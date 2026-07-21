use crate::{
    dispatch::{
        CompilationOptions, GpuBackend,
        backend::{
            DType, Graph, GraphOp, MetaId, Metadata, NodeId, Op, Param, ParamId, SharedAlloc,
            SharedId, Value, ValueId, ValueState,
        },
    },
    errors::{Error, ErrorKind, GraphErrorContext},
};
use alloc::{boxed::Box, vec, vec::Vec};

mod forward;

mod backward;

mod loss;

/// Forward, backward, and loss kernel IR.
#[derive(Debug, Clone)]
pub struct KernelGroup {
    pub(crate) forward: Vec<Dependencies<Kernel>>,
    pub(crate) backward: Vec<Dependencies<Kernel>>,
    pub(crate) loss: Kernel,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Dependencies<T> {
    pub(crate) val: T,
    pub(crate) dep: Vec<usize>,
}

pub fn topo_sort<T: Clone>(nodes: &mut [Dependencies<T>]) -> Result<(), Error<GraphErrorContext>> {
    let n = nodes.len();

    let mut in_degree = vec![0_usize; n];
    let mut adj = vec![Vec::new(); n];

    for (node_id, node) in nodes.iter().enumerate() {
        for &inp in &node.dep {
            if inp >= n {
                return Err(Error {
                    msg: "invalid node reference in graph",
                    kind: ErrorKind::ComputeGraphError,
                    ctx: GraphErrorContext::MissingInput {
                        node: node_id,
                        input: inp,
                    },
                });
            }

            adj[inp].push(node_id);
            in_degree[node_id] += 1;
        }
    }

    let mut zeros: Vec<_> = (0..n).filter(|&i| in_degree[i] == 0).collect();

    zeros.sort_unstable();

    let mut order = Vec::with_capacity(n);

    let mut idx = 0;
    while idx < zeros.len() {
        let node = zeros[idx];
        idx += 1;

        order.push(node);

        let mut nexts = adj[node].clone();
        nexts.sort_unstable();

        for nxt in nexts {
            in_degree[nxt] -= 1;
            if in_degree[nxt] == 0 {
                zeros.push(nxt);
            }
        }
    }

    if order.len() != n {
        return Err(Error {
            msg: "cycle detected in graph",
            kind: ErrorKind::ComputeGraphError,
            ctx: GraphErrorContext::CycleDetected {
                node: 0,
                path: order,
            },
        });
    }

    let mut new_index = vec![0_usize; n];
    for (i, &old) in order.iter().enumerate() {
        new_index[old] = i;
    }

    let mut new_nodes = Vec::with_capacity(n);

    for &old_id in &order {
        let mut node = nodes[old_id].clone();

        for inp in &mut node.dep {
            *inp = new_index[*inp];
        }

        new_nodes.push(node);
    }

    nodes.clone_from_slice(&new_nodes);

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SaveIndicator {
    flags: u8,
}

impl core::ops::BitOr for SaveIndicator {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.or(rhs)
    }
}

impl core::ops::BitOrAssign for SaveIndicator {
    fn bitor_assign(&mut self, rhs: Self) {
        self.flags |= rhs.flags;
    }
}

impl SaveIndicator {
    pub const DEFINED_IN_FORWARD: Self = Self { flags: 0b0000_0001 };
    pub const USED_BY_FORWARD: Self = Self { flags: 0b0000_0010 };
    pub const DEFINED_IN_BACKWARD: Self = Self { flags: 0b0000_0100 };
    pub const USED_BY_BACKWARD: Self = Self { flags: 0b0000_1000 };

    #[must_use]
    pub const fn or(self, other: Self) -> Self {
        Self {
            flags: self.flags | other.flags,
        }
    }

    #[must_use]
    pub const fn is_defined_in_forward(&self) -> bool {
        self.flags & Self::DEFINED_IN_FORWARD.flags == Self::DEFINED_IN_FORWARD.flags
    }

    #[must_use]
    pub const fn is_used_by_forward(&self) -> bool {
        self.flags & Self::USED_BY_FORWARD.flags == Self::USED_BY_FORWARD.flags
    }

    #[must_use]
    pub const fn is_defined_in_backward(&self) -> bool {
        self.flags & Self::DEFINED_IN_BACKWARD.flags == Self::DEFINED_IN_BACKWARD.flags
    }

    #[must_use]
    pub const fn is_used_by_backward(&self) -> bool {
        self.flags & Self::USED_BY_BACKWARD.flags == Self::USED_BY_BACKWARD.flags
    }
}

/// Intermediate representation of a GPU kernel.
#[derive(Debug, Clone)]
pub struct Kernel {
    pub(crate) meta: Metadata,

    pub(crate) params: Vec<Param>,

    pub(crate) shared: Vec<SharedAlloc>,

    pub(crate) values: Vec<Value>,

    pub(crate) ops: Vec<Op>,

    pub(crate) block: [u32; 3],

    pub(crate) iter_space: Vec<MetaId>,

    pub(crate) root: NodeId,
}

impl Kernel {
    #[inline]
    #[must_use]
    pub fn compute_saved_nodes(graph: &Graph) -> Vec<SaveIndicator> {
        let mut saved = vec![SaveIndicator { flags: 0 }; graph.nodes.len()];

        for (node_id, node) in graph.nodes.iter().enumerate() {
            match node.op {
                GraphOp::Input => {
                    saved[node_id] |=
                        SaveIndicator::DEFINED_IN_BACKWARD | SaveIndicator::USED_BY_BACKWARD;
                }

                GraphOp::Custom { save, .. } => {
                    save(node_id, node, graph, &mut saved);
                }

                _ => {}
            }
        }

        saved
    }

    /// Lowers an execution graph into kernels.
    pub fn lower<B: GpuBackend + Clone>(
        graph: &Graph<B>,
        meta: Metadata,
        saved: &[SaveIndicator],
        options: &CompilationOptions<B>,
    ) -> Result<KernelGroup, Error> {
        if graph.nodes.is_empty() {
            return Err(Error {
                msg: "graph empty",
                kind: ErrorKind::GraphEmpty,
                ctx: (),
            });
        }

        let forward = forward::lower_forward(graph, meta, saved, options)?;
        let backward = backward::lower_backward(graph, meta, saved, options)?;
        let loss = loss::lower_loss(graph, meta);

        Ok(KernelGroup {
            forward,
            backward,
            loss,
        })
    }

    pub fn def_var(&mut self, dtype: DType, state: ValueState, init: Option<Op>) -> ValueId {
        let id = self.values.len();
        self.values.push(Value { state, dtype, init });
        self.ops.push(Op::DefineVar { id });
        id
    }

    pub fn overwrite_var(&mut self, id: ValueId, op: Op) {
        self.ops.push(Op::OverwriteVar {
            id,
            val: Box::new(op),
        });
    }

    pub fn accum_var(&mut self, id: ValueId, op: Op) {
        self.ops.push(Op::AccumVar {
            id,
            val: Box::new(op),
        });
    }

    pub fn update_state(&mut self, id: ValueId, state: ValueState) {
        self.values[id].state = state;
    }

    pub fn new_shared(&mut self, dtype: DType, size: u32) -> SharedId {
        let id = self.shared.len();
        self.shared.push(SharedAlloc { dtype, size });
        id
    }

    pub fn push_barrier(&mut self) {
        self.ops.push(Op::Barrier);
    }

    pub fn push_return(&mut self) {
        self.ops.push(Op::Return);
    }

    pub fn push_if(&mut self, cond: ValueId, content: impl FnOnce(&mut Self)) {
        self.ops.push(Op::IfBegin { cond });
        content(self);
        self.ops.push(Op::EndScope);
    }

    pub fn push_if_else(
        &mut self,
        cond: ValueId,
        content: impl FnOnce(&mut Self),
        else_content: impl FnOnce(&mut Self),
    ) {
        self.ops.push(Op::IfBegin { cond });
        content(self);
        self.ops.push(Op::EndScope);
        self.ops.push(Op::ElseBegin);
        else_content(self);
        self.ops.push(Op::EndScope);
    }

    pub fn push_for_loop<F: FnOnce(&mut Self) -> R, R>(
        &mut self,
        index: ValueId,
        end: ValueId,
        step: ValueId,
        content: F,
    ) -> R {
        self.ops.push(Op::ForLoopBegin { index, end, step });
        let ret = content(self);
        self.ops.push(Op::EndScope);
        ret
    }

    pub fn push_continue(&mut self) {
        self.ops.push(Op::Continue);
    }

    pub fn push_break(&mut self) {
        self.ops.push(Op::Break);
    }

    pub fn param_store(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.ops.push(Op::Store {
            param,
            index,
            value,
        });
    }

    pub fn shared_store(&mut self, mem: SharedId, index: ValueId, value: ValueId) {
        self.ops.push(Op::SharedStore { mem, index, value });
    }
}

pub enum NodeInput<'a> {
    Raw { param: ParamId, shape: &'a [MetaId] },

    Node(NodeId),
}
