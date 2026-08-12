use crate::{
    dispatch::{
        CompilationOptions, GpuBackend,
        backend::{
            DType, GpuContext, Graph, GraphOp, MetaId, Metadata, NodeId, Op, Param, ParamId,
            SharedAlloc, SharedId, Value, ValueId, ValueState,
        },
    },
    errors::{Error, ErrorKind, GraphErrorContext},
};
use alloc::{boxed::Box, vec, vec::Vec};

mod forward;

mod backward;

mod loss;

/// Forward, backward, and loss kernel IR.
#[derive(Debug)]
pub struct KernelGroup<'a, B: GpuBackend = GpuContext> {
    pub(crate) forward: KernelsRedirected<'a, B>,
    pub(crate) backward: KernelsRedirected<'a, B>,
    pub(crate) loss: Kernel,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Dependencies<T> {
    pub(crate) val: T,
    pub(crate) dep: Vec<usize>,
}

pub fn topo_sort<B: GpuBackend, T: Clone>(
    nodes: &mut [Dependencies<T>],
) -> Result<(), Error<GraphErrorContext<'_, B>>> {
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
    pub(crate) flags: u8,
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

#[derive(Debug)]
pub struct KernelsChained<'a, B: GpuBackend = GpuContext> {
    pub kernels: Vec<Dependencies<LinkedKernel<'a, B>>>,
    pub params: Vec<Param>,
}

#[derive(Debug)]
pub struct KernelsRedirected<'a, B: GpuBackend = GpuContext> {
    pub kernels: Vec<Dependencies<Redirect<LinkedKernel<'a, B>>>>,
    pub params: Vec<Param>,
}

impl<'a> KernelsChained<'a> {
    pub(crate) fn lower<B: GpuBackend>(
        graph: &'a Graph<'a, B>,
        meta: Metadata,
        saved: &[SaveIndicator],
        options: &CompilationOptions<B>,
    ) -> Result<KernelGroup<'a, B>, Error> {
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

        let forward = eliminate_kernels(forward);
        let backward = eliminate_kernels(backward);

        Ok(KernelGroup {
            forward,
            backward,
            loss,
        })
    }
}

#[derive(Debug)]
pub enum Redirect<T> {
    Unmasked(T),
    Redirected(usize),
}

fn eliminate_kernels<'a, B: GpuBackend>(kernels: KernelsChained<'a, B>) -> KernelsRedirected<'a, B> {
    let mut kernels_optimized: Vec<Dependencies<Redirect<LinkedKernel<'a, B>>>> = Vec::new();

    for kernel in kernels.kernels {
        let mut unique_id = usize::MAX;

        for (idx, resolved_kernel) in kernels_optimized.iter().enumerate() {
            if let Redirect::Unmasked(resolved_kernel) = &resolved_kernel.val 
                && kernel.val.ops == resolved_kernel.ops
                && kernel.val.meta == resolved_kernel.meta
                && kernel.val.params == resolved_kernel.params
            {
                unique_id = idx;
            }
        }

        if unique_id == usize::MAX {
            kernels_optimized.push(Dependencies {
                val: Redirect::Unmasked(kernel.val),
                dep: kernel.dep,
            });
        } else {
            kernels_optimized.push(Dependencies {
                val: Redirect::Redirected(unique_id),
                dep: kernel.dep,
            });
        }
    }

    KernelsRedirected {
        kernels: kernels_optimized,
        params: kernels.params,
    }
}

#[derive(Debug)]
pub struct Kernel {
    pub raw: RawKernel,
    pub params: Vec<Param>,
    pub meta: Vec<bool>,
}

impl Kernel {
    pub fn update_param_dtype(&mut self, id: ValueId, dtype: DType) {
        self.params[id].dtype = dtype;
    }

    pub fn register_meta(&mut self, field: MetaId) {
        self.meta[field] = true;
    }

    pub fn unregister_meta(&mut self, field: MetaId) {
        self.meta[field] = false;
    }
}

#[derive(Debug)]
pub struct LinkedKernel<'a, B: GpuBackend = GpuContext> {
    pub raw: RawKernel,
    pub params: Vec<bool>,
    pub meta: Vec<bool>,
    ops: Vec<&'a GraphOp<'a, B>>,
}

impl<B: GpuBackend> LinkedKernel<'_, B> {
    pub fn push_if<F: FnOnce(&mut Self) -> Result<R, Error>, R>(
        &mut self,
        cond: ValueId,
        content: F,
    ) -> Result<R, Error> {
        self.raw.ops.push(Op::IfBegin { cond });
        let ret = content(self);
        self.raw.ops.push(Op::EndScope);
        ret
    }

    pub fn push_if_else(
        &mut self,
        cond: ValueId,
        content: impl FnOnce(&mut Self) -> Result<(), Error>,
        else_content: impl FnOnce(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.raw.ops.push(Op::IfBegin { cond });
        content(self)?;
        self.raw.ops.push(Op::EndScope);
        self.raw.ops.push(Op::ElseBegin);
        else_content(self)?;
        self.raw.ops.push(Op::EndScope);
        Ok(())
    }

    pub fn push_for_loop<F: FnOnce(&mut Self) -> Result<R, Error>, R>(
        &mut self,
        index: ValueId,
        end: ValueId,
        step: ValueId,
        content: F,
    ) -> Result<R, Error> {
        self.raw.ops.push(Op::ForLoopBegin { index, end, step });
        let ret = content(self);
        self.raw.ops.push(Op::EndScope);
        ret
    }

    pub fn push_while_loop<F: FnOnce(&mut Self) -> Result<R, Error>, R>(
        &mut self,
        cond: Op,
        content: F,
    ) -> Result<R, Error> {
        self.raw.ops.push(Op::WhileLoopBegin {
            cond: Box::new(cond),
        });
        let ret = content(self);
        self.raw.ops.push(Op::EndScope);
        ret
    }

    pub fn push_forever_loop<F: FnOnce(&mut Self) -> Result<R, Error>, R>(
        &mut self,
        content: F,
    ) -> Result<R, Error> {
        self.raw.ops.push(Op::ForeverLoopBegin);
        let ret = content(self);
        self.raw.ops.push(Op::EndScope);
        ret
    }

    pub fn param_store(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.register_param(param);

        self.raw.ops.push(Op::ParamStore {
            param,
            index,
            value,
        });
    }

    pub fn param_accum(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.register_param(param);

        self.raw.ops.push(Op::ParamAccum {
            param,
            index,
            value,
        });
    }

    pub fn param_mul(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.register_param(param);

        self.raw.ops.push(Op::ParamMul {
            param,
            index,
            value,
        });
    }

    pub fn param_div(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.register_param(param);

        self.raw.ops.push(Op::ParamDiv {
            param,
            index,
            value,
        });
    }

    pub fn param_sub(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.register_param(param);

        self.raw.ops.push(Op::ParamSub {
            param,
            index,
            value,
        });
    }

    pub fn param_shl(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.register_param(param);

        self.raw.ops.push(Op::ParamShl {
            param,
            index,
            value,
        });
    }

    pub fn param_shr(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.register_param(param);

        self.raw.ops.push(Op::ParamShr {
            param,
            index,
            value,
        });
    }
}

impl<B: GpuBackend> LinkedKernel<'_, B> {
    pub fn register_param(&mut self, id: ParamId) {
        self.params[id] = true;
    }

    pub fn register_meta(&mut self, field: MetaId) {
        self.meta[field] = true;
    }

    pub fn unregister_param(&mut self, id: ParamId) {
        self.params[id] = false;
    }

    pub fn unregister_meta(&mut self, field: MetaId) {
        self.meta[field] = false;
    }
}

#[derive(Debug)]
pub struct RawKernel {
    pub meta: Metadata,

    pub shared: Vec<SharedAlloc>,

    pub values: Vec<Value>,

    pub ops: Vec<Op>,

    pub block: [u32; 3],

    pub iter_space: Vec<MetaId>,

    pub root: NodeId,
}

impl RawKernel {
    pub fn push_if<F: FnOnce(&mut Self) -> Result<R, Error>, R>(
        &mut self,
        cond: ValueId,
        content: F,
    ) -> Result<R, Error> {
        self.ops.push(Op::IfBegin { cond });
        let ret = content(self);
        self.ops.push(Op::EndScope);
        ret
    }

    pub fn push_if_else(
        &mut self,
        cond: ValueId,
        content: impl FnOnce(&mut Self) -> Result<(), Error>,
        else_content: impl FnOnce(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.ops.push(Op::IfBegin { cond });
        content(self)?;
        self.ops.push(Op::EndScope);
        self.ops.push(Op::ElseBegin);
        else_content(self)?;
        self.ops.push(Op::EndScope);
        Ok(())
    }

    pub fn push_for_loop<F: FnOnce(&mut Self) -> Result<R, Error>, R>(
        &mut self,
        index: ValueId,
        end: ValueId,
        step: ValueId,
        content: F,
    ) -> Result<R, Error> {
        self.ops.push(Op::ForLoopBegin { index, end, step });
        let ret = content(self);
        self.ops.push(Op::EndScope);
        ret
    }

    pub fn push_while_loop<F: FnOnce(&mut Self) -> Result<R, Error>, R>(
        &mut self,
        cond: Op,
        content: F,
    ) -> Result<R, Error> {
        self.ops.push(Op::WhileLoopBegin {
            cond: Box::new(cond),
        });
        let ret = content(self);
        self.ops.push(Op::EndScope);
        ret
    }

    pub fn push_forever_loop<F: FnOnce(&mut Self) -> Result<R, Error>, R>(
        &mut self,
        content: F,
    ) -> Result<R, Error> {
        self.ops.push(Op::ForeverLoopBegin);
        let ret = content(self);
        self.ops.push(Op::EndScope);
        ret
    }

    pub fn param_store(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.ops.push(Op::ParamStore {
            param,
            index,
            value,
        });
    }

    pub fn param_accum(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.ops.push(Op::ParamAccum {
            param,
            index,
            value,
        });
    }

    pub fn param_mul(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.ops.push(Op::ParamMul {
            param,
            index,
            value,
        });
    }

    pub fn param_div(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.ops.push(Op::ParamDiv {
            param,
            index,
            value,
        });
    }

    pub fn param_sub(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.ops.push(Op::ParamSub {
            param,
            index,
            value,
        });
    }

    pub fn param_shl(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.ops.push(Op::ParamShl {
            param,
            index,
            value,
        });
    }

    pub fn param_shr(&mut self, param: ParamId, index: ValueId, value: ValueId) {
        self.ops.push(Op::ParamShr {
            param,
            index,
            value,
        });
    }
}

impl RawKernel {
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
        self.ops.push(Op::AddAssign {
            id,
            val: Box::new(op),
        });
    }

    pub fn mul_assign_var(&mut self, id: ValueId, op: Op) {
        self.ops.push(Op::MulAssign {
            id,
            val: Box::new(op),
        });
    }

    pub fn div_assign_var(&mut self, id: ValueId, op: Op) {
        self.ops.push(Op::DivAssign {
            id,
            val: Box::new(op),
        });
    }

    pub fn sub_assign_var(&mut self, id: ValueId, op: Op) {
        self.ops.push(Op::SubAssign {
            id,
            val: Box::new(op),
        });
    }

    pub fn shl_assign_var(&mut self, id: ValueId, op: Op) {
        self.ops.push(Op::ShlAssign {
            id,
            val: Box::new(op),
        });
    }

    pub fn shr_assign_var(&mut self, id: ValueId, op: Op) {
        self.ops.push(Op::ShrAssign {
            id,
            val: Box::new(op),
        });
    }

    pub fn update_var_state(&mut self, id: ValueId, state: ValueState) {
        self.values[id].state = state;
    }

    pub fn update_var_dtype(&mut self, id: ValueId, dtype: DType) {
        self.values[id].dtype = dtype;
    }

    pub fn update_var_init(&mut self, id: ValueId, init: Op) {
        self.values[id].init = Some(init);
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

    pub fn push_continue(&mut self) {
        self.ops.push(Op::Continue);
    }

    pub fn push_break(&mut self) {
        self.ops.push(Op::Break);
    }

    pub fn shared_store(&mut self, mem: SharedId, index: ValueId, value: ValueId) {
        self.ops.push(Op::SharedStore { mem, index, value });
    }

    pub fn shared_accum(&mut self, mem: SharedId, index: ValueId, value: ValueId) {
        self.ops.push(Op::SharedAccum { mem, index, value });
    }

    pub fn shared_mul(&mut self, mem: SharedId, index: ValueId, value: ValueId) {
        self.ops.push(Op::SharedMul { mem, index, value });
    }

    pub fn shared_div(&mut self, mem: SharedId, index: ValueId, value: ValueId) {
        self.ops.push(Op::SharedDiv { mem, index, value });
    }

    pub fn shared_sub(&mut self, mem: SharedId, index: ValueId, value: ValueId) {
        self.ops.push(Op::SharedSub { mem, index, value });
    }

    pub fn shared_shl(&mut self, mem: SharedId, index: ValueId, value: ValueId) {
        self.ops.push(Op::SharedShl { mem, index, value });
    }

    pub fn shared_shr(&mut self, mem: SharedId, index: ValueId, value: ValueId) {
        self.ops.push(Op::SharedShr { mem, index, value });
    }
}

pub enum NodeInput<'a> {
    Raw { param: ParamId, shape: &'a [MetaId] },

    Node(NodeId),
}
