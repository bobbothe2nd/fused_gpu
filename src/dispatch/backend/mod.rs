use alloc::{boxed::Box, string::String, vec, vec::Vec};
use core::{cmp::Ordering, fmt::Debug};

use crate::{
    dispatch::{
        CompilationOptions, GpuBackend,
        backend::kernel::{Kernel, KernelGroup, NodeInput, SaveIndicator},
    },
    errors::{Error, ErrorKind, GraphErrorContext},
};

mod std_lib;

pub mod kernel;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "rocm")]
pub mod rocm;

#[cfg(feature = "wgsl")]
pub mod wgsl;

#[cfg(all(feature = "wgsl", not(any(feature = "rocm", feature = "cuda"))))]
pub type GpuBuffer = wgsl::GpuBuffer;

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub type GpuBuffer = rocm::GpuBuffer;

#[cfg(feature = "cuda")]
pub type GpuBuffer = cuda::GpuBuffer;

#[cfg(all(feature = "wgsl", not(any(feature = "rocm", feature = "cuda"))))]
pub type GpuContext = wgsl::GpuContext;

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub type GpuContext = rocm::GpuContext;

#[cfg(feature = "cuda")]
pub type GpuContext = cuda::GpuContext;

/// Identifier of a node in a graph.
pub type NodeId = usize;

/// Identifier of a parameter in kernel IR.
pub type ParamId = usize;

/// Identifier of a metadata field in kernel IR.
pub type MetaId = usize;

/// Identifier of a shared buffer in kernel IR.
pub type SharedId = usize;

/// Identifier of a value in kernel IR.
pub type ValueId = usize;

/// Data type used in kernel IR.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float,
    SignedInt,
    UnsignedInt,
    Bool,
}

/// Parameter type used in kernel IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Param {
    pub dtype: DType,
    pub ty: ParamTy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamTy {
    Uniform,
    ReadWrite,
    ReadOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedAlloc {
    pub dtype: DType,
    pub size: u32,
}

/// Value type used in kernel IR.
#[derive(Debug, Clone, PartialEq)]
pub struct Value {
    pub state: ValueState,
    pub dtype: DType,
    pub init: Option<Op>,
}

/// Value state type that defines a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueState {
    Mut,
    Immut,
    Masked,
    Inline,
    Const,
}

/// Axis type used in kernel IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// x-axis (0)
    X,

    /// y-axis (1)
    Y,

    /// z-axis (2)
    Z,
}

impl TryFrom<u8> for Axis {
    type Error = Error;

    #[inline]
    fn try_from(value: u8) -> Result<Self, Error> {
        match value {
            0 => Ok(Self::X),
            1 => Ok(Self::Y),
            2 => Ok(Self::Z),
            _ => Err(Error {
                msg: "could not coerce invalid integer to axis",
                kind: ErrorKind::InvalidArgument,
                ctx: (),
            }),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    DefineVar {
        id: ValueId,
    },
    OverwriteVar {
        id: ValueId,
        val: Box<Self>,
    },
    AccumVar {
        id: ValueId,
        val: Box<Self>,
    },
    CopyVar {
        id: ValueId,
    },

    ConstF32 {
        value: f32,
    },
    ConstU32 {
        value: u32,
    },
    ConstI32 {
        value: i32,
    },

    ReadMeta {
        param: ParamId,
        field: MetaId,
    },

    LocalId {
        axis: Axis,
    },
    BlockId {
        axis: Axis,
    },
    GlobalId {
        axis: Axis,
    },

    Add {
        a: ValueId,
        b: ValueId,
    },
    Sub {
        a: ValueId,
        b: ValueId,
    },
    Mul {
        a: ValueId,
        b: ValueId,
    },
    Div {
        a: ValueId,
        b: ValueId,
    },
    Mod {
        a: ValueId,
        b: ValueId,
    },
    Pow {
        a: ValueId,
        b: ValueId,
    },

    /// (Fused) Operation `a * b + c`
    Fma {
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },

    Exp {
        x: ValueId,
    },
    Abs {
        x: ValueId,
    },
    Neg {
        x: ValueId,
    },
    Log {
        x: ValueId,
    },
    Tanh {
        x: ValueId,
    },
    Sqrt {
        x: ValueId,
    },

    Load {
        param: ParamId,
        index: ValueId,
    },
    Store {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },

    SharedLoad {
        mem: SharedId,
        index: ValueId,
    },
    SharedStore {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },

    Lt {
        a: ValueId,
        b: ValueId,
    },

    Gt {
        a: ValueId,
        b: ValueId,
    },

    Le {
        a: ValueId,
        b: ValueId,
    },

    Ge {
        a: ValueId,
        b: ValueId,
    },

    Max {
        a: ValueId,
        b: ValueId,
    },
    Min {
        a: ValueId,
        b: ValueId,
    },

    Select {
        cond: ValueId,
        a: ValueId,
        b: ValueId,
    },

    ForLoopBegin {
        index: ValueId,
        end: ValueId,
        step: ValueId,
    },

    Continue,

    Break,

    IfBegin {
        cond: ValueId,
    },

    ElseBegin,

    EndScope,

    Barrier,

    Return,
}

#[derive(Debug, Clone)]
pub struct Node<B: GpuBackend + Clone = GpuContext> {
    pub(crate) op: GraphOp<B>,
    pub(crate) inputs: Vec<NodeId>,
    pub(crate) outputs: Vec<NodeId>,
    pub(crate) shape: Vec<MetaId>,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LossType {
    MeanSquaredError,
    CrossEntropy,
    BinaryCrossEntropy,
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Metadata {
    fields: MetaId,
}

impl Metadata {
    /// Defines an empty metadata instance.
    #[must_use]
    pub const fn new() -> Self {
        Self { fields: 0 }
    }

    /// Defines a new metadata field, returning the identifier.
    ///
    /// The identifier will always be 1 more than than the previous field, starting from 0.
    pub const fn new_field(&mut self) -> MetaId {
        let field = self.fields;
        self.fields += 1;
        field
    }

    /// Returns an iterator over each index.
    ///
    /// This is numerically `0..[len]`.
    #[must_use]
    pub const fn iter_fields(&self) -> core::ops::Range<MetaId> {
        0..self.fields
    }

    /// Checks if metadata is valid.
    ///
    /// Valid metadata is defined as:
    ///
    /// - instantiation must be the same length as compile time definition
    /// - all fields must be multiples of 16 (as of now)
    ///
    /// If valid, return `true`. If any of the above criteria fail, return `false`.
    #[must_use]
    pub fn validate_meta(&self, meta: &[u32]) -> bool {
        meta.len() == self.fields && meta.iter().all(|x| x.is_multiple_of(16))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchOptions {
    Any,
    ReqRow,
    ReqCol,
    ReqRowCol,
}

impl PartialOrd for DispatchOptions {
    fn ge(&self, other: &Self) -> bool {
        self == other || *self == Self::ReqRowCol || *other != Self::ReqRowCol
    }

    fn gt(&self, other: &Self) -> bool {
        *self == Self::ReqRowCol || *other != Self::ReqRowCol
    }

    fn le(&self, other: &Self) -> bool {
        self == other || *self != Self::ReqRowCol || *other == Self::ReqRowCol
    }

    fn lt(&self, other: &Self) -> bool {
        *self != Self::ReqRowCol || *other == Self::ReqRowCol
    }

    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }

        match (self, other) {
            (Self::ReqRowCol, _) => Some(Ordering::Greater),
            (_, Self::ReqRowCol) => Some(Ordering::Less),
            _ => None,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum GraphOp<B: GpuBackend + Clone = GpuContext> {
    Input,
    ConstF32(f32),
    ConstI32(i32),
    ConstU32(u32),

    Custom {
        lower: fn(
            fn(
                NodeId,
                NodeId,
                &NodeInput,
                ValueId,
                &mut Vec<NodeId>,
                &Graph<B>,
                &[Option<ParamId>],
                &[Option<ParamId>],
                &mut Kernel,
                ValueId,
                ValueId,
                ValueId,
                ValueId,
                u32,
                ValueId,
                bool,
                &CompilationOptions<B>,
            ) -> Result<Vec<NodeId>, Error>,
            NodeId,
            NodeId,
            &mut Vec<NodeId>,
            Option<u8>,
            NodeId,
            &Graph<B>,
            ValueId,
            &[Option<ParamId>],
            &[Option<ParamId>],
            &mut Kernel,
            ValueId,
            ValueId,
            ValueId,
            ValueId,
            u32,
            ValueId,
            bool,
            &CompilationOptions<B>,
        ) -> Result<Vec<NodeId>, Error>,
        display: fn(&[Vec<MetaId>]) -> String,
        save: fn(NodeId, &Node<B>, &Graph<B>, &mut [SaveIndicator]),
        valid_shape: fn(NodeId, &Node<B>, &Graph<B>, &mut Vec<Error<GraphErrorContext<B>>>),
        arity: u8,
        need_dims: bool,
        stable_iter: bool,
        iter_space: Vec<bool>,
        auto_save: bool,
        valid_dispatch: DispatchOptions,
    },
}

impl<B: GpuBackend + Clone> GraphOp<B> {
    #[must_use]
    pub const fn is_elementwise(&self) -> bool {
        if let Self::Custom { need_dims, .. } = self {
            !*need_dims
        } else {
            false
        }
    }

    #[must_use]
    pub const fn is_transform(&self) -> bool {
        if let Self::Custom { stable_iter, .. } = self {
            !*stable_iter
        } else {
            false
        }
    }

    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        matches!(
            self,
            Self::Input
                | Self::ConstF32(_)
                | Self::ConstI32(_)
                | Self::ConstU32(_)
        )
    }

    #[must_use]
    pub const fn is_auto_save(&self) -> bool {
        if let Self::Custom { auto_save, .. } = self {
            *auto_save
        } else {
            false
        }
    }

    #[must_use]
    pub const fn arity(&self) -> u8 {
        if let Self::Custom { arity, .. } = self {
            *arity
        } else {
            0
        }
    }

    #[must_use]
    pub fn debug(&self, all_hand_sides: &[Vec<MetaId>]) -> String {
        match self {
            Self::Custom { display, .. } => display(all_hand_sides),
            _ => String::new(),
        }
    }
}

#[inline]
fn check_acrylicity<B: GpuBackend + Clone>(
    graph: &Graph<B>,
    errors: &mut Vec<Error<GraphErrorContext<B>>>,
) {
    #[inline]
    fn dfs<B: GpuBackend + Clone>(
        node: NodeId,
        graph: &Graph<B>,
        visited: &mut [bool],
        stack: &mut [bool],
        errors: &mut Vec<Error<GraphErrorContext<B>>>,
        path: &mut Vec<NodeId>,
    ) {
        if stack[node] {
            errors.push(Error {
                msg: "graph must be acrylic",
                kind: ErrorKind::ComputeGraphError,
                ctx: GraphErrorContext::CycleDetected {
                    node,
                    path: path.clone(),
                },
            });
            return;
        }

        if visited[node] {
            return;
        }

        visited[node] = true;
        stack[node] = true;
        path.push(node);

        let n = &graph.nodes[node];

        for input in &n.inputs {
            dfs(*input, graph, visited, stack, errors, path);
        }

        stack[node] = false;
        path.clear();
    }

    let mut visited = vec![false; graph.nodes.len()];
    let mut stack = vec![false; graph.nodes.len()];

    let mut path = Vec::new();
    for node_id in 0..graph.nodes.len() {
        dfs(node_id, graph, &mut visited, &mut stack, errors, &mut path);
    }
}

#[inline]
fn check_inputs_exist<B: GpuBackend + Clone>(
    graph: &Graph<B>,
    errors: &mut Vec<Error<GraphErrorContext<B>>>,
) {
    let node_count = graph.nodes.len();

    for (node_id, node) in graph.nodes.iter().enumerate() {
        for input in &node.inputs {
            if *input >= node_count {
                errors.push(Error {
                    msg: "node referenced non-existent input(s)",
                    kind: ErrorKind::ComputeGraphError,
                    ctx: GraphErrorContext::MissingInput {
                        node: node_id,
                        input: *input,
                    },
                });
            }
        }
    }
}

#[inline]
fn check_metadata<B: GpuBackend + Clone>(
    graph: &Graph<B>,
    meta: Metadata,
    errors: &mut Vec<Error<GraphErrorContext<B>>>,
) {
    for (node_id, node) in graph.nodes.iter().enumerate() {
        for &dim in &node.shape {
            if dim >= meta.fields {
                errors.push(Error {
                    msg: "node has no metadata entry",
                    kind: ErrorKind::ComputeGraphError,
                    ctx: GraphErrorContext::MissingMetadata {
                        node: node_id,
                        meta: dim,
                    },
                });
            }
        }
    }
}

fn check_shapes<B: GpuBackend + Clone>(
    graph: &Graph<B>,
    errors: &mut Vec<Error<GraphErrorContext<B>>>,
) {
    for (node_id, node) in graph.nodes.iter().enumerate() {
        if node.shape.len() < 2 {
            errors.push(Error {
                msg: "no node can have less than two dimensions",
                kind: ErrorKind::ComputeGraphError,
                ctx: GraphErrorContext::LowRank {
                    node: node_id,
                    rank: node.shape.len(),
                    required: 2,
                }
            });
        }

        match node.op {
            GraphOp::Custom { valid_shape, .. } => {
                valid_shape(node_id, node, graph, errors);
            }

            GraphOp::Input
            | GraphOp::ConstF32(_)
            | GraphOp::ConstI32(_)
            | GraphOp::ConstU32(_) => {
                if !node.inputs.is_empty() {
                    errors.push(Error {
                        msg: "leaf node accepting inputs",
                        kind: ErrorKind::ComputeGraphError,
                        ctx: GraphErrorContext::InvalidInputs {
                            node: node_id,
                            arity: 0,
                            args: node.inputs.len(),
                        },
                    });
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Graph<B: GpuBackend + Clone = GpuContext> {
    pub(crate) nodes: Vec<Node<B>>,
    pub(crate) loss: LossType,
}

impl<B: GpuBackend + Clone> Graph<B> {
    #[must_use]
    pub const fn new(loss: LossType) -> Self {
        Self {
            nodes: Vec::new(),
            loss,
        }
    }

    #[must_use]
    pub fn get_adjacent(&self, node: NodeId, user: NodeId) -> &[NodeId] {
        let user_inputs = &self.nodes[user].inputs;

        let mut adjacent: &[NodeId] = &[];

        for input in user_inputs {
            if *input != node {
                continue;
            }

            adjacent = &self.nodes[*input].outputs;
        }

        adjacent
    }

    #[must_use]
    pub fn is_lhs_edge(&self, node: NodeId, user: NodeId) -> bool {
        let user_inputs = &self.nodes[user].inputs;

        !user_inputs.is_empty() && user_inputs[0] == node
    }

    #[must_use]
    pub fn is_rhs_edge(&self, node: NodeId, user: NodeId) -> bool {
        let user_inputs = &self.nodes[user].inputs;

        user_inputs.len() > 1 && user_inputs[1] == node
    }

    #[must_use]
    pub fn is_edge(&self, node: NodeId, user: NodeId, edge: usize) -> bool {
        let user_inputs = &self.nodes[user].inputs;

        user_inputs.len() > edge && user_inputs[edge] == node
    }

    #[must_use]
    pub fn get_edge(&self, node: NodeId, user: NodeId) -> Option<usize> {
        let user_inputs = &self.nodes[user].inputs;

        user_inputs.iter().position(|input| *input == node)
    }

    pub fn topo_sort(&mut self) -> Result<(), Error<GraphErrorContext<B>>> {
        let n = self.nodes.len();

        let mut in_degree = vec![0usize; n];
        let mut adj = vec![Vec::<usize>::new(); n];

        for (node_id, node) in self.nodes.iter().enumerate() {
            for &inp in &node.inputs {
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

        let mut new_index = vec![0usize; n];
        for (i, &old) in order.iter().enumerate() {
            new_index[old] = i;
        }

        let mut new_nodes = Vec::with_capacity(n);

        for &old_id in &order {
            let mut node = self.nodes[old_id].clone();

            for inp in &mut node.inputs {
                *inp = new_index[*inp];
            }

            node.outputs.clear();

            new_nodes.push(node);
        }

        self.nodes = new_nodes;

        Ok(())
    }

    pub fn rebuild_outputs(&mut self) {
        let mut outs = vec![Vec::new(); self.nodes.len()];

        for (node_id, node) in self.nodes.iter().enumerate() {
            for &inp in &node.inputs {
                outs[inp].push(node_id);
            }
        }

        for (i, o) in outs.into_iter().enumerate() {
            self.nodes[i].outputs = o;
        }
    }

    /// Validates the graph for mathematical correctness given the metadata usesd during lowering.
    ///
    /// This function will not panic, but it can stop many panics and bugs in:
    ///
    /// - [`Self::lower`]
    /// - [`GpuContext::compile`](`super::GpuContext::compile`)
    /// - [`GpuContext::launch_forward`](`super::GpuContext::launch_forward`)
    /// - [`GpuContext::launch_backward`](`super::GpuContext::launch_backward`)
    /// - [`GpuContext::launch_loss`](`super::GpuContext::launch_loss`)
    ///
    /// It is recommended that you run this function on your graph at least in debug mode, or
    /// you could have panics in production code.
    pub fn validate(&self, meta: Metadata) -> Result<(), Vec<Error<GraphErrorContext<B>>>>
    where
        B: Clone,
    {
        let mut errors = Vec::new();

        check_acrylicity(self, &mut errors);
        check_inputs_exist(self, &mut errors);
        check_metadata(self, meta, &mut errors);
        check_shapes(self, &mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn lower(
        &self,
        meta: Metadata,
        options: &CompilationOptions<B>,
        saved: &[SaveIndicator],
    ) -> Result<KernelGroup, Error> {
        Kernel::lower(self, meta, saved, options)
    }

    fn add_node(&mut self, op: GraphOp<B>, inputs: Vec<NodeId>, shape: Vec<MetaId>) -> NodeId {
        let id = self.nodes.len();

        for node_id in &inputs {
            let node = &mut self.nodes[*node_id];
            node.outputs.push(id);
        }

        self.nodes.push(Node {
            outputs: Vec::new(),
            op,
            inputs,
            shape,
        });

        id
    }

    pub fn input(&mut self, shape: &[MetaId]) -> NodeId {
        self.add_node(GraphOp::Input, Vec::new(), shape.to_vec())
    }

    pub fn constant_f32(&mut self, data: f32) -> NodeId {
        self.add_node(GraphOp::ConstF32(data), Vec::new(), Vec::new())
    }

    pub fn constant_i32(&mut self, data: i32) -> NodeId {
        self.add_node(GraphOp::ConstI32(data), Vec::new(), Vec::new())
    }

    pub fn constant_u32(&mut self, data: u32) -> NodeId {
        self.add_node(GraphOp::ConstU32(data), Vec::new(), Vec::new())
    }
}
