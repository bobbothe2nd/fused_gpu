use alloc::{string::String, vec, vec::Vec};
use core::{cmp::Ordering, fmt::Debug};

use crate::{
    dispatch::{
        CompilationOptions, GpuBackend, GpuBufferBackend, GpuKernelBackend,
        TargetCompilationOptions,
        backend::kernel::{
            Kernel, KernelGroup, KernelsChained, LinkedKernel, NodeInput, RawKernel, SaveIndicator,
        },
    },
    errors::{Error, ErrorKind, GraphErrorContext},
};

#[cfg(feature = "std")]
mod std_lib;

pub mod kernel;

#[cfg(feature = "wgsl")]
pub mod wgsl;

mod ops;

pub use ops::Op;

#[cfg(feature = "wgsl")]
pub type GpuContext = wgsl::GpuContext;
#[cfg(not(feature = "wgsl"))]
pub type GpuContext = NopGpuContext;

#[cfg(feature = "wgsl")]
pub type GpuBuffer = wgsl::GpuBuffer;
#[cfg(not(feature = "wgsl"))]
pub type GpuBuffer = NopGpuBuffer;

#[derive(Debug, Clone)]
pub struct NopGpuBuffer;

impl GpuBufferBackend for NopGpuBuffer {
    fn size(&self) -> u32 {
        0
    }

    fn size_bytes(&self) -> u32 {
        0
    }
}

pub struct NopGpuKernel;

impl GpuKernelBackend for NopGpuKernel {
    fn block(&self) -> &[u32; 3] {
        &[0, 0, 0]
    }

    fn iteration_space(&self) -> &[MetaId] {
        &[]
    }
}

#[derive(Clone)]
pub struct NopGpuContext;

impl NopGpuContext {
    #[allow(clippy::unused_async)]
    pub async fn new() -> Result<Self, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }
}

impl GpuBackend for NopGpuContext {
    type Buffer = NopGpuBuffer;
    type Kernel = NopGpuKernel;
    type ParamLayout = ();
    type SubmissionIndex = ();
    type Schedule<'a> = ();
    type SyncSubmissions = ();
    type Batcher<'a> = ();
    type BatchState = ();

    const TARGET_SPEC: TargetCompilationOptions<Self> =
        TargetCompilationOptions::new(false, false, false);

    fn alloc(&self, _len: usize) -> Self::Buffer {
        NopGpuBuffer
    }

    fn alloc_init(&self, _data: &[u8]) -> Self::Buffer {
        NopGpuBuffer
    }

    fn alloc_meta(&self, _data: &[u32]) -> Self::Buffer {
        NopGpuBuffer
    }

    fn compile(
        &self,
        _src: &RawKernel,
        _params: &[Param],
        _options: &CompilationOptions<Self>,
    ) -> Result<Self::Kernel, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn download(&self, _buffer: &Self::Buffer, _out: &mut [u8]) -> Result<(), Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn dispatch_kernel(
        &self,
        _batcher: &mut Self::Batcher<'_>,
        _kernel: &Self::Kernel,
        _wg: [u32; 3],
        _bindings: &[&Self::Buffer],
    ) {
    }

    fn dispatch_schedule(&self, _batcher: &mut Self::Batcher<'_>, _schedule: &Self::Schedule<'_>) {}

    fn encode(&self, _state: Self::BatchState) -> Self::SyncSubmissions {}

    fn prepare_batch(&self) -> Self::BatchState {}

    fn start_batch<'a>(&self, _state: &'a mut Self::BatchState) -> Self::Batcher<'a> {}

    fn schedule<'a>(
        &self,
        _kernels: &'a [kernel::Dependencies<kernel::Redirect<(Self::Kernel, NodeId, &[bool])>>],
        _bindings: &[&'a Self::Buffer],
        _meta: &[u32],
    ) -> Result<Self::Schedule<'a>, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn poll(&self) -> super::PollStatus {
        super::PollStatus::Failed
    }

    fn sync(&self, _submission_index: Self::SubmissionIndex) {}

    fn submit(&self, _submission: Self::SyncSubmissions) -> Self::SubmissionIndex {}

    fn upload(&self, _buffer: &Self::Buffer, _data: &[u8]) -> Result<Self::SubmissionIndex, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }

    fn pipe(
        &self,
        _src: &Self::Buffer,
        _dst: &Self::Buffer,
    ) -> Result<Self::SubmissionIndex, Error> {
        Err(Error {
            msg: "using nop backend",
            kind: ErrorKind::UnsupportedFeature,
            ctx: (),
        })
    }
}

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
    pub pid: ParamId,
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
#[derive(Debug, Clone, Copy, PartialEq)]
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

#[derive(Debug)]
pub struct Node<'a, B: GpuBackend = GpuContext> {
    pub op: GraphOp<'a, B>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub shape: Vec<MetaId>,
}

impl<B: GpuBackend> Clone for Node<'_, B> {
    fn clone(&self) -> Self {
        Self {
            op: self.op,
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            shape: self.shape.clone(),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Hash)]
pub struct LossType {
    /// Signature:
    /// ```ignore
    /// fn(
    ///     kernel: &mut Kernel,
    ///     pred: ValueId,
    ///     target: ValueId,
    ///     pred_param: ParamId,
    ///     target_param: ParamId,
    ///     row: ValueId,
    ///     col: ValueId,
    /// ) -> (
    ///     loss_val: ValueId,
    ///     grad_val: ValueId
    /// )
    /// ```
    pub lower:
        fn(&mut Kernel, ValueId, ValueId, ParamId, ParamId, ValueId, ValueId) -> (ValueId, ValueId),
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
}

impl PartialOrd for DispatchOptions {
    fn ge(&self, other: &Self) -> bool {
        self == other
    }

    fn gt(&self, other: &Self) -> bool {
        *self != Self::Any && *other == Self::Any
    }

    fn le(&self, other: &Self) -> bool {
        self == other
    }

    fn lt(&self, other: &Self) -> bool {
        *self == Self::Any && *other != Self::Any
    }

    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }

        match (self, other) {
            (Self::Any, _) => Some(Ordering::Less),
            (_, Self::Any) => Some(Ordering::Greater),
            _ => None,
        }
    }
}

/// A linear operation in an execution graph.
///
/// This treats lowering functions as unique identifiers, so don't reuse functions.
#[non_exhaustive]
#[derive(Debug)]
pub enum GraphOp<'a, B: GpuBackend = GpuContext> {
    Input,
    ConstF32(f32),

    Custom {
        lower: fn(
            fn(
                NodeId,
                NodeId,
                &NodeInput,
                ValueId,
                &mut Vec<NodeId>,
                &'a Graph<'a, B>,
                &[Option<ParamId>],
                &[Option<ParamId>],
                &mut LinkedKernel<'a, B>,
                ValueId,
                ValueId,
                ValueId,
                ValueId,
                u32,
                ValueId,
                &mut bool,
                &CompilationOptions<B>,
            ) -> Result<Vec<NodeId>, Error>,
            NodeId,
            NodeId,
            &mut Vec<NodeId>,
            Option<u8>,
            NodeId,
            &'a Graph<'a, B>,
            ValueId,
            &[Option<ParamId>],
            &[Option<ParamId>],
            &mut LinkedKernel<'a, B>,
            ValueId,
            ValueId,
            ValueId,
            ValueId,
            u32,
            ValueId,
            &mut bool,
            &CompilationOptions<B>,
        ) -> Result<Vec<NodeId>, Error>,
        display: fn(&[Vec<MetaId>]) -> String,
        save: fn(NodeId, &Node<'_, B>, &Graph<'_, B>, &mut [SaveIndicator]),
        valid_shape:
            fn(NodeId, &Node<'a, B>, &Graph<'a, B>, &mut Vec<Error<GraphErrorContext<'a, B>>>),
        arity: u8,
        need_dims: bool,
        stable_iter: bool,
        auto_save: bool,
        computes_gid: bool,
        prefer_separate: bool,
        valid_dispatch: DispatchOptions,
    },
}

impl<B: GpuBackend> Copy for GraphOp<'_, B> {}

impl<B: GpuBackend> Clone for GraphOp<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: GpuBackend> PartialEq for GraphOp<'_, B> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Input, Self::Input) => true,
            (Self::ConstF32(a), Self::ConstF32(b)) => a == b,
            (Self::Custom { lower: a_lower, .. }, Self::Custom { lower: b_lower, .. }) => {
                (*a_lower as usize) == (*b_lower as usize)
            }
            _ => false,
        }
    }
}

impl<B: GpuBackend> GraphOp<'_, B> {
    #[inline]
    #[must_use]
    pub const fn is_elementwise(&self) -> bool {
        if let Self::Custom { need_dims, .. } = self {
            !*need_dims
        } else {
            false
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_transform(&self) -> bool {
        if let Self::Custom { stable_iter, .. } = self {
            !*stable_iter
        } else {
            false
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        matches!(self, Self::Input | Self::ConstF32(_))
    }

    #[inline]
    #[must_use]
    pub const fn is_const(&self) -> bool {
        matches!(self, Self::ConstF32(_))
    }

    #[inline]
    #[must_use]
    pub const fn is_input(&self) -> bool {
        matches!(self, Self::Input)
    }

    #[inline]
    #[must_use]
    pub const fn is_auto_save(&self) -> bool {
        if let Self::Custom { auto_save, .. } = self {
            *auto_save
        } else {
            false
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_compute_gid(&self) -> bool {
        if let Self::Custom { computes_gid, .. } = self {
            *computes_gid
        } else {
            false
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_prefer_separate(&self) -> bool {
        if let Self::Custom {
            prefer_separate, ..
        } = self
        {
            *prefer_separate
        } else {
            false
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_need_dims(&self) -> bool {
        if let Self::Custom { need_dims, .. } = self {
            *need_dims
        } else {
            false
        }
    }

    #[inline]
    #[must_use]
    pub const fn valid_dispatch(&self) -> DispatchOptions {
        if let Self::Custom { valid_dispatch, .. } = self {
            *valid_dispatch
        } else {
            DispatchOptions::Any
        }
    }

    #[inline]
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
fn check_acrylicity<B: GpuBackend>(
    graph: &Graph<B>,
    errors: &mut Vec<Error<GraphErrorContext<B>>>,
) {
    #[inline]
    fn dfs<B: GpuBackend>(
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
fn check_inputs_exist<B: GpuBackend>(
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
fn check_metadata<B: GpuBackend>(
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

fn check_shapes<'a, B: GpuBackend>(
    graph: &Graph<'a, B>,
    errors: &mut Vec<Error<GraphErrorContext<'a, B>>>,
) {
    for (node_id, node) in graph.nodes.iter().enumerate() {
        if node.shape.len() < 2 && !node.op.is_leaf() {
            errors.push(Error {
                msg: "no compute node can have less than two dimensions",
                kind: ErrorKind::ComputeGraphError,
                ctx: GraphErrorContext::LowRank {
                    node: node_id,
                    rank: node.shape.len(),
                    required: 2,
                },
            });
        }

        match node.op {
            GraphOp::Custom { valid_shape, .. } => {
                valid_shape(node_id, node, graph, errors);
            }

            GraphOp::Input | GraphOp::ConstF32(_) => {
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

#[derive(Debug)]
pub struct Graph<'a, B: GpuBackend = GpuContext> {
    pub(crate) nodes: Vec<Node<'a, B>>,
    pub(crate) loss: LossType,
}

impl<'a, B: GpuBackend> Graph<'a, B> {
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

    /// Topologically sorts the nodes in the graph by inputs.
    ///
    /// After sorting, it's important to rebuild the outputs ([`Self::rebuild_outputs`]) before
    /// lowering. Its probably a good idea to sort it before lowering and validate it before
    /// sorting too.
    ///
    /// # Errors
    ///
    /// Fails if the graph is structurally broken (e.g. containing loops).
    pub fn topo_sort<'b>(&'b mut self) -> Result<(), Error<GraphErrorContext<'a, B>>> {
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
    /// # Errors
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
    pub fn validate(&self, meta: Metadata) -> Result<(), Vec<Error<GraphErrorContext<'a, B>>>> {
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

    #[inline]
    #[must_use]
    pub fn compute_saved_nodes(&self) -> Vec<SaveIndicator> {
        let mut saved = vec![SaveIndicator { flags: 0 }; self.nodes.len()];

        for (node_id, node) in self.nodes.iter().enumerate() {
            match node.op {
                GraphOp::Input => {
                    saved[node_id] |=
                        SaveIndicator::DEFINED_IN_BACKWARD | SaveIndicator::USED_BY_BACKWARD;
                }

                GraphOp::Custom { save, .. } => {
                    save(node_id, node, self, &mut saved);
                }

                _ => {}
            }
        }

        saved
    }

    /// Lowers an execution graph into kernels.
    ///
    /// # Errors
    ///
    ///
    pub fn lower(
        &'a self,
        meta: Metadata,
        options: &CompilationOptions<B>,
        saved: &[SaveIndicator],
    ) -> Result<KernelGroup<'a, B>, Error> {
        KernelsChained::lower(self, meta, saved, options)
    }

    fn add_node(&mut self, op: GraphOp<'a, B>, inputs: Vec<NodeId>, shape: Vec<MetaId>) -> NodeId {
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

    pub fn repeat<I>(&mut self, count: usize, mut start: I, structure: fn(&mut Self, I) -> I) {
        for _ in 0..count {
            start = structure(self, start);
        }
    }
}
