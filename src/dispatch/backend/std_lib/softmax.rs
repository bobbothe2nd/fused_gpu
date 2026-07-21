use crate::{
    dispatch::{
        CompilationOptions, GpuBackend, backend::{
            Axis, DType, DispatchOptions, Graph, GraphOp, Node, NodeId, Op, ParamId, ValueId, ValueState, kernel::{Kernel, NodeInput, SaveIndicator},
        },
    }, errors::{Error, ErrorKind, GraphErrorContext},
};
use alloc::{format, vec, vec::Vec};

pub fn lower_softmax_recursive<B: GpuBackend + Clone>(
    eval_node: impl Fn(
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
    root: NodeId,
    input: NodeId,
    resolved: &mut Vec<NodeId>,
    backwardness: Option<u8>,
    node_id: NodeId,
    graph: &Graph<B>,
    out: ValueId,
    node_params: &[Option<ParamId>],
    saved_params: &[Option<ParamId>],
    kernel: &mut Kernel,
    base: ValueId,
    _idx: ValueId,
    local_row: ValueId,
    local_col: ValueId,
    shared_size: u32,
    tile_size: ValueId,
    _stable_iteration_space: bool,
    options: &CompilationOptions<B>,
) -> Result<Vec<NodeId>, Error> {
    Ok(Vec::new())
}

impl<B: GpuBackend + Clone> Graph<B> {
    pub fn softmax(&mut self, x: NodeId) -> NodeId {
        fn save<B: GpuBackend + Clone>(
            node_id: NodeId,
            _node: &Node<B>,
            _graph: &Graph<B>,
            saved: &mut [SaveIndicator],
        ) {
            saved[node_id] |= SaveIndicator::DEFINED_IN_FORWARD
                | SaveIndicator::USED_BY_FORWARD
                | SaveIndicator::DEFINED_IN_BACKWARD
                | SaveIndicator::USED_BY_BACKWARD;
        }

        fn valid_shape<B: GpuBackend + Clone>(
            node_id: NodeId,
            node: &Node<B>,
            graph: &Graph<B>,
            errors: &mut Vec<Error<GraphErrorContext<B>>>,
        ) {
            if node.inputs.len() != 1 {
                errors.push(Error {
                    msg: "softmax has invalid input count",
                    kind: ErrorKind::ComputeGraphError,
                    ctx: GraphErrorContext::InvalidInputs {
                        node: node_id,
                        arity: 1,
                        args: node.inputs.len(),
                    },
                });

                return;
            }
        }

        self.add_node(
            GraphOp::Custom {
                lower: lower_softmax_recursive::<B>,
                arity: 1,
                need_dims: true,
                stable_iter: false,
                auto_save: true,
                save,
                valid_shape,
                display: |inputs| format!("softmax({:?})", inputs[0]),
                iter_space: vec![true, false],
                valid_dispatch: DispatchOptions::ReqCol,
            },
            vec![x],
            self.nodes[x].shape.clone(),
        )
    }
}
