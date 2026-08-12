use crate::{
    dispatch::{
        CompilationOptions, GpuBackend,
        backend::{
            Axis, DType, DispatchOptions, Graph, GraphOp, Node, NodeId, Op, ParamId, ValueId,
            ValueState,
            kernel::{LinkedKernel, NodeInput, SaveIndicator},
        },
    },
    errors::{Error, ErrorKind, GraphErrorContext},
};
use alloc::{format, vec, vec::Vec};

pub fn lower_matmul_recursive<'a, B: GpuBackend>(
    eval_node: impl Fn(
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
        bool,
        &CompilationOptions<B>,
    ) -> Result<Vec<NodeId>, Error>,
    root: NodeId,
    input: NodeId,
    resolved: &mut Vec<NodeId>,
    backwardness: Option<u8>,
    node_id: NodeId,
    graph: &'a Graph<'a, B>,
    out: ValueId,
    node_params: &[Option<ParamId>],
    saved_params: &[Option<ParamId>],
    kernel: &mut LinkedKernel<'a, B>,
    base: ValueId,
    _idx: ValueId,
    local_row: ValueId,
    local_col: ValueId,
    shared_size: u32,
    tile_size: ValueId,
    _stable_iteration_space: bool,
    options: &CompilationOptions<B>,
) -> Result<Vec<NodeId>, Error> {
    let row = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Immut,
        Some(Op::GlobalId { axis: Axis::Y }),
    );
    let col = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Immut,
        Some(Op::GlobalId { axis: Axis::X }),
    );

    let node = &graph.nodes[node_id];

    let (a_node, b_node);

    if backwardness == Some(0) {
        let param = saved_params[node.inputs[1]].ok_or(Error {
            msg: "saved input parameter could not be materialized",
            kind: ErrorKind::ParamNotMaterialized,
            ctx: (),
        })?;
        let shape = &graph.nodes[node.inputs[1]].shape;

        a_node = NodeInput::Node(node_id);
        b_node = NodeInput::Raw { param, shape };
    } else if backwardness == Some(1) {
        let param = saved_params[node.inputs[0]].ok_or(Error {
            msg: "saved input parameter could not be materialized",
            kind: ErrorKind::ParamNotMaterialized,
            ctx: (),
        })?;
        let shape = &graph.nodes[node.inputs[0]].shape;

        a_node = NodeInput::Raw { param, shape };
        b_node = NodeInput::Node(node_id);
    } else {
        a_node = NodeInput::Node(node.inputs[0]);
        b_node = NodeInput::Node(node.inputs[1]);
    }

    let transpose_a = backwardness == Some(1);
    let transpose_b = backwardness == Some(0);

    let mut a_node_shape = match a_node {
        NodeInput::Node(node) => graph.nodes[node].shape.clone(),
        NodeInput::Raw { param: _, shape } => shape.to_vec(),
    };

    if transpose_a {
        let len = a_node_shape.len();
        a_node_shape.swap(len - 1, len - 2);
    }

    let mut b_node_shape = match b_node {
        NodeInput::Node(node) => graph.nodes[node].shape.clone(),
        NodeInput::Raw { param: _, shape } => shape.to_vec(),
    };

    if transpose_b {
        let len = b_node_shape.len();
        b_node_shape.swap(len - 1, len - 2);
    }

    let m = a_node_shape[a_node_shape.len() - 2];
    let n = b_node_shape[b_node_shape.len() - 1];
    let k = a_node_shape[a_node_shape.len() - 1];

    kernel.register_meta(m);
    kernel.register_meta(n);
    kernel.register_meta(k);

    let m = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Immut,
        Some(Op::ReadMeta {
            param: 0,
            field: m,
        }),
    );
    let n = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Immut,
        Some(Op::ReadMeta {
            param: 0,
            field: n,
        }),
    );
    let k = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Immut,
        Some(Op::ReadMeta {
            param: 0,
            field: k,
        }),
    );

    if backwardness.is_none() {
        kernel.raw.overwrite_var(out, Op::ConstF32 { value: 0.0 });
    }

    forward_matmul(
        eval_node,
        root,
        input,
        resolved,
        m,
        n,
        k,
        transpose_a,
        transpose_b,
        &a_node,
        &b_node,
        graph,
        out,
        node_params,
        saved_params,
        kernel,
        base,
        row,
        col,
        local_row,
        local_col,
        shared_size,
        tile_size,
        options,
    )
}

pub fn forward_matmul<'a, B: GpuBackend>(
    eval_node: impl Fn(
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
        bool,
        &CompilationOptions<B>,
    ) -> Result<Vec<NodeId>, Error>,
    root: NodeId,
    input: NodeId,
    resolved: &mut Vec<NodeId>,
    m: ValueId,
    n: ValueId,
    k: ValueId,
    swap_a: bool,
    swap_b: bool,
    a_node: &NodeInput,
    b_node: &NodeInput,
    graph: &'a Graph<'a, B>,
    out: ValueId,
    node_params: &[Option<ParamId>],
    saved_params: &[Option<ParamId>],
    kernel: &mut LinkedKernel<'a, B>,
    base: ValueId,
    row: ValueId,
    col: ValueId,
    local_row: ValueId,
    local_col: ValueId,
    shared_size: u32,
    tile_size: ValueId,
    options: &CompilationOptions<B>,
) -> Result<Vec<NodeId>, Error> {
    let mut deepest = Vec::new();

    let a_tile = kernel.raw.new_shared(DType::Float, shared_size);
    let b_tile = kernel.raw.new_shared(DType::Float, shared_size);

    let one = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Inline,
        Some(Op::ConstU32 { value: 1 }),
    );

    let tk = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Mut,
        Some(Op::ConstU32 { value: 0 }),
    );

    let tile_row = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Inline,
        Some(Op::Mul {
            a: local_row,
            b: tile_size,
        }),
    );
    let shared_idx = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Mut,
        Some(Op::Add {
            a: tile_row,
            b: local_col,
        }),
    );

    let mut a_deepest = Vec::new();
    let mut b_deepest = Vec::new();

    kernel.push_for_loop(tk, k, tile_size, |kernel| {
        let a_k = kernel.raw.def_var(
            DType::UnsignedInt,
            ValueState::Immut,
            Some(Op::Add {
                a: tk,
                b: local_col,
            }),
        );

        let b_k = kernel.raw.def_var(
            DType::UnsignedInt,
            ValueState::Immut,
            Some(Op::Add {
                a: tk,
                b: local_row,
            }),
        );

        let a_idx = if swap_a {
            let a_row = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Mul { a: a_k, b: m }),
            );
            let a_col = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Add { a: a_row, b: row }),
            );
            kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::Add { a: a_col, b: base }),
            )
        } else {
            let a_row = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Mul { a: row, b: k }),
            );
            let a_col = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Add { a: a_row, b: a_k }),
            );
            kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::Add { a: a_col, b: base }),
            )
        };

        let a_val = kernel.raw.def_var(
            DType::Float,
            ValueState::Mut,
            Some(Op::ConstF32 { value: 0.0 }),
        );

        a_deepest = eval_node(
            root,
            input,
            a_node,
            a_val,
            resolved,
            graph,
            node_params,
            saved_params,
            kernel,
            a_idx,
            base,
            local_row,
            local_col,
            shared_size,
            tile_size,
            false,
            options,
        )?;

        kernel.raw.shared_store(a_tile, shared_idx, a_val);

        let b_idx = if swap_b {
            let b_row = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Mul { a: col, b: k }),
            );
            let b_col = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Add { a: b_row, b: b_k }),
            );
            kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::Add { a: b_col, b: base }),
            )
        } else {
            let b_row = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Mul { a: b_k, b: n }),
            );
            let b_col = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Add { a: b_row, b: col }),
            );
            kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::Add { a: b_col, b: base }),
            )
        };

        let b_val = kernel.raw.def_var(
            DType::Float,
            ValueState::Mut,
            Some(Op::ConstF32 { value: 0.0 }),
        );

        b_deepest = eval_node(
            root,
            input,
            b_node,
            b_val,
            resolved,
            graph,
            node_params,
            saved_params,
            kernel,
            b_idx,
            base,
            local_row,
            local_col,
            shared_size,
            tile_size,
            false,
            options,
        )?;

        kernel.raw.shared_store(b_tile, shared_idx, b_val);

        kernel.raw.push_barrier();

        let inner = kernel.raw.def_var(
            DType::UnsignedInt,
            ValueState::Mut,
            Some(Op::ConstU32 { value: 0 }),
        );

        kernel.push_for_loop(inner, tile_size, one, |kernel| {
            let a_s_row = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Mul {
                    a: local_row,
                    b: tile_size,
                }),
            );
            let a_s_idx = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::Add {
                    a: a_s_row,
                    b: inner,
                }),
            );

            let a_val = kernel.raw.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::SharedLoad {
                    mem: a_tile,
                    index: a_s_idx,
                }),
            );

            let b_s_row = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Mul {
                    a: inner,
                    b: tile_size,
                }),
            );
            let b_s_idx = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::Add {
                    a: b_s_row,
                    b: local_col,
                }),
            );

            let b_val = kernel.raw.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::SharedLoad {
                    mem: b_tile,
                    index: b_s_idx,
                }),
            );

            kernel.raw.overwrite_var(
                out,
                Op::Fma {
                    a: a_val,
                    b: b_val,
                    c: out,
                },
            );

            Ok(())
        })?;

        kernel.raw.push_barrier();

        Ok(())
    })?;

    deepest.append(&mut a_deepest);
    deepest.append(&mut b_deepest);

    Ok(deepest)
}

impl<'a, B: GpuBackend> Graph<'a, B> {
    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        fn save<B: GpuBackend>(
            node_id: NodeId,
            node: &Node<'_, B>,
            graph: &Graph<'_, B>,
            saved: &mut [SaveIndicator],
        ) {
            saved[node_id] |= SaveIndicator::DEFINED_IN_FORWARD
                | SaveIndicator::USED_BY_FORWARD
                | SaveIndicator::DEFINED_IN_BACKWARD
                | SaveIndicator::USED_BY_BACKWARD;

            for &inp in &node.inputs {
                if !graph.nodes[inp].inputs.is_empty() {
                    saved[inp] |=
                        SaveIndicator::DEFINED_IN_FORWARD | SaveIndicator::USED_BY_FORWARD;
                }
            }
        }

        fn valid_shape<'a, B: GpuBackend>(
            node_id: NodeId,
            node: &Node<'a, B>,
            graph: &Graph<'a, B>,
            errors: &mut Vec<Error<GraphErrorContext<'a, B>>>,
        ) {
            if node.inputs.len() != 2 {
                errors.push(Error {
                    msg: "matrix multiplication has invalid input count",
                    kind: ErrorKind::ComputeGraphError,
                    ctx: GraphErrorContext::InvalidInputs {
                        node: node_id,
                        arity: 2,
                        args: node.inputs.len(),
                    },
                });

                return;
            }

            let Some(a) = graph.nodes.get(node.inputs[0]) else {
                return;
            };
            let Some(b) = graph.nodes.get(node.inputs[1]) else {
                return;
            };

            let rank_a = a.shape.len();
            let rank_b = b.shape.len();

            if rank_a != rank_b {
                errors.push(Error {
                    msg: "matrix multiplication has invalid input rank(s)",
                    kind: ErrorKind::ComputeGraphError,
                    ctx: GraphErrorContext::RankMismatch {
                        node: node_id,
                        all_hand_sides: vec![rank_a, rank_b],
                    },
                });
            }

            let k1 = a.shape[a.shape.len() - 1];
            let k2 = b.shape[b.shape.len() - 2];

            if k1 != k2 {
                errors.push(Error {
                    msg: "inner dimensions don't match for matrix multiplication",
                    kind: ErrorKind::ComputeGraphError,
                    ctx: GraphErrorContext::ShapeMismatch {
                        node: node_id,
                        all_hand_sides: vec![a.shape.clone(), b.shape.clone()],
                        op: node.op,
                    },
                });
            }

            if a.op.is_const() && b.op.is_const() {
                errors.push(Error {
                    msg: "binary operation has only constant inputs",
                    kind: ErrorKind::ComputeGraphError,
                    ctx: GraphErrorContext::CannotInferShape {
                        node: node_id,
                        all_hand_sides: vec![a.shape.clone(), b.shape.clone()],
                        op: node.op,
                    },
                });
            }
        }

        let mut shape = self.nodes[a].shape.clone();
        let last_idx = shape.len() - 1;
        shape[last_idx] = self.nodes[b].shape[last_idx];

        self.add_node(
            GraphOp::Custom {
                lower: lower_matmul_recursive::<B>,
                arity: 2,
                need_dims: true,
                stable_iter: false,
                auto_save: true,
                computes_gid: true,
                prefer_separate: true,
                save,
                valid_shape,
                display: |inputs| format!("{:?} @ {:?}", inputs[0], inputs[1]),
                valid_dispatch: DispatchOptions::Any,
            },
            vec![a, b],
            shape,
        )
    }
}
