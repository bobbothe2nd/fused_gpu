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

pub fn lower_softmax_recursive<'a, B: GpuBackend>(
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
    _out: ValueId,
    node_params: &[Option<ParamId>],
    saved_params: &[Option<ParamId>],
    kernel: &mut LinkedKernel<'a, B>,
    base: ValueId,
    _idx: ValueId,
    local_row: ValueId,
    tid: ValueId,
    shared_size: u32,
    tile_size: ValueId,
    _stable_iteration_space: bool,
    options: &CompilationOptions<B>,
) -> Result<Vec<NodeId>, Error> {
    let node = &graph.nodes[node_id];

    let saved_param = saved_params[node_id].ok_or(Error {
        msg: "could not materialize saved forward param",
        kind: ErrorKind::ParamNotMaterialized,
        ctx: (),
    })?;
    let node_param = node_params[node_id];

    let cols_field = node.shape[node.shape.len() - 1];
    let cols = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Immut,
        Some(Op::ReadMeta {
            param: 0,
            field: cols_field,
        }),
    );

    kernel.register_meta(cols_field);

    let shared_size_const = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Inline,
        Some(Op::ConstU32 { value: shared_size }),
    );
    let zero = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Inline,
        Some(Op::ConstU32 { value: 0 }),
    );
    let one = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Inline,
        Some(Op::ConstU32 { value: 1 }),
    );
    let one_f = kernel.raw.def_var(
        DType::Float,
        ValueState::Inline,
        Some(Op::ConstF32 { value: 1.0 }),
    );

    let row = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Immut,
        Some(Op::BlockId { axis: Axis::X }),
    );

    let tmp_shared = kernel.raw.new_shared(DType::Float, shared_size);

    let col = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Mut,
        Some(Op::CopyVar { id: tid }),
    );

    match backwardness {
        None => {
            let local_max = kernel.raw.def_var(
                DType::Float,
                ValueState::Mut,
                Some(Op::ConstF32 { value: f32::MIN }),
            );

            let mut x_deep = Ok(Vec::new());

            kernel.push_while_loop(Op::Lt { a: col, b: cols }, |kernel| {
                let row_flat = kernel.raw.def_var(
                    DType::UnsignedInt,
                    ValueState::Inline,
                    Some(Op::Mul { a: row, b: cols }),
                );
                let idx = kernel.raw.def_var(
                    DType::UnsignedInt,
                    ValueState::Immut,
                    Some(Op::Add {
                        a: row_flat,
                        b: col,
                    }),
                );

                let x_val = kernel.raw.def_var(DType::Float, ValueState::Mut, None);

                x_deep = eval_node(
                    root,
                    input,
                    &NodeInput::Node(node.inputs[0]),
                    x_val,
                    resolved,
                    graph,
                    node_params,
                    saved_params,
                    kernel,
                    idx,
                    base,
                    local_row,
                    tid,
                    shared_size,
                    tile_size,
                    false,
                    options,
                );

                kernel.raw.overwrite_var(
                    local_max,
                    Op::Max {
                        a: local_max,
                        b: x_val,
                    },
                );

                kernel
                    .raw
                    .accum_var(col, Op::ConstU32 { value: shared_size });

                Ok(())
            })?;

            kernel.raw.shared_store(tmp_shared, tid, local_max);

            kernel.raw.push_barrier();

            let stride_const = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Const,
                Some(Op::Shr {
                    a: shared_size_const,
                    b: one,
                }),
            );
            let stride = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Mut,
                Some(Op::CopyVar { id: stride_const }),
            );

            kernel.push_forever_loop(|kernel| {
                let tid_less_stride = kernel.raw.def_var(
                    DType::Bool,
                    ValueState::Inline,
                    Some(Op::Lt { a: tid, b: stride }),
                );

                kernel.push_if(tid_less_stride, |kernel| {
                    let scratch_tid = kernel.raw.def_var(
                        DType::Float,
                        ValueState::Inline,
                        Some(Op::SharedLoad {
                            mem: tmp_shared,
                            index: tid,
                        }),
                    );

                    let tid_stride = kernel.raw.def_var(
                        DType::UnsignedInt,
                        ValueState::Inline,
                        Some(Op::Add { a: stride, b: tid }),
                    );
                    let scratch_tid_stride = kernel.raw.def_var(
                        DType::Float,
                        ValueState::Inline,
                        Some(Op::SharedLoad {
                            mem: tmp_shared,
                            index: tid_stride,
                        }),
                    );

                    let max_scratch_stride = kernel.raw.def_var(
                        DType::Float,
                        ValueState::Inline,
                        Some(Op::Max {
                            a: scratch_tid,
                            b: scratch_tid_stride,
                        }),
                    );

                    kernel.raw.shared_store(tmp_shared, tid, max_scratch_stride);

                    Ok(())
                })?;

                kernel.raw.push_barrier();

                let stride_is_one = kernel.raw.def_var(
                    DType::Bool,
                    ValueState::Inline,
                    Some(Op::Eq { a: stride, b: one }),
                );

                kernel.push_if(stride_is_one, |kernel| {
                    kernel.raw.push_break();
                    Ok(())
                })?;

                kernel
                    .raw
                    .overwrite_var(stride, Op::Shr { a: stride, b: one });

                Ok(())
            })?;

            let row_max = kernel.raw.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::SharedLoad {
                    mem: tmp_shared,
                    index: zero,
                }),
            );

            kernel.raw.push_barrier();

            let local_sum = kernel.raw.def_var(
                DType::Float,
                ValueState::Mut,
                Some(Op::ConstF32 { value: 0.0 }),
            );

            kernel.raw.overwrite_var(col, Op::CopyVar { id: tid });

            kernel.push_while_loop(Op::Lt { a: col, b: cols }, |kernel| {
                let row_flat = kernel.raw.def_var(
                    DType::UnsignedInt,
                    ValueState::Inline,
                    Some(Op::Mul { a: row, b: cols }),
                );
                let idx = kernel.raw.def_var(
                    DType::UnsignedInt,
                    ValueState::Immut,
                    Some(Op::Add {
                        a: row_flat,
                        b: col,
                    }),
                );

                let x_val = kernel.raw.def_var(DType::Float, ValueState::Mut, None);

                eval_node(
                    root,
                    input,
                    &NodeInput::Node(node.inputs[0]),
                    x_val,
                    resolved,
                    graph,
                    node_params,
                    saved_params,
                    kernel,
                    idx,
                    base,
                    local_row,
                    tid,
                    shared_size,
                    tile_size,
                    false,
                    options,
                )?;

                let x_minus_max = kernel.raw.def_var(
                    DType::Float,
                    ValueState::Inline,
                    Some(Op::Sub {
                        a: x_val,
                        b: row_max,
                    }),
                );

                let e = kernel.raw.def_var(
                    DType::Float,
                    ValueState::Immut,
                    Some(Op::Exp { x: x_minus_max }),
                );

                kernel.param_store(saved_param, idx, e);

                if let Some(output) = node_param {
                    kernel.param_store(output, idx, e);
                }

                kernel.raw.accum_var(local_sum, Op::CopyVar { id: e });

                kernel
                    .raw
                    .accum_var(col, Op::ConstU32 { value: shared_size });

                Ok(())
            })?;

            kernel.raw.shared_store(tmp_shared, tid, local_sum);

            kernel.raw.push_barrier();

            kernel
                .raw
                .overwrite_var(stride, Op::CopyVar { id: stride_const });

            kernel.push_forever_loop(|kernel| {
                let tid_less_stride = kernel.raw.def_var(
                    DType::Bool,
                    ValueState::Inline,
                    Some(Op::Lt { a: tid, b: stride }),
                );

                kernel.push_if(tid_less_stride, |kernel| {
                    let tid_stride = kernel.raw.def_var(
                        DType::UnsignedInt,
                        ValueState::Inline,
                        Some(Op::Add { a: stride, b: tid }),
                    );
                    let scratch_tid_stride = kernel.raw.def_var(
                        DType::Float,
                        ValueState::Inline,
                        Some(Op::SharedLoad {
                            mem: tmp_shared,
                            index: tid_stride,
                        }),
                    );

                    kernel.raw.shared_accum(tmp_shared, tid, scratch_tid_stride);

                    Ok(())
                })?;

                kernel.raw.push_barrier();

                let stride_is_one = kernel.raw.def_var(
                    DType::Bool,
                    ValueState::Inline,
                    Some(Op::Eq { a: stride, b: one }),
                );

                kernel.push_if(stride_is_one, |kernel| {
                    kernel.raw.push_break();
                    Ok(())
                })?;

                kernel
                    .raw
                    .overwrite_var(stride, Op::Shr { a: stride, b: one });

                Ok(())
            })?;

            let row_sum = kernel.raw.def_var(
                DType::Float,
                ValueState::Inline,
                Some(Op::SharedLoad {
                    mem: tmp_shared,
                    index: zero,
                }),
            );

            let inv_row_sum = kernel.raw.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Div {
                    a: one_f,
                    b: row_sum,
                }),
            );

            kernel.raw.push_barrier();

            kernel.raw.overwrite_var(col, Op::CopyVar { id: tid });

            kernel.push_while_loop(Op::Lt { a: col, b: cols }, |kernel| {
                let row_flat = kernel.raw.def_var(
                    DType::UnsignedInt,
                    ValueState::Inline,
                    Some(Op::Mul { a: row, b: cols }),
                );
                let idx = kernel.raw.def_var(
                    DType::UnsignedInt,
                    ValueState::Immut,
                    Some(Op::Add {
                        a: row_flat,
                        b: col,
                    }),
                );

                kernel.param_mul(saved_param, idx, inv_row_sum);

                if let Some(output) = node_param {
                    kernel.param_mul(output, idx, inv_row_sum);
                }

                kernel
                    .raw
                    .accum_var(col, Op::ConstU32 { value: shared_size });

                Ok(())
            })?;

            x_deep
        }

        Some(0) => {
            let local_dot = kernel.raw.def_var(
                DType::Float,
                ValueState::Mut,
                Some(Op::ConstF32 { value: 0.0 }),
            );

            let mut dy_deep = Ok(Vec::new());

            kernel.push_while_loop(Op::Lt { a: col, b: cols }, |kernel| {
                let row_flat = kernel.raw.def_var(
                    DType::UnsignedInt,
                    ValueState::Inline,
                    Some(Op::Mul { a: row, b: cols }),
                );
                let idx = kernel.raw.def_var(
                    DType::UnsignedInt,
                    ValueState::Immut,
                    Some(Op::Add {
                        a: row_flat,
                        b: col,
                    }),
                );

                let dy_idx = kernel.raw.def_var(
                    DType::Float,
                    ValueState::Mut,
                    Some(Op::ConstF32 { value: 0.0 }),
                );

                dy_deep = eval_node(
                    root,
                    input,
                    &NodeInput::Node(node_id),
                    dy_idx,
                    resolved,
                    graph,
                    node_params,
                    saved_params,
                    kernel,
                    idx,
                    base,
                    local_row,
                    tid,
                    shared_size,
                    tile_size,
                    false,
                    options,
                );

                let y_idx = kernel.raw.def_var(
                    DType::Float,
                    ValueState::Inline,
                    Some(Op::ParamLoad {
                        param: saved_param,
                        index: idx,
                    }),
                );

                kernel.register_param(saved_param);

                kernel.raw.accum_var(
                    local_dot,
                    Op::Mul {
                        a: dy_idx,
                        b: y_idx,
                    },
                );

                kernel
                    .raw
                    .accum_var(col, Op::ConstU32 { value: shared_size });

                Ok(())
            })?;

            kernel.raw.shared_store(tmp_shared, tid, local_dot);

            kernel.raw.push_barrier();

            let stride_const = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Const,
                Some(Op::Shr {
                    a: shared_size_const,
                    b: one,
                }),
            );
            let stride = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Mut,
                Some(Op::CopyVar { id: stride_const }),
            );

            kernel.push_forever_loop(|kernel| {
                let tid_less_stride = kernel.raw.def_var(
                    DType::Bool,
                    ValueState::Inline,
                    Some(Op::Lt { a: tid, b: stride }),
                );

                kernel.push_if(tid_less_stride, |kernel| {
                    let tid_stride = kernel.raw.def_var(
                        DType::UnsignedInt,
                        ValueState::Inline,
                        Some(Op::Add { a: stride, b: tid }),
                    );
                    let scratch_tid_stride = kernel.raw.def_var(
                        DType::Float,
                        ValueState::Inline,
                        Some(Op::SharedLoad {
                            mem: tmp_shared,
                            index: tid_stride,
                        }),
                    );

                    kernel.raw.shared_accum(tmp_shared, tid, scratch_tid_stride);

                    Ok(())
                })?;

                kernel.raw.push_barrier();

                let stride_is_one = kernel.raw.def_var(
                    DType::Bool,
                    ValueState::Inline,
                    Some(Op::Eq { a: stride, b: one }),
                );

                kernel.push_if(stride_is_one, |kernel| {
                    kernel.raw.push_break();
                    Ok(())
                })?;

                kernel
                    .raw
                    .overwrite_var(stride, Op::Shr { a: stride, b: one });

                Ok(())
            })?;

            let row_dot = kernel.raw.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::SharedLoad {
                    mem: tmp_shared,
                    index: zero,
                }),
            );

            kernel.raw.push_barrier();

            kernel.raw.overwrite_var(col, Op::ConstU32 { value: 0 });

            kernel.push_while_loop(Op::Lt { a: col, b: cols }, |kernel| {
                let row_flat = kernel.raw.def_var(
                    DType::UnsignedInt,
                    ValueState::Inline,
                    Some(Op::Mul { a: row, b: cols }),
                );
                let idx = kernel.raw.def_var(
                    DType::UnsignedInt,
                    ValueState::Immut,
                    Some(Op::Add {
                        a: row_flat,
                        b: col,
                    }),
                );

                let dy_idx = kernel.raw.def_var(DType::Float, ValueState::Mut, None);

                eval_node(
                    root,
                    input,
                    &NodeInput::Node(node_id),
                    dy_idx,
                    resolved,
                    graph,
                    node_params,
                    saved_params,
                    kernel,
                    idx,
                    base,
                    local_row,
                    tid,
                    shared_size,
                    tile_size,
                    false,
                    options,
                )?;

                let y_idx = kernel.raw.def_var(
                    DType::Float,
                    ValueState::Inline,
                    Some(Op::ParamLoad {
                        param: saved_param,
                        index: idx,
                    }),
                );

                kernel.register_param(saved_param);

                let dy_minus_row_dot = kernel.raw.def_var(
                    DType::Float,
                    ValueState::Inline,
                    Some(Op::Sub {
                        a: dy_idx,
                        b: row_dot,
                    }),
                );

                let y_scaled_dy_dot = kernel.raw.def_var(
                    DType::Float,
                    ValueState::Inline,
                    Some(Op::Mul {
                        a: y_idx,
                        b: dy_minus_row_dot,
                    }),
                );

                let grad = node_param.ok_or(Error {
                    msg: "could not materialize grad param in softmax backward",
                    kind: ErrorKind::ParamNotMaterialized,
                    ctx: (),
                })?;

                kernel.param_accum(grad, idx, y_scaled_dy_dot);

                kernel
                    .raw
                    .accum_var(col, Op::ConstU32 { value: shared_size });

                Ok(())
            })?;

            dy_deep
        }

        Some(_) => Err(Error {
            msg: "backwardness must be restricted to input count",
            kind: ErrorKind::UnresolvedInput,
            ctx: (),
        }),
    }
}

impl<B: GpuBackend> Graph<'_, B> {
    pub fn softmax(&mut self, x: NodeId) -> NodeId {
        fn save<B: GpuBackend>(
            node_id: NodeId,
            node: &Node<B>,
            graph: &Graph<B>,
            saved: &mut [SaveIndicator],
        ) {
            saved[node_id] |= SaveIndicator::DEFINED_IN_FORWARD
                | SaveIndicator::USED_BY_FORWARD
                | SaveIndicator::DEFINED_IN_BACKWARD
                | SaveIndicator::USED_BY_BACKWARD;

            for inp in &node.inputs {
                let inp_node = &graph.nodes[*inp];

                if inp_node.op.is_need_dims() || inp_node.op.is_prefer_separate() {
                    saved[*inp] |= SaveIndicator::DEFINED_IN_FORWARD
                        | SaveIndicator::USED_BY_FORWARD
                        | SaveIndicator::DEFINED_IN_BACKWARD
                        | SaveIndicator::USED_BY_BACKWARD;
                }
            }

            for out in &node.outputs {
                saved[*out] |= SaveIndicator::DEFINED_IN_BACKWARD | SaveIndicator::USED_BY_BACKWARD;
            }
        }

        fn valid_shape<B: GpuBackend>(
            node_id: NodeId,
            node: &Node<B>,
            _graph: &Graph<B>,
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
            }
        }

        self.add_node(
            GraphOp::Custom {
                lower: lower_softmax_recursive::<B>,
                arity: 1,
                need_dims: true,
                stable_iter: false,
                auto_save: false,
                computes_gid: false,
                prefer_separate: true,
                save,
                valid_shape,
                display: |inputs| format!("softmax({:?})", inputs[0]),
                valid_dispatch: DispatchOptions::ReqCol,
            },
            vec![x],
            self.nodes[x].shape.clone(),
        )
    }
}
