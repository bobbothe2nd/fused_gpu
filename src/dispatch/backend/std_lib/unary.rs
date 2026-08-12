use crate::{
    dispatch::{
        CompilationOptions, GpuBackend,
        backend::{
            DType, DispatchOptions, Graph, GraphOp, Node, NodeId, Op, ParamId, ValueId, ValueState,
            kernel::{LinkedKernel, NodeInput, SaveIndicator},
        },
    },
    errors::{Error, ErrorKind, GraphErrorContext},
};

use alloc::{format, vec, vec::Vec};

fn save_unary<B: GpuBackend>(
    _: NodeId,
    node: &Node<B>,
    graph: &Graph<B>,
    saved: &mut [SaveIndicator],
) {
    let inp = node.inputs[0];

    if !graph.nodes[inp].inputs.is_empty() {
        saved[inp] |= SaveIndicator::DEFINED_IN_FORWARD | SaveIndicator::USED_BY_BACKWARD;
    }
}

fn valid_unary<B: GpuBackend>(
    node_id: NodeId,
    node: &Node<B>,
    _: &Graph<B>,
    errors: &mut Vec<Error<GraphErrorContext<B>>>,
) {
    if node.inputs.len() != 1 {
        errors.push(Error {
            msg: "unary operation has invalid input count",
            kind: ErrorKind::ComputeGraphError,
            ctx: GraphErrorContext::InvalidInputs {
                node: node_id,
                arity: 1,
                args: node.inputs.len(),
            },
        });
    }
}

macro_rules! lower_unary {
    (0, $out:expr, $kernel:expr, $saved_param:expr, $index:expr) => {
        $kernel.param_store($saved_param, $index, $out)
    };

    (
        $load_forward:tt,
        $op:expr,
        $back_op:expr,
        $($auto_save:tt)?
    ) => {
        |eval_node: fn(
            NodeId,
            NodeId,
            &NodeInput,
            ValueId,
            &mut Vec<NodeId>,
            &Graph<B>,
            &[Option<ParamId>],
            &[Option<ParamId>],
            &mut LinkedKernel,
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
        kernel: &mut LinkedKernel,
        base: ValueId,
        idx: ValueId,
        local_row: ValueId,
        local_col: ValueId,
        shared_size: u32,
        tile_size: ValueId,
        stable_iteration_space: bool,
        options: &CompilationOptions<B>| {
            let mut deepest;

            match backwardness {
                None => {
                    let node_input = graph.nodes[node_id].inputs[0];
                    let upstream = kernel.raw.def_var(
                        DType::Float,
                        ValueState::Mut,
                        Some(Op::ConstF32 { value: 0.0 }),
                    );
                    deepest = eval_node(
                        root,
                        input,
                        &NodeInput::Node(node_input),
                        upstream,
                        resolved,
                        graph,
                        node_params,
                        saved_params,
                        kernel,
                        idx,
                        base,
                        local_row,
                        local_col,
                        shared_size,
                        tile_size,
                        stable_iteration_space,
                        options,
                    )?;

                    #[allow(unused_variables)]
                    let save = $op(kernel, out, upstream);

                    $(
                        lower_unary!(
                            $auto_save,
                            save,
                            kernel,
                            saved_params[node_id].ok_or(Error {
                                msg: "could not materialize saved forward param",
                                kind: ErrorKind::ParamNotMaterialized,
                                ctx: (),
                            })?,
                            idx
                        );
                    )?
                }

                Some(0) => {
                    let node = &graph.nodes[node_id];

                    let saved = kernel.raw.def_var(
                        DType::Float,
                        ValueState::Mut,
                        Some(Op::ConstF32 { value: 0.0 }),
                    );
                    deepest = if $load_forward {
                        eval_node(
                            root,
                            input,
                            &NodeInput::Raw {
                                param: saved_params[node_id].ok_or(Error {
                                    msg: "could not materialize saved forward param",
                                    kind: ErrorKind::ParamNotMaterialized,
                                    ctx: (),
                                })?,
                                shape: &node.shape,
                            },
                            saved,
                            resolved,
                            graph,
                            node_params,
                            saved_params,
                            kernel,
                            idx,
                            base,
                            local_row,
                            local_col,
                            shared_size,
                            tile_size,
                            stable_iteration_space,
                            options,
                        )?
                    } else {
                        Vec::new()
                    };

                    let acc = kernel.raw.def_var(
                        DType::Float,
                        ValueState::Mut,
                        Some(Op::ConstF32 { value: 0.0 }),
                    );

                    let mut back_deepest = eval_node(
                        root,
                        input,
                        &NodeInput::Node(node_id),
                        acc,
                        resolved,
                        graph,
                        node_params,
                        saved_params,
                        kernel,
                        idx,
                        base,
                        local_row,
                        local_col,
                        shared_size,
                        tile_size,
                        stable_iteration_space,
                        options,
                    )?;

                    deepest.append(&mut back_deepest);

                    $back_op(kernel, out, saved, acc);
                }

                _ => {
                    return Err(Error {
                        msg: "backwardness must be restricted to input count",
                        kind: ErrorKind::UnresolvedInput,
                        ctx: (),
                    });
                }
            };

            Ok(deepest)
        }
    };
}

impl<B: GpuBackend> Graph<B> {
    pub fn log(&mut self, x: NodeId) -> NodeId {
        self.add_node(
            GraphOp::Custom {
                lower: lower_unary!(
                    true,
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId| kernel
                        .raw
                        .overwrite_var(out, Op::Log { x: inp }),
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId, upstream: ValueId| {
                        kernel.raw.accum_var(
                            out,
                            Op::Div {
                                a: upstream,
                                b: inp,
                            },
                        )
                    },
                ),
                display: |inputs| format!("log2({:?})", inputs[0]),
                save: save_unary::<B>,
                valid_shape: valid_unary::<B>,
                arity: 1,
                need_dims: false,
                stable_iter: true,
                auto_save: true,
                computes_gid: true,
                prefer_separate: false,
                valid_dispatch: DispatchOptions::Any,
            },
            vec![x],
            self.nodes[x].shape.clone(),
        )
    }

    pub fn tanh(&mut self, x: NodeId) -> NodeId {
        self.add_node(
            GraphOp::Custom {
                lower: lower_unary!(
                    true,
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId| kernel
                        .raw
                        .overwrite_var(out, Op::Tanh { x: inp }),
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId, upstream: ValueId| {
                        let one = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 { value: 1.0 }),
                        );
                        let forward_squared = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Mul { a: inp, b: inp }),
                        );
                        let one_minus_forward_squared = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Sub {
                                a: one,
                                b: forward_squared,
                            }),
                        );
                        kernel.raw.accum_var(
                            out,
                            Op::Mul {
                                a: upstream,
                                b: one_minus_forward_squared,
                            },
                        );
                    },
                ),
                display: |inputs| format!("tanh({:?})", inputs[0]),
                save: save_unary::<B>,
                valid_shape: valid_unary::<B>,
                arity: 1,
                need_dims: false,
                stable_iter: true,
                auto_save: true,
                computes_gid: true,
                prefer_separate: false,
                valid_dispatch: DispatchOptions::Any,
            },
            vec![x],
            self.nodes[x].shape.clone(),
        )
    }

    pub fn elu(&mut self, x: NodeId) -> NodeId {
        self.add_node(
            GraphOp::Custom {
                lower: lower_unary!(
                    true,
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId| {
                        let zero = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 { value: 0.0 }),
                        );
                        let one = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 { value: 1.0 }),
                        );

                        let exp_inp = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Exp { x: inp }),
                        );
                        let exp_inp_minus_one = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Sub { a: exp_inp, b: one }),
                        );

                        let cond = kernel.raw.def_var(
                            DType::Bool,
                            ValueState::Inline,
                            Some(Op::Ge { a: inp, b: zero }),
                        );
                        kernel.raw.overwrite_var(
                            out,
                            Op::Select {
                                cond,
                                a: inp,
                                b: exp_inp_minus_one,
                            },
                        );
                    },
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId, upstream: ValueId| {
                        let zero = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 { value: 0.0 }),
                        );
                        let one = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 { value: 1.0 }),
                        );

                        let forward_plus_one = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Add { a: inp, b: one }),
                        );
                        let upstream_forward_plus_one = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Mul {
                                a: upstream,
                                b: forward_plus_one,
                            }),
                        );

                        let cond = kernel.raw.def_var(
                            DType::Bool,
                            ValueState::Inline,
                            Some(Op::Ge { a: inp, b: zero }),
                        );
                        kernel.raw.accum_var(
                            out,
                            Op::Select {
                                cond,
                                a: upstream,
                                b: upstream_forward_plus_one,
                            },
                        );
                    },
                ),
                display: |inputs| format!("ELU({:?})", inputs[0]),
                save: save_unary::<B>,
                valid_shape: valid_unary::<B>,
                arity: 1,
                need_dims: false,
                stable_iter: true,
                auto_save: true,
                computes_gid: true,
                prefer_separate: false,
                valid_dispatch: DispatchOptions::Any,
            },
            vec![x],
            self.nodes[x].shape.clone(),
        )
    }

    pub fn relu(&mut self, x: NodeId) -> NodeId {
        self.add_node(
            GraphOp::Custom {
                lower: lower_unary!(
                    true,
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId| {
                        let zero = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 { value: 0.0 }),
                        );

                        kernel.raw.overwrite_var(out, Op::Max { a: inp, b: zero });
                    },
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId, upstream: ValueId| {
                        let zero = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 { value: 0.0 }),
                        );

                        let cond = kernel.raw.def_var(
                            DType::Bool,
                            ValueState::Inline,
                            Some(Op::Ge { a: inp, b: zero }),
                        );

                        kernel.push_if(cond, |kernel| {
                            kernel.raw.accum_var(out, Op::CopyVar { id: upstream });
                        });
                    },
                ),
                display: |inputs| format!("ReLU({:?})", inputs[0]),
                save: save_unary::<B>,
                valid_shape: valid_unary::<B>,
                arity: 1,
                need_dims: false,
                stable_iter: true,
                auto_save: true,
                computes_gid: true,
                prefer_separate: false,
                valid_dispatch: DispatchOptions::Any,
            },
            vec![x],
            self.nodes[x].shape.clone(),
        )
    }

    pub fn gelu(&mut self, x: NodeId) -> NodeId {
        const THAT_RANDOM_DECIMAL: f32 = 0.044_715;
        const SQRT_FRAC_PI_2: f32 = 0.797_884_6;

        self.add_node(
            GraphOp::Custom {
                lower: lower_unary!(
                    true,
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId| {
                        let gelu_a = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 {
                                value: THAT_RANDOM_DECIMAL,
                            }),
                        );
                        let gelu_b = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 {
                                value: SQRT_FRAC_PI_2,
                            }),
                        );
                        let half = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 { value: 0.5 }),
                        );
                        let one = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 { value: 1.0 }),
                        );

                        let xx = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Mul { a: inp, b: inp }),
                        );

                        let xxx = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Mul { a: xx, b: inp }),
                        );

                        let a_x3 = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Mul { a: xxx, b: gelu_a }),
                        );

                        let x_plus_a_x3 = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Add { a: inp, b: a_x3 }),
                        );

                        let b_x_plus_a_x3 = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Mul {
                                a: gelu_b,
                                b: x_plus_a_x3,
                            }),
                        );

                        let tanh_u = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Tanh { x: b_x_plus_a_x3 }),
                        );

                        let one_plus_tanh_u = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Add { a: one, b: tanh_u }),
                        );

                        let half_x = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Mul { a: half, b: inp }),
                        );

                        kernel.raw.overwrite_var(
                            out,
                            Op::Mul {
                                a: half_x,
                                b: one_plus_tanh_u,
                            },
                        );

                        tanh_u
                    },
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId, upstream: ValueId| {
                        let half = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 { value: 0.5 }),
                        );
                        let one = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 { value: 1.0 }),
                        );
                        let three = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 { value: 3.0 }),
                        );
                        let gelu_a = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 {
                                value: THAT_RANDOM_DECIMAL,
                            }),
                        );
                        let gelu_b = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 {
                                value: SQRT_FRAC_PI_2,
                            }),
                        );
                        let gelu_c = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Const,
                            Some(Op::Mul {
                                a: gelu_a,
                                b: three,
                            }),
                        );

                        let xx = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Mul { a: inp, b: inp }),
                        );

                        let one_plus_t = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Immut,
                            Some(Op::Add { a: one, b: inp }),
                        );

                        let tt = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Mul { a: inp, b: inp }),
                        );

                        let one_minus_tt = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Immut,
                            Some(Op::Sub { a: one, b: tt }),
                        );

                        let c_xx = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Mul { a: gelu_c, b: xx }),
                        );

                        let one_plus_c_xx = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Add { a: one, b: c_xx }),
                        );

                        let du_dx = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Immut,
                            Some(Op::Mul {
                                a: gelu_b,
                                b: one_plus_c_xx,
                            }),
                        );

                        let x_du_dx = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Immut,
                            Some(Op::Mul { a: inp, b: du_dx }),
                        );

                        let one_minus_tt_x_du_dx = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Immut,
                            Some(Op::Mul {
                                a: one_minus_tt,
                                b: x_du_dx,
                            }),
                        );

                        let two_dy_dx = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Immut,
                            Some(Op::Add {
                                a: one_plus_t,
                                b: one_minus_tt_x_du_dx,
                            }),
                        );

                        let dy_dx = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Immut,
                            Some(Op::Mul {
                                a: half,
                                b: two_dy_dx,
                            }),
                        );

                        kernel.raw.accum_var(
                            out,
                            Op::Mul {
                                a: upstream,
                                b: dy_dx,
                            },
                        );
                    },
                    0
                ),
                display: |inputs| format!("GELU({:?})", inputs[0]),
                save: save_unary::<B>,
                valid_shape: valid_unary::<B>,
                arity: 1,
                need_dims: false,
                stable_iter: true,
                auto_save: false,
                computes_gid: true,
                prefer_separate: true,
                valid_dispatch: DispatchOptions::Any,
            },
            vec![x],
            self.nodes[x].shape.clone(),
        )
    }

    pub fn exp(&mut self, x: NodeId) -> NodeId {
        self.add_node(
            GraphOp::Custom {
                lower: lower_unary!(
                    true,
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId| kernel
                        .raw
                        .overwrite_var(out, Op::Exp { x: inp }),
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId, upstream: ValueId| {
                        kernel.raw.accum_var(
                            out,
                            Op::Mul {
                                a: upstream,
                                b: inp,
                            },
                        )
                    },
                ),
                display: |inputs| format!("e^{:?}", inputs[0]),
                save: save_unary::<B>,
                valid_shape: valid_unary::<B>,
                arity: 1,
                need_dims: false,
                stable_iter: true,
                auto_save: true,
                computes_gid: true,
                prefer_separate: false,
                valid_dispatch: DispatchOptions::Any,
            },
            vec![x],
            self.nodes[x].shape.clone(),
        )
    }

    pub fn abs(&mut self, x: NodeId) -> NodeId {
        self.add_node(
            GraphOp::Custom {
                lower: lower_unary!(
                    true,
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId| kernel
                        .raw
                        .overwrite_var(out, Op::Abs { x: inp }),
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId, upstream: ValueId| {
                        let zero = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::ConstF32 { value: 0.0 }),
                        );
                        let ge0 = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Ge { a: inp, b: zero }),
                        );
                        let le0 = kernel.raw.def_var(
                            DType::Float,
                            ValueState::Inline,
                            Some(Op::Ge { a: inp, b: zero }),
                        );
                        kernel.push_if_else(
                            ge0,
                            |kernel| {
                                kernel.raw.accum_var(out, Op::CopyVar { id: upstream });
                            },
                            |kernel: &mut LinkedKernel| {
                                kernel.push_if_else(
                                    le0,
                                    |kernel| {
                                        kernel.raw.accum_var(out, Op::Neg { x: upstream });
                                    },
                                    |kernel: &mut LinkedKernel| {
                                        kernel.raw.accum_var(out, Op::ConstF32 { value: 0.0 });
                                    },
                                );
                            },
                        );
                    },
                ),
                display: |inputs| format!("|{:?}|", inputs[0]),
                save: save_unary::<B>,
                valid_shape: valid_unary::<B>,
                arity: 1,
                need_dims: false,
                stable_iter: true,
                auto_save: true,
                computes_gid: true,
                prefer_separate: false,
                valid_dispatch: DispatchOptions::Any,
            },
            vec![x],
            self.nodes[x].shape.clone(),
        )
    }

    pub fn neg(&mut self, x: NodeId) -> NodeId {
        self.add_node(
            GraphOp::Custom {
                lower: lower_unary!(
                    false,
                    |kernel: &mut LinkedKernel, out: ValueId, inp: ValueId| kernel
                        .raw
                        .overwrite_var(out, Op::Neg { x: inp }),
                    |kernel: &mut LinkedKernel, out: ValueId, _: ValueId, upstream: ValueId| kernel
                        .raw
                        .accum_var(out, Op::Neg { x: upstream }),
                ),
                display: |inputs| format!("-{:?}", inputs[0]),
                save: |_node_id, node, graph, saved| {
                    let inp = node.inputs[0];

                    if !graph.nodes[inp].inputs.is_empty() {
                        saved[inp] |=
                            SaveIndicator::DEFINED_IN_FORWARD | SaveIndicator::USED_BY_BACKWARD;
                    }
                },
                valid_shape: valid_unary::<B>,
                arity: 1,
                need_dims: false,
                stable_iter: true,
                auto_save: true,
                computes_gid: true,
                prefer_separate: false,
                valid_dispatch: DispatchOptions::Any,
            },
            vec![x],
            self.nodes[x].shape.clone(),
        )
    }
}
