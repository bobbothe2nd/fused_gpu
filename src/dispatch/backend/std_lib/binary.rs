use crate::{
    dispatch::{
        GpuBackend,
        backend::{
            CompilationOptions, DType, DispatchOptions, Graph, GraphOp, MetaId, Node,
            NodeId, NodeInput, Op, ParamId, ValueId, ValueState, kernel::{SaveIndicator, LinkedKernel},
        },
    },
    errors::{Error, ErrorKind, GraphErrorContext},
};

use alloc::{format, vec, vec::Vec};

fn valid_binary<B: GpuBackend + Clone>(
    node_id: NodeId,
    node: &Node<B>,
    graph: &Graph<B>,
    errors: &mut Vec<Error<GraphErrorContext<B>>>,
) {
    if node.inputs.len() != 2 {
        errors.push(Error {
            msg: "binary operation has invalid input count",
            kind: ErrorKind::ComputeGraphError,
            ctx: GraphErrorContext::InvalidInputs {
                node: node_id,
                arity: 2,
                args: node.inputs.len(),
            },
        });
    }

    let a = node.inputs[0];
    let b = node.inputs[1];

    let a_node = &graph.nodes[a];
    let b_node = &graph.nodes[b];

    let a_shape = &a_node.shape;
    let b_shape = &b_node.shape;

    if a_node.op.is_const() && b_node.op.is_const() {
        errors.push(Error {
            msg: "binary operation has only constant inputs",
            kind: ErrorKind::ComputeGraphError,
            ctx: GraphErrorContext::CannotInferShape {
                node: node_id,
                all_hand_sides: vec![a_shape.clone(), b_shape.clone()],
                op: node.op.clone(),
            },
        });
    }

    if !(a_node.op.is_const() || b_node.op.is_const()) && a_shape != b_shape {
        errors.push(Error {
            msg: "binary operation has different shaped inputs",
            kind: ErrorKind::ComputeGraphError,
            ctx: GraphErrorContext::ShapeMismatch {
                node: node_id,
                all_hand_sides: vec![a_shape.clone(), b_shape.clone()],
                op: node.op.clone(),
            },
        });
    }
}

fn save_mul_div<B: GpuBackend + Clone>(
    _node_id: NodeId,
    node: &Node<B>,
    graph: &Graph<B>,
    saved: &mut [SaveIndicator],
) {
    for &inp in &node.inputs {
        if !graph.nodes[inp].inputs.is_empty() {
            saved[inp] |= SaveIndicator::DEFINED_IN_FORWARD | SaveIndicator::USED_BY_BACKWARD;
        }
    }
}

impl<B: GpuBackend + Clone> Graph<B> {
    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_node(
            GraphOp::Custom {
                lower: |eval_node: fn(
                    NodeId,
                    NodeId,
                    &NodeInput,
                    ValueId,
                    &mut Vec<NodeId>,
                    &Self,
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
                        graph: &Self,
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
                    let mut deepest = Vec::new();

                    match backwardness {
                        None => {
                            let a = graph.nodes[node_id].inputs[0];
                            let b = graph.nodes[node_id].inputs[1];

                            let a_val = kernel.raw.def_var(DType::Float, ValueState::Mut, None);

                            let mut a_deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(a),
                                a_val,
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

                            let b_val = kernel.raw.def_var(DType::Float, ValueState::Mut, None);

                            let mut b_deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(b),
                                b_val,
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

                            deepest.append(&mut a_deep);
                            deepest.append(&mut b_deep);

                            kernel.raw.overwrite_var(out, Op::Add { a: a_val, b: b_val });
                        }

                        Some(0 | 1) => {
                            let mut deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(node_id),
                                out,
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

                            deepest.append(&mut deep);
                        }

                        _ => {
                            return Err(Error {
                                msg: "backwardness must be restricted to input count",
                                kind: ErrorKind::UnresolvedInput,
                                ctx: (),
                            });
                        }
                    }

                    Ok(deepest)
                },
                display: |inputs| format!("{:?} + {:?}", inputs[0], inputs[1]),
                save: |_, _, _, _| {},
                valid_shape: valid_binary::<B>,
                arity: 2,
                need_dims: false,
                stable_iter: true,
                auto_save: true,
                computes_gid: true,
                prefer_separate: false,
                valid_dispatch: DispatchOptions::Any,
            },
            vec![a, b],
            get_shape(a, b, self),
        )
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_node(
            GraphOp::Custom {
                lower: |eval_node: fn(
                    NodeId,
                    NodeId,
                    &NodeInput,
                    ValueId,
                    &mut Vec<NodeId>,
                    &Self,
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
                        graph: &Self,
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
                    let mut deepest = Vec::new();

                    match backwardness {
                        None => {
                            let a = graph.nodes[node_id].inputs[0];
                            let b = graph.nodes[node_id].inputs[1];

                            let a_val = kernel.raw.def_var(DType::Float, ValueState::Mut, None);

                            let mut a_deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(a),
                                a_val,
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

                            let b_val = kernel.raw.def_var(DType::Float, ValueState::Mut, None);

                            let mut b_deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(b),
                                b_val,
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

                            deepest.append(&mut a_deep);
                            deepest.append(&mut b_deep);

                            kernel.raw.overwrite_var(out, Op::Mul { a: a_val, b: b_val });
                        }

                        Some(0) => {
                            let g_val = kernel.raw.def_var(
                                DType::Float,
                                ValueState::Mut,
                                Some(Op::ConstF32 { value: 0.0 }),
                            );

                            let mut deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(node_id),
                                g_val,
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

                            let user = graph.nodes[node_id].inputs[1];

                            let saved = read_saved(kernel, idx, saved_params[user]);

                            kernel.raw.accum_var(out, Op::Mul { a: g_val, b: saved });

                            deepest.append(&mut deep);
                        }

                        Some(1) => {
                            let g_val = kernel.raw.def_var(
                                DType::Float,
                                ValueState::Mut,
                                Some(Op::ConstF32 { value: 0.0 }),
                            );

                            let mut deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(node_id),
                                g_val,
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

                            let user = graph.nodes[node_id].inputs[0];

                            let saved = read_saved(kernel, idx, saved_params[user]);

                            kernel.raw.accum_var(out, Op::Mul { a: g_val, b: saved });

                            deepest.append(&mut deep);
                        }

                        _ => {
                            return Err(Error {
                                msg: "backwardness must be restricted to input count",
                                kind: ErrorKind::UnresolvedInput,
                                ctx: (),
                            });
                        }
                    }

                    Ok(deepest)
                },
                display: |inputs| format!("{:?} * {:?}", inputs[0], inputs[1]),
                save: save_mul_div::<B>,
                valid_shape: valid_binary::<B>,
                arity: 2,
                need_dims: false,
                stable_iter: true,
                auto_save: true,
                computes_gid: true,
                prefer_separate: false,
                valid_dispatch: DispatchOptions::Any,
            },
            vec![a, b],
            get_shape(a, b, self),
        )
    }

    pub fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_node(
            GraphOp::Custom {
                lower: |eval_node: fn(
                    NodeId,
                    NodeId,
                    &NodeInput,
                    ValueId,
                    &mut Vec<NodeId>,
                    &Self,
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
                        graph: &Self,
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
                    let mut deepest = Vec::new();

                    match backwardness {
                        None => {
                            let a = graph.nodes[node_id].inputs[0];
                            let b = graph.nodes[node_id].inputs[1];

                            let a_val = kernel.raw.def_var(DType::Float, ValueState::Mut, None);

                            let mut a_deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(a),
                                a_val,
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

                            let b_val = kernel.raw.def_var(DType::Float, ValueState::Mut, None);

                            let mut b_deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(b),
                                b_val,
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

                            deepest.append(&mut a_deep);
                            deepest.append(&mut b_deep);

                            kernel.raw.overwrite_var(out, Op::Sub { a: a_val, b: b_val });
                        }

                        Some(0) => {
                            let mut deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(node_id),
                                out,
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

                            deepest.append(&mut deep);
                        }

                        Some(1) => {
                            let g_val = kernel.raw.def_var(
                                DType::Float,
                                ValueState::Mut,
                                Some(Op::ConstF32 { value: 0.0 }),
                            );

                            let mut deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(node_id),
                                g_val,
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

                            kernel.raw.accum_var(out, Op::Neg { x: g_val });

                            deepest.append(&mut deep);
                        }

                        _ => {
                            return Err(Error {
                                msg: "backwardness must be restricted to input count",
                                kind: ErrorKind::UnresolvedInput,
                                ctx: (),
                            });
                        }
                    }

                    Ok(deepest)
                },
                display: |inputs| format!("{:?} - {:?}", inputs[0], inputs[1]),
                save: |_, _, _, _| {},
                valid_shape: valid_binary::<B>,
                arity: 2,
                need_dims: false,
                stable_iter: true,
                auto_save: true,
                computes_gid: true,
                prefer_separate: false,
                valid_dispatch: DispatchOptions::Any,
            },
            vec![a, b],
            get_shape(a, b, self),
        )
    }

    pub fn div(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.add_node(
            GraphOp::Custom {
                lower: |eval_node: fn(
                    NodeId,
                    NodeId,
                    &NodeInput,
                    ValueId,
                    &mut Vec<NodeId>,
                    &Self,
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
                        graph: &Self,
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
                    let mut deepest = Vec::new();

                    match backwardness {
                        None => {
                            let a = graph.nodes[node_id].inputs[0];
                            let b = graph.nodes[node_id].inputs[1];

                            let a_val = kernel.raw.def_var(DType::Float, ValueState::Mut, None);

                            let mut a_deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(a),
                                a_val,
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

                            let b_val = kernel.raw.def_var(DType::Float, ValueState::Mut, None);

                            let mut b_deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(b),
                                b_val,
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

                            deepest.append(&mut a_deep);
                            deepest.append(&mut b_deep);

                            kernel.raw.overwrite_var(out, Op::Div { a: a_val, b: b_val });
                        }

                        Some(0) => {
                            let g_val = kernel.raw.def_var(
                                DType::Float,
                                ValueState::Mut,
                                Some(Op::ConstF32 { value: 0.0 }),
                            );

                            let mut deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(node_id),
                                g_val,
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

                            let user = graph.nodes[node_id].inputs[1];

                            let saved = read_saved(kernel, idx, saved_params[user]);

                            kernel.raw.accum_var(out, Op::Div { a: g_val, b: saved });

                            deepest.append(&mut deep);
                        }

                        Some(1) => {
                            let g_val = kernel.raw.def_var(
                                DType::Float,
                                ValueState::Mut,
                                Some(Op::ConstF32 { value: 0.0 }),
                            );

                            let mut deep = eval_node(
                                root,
                                input,
                                &NodeInput::Node(node_id),
                                g_val,
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

                            let a = graph.nodes[node_id].inputs[0];
                            let b = graph.nodes[node_id].inputs[1];

                            let a_val = read_saved(kernel, idx, saved_params[a]);

                            let b_val = read_saved(kernel, idx, saved_params[b]);

                            let bb = kernel.raw.def_var(
                                DType::Float,
                                ValueState::Inline,
                                Some(Op::Mul { a: b_val, b: b_val }),
                            );

                            let neg_g_val = kernel.raw.def_var(
                                DType::Float,
                                ValueState::Inline,
                                Some(Op::Neg { x: g_val }),
                            );

                            let a_div_bb = kernel.raw.def_var(
                                DType::Float,
                                ValueState::Inline,
                                Some(Op::Div { a: a_val, b: bb }),
                            );

                            kernel.raw.accum_var(
                                out,
                                Op::Mul {
                                    a: neg_g_val,
                                    b: a_div_bb,
                                },
                            );

                            deepest.append(&mut deep);
                        }

                        _ => {
                            return Err(Error {
                                msg: "backwardness must be restricted to input count",
                                kind: ErrorKind::UnresolvedInput,
                                ctx: (),
                            });
                        }
                    }

                    Ok(deepest)
                },
                display: |inputs| format!("{:?} / {:?}", inputs[0], inputs[1]),
                save: save_mul_div::<B>,
                valid_shape: valid_binary::<B>,
                arity: 2,
                need_dims: false,
                stable_iter: true,
                auto_save: true,
                computes_gid: true,
                prefer_separate: false,
                valid_dispatch: DispatchOptions::Any,
            },
            vec![a, b],
            get_shape(a, b, self),
        )
    }
}

fn get_shape<B: GpuBackend + Clone>(a: NodeId, b: NodeId, graph: &Graph<B>) -> Vec<MetaId> {
    let a_node = &graph.nodes[a];
    let b_node = &graph.nodes[b];

    if a_node.op.is_const() {
        b_node.shape.clone()
    } else {
        a_node.shape.clone()
    }
}

fn read_saved(kernel: &mut LinkedKernel, index: ValueId, param: Option<ParamId>) -> ValueId {
    if let Some(pid) = param {
        kernel.register_param(pid);

        kernel.raw.def_var(
            DType::Float,
            ValueState::Immut,
            Some(Op::ParamLoad { param: pid, index }),
        )
    } else {
        kernel.raw.def_var(
            DType::Float,
            ValueState::Inline,
            Some(Op::ConstF32 { value: 0.0 }),
        )
    }
}
