use crate::{
    dispatch::{
        CompilationOptions, GpuBackend,
        backend::{
            Axis, DType, Graph, GraphOp, Metadata, NodeId, Op, Param, ParamId, ParamTy, ValueId,
            ValueState,
            kernel::{
                Dependencies, Kernel, NodeInput, SaveIndicator, TILE_SIZE,
                matmul::lower_matmul_recursive,
            },
        },
    },
    errors::{Error, ErrorKind},
};
use alloc::{vec, vec::Vec};

#[inline]
pub fn lower_forward<B: GpuBackend>(
    graph: &Graph,
    meta: Metadata,
    saved: &[SaveIndicator],
    options: &CompilationOptions<B>,
) -> Result<Vec<Dependencies<Kernel>>, Error> {
    let mut kernels: Vec<Dependencies<Kernel>> = Vec::new();

    let mut roots = Vec::new();

    for (node_id, node) in graph.nodes.iter().enumerate() {
        if node.outputs.is_empty() {
            roots.push(node_id);
        }
    }

    let block = if graph
        .nodes
        .iter()
        .any(|x| matches!(x.op, GraphOp::Matmul | GraphOp::Transpose))
    {
        [TILE_SIZE, TILE_SIZE, 1]
    } else {
        [TILE_SIZE * TILE_SIZE, 1, 1]
    };

    let mut params = Vec::new();

    params.push(Param {
        dtype: DType::UnsignedInt,
        ty: ParamTy::Uniform,
    });

    let mut saved_params = vec![None; graph.nodes.len()];

    for node_id in 0..graph.nodes.len() {
        if saved[node_id].is_defined_in_forward() {
            let pid = params.len();
            saved_params[node_id] = Some(pid);
            params.push(Param {
                dtype: DType::Float,
                ty: ParamTy::ReadWrite,
            });
        }
    }

    let mut node_params = vec![None; graph.nodes.len()];

    for node_id in 0..graph.nodes.len() {
        let node = &graph.nodes[node_id];

        for &input in &node.inputs {
            let input_node = &graph.nodes[input];

            if input_node.op == GraphOp::Input {
                ensure_param_slot(
                    &mut node_params,
                    input,
                    Param {
                        dtype: DType::Float,
                        ty: ParamTy::ReadOnly,
                    },
                    &mut params,
                );
            }
        }
    }

    let output_param = params.len();
    params.push(Param {
        dtype: DType::Float,
        ty: ParamTy::ReadWrite,
    });

    let mut resolved = Vec::new();

    while roots.iter().any(|x| !graph.nodes[*x].inputs.is_empty()) {
        let depth = kernels.len();

        if depth > 10 {
            break;
        }

        let roots_clone = roots.clone();

        roots.clear();

        for root in roots_clone {
            if resolved.contains(&root) {
                continue;
            }

            let root_node = &graph.nodes[root];

            let mut kernel = Kernel {
                meta,
                params: params.clone(),
                shared: Vec::new(),
                values: Vec::new(),
                ops: Vec::new(),
                block,
                root,
                iter_space: root_node.shape.clone(),
            };

            let shared_size = kernel.block.iter().product::<u32>();

            let mut dims = Vec::new();

            for &meta_index in &root_node.shape {
                let dim_val = kernel.def_var(
                    DType::UnsignedInt,
                    ValueState::Immut,
                    Some(Op::ReadMeta {
                        param: 0,
                        field: meta_index,
                    }),
                );

                dims.push(dim_val);
            }

            let gid = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Mut,
                Some(Op::GlobalId { axis: Axis::X }),
            );

            let mut base = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::ConstU32 { value: 0 }),
            );

            let total = dims[0];
            kernel.update_state(total, ValueState::Mut);

            if dims.len() > 1 {
                let gid2 = kernel.def_var(DType::UnsignedInt, ValueState::Mut, None);

                for (i, &d) in dims.iter().enumerate().skip(1) {
                    kernel.overwrite_var(
                        gid2,
                        Op::GlobalId {
                            axis: (i as u8).try_into().unwrap_or(Axis::Z),
                        },
                    );

                    if i >= 2 {
                        kernel.overwrite_var(gid2, Op::Mul { a: total, b: gid2 });

                        if i == 2 {
                            kernel.values[base].state = ValueState::Masked;
                            base = gid2;
                        }
                    } else {
                        kernel.overwrite_var(gid2, Op::Mul { a: d, b: gid2 });
                    }

                    kernel.overwrite_var(gid, Op::Add { a: gid, b: gid2 });

                    kernel.overwrite_var(total, Op::Mul { a: total, b: d });
                }
            }

            let tile_size = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::ConstU32 { value: TILE_SIZE }),
            );
            let local_row = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::LocalId { axis: Axis::Y }),
            );
            let local_col = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::LocalId { axis: Axis::X }),
            );

            let out = kernel.def_var(DType::Float, ValueState::Mut, None);

            let new_roots = eval_node(
                root,
                0,
                &NodeInput::Node(root),
                out,
                &mut resolved,
                graph,
                &node_params,
                &saved_params,
                &mut kernel,
                gid,
                base,
                local_row,
                local_col,
                shared_size,
                tile_size,
                true,
                options,
            )?;

            for root in &new_roots[1..] {
                if roots.contains(root) {
                    continue;
                }

                roots.push(*root);
            }

            if depth == 0 {
                kernel.param_store(output_param, gid, out);
            }

            for kernel in &mut kernels {
                if kernel.val.root > root {
                    kernel.dep.push(root);
                }
            }

            kernels.push(Dependencies {
                val: kernel,
                dep: Vec::new(),
            });
        }
    }

    Ok(kernels)
}

#[inline]
fn ensure_param_slot(
    node_params: &mut [Option<ParamId>],
    node_id: NodeId,
    param: Param,
    params: &mut Vec<Param>,
) -> ParamId {
    node_params[node_id].unwrap_or_else(|| {
        let pid = params.len();
        params.push(param);
        node_params[node_id] = Some(pid);
        pid
    })
}

fn eval_node<B: GpuBackend>(
    root: NodeId,
    input: NodeId,
    node_id: &NodeInput,
    out: ValueId,
    resolved: &mut Vec<NodeId>,
    graph: &Graph,
    node_params: &[Option<ParamId>],
    saved_params: &[Option<ParamId>],
    kernel: &mut Kernel,
    idx: ValueId,
    base: ValueId,
    local_row: ValueId,
    local_col: ValueId,
    shared_size: u32,
    tile_size: ValueId,
    stable_iteration_space: bool,
    options: &CompilationOptions<B>,
) -> Result<Vec<NodeId>, Error> {
    let node_id = match node_id {
        NodeInput::Node(node_id) => *node_id,
        NodeInput::Raw { .. } => {
            return Err(Error {
                msg: "cannot evaluate raw node input during forward pass",
                kind: ErrorKind::InvalidArgument,
                ctx: (),
            });
        }
    };

    let node = &graph.nodes[node_id];

    let mut deepest = Vec::new();

    if !node.inputs.is_empty() {
        deepest.push(node_id);
    }

    match node.op {
        GraphOp::Input => {
            let param = node_params[node_id].ok_or(Error {
                msg: "input param missing",
                kind: ErrorKind::ParamNotMaterialized,
                ctx: (),
            })?;

            kernel.overwrite_var(out, Op::Load { param, index: idx });
        }

        GraphOp::ConstF32(value) => {
            kernel.overwrite_var(out, Op::ConstF32 { value });
        }

        GraphOp::ConstI32(value) => {
            kernel.overwrite_var(out, Op::ConstI32 { value });
        }

        GraphOp::ConstU32(value) => {
            kernel.overwrite_var(out, Op::ConstU32 { value });
        }

        GraphOp::Add | GraphOp::Sub | GraphOp::Mul | GraphOp::Div => {
            let a = kernel.def_var(DType::Float, ValueState::Mut, None);

            let mut a_deep = eval_node(
                root,
                input,
                &NodeInput::Node(node.inputs[0]),
                a,
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

            let b = kernel.def_var(DType::Float, ValueState::Mut, None);

            let mut b_deep = eval_node(
                root,
                input,
                &NodeInput::Node(node.inputs[1]),
                b,
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

            let op = match node.op {
                GraphOp::Add => Op::Add { a, b },
                GraphOp::Sub => Op::Sub { a, b },
                GraphOp::Mul => Op::Mul { a, b },
                GraphOp::Div => Op::Div { a, b },
                _ => unreachable!(),
            };

            kernel.overwrite_var(out, op);

            deepest.append(&mut a_deep);
            deepest.append(&mut b_deep);
        }

        GraphOp::Matmul => {
            if !stable_iteration_space {
                if !graph.nodes[node_id].outputs.is_empty() {
                    let param = saved_params[node_id].ok_or(Error {
                        msg: "saved root param not materialized",
                        kind: ErrorKind::ParamNotMaterialized,
                        ctx: (),
                    })?;
                    kernel.overwrite_var(out, Op::Load { param, index: idx });
                }

                return Ok(deepest);
            }

            let row = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::GlobalId { axis: Axis::Y }),
            );
            let col = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::GlobalId { axis: Axis::X }),
            );

            let mut matmul_deep = lower_matmul_recursive(
                eval_node::<B>,
                root,
                input,
                resolved,
                false,
                false,
                &NodeInput::Node(node.inputs[0]),
                &NodeInput::Node(node.inputs[1]),
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
            )?;

            deepest.append(&mut matmul_deep);
        }

        _ => {
            unimplemented!()
        }
    }

    if let Some(saved) = saved_params[node_id] {
        kernel.param_store(saved, idx, out);
    }

    if !resolved.contains(&node_id) {
        resolved.push(node_id);
    }

    Ok(deepest)
}
