use crate::{
    dispatch::{
        CompilationOptions, GpuBackend, backend::{
            Axis, DType, DispatchOptions, Graph, GraphOp, Metadata, Node, NodeId, Op, Param, ParamId, ParamTy, ValueId, ValueState, kernel::{Dependencies, Kernel, NodeInput, SaveIndicator},
        },
    }, errors::{Error, ErrorKind},
};
use alloc::{vec, vec::Vec};

const MAX_KERNEL_DEPTH: usize = 10;

#[inline]
pub fn lower_backward<B: GpuBackend + Clone>(
    graph: &Graph<B>,
    meta: Metadata,
    saved: &[SaveIndicator],
    options: &CompilationOptions<B>,
) -> Result<Vec<Dependencies<Kernel>>, Error> {
    let tile_size = options.opt.tile_size;

    let shared_size = tile_size * tile_size;

    let mut params = Vec::new();

    params.push(Param {
        dtype: DType::UnsignedInt,
        ty: ParamTy::Uniform,
    });

    params.push(Param {
        dtype: DType::Float,
        ty: ParamTy::ReadOnly,
    });

    let mut grad_params = vec![None; graph.nodes.len()];

    for node_id in 0..graph.nodes.len() {
        if saved[node_id].is_defined_in_backward() {
            let pid = params.len();
            grad_params[node_id] = Some(pid);

            params.push(Param {
                dtype: DType::Float,
                ty: ParamTy::ReadWrite,
            });
        }
    }

    let mut saved_params = vec![None; graph.nodes.len()];

    for (node_id, param) in saved_params.iter_mut().enumerate() {
        if matches!(graph.nodes[node_id].op, GraphOp::Input) {
            let pid = params.len();
            *param = Some(pid);

            params.push(Param {
                dtype: DType::Float,
                ty: ParamTy::ReadOnly,
            });
        }
    }

    for (node_id, param) in saved_params.iter_mut().enumerate() {
        if saved[node_id].is_defined_in_forward() {
            let pid = params.len();
            *param = Some(pid);

            params.push(Param {
                dtype: DType::Float,
                ty: ParamTy::ReadOnly,
            });
        }
    }

    let mut grad_kernels: Vec<Dependencies<Kernel>> = Vec::new();
    let mut kernels = Vec::new();

    for input in graph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(node_id, node)| {
            if matches!(node.op, GraphOp::Input) {
                Some(node_id)
            } else {
                None
            }
        })
    {
        let mut resolved = Vec::new();
        let mut roots = vec![input];

        while !roots.iter().all(|x| graph.nodes[*x].outputs.is_empty()) {
            let depth = grad_kernels.len();

            if depth > MAX_KERNEL_DEPTH {
                break;
            }

            let roots_clone = roots.clone();

            roots.clear();

            for &root in &roots_clone {
                if resolved.contains(&root) {
                    continue;
                }

                let root_node = &graph.nodes[root];

                if root_node.outputs.is_empty() {
                    continue;
                }

                let (mut kernel, base, gid) =
                    gen_kernel(meta, params.clone(), [tile_size, tile_size, 1], root, root_node);

                let upstream = kernel.def_var(DType::Float, ValueState::Mut, Some(Op::ConstF32 { value: 0.0 }));

                let tile_size = kernel.def_var(
                    DType::UnsignedInt,
                    ValueState::Const,
                    Some(Op::ConstU32 { value: tile_size }),
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

                let new_roots = eval_grad::<B>(
                    root,
                    input,
                    &NodeInput::Node(root),
                    upstream,
                    &mut resolved,
                    graph,
                    &grad_params,
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

                let pid = grad_params[root].ok_or(Error {
                    msg: "grad param could not be materialized",
                    kind: ErrorKind::ParamNotMaterialized,
                    ctx: (),
                })?;
                kernel.param_store(pid, gid, upstream);

                for root in &new_roots[1..] {
                    let node = &graph.nodes[*root];

                    if let GraphOp::Custom { valid_dispatch, .. } = node.op {
                        match valid_dispatch {
                            DispatchOptions::Any => {}

                            DispatchOptions::ReqRow => {
                                kernel.block = [1, shared_size, 1];
                            }

                            DispatchOptions::ReqCol => {
                                kernel.block = [shared_size, 1, 1];
                            }

                            DispatchOptions::ReqRowCol => {
                                kernel.block = [1, 1, 1];
                            }
                        }
                    }

                    if roots.contains(root) {
                        continue;
                    }

                    roots.push(*root);
                }

                for kernel in &mut grad_kernels {
                    if kernel.val.root < root {
                        kernel.dep.push(root);
                    }
                }

                let kernel = Dependencies {
                    val: kernel,
                    dep: Vec::new(),
                };

                grad_kernels.push(kernel);
            }
        }

        kernels.append(&mut grad_kernels);
    }

    Ok(kernels)
}

fn eval_grad<B: GpuBackend + Clone>(
    root: NodeId,
    input: NodeId,
    node_id: &NodeInput,
    upstream: ValueId,
    resolved: &mut Vec<NodeId>,
    graph: &Graph<B>,
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
        NodeInput::Raw { param, shape: _ } => {
            kernel.accum_var(
                upstream,
                Op::Load {
                    param: *param,
                    index: idx,
                },
            );

            return Ok(Vec::new());
        }
    };

    let mut deepest = Vec::new();

    deepest.push(node_id);

    let node = &graph.nodes[node_id];

    if node.outputs.is_empty() {
        kernel.accum_var(
            upstream,
            Op::Load {
                param: 1,
                index: idx,
            },
        );

        return Ok(deepest);
    }

    if node.op.is_transform()
        && !stable_iteration_space
        && !graph.nodes[node_id].outputs.is_empty()
    {
        let param = node_params[node_id].ok_or(Error {
            msg: "grad root param not materialized",
            kind: ErrorKind::ParamNotMaterialized,
            ctx: (),
        })?;
        kernel.accum_var(upstream, Op::Load { param, index: idx });

        return Ok(deepest);
    }

    for &user in &node.outputs {
        let user_node = &graph.nodes[user];

        if let GraphOp::Custom { lower, .. } = user_node.op {
            let edge = graph.get_edge(node_id, user).ok_or(Error {
                msg: "recieved invalid node edge",
                kind: ErrorKind::UnresolvedInput,
                ctx: (),
            })?;

            let mut deep = lower(
                eval_grad::<B>,
                root,
                input,
                resolved,
                Some(edge as u8),
                user,
                graph,
                upstream,
                node_params,
                saved_params,
                kernel,
                base,
                idx,
                local_row,
                local_col,
                shared_size,
                tile_size,
                stable_iteration_space,
                options,
            )?;

            deepest.append(&mut deep);
        }
    }

    if !resolved.contains(&node_id) {
        resolved.push(node_id);
    }

    Ok(deepest)
}

fn gen_kernel<B: GpuBackend + Clone>(
    meta: Metadata,
    params: Vec<Param>,
    block: [u32; 3],
    input: NodeId,
    root_node: &Node<B>,
) -> (Kernel, ValueId, ValueId) {
    let mut kernel = Kernel {
        meta,
        params,
        shared: Vec::new(),
        values: Vec::new(),
        ops: Vec::new(),
        block,
        root: input,
        iter_space: root_node.shape.clone(),
    };

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

    if dims.len() > 3 {
        let m = dims.len() - 2;
        let n = dims.len() - 1;

        dims.swap(m, n);
    }

    let gid = kernel.def_var(
        DType::UnsignedInt,
        ValueState::Mut,
        Some(Op::GlobalId { axis: Axis::X }),
    );

    let mut base = kernel.def_var(
        DType::UnsignedInt,
        ValueState::Mut,
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
                    kernel.update_state(base, ValueState::Masked);
                    base = gid2;
                }
            } else {
                kernel.overwrite_var(gid2, Op::Mul { a: d, b: gid2 });
            }

            kernel.accum_var(gid, Op::CopyVar { id: gid2 });

            kernel.overwrite_var(total, Op::Mul { a: total, b: d });
        }
    }

    (kernel, base, gid)
}
