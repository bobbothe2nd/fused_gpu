use crate::{
    dispatch::{
        CompilationOptions, GpuBackend,
        backend::{
            Axis, DType, DispatchOptions, Graph, GraphOp, Metadata, Node, NodeId, Op, Param,
            ParamId, ParamTy, ValueId, ValueState,
            kernel::{
                Dependencies, KernelsChained, LinkedKernel, NodeInput, RawKernel, SaveIndicator,
            },
        },
    },
    errors::{Error, ErrorKind},
};
use alloc::{vec, vec::Vec};
use core::cmp::Ordering;

#[inline]
pub fn lower_backward<'a, B: GpuBackend>(
    graph: &'a Graph<'a, B>,
    meta: Metadata,
    saved: &[SaveIndicator],
    options: &CompilationOptions<B>,
) -> Result<KernelsChained<'a, B>, Error> {
    let tile_size = options.opt.tile_size;

    let shared_size = tile_size * tile_size;

    let mut params = Vec::new();

    params.push(Param {
        dtype: DType::UnsignedInt,
        ty: ParamTy::Uniform,
        pid: 0,
    });

    params.push(Param {
        dtype: DType::Float,
        ty: ParamTy::ReadOnly,
        pid: 1,
    });

    let mut grad_params = vec![None; graph.nodes.len()];

    for node_id in 0..graph.nodes.len() {
        if saved[node_id].is_defined_in_backward() {
            let pid = params.len();
            grad_params[node_id] = Some(pid);

            params.push(Param {
                dtype: DType::Float,
                ty: ParamTy::ReadWrite,
                pid,
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
                pid,
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
                pid,
            });
        }
    }

    let mut grad_kernels: Vec<Dependencies<LinkedKernel<'a, B>>> = Vec::new();
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
            let roots_clone = roots.clone();

            roots.clear();

            for &root in &roots_clone {
                if resolved.contains(&root) {
                    continue;
                }

                let root_node = &graph.nodes[root];

                let (mut kernel, base, gid) =
                    gen_kernel(meta, &params, [tile_size, tile_size, 1], root, root_node);

                let upstream = kernel.raw.def_var(
                    DType::Float,
                    ValueState::Mut,
                    Some(Op::ConstF32 { value: 0.0 }),
                );

                let tile_size = kernel.raw.def_var(
                    DType::UnsignedInt,
                    ValueState::Const,
                    Some(Op::ConstU32 { value: tile_size }),
                );
                let local_row = kernel.raw.def_var(
                    DType::UnsignedInt,
                    ValueState::Immut,
                    Some(Op::LocalId { axis: Axis::Y }),
                );
                let local_col = kernel.raw.def_var(
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

                let root_op = &graph.nodes[root].op;

                if root_op.is_compute_gid() || root_op.is_leaf() {
                    let pid = grad_params[root].ok_or(Error {
                        msg: "grad param could not be materialized during invalid dispatch fusion",
                        kind: ErrorKind::ParamNotMaterialized,
                        ctx: (),
                    })?;
                    kernel.param_store(pid, gid, upstream);
                }

                for inner_root in &new_roots {
                    let node = &graph.nodes[*inner_root];

                    if let GraphOp::Custom { valid_dispatch, .. } = node.op {
                        match valid_dispatch {
                            DispatchOptions::Any => {}

                            DispatchOptions::ReqRow => {
                                kernel.raw.block = [1, shared_size, 1];
                            }

                            DispatchOptions::ReqCol => {
                                kernel.raw.block = [shared_size, 1, 1];
                            }
                        }
                    }

                    if roots.contains(inner_root)
                        || root == *inner_root
                        || resolved.contains(inner_root)
                    {
                        continue;
                    }

                    roots.push(*inner_root);
                }

                for kernel in &mut grad_kernels {
                    if kernel.val.raw.root < root {
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

    Ok(KernelsChained { kernels, params })
}

fn eval_grad<'a, B: GpuBackend>(
    root: NodeId,
    input: NodeId,
    node_id: &NodeInput,
    upstream: ValueId,
    resolved: &mut Vec<NodeId>,
    graph: &'a Graph<'a, B>,
    node_params: &[Option<ParamId>],
    saved_params: &[Option<ParamId>],
    kernel: &mut LinkedKernel<'a, B>,
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
            kernel.raw.accum_var(
                upstream,
                Op::ParamLoad {
                    param: *param,
                    index: idx,
                },
            );

            kernel.register_param(*param);

            return Ok(Vec::new());
        }
    };

    kernel.ops.push(&graph.nodes[node_id].op);

    let mut deepest = Vec::new();

    deepest.push(node_id);

    let node = &graph.nodes[node_id];

    if node.outputs.is_empty() {
        kernel.raw.accum_var(
            upstream,
            Op::ParamLoad {
                param: 1,
                index: idx,
            },
        );

        kernel.register_param(1);

        return Ok(deepest);
    }

    let mut least_valid_dispatch = DispatchOptions::Any;

    if node.op.is_need_dims() || node.op.is_prefer_separate() {
        for resolved_id in resolved.iter() {
            let resolved_node = &graph.nodes[*resolved_id];

            if let GraphOp::Custom { valid_dispatch, .. } = resolved_node.op
                && valid_dispatch > least_valid_dispatch
            {
                least_valid_dispatch = valid_dispatch;
            }
        }
    }

    let dims_invalid = (least_valid_dispatch.partial_cmp(&node.op.valid_dispatch())
        != Some(Ordering::Less))
        && least_valid_dispatch != DispatchOptions::Any;

    if (node.op.is_transform() && !stable_iteration_space)
        || (!node.op.is_compute_gid() && (dims_invalid || resolved.is_empty()))
    {
        let param = node_params[node_id].ok_or(Error {
            msg: "grad root param not materialized",
            kind: ErrorKind::ParamNotMaterialized,
            ctx: (),
        })?;
        kernel
            .raw
            .accum_var(upstream, Op::ParamLoad { param, index: idx });

        kernel.register_param(param);

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
        } else {
            return Err(Error {
                msg: "node output accepting no inputs",
                kind: ErrorKind::UnresolvedOutput,
                ctx: (),
            });
        }
    }

    if !resolved.contains(&node_id) {
        resolved.push(node_id);
    }

    Ok(deepest)
}

fn gen_kernel<'a, B: GpuBackend>(
    meta: Metadata,
    params: &[Param],
    block: [u32; 3],
    input: NodeId,
    root_node: &Node<B>,
) -> (LinkedKernel<'a, B>, ValueId, ValueId) {
    let mut kernel = LinkedKernel {
        raw: RawKernel {
            meta,
            shared: Vec::new(),
            values: Vec::new(),
            ops: Vec::new(),
            block,
            root: input,
            iter_space: root_node.shape.clone(),
        },
        params: vec![false; params.len()],
        meta: vec![false; meta.fields],
        ops: Vec::new(),
    };

    kernel.register_param(0);

    let mut dims = Vec::new();

    for &meta_index in &root_node.shape {
        let dim_val = kernel.raw.def_var(
            DType::UnsignedInt,
            ValueState::Immut,
            Some(Op::ReadMeta {
                param: 0,
                field: meta_index,
            }),
        );

        kernel.register_meta(meta_index);

        dims.push(dim_val);
    }

    if dims.len() > 3 {
        let m = dims.len() - 2;
        let n = dims.len() - 1;

        dims.swap(m, n);
    }

    let gid = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Mut,
        Some(Op::GlobalId { axis: Axis::X }),
    );

    let mut base = kernel.raw.def_var(
        DType::UnsignedInt,
        ValueState::Mut,
        Some(Op::ConstU32 { value: 0 }),
    );

    let total = dims[0];
    kernel.raw.update_var_state(total, ValueState::Mut);

    if dims.len() > 1 {
        let gid2 = kernel
            .raw
            .def_var(DType::UnsignedInt, ValueState::Mut, None);

        for (i, &d) in dims.iter().enumerate().skip(1) {
            kernel.raw.overwrite_var(
                gid2,
                Op::GlobalId {
                    axis: (i as u8).try_into().unwrap_or(Axis::Z),
                },
            );

            if i >= 2 {
                kernel
                    .raw
                    .overwrite_var(gid2, Op::Mul { a: total, b: gid2 });

                if i == 2 {
                    kernel.raw.update_var_state(base, ValueState::Masked);
                    base = gid2;
                }
            } else {
                kernel.raw.overwrite_var(gid2, Op::Mul { a: d, b: gid2 });
            }

            kernel.raw.accum_var(gid, Op::CopyVar { id: gid2 });

            kernel.raw.overwrite_var(total, Op::Mul { a: total, b: d });
        }
    }

    (kernel, base, gid)
}
