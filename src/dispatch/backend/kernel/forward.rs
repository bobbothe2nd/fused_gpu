use core::cmp::Ordering;

use crate::{
    dispatch::{
        CompilationOptions, GpuBackend,
        backend::{
            Axis, DType, DispatchOptions, Graph, GraphOp, Metadata, NodeId, Op, Param, ParamId,
            ParamTy, ValueId, ValueState,
            kernel::{
                Dependencies, KernelsChained, LinkedKernel, NodeInput, RawKernel, SaveIndicator,
            },
        },
    },
    errors::{Error, ErrorKind},
};
use alloc::{vec, vec::Vec};

#[inline]
pub fn lower_forward<'a, B: GpuBackend>(
    graph: &'a Graph<'a, B>,
    meta: Metadata,
    saved: &[SaveIndicator],
    options: &CompilationOptions<B>,
) -> Result<KernelsChained<'a, B>, Error> {
    let mut kernels: Vec<Dependencies<LinkedKernel<'a, B>>> = Vec::new();

    let mut roots = Vec::new();

    for (node_id, node) in graph.nodes.iter().enumerate() {
        if node.outputs.is_empty() {
            roots.push(node_id);
        }
    }

    let tile_size = options.opt.tile_size;

    let mut params = Vec::new();

    params.push(Param {
        dtype: DType::UnsignedInt,
        ty: ParamTy::Uniform,
        pid: 0,
    });

    let mut saved_params = vec![None; graph.nodes.len()];

    for node_id in 0..graph.nodes.len() {
        if saved[node_id].is_defined_in_forward() {
            let pid = params.len();
            saved_params[node_id] = Some(pid);
            params.push(Param {
                dtype: DType::Float,
                ty: ParamTy::ReadWrite,
                pid,
            });
        }
    }

    let mut node_params = vec![None; graph.nodes.len()];

    for node_id in 0..graph.nodes.len() {
        let node = &graph.nodes[node_id];

        for &input in &node.inputs {
            let input_node = &graph.nodes[input];

            if matches!(input_node.op, GraphOp::Input) {
                ensure_param_slot(
                    &mut node_params,
                    input,
                    Param {
                        dtype: DType::Float,
                        ty: ParamTy::ReadOnly,
                        pid: params.len(),
                    },
                    &mut params,
                );
            }
        }
    }

    let output_param = params.len();
    node_params[graph.nodes.len() - 1] = Some(output_param);

    params.push(Param {
        dtype: DType::Float,
        ty: ParamTy::ReadWrite,
        pid: output_param,
    });

    let mut resolved = Vec::new();

    while roots.iter().any(|x| !graph.nodes[*x].inputs.is_empty()) {
        let depth = kernels.len();

        let roots_clone = roots.clone();

        roots.clear();

        for root in roots_clone {
            if resolved.contains(&root) {
                continue;
            }

            let root_node = &graph.nodes[root];

            let mut kernel = LinkedKernel {
                raw: RawKernel {
                    meta,
                    shared: Vec::new(),
                    values: Vec::new(),
                    ops: Vec::new(),
                    block: [tile_size, tile_size, 1],
                    root,
                    iter_space: root_node.shape.clone(),
                },
                params: vec![false; params.len()],
                meta: vec![false; meta.fields],
                ops: Vec::new(),
            };

            kernel.register_param(0);

            let shared_size = tile_size * tile_size;

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

            let gid = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Mut,
                Some(Op::GlobalId { axis: Axis::X }),
            );

            let mut base = kernel.raw.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
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

                    kernel.raw.overwrite_var(gid, Op::Add { a: gid, b: gid2 });

                    kernel.raw.overwrite_var(total, Op::Mul { a: total, b: d });
                }
            }

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

            let out = kernel.raw.def_var(DType::Float, ValueState::Mut, None);

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

                if roots.contains(inner_root) || root == *inner_root {
                    continue;
                }

                roots.push(*inner_root);
            }

            if depth == 0 && graph.nodes[kernel.raw.root].op.is_compute_gid() {
                kernel.param_store(output_param, gid, out);
            }

            for kernel in &mut kernels {
                if kernel.val.raw.root > root {
                    kernel.dep.push(root);
                }
            }

            kernels.push(Dependencies {
                val: kernel,
                dep: Vec::new(),
            });
        }
    }

    Ok(KernelsChained { kernels, params })
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

fn eval_node<'a, B: GpuBackend>(
    root: NodeId,
    input: NodeId,
    node_id: &NodeInput,
    out: ValueId,
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
        NodeInput::Raw { .. } => {
            return Err(Error {
                msg: "cannot evaluate raw node input during forward pass",
                kind: ErrorKind::InvalidArgument,
                ctx: (),
            });
        }
    };

    kernel.ops.push(&graph.nodes[node_id].op);

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

            kernel
                .raw
                .overwrite_var(out, Op::ParamLoad { param, index: idx });
            kernel.register_param(param);
        }

        GraphOp::ConstF32(value) => {
            kernel.raw.overwrite_var(out, Op::ConstF32 { value });
        }

        GraphOp::Custom {
            lower,
            stable_iter,
            need_dims,
            valid_dispatch,
            computes_gid,
            prefer_separate,
            ..
        } => {
            let mut least_valid_dispatch = DispatchOptions::Any;

            if need_dims || prefer_separate {
                for resolved_id in resolved.iter() {
                    let resolved_node = &graph.nodes[*resolved_id];

                    if let GraphOp::Custom { valid_dispatch, .. } = resolved_node.op
                        && valid_dispatch > least_valid_dispatch
                    {
                        least_valid_dispatch = valid_dispatch;
                    }
                }
            }

            let dims_invalid = (least_valid_dispatch.partial_cmp(&valid_dispatch)
                != Some(Ordering::Less))
                && least_valid_dispatch != DispatchOptions::Any;

            if !(stable_iter || stable_iteration_space)
                || dims_invalid
                || !(computes_gid || resolved.is_empty())
            {
                let param = saved_params[node_id].ok_or(Error {
                    msg: "saved root param not materialized",
                    kind: ErrorKind::ParamNotMaterialized,
                    ctx: (),
                })?;
                kernel
                    .raw
                    .overwrite_var(out, Op::ParamLoad { param, index: idx });

                kernel.register_param(param);

                return Ok(deepest);
            }

            let mut deep = lower(
                eval_node::<B>,
                root,
                input,
                resolved,
                None,
                node_id,
                graph,
                out,
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

    if let Some(saved) = saved_params[node_id]
        && node.op.is_auto_save()
    {
        kernel.param_store(saved, idx, out);
    }

    if !resolved.contains(&node_id) {
        resolved.push(node_id);
    }

    Ok(deepest)
}
