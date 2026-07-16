use crate::{
    dispatch::{
        CompilationOptions, GpuBackend,
        backend::{
            Axis, DType, Graph, GraphOp, Metadata, Node, NodeId, Op, Param, ParamId, ParamTy,
            ValueId, ValueState,
            kernel::{
                Dependencies, Kernel, NodeInput, SaveIndicator, TILE_SIZE,
                matmul::lower_matmul_recursive,
            },
        },
    },
    errors::{Error, ErrorKind},
};
use alloc::{vec, vec::Vec};

const MAX_KERNEL_DEPTH: usize = 10;

#[inline]
pub fn lower_backward<B: GpuBackend>(
    graph: &Graph,
    meta: Metadata,
    saved: &[SaveIndicator],
    options: &CompilationOptions<B>,
) -> Result<Vec<Dependencies<Kernel>>, Error> {
    let block = if graph
        .nodes
        .iter()
        .all(|x| x.op.is_elementwise() || x.op.is_leaf())
    {
        [256, 1, 1]
    } else {
        [16, 16, 1]
    };

    let shared_size = block.iter().product::<u32>();

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
        if graph.nodes[node_id].op == GraphOp::Input {
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
            if node.op == GraphOp::Input {
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
                    gen_kernel(meta, params.clone(), block, root, root_node);

                let upstream = kernel.def_var(DType::Float, ValueState::Mut, None);

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

                let mut new_roots = eval_grad::<B>(
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

                roots.append(&mut new_roots);

                let pid = grad_params[root].ok_or(Error {
                    msg: "grad param could not be materialized",
                    kind: ErrorKind::ParamNotMaterialized,
                    ctx: (),
                })?;
                kernel.param_store(pid, gid, upstream);

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

#[inline]
fn load_value(
    kernel: &mut Kernel,
    params: &[Option<ParamId>],
    index: ValueId,
    node: NodeId,
    dtype: DType,
) -> Result<ValueId, Error> {
    let pid = params[node].ok_or(Error {
        msg: "no saved value available",
        kind: ErrorKind::ParamNotMaterialized,
        ctx: (),
    })?;

    let v = kernel.def_var(
        dtype,
        ValueState::Immut,
        Some(Op::Load { param: pid, index }),
    );

    Ok(v)
}

fn eval_grad<B: GpuBackend>(
    root: NodeId,
    input: NodeId,
    node_id: &NodeInput,
    upstream: ValueId,
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
        NodeInput::Raw { param, shape: _ } => {
            kernel.overwrite_var(
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
        kernel.overwrite_var(
            upstream,
            Op::Load {
                param: 1,
                index: idx,
            },
        );

        return Ok(deepest);
    }

    for &user in &node.outputs {
        let user_node = &graph.nodes[user];

        let mut deep = match user_node.op {
            GraphOp::Add => eval_grad(
                root,
                input,
                &NodeInput::Node(user),
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
            )?,

            GraphOp::Sub => {
                let is_rhs = graph.is_rhs_edge(node_id, user);

                let g = if is_rhs {
                    kernel.def_var(DType::Float, ValueState::Immut, None)
                } else {
                    upstream
                };

                let deep = eval_grad(
                    root,
                    input,
                    &NodeInput::Node(user),
                    g,
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

                if is_rhs {
                    kernel.overwrite_var(upstream, Op::Neg { x: g });
                }

                deep
            }

            GraphOp::Mul => {
                let g = kernel.def_var(DType::Float, ValueState::Mut, None);

                let deep = eval_grad(
                    root,
                    input,
                    &NodeInput::Node(user),
                    g,
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

                if graph.is_lhs_edge(node_id, user) {
                    let b =
                        load_value(kernel, saved_params, idx, user_node.inputs[1], DType::Float)?;

                    kernel.overwrite_var(upstream, Op::Mul { a: g, b });
                } else {
                    let a =
                        load_value(kernel, saved_params, idx, user_node.inputs[0], DType::Float)?;

                    kernel.overwrite_var(upstream, Op::Mul { a, b: g });
                }

                deep
            }

            GraphOp::Matmul => {
                if !stable_iteration_space {
                    let param = node_params[node_id].ok_or(Error {
                        msg: "saved root param not materialized",
                        kind: ErrorKind::ParamNotMaterialized,
                        ctx: (),
                    })?;

                    kernel.overwrite_var(upstream, Op::Load { param, index: idx });

                    return Ok(deepest);
                }

                let (a, b, a_t, b_t);

                if graph.is_lhs_edge(node_id, user) {
                    let param = saved_params[user_node.inputs[1]].ok_or(Error {
                        msg: "saved input parameter could not be materialized",
                        kind: ErrorKind::ParamNotMaterialized,
                        ctx: (),
                    })?;
                    let shape = &graph.nodes[user_node.inputs[1]].shape;

                    a = NodeInput::Node(user);
                    b = NodeInput::Raw { param, shape };
                    a_t = false;
                    b_t = true;
                } else {
                    let param = saved_params[user_node.inputs[0]].ok_or(Error {
                        msg: "saved input parameter could not be materialized",
                        kind: ErrorKind::ParamNotMaterialized,
                        ctx: (),
                    })?;
                    let shape = &graph.nodes[user_node.inputs[0]].shape;

                    a = NodeInput::Raw { param, shape };
                    b = NodeInput::Node(user);
                    a_t = true;
                    b_t = false;
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

                lower_matmul_recursive(
                    eval_grad,
                    root,
                    input,
                    resolved,
                    a_t,
                    b_t,
                    &a,
                    &b,
                    graph,
                    upstream,
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
                )?
            }

            _ => unimplemented!(),
        };

        deepest.append(&mut deep);
    }

    if !resolved.contains(&node_id) {
        resolved.push(node_id);
    }

    Ok(deepest)
}

fn gen_kernel(
    meta: Metadata,
    params: Vec<Param>,
    block: [u32; 3],
    input: NodeId,
    root_node: &Node,
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

            kernel.overwrite_var(gid, Op::Add { a: gid, b: gid2 });

            kernel.overwrite_var(total, Op::Mul { a: total, b: d });
        }
    }

    (kernel, base, gid)
}
