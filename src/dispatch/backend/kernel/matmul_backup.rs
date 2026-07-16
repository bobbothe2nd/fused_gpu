use crate::{
    dispatch::{
        CompilationOptions, GpuBackend,
        backend::{
            DType, Graph, NodeId, ParamId, ValueId, ValueState,
            kernel::{Kernel, NodeInput},
        },
    },
    errors::Error,
};
use alloc::vec::Vec;

pub fn lower_matmul_recursive<B: GpuBackend, G: Copy>(
    eval_node: impl Fn(
        NodeId,
        &NodeInput,
        ValueId,
        &mut Vec<NodeId>,
        &Graph,
        G,
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
    resolved: &mut Vec<NodeId>,
    transpose_a: bool,
    transpose_b: bool,
    a_node: &NodeInput,
    b_node: &NodeInput,
    graph: &Graph,
    out: ValueId,
    node_params: G,
    saved_params: &[Option<ParamId>],
    kernel: &mut Kernel,
    base: ValueId,
    row: ValueId,
    col: ValueId,
    local_row: ValueId,
    local_col: ValueId,
    shared_size: u32,
    tile_size: ValueId,
    options: &CompilationOptions<B>,
) -> Result<Vec<NodeId>, Error> {
    let mut a_node_shape = match a_node {
        NodeInput::Node(node) => graph.nodes[*node].shape.clone(),
        NodeInput::Raw { param: _, shape } => shape.to_vec(),
    };

    if transpose_a {
        let len = a_node_shape.len();
        a_node_shape.swap(len - 1, len - 2);
    }

    let mut b_node_shape = match b_node {
        NodeInput::Node(node) => graph.nodes[*node].shape.clone(),
        NodeInput::Raw { param: _, shape } => shape.to_vec(),
    };

    if transpose_b {
        let len = b_node_shape.len();
        b_node_shape.swap(len - 1, len - 2);
    }

    std::eprintln!(" a={a_node_shape:?},b={b_node_shape:?}");

    let m = kernel.def_var(DType::UnsignedInt, ValueState::Immut, None);
    let n = kernel.def_var(DType::UnsignedInt, ValueState::Immut, None);
    let k = kernel.def_var(DType::UnsignedInt, ValueState::Immut, None);

    kernel.push_read_meta(
        a_node_shape[a_node_shape.len() - 2],
        m,
    );

    kernel.push_read_meta(
        b_node_shape[b_node_shape.len() - 1],
        n,
    );

    kernel.push_read_meta(
        a_node_shape[a_node_shape.len() - 1],
        k,
    );

    forward_matmul(
        eval_node,
        root,
        resolved,
        m,
        n,
        k,
        transpose_a,
        transpose_b,
        a_node,
        b_node,
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

pub fn forward_matmul<B: GpuBackend, G: Copy>(
    eval_node: impl Fn(
        NodeId,
        &NodeInput,
        ValueId,
        &mut Vec<NodeId>,
        &Graph,
        G,
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
    resolved: &mut Vec<NodeId>,
    m: ValueId,
    n: ValueId,
    k: ValueId,
    swap_a: bool,
    swap_b: bool,
    a_node: &NodeInput,
    b_node: &NodeInput,
    graph: &Graph,
    out: ValueId,
    node_params: G,
    saved_params: &[Option<ParamId>],
    kernel: &mut Kernel,
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

    let a_tile = kernel.new_shared(DType::Float, shared_size);
    let b_tile = kernel.new_shared(DType::Float, shared_size);

    let zero = kernel.def_var(DType::UnsignedInt, ValueState::Immut, None);
    let one = kernel.def_var(DType::UnsignedInt, ValueState::Immut, None);

    kernel.push_const_u32(0, zero);
    kernel.push_const_u32(1, one);

    kernel.push_const_f32(0.0, out);

    let tk = kernel.def_var(DType::UnsignedInt, ValueState::Mut, None);

    let shared_idx = kernel.def_var(DType::UnsignedInt, ValueState::Mut, None);

    kernel.push_mul(
        local_row,
        tile_size,
        shared_idx,
    );
    kernel.push_add(
        shared_idx,
        local_col,
        shared_idx,
    );

    let mut a_deepest = Vec::new();
    let mut b_deepest = Vec::new();

    kernel.push_for_loop(
        tk,
        zero,
        k,
        tile_size,
        |kernel| {
            let a_k = kernel.def_var(DType::UnsignedInt, ValueState::Immut, None);
            kernel.push_add(
                tk,
                local_col,
                a_k,
            );

            let b_k = kernel.def_var(DType::UnsignedInt, ValueState::Immut, None);
            kernel.push_add(
                tk,
                local_row,
                b_k,
            );

            let a_idx = kernel.def_var(DType::UnsignedInt, ValueState::Mut, None);

            if swap_a {
                kernel.push_mul(
                    a_k,
                    m,
                    a_idx,
                );
                kernel.push_add(
                    a_idx,
                    row,
                    a_idx,
                );
            } else {
                kernel.push_mul(
                    row,
                    k,
                    a_idx,
                );
                kernel.push_add(
                    a_idx,
                    a_k,
                    a_idx,
                );
            }

            kernel.push_add(
                a_idx,
                base,
                a_idx,
            );

            let a_val = kernel.def_var(DType::Float, ValueState::Mut, None);

            a_deepest = eval_node(
                root,
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

            kernel.push_shared_store(
                a_tile,
                shared_idx,
                a_val,
            );

            let b_idx = kernel.def_var(DType::UnsignedInt, ValueState::Mut, None);

            if swap_b {
                kernel.push_mul(
                    col,
                    k,
                    b_idx,
                );
                kernel.push_add(
                    b_idx,
                    b_k,
                    b_idx,
                );
            } else {
                kernel.push_mul(
                    b_k,
                    n,
                    b_idx,
                );
                kernel.push_add(
                    b_idx,
                    col,
                    b_idx,
                );
            }

            kernel.push_add(
                b_idx,
                base,
                b_idx,
            );

            let b_val = kernel.def_var(DType::Float, ValueState::Mut, None);

            b_deepest = eval_node(
                root,
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

            kernel.push_shared_store(
                b_tile,
                shared_idx,
                b_val,
            );

            kernel.push_barrier();

            let inner = kernel.def_var(DType::UnsignedInt, ValueState::Mut, None);

            kernel.push_for_loop(
                inner,
                zero,
                tile_size,
                one,
                |kernel| {
                    let a_s_idx = kernel.def_var(DType::UnsignedInt, ValueState::Immut, None);

                    kernel.push_mul(
                        local_row,
                        tile_size,
                        a_s_idx,
                    );
                    kernel.push_add(
                        a_s_idx,
                        inner,
                        a_s_idx,
                    );

                    let a_val = kernel.def_var(DType::Float, ValueState::Immut, None);
                    kernel.push_shared_load(
                        a_tile,
                        a_s_idx,
                        a_val,
                    );

                    let b_s_idx = kernel.def_var(DType::UnsignedInt, ValueState::Immut, None);

                    kernel.push_mul(
                        local_col,
                        tile_size,
                        b_s_idx,
                    );
                    kernel.push_add(
                        b_s_idx,
                        inner,
                        b_s_idx,
                    );

                    let b_val = kernel.def_var(DType::Float, ValueState::Immut, None);
                    kernel.push_shared_load(
                        b_tile,
                        b_s_idx,
                        b_val,
                    );

                    kernel.push_fma(
                        a_val,
                        b_val,
                        out,
                        out,
                    );

                    Ok(())
            })?;

            kernel.push_barrier();

            Ok(())
    })?;

    deepest.append(&mut a_deepest);
    deepest.append(&mut b_deepest);

    Ok(deepest)
}
