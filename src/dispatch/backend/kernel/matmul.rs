use crate::{
    dispatch::{
        CompilationOptions, GpuBackend,
        backend::{
            DType, Graph, NodeId, Op, ParamId, ValueId, ValueState,
            kernel::{Kernel, NodeInput},
        },
    },
    errors::Error,
};
use alloc::vec::Vec;

pub fn lower_matmul_recursive<B: GpuBackend>(
    eval_node: impl Fn(
        NodeId,
        NodeId,
        &NodeInput,
        ValueId,
        &mut Vec<NodeId>,
        &Graph,
        &[Option<ParamId>],
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
    input: NodeId,
    resolved: &mut Vec<NodeId>,
    transpose_a: bool,
    transpose_b: bool,
    a_node: &NodeInput,
    b_node: &NodeInput,
    graph: &Graph,
    out: ValueId,
    node_params: &[Option<ParamId>],
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

    let m = kernel.def_var(
        DType::UnsignedInt,
        ValueState::Immut,
        Some(Op::ReadMeta {
            param: 0,
            field: a_node_shape[a_node_shape.len() - 2],
        }),
    );
    let n = kernel.def_var(
        DType::UnsignedInt,
        ValueState::Immut,
        Some(Op::ReadMeta {
            param: 0,
            field: b_node_shape[b_node_shape.len() - 1],
        }),
    );
    let k = kernel.def_var(
        DType::UnsignedInt,
        ValueState::Immut,
        Some(Op::ReadMeta {
            param: 0,
            field: a_node_shape[a_node_shape.len() - 1],
        }),
    );

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

pub fn forward_matmul<B: GpuBackend>(
    eval_node: impl Fn(
        NodeId,
        NodeId,
        &NodeInput,
        ValueId,
        &mut Vec<NodeId>,
        &Graph,
        &[Option<ParamId>],
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
    input: NodeId,
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
    node_params: &[Option<ParamId>],
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

    let one = kernel.def_var(
        DType::UnsignedInt,
        ValueState::Immut,
        Some(Op::ConstU32 { value: 1 }),
    );

    kernel.overwrite_var(out, Op::ConstF32 { value: 0.0 });

    let tk = kernel.def_var(
        DType::UnsignedInt,
        ValueState::Mut,
        Some(Op::ConstU32 { value: 0 }),
    );

    let tile_row = kernel.def_var(
        DType::UnsignedInt,
        ValueState::Inline,
        Some(Op::Mul {
            a: local_row,
            b: tile_size,
        }),
    );
    let shared_idx = kernel.def_var(
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
        let a_k = kernel.def_var(
            DType::UnsignedInt,
            ValueState::Immut,
            Some(Op::Add {
                a: tk,
                b: local_col,
            }),
        );

        let b_k = kernel.def_var(
            DType::UnsignedInt,
            ValueState::Immut,
            Some(Op::Add {
                a: tk,
                b: local_row,
            }),
        );

        let a_idx = if swap_a {
            let a_row = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Mul { a: a_k, b: m }),
            );
            let a_col = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Add { a: a_row, b: row }),
            );
            kernel.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::Add { a: a_col, b: base }),
            )
        } else {
            let a_row = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Mul { a: row, b: k }),
            );
            let a_col = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Add { a: a_row, b: a_k }),
            );
            kernel.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::Add { a: a_col, b: base }),
            )
        };

        let a_val = kernel.def_var(DType::Float, ValueState::Mut, None);

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

        kernel.shared_store(a_tile, shared_idx, a_val);

        let b_idx = if swap_b {
            let b_row = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Mul { a: col, b: k }),
            );
            let b_col = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Add { a: b_row, b: b_k }),
            );
            kernel.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::Add { a: b_col, b: base }),
            )
        } else {
            let b_row = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Mul { a: b_k, b: n }),
            );
            let b_col = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Add { a: b_row, b: col }),
            );
            kernel.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::Add { a: b_col, b: base }),
            )
        };

        let b_val = kernel.def_var(DType::Float, ValueState::Mut, None);

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

        kernel.shared_store(b_tile, shared_idx, b_val);

        kernel.push_barrier();

        let inner = kernel.def_var(
            DType::UnsignedInt,
            ValueState::Mut,
            Some(Op::ConstU32 { value: 0 }),
        );

        kernel.push_for_loop(inner, tile_size, one, |kernel| {
            let a_s_row = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Mul {
                    a: local_row,
                    b: tile_size,
                }),
            );
            let a_s_idx = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::Add {
                    a: a_s_row,
                    b: inner,
                }),
            );

            let a_val = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::SharedLoad {
                    mem: a_tile,
                    index: a_s_idx,
                }),
            );

            let b_s_row = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Inline,
                Some(Op::Mul {
                    a: inner,
                    b: tile_size,
                }),
            );
            let b_s_idx = kernel.def_var(
                DType::UnsignedInt,
                ValueState::Immut,
                Some(Op::Add {
                    a: b_s_row,
                    b: local_col,
                }),
            );

            let b_val = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::SharedLoad {
                    mem: b_tile,
                    index: b_s_idx,
                }),
            );

            kernel.overwrite_var(
                out,
                Op::Fma {
                    a: a_val,
                    b: b_val,
                    c: out,
                },
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
