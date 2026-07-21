use crate::dispatch::{
    GpuBackend,
    backend::{Axis, DType, Graph, Kernel, LossType, Metadata, Op, Param, ParamTy, ValueState},
};
use alloc::vec::Vec;

#[inline]
pub fn lower_loss<B: GpuBackend + Clone>(graph: &Graph<B>, meta: Metadata) -> Kernel {
    let root = graph.nodes.len() - 1;
    let root_node = &graph.nodes[root];

    let mut kernel = Kernel {
        meta,
        params: Vec::new(),
        shared: Vec::new(),
        values: Vec::new(),
        ops: Vec::new(),
        block: [0; 3],
        root,
        iter_space: root_node.shape.clone(),
    };

    if graph
        .nodes
        .iter()
        .all(|x| x.op.is_elementwise() || x.op.is_leaf())
    {
        kernel.block = [256, 1, 1];
    } else {
        kernel.block = [16, 16, 1];
    }

    kernel.params.push(Param {
        dtype: DType::UnsignedInt,
        ty: ParamTy::Uniform,
    });

    let loss_param = kernel.params.len();
    kernel.params.push(Param {
        dtype: DType::Float,
        ty: ParamTy::ReadWrite,
    });

    let grad_param = kernel.params.len();
    kernel.params.push(Param {
        dtype: DType::Float,
        ty: ParamTy::ReadWrite,
    });

    let pred_param = kernel.params.len();
    kernel.params.push(Param {
        dtype: DType::Float,
        ty: ParamTy::ReadOnly,
    });

    let target_param = kernel.params.len();
    kernel.params.push(Param {
        dtype: DType::Float,
        ty: ParamTy::ReadOnly,
    });

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

            kernel.overwrite_var(gid2, Op::Mul { a: gid2, b: d });

            kernel.overwrite_var(gid, Op::Add { a: gid, b: gid2 });

            kernel.overwrite_var(total, Op::Mul { a: total, b: d });
        }
    }

    let pred = kernel.def_var(
        DType::Float,
        ValueState::Immut,
        Some(Op::Load {
            param: pred_param,
            index: gid,
        }),
    );
    let target = kernel.def_var(
        DType::Float,
        ValueState::Immut,
        Some(Op::Load {
            param: target_param,
            index: gid,
        }),
    );

    let loss_val;
    let grad_val;

    match graph.loss {
        LossType::MeanSquaredError => {
            let diff = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Sub { a: pred, b: target }),
            );

            loss_val = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Mul { a: diff, b: diff }),
            );

            let two = kernel.def_var(
                DType::Float,
                ValueState::Inline,
                Some(Op::ConstF32 { value: 2.0 }),
            );

            grad_val = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Mul { a: two, b: diff }),
            );
        }

        LossType::BinaryCrossEntropy => {
            let one = kernel.def_var(
                DType::Float,
                ValueState::Inline,
                Some(Op::ConstF32 { value: 1.0 }),
            );

            let log_pred =
                kernel.def_var(DType::Float, ValueState::Immut, Some(Op::Log { x: pred }));

            let one_minus_target = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Sub { a: one, b: target }),
            );

            let one_minus_pred = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Sub { a: one, b: target }),
            );

            let log_one_minus_pred = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Log { x: one_minus_pred }),
            );

            let term1 = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Mul {
                    a: target,
                    b: log_pred,
                }),
            );

            let term2 = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Mul {
                    a: one_minus_target,
                    b: log_one_minus_pred,
                }),
            );

            loss_val = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Sub { a: term2, b: term1 }),
            );

            grad_val = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Sub { a: pred, b: target }),
            );
        }

        LossType::CrossEntropy => {
            let log_pred =
                kernel.def_var(DType::Float, ValueState::Immut, Some(Op::Log { x: pred }));

            let term = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Mul {
                    a: target,
                    b: log_pred,
                }),
            );

            loss_val = kernel.def_var(DType::Float, ValueState::Immut, Some(Op::Neg { x: term }));

            grad_val = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Sub { a: pred, b: target }),
            );
        }
    }

    kernel.param_store(loss_param, gid, loss_val);
    kernel.param_store(grad_param, gid, grad_val);

    kernel
}
