use crate::dispatch::backend::{DType, LossType, Op, ValueState};

impl LossType {
    pub const LOSS_MSE: Self = Self {
        lower: |kernel, pred, target, _, _, _, _| {
            let diff = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Sub { a: pred, b: target }),
            );

            let loss_val = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Mul { a: diff, b: diff }),
            );

            let two = kernel.def_var(
                DType::Float,
                ValueState::Inline,
                Some(Op::ConstF32 { value: 2.0 }),
            );

            let grad_val = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Mul { a: two, b: diff }),
            );

            (loss_val, grad_val)
        },
    };

    pub const LOSS_BCE: Self = Self {
        lower: |kernel, pred, target, _, _, _, _| {
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

            let loss_val = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Sub { a: term2, b: term1 }),
            );

            let grad_val = kernel.def_var(
                DType::Float,
                ValueState::Immut,
                Some(Op::Sub { a: pred, b: target }),
            );

            (loss_val, grad_val)
        },
    };

    pub const LOSS_CROSS_ENTROPY: Self = Self {
        lower: |kernel, pred, target, _, target_param, row, col| {
            kernel.update_param_dtype(target_param, DType::UnsignedInt);
            kernel.update_var_dtype(target, DType::UnsignedInt);
            kernel.update_var_init(
                target,
                Op::ParamLoad {
                    param: target_param,
                    index: row,
                },
            );

            let target_eq_col = kernel.def_var(
                DType::Bool,
                ValueState::Inline,
                Some(Op::Eq { a: target, b: col }),
            );

            let loss_val = kernel.def_var(
                DType::Float,
                ValueState::Mut,
                Some(Op::ConstF32 { value: 0.0 }),
            );

            let grad_val = kernel.def_var(
                DType::Float,
                ValueState::Mut,
                Some(Op::CopyVar { id: pred }),
            );

            kernel.push_if(target_eq_col, |kernel| {
                let one = kernel.def_var(
                    DType::Float,
                    ValueState::Inline,
                    Some(Op::ConstF32 { value: 1.0 }),
                );

                let log_pred =
                    kernel.def_var(DType::Float, ValueState::Inline, Some(Op::Log { x: pred }));

                kernel.overwrite_var(loss_val, Op::Neg { x: log_pred });

                kernel.overwrite_var(grad_val, Op::Sub { a: pred, b: one });
            });

            (loss_val, grad_val)
        },
    };
}
