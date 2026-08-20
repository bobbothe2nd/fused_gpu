use crate::dispatch::{
    CompilationOptions, GpuBackend, OptFlags,
    backend::{DType, Op, ValueId, ValueState, kernel::RawKernel},
};
use alloc::vec::Vec;

pub fn optimize<B: GpuBackend>(kernel: &mut RawKernel, options: &CompilationOptions<B>) {
    for _ in 0..options.opt.passes {
        dead_code_elimination(kernel, options);

        if options.opt.flags.contains(OptFlags::MUL_ADD) {
            for value_id in 0..kernel.values.len() {
                if kernel.values[value_id].state == ValueState::Masked
                    || kernel.values[value_id].dtype != DType::Float
                {
                    continue;
                }

                if let Some(Op::Add { a, b }) = kernel.values[value_id].init {
                    if let Some(Op::Mul { a: a_mul, b: b_mul }) = kernel.values[a].init {
                        kernel.values[a].state = ValueState::Masked;

                        kernel.values[value_id].init.replace(Op::Fma {
                            a: a_mul,
                            b: b_mul,
                            c: b,
                        });
                    } else if let Some(Op::Mul { a: a_mul, b: b_mul }) = kernel.values[b].init {
                        kernel.values[b].state = ValueState::Masked;

                        kernel.values[value_id].init.replace(Op::Fma {
                            a: a_mul,
                            b: b_mul,
                            c: a,
                        });
                    }
                }
            }

            let additions = iter_values(kernel, |op, value_id| {
                if let Op::Add { a, b } = &op {
                    Some((*a, *b, value_id))
                } else {
                    None
                }
            });

            for (a, b, value_id) in additions {
                if let Some(Op::Mul { a: a_mul, b: b_mul }) = kernel.values[a].init {
                    kernel.values[value_id].init.replace(Op::Fma {
                        a: a_mul,
                        b: b_mul,
                        c: b,
                    });
                }

                if let Some(Op::Mul { a: a_mul, b: b_mul }) = kernel.values[b].init {
                    kernel.values[value_id].init.replace(Op::Fma {
                        a: a_mul,
                        b: b_mul,
                        c: a,
                    });
                }
            }
        }

        if options.opt.flags.contains(OptFlags::DIV_CONST) {
            let divisions = iter_values(kernel, |op, value_id| {
                if let Op::Div { a, b } = &op {
                    Some((*a, *b, value_id))
                } else {
                    None
                }
            });

            for (a, b, value_id) in divisions {
                if kernel.values[b].state == ValueState::Mut {
                    continue;
                }

                if let Some(Op::ConstF32 { value }) = kernel.values[b].init {
                    kernel.values[b]
                        .init
                        .replace(Op::ConstF32 { value: 1.0 / value });

                    kernel.values[value_id].init.replace(Op::Mul { a, b });
                }
            }
        }

        if options.opt.flags.contains(OptFlags::CONST_FOLD) {
            for value_id in 0..kernel.values.len() {
                if kernel.values[value_id].state == ValueState::Masked {
                    continue;
                }

                const_fold(kernel, value_id);
            }
        }

        if options.opt.flags.contains(OptFlags::IDENTITY) {
            for value_id in 0..kernel.values.len() {
                if kernel.values[value_id].state == ValueState::Masked {
                    continue;
                }

                identity(kernel, value_id);
            }
        }

        kernel.ops = erase_nops(&kernel.ops);
    }

    dead_code_elimination(kernel, options);
}

fn dead_code_elimination<B: GpuBackend>(kernel: &mut RawKernel, options: &CompilationOptions<B>) {
    if options.opt.flags.contains(OptFlags::DEAD_CODE) {
        for value_id in 0..kernel.values.len() {
            if kernel.values[value_id].state == ValueState::Masked {
                continue;
            }

            if !any_ops(kernel, |op| {
                op.does_read(value_id) && !op.does_write(value_id)
            }) {
                kernel.values[value_id].state = ValueState::Masked;
                erase_writes(kernel, value_id);
            }
        }
    }
}

fn iter_values<R>(kernel: &RawKernel, mut f: impl FnMut(&Op, ValueId) -> Option<R>) -> Vec<R> {
    let mut value_ids = Vec::new();

    for value_id in 0..kernel.values.len() {
        if kernel.values[value_id].state == ValueState::Masked
            || kernel.values[value_id].dtype != DType::Float
        {
            continue;
        }

        if let Some(op) = &kernel.values[value_id].init
            && let Some(value) = f(op, value_id)
        {
            value_ids.push(value);
        }
    }

    value_ids
}

fn any_ops(kernel: &RawKernel, mut f: impl FnMut(&Op) -> bool) -> bool {
    for op in &kernel.ops {
        if f(op) {
            return true;
        }
    }

    for value in &kernel.values {
        if let Some(init) = &value.init
            && f(init)
        {
            return true;
        }
    }

    false
}

fn erase_writes(kernel: &mut RawKernel, value_id: ValueId) {
    for op in &mut kernel.ops {
        if op.does_write(value_id) {
            *op = Op::Nop;
        }
    }
}

fn erase_nops(ops: &[Op]) -> Vec<Op> {
    let mut new_ops = Vec::new();

    for op in ops {
        if *op != Op::Nop {
            new_ops.push(*op);
        }
    }

    new_ops
}

macro_rules! const_fold_op {
    ($kernel:ident, $value_id:ident, $a:ident, $b:expr, $opu:path, $opi:path, $($opf:path)?) => {{
        let a = $kernel.values[*$a];
        let b = $kernel.values[*$b];

        if a.state == ValueState::Mut
        || b.state == ValueState::Mut {
            return;
        }

        if let Some(a_op) = a.init
        && let Some(b_op) = b.init {
            match (a_op, b_op) {
                $((Op::ConstF32 { value: a }, Op::ConstF32 { value: b }) => {
                    $kernel.values[$value_id].init.replace(Op::ConstF32 { value: $opf(a, b) });
                })?
                (Op::ConstU32 { value: a }, Op::ConstU32 { value: b }) => {
                    $kernel.values[$value_id].init.replace(Op::ConstU32 { value: $opu(a, b) });
                }
                (Op::ConstI32 { value: a }, Op::ConstI32 { value: b }) => {
                    $kernel.values[$value_id].init.replace(Op::ConstI32 { value: $opi(a, b) });
                }
                _ => {}
            }
        }
    }};
}

fn const_fold(kernel: &mut RawKernel, value_id: ValueId) {
    use core::ops::{Add, Div, Mul, Shl, Shr, Sub};

    if let Some(init) = &kernel.values[value_id].init {
        match init {
            Op::Add { a, b } => {
                const_fold_op!(kernel, value_id, a, b, u32::add, i32::add, f32::add);
            }
            Op::Mul { a, b } => {
                const_fold_op!(kernel, value_id, a, b, u32::mul, i32::mul, f32::mul);
            }
            Op::Sub { a, b } => {
                const_fold_op!(kernel, value_id, a, b, u32::sub, i32::sub, f32::sub);
            }
            Op::Div { a, b } => {
                const_fold_op!(kernel, value_id, a, b, u32::div, i32::div, f32::div);
            }
            Op::Shr { a, b } => const_fold_op!(kernel, value_id, a, b, u32::shr, i32::shr,),
            Op::Shl { a, b } => const_fold_op!(kernel, value_id, a, b, u32::shl, i32::shl,),
            Op::Fma { a, b, c } => {
                let a = kernel.values[*a];
                let b = kernel.values[*b];
                let c = kernel.values[*c];

                if a.state == ValueState::Mut
                    || b.state == ValueState::Mut
                    || c.state == ValueState::Mut
                {
                    return;
                }

                if let Some(Op::ConstF32 { value: a }) = a.init
                    && let Some(Op::ConstF32 { value: b }) = b.init
                    && let Some(Op::ConstF32 { value: c }) = c.init
                {
                    kernel.values[value_id]
                        .init
                        .replace(Op::ConstF32 { value: (a * b) + c });
                }
            }
            _ => {}
        }
    }
}

fn identity(kernel: &mut RawKernel, value_id: ValueId) {
    if let Some(init) = kernel.values[value_id].init {
        match init {
            Op::Add { a, b } => {
                if kernel.values[a].init.is_some_and(|op| op.is_zero())
                    && kernel.values[a].state != ValueState::Mut
                {
                    kernel.values[value_id].init.replace(Op::CopyVar { id: b });
                } else if kernel.values[b].init.is_some_and(|op| op.is_zero())
                    && kernel.values[b].state != ValueState::Mut
                {
                    kernel.values[value_id].init.replace(Op::CopyVar { id: a });
                }
            }
            Op::Sub { a, b } => {
                if kernel.values[b].init.is_some_and(|op| op.is_zero())
                    && kernel.values[b].state != ValueState::Mut
                {
                    kernel.values[value_id].init.replace(Op::CopyVar { id: a });
                }
            }
            Op::Mul { a, b } => {
                if kernel.values[a].init.is_some_and(|op| op.is_one())
                    && kernel.values[a].state != ValueState::Mut
                {
                    kernel.values[value_id].init.replace(Op::CopyVar { id: b });
                }
                if kernel.values[b].init.is_some_and(|op| op.is_one())
                    && kernel.values[b].state != ValueState::Mut
                {
                    kernel.values[value_id].init.replace(Op::CopyVar { id: a });
                }
            }
            Op::Div { a, b }
                if kernel.values[b].init.is_some_and(|op| op.is_one())
                    && kernel.values[b].state != ValueState::Mut =>
            {
                kernel.values[value_id].init.replace(Op::CopyVar { id: a });
            }
            _ => {}
        }
    }
}
