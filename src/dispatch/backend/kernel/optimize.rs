use crate::dispatch::{
    CompilationOptions, GpuBackend, OptFlags, backend::{DType, Op, ValueId, ValueState, kernel::RawKernel},
};
use alloc::vec::Vec;

pub fn optimize<B: GpuBackend>(kernel: &mut RawKernel, options: &CompilationOptions<B>) {
    for _ in 0..options.opt.passes {
        if options.opt.flags.contains(OptFlags::DEAD_CODE) {
            for value_id in 0..kernel.values.len() {
                if kernel.values[value_id].state == ValueState::Masked {
                    continue;
                }

                if !any_ops(kernel, |op| op.does_read(value_id)) {
                    kernel.values[value_id].state = ValueState::Masked;
                    erase_writes(kernel, value_id);
                }
            }
        }

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
                if let Op::Div { a, b } = &op {
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
                if let Some(Op::ConstF32 { value }) = kernel.values[b].init {
                    kernel.values[b].init.replace(Op::ConstF32 { value: 1.0 / value });

                    kernel.values[value_id].init.replace(Op::Mul { a, b, });
                }
            }
        }

        kernel.ops = erase_nops(&kernel.ops);
    }

    if options.opt.flags.contains(OptFlags::DEAD_CODE) {
        for value_id in 0..kernel.values.len() {
            if kernel.values[value_id].state == ValueState::Masked {
                continue;
            }

            if !any_ops(kernel, |op| op.does_read(value_id)) {
                kernel.values[value_id].state = ValueState::Masked;
                erase_writes(kernel, value_id);
            }
        }
    }
}

fn iter_values<R>(kernel: &mut RawKernel, mut f: impl FnMut(&mut Op, ValueId) -> Option<R>) -> Vec<R> {
    let mut value_ids = Vec::new();

    for value_id in 0..kernel.values.len() {
        if kernel.values[value_id].state == ValueState::Masked
            || kernel.values[value_id].dtype != DType::Float
        {
            continue;
        }

        if let Some(op) = &mut kernel.values[value_id].init
        && let Some(value) = f(op, value_id) {
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
