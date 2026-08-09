use std::time::Instant;

use fused_gpu::dispatch::{
    CompilationOptions, GpuContext, Schedule, backend::{Graph, LossType, Metadata, kernel::KernelsChained},
};

fn matmul_chain3_forward_backward() {
    const M: u32 = 16;
    const N: u32 = 16;
    const K: u32 = 128;
    const H: u32 = 64;

    const A_VAL: f32 = 3.0;
    const B_VAL: f32 = 2.0;
    const C_VAL: f32 = 1.0;
    const D_VAL: f32 = 0.5;
    const E_VAL: f32 = 1.0;

    let mut meta = Metadata::new();
    let m = meta.new_field();
    let n = meta.new_field();
    let k = meta.new_field();
    let h = meta.new_field();

    let mut graph = Graph::new(LossType::LOSS_MSE);

    let a = graph.input(&[m, k]);
    let b = graph.input(&[k, n]);
    let c = graph.input(&[h, m]);
    let d = graph.input(&[n, h]);
    let e = graph.input(&[h, h]);

    let x = graph.matmul(a, b);
    let y = graph.matmul(c, x);
    let z = graph.matmul(y, d);
    graph.add(z, e);

    let saved = KernelsChained::compute_saved_nodes(&graph);
    let options = CompilationOptions::default();

    let compile_start = Instant::now();

    let ctx = pollster::block_on(GpuContext::new()).unwrap();
    graph.validate(meta).unwrap();
    graph.topo_sort().unwrap();
    graph.rebuild_outputs();
    let ir = graph.lower(meta, &options, &saved).unwrap();
    let kernels = ctx.compile(&ir, &options).unwrap();

    let compile_elapsed = compile_start.elapsed();

    println!("COMPILE TIME: {compile_elapsed:?} elapsed");

    let tensor_start = Instant::now();

    let in_tensors = [
        ctx.new_tensor_init(&[M, K], &[A_VAL; (M * K) as usize]),
        ctx.new_tensor_init(&[K, N], &[B_VAL; (K * N) as usize]),
        ctx.new_tensor_init(&[H, M], &[C_VAL; (H * M) as usize]),
        ctx.new_tensor_init(&[N, H], &[D_VAL; (N * H) as usize]),
        ctx.new_tensor_init(&[H, H], &[E_VAL; (H * H) as usize]),
    ];

    let meta_binding = [M, N, K, H];
    assert!(meta.validate_meta(&meta_binding));

    let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding);

    ctx.upload(&saved_tensors.seed, &[1_f32; (H * H) as usize])
        .unwrap()
        .sync();

    let tensor_elapsed = tensor_start.elapsed();

    println!("TENSOR INIT TIME: {tensor_elapsed:?} elapsed");

    let schedule = ctx.schedule(&kernels, &meta_binding, &in_tensors, &saved_tensors);

    model_runtime(&ctx, &schedule);
}

fn model_runtime(
    ctx: &GpuContext,
    schedule: &Schedule,
) {
    for iters in [100, 10, 1] {
        let runtime_start = Instant::now();

        for _ in 0..iters {
            ctx.launch_forward(schedule);

            ctx.launch_backward(schedule);
        }

        let runtime_elapsed = runtime_start.elapsed();

        println!("MODEL RUNTIME {iters} TIME: {runtime_elapsed:?} elapsed");
    }
}

fn main() {
    matmul_chain3_forward_backward();
}
