use fused_gpu::dispatch::{
    CompilationOptions, GpuContext, Schedule,
    backend::{Graph, LossType, Metadata},
};
use gpu_telemetry::monitor::{GpuMonitor, telemetry::Telemetry};
use std::time::{Duration, Instant};

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

    let mut graph = Graph::new(LossType::MEAN_SQUARED_ERROR);

    let a = graph.input(&[m, k]);
    let b = graph.input(&[k, n]);
    let c = graph.input(&[h, m]);
    let d = graph.input(&[n, h]);
    let e = graph.input(&[h, h]);

    let x = graph.matmul(a, b);
    let y = graph.matmul(c, x);
    let z = graph.matmul(y, d);
    graph.add(z, e);

    let saved = graph.compute_saved_nodes();
    let options = CompilationOptions::default();

    let compile_start = Instant::now();

    let ctx = GpuContext::new().unwrap();
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

    let schedule = ctx.schedule(&kernels, &meta_binding, &in_tensors, &saved_tensors).unwrap();

    let monitor: GpuMonitor<Telemetry> = GpuMonitor::start(Duration::from_millis(3)).unwrap();

    model_runtime(&ctx, &schedule);

    let telemetry = monitor.stop().unwrap();

    for (i, sample) in telemetry.samples.iter().enumerate() {
        println!("SAMPLE {i}:");

        for heap in &sample.heaps {
            println!(" {} HEAP:", if heap.dev_local { "LOCAL" } else { "SHARED" });

            if let Some(size) = heap.size {
                println!("  size: {size},");
            }

            if let Some(budget) = heap.budget {
                println!("  budget: {budget},");
            }

            if let Some(usage) = heap.usage {
                println!("  usage: {usage},");
            }

            if let Some(reservation) = heap.reservation {
                println!("  reservation: {reservation},");
            }

            if let Some(available) = heap.available_for_reservation {
                println!("  available for reservation: {available},");
            }
        }
    }
}

fn model_runtime(ctx: &GpuContext, schedule: &Schedule) {
    for iters in [100, 10, 1] {
        let mut state = ctx.prepare_batch();

        {
            let mut pass = ctx.start_batch(&mut state);

            for _ in 0..iters {
                pass.dispatch_forward(schedule);

                pass.dispatch_backward(schedule);
            }
        }

        let encoded = state.encode();

        let runtime_start = Instant::now();

        let _ = encoded.submit().sync();

        let runtime_elapsed = runtime_start.elapsed();

        println!("MODEL RUNTIME + SYNCHRONIZATION {iters} TIME: {runtime_elapsed:?} elapsed");
    }
}

fn main() {
    matmul_chain3_forward_backward();
}
