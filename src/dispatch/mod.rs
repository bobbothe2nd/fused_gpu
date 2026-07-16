//! Full dispatch of math operations and GPU contVec;

use alloc::vec::Vec;
use core::{fmt::Debug, marker::PhantomData};

use crate::{
    dispatch::backend::{
        Graph, MetaId, Param,
        kernel::{Dependencies, Kernel, SaveIndicator},
    },
    errors::Error,
    tensor::{Tensor, build_dims, calc_grid},
};
use briny::{
    raw::{slice_to_bytes, slice_to_bytes_mut},
    traits::Pod,
};

pub mod backend;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PollStatus {
    QueueEmpty,
    Failed,
    Pending,
    Ready,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct CompilationOptions<B: GpuBackend + ?Sized> {
    pub target: TargetCompilationOptions<B>,
    pub debug: DebugCompilationOptions,
    pub opt: OptCompilationOptions,
}

impl<B: GpuBackend> Default for CompilationOptions<B> {
    fn default() -> Self {
        Self {
            target: B::TARGET_SPEC,
            debug: DebugCompilationOptions {
                pretty_print_ir: false,
            },
            opt: OptCompilationOptions {
                opt_passes: 0,
                opt_level: 0,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TargetCompilationOptions<B: GpuBackend + ?Sized> {
    /// Support for linear algebra accelerator (e.g. tensor cores)?
    lin_acc: bool,

    /// Support for asyncronous memory loads/reads?
    async_mem_load: bool,

    /// Support for asyncronous memory stores/writes?
    async_mem_store: bool,

    _marker: PhantomData<B>,
}

impl<B: GpuBackend> TargetCompilationOptions<B> {
    #[must_use]
    pub const fn new(lin_acc: bool, async_mem_load: bool, async_mem_store: bool) -> Self {
        Self {
            lin_acc,
            async_mem_load,
            async_mem_store,
            _marker: PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct DebugCompilationOptions {
    pub pretty_print_ir: bool,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct OptCompilationOptions {
    pub opt_passes: u8,
    pub opt_level: u8,
}

pub trait GpuBufferBackend: Debug + Clone {
    fn size(&self) -> u32;

    fn size_bytes(&self) -> u32;
}

pub trait GpuKernelBackend {
    fn iteration_space(&self) -> &[MetaId];
}

pub trait GpuBackend {
    const TARGET_SPEC: TargetCompilationOptions<Self>;

    type Buffer: GpuBufferBackend;
    type Kernel: GpuKernelBackend;
    type SubmissionIndex;
    type ParamLayout;

    fn alloc(&self, len: usize) -> Self::Buffer;

    fn alloc_init(&self, data: &[u8]) -> Self::Buffer;

    fn alloc_meta(&self, data: &[u32]) -> Self::Buffer;

    fn upload(&self, buffer: &Self::Buffer, data: &[u8]) -> Result<Self::SubmissionIndex, Error>;

    fn download(&self, buffer: &Self::Buffer, out: &mut [u8]) -> Result<(), Error>;

    fn init_params(
        &self,
        src: &[Param],
        options: &CompilationOptions<Self>,
    ) -> Result<Self::ParamLayout, Error>;

    fn compile(
        &self,
        src: &Kernel,
        params: &Self::ParamLayout,
        options: &CompilationOptions<Self>,
    ) -> Result<Self::Kernel, Error>;

    fn launch(
        &self,
        kernel: &Self::Kernel,
        wg: [u32; 3],
        bindings: &[&Self::Buffer],
    ) -> Self::SubmissionIndex;

    fn sync(&self, submission_index: Self::SubmissionIndex);

    fn poll(&self) -> PollStatus;
}

/// Allocate a buffer on the GPU.
pub fn gpu_alloc<B: GpuBackend>(context: &B, len: usize) -> GpuBuffer<B> {
    GpuBuffer {
        inner: context.alloc(len),
    }
}

/// Allocate a buffer on the GPU and copy CPU memory into it.
pub fn gpu_alloc_init<T: Pod, B: GpuBackend>(context: &B, data: &[T]) -> GpuBuffer<B> {
    GpuBuffer {
        inner: context.alloc_init(slice_to_bytes(data)),
    }
}

/// Generic GPU buffer for any backend.
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct GpuBuffer<B: GpuBackend = backend::GpuContext> {
    pub(crate) inner: B::Buffer,
}

impl<B: GpuBackend> GpuBuffer<B> {
    /// Requests the length of the buffer in 32-bit chunks (`f32`, `u32`, etc.).
    pub fn size(&self) -> u32 {
        self.inner.size()
    }

    /// Requests the length of the buffer in bytes.
    pub fn size_bytes(&self) -> u32 {
        self.inner.size_bytes()
    }
}

#[derive(Debug)]
pub struct KernelGroup<B: GpuBackend = backend::GpuContext> {
    pub(crate) forward: Vec<(Dependencies<B::Kernel>, usize)>,
    pub(crate) backward: Vec<(Dependencies<B::Kernel>, usize)>,
    pub(crate) loss: B::Kernel,
    pub(crate) block: [u32; 3],
}

#[must_use]
pub struct SubmissionIndex<'a, B: GpuBackend>(B::SubmissionIndex, &'a GpuContext<B>);

impl<B: GpuBackend> SubmissionIndex<'_, B> {
    pub fn sync(self) {
        self.1.inner.sync(self.0);
    }
}

/// Generic GPU context storing a handle to the device and shaders.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuContext<B: GpuBackend = backend::GpuContext> {
    pub(crate) inner: B,
}

impl GpuContext<backend::GpuContext> {
    /// Non-blocking creation of the context by searching for GPU.
    pub async fn new() -> Result<Self, Error> {
        Ok(Self {
            inner: backend::GpuContext::new().await?,
        })
    }
}

impl<B: GpuBackend> GpuContext<B> {
    /// Wrapping the inner context.
    pub const fn new_with_context(ctx: B) -> Self {
        Self { inner: ctx }
    }

    /// Compiles a graph into a kernels.
    pub fn compile(
        &self,
        ir: &backend::kernel::KernelGroup,
        options: &CompilationOptions<B>,
    ) -> Result<KernelGroup<B>, Error> {
        let pipeline_layout = self.inner.init_params(&ir.forward[0].val.params, options)?;
        let forward = ir
            .forward
            .iter()
            .map(|kernel| {
                Ok((
                    Dependencies {
                        val: self.inner.compile(&kernel.val, &pipeline_layout, options)?,
                        dep: kernel.dep.clone(),
                    },
                    kernel.val.root,
                ))
            })
            .collect::<Result<Vec<(Dependencies<<B as GpuBackend>::Kernel>, usize)>, Error>>()?;

        let pipeline_layout = self
            .inner
            .init_params(&ir.backward[0].val.params, options)?;
        let backward = ir
            .backward
            .iter()
            .map(|kernel| {
                Ok((
                    Dependencies {
                        val: self.inner.compile(&kernel.val, &pipeline_layout, options)?,
                        dep: kernel.dep.clone(),
                    },
                    kernel.val.root,
                ))
            })
            .collect::<Result<Vec<(Dependencies<<B as GpuBackend>::Kernel>, usize)>, Error>>()?;

        let pipeline_layout = self.inner.init_params(&ir.loss.params, options)?;
        let loss = self.inner.compile(&ir.loss, &pipeline_layout, options)?;

        Ok(KernelGroup {
            forward,
            backward,
            loss,
            block: ir.loss.block,
        })
    }

    pub fn download<T: Pod>(&self, tensor: &Tensor<B>, dst: &mut [T]) -> Result<(), Error> {
        self.inner
            .download(&tensor.data.inner, slice_to_bytes_mut(dst))
    }

    pub fn upload<T: Pod>(
        &self,
        tensor: &Tensor<B>,
        dst: &[T],
    ) -> Result<SubmissionIndex<'_, B>, Error> {
        self.inner
            .upload(&tensor.data.inner, slice_to_bytes(dst))
            .map(|x| SubmissionIndex(x, self))
    }

    /// Launches a forward kernel from the compiled kernels with metadata and tensors.
    ///
    /// This function relies on the assumption that tensors matches the graph inputs and
    /// the metadata matches the graph metadata when compiling.
    pub fn launch_forward(
        &self,
        kernels: &KernelGroup<B>,
        meta: &[u32],
        in_tensors: &[Tensor<B>],
        alloc_tensors: &AllocTensors<B>,
    ) {
        let mut bindings = Vec::with_capacity(2 + in_tensors.len());

        let meta_binding = self.inner.alloc_meta(meta);
        bindings.push(&meta_binding);

        for t in &alloc_tensors.forward_saved {
            bindings.push(&t.data.inner);
        }

        for t in in_tensors {
            bindings.push(&t.data.inner);
        }

        bindings.push(&alloc_tensors.forward_out.data.inner);

        let mut resolved = Vec::new();
        let mut tmp_res = Vec::new();

        while resolved.len() < kernels.forward.len() {
            let submissions = kernels.forward.iter().filter_map(|(kernel, idx)| {
                if resolved.contains(idx) {
                    return None;
                }

                if kernel.dep.iter().all(|x| resolved.contains(x)) {
                    tmp_res.push(*idx);

                    let iter_space = build_dims(kernel.val.iteration_space(), meta);
                    let grid = calc_grid(&iter_space, kernels.block);

                    Some(self.inner.launch(&kernel.val, grid, &bindings))
                } else {
                    None
                }
            });

            let mut subs = 0;

            submissions.for_each(|submission_index| {
                subs += 1;
                self.inner.sync(submission_index);
            });

            if subs == 0 {
                break;
            }

            resolved.append(&mut tmp_res);
        }
    }

    /// Launches a backward kernel from the compiled kernels with metadata and tensors.
    ///
    /// This function relies on the assumption that tensors matches the graph inputs and
    /// the metadata matches the graph metadata when compiling.
    pub fn launch_backward(
        &self,
        kernels: &KernelGroup<B>,
        meta: &[u32],
        forward_tensors: &[Tensor<B>],
        tensors: &AllocTensors<B>,
    ) {
        let mut bindings = Vec::new();

        let meta_binding = self.inner.alloc_meta(meta);
        bindings.push(&meta_binding);
        bindings.push(&tensors.seed.data.inner);

        for t in &tensors.grad_tensors {
            bindings.push(&t.data.inner);
        }

        for t in forward_tensors {
            bindings.push(&t.data.inner);
        }

        for t in &tensors.forward_saved {
            bindings.push(&t.data.inner);
        }

        let mut resolved = Vec::new();
        let mut tmp_res = Vec::new();

        while resolved.len() < kernels.backward.len() {
            let submissions = kernels.backward.iter().filter_map(|(kernel, idx)| {
                if resolved.contains(idx) {
                    return None;
                }

                if kernel.dep.iter().all(|x| resolved.contains(x)) {
                    tmp_res.push(*idx);

                    let iter_space = build_dims(kernel.val.iteration_space(), meta);
                    let grid = calc_grid(&iter_space, kernels.block);

                    Some(self.inner.launch(&kernel.val, grid, &bindings))
                } else {
                    None
                }
            });

            submissions.for_each(|submission_index| self.inner.sync(submission_index));

            resolved.append(&mut tmp_res);
        }
    }

    pub fn launch_loss(
        &self,
        kernels: &KernelGroup<B>,
        meta: &[u32],
        target: &Tensor<B>,
        tensors: &AllocTensors<B>,
    ) {
        let grid = tensors.loss_t.calc_grid(kernels.block);

        let meta_binding = self.inner.alloc_meta(meta);
        let bindings = [
            &meta_binding,
            &tensors.loss_t.data.inner,
            &tensors.seed.data.inner,
            &tensors.forward_out.data.inner,
            &target.data.inner,
        ];

        let submission_index = self.inner.launch(&kernels.loss, grid, &bindings);

        self.inner.sync(submission_index);
    }

    pub fn alloc_tensors(
        &self,
        graph: &Graph,
        saved: &[SaveIndicator],
        meta: &[u32],
    ) -> AllocTensors<B> {
        let mut forward_saved = Vec::new();
        let mut grad_tensors = Vec::new();

        for (idx, save) in saved.iter().enumerate() {
            let node_shape = &graph.nodes[idx].shape;
            let num_shape = build_dims(node_shape, meta);

            if save.is_defined_in_forward() {
                let tensor = self.new_tensor(&num_shape);
                forward_saved.push(tensor);
            }

            if save.is_defined_in_backward() {
                let tensor = self.new_tensor(&num_shape);
                grad_tensors.push(tensor);
            }
        }

        let meta_buffer = self.inner.alloc_meta(meta);

        let node_shape = &graph.nodes[graph.nodes.len() - 1].shape;
        let shape = build_dims(node_shape, meta);
        let forward_out = self.new_tensor(&shape);
        let loss_t = self.new_tensor(&shape);
        let seed = self.new_tensor(&shape);

        AllocTensors {
            meta: meta_buffer,
            forward_saved,
            grad_tensors,
            forward_out,
            loss_t,
            seed,
        }
    }

    pub fn new_tensor(&self, shape: &[u32]) -> Tensor<B> {
        let len = shape.iter().product::<u32>() as usize;
        let data = self.inner.alloc(len);
        Tensor {
            shape: Some(shape.to_vec()),
            data: GpuBuffer { inner: data },
        }
    }

    pub fn new_tensor_init(&self, shape: &[u32], data: &[f32]) -> Tensor<B> {
        debug_assert_eq!(
            shape.iter().product::<u32>(),
            data.len() as u32,
            "shape product (left) and data length (right) mismatch"
        );

        let data = gpu_alloc_init(&self.inner, data);
        Tensor {
            shape: Some(shape.to_vec()),
            data,
        }
    }

    /// Allocates an empty one-hot vector.
    pub fn new_onehot(&self, classes: u32) -> Tensor<B> {
        let data = gpu_alloc(&self.inner, classes as usize);
        Tensor { data, shape: None }
    }

    /// Allocates a new one-hot vector with the defined classes.
    pub fn new_onehot_init(&self, indices: &[u32]) -> Tensor<B> {
        let data = gpu_alloc_init(&self.inner, indices);
        Tensor { data, shape: None }
    }
}

#[derive(Debug, Clone)]
pub struct AllocTensors<B: GpuBackend = backend::GpuContext> {
    pub meta: B::Buffer,
    pub forward_saved: Vec<Tensor<B>>,
    pub grad_tensors: Vec<Tensor<B>>,
    pub forward_out: Tensor<B>,
    pub loss_t: Tensor<B>,

    pub seed: Tensor<B>,
}

#[cfg(test)]
mod tests {
    use crate::dispatch::{
        CompilationOptions, GpuContext,
        backend::{Graph, LossType, Metadata, kernel::Kernel},
    };

    #[test]
    fn mul_add_forward_backward() {
        let mut meta = Metadata::new();
        let len = meta.new_field();

        let mut graph = Graph::new(LossType::MeanSquaredError);
        let a = graph.input(&[len]);
        let b = graph.input(&[len]);
        let c = graph.input(&[len]);

        let x = graph.mul(a, b);
        graph.add(c, x);

        let saved = Kernel::compute_saved_nodes(&graph);
        let options = CompilationOptions::default();

        let ctx = pollster::block_on(GpuContext::new()).unwrap();
        graph.validate(meta).unwrap();
        graph.topo_sort().unwrap();
        graph.rebuild_outputs();
        let ir = graph.lower(meta, &options, &saved).unwrap();
        let kernels = ctx.compile(&ir, &options).unwrap();

        let in_tensors = [
            ctx.new_tensor_init(&[32, 32], &[3.0; 1024]),
            ctx.new_tensor_init(&[32, 32], &[2.0; 1024]),
            ctx.new_tensor_init(&[32, 32], &[1.0; 1024]),
        ];

        let meta_binding = [1024];
        assert!(meta.validate_meta(&meta_binding));

        let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding);

        ctx.launch_forward(&kernels, &meta_binding, &in_tensors, &saved_tensors);

        ctx.upload(&saved_tensors.seed, &[1_f32; 1024])
            .unwrap()
            .sync();

        ctx.launch_backward(&kernels, &meta_binding, &in_tensors, &saved_tensors);

        let mut dst = [0_f32; 1024];

        let out_tensor = &saved_tensors.forward_out;
        let grad_tensors = &saved_tensors.grad_tensors;

        ctx.download(&out_tensor, &mut dst).unwrap();
        assert!(dst.iter().all(|x| *x == 7.0));

        ctx.download(&grad_tensors[0], &mut dst).unwrap();
        assert!(dst.iter().all(|x| *x == 2.0));

        ctx.download(&grad_tensors[1], &mut dst).unwrap();
        assert!(dst.iter().all(|x| *x == 3.0));

        ctx.download(&grad_tensors[2], &mut dst).unwrap();
        assert!(dst.iter().all(|x| *x == 1.0));
    }

    #[test]
    fn matmul_div_softmax_forward_backward() {
        let mut meta = Metadata::new();
        let m = meta.new_field();
        let n = meta.new_field();
        let k = meta.new_field();

        let mut graph = Graph::new(LossType::CrossEntropy);
        let a = graph.input(&[m, k]);
        let b = graph.input(&[k, n]);

        let x = graph.matmul(a, b);
        let s = graph.softmax(x);
        graph.mul(x, s);

        let saved = Kernel::compute_saved_nodes(&graph);
        let options = CompilationOptions::default();

        let ctx = pollster::block_on(GpuContext::new()).unwrap();
        graph.validate(meta).unwrap();
        graph.topo_sort().unwrap();
        graph.rebuild_outputs();
        let ir = graph.lower(meta, &options, &saved).unwrap();
        let kernels = ctx.compile(&ir, &options).unwrap();

        let in_tensors = [
            ctx.new_tensor_init(&[16, 32], &[3.0; 512]),
            ctx.new_tensor_init(&[32, 64], &[2.0; 2048]),
        ];

        let meta_binding = [16, 64, 32];
        assert!(meta.validate_meta(&meta_binding));

        let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding);

        ctx.launch_forward(&kernels, &meta_binding, &in_tensors, &saved_tensors);

        let target = ctx.new_tensor_init(&[16, 64], &[1.0; 1024]);

        ctx.launch_loss(&kernels, &meta_binding, &target, &saved_tensors);

        ctx.launch_backward(&kernels, &meta_binding, &in_tensors, &saved_tensors);

        let mut dst = [0_f32; 1024];

        let out_tensor = &saved_tensors.forward_out;
        let grad_tensors = &saved_tensors.grad_tensors;

        ctx.download(&out_tensor, &mut dst).unwrap();
        assert!(dst.iter().all(|x| *x == 7.0));

        ctx.download(&saved_tensors.loss_t, &mut dst).unwrap();
        assert!(dst.iter().all(|x| *x == 0.0));

        ctx.download(&saved_tensors.seed, &mut dst).unwrap();
        assert!(dst.iter().all(|x| *x == 0.0));

        ctx.download(&grad_tensors[0], &mut dst).unwrap();
        assert!(dst.iter().all(|x| *x == 2.0));

        ctx.download(&grad_tensors[1], &mut dst).unwrap();
        assert!(dst.iter().all(|x| *x == 3.0));
    }

    #[test]
    fn matmul_add_forward_backward() {
        // works only when K<=N
        const M: u32 = 32;
        const N: u32 = 64;
        const K: u32 = 16;

        const A_VAL: f32 = 3.0;
        const B_VAL: f32 = 2.0;
        const C_VAL: f32 = 1.0;

        let mut meta = Metadata::new();
        let m = meta.new_field();
        let n = meta.new_field();
        let k = meta.new_field();

        let mut graph = Graph::new(LossType::MeanSquaredError);
        let a = graph.input(&[m, k]);
        let b = graph.input(&[k, n]);
        let c = graph.input(&[m, n]);

        let x = graph.matmul(a, b);
        graph.add(x, c);

        let saved = Kernel::compute_saved_nodes(&graph);
        let options = CompilationOptions::default();

        let ctx = pollster::block_on(GpuContext::new()).unwrap();
        graph.validate(meta).unwrap();
        graph.topo_sort().unwrap();
        graph.rebuild_outputs();
        let ir = graph.lower(meta, &options, &saved).unwrap();
        let kernels = ctx.compile(&ir, &options).unwrap();

        let in_tensors = [
            ctx.new_tensor_init(&[M, K], &[A_VAL; (M * K) as usize]),
            ctx.new_tensor_init(&[K, N], &[B_VAL; (K * N) as usize]),
            ctx.new_tensor_init(&[M, N], &[C_VAL; (M * N) as usize]),
        ];

        let meta_binding = [M, N, K];
        assert!(meta.validate_meta(&meta_binding));

        let saved_tensors = ctx.alloc_tensors(&graph, &saved, &meta_binding);

        let upload = ctx
            .upload(&saved_tensors.seed, &[1_f32; (M * N) as usize])
            .unwrap();

        ctx.launch_forward(&kernels, &meta_binding, &in_tensors, &saved_tensors);

        upload.sync();

        ctx.launch_backward(&kernels, &meta_binding, &in_tensors, &saved_tensors);

        let mut dst = alloc::vec![0_f32; (M * N).max(M * K).max(K * N) as usize];

        let out_tensor = &saved_tensors.forward_out;
        let grad_tensors = &saved_tensors.grad_tensors;

        ctx.download(&out_tensor, &mut dst).unwrap();
        let download = &dst[..(M * N) as usize];
        std::eprintln!("{:?}!={}", download[0], (A_VAL * B_VAL * K as f32) + C_VAL);
        assert!(
            download
                .iter()
                .all(|x| *x == (A_VAL * B_VAL * K as f32) + C_VAL)
        );

        ctx.download(&grad_tensors[0], &mut dst).unwrap();
        let download = &dst[..(M * K) as usize];
        std::eprintln!("{:?}!={}", download[0], B_VAL * N as f32);
        assert!(download.iter().all(|x| *x == B_VAL * N as f32));

        ctx.download(&grad_tensors[1], &mut dst).unwrap();
        let download = &dst[..(K * N) as usize];
        std::eprintln!("{:?}!={}", download[0], A_VAL * M as f32);
        assert!(download.iter().all(|x| *x == A_VAL * M as f32));

        ctx.download(&grad_tensors[2], &mut dst).unwrap();
        let download = &dst[..(M * N) as usize];
        assert!(download.iter().all(|x| *x == 1.0));
    }

    #[test]
    fn matmul_chain3_forward_backward() {
        const M: u32 = 32;
        const N: u32 = 64;
        const K: u32 = 128;
        const H: u32 = 256;

        const A_VAL: f32 = 3.0;
        const B_VAL: f32 = 2.0;
        const C_VAL: f32 = 1.0;
        const D_VAL: f32 = 0.5;
        const E_VAL: f32 = 1.0;
        const X_VAL: f32 = A_VAL * B_VAL * K as f32;
        const Y_VAL: f32 = C_VAL * X_VAL * M as f32;
        const Z_VAL: f32 = Y_VAL * D_VAL * N as f32;
        const Y_GRAD: f32 = D_VAL * H as f32;
        const X_GRAD: f32 = Y_GRAD * C_VAL * H as f32;
        const D_GRAD: f32 = Y_VAL * H as f32;
        const C_GRAD: f32 = Y_GRAD * X_VAL * N as f32;
        const B_GRAD: f32 = A_VAL * X_GRAD * M as f32;
        const A_GRAD: f32 = X_GRAD * B_VAL * N as f32;

        let mut meta = Metadata::new();
        let m = meta.new_field();
        let n = meta.new_field();
        let k = meta.new_field();
        let h = meta.new_field();

        let mut graph = Graph::new(LossType::MeanSquaredError);

        let a = graph.input(&[m, k]);
        let b = graph.input(&[k, n]);
        let c = graph.input(&[h, m]);
        let d = graph.input(&[n, h]);
        let e = graph.input(&[h, h]);

        let x = graph.matmul(a, b);
        let y = graph.matmul(c, x);
        let z = graph.matmul(y, d);
        graph.add(z, e);

        let saved = Kernel::compute_saved_nodes(&graph);
        let options = CompilationOptions::default();

        let ctx = pollster::block_on(GpuContext::new()).unwrap();
        graph.validate(meta).unwrap();
        graph.topo_sort().unwrap();
        graph.rebuild_outputs();
        let ir = graph.lower(meta, &options, &saved).unwrap();
        let kernels = ctx.compile(&ir, &options).unwrap();

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

        let upload = ctx
            .upload(&saved_tensors.seed, &[1_f32; (H * H) as usize])
            .unwrap();

        ctx.launch_forward(&kernels, &meta_binding, &in_tensors, &saved_tensors);

        upload.sync();

        ctx.launch_backward(&kernels, &meta_binding, &in_tensors, &saved_tensors);

        let max_len = in_tensors.iter().map(|x| x.len()).max().unwrap_or(0) as usize;
        let mut dst = alloc::vec![0.0; max_len];

        let out_tensor = &saved_tensors.forward_out;
        let grad_tensors = &saved_tensors.grad_tensors;
        let saved_tensors = &saved_tensors.forward_saved;

        ctx.download(&grad_tensors[3], &mut dst).unwrap();
        let download = &dst[..(N * H) as usize];
        std::eprintln!("dD={:?}... {:?}?", download[0], D_GRAD); // move to end later
        assert!(download.iter().all(|x| *x == D_GRAD));

        ctx.download(&saved_tensors[0], &mut dst).unwrap();
        let download = &dst[..(M * N) as usize];
        std::eprintln!("X={:?}... {:?}?", &download[0], X_VAL);
        assert!(download.iter().all(|x| *x == X_VAL));

        ctx.download(&saved_tensors[1], &mut dst).unwrap();
        let download = &dst[..(H * N) as usize];
        std::eprintln!("Y={:?}... {:?}?", &download[0], Y_VAL);
        assert!(download.iter().all(|x| *x == Y_VAL));

        ctx.download(&saved_tensors[2], &mut dst).unwrap();
        let download = &dst[..(H * H) as usize];
        std::eprintln!("Z={:?}... {:?}?", &download[0], Z_VAL);
        assert!(download.iter().all(|x| *x == Z_VAL));

        ctx.download(&out_tensor, &mut dst).unwrap();
        let download = &dst[..(H * H) as usize];
        std::eprintln!("forward={:?}... {:?}?", &download[0], Z_VAL + E_VAL);
        assert!(download.iter().all(|x| *x == Z_VAL + E_VAL));

        ctx.download(&grad_tensors[0], &mut dst).unwrap();
        let download = &dst[..(M * K) as usize];
        std::eprintln!("dA={:?}... {:?}?", download[0], A_GRAD);
        // assert!(download.iter().all(|x| *x == A_GRAD));

        ctx.download(&grad_tensors[1], &mut dst).unwrap();
        let download = &dst[..(K * N) as usize];
        std::eprintln!("dB={:?}... {:?}?", download[0], B_GRAD);
        // assert!(download.iter().all(|x| *x == B_GRAD));

        ctx.download(&grad_tensors[2], &mut dst).unwrap();
        let download = &dst[..(H * M) as usize];
        std::eprintln!("dC={:?}... {:?}?", download[0], C_GRAD);
        assert!(download.iter().all(|x| *x == C_GRAD));

        ctx.download(&grad_tensors[4], &mut dst).unwrap();
        let download = &dst[..(H * H) as usize];
        assert!(download.iter().all(|x| *x == 1.0));
    }
}
