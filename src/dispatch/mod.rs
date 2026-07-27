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
            debug: DebugCompilationOptions::default(),
            opt: OptCompilationOptions::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TargetCompilationOptions<B: GpuBackend + ?Sized> {
    /// Support for linear algebra accelerator (e.g. tensor cores)?
    pub lin_acc: bool,

    /// Support for asyncronous memory loads/reads?
    pub async_mem_load: bool,

    /// Support for asyncronous memory stores/writes?
    pub async_mem_store: bool,

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

impl Default for DebugCompilationOptions {
    fn default() -> Self {
        Self {
            pretty_print_ir: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct OptCompilationOptions {
    pub tile_size: u32,
    pub opt_passes: u8,
    pub opt_level: u8,
}

impl Default for OptCompilationOptions {
    fn default() -> Self {
        Self {
            tile_size: 16,
            opt_passes: 0,
            opt_level: 0,
        }
    }
}

pub trait GpuBufferBackend: Debug + Clone {
    fn size(&self) -> u32;

    fn size_bytes(&self) -> u32;
}

pub trait GpuKernelBackend {
    fn iteration_space(&self) -> &[MetaId];
    fn block(&self) -> &[u32; 3];
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
            for (kernel, idx) in &kernels.forward {
                if resolved.contains(idx) {
                    continue;
                }

                if kernel.dep.iter().all(|x| resolved.contains(x)) {
                    tmp_res.push(*idx);

                    let iter_space = build_dims(kernel.val.iteration_space(), meta);
                    let grid = calc_grid(&iter_space, *kernel.val.block());

                    self.inner.launch(&kernel.val, grid, &bindings);
                }
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
            for (kernel, idx) in &kernels.backward {
                if resolved.contains(idx) {
                    continue;
                }

                if kernel.dep.iter().all(|x| resolved.contains(x)) {
                    tmp_res.push(*idx);

                    let iter_space = build_dims(kernel.val.iteration_space(), meta);
                    let grid = calc_grid(&iter_space, *kernel.val.block());

                    self.inner.launch(&kernel.val, grid, &bindings);
                }
            }

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
        let grid = tensors.loss_t.calc_grid(*kernels.loss.block());

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
