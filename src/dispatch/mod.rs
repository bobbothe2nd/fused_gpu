//! Full dispatch of math operations and GPU contVec;

use alloc::vec::Vec;
use core::{fmt::Debug, marker::PhantomData};

use crate::{
    dispatch::backend::{
        Graph, MetaId, NodeId, Param, kernel::{Dependencies, RawKernel, SaveIndicator},
    }, errors::Error, tensor::{Tensor, build_dims},
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

pub trait GpuBackend: Sized {
    const TARGET_SPEC: TargetCompilationOptions<Self>;

    type Buffer: GpuBufferBackend;
    type Kernel: GpuKernelBackend;
    type SubmissionIndex;
    type ParamLayout;
    type Schedule<'a>;
    type SyncSubmissions;

    fn alloc(&self, len: usize) -> Self::Buffer;

    fn alloc_init(&self, data: &[u8]) -> Self::Buffer;

    fn alloc_meta(&self, data: &[u32]) -> Self::Buffer;

    fn upload(&self, buffer: &Self::Buffer, data: &[u8]) -> Result<Self::SubmissionIndex, Error>;

    fn download(&self, buffer: &Self::Buffer, out: &mut [u8]) -> Result<(), Error>;

    fn compile(
        &self,
        src: &RawKernel,
        params: &[Param],
        options: &CompilationOptions<Self>,
    ) -> Result<Self::Kernel, Error>;

    fn schedule<'a>(
        &self,
        kernels: &'a [(Dependencies<Self::Kernel>, NodeId, &[bool])],
        bindings: &[&'a Self::Buffer],
        meta: &[u32],
    ) -> Self::Schedule<'a>;

    fn launch_schedule(
        &self,
        schedule: &Self::Schedule<'_>,
    );

    fn launch_kernel(
        &self,
        kernel: &Self::Kernel,
        wg: [u32; 3],
        bindings: &[&Self::Buffer],
    ) -> Self::SyncSubmissions;

    fn sync(&self, submission_index: Self::SubmissionIndex);

    fn submit(&self, submission: Self::SyncSubmissions) -> Self::SubmissionIndex;

    fn poll(&self) -> PollStatus;
}

pub trait Pipeline<B: GpuBackend> {
    fn submit(&self) -> B::SubmissionIndex;
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
pub struct KernelGroup<'a, B: GpuBackend = backend::GpuContext> {
    pub(crate) forward: Vec<(Dependencies<B::Kernel>, usize, &'a [bool])>,
    pub(crate) backward: Vec<(Dependencies<B::Kernel>, usize, &'a [bool])>,
    pub(crate) loss: B::Kernel,
}

#[must_use]
pub struct SubmissionIndex<'a, B: GpuBackend>(B::SubmissionIndex, &'a GpuContext<B>);

impl<B: GpuBackend> SubmissionIndex<'_, B> {
    pub fn sync(self) {
        self.1.inner.sync(self.0);
    }
}

#[must_use]
pub struct SyncSubmission<'a, B: GpuBackend>(B::SyncSubmissions, &'a GpuContext<B>);

impl<'a, B: GpuBackend> SyncSubmission<'a, B> {
    pub fn submit(self) -> SubmissionIndex<'a, B> {
        SubmissionIndex(self.1.inner.submit(self.0), self.1)
    }
}

pub struct Schedule<'a, B: GpuBackend = backend::GpuContext> {
    forward: B::Schedule<'a>,
    backward: B::Schedule<'a>,
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
    pub fn compile<'a>(
        &self,
        ir: &'a backend::kernel::KernelGroup,
        options: &CompilationOptions<B>,
    ) -> Result<KernelGroup<'a, B>, Error> {
        let forward = ir
            .forward
            .kernels
            .iter()
            .map(|kernel| {
                let params = ir.forward.params.iter().enumerate().filter_map(|(i, param)| if kernel.val.params[i] {
                    Some(*param)
                } else {
                    None
                }).collect::<Vec<_>>();

                Ok((
                    Dependencies {
                        val: self.inner.compile(&kernel.val.raw, &params, options)?,
                        dep: kernel.dep.clone(),
                    },
                    kernel.val.raw.root,
                    kernel.val.params.as_slice(),
                ))
            })
            .collect::<Result<Vec<(Dependencies<<B as GpuBackend>::Kernel>, usize, &'a [bool])>, Error>>()?;

        let backward = ir
            .backward
            .kernels
            .iter()
            .map(|kernel| {
                let params = ir.backward.params.iter().enumerate().filter_map(|(i, param)| if kernel.val.params[i] {
                    Some(*param)
                } else {
                    None
                }).collect::<Vec<_>>();

                Ok((
                    Dependencies {
                        val: self.inner.compile(&kernel.val.raw, &params, options)?,
                        dep: kernel.dep.clone(),
                    },
                    kernel.val.raw.root,
                    kernel.val.params.as_slice(),
                ))
            })
            .collect::<Result<Vec<(Dependencies<<B as GpuBackend>::Kernel>, usize, &'a [bool])>, Error>>()?;

        let loss = self.inner.compile(&ir.loss.raw, &ir.loss.params, options)?;

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

    pub fn schedule<'a>(
        &self,
        kernels: &'a KernelGroup<B>,
        meta: &[u32],
        in_tensors: &'a [Tensor<B>],
        alloc_tensors: &'a AllocTensors<B>,
    ) -> Schedule<'a, B> {
        let mut bindings = Vec::new();

        bindings.push(&alloc_tensors.meta);

        for t in &alloc_tensors.forward_saved {
            bindings.push(&t.data.inner);
        }

        for t in in_tensors {
            bindings.push(&t.data.inner);
        }

        bindings.push(&alloc_tensors.forward_out.data.inner);

        let forward = self.inner.schedule(&kernels.forward, &bindings, meta);

        bindings.truncate(1);

        bindings.push(&alloc_tensors.seed.data.inner);

        for t in &alloc_tensors.grad_tensors {
            bindings.push(&t.data.inner);
        }

        for t in in_tensors {
            bindings.push(&t.data.inner);
        }

        for t in &alloc_tensors.forward_saved {
            bindings.push(&t.data.inner);
        }

        let backward = self.inner.schedule(&kernels.backward, &bindings, meta);

        Schedule {
            forward,
            backward,
        }
    }

    /// Launches a forward kernel from the compiled kernels with metadata and tensors.
    ///
    /// This function relies on the assumption that tensors matches the graph inputs and
    /// the metadata matches the graph metadata when compiling.
    pub fn launch_forward(
        &self,
        schedule: &Schedule<'_, B>,
    ) {
        self.inner.launch_schedule(&schedule.forward);
    }

    /// Launches a backward kernel from the compiled kernels with metadata and tensors.
    ///
    /// This function relies on the assumption that tensors matches the graph inputs and
    /// the metadata matches the graph metadata when compiling.
    pub fn launch_backward(
        &self,
        schedule: &Schedule<'_, B>,
    ) {
        self.inner.launch_schedule(&schedule.backward);
    }

    pub fn launch_loss(
        &self,
        kernels: &KernelGroup<B>,
        target: &Tensor<B>,
        tensors: &AllocTensors<B>,
    ) -> SyncSubmission<'_, B> {
        let grid = tensors.loss_t.calc_grid(*kernels.loss.block());

        let bindings = [
            &tensors.meta,
            &tensors.loss_t.data.inner,
            &tensors.seed.data.inner,
            &tensors.forward_out.data.inner,
            &target.data.inner,
        ];

        SyncSubmission(self.inner.launch_kernel(&kernels.loss, grid, &bindings), self)
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
