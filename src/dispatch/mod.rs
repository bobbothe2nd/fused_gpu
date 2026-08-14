//! Full dispatch of math operations and GPU contVec;

use crate::{
    dispatch::backend::{
        Graph, MetaId, NodeId, Param,
        kernel::{Dependencies, RawKernel, Redirect, SaveIndicator},
    },
    errors::Error,
    tensor::{Tensor, build_dims},
};
use alloc::{vec, vec::Vec};
use briny::{
    raw::cast::{slice_to_bytes, slice_to_bytes_mut},
    traits::Pod,
};
use core::{fmt::Debug, marker::PhantomData};

pub mod backend;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum PollStatus {
    QueueEmpty,
    Failed,
    Pending,
    Ready,
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct CompilationOptions<B: GpuBackend> {
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

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct TargetCompilationOptions<B: GpuBackend> {
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

#[allow(clippy::derivable_impls)]
impl Default for DebugCompilationOptions {
    fn default() -> Self {
        Self {
            pretty_print_ir: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct OptCompilationOptions {
    /// Sets tile size or block size.
    pub tile_size: u32,

    /// Sets the amount of passes to optimize each kernel.
    ///
    /// The compiler will quickly run out of optimizations if this is set too high.
    pub passes: u8,

    /// Defines the set of optimizations the compiler will run each pass.
    pub flags: OptFlags,
}

impl Default for OptCompilationOptions {
    fn default() -> Self {
        Self {
            tile_size: 16,
            passes: 1,
            flags: OptFlags::all(),
        }
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub struct OptFlags: u8 {
        /// Enables or disables dead code elimination.
        ///
        /// `let useless = 1234;` -> [[nothing]]
        const DEAD_CODE = 1 << 0;

        /// Enables or disables constant folding.
        ///
        /// `2*2` -> `4`
        const CONST_FOLD = 1 << 1;

        /// Converts assignments to op-assignments.
        ///
        /// `a = b + a;` -> `a += b;`
        const OP_ASSIGN = 1 << 2;

        /// Applies the identity prperty to arithmetic.
        ///
        /// `x*1` or `x/1` or `x+0` or `x-0` -> `x`
        const IDENTITY = 1 << 3;

        /// Converts constant division statements to multiplication.
        ///
        /// `x/2` -> `x*0.5`
        const DIV_CONST = 1 << 4;

        /// Fuses multiply-add operations.
        ///
        /// `a*b+c` -> `fma(a,b,c)`
        const MUL_ADD = 1 << 5;
    }
}

pub trait GpuBufferBackend: Debug {
    fn size(&self) -> u32;

    fn size_bytes(&self) -> u32;
}

pub trait GpuKernelBackend {
    fn iteration_space(&self) -> &[MetaId];
    fn block(&self) -> &[u32; 3];
}

pub trait GpuBackend: Sized {
    /// The target-specific configuration for the compiler.
    const TARGET_SPEC: TargetCompilationOptions<Self>;

    type Buffer: GpuBufferBackend;
    type Kernel: GpuKernelBackend;
    type Batcher<'a>;
    type BatchState;
    type SubmissionIndex;
    type ParamLayout;
    type Schedule<'a>;
    type SyncSubmissions;

    /// Allocates an uninitialized GPU buffer with the size `len` in bytes.
    fn alloc(&self, len: usize) -> Self::Buffer;

    /// Allocates a slice of `u8` (to be reinterpreted as larger datatypes).
    fn alloc_init(&self, data: &[u8]) -> Self::Buffer;

    /// Allocates a slice of `u32` on the GPU.
    ///
    /// This allocation is often small and uniform.
    fn alloc_meta(&self, data: &[u32]) -> Self::Buffer;

    /// Copies the content of a CPU buffer to a GPU buffer.
    ///
    /// # Errors
    ///
    /// Should return an [`ErrorKind::FailedBufferCopy`](`crate::errors::ErrorKind::FailedBufferCopy`).
    fn upload(&self, buffer: &Self::Buffer, data: &[u8]) -> Result<Self::SubmissionIndex, Error>;

    /// Copies the content of one buffer to another.
    ///
    /// # Errors
    ///
    /// Should return an [`ErrorKind::FailedBufferCopy`](`crate::errors::ErrorKind::FailedBufferCopy`).
    fn pipe(&self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<Self::SubmissionIndex, Error>;

    /// Copies the content of a GPU buffer to a CPU buffer.
    ///
    /// # Errors
    ///
    /// Should return an [`ErrorKind::FailedBufferCopy`](`crate::errors::ErrorKind::FailedBufferCopy`).
    fn download(&self, buffer: &Self::Buffer, out: &mut [u8]) -> Result<(), Error>;

    /// Compiles the `RawKernel` (containing ops+vars) to a GPU kernel ([`Self::Kernel`]).
    ///
    /// # Errors
    ///
    /// For an invalid kernel, param layout, or compilation options and all related errors.
    fn compile(
        &self,
        src: &RawKernel,
        params: &[Param],
        options: &CompilationOptions<Self>,
    ) -> Result<Self::Kernel, Error>;

    fn schedule<'a>(
        &self,
        kernels: &'a [Dependencies<Redirect<(Self::Kernel, NodeId, &[bool])>>],
        bindings: &[&'a Self::Buffer],
        meta: &[u32],
    ) -> Result<Self::Schedule<'a>, Error>;

    fn dispatch_kernel(
        &self,
        batcher: &mut Self::Batcher<'_>,
        kernel: &Self::Kernel,
        wg: [u32; 3],
        bindings: &[&Self::Buffer],
    );

    fn dispatch_schedule(&self, batcher: &mut Self::Batcher<'_>, schedule: &Self::Schedule<'_>);

    fn sync(&self, submission_index: Self::SubmissionIndex);

    fn prepare_batch(&self) -> Self::BatchState;

    fn start_batch<'a>(&self, state: &'a mut Self::BatchState) -> Self::Batcher<'a>;

    fn encode(&self, state: Self::BatchState) -> Self::SyncSubmissions;

    fn submit(&self, submission: Self::SyncSubmissions) -> Self::SubmissionIndex;

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

/// A group of compiled kernels including all required metadata.
///
/// It includes the kernels of the forward, backward, and loss passes, the order
/// in which to execute them, and the parameters they use.
#[derive(Debug)]
pub struct KernelGroup<'a, B: GpuBackend = backend::GpuContext> {
    pub(crate) forward: Vec<Dependencies<Redirect<(B::Kernel, usize, &'a [bool])>>>,
    pub(crate) backward: Vec<Dependencies<Redirect<(B::Kernel, usize, &'a [bool])>>>,
    pub(crate) loss: B::Kernel,
}

/// Handle to a series of GPU submissions.
#[must_use]
pub struct SubmissionIndex<'a, B: GpuBackend = backend::GpuContext>(
    B::SubmissionIndex,
    &'a GpuContext<B>,
);

impl<B: GpuBackend> SubmissionIndex<'_, B> {
    pub fn sync(self) {
        self.1.inner.sync(self.0);
    }
}

/// Handle to a series of unsubmitted GPU kernels.
#[must_use]
pub struct SyncSubmission<'a, B: GpuBackend = backend::GpuContext>(
    B::SyncSubmissions,
    &'a GpuContext<B>,
);

impl<'a, B: GpuBackend> SyncSubmission<'a, B> {
    pub fn submit(self) -> SubmissionIndex<'a, B> {
        SubmissionIndex(self.1.inner.submit(self.0), self.1)
    }
}

#[must_use]
pub struct BatchState<'a, B: GpuBackend = backend::GpuContext>(B::BatchState, &'a GpuContext<B>);

impl<'a, B: GpuBackend> BatchState<'a, B> {
    pub fn encode(self) -> SyncSubmission<'a, B> {
        SyncSubmission(self.1.inner.encode(self.0), self.1)
    }
}

/// Schedule used to improve performance by caching critical launch information.
pub struct Schedule<'a, B: GpuBackend = backend::GpuContext> {
    forward: B::Schedule<'a>,
    backward: B::Schedule<'a>,
}

/// Generic GPU context storing a handle to the device and shaders.
///
/// Perhaps most important structure in all of Fused GPU this is. Without it,
/// you couldn't compile kernels, execute kernels, create tensors, copy data.
#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuContext<B: GpuBackend = backend::GpuContext> {
    pub(crate) inner: B,
}

impl GpuContext<backend::GpuContext> {
    /// Blocking creation of the context by searching for GPU.
    ///
    /// # Errors
    ///
    /// Failure is platform-specific and backend-dependent. It is likely a result of
    /// not finding a supported device. Errors must be handled properly in critical code.
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            inner: pollster::block_on(backend::GpuContext::new())?,
        })
    }

    /// Non-blocking creation of the context by searching for GPU.
    ///
    /// # Errors
    ///
    /// Failure is platform-specific and backend-dependent. It is likely a result of
    /// not finding a supported device. Errors must be handled properly in critical code.
    pub async fn new_async() -> Result<Self, Error> {
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
    ///
    /// # Errors
    ///
    /// Failure is platform-specific and backend-dependent but often a result of invalid input.
    /// Regardless, errors must be handled properly in critical code.
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
                let dep = kernel.dep.clone();

                let compiled = match &kernel.val {
                    Redirect::Unmasked(kernel) => {
                        let params = ir
                            .forward
                            .params
                            .iter()
                            .enumerate()
                            .filter_map(
                                |(i, param)| if kernel.params[i] { Some(*param) } else { None },
                            )
                            .collect::<Vec<_>>();

                        Redirect::Unmasked((
                            self.inner.compile(&kernel.raw, &params, options)?,
                            kernel.raw.root,
                            kernel.params.as_slice(),
                        ))
                    }
                    Redirect::Redirected(idx) => Redirect::Redirected(*idx),
                };

                Ok(Dependencies { val: compiled, dep })
            })
            .collect::<Result<
                Vec<Dependencies<Redirect<(<B as GpuBackend>::Kernel, usize, &'a [bool])>>>,
                Error,
            >>()?;

        let backward = ir
            .backward
            .kernels
            .iter()
            .map(|kernel| {
                let dep = kernel.dep.clone();

                let compiled = match &kernel.val {
                    Redirect::Unmasked(kernel) => {
                        let params = ir
                            .backward
                            .params
                            .iter()
                            .enumerate()
                            .filter_map(
                                |(i, param)| if kernel.params[i] { Some(*param) } else { None },
                            )
                            .collect::<Vec<_>>();

                        Redirect::Unmasked((
                            self.inner.compile(&kernel.raw, &params, options)?,
                            kernel.raw.root,
                            kernel.params.as_slice(),
                        ))
                    }
                    Redirect::Redirected(idx) => Redirect::Redirected(*idx),
                };

                Ok(Dependencies { val: compiled, dep })
            })
            .collect::<Result<
                Vec<Dependencies<Redirect<(<B as GpuBackend>::Kernel, usize, &'a [bool])>>>,
                Error,
            >>()?;

        let loss = self.inner.compile(&ir.loss.raw, &ir.loss.params, options)?;

        Ok(KernelGroup {
            forward,
            backward,
            loss,
        })
    }

    /// Copies the content of a GPU buffer into a CPU buffer.
    ///
    /// # Errors
    ///
    /// Failure is platform-specific and backend-dependent. It might only return an error
    /// if buffer lengths are unequal, but its behavior should not be assumed. Errors
    /// must be handled properly in critical code.
    pub fn download<T: Pod>(&self, tensor: &Tensor<B>, dst: &mut [T]) -> Result<(), Error> {
        self.inner
            .download(&tensor.data.inner, slice_to_bytes_mut(dst))
    }

    /// Copies the content of a CPU buffer into GPU buffer.
    ///
    /// # Errors
    ///
    /// Failure is platform-specific and backend-dependent. It might only return an error
    /// if buffer lengths are unequal, but its behavior should not be assumed. Errors
    /// must be handled properly in critical code.
    pub fn upload<T: Pod>(
        &self,
        tensor: &Tensor<B>,
        dst: &[T],
    ) -> Result<SubmissionIndex<'_, B>, Error> {
        self.inner
            .upload(&tensor.data.inner, slice_to_bytes(dst))
            .map(|x| SubmissionIndex(x, self))
    }

    /// Copies the content of one buffer to another without mutating the source.
    ///
    /// This is faster than chaining [`Self::download`] into [`Self::upload`] because it
    /// bypasses CPU memory.
    ///
    /// # Errors
    ///
    /// Failure is platform-specific and backend-dependent. It might only return an error
    /// if buffer lengths are unequal, but its behavior should not be assumed. Errors
    /// must be handled properly in critical code.
    pub fn pipe<T: Pod>(
        &self,
        src: &Tensor<B>,
        dst: &Tensor<B>,
    ) -> Result<SubmissionIndex<'_, B>, Error> {
        self.inner
            .pipe(&src.data.inner, &dst.data.inner)
            .map(|x| SubmissionIndex(x, self))
    }

    pub fn schedule<'a>(
        &self,
        kernels: &'a KernelGroup<B>,
        meta: &[u32],
        in_tensors: &'a [Tensor<B>],
        alloc_tensors: &'a AllocTensors<B>,
    ) -> Result<Schedule<'a, B>, Error> {
        let mut bindings = Vec::new();

        bindings.push(&alloc_tensors.meta);

        for t in &alloc_tensors.forward_saved {
            bindings.push(&t.data.inner);
        }

        for t in in_tensors {
            bindings.push(&t.data.inner);
        }

        bindings.push(&alloc_tensors.forward_out.data.inner);

        let forward = self.inner.schedule(&kernels.forward, &bindings, meta)?;

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

        let backward = self.inner.schedule(&kernels.backward, &bindings, meta)?;

        Ok(Schedule { forward, backward })
    }

    pub fn prepare_batch(&self) -> BatchState<'_, B> {
        BatchState(self.inner.prepare_batch(), self)
    }

    pub fn start_batch<'a>(&'a self, state: &'a mut BatchState<B>) -> Batcher<'a, B> {
        Batcher(self.inner.start_batch(&mut state.0), self)
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
            shape: shape.to_vec(),
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
            shape: shape.to_vec(),
            data,
        }
    }

    /// Allocates an empty one-hot vector.
    pub fn new_onehot(&self, classes: u32) -> Tensor<B> {
        let data = gpu_alloc(&self.inner, classes as usize);
        Tensor {
            data,
            shape: vec![classes],
        }
    }

    /// Allocates a new one-hot vector with the defined classes.
    pub fn new_onehot_init(&self, indices: &[u32]) -> Tensor<B> {
        let data = gpu_alloc_init(&self.inner, indices);
        Tensor {
            data,
            shape: vec![indices.len() as u32],
        }
    }
}

#[must_use]
pub struct Batcher<'a, B: GpuBackend = backend::GpuContext>(B::Batcher<'a>, &'a GpuContext<B>);

impl<B: GpuBackend> Batcher<'_, B> {
    pub fn dispatch_forward(&mut self, schedule: &Schedule<'_, B>) {
        self.1
            .inner
            .dispatch_schedule(&mut self.0, &schedule.forward);
    }

    pub fn dispatch_backward(&mut self, schedule: &Schedule<'_, B>) {
        self.1
            .inner
            .dispatch_schedule(&mut self.0, &schedule.backward);
    }

    pub fn launch_loss(
        &mut self,
        kernels: &KernelGroup<B>,
        target: &Tensor<B>,
        tensors: &AllocTensors<B>,
    ) {
        let grid = tensors.loss_t.calc_grid(*kernels.loss.block());

        let bindings = [
            &tensors.meta,
            &tensors.loss_t.data.inner,
            &tensors.seed.data.inner,
            &tensors.forward_out.data.inner,
            &target.data.inner,
        ];

        self.1
            .inner
            .dispatch_kernel(&mut self.0, &kernels.loss, grid, &bindings);
    }
}

#[derive(Debug)]
pub struct AllocTensors<B: GpuBackend = backend::GpuContext> {
    pub meta: B::Buffer,
    pub forward_saved: Vec<Tensor<B>>,
    pub grad_tensors: Vec<Tensor<B>>,
    pub forward_out: Tensor<B>,
    pub loss_t: Tensor<B>,

    pub seed: Tensor<B>,
}
