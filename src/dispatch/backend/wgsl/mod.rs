use crate::{
    dispatch::{
        CompilationOptions, GpuBackend, GpuBufferBackend, GpuKernelBackend, PollStatus,
        TargetCompilationOptions,
        backend::{Axis, DType, MetaId, Op, Param, ParamTy, ValueId, ValueState, kernel::Kernel},
    },
    errors::{Error, ErrorKind},
};
use alloc::{string::String, vec::Vec};
use briny::raw::cast_slice;
use core::fmt::Write;
use wgpu::{
    BackendOptions, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, DeviceDescriptor, Dx12BackendOptions, Dx12Compiler, Dx12SwapchainKind,
    Dx12UseFrameLatencyWaitableObject, ExperimentalFeatures, Features, ForceShaderModelToken,
    GlBackendOptions, GlDebugFns, GlFenceBehavior, Gles3MinorVersion, Instance, InstanceDescriptor,
    InstanceFlags, Limits, MemoryBudgetThresholds, MemoryHints, NoopBackendOptions,
    PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, PollType,
    PowerPreference, Queue, RequestAdapterError, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, SubmissionIndex, Trace,
    util::{BufferInitDescriptor, DeviceExt},
};

/// WGPU context for device and queue.
#[derive(Debug, Clone)]
pub struct GpuContext {
    device: Device,
    queue: Queue,
}

impl GpuContext {
    /// Constructs a new context asynchronously.
    pub async fn new() -> Result<Self, crate::errors::Error> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            flags: InstanceFlags::empty(),
            memory_budget_thresholds: MemoryBudgetThresholds::default(),
            backend_options: BackendOptions {
                gl: GlBackendOptions {
                    gles_minor_version: Gles3MinorVersion::Automatic,
                    fence_behavior: GlFenceBehavior::AutoFinish,
                    debug_fns: GlDebugFns::Disabled,
                },
                dx12: Dx12BackendOptions {
                    shader_compiler: Dx12Compiler::Auto,
                    presentation_system: Dx12SwapchainKind::default(),
                    latency_waitable_object: Dx12UseFrameLatencyWaitableObject::None,
                    force_shader_model: ForceShaderModelToken::default(),
                    agility_sdk: None,
                },
                noop: NoopBackendOptions::default(),
            },
            display: None,
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| match e {
                RequestAdapterError::EnvNotSet => Error {
                    msg: "adapter environment variables not set",
                    kind: ErrorKind::EnvNotSet,
                    ctx: (),
                },
                _ => Error {
                    msg: "no adapter found",
                    kind: ErrorKind::AdapterNotFound,
                    ctx: (),
                },
            })?;

        let adapter_limits = adapter.limits();

        let (device, queue) = match adapter
            .request_device(&DeviceDescriptor {
                label: Some("device"),
                required_limits: adapter_limits,
                required_features: Features::empty(),
                experimental_features: ExperimentalFeatures::disabled(),
                memory_hints: MemoryHints::Performance,
                trace: Trace::Off,
            })
            .await
        {
            Ok(dev) => dev,
            Err(_) => adapter
                .request_device(&DeviceDescriptor {
                    label: Some("device"),
                    required_limits: Limits::defaults(),
                    required_features: Features::empty(),
                    experimental_features: ExperimentalFeatures::disabled(),
                    memory_hints: MemoryHints::Performance,
                    trace: Trace::Off,
                })
                .await
                .map_err(|_| Error {
                    msg: "failed to request device",
                    kind: ErrorKind::DeviceNotFound,
                    ctx: (),
                })?,
        };

        Ok(Self { device, queue })
    }
}

#[derive(Debug)]
pub struct GpuKernel {
    kernel: ComputePipeline,
    iter_space: Vec<MetaId>,
    block: [u32; 3],
}

impl GpuKernelBackend for GpuKernel {
    fn iteration_space(&self) -> &[MetaId] {
        &self.iter_space
    }

    fn block(&self) -> &[u32; 3] {
        &self.block
    }
}

impl GpuBackend for GpuContext {
    const TARGET_SPEC: TargetCompilationOptions<Self> =
        TargetCompilationOptions::new(false, false, false);

    type Buffer = GpuBuffer;
    type Kernel = GpuKernel;
    type SubmissionIndex = SubmissionIndex;
    type ParamLayout = PipelineLayout;

    #[inline]
    fn alloc(&self, len: usize) -> Self::Buffer {
        GpuBuffer(self.device.create_buffer(&BufferDescriptor {
            label: Some("gpu_tensor"),
            size: (len * core::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    #[inline]
    fn alloc_init(&self, data: &[u8]) -> Self::Buffer {
        GpuBuffer(self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("gpu_tensor"),
            contents: cast_slice(data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        }))
    }

    #[inline]
    fn alloc_meta(&self, data: &[u32]) -> Self::Buffer {
        let offset = (4 - (data.len() % 4)) % 4;
        let new_len = data.len() + offset;
        let mut aligned = alloc::vec![u32::MAX; new_len];
        aligned[..data.len()].copy_from_slice(data);

        GpuBuffer(self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("gpu_meta"),
            contents: cast_slice(&aligned),
            usage: BufferUsages::UNIFORM,
        }))
    }

    #[inline]
    fn upload(&self, buffer: &Self::Buffer, data: &[u8]) -> Result<Self::SubmissionIndex, Error> {
        if buffer.size_bytes() as usize != data.len() {
            return Err(Error {
                msg: "CPU and GPU buffers of unequal sizes during upload",
                kind: ErrorKind::FailedDownload,
                ctx: (),
            });
        }

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        let src = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("gpu_tensor"),
            contents: cast_slice(data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });
        encoder.copy_buffer_to_buffer(&src, 0, &buffer.0, 0, buffer.0.size());

        Ok(self.queue.submit(Some(encoder.finish())))
    }

    #[inline]
    fn download(&self, buffer: &Self::Buffer, data: &mut [u8]) -> Result<(), Error> {
        if buffer.size_bytes() as usize > data.len() {
            return Err(Error {
                msg: "insufficient CPU memory allocated for GPU download",
                kind: ErrorKind::FailedDownload,
                ctx: (),
            });
        }

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        let dst = self.device.create_buffer(&BufferDescriptor {
            label: Some("download"),
            size: buffer.0.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buffer.0, 0, &dst, 0, buffer.0.size());
        let submission_index = self.queue.submit(Some(encoder.finish()));
        let buffer_slice = dst.slice(..);

        let (send, recv) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            if res.is_ok() {
                let _ = send.send(());
            }
        });

        let _ = self.device.poll(PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        });

        let _ = recv.recv();

        let len = data.len().min(buffer.size_bytes() as usize);

        data[..len].copy_from_slice(
            &buffer_slice.get_mapped_range().map_err(|_| Error {
                msg: "failed to map GPU memory to CPU",
                kind: ErrorKind::FailedDownload,
                ctx: (),
            })?[..len],
        );
        dst.unmap();

        Ok(())
    }

    #[inline]
    fn init_params(
        &self,
        src: &[Param],
        _options: &CompilationOptions<Self>,
    ) -> Result<Self::ParamLayout, Error> {
        let entries = generate_layout_desc(src);
        let desc = BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &entries,
        };

        let bind_group_layout = self.device.create_bind_group_layout(&desc);
        let bind_group_layouts = &[Some(&bind_group_layout)];

        let desc = PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts,
            immediate_size: 0,
        };
        let pipeline_layout = self.device.create_pipeline_layout(&desc);

        Ok(pipeline_layout)
    }

    #[inline]
    fn compile(
        &self,
        src: &Kernel,
        pipeline_layout: &Self::ParamLayout,
        options: &CompilationOptions<Self>,
    ) -> Result<Self::Kernel, Error> {
        let source = generate_wgsl(src, options.debug.pretty_print_ir);

        std::eprintln!("root {} = {source:?}", src.root);

        let shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("shader"),
            source: ShaderSource::Wgsl(source.into()),
        });

        Ok(GpuKernel {
            kernel: self
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("pipeline"),
                    layout: Some(pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                }),
            iter_space: src.iter_space.clone(),
            block: src.block,
        })
    }

    #[inline]
    fn launch(
        &self,
        kernel: &Self::Kernel,
        wg: [u32; 3],
        bindings: &[&Self::Buffer],
    ) -> Self::SubmissionIndex {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        {
            let kernel = &kernel.kernel;

            let entries = bindings
                .iter()
                .enumerate()
                .map(|(i, x)| BindGroupEntry {
                    binding: i as u32,
                    resource: x.0.as_entire_binding(),
                })
                .collect::<Vec<_>>();

            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                layout: &kernel.get_bind_group_layout(0),
                entries: &entries,
                label: Some("bind_group"),
            });

            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(kernel);
            pass.set_bind_group(0, &bind_group, &[]);

            let dispatch_x = wg[0];
            let dispatch_y = wg[1];
            let dispatch_z = wg[2];

            pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
        }

        self.queue.submit([encoder.finish()])
    }

    #[inline]
    fn sync(&self, submission_index: Self::SubmissionIndex) {
        let _ = self.device.poll(PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        });
    }

    #[inline]
    fn poll(&self) -> PollStatus {
        let res = self.device.poll(PollType::Poll);

        match res {
            Err(_) => PollStatus::Failed,
            Ok(wgpu::PollStatus::Poll) => PollStatus::Pending,
            Ok(wgpu::PollStatus::WaitSucceeded) => PollStatus::Ready,
            Ok(wgpu::PollStatus::QueueEmpty) => PollStatus::QueueEmpty,
        }
    }
}

/// Handle to a GPU allocation.
#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct GpuBuffer(Buffer);

impl GpuBufferBackend for GpuBuffer {
    fn size_bytes(&self) -> u32 {
        self.0.size() as u32
    }

    fn size(&self) -> u32 {
        (self.0.size() / (size_of::<f32>() as u64)) as u32
    }
}

#[inline]
fn generate_layout_desc(params: &[Param]) -> Vec<BindGroupLayoutEntry> {
    params
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let ty = match x.ty {
                ParamTy::Uniform => BufferBindingType::Uniform,
                ParamTy::ReadOnly => BufferBindingType::Storage { read_only: true },
                ParamTy::ReadWrite => BufferBindingType::Storage { read_only: false },
            };

            BindGroupLayoutEntry {
                binding: i as u32,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        })
        .collect::<Vec<_>>()
}

#[inline]
fn generate_wgsl(kernel: &Kernel, pretty_print: bool) -> String {
    let mut out = String::new();

    emit_bindings(kernel, &mut out, pretty_print);
    newline(pretty_print, &mut out, 0);

    emit_entry(kernel, &mut out, pretty_print);

    out
}

#[inline]
const fn get_axis(axis: Axis) -> &'static str {
    match axis {
        Axis::X => "x",
        Axis::Y => "y",
        Axis::Z => "z",
    }
}

#[inline]
const fn get_dtype(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::Float => "f32",
        DType::SignedInt => "i32",
        DType::UnsignedInt => "u32",
    }
}

#[inline]
fn newline(pretty_print: bool, out: &mut String, nesting: usize) {
    if pretty_print {
        let _ = write!(out, "\n{}", "  ".repeat(nesting));
    } else {
        let _ = write!(out, " ");
    }
}

#[inline]
fn tab(pretty_print: bool, out: &mut String) {
    if pretty_print {
        let _ = write!(out, "  ");
    }
}

#[inline]
fn emit_bindings(kernel: &Kernel, out: &mut String, pretty_print: bool) {
    let _ = write!(out, "struct Meta {{");
    for f in 0..kernel.meta.fields {
        newline(pretty_print, out, 1);
        let _ = write!(out, "f{f}: u32,");
    }
    newline(pretty_print, out, 0);
    let _ = write!(out, "}}");

    newline(pretty_print, out, 0);

    for (i, p) in kernel.params.iter().enumerate() {
        let var_type = match p.ty {
            ParamTy::ReadOnly => "var<storage, read>",
            ParamTy::ReadWrite => "var<storage, read_write>",
            ParamTy::Uniform => "var<uniform>",
        };

        newline(pretty_print, out, 0);
        if p.ty == ParamTy::Uniform && p.dtype == DType::UnsignedInt {
            let _ = write!(out, "@group(0) @binding({i}) {var_type} param{i}: Meta;");
        } else {
            let _ = write!(
                out,
                "@group(0) @binding({i}) {var_type} param{i}: array<{}>;",
                get_dtype(p.dtype)
            );
        }
    }

    newline(pretty_print, out, 0);

    for (i, s) in kernel.shared.iter().enumerate() {
        newline(pretty_print, out, 0);
        let _ = write!(
            out,
            "var<workgroup> shared{i}: array<{}, ({})>;",
            get_dtype(s.dtype),
            s.size
        );
    }
}

#[inline]
fn emit_entry(kernel: &Kernel, out: &mut String, pretty_print: bool) {
    newline(pretty_print, out, 0);
    let _ = write!(
        out,
        "@compute @workgroup_size({}, ({}), ({})) ",
        kernel.block[0], kernel.block[1], kernel.block[2]
    );

    newline(pretty_print, out, 0);
    let _ = write!(out, "fn main(");

    newline(pretty_print, out, 1);
    let _ = write!(out, "@builtin(local_invocation_id) lid: vec3<u32>,");

    newline(pretty_print, out, 1);
    let _ = write!(out, "@builtin(workgroup_id) bid: vec3<u32>,");

    newline(pretty_print, out, 1);
    let _ = write!(out, "@builtin(global_invocation_id) gid: vec3<u32>,");

    newline(pretty_print, out, 0);
    let _ = write!(out, ") {{");

    emit_ops(kernel, out, pretty_print);
    newline(pretty_print, out, 0);

    let _ = write!(out, "}}");
}

#[inline]
fn emit_ops(kernel: &Kernel, out: &mut String, pretty_print: bool) {
    let mut nesting = 0;

    for op in &kernel.ops {
        newline(pretty_print, out, nesting);

        if *op != Op::EndScope {
            tab(pretty_print, out);
        }

        process_op(out, op, &mut nesting, kernel);
    }
}

fn process_op(out: &mut String, op: &Op, nesting: &mut usize, kernel: &Kernel) {
    match op {
        Op::DefineVar { id } => {
            let val = &kernel.values[*id];
            match val.state {
                ValueState::Masked | ValueState::Inline => {}
                ValueState::Const => {
                    let _ = write!(out, "const v{id}: {}", get_dtype(val.dtype));

                    if let Some(op) = &val.init {
                        let _ = out.write_str(" = ");
                        process_op(out, op, nesting, kernel);
                    }

                    let _ = out.write_char(';');
                }
                var => {
                    if var == ValueState::Immut {
                        let _ = write!(out, "let v{id}: {}", get_dtype(val.dtype));
                    } else {
                        let _ = write!(out, "var v{id}: {}", get_dtype(val.dtype));
                    }

                    if let Some(op) = &val.init {
                        let _ = out.write_str(" = ");
                        process_op(out, op, nesting, kernel);
                    }

                    let _ = out.write_char(';');
                }
            }
        }

        Op::OverwriteVar { id, val } => {
            let _ = write!(out, "v{id} = ");
            process_op(out, val, nesting, kernel);
            let _ = write!(out, ";");
        }

        Op::AccumVar { id, val } => {
            let _ = write!(out, "v{id} += ");
            process_op(out, val, nesting, kernel);
            let _ = write!(out, ";");
        }

        Op::CopyVar { id } => {
            let _ = out.write_str(&render_val(*id, kernel));
        }

        Op::ConstF32 { value } => {
            let _ = write!(out, "{value}f");
        }

        Op::ConstU32 { value } => {
            let _ = write!(out, "{value}u");
        }

        Op::ConstI32 { value } => {
            let _ = write!(out, "{value}");
        }

        Op::ReadMeta { param, field } => {
            let _ = write!(out, "param{param}.f{field}");
        }

        Op::Lt { a, b } => {
            let _ = write!(
                out,
                "({}) < ({})",
                render_val(*a, kernel),
                render_val(*b, kernel)
            );
        }

        Op::Gt { a, b } => {
            let _ = write!(
                out,
                "({}) > ({})",
                render_val(*a, kernel),
                render_val(*b, kernel)
            );
        }

        Op::Le { a, b } => {
            let _ = write!(
                out,
                "({}) <= ({})",
                render_val(*a, kernel),
                render_val(*b, kernel)
            );
        }

        Op::Ge { a, b } => {
            let _ = write!(
                out,
                "({}) >= ({})",
                render_val(*a, kernel),
                render_val(*b, kernel)
            );
        }

        Op::LocalId { axis } => {
            let _ = write!(out, "lid.{}", get_axis(*axis));
        }

        Op::BlockId { axis } => {
            let _ = write!(out, "bid.{}", get_axis(*axis));
        }

        Op::GlobalId { axis } => {
            let _ = write!(out, "gid.{}", get_axis(*axis));
        }

        Op::Add { a, b } => {
            let _ = write!(
                out,
                "({}) + ({})",
                render_val(*a, kernel),
                render_val(*b, kernel)
            );
        }

        Op::Sub { a, b } => {
            let _ = write!(
                out,
                "({}) - ({})",
                render_val(*a, kernel),
                render_val(*b, kernel)
            );
        }

        Op::Mul { a, b } => {
            let _ = write!(
                out,
                "({}) * ({})",
                render_val(*a, kernel),
                render_val(*b, kernel)
            );
        }

        Op::Div { a, b } => {
            let _ = write!(
                out,
                "({}) / ({})",
                render_val(*a, kernel),
                render_val(*b, kernel)
            );
        }

        Op::Mod { a, b } => {
            let _ = write!(
                out,
                "({}) % ({})",
                render_val(*a, kernel),
                render_val(*b, kernel)
            );
        }

        Op::Pow { a, b } => {
            let _ = write!(
                out,
                "pow({}, {})",
                render_val(*a, kernel),
                render_val(*b, kernel)
            );
        }

        Op::Fma { a, b, c } => {
            let _ = write!(
                out,
                "fma({}, {}, {})",
                render_val(*a, kernel),
                render_val(*b, kernel),
                render_val(*c, kernel)
            );
        }

        Op::Max { a, b } => {
            let _ = write!(
                out,
                "max({}, {})",
                render_val(*a, kernel),
                render_val(*b, kernel)
            );
        }

        Op::Min { a, b } => {
            let _ = write!(
                out,
                "min({}, {})",
                render_val(*a, kernel),
                render_val(*b, kernel)
            );
        }

        Op::Select { cond, a, b } => {
            let _ = write!(
                out,
                "select({}, {}, {})",
                render_val(*cond, kernel),
                render_val(*a, kernel),
                render_val(*b, kernel),
            );
        }

        Op::Exp { x } => {
            let _ = write!(out, "exp({})", render_val(*x, kernel));
        }

        Op::Abs { x } => {
            let _ = write!(out, "abs({})", render_val(*x, kernel));
        }

        Op::Neg { x } => {
            let _ = write!(out, "-({})", render_val(*x, kernel));
        }

        Op::Log { x } => {
            let _ = write!(out, "log({})", render_val(*x, kernel));
        }

        Op::Tanh { x } => {
            let _ = write!(out, "tanh({})", render_val(*x, kernel));
        }

        Op::Sqrt { x } => {
            let _ = write!(out, "sqrt({})", render_val(*x, kernel));
        }

        Op::Load { param, index } => {
            let _ = write!(out, "param{param}[{}]", render_val(*index, kernel));
        }

        Op::Store {
            param,
            index,
            value,
        } => {
            let _ = write!(
                out,
                "param{param}[{}] = {};",
                render_val(*index, kernel),
                render_val(*value, kernel)
            );
        }

        Op::SharedLoad { mem, index } => {
            let _ = write!(out, "shared{mem}[{}]", render_val(*index, kernel));
        }

        Op::SharedStore { mem, index, value } => {
            let _ = write!(
                out,
                "shared{mem}[{}] = {};",
                render_val(*index, kernel),
                render_val(*value, kernel)
            );
        }

        Op::ForLoopBegin { index, end, step } => {
            let _ = write!(
                out,
                "for (; v{index} < {}; v{index} += {}) {{",
                render_val(*end, kernel),
                render_val(*step, kernel),
            );
            *nesting += 1;
        }

        Op::IfBegin { cond } => {
            let _ = write!(out, "if ({}) {{", render_val(*cond, kernel));
            *nesting += 1;
        }

        Op::ElseBegin => {
            let _ = write!(out, "else {{");
            *nesting += 1;
        }

        Op::EndScope => {
            let _ = write!(out, "}}");
            *nesting -= 1;
        }

        Op::Barrier => {
            let _ = write!(out, "workgroupBarrier();");
        }

        Op::Return => {
            let _ = write!(out, "return;");
        }

        Op::Continue => {
            let _ = write!(out, "continue;");
        }

        Op::Break => {
            let _ = write!(out, "break;");
        }
    }
}

fn render_val(id: ValueId, kernel: &Kernel) -> String {
    let mut out = String::new();
    let val = &kernel.values[id];
    match val.state {
        ValueState::Inline => {
            if let Some(op) = &val.init {
                process_op(&mut out, op, &mut 0, kernel);
            }
        }
        ValueState::Masked => {}
        _ => {
            let _ = write!(out, "v{id}");
        }
    }
    out
}
