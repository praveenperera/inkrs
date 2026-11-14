//! Thin Metal layer: one device and command queue, shared-memory buffers, the MSL kernel library compiled at load,
//! and command batching
//!
//! Every dispatch goes into one serial compute encoder, so kernels run in the order they were encoded and each sees
//! the writes of the ones before it, like a CUDA stream. The open command buffer is committed every
//! [`FLUSH_EVERY`] dispatches so the GPU starts while the host is still encoding; [`Gpu::flush`] commits it and
//! returns a [`Fence`] a writer thread can wait on, and [`Gpu::synchronize`] commits and waits for everything.
//!
//! Apple silicon memory is unified: buffers use shared storage, so uploads and downloads are host copies into and out
//! of a buffer's contents, done only when no queued kernel can still touch those bytes. Adapted from sam3rs's Metal
//! backend (`src/metal/gpu.rs` there), without its buffer pool: inkrs allocates its buffers once per model or pass

use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::thread::JoinHandle;

use bytemuck::Pod;
use color_eyre::eyre::{Result, bail, ensure, eyre};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSOperatingSystemVersion, NSProcessInfo, NSRange, NSString};
use objc2_metal::{
    MTLBinding, MTLBindingType, MTLBuffer, MTLBufferBinding, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLCommonCounterSetTimestamp, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePassDescriptor, MTLComputePipelineState, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLGPUFamily,
    MTLLanguageVersion, MTLLibrary, MTLMathMode, MTLPipelineOption, MTLResourceOptions, MTLSize, MTLStorageMode,
};

use crate::fast::ReaderPrecision;

/// The kernel library's sources, compiled together in this order: the shared header first
const SOURCES: [&str; 5] = [
    include_str!("common.metal"),
    include_str!("conv.metal"),
    include_str!("glue.metal"),
    include_str!("hecate.metal"),
    include_str!("v8in.metal"),
];

/// The sources of the hybrid_3d2d reader's kernels, the fp32 library's only ones: Hecate and v8-in have no fp32
/// reader to serve, and leaving them out keeps that library's build short
const READER_SOURCES: usize = 3;

/// The macro that switches the reader kernels' storage type `real` (common.metal) from half to float
const FP32_DEFINE: &str = "#define INKRS_FP32 1\n";

/// Dispatches encoded before the open command buffer is committed
const FLUSH_EVERY: usize = 64;

/// A device buffer in shared storage, with its length in bytes
pub struct Buf {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    bytes: usize,
}

// SAFETY: Metal buffers are thread-safe objects; touching their contents while a kernel uses them is not, which the
// engine orders through command-buffer completion
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

impl Buf {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The whole buffer as a kernel argument
    pub fn arg(&self) -> Arg<'_> {
        Arg::Buf(self, 0)
    }

    /// The buffer from element `offset` of `T` on
    pub fn at<T>(&self, offset: usize) -> Arg<'_> {
        Arg::Buf(self, offset * std::mem::size_of::<T>())
    }

    /// The contents as a host slice; the caller must know no queued kernel writes them
    ///
    /// # Safety
    /// No command buffer that writes this buffer may be pending, and `T` must be valid for every bit pattern
    pub unsafe fn host<T: Pod>(&self) -> &[T] {
        let n = self.bytes / std::mem::size_of::<T>();
        // SAFETY: shared storage of at least `bytes` bytes, aligned for any Pod type (Metal aligns to a page), and
        // the caller rules out concurrent GPU writes
        unsafe { std::slice::from_raw_parts(self.raw.contents().as_ptr().cast::<T>(), n) }
    }

    /// The start of the contents, for host code that reads or writes disjoint ranges from several threads; every
    /// access through it must follow the rules of `host` and `host_mut` for the bytes it touches
    pub fn ptr(&self) -> NonNull<u8> {
        self.raw.contents().cast()
    }

    /// The contents as a mutable host slice
    ///
    /// # Safety
    /// No queued kernel may read or write this buffer
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn host_mut<T: Pod>(&self) -> &mut [T] {
        let n = self.bytes / std::mem::size_of::<T>();
        // SAFETY: as `host`, and the caller rules out any GPU access
        unsafe { std::slice::from_raw_parts_mut(self.raw.contents().as_ptr().cast::<T>(), n) }
    }
}

/// A kernel argument: a buffer from a byte offset, or a 4-byte scalar
#[derive(Clone, Copy)]
pub enum Arg<'a> {
    Buf(&'a Buf, usize),
    U32(u32),
    I32(i32),
    F32(f32),
}

impl From<u32> for Arg<'_> {
    fn from(v: u32) -> Self {
        Arg::U32(v)
    }
}

impl From<usize> for Arg<'_> {
    fn from(v: usize) -> Self {
        Arg::U32(u32::try_from(v).expect("kernel size argument overflows u32"))
    }
}

impl From<f32> for Arg<'_> {
    fn from(v: f32) -> Self {
        Arg::F32(v)
    }
}

impl<'a> From<&'a Buf> for Arg<'a> {
    fn from(b: &'a Buf) -> Self {
        Arg::Buf(b, 0)
    }
}

/// A kernel parameter as reflection reports it
#[derive(Clone, Copy, Debug)]
enum Param {
    Pointer,
    Scalar(MTLDataType),
}

/// A compiled kernel and its parameter list
struct Kernel {
    pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    params: Vec<Param>,
}

type Kernels = HashMap<String, Kernel>;

/// The kernel library's source for reader precision `precision`: at fp16 every kernel, as it always was; at fp32 the
/// reader's kernels only, with `real` defined as float
fn library_source(precision: ReaderPrecision) -> String {
    match precision {
        ReaderPrecision::Fp16 => SOURCES.concat(),
        ReaderPrecision::Fp32 => [FP32_DEFINE].iter().chain(&SOURCES[..READER_SOURCES]).copied().collect(),
    }
}

/// Compiles the kernel library and builds a pipeline for every kernel, spread over a few threads, so pipeline builds
/// stay out of the first forward; [`Gpu::open`] runs this on a background thread behind the weight load
fn build_kernels(device: &ProtocolObject<dyn MTLDevice>, precision: ReaderPrecision) -> Result<Kernels> {
    let options = MTLCompileOptions::new();
    // Metal 4 for the tensor ops; safe math so exp, division and rounding follow IEEE as the CUDA kernels' do
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    options.setMathMode(MTLMathMode::Safe);
    let source = NSString::from_str(&library_source(precision));
    let library = device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(|e| eyre!("compile the Metal kernels: {}", e.localizedDescription()))?;
    let names: Vec<String> = library.functionNames().iter().map(|n| n.to_string()).collect();
    let threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get).min(8);
    let library = &library;
    std::thread::scope(|s| {
        let workers: Vec<_> = names
            .chunks(names.len().div_ceil(threads).max(1))
            .map(|chunk| {
                s.spawn(move || {
                    chunk
                        .iter()
                        .map(|name| Ok((name.clone(), build_kernel(device, library, name)?)))
                        .collect::<Result<Vec<_>>>()
                })
            })
            .collect();
        let mut kernels = Kernels::new();
        for w in workers {
            kernels.extend(w.join().map_err(|_| eyre!("a Metal pipeline build panicked"))??);
        }

        Ok(kernels)
    })
}

/// The pipeline of kernel `name` and its parameter list, read from reflection
fn build_kernel(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Kernel> {
    let func =
        library.newFunctionWithName(&NSString::from_str(name)).ok_or_else(|| eyre!("no Metal kernel named {name}"))?;
    let mut reflection = None;
    // SAFETY: the function comes from our library and is only dispatched through `Gpu::dispatch`, which checks its
    // arguments against this reflection
    let pso = unsafe {
        device.newComputePipelineStateWithFunction_options_reflection_error(
            &func,
            MTLPipelineOption::BindingInfo | MTLPipelineOption::BufferTypeInfo,
            Some(&mut reflection),
        )
    }
    .map_err(|e| eyre!("build Metal pipeline {name}: {}", e.localizedDescription()))?;
    let reflection = reflection.ok_or_else(|| eyre!("no reflection for Metal kernel {name}"))?;
    let mut bindings: Vec<(usize, Param)> = Vec::new();
    for b in reflection.bindings().iter() {
        if b.r#type() != MTLBindingType::Buffer {
            continue;
        }
        // SAFETY: a binding of type Buffer is an MTLBufferBinding
        let buffer: &ProtocolObject<dyn MTLBufferBinding> =
            unsafe { &*(&*b as *const ProtocolObject<dyn MTLBinding>).cast() };
        let ty = buffer.bufferDataType();
        // pointers are named p_*; everything else must be a 4-byte scalar
        let param = if b.name().to_string().starts_with("p_") {
            Param::Pointer
        } else {
            ensure!(
                buffer.bufferDataSize() == 4 && matches!(ty, MTLDataType::UInt | MTLDataType::Int | MTLDataType::Float),
                "Metal kernel {name}: parameter {} is neither a p_* pointer nor a 4-byte scalar",
                b.name()
            );
            Param::Scalar(ty)
        };
        bindings.push((b.index(), param));
    }
    bindings.sort_by_key(|(i, _)| *i);
    for (pos, (i, _)) in bindings.iter().enumerate() {
        ensure!(*i == pos, "Metal kernel {name}: buffer indices must run 0..n without gaps, found {i} at {pos}");
    }

    Ok(Kernel { pso, params: bindings.into_iter().map(|(_, p)| p).collect() })
}

/// The open command buffer and its encoder
struct Batch {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    dispatches: usize,
}

/// The committed command buffers a writer thread can wait on: the newest one and every one committed before it since
/// the last `synchronize`, which reported on the ones before that
///
/// A map's output depends on logits written by earlier command buffers, and Metal can complete a buffer after an
/// earlier one failed, so a fence checks all of them, not only the buffer that wrote the map
#[derive(Clone)]
pub struct Fence(Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>);

// SAFETY: waiting on and reading the status of a committed command buffer is thread-safe
unsafe impl Send for Fence {}

impl Fence {
    /// Waits for every covered command buffer, failing if any failed
    pub fn wait(&self) -> Result<()> {
        for cb in &self.0 {
            cb.waitUntilCompleted();
            ensure!(
                cb.status() == MTLCommandBufferStatus::Completed,
                "a Metal command buffer failed: {}",
                failure_reason(cb)
            );
        }
        Ok(())
    }

    /// Whether every covered buffer has finished, without waiting; a failed buffer counts as finished
    pub fn done(&self) -> bool {
        self.0.iter().all(|cb| matches!(cb.status(), MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error))
    }

    /// GPU seconds from the oldest covered buffer's start to the newest one's end; the buffers run in commit order
    /// on one queue, so this spans all of them
    pub fn gpu_span(&self) -> (f64, f64) {
        let (first, last) = (self.0.first(), self.0.last());
        let (first, last) = first.zip(last).expect("a fence covers at least one command buffer");
        (first.GPUStartTime(), last.GPUEndTime())
    }
}

/// Timestamp samples a profiling window holds, the most one counter sample buffer may take
const PROFILE_SAMPLES: usize = 4096;

/// GPU timestamps around every dispatch, for `INKRS_METAL_PROFILE`
///
/// Each dispatch gets its own encoder, sampled at its start and end, since Apple GPUs sample only at encoder
/// boundaries. A dispatch is charged the time it adds to the GPU timeline, `end - max(start, previous end)`, so the
/// times sum to the busy time
struct Profiler {
    samples: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    pending: Vec<&'static str>,
    done: Vec<(&'static str, f64)>,
    origin: (u64, u64),
    last_end: u64,
}

impl Profiler {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        ensure!(
            device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary),
            "INKRS_METAL_PROFILE: this GPU can't sample timestamps at encoder boundaries"
        );
        // SAFETY: a static NSString constant of the Metal framework
        let want = unsafe { MTLCommonCounterSetTimestamp };
        let set = device
            .counterSets()
            .and_then(|sets| sets.iter().find(|s| MTLCounterSet::name(&**s).isEqualToString(want)))
            .ok_or_else(|| eyre!("INKRS_METAL_PROFILE: no timestamp counter set"))?;
        let desc = MTLCounterSampleBufferDescriptor::new();
        desc.setCounterSet(Some(&set));
        desc.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: within the 32 KB a sample buffer may hold
        unsafe { desc.setSampleCount(PROFILE_SAMPLES) };
        let samples = device
            .newCounterSampleBufferWithDescriptor_error(&desc)
            .map_err(|e| eyre!("INKRS_METAL_PROFILE: {}", e.localizedDescription()))?;
        Ok(Self { samples, pending: Vec::new(), done: Vec::new(), origin: timestamps(device), last_end: 0 })
    }

    fn full(&self) -> bool {
        2 * self.pending.len() >= PROFILE_SAMPLES
    }

    fn sample(&mut self, name: &'static str) -> Retained<MTLComputePassDescriptor> {
        let i = 2 * self.pending.len();
        self.pending.push(name);
        let desc = MTLComputePassDescriptor::computePassDescriptor();
        // SAFETY: index 0 exists on every pass descriptor, and both sample indices are inside the buffer, since
        // `full` synchronises before it runs out
        unsafe {
            let att = desc.sampleBufferAttachments().objectAtIndexedSubscript(0);
            att.setSampleBuffer(Some(&self.samples));
            att.setStartOfEncoderSampleIndex(i);
            att.setEndOfEncoderSampleIndex(i + 1);
        }
        desc
    }

    fn resolve(&mut self, device: &ProtocolObject<dyn MTLDevice>) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let n = 2 * self.pending.len();
        // SAFETY: the range lies inside the buffer and every command buffer that wrote it has completed
        let data = unsafe { self.samples.resolveCounterRange(NSRange::new(0, n)) }
            .ok_or_else(|| eyre!("INKRS_METAL_PROFILE: could not resolve the timestamps"))?;
        let bytes = data.to_vec();
        ensure!(bytes.len() >= 8 * n, "INKRS_METAL_PROFILE: {} bytes of timestamps for {n}", bytes.len());
        let ticks: Vec<u64> = bytes.as_chunks::<8>().0.iter().map(|c| u64::from_ne_bytes(*c)).collect();
        let now = timestamps(device);
        let ns_per_tick = (now.0 - self.origin.0) as f64 / (now.1 - self.origin.1).max(1) as f64;
        for (i, name) in self.pending.drain(..).enumerate() {
            let (a, b) = (ticks[2 * i], ticks[2 * i + 1]);
            // a sample the GPU couldn't take reads as all ones
            if a == u64::MAX || b == u64::MAX || b < a {
                continue;
            }
            let from = a.max(self.last_end);
            self.done.push((name, b.saturating_sub(from) as f64 * ns_per_tick * 1e-9));
            self.last_end = self.last_end.max(b);
        }
        Ok(())
    }
}

fn timestamps(device: &ProtocolObject<dyn MTLDevice>) -> (u64, u64) {
    let (mut cpu, mut gpu) = (0u64, 0u64);
    // SAFETY: both pointers are to live u64s
    unsafe { device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu), NonNull::from(&mut gpu)) };
    (cpu, gpu)
}

fn failure_reason(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> String {
    cb.error().map_or_else(|| "unknown error".into(), |e| e.localizedDescription().to_string())
}

/// The Metal device, its queue and the kernel library
pub struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// The reader precision the library was compiled for; the reader's model takes it from here, so its buffers and
    /// the kernels that read them always agree on the storage type
    precision: ReaderPrecision,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    build: RefCell<Option<JoinHandle<Result<Kernels>>>>,
    kernels: OnceCell<Kernels>,
    batch: RefCell<Option<Batch>>,
    /// Committed command buffers not yet waited on
    inflight: RefCell<Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    /// The newest buffer the last [`Gpu::synchronize`] waited on, for a flush with nothing committed since
    settled: RefCell<Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    profile: Option<RefCell<Profiler>>,
}

// SAFETY: `Gpu` is not `Sync` (its `RefCell`s), so one thread at a time uses it, and Metal's command buffers and
// encoders may move between threads between uses
unsafe impl Send for Gpu {}

impl Gpu {
    /// Opens the system's default Metal device and starts compiling the kernel library, with the reader's kernels at
    /// fp16, on a background thread; the first dispatch or [`Gpu::ready`] waits for it. Apple silicon has one GPU,
    /// device 0
    pub fn open(ordinal: usize) -> Result<Self> {
        Self::open_at(ordinal, ReaderPrecision::Fp16)
    }

    /// [`Gpu::open`] with the reader's kernels at `precision`. At fp32 the library holds the reader's kernels only,
    /// so Hecate and v8-in need a library opened at fp16
    pub fn open_at(ordinal: usize, precision: ReaderPrecision) -> Result<Self> {
        crate::ensure_supported!(ordinal == 0, "Metal device {ordinal} requested, but Apple silicon has only device 0");
        // older systems lack Metal 4 and `setMathMode:`, which raises an Objective-C exception instead of failing
        let info = NSProcessInfo::processInfo();
        let v = info.operatingSystemVersion();
        crate::ensure_supported!(
            info.isOperatingSystemAtLeastVersion(NSOperatingSystemVersion {
                majorVersion: 26,
                minorVersion: 0,
                patchVersion: 0
            }),
            "the Metal backend needs macOS 26 or newer (Metal 4), and this is macOS {}.{}.{}",
            v.majorVersion,
            v.minorVersion,
            v.patchVersion
        );
        let device = MTLCreateSystemDefaultDevice().ok_or_else(|| eyre!("no Metal device"))?;
        crate::ensure_supported!(
            device.supportsFamily(MTLGPUFamily::Apple7),
            "the Metal backend needs Apple silicon (M1 or newer), and this GPU is {}",
            device.name()
        );
        let queue = device.newCommandQueue().ok_or_else(|| eyre!("Metal could not create a command queue"))?;
        let build = {
            let device = device.clone();
            std::thread::Builder::new()
                .name("inkrs-metal-kernels".into())
                .spawn(move || build_kernels(&device, precision))?
        };
        let profile = std::env::var_os("INKRS_METAL_PROFILE").map(|_| Profiler::new(&device)).transpose()?;
        Ok(Self {
            device,
            precision,
            queue,
            build: RefCell::new(Some(build)),
            kernels: OnceCell::new(),
            batch: RefCell::default(),
            inflight: RefCell::default(),
            settled: RefCell::default(),
            profile: profile.map(RefCell::new),
        })
    }

    /// The reader precision the kernel library was compiled for
    pub fn precision(&self) -> ReaderPrecision {
        self.precision
    }

    /// The GPU's name, such as "Apple M4 Pro"
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    /// This process's Metal allocations on the device and the device's recommended working set, in bytes
    pub fn memory(&self) -> (usize, usize) {
        (self.device.currentAllocatedSize(), self.device.recommendedMaxWorkingSetSize() as usize)
    }

    /// Waits for the kernel build started by [`Gpu::open`], surfacing a compile error
    pub fn ready(&self) -> Result<()> {
        self.kernels().map(|_| ())
    }

    fn kernels(&self) -> Result<&Kernels> {
        if let Some(k) = self.kernels.get() {
            return Ok(k);
        }
        let build = self.build.borrow_mut().take().ok_or_else(|| eyre!("the Metal kernel build failed earlier"))?;
        let kernels = build.join().map_err(|_| eyre!("the Metal kernel build panicked"))??;
        Ok(self.kernels.get_or_init(|| kernels))
    }

    fn kernel(&self, name: &str) -> Result<&Kernel> {
        self.kernels()?.get(name).ok_or_else(|| eyre!("no Metal kernel named {name}"))
    }

    /// A zero-filled buffer of `bytes` (at least 4)
    pub fn alloc(&self, bytes: usize) -> Result<Buf> {
        self.alloc_fill(bytes, 0)
    }

    /// A buffer of `bytes` with every byte `value`, written by the host before any kernel can see it
    pub fn alloc_fill(&self, bytes: usize, value: u8) -> Result<Buf> {
        let len = bytes.max(4);
        let raw = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| eyre!("Metal could not allocate {len} bytes"))?;
        // SAFETY: a fresh shared buffer of `len` bytes no command uses yet
        unsafe { std::ptr::write_bytes(raw.contents().as_ptr().cast::<u8>(), value, len) };
        Ok(Buf { raw, bytes })
    }

    /// A buffer holding `data`
    pub fn upload<T: Pod>(&self, data: &[T]) -> Result<Buf> {
        let bytes = std::mem::size_of_val(data);
        let buf = self.alloc_fill(bytes, 0)?;
        // SAFETY: the buffer is fresh, at least `bytes` long, and unused by any command
        unsafe { buf.host_mut::<u8>()[..bytes].copy_from_slice(bytemuck::cast_slice(data)) };
        Ok(buf)
    }

    /// A buffer of `n` values in the reader's storage type ([`Gpu::precision`]), every one NaN, so a value no kernel
    /// writes fails the finiteness checks: all-ones bytes are NaN in FP16 and FP32 alike
    pub fn alloc_real_nan(&self, n: usize) -> Result<Buf> {
        self.alloc_fill(n * self.precision.bytes(), 0xFF)
    }

    /// A buffer holding `values` in the reader's storage type: rounded to FP16 at fp16, as they are at fp32
    pub fn upload_real(&self, values: &[f32]) -> Result<Buf> {
        match self.precision {
            ReaderPrecision::Fp16 => self.upload(&values.iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<_>>()),
            ReaderPrecision::Fp32 => self.upload(values),
        }
    }

    /// How many of the first `n` values of `buf`, in the reader's storage type, are not finite; the copy stays in that
    /// type, so an fp16 check holds 2 bytes per value
    pub fn nonfinite_real(&self, buf: &Buf, n: usize) -> Result<usize> {
        match self.precision {
            ReaderPrecision::Fp16 => {
                let mut v = vec![half::f16::ZERO; n];
                self.download(buf, 0, &mut v)?;
                Ok(v.iter().filter(|x| !x.is_finite()).count())
            }
            ReaderPrecision::Fp32 => {
                let mut v = vec![0.0f32; n];
                self.download(buf, 0, &mut v)?;
                Ok(v.iter().filter(|x| !x.is_finite()).count())
            }
        }
    }

    /// Waits for the GPU, then copies `out.len()` elements of `T` from element `offset` of `buf`
    pub fn download<T: Pod>(&self, buf: &Buf, offset: usize, out: &mut [T]) -> Result<()> {
        let size = std::mem::size_of::<T>();
        ensure!((offset + out.len()) * size <= buf.bytes, "download past the end of a {}-byte buffer", buf.bytes);
        self.synchronize()?;
        // SAFETY: in bounds and no kernel runs after the sync
        out.copy_from_slice(&unsafe { buf.host::<T>() }[offset..offset + out.len()]);
        Ok(())
    }

    /// Encodes kernel `name` over `groups` threadgroups of `threads` threads
    pub fn dispatch(&self, name: &'static str, groups: [usize; 3], threads: [usize; 3], args: &[Arg]) -> Result<()> {
        if groups.contains(&0) {
            return Ok(());
        }
        self.encode(name, args, |enc, k| {
            ensure!(
                threads.iter().product::<usize>() <= k.pso.maxTotalThreadsPerThreadgroup(),
                "Metal kernel {name}: {threads:?} threads exceed its limit of {}",
                k.pso.maxTotalThreadsPerThreadgroup()
            );
            enc.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threads));
            Ok(())
        })
    }

    /// Encodes kernel `name` over exactly `n` threads, for one-thread-per-element kernels
    pub fn dispatch_1d(&self, name: &'static str, n: usize, args: &[Arg]) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.encode(name, args, |enc, k| {
            let per_group = k.pso.maxTotalThreadsPerThreadgroup().min(256);
            enc.dispatchThreads_threadsPerThreadgroup(size([n, 1, 1]), size([per_group, 1, 1]));
            Ok(())
        })
    }

    fn encode(
        &self,
        name: &'static str,
        args: &[Arg],
        run: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>, &Kernel) -> Result<()>,
    ) -> Result<()> {
        let k = self.kernel(name)?;
        ensure!(
            args.len() == k.params.len(),
            "Metal kernel {name} takes {} buffer arguments, got {}",
            k.params.len(),
            args.len()
        );
        for (i, (arg, param)) in args.iter().zip(&k.params).enumerate() {
            let ok = match (arg, param) {
                (Arg::Buf(b, offset), Param::Pointer) => *offset <= b.bytes,
                (Arg::U32(_), Param::Scalar(MTLDataType::UInt))
                | (Arg::I32(_), Param::Scalar(MTLDataType::Int))
                | (Arg::F32(_), Param::Scalar(MTLDataType::Float)) => true,
                _ => false,
            };
            if !ok {
                bail!("Metal kernel {name}: argument {i} doesn't match parameter {param:?}");
            }
        }

        let mut batch = self.batch.borrow_mut();
        if batch.is_none() {
            let cb = self.queue.commandBuffer().ok_or_else(|| eyre!("Metal could not create a command buffer"))?;
            let enc = self.encoder(&cb, name)?;
            *batch = Some(Batch { cb, enc, dispatches: 0 });
        } else if self.profile.is_some() {
            let b = batch.as_mut().expect("checked above");
            b.enc.endEncoding();
            b.enc = self.encoder(&b.cb, name)?;
        }
        let b = batch.as_mut().expect("opened above");
        b.enc.setComputePipelineState(&k.pso);
        for (i, arg) in args.iter().enumerate() {
            // SAFETY: the buffer outlives the dispatch (Metal retains it until the command buffer completes) and the
            // offset is inside it; scalars are 4 bytes copied now
            unsafe {
                match *arg {
                    Arg::Buf(buf, offset) => b.enc.setBuffer_offset_atIndex(Some(&buf.raw), offset, i),
                    Arg::U32(v) => b.enc.setBytes_length_atIndex(NonNull::from(&v).cast(), 4, i),
                    Arg::I32(v) => b.enc.setBytes_length_atIndex(NonNull::from(&v).cast(), 4, i),
                    Arg::F32(v) => b.enc.setBytes_length_atIndex(NonNull::from(&v).cast(), 4, i),
                }
            }
        }
        run(&b.enc, k)?;
        b.dispatches += 1;
        if self.profile.as_ref().is_some_and(|p| p.borrow().full()) {
            drop(batch);
            return self.synchronize();
        }
        if b.dispatches >= FLUSH_EVERY {
            drop(batch);
            self.commit();
        }
        Ok(())
    }

    fn encoder(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        name: &'static str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
        let enc = match &self.profile {
            None => cb.computeCommandEncoder(),
            Some(p) => {
                let desc = p.borrow_mut().sample(name);
                cb.computeCommandEncoderWithDescriptor(&desc)
            }
        };
        enc.ok_or_else(|| eyre!("Metal could not create an encoder"))
    }

    /// Commits the open command buffer without waiting, returning a fence on everything queued so far
    pub fn flush(&self) -> Result<Option<Fence>> {
        self.commit();
        let inflight = self.inflight.borrow();
        if !inflight.is_empty() {
            return Ok(Some(Fence(inflight.clone())));
        }
        Ok(self.settled.borrow().clone().map(|cb| Fence(vec![cb])))
    }

    /// Commits the open command buffer without waiting, returning a fence on the newest committed buffer only
    ///
    /// One queue runs its buffers in commit order, so this still orders the host after everything queued so far,
    /// but it reports only the newest buffer's failure; a later `flush` fence checks the rest
    pub fn flush_newest(&self) -> Result<Option<Fence>> {
        self.commit();
        let newest = self.inflight.borrow().last().cloned().or_else(|| self.settled.borrow().clone());
        Ok(newest.map(|cb| Fence(vec![cb])))
    }

    /// Commits the open command buffer, if any, without waiting
    fn commit(&self) {
        if let Some(b) = self.batch.borrow_mut().take() {
            b.enc.endEncoding();
            b.cb.commit();
            self.inflight.borrow_mut().push(b.cb);
        }
    }

    /// Commits pending work and waits for all of it, failing if any command buffer failed
    pub fn synchronize(&self) -> Result<()> {
        self.commit();
        let inflight = std::mem::take(&mut *self.inflight.borrow_mut());
        let mut failure = None;
        for cb in &inflight {
            cb.waitUntilCompleted();
            if cb.status() != MTLCommandBufferStatus::Completed {
                failure.get_or_insert_with(|| failure_reason(cb));
            }
        }
        // keep the newest buffer, checked above, so a flush with nothing new still returns a fence on everything
        // queued so far: a profiling window can synchronize between a map's last dispatch and the flush that fences it.
        // It stays out of `inflight`, so the next work's fences, and the GPU spans timed from them, start at that work
        if failure.is_none() && !inflight.is_empty() {
            *self.settled.borrow_mut() = inflight.last().cloned();
        }
        if let Some(p) = &self.profile {
            p.borrow_mut().resolve(&self.device)?;
        }
        match failure {
            Some(why) => Err(eyre!("a Metal command buffer failed ({why})")),
            None => Ok(()),
        }
    }

    /// Per-kernel GPU seconds since the last call, by kernel name, sorted by total, with dispatch counts; empty
    /// unless `INKRS_METAL_PROFILE` is set
    pub fn take_profile(&self) -> Result<Vec<(&'static str, usize, f64)>> {
        let Some(profile) = &self.profile else { return Ok(Vec::new()) };
        self.synchronize()?;
        let mut by_name: HashMap<&'static str, (usize, f64)> = HashMap::new();
        for (name, t) in profile.borrow_mut().done.drain(..) {
            let e = by_name.entry(name).or_default();
            e.0 += 1;
            e.1 += t;
        }
        let mut rows: Vec<_> = by_name.into_iter().map(|(n, (c, t))| (n, c, t)).collect();
        rows.sort_by(|a, b| b.2.total_cmp(&a.2));
        Ok(rows)
    }

    /// Prints [`Gpu::take_profile`] to stderr, if `INKRS_METAL_PROFILE` is set
    pub fn print_profile(&self) -> Result<()> {
        let profile = self.take_profile()?;
        if profile.is_empty() {
            return Ok(());
        }
        let total: f64 = profile.iter().map(|r| r.2).sum();
        eprintln!("INKRS_METAL_PROFILE: {:.1} ms of GPU time by kernel", total * 1e3);
        for (name, count, t) in profile {
            eprintln!("  {name:>28} {count:>8} {:>10.1} ms {:>5.1}%", t * 1e3, 100.0 * t / total);
        }
        Ok(())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // queued kernels may still use buffers that drop after this
        let _ = self.synchronize();
    }
}

fn size(v: [usize; 3]) -> MTLSize {
    MTLSize { width: v[0], height: v[1], depth: v[2] }
}
