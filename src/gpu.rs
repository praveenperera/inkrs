//! The device: cudakit's GPU and stream, the embedded inkrs PTX module, device and pinned
//! buffers, and kernel launches checked against the PTX param lists

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use cudakit::{ComputeCapability, ExtraSymbols, Library, LibraryPolicy, Profile, PtxTier, TierSet};
use cudarc::driver::CudaModule;
use cudarc::driver::result as cur;
use cudarc::driver::sys as cu;
use cudarc::nvrtc::Ptx;

/// The kernels for each shipped tier, built by `scripts/build-ptx.sh` from `inkrs-kernels`
///
/// sm75 is the Turing build (T4, RTX 20), from the same source with the kernel crate's `sm75`
/// feature, which swaps the two FP16 instructions sm_75 lacks for ones it has. sm80 (PTX ISA 7.0)
/// runs on every GPU from RTX 30 / A100 up, JIT-compiled by the driver; sm120 (ISA 8.7) is the
/// RTX 50 build. sm80 and sm120 build from the same source without the feature
pub const PTX: &[(PtxTier, &str)] = &[
    (PtxTier::Sm75, include_str!("../ptx/inkrs.sm75.ptx")),
    (PtxTier::Sm80, include_str!("../ptx/inkrs.sm80.ptx")),
    (PtxTier::Sm120, include_str!("../ptx/inkrs.sm120.ptx")),
];

/// One tier's PTX compiled ahead to SASS for one GPU family by `scripts/build-cubin.sh`
pub struct Cubin {
    /// The tier whose PTX it was compiled from
    pub tier: PtxTier,
    /// The family it was compiled for; it also runs on later minor versions of the same major
    pub arch: ComputeCapability,
    pub bytes: &'static [u8],
}

/// The cubins each tier ships, so a GPU of these families loads its kernels without the driver's
/// JIT, which costs over a second on a machine's first run. A GPU with none here, such as 11.x, or a
/// tier forced below the device's, JITs the PTX instead
pub const CUBINS: &[Cubin] = &[
    cubin(PtxTier::Sm75, 7, 5, include_bytes!("../ptx/inkrs.sm75.sm_75.cubin")),
    cubin(PtxTier::Sm80, 8, 0, include_bytes!("../ptx/inkrs.sm80.sm_80.cubin")),
    cubin(PtxTier::Sm80, 8, 6, include_bytes!("../ptx/inkrs.sm80.sm_86.cubin")),
    cubin(PtxTier::Sm80, 8, 9, include_bytes!("../ptx/inkrs.sm80.sm_89.cubin")),
    cubin(PtxTier::Sm80, 9, 0, include_bytes!("../ptx/inkrs.sm80.sm_90.cubin")),
    cubin(PtxTier::Sm80, 10, 0, include_bytes!("../ptx/inkrs.sm80.sm_100.cubin")),
    cubin(PtxTier::Sm120, 12, 0, include_bytes!("../ptx/inkrs.sm120.sm_120.cubin")),
];

const fn cubin(tier: PtxTier, major: u32, minor: u32, bytes: &'static [u8]) -> Cubin {
    Cubin { tier, arch: ComputeCapability::new(major, minor), bytes }
}

/// The cubin of `tier` a device of `capability` runs: the same major version, and the highest minor
/// version at or below the device's
fn cubin_for(tier: PtxTier, capability: ComputeCapability) -> Option<&'static Cubin> {
    CUBINS
        .iter()
        .filter(|c| c.tier == tier && c.arch.major == capability.major && c.arch.minor <= capability.minor)
        .max_by_key(|c| c.arch.minor)
}

/// How the loaded kernels reached the GPU
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernels {
    /// A cubin compiled ahead for this family
    Sass(ComputeCapability),
    /// The tier's PTX, compiled by the driver's JIT
    Ptx,
}

impl std::fmt::Display for Kernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Kernels::Sass(arch) => write!(f, "sm_{}{} SASS", arch.major, arch.minor),
            Kernels::Ptx => f.write_str("PTX, JIT-compiled"),
        }
    }
}

/// Forces a PTX tier at or below the device's, such as `sm80` to run the RTX 30 kernels on an RTX 50,
/// or `sm75` to run the Turing kernels on a newer GPU
pub const TIER_ENV: &str = "INKRS_PTX_TIER";

/// Each embedded PTX tier's `.target` and `.version` (PTX ISA) lines, lowest tier first, for build
/// reports and the minimum-driver figures of the Python package
pub fn ptx_targets() -> Vec<(String, String)> {
    PTX.iter().map(|(_, ptx)| (ptx_directive(ptx, ".target "), ptx_directive(ptx, ".version "))).collect()
}

/// Each embedded cubin as `(PTX target it was compiled from, GPU family it runs on)`, such as
/// `("sm_80", "sm_86")`, in [`CUBINS`] order, for build reports
pub fn cubin_targets() -> Vec<(String, String)> {
    CUBINS
        .iter()
        .map(|c| {
            let ptx = PTX.iter().find(|(tier, _)| *tier == c.tier).map(|(_, ptx)| *ptx).unwrap_or_default();
            (ptx_directive(ptx, ".target "), format!("sm_{}{}", c.arch.major, c.arch.minor))
        })
        .collect()
}

fn ptx_directive(ptx: &str, name: &str) -> String {
    ptx.lines().find_map(|l| l.trim().strip_prefix(name).map(|v| v.trim().to_owned())).unwrap_or_default()
}

/// cuDNN entry points the conv plans call directly, beyond cudakit's own list
const CUDNN_SYMBOLS: &[&std::ffi::CStr] = &[
    c"cudnnSetTensorNdDescriptorEx",
    c"cudnnSetFilterNdDescriptor",
    c"cudnnSetConvolutionNdDescriptor",
    c"cudnnGetConvolutionNdForwardOutputDim",
    c"cudnnFindConvolutionForwardAlgorithmEx",
    c"cudnnGetErrorString",
    // the fused Hecate convs' backend graph API (`dnn::fused`)
    c"cudnnBackendCreateDescriptor",
    c"cudnnBackendDestroyDescriptor",
    c"cudnnBackendSetAttribute",
    c"cudnnBackendGetAttribute",
    c"cudnnBackendFinalize",
    c"cudnnBackendExecute",
];

/// cuBLAS entry points the transposed convs call
const CUBLAS_SYMBOLS: &[&std::ffi::CStr] = &[c"cublasGemmEx", c"cublasCreate_v2", c"cublasSetStream_v2"];

/// The inkrs build policy: sm75, sm80 and sm120 PTX, cuDNN and cuBLAS allowed
pub static PROFILE: Profile = Profile {
    consumer: "inkrs",
    compiled: TierSet::of(&[PtxTier::Sm75, PtxTier::Sm80, PtxTier::Sm120]),
    shipped: TierSet::of(&[PtxTier::Sm75, PtxTier::Sm80, PtxTier::Sm120]),
    tier_features: &[],
    libraries: LibraryPolicy::Allowed,
    extra_symbols: ExtraSymbols { cublas: CUBLAS_SYMBOLS, cudnn: CUDNN_SYMBOLS, nvrtc: &[] },
};

/// One PTX parameter's kind
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Param {
    Ptr,
    U64,
    U32,
    F32,
}

/// A kernel argument
///
/// Each `&[T]` or `DisjointSlice<T>` parameter of a cuda-oxide kernel lowers to two PTX
/// params, the address then the length in elements of `T`, which is [`Arg::Slice`]
#[derive(Clone, Copy, Debug)]
pub enum Arg {
    /// Address and length in elements of the kernel's slice type
    Slice(u64, u64),
    U32(u32),
    F32(f32),
}

impl Arg {
    /// An empty slice for an optional input the kernel skips when empty; the address is
    /// never read but must be non-null and aligned like any Rust slice's
    pub const EMPTY: Arg = Arg::Slice(256, 0);

    fn params(&self) -> &'static [Param] {
        match self {
            Arg::Slice(..) => &[Param::Ptr, Param::U64],
            // the PTX declares i32 params as .u32 too
            Arg::U32(_) => &[Param::U32],
            Arg::F32(_) => &[Param::F32],
        }
    }
}

/// An owned device allocation
#[derive(Debug)]
pub struct Buf {
    ptr: u64,
    bytes: usize,
}

impl Buf {
    /// The device address
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Size in bytes
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The buffer as a kernel slice of `T`
    pub fn arg<T>(&self) -> Arg {
        Arg::Slice(self.ptr, (self.bytes / size_of::<T>()) as u64)
    }

    /// `len` elements of `T` starting `offset` elements in
    pub fn view<T>(&self, offset: usize, len: usize) -> Arg {
        assert!((offset + len) * size_of::<T>() <= self.bytes, "view past the end of a {}-byte buffer", self.bytes);
        Arg::Slice(self.ptr + (offset * size_of::<T>()) as u64, len as u64)
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: the allocation is ours and every queued use finished, since owners
            // synchronize the stream before dropping their buffers
            let _ = unsafe { cur::free_sync(self.ptr) };
        }
    }
}

/// Page-locked host memory, so async device-to-host copies are truly async and the
/// buffer outlives them by construction (the owner synchronizes before reading or dropping)
#[derive(Debug)]
pub struct Pinned {
    ptr: *mut u8,
    bytes: usize,
}

// SAFETY: plain host memory; access is coordinated by stream synchronization
unsafe impl Send for Pinned {}

impl Pinned {
    /// The contents; only valid once the copies into it have completed
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `bytes` were allocated at `ptr`
        unsafe { std::slice::from_raw_parts(self.ptr, self.bytes) }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemHostAlloc` and no copy into it is pending
        let _ = unsafe { cur::free_host(self.ptr.cast()) };
    }
}

/// The GPU, its stream and the loaded inkrs kernels
pub struct Gpu {
    // the module and handles must drop before the cudakit GPU's context
    functions: RefCell<HashMap<&'static str, (cu::CUfunction, Vec<Param>)>>,
    module: Arc<CudaModule>,
    ptx: &'static str,
    kernels: Kernels,
    pub inner: cudakit::Gpu,
}

impl Gpu {
    /// Opens device 0, prepares cuDNN and cuBLAS and loads the kernels of the highest shipped tier
    /// the device runs, or of the tier `INKRS_PTX_TIER` names: the tier's cubin for the device's
    /// family when there is one, else its PTX
    pub fn open() -> Result<Self> {
        Self::open_device(0)
    }

    /// [`Gpu::open`] on device `ordinal`
    pub fn open_device(ordinal: usize) -> Result<Self> {
        let requested = match std::env::var(TIER_ENV) {
            Ok(value) if !value.is_empty() => {
                Some(value.parse::<PtxTier>().wrap_err_with(|| format!("{TIER_ENV}={value}"))?)
            }
            Ok(_) | Err(std::env::VarError::NotPresent) => None,
            Err(e) => bail!("{TIER_ENV}: {e}"),
        };
        // cudarc opens each library by bare name with Windows' default search, which reaches the current directory
        #[cfg(windows)]
        crate::dll::pin().wrap_err("find the CUDA libraries")?;
        let inner = cudakit::Gpu::open(ordinal, &PROFILE, requested)
            .wrap_err_with(|| format!("open GPU {ordinal} through cudakit"))?;
        inner.prepare(Library::Cudnn).wrap_err("prepare cuDNN")?;
        inner.prepare(Library::Cublas).wrap_err("prepare cuBLAS")?;
        let tier = inner.ptx_tier();
        let ptx = ptx_for(tier)?;
        let (module, kernels) = Self::load_kernels(&inner, tier, ptx)?;
        Ok(Self { functions: RefCell::new(HashMap::new()), module, ptx, kernels, inner })
    }

    /// Loads `tier`'s cubin for the device, or its PTX when there is none or the driver refuses the
    /// cubin, as a driver older than the ptxas that built it may
    fn load_kernels(inner: &cudakit::Gpu, tier: PtxTier, ptx: &str) -> Result<(Arc<CudaModule>, Kernels)> {
        if let Some(cubin) = cubin_for(tier, inner.compute_capability()) {
            match inner.context().load_module(Ptx::from_binary(cubin.bytes.to_vec())) {
                Ok(module) => return Ok((module, Kernels::Sass(cubin.arch))),
                // the PTX still runs, only the first run is slower, so this warns rather than fails
                Err(e) => eprintln!("inkrs: the driver refused the {tier} cubin ({e}); JIT-compiling the PTX"),
            }
        }

        let module =
            inner.context().load_module(Ptx::from_src(ptx)).wrap_err_with(|| format!("load the inkrs {tier} PTX"))?;
        Ok((module, Kernels::Ptx))
    }

    /// How the kernels were loaded: a cubin compiled ahead, or the PTX through the driver's JIT
    pub fn kernels(&self) -> Kernels {
        self.kernels
    }

    /// The raw stream
    pub fn cu_stream(&self) -> cu::CUstream {
        self.inner.stream().cu_stream()
    }

    /// Allocates `bytes` of device memory, uninitialised
    pub fn alloc(&self, bytes: usize) -> Result<Buf> {
        self.inner.context().bind_to_thread()?;
        // SAFETY: a plain allocation; a zero-byte request still gets a valid address
        let ptr =
            unsafe { cur::malloc_sync(bytes.max(256)) }.wrap_err_with(|| format!("allocate {bytes} device bytes"))?;
        Ok(Buf { ptr, bytes })
    }

    /// Allocates `bytes` and fills every byte with `byte`, queued on the stream
    pub fn alloc_fill(&self, bytes: usize, byte: u8) -> Result<Buf> {
        let buf = self.alloc(bytes)?;
        self.fill(&buf, byte)?;
        Ok(buf)
    }

    /// Fills `buf` with `byte`, queued on the stream; 0xFF fills FP16 and FP32 with NaN
    pub fn fill(&self, buf: &Buf, byte: u8) -> Result<()> {
        // SAFETY: `buf` is a live allocation of `bytes`
        unsafe { cur::memset_d8_async(buf.ptr, byte, buf.bytes, self.cu_stream()) }?;
        Ok(())
    }

    /// Uploads `data` into a new buffer and waits for the copy, so `data` may drop after
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<Buf> {
        let buf = self.alloc(std::mem::size_of_val(data))?;
        self.upload_into(&buf, 0, data)?;
        Ok(buf)
    }

    /// Copies `data` to `buf` at byte `offset` and waits, so `data` may drop after
    pub fn upload_into<T: Copy>(&self, buf: &Buf, offset: usize, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        ensure!(
            offset + bytes <= buf.bytes,
            "upload of {bytes} bytes at {offset} overruns a {}-byte buffer",
            buf.bytes
        );
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: the range is in bounds and the synchronize below keeps `data` alive
        // until the copy is done
        unsafe {
            cur::memcpy_htod_async(buf.ptr + offset as u64, data, self.cu_stream())?;
            cur::stream::synchronize(self.cu_stream())?;
        }
        Ok(())
    }

    /// Copies `buf` back to a vector, synchronously
    pub fn download<T: Copy + Default>(&self, buf: &Buf) -> Result<Vec<T>> {
        self.download_range(buf, 0, buf.bytes / size_of::<T>())
    }

    /// Copies `len` elements of `T` starting `offset` elements in, synchronously
    pub fn download_range<T: Copy + Default>(&self, buf: &Buf, offset: usize, len: usize) -> Result<Vec<T>> {
        let bytes = len * size_of::<T>();
        ensure!(offset * size_of::<T>() + bytes <= buf.bytes, "download past the end of a {}-byte buffer", buf.bytes);
        let mut out = vec![T::default(); len];
        if len == 0 {
            return Ok(out);
        }
        // SAFETY: in bounds, and the synchronize keeps `out` alive until the copy lands
        unsafe {
            cur::memcpy_dtoh_async(&mut out, buf.ptr + (offset * size_of::<T>()) as u64, self.cu_stream())?;
            cur::stream::synchronize(self.cu_stream())?;
        }
        Ok(out)
    }

    /// Allocates `bytes` of pinned host memory
    pub fn pinned(&self, bytes: usize) -> Result<Pinned> {
        self.inner.context().bind_to_thread()?;
        // SAFETY: a plain allocation
        let ptr = unsafe { cur::malloc_host(bytes.max(1), 0) }?;
        Ok(Pinned { ptr: ptr.cast(), bytes })
    }

    /// Queues a copy of `bytes` from `src` at byte `offset` into `dst`
    ///
    /// # Safety
    ///
    /// `dst` must not be read or dropped until the stream has passed this copy
    pub unsafe fn download_async(&self, dst: &mut Pinned, src: &Buf, offset: usize, bytes: usize) -> Result<()> {
        ensure!(bytes <= dst.bytes && offset + bytes <= src.bytes, "async download out of bounds");
        // SAFETY: bounds checked; the caller keeps `dst` alive past the copy
        unsafe {
            let dst = std::slice::from_raw_parts_mut(dst.ptr, bytes);
            cur::memcpy_dtoh_async(dst, src.ptr + offset as u64, self.cu_stream())?;
        }
        Ok(())
    }

    /// Waits for every queued operation
    pub fn sync(&self) -> Result<()> {
        // SAFETY: our own live stream
        unsafe { cur::stream::synchronize(self.cu_stream()) }.wrap_err("synchronize the stream")?;
        Ok(())
    }

    fn function(&self, name: &'static str, args: &[Arg]) -> Result<cu::CUfunction> {
        let mut functions = self.functions.borrow_mut();
        if !functions.contains_key(name) {
            let f = self.module.load_function(name).wrap_err_with(|| format!("kernel {name}"))?;
            functions.insert(name, (f.cu_function(), entry_params(self.ptx, name)?));
        }
        let (f, params) = &functions[name];
        let given: Vec<Param> = args.iter().flat_map(|a| a.params().iter().copied()).collect();
        ensure!(&given == params, "launch {name}: arguments {given:?} do not match the PTX params {params:?}");
        Ok(*f)
    }

    /// Launches `name`, failing without launching if `args` don't match its PTX params
    pub fn launch(
        &self,
        name: &'static str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        smem: u32,
        args: &[Arg],
    ) -> Result<()> {
        let f = self.function(name, args)?;
        ensure!(grid.0 as u64 * block.0 as u64 <= 1 << 32, "launch {name}: more than 2^32 threads along x");
        let mut params: Vec<*mut c_void> = Vec::with_capacity(2 * args.len());
        for a in args {
            match a {
                Arg::Slice(ptr, len) => {
                    params.extend([ptr as *const u64 as *mut c_void, len as *const u64 as *mut c_void]);
                }
                Arg::U32(v) => params.push(v as *const u32 as *mut c_void),
                Arg::F32(v) => params.push(v as *const f32 as *mut c_void),
            }
        }
        if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
            return Ok(());
        }
        // SAFETY: the params match the kernel's PTX signature and point at `args`, which
        // outlive the call; the driver copies them at launch
        unsafe { cur::launch_kernel(f, grid, block, smem, self.cu_stream(), &mut params) }
            .wrap_err_with(|| format!("launch {name}"))?;
        Ok(())
    }

    /// Launches a one-thread-per-element kernel over `n` elements, 256 threads per block
    pub fn launch_1d(&self, name: &'static str, n: usize, args: &[Arg]) -> Result<()> {
        let n = u32::try_from(n).map_err(|_| eyre!("launch {name}: {n} threads overflow u32"))?;
        self.launch(name, (n.div_ceil(256), 1, 1), (256, 1, 1), 0, args)
    }
}

/// The embedded PTX of `tier`, checked to be a non-empty build for that target
fn ptx_for(tier: PtxTier) -> Result<&'static str> {
    let Some((_, ptx)) = PTX.iter().find(|(t, _)| *t == tier) else {
        bail!("no embedded PTX for tier {tier}");
    };
    let target = format!("\n.target sm_{}\n", &tier.name()[2..]);
    ensure!(ptx.contains(&target), "the embedded {tier} PTX lacks `{}`; run scripts/build-ptx.sh", target.trim());
    ensure!(ptx.contains(".entry "), "the embedded {tier} PTX has no kernels; run scripts/build-ptx.sh");
    Ok(ptx)
}

/// The params of kernel `name` in `ptx`, in order, read from the PTX text
fn entry_params(ptx: &str, name: &str) -> Result<Vec<Param>> {
    let entry = format!(".entry {name}(");
    let Some(start) = ptx.find(&entry) else {
        bail!("kernel {name} missing from the PTX");
    };
    let start = start + entry.len();
    let end = ptx[start..].find(')').ok_or_else(|| eyre!("kernel {name}: unterminated PTX param list"))?;
    ptx[start..start + end]
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let ty = line.strip_prefix(".param ").and_then(|rest| rest.split_whitespace().next());
            match (ty, line.contains(" .ptr ")) {
                (Some(".u64"), true) => Ok(Param::Ptr),
                (Some(".u64"), false) => Ok(Param::U64),
                (Some(".u32"), false) => Ok(Param::U32),
                (Some(".f32"), false) => Ok(Param::F32),
                _ => Err(eyre!("kernel {name}: unexpected PTX param `{line}`")),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_params_reads_slices_and_scalars() {
        let ptx = ".visible .entry k(\n\t.param .u64 .ptr .global .align 4 k_param_0,\n\t.param .u64 k_param_1,\n\t.param .u32 k_param_2,\n\t.param .f32 k_param_3\n)\n{";
        assert_eq!(entry_params(ptx, "k").unwrap(), vec![Param::Ptr, Param::U64, Param::U32, Param::F32]);
        assert!(entry_params(ptx, "missing").is_err());
    }

    /// Kernel name prefixes of the `inkrs-kernels` modules the sm75 tier leaves out
    /// (`#[cfg(not(feature = "sm75"))]` in its lib.rs): the own 2D convs and the 3D ink engine
    const NOT_ON_SM75: &[&str] = &["conv_", "u3_"];

    /// Every profile tier has its PTX, for its own target, with the same kernels and params as
    /// every other tier, so the launch checks hold whichever tier loads. The one exception is the
    /// [`NOT_ON_SM75`] families, which sm75 leaves out entirely and the other tiers must all carry
    #[test]
    fn tiers_ship_matching_ptx() {
        let tiers: Vec<PtxTier> = PtxTier::ALL.iter().copied().filter(|t| PROFILE.is_compiled_in(*t)).collect();
        assert_eq!(tiers, PTX.iter().map(|(t, _)| *t).collect::<Vec<_>>());
        let kernels = |tier: PtxTier| -> Vec<(String, Vec<Param>)> {
            let ptx = ptx_for(tier).unwrap();
            ptx.match_indices(".entry ")
                .map(|(i, _)| {
                    let name = ptx[i + 7..].split('(').next().unwrap().to_string();
                    let params = entry_params(ptx, &name).unwrap();
                    (name, params)
                })
                .collect()
        };
        let gated = |name: &str| NOT_ON_SM75.iter().any(|p| name.starts_with(p));
        let full_tiers: Vec<PtxTier> = tiers.iter().copied().filter(|t| *t != PtxTier::Sm75).collect();
        let full = kernels(full_tiers[0]);
        for p in NOT_ON_SM75 {
            assert!(full.iter().any(|(n, _)| n.starts_with(p)), "{} has no {p}* kernels", full_tiers[0]);
        }
        for tier in &full_tiers[1..] {
            assert_eq!(kernels(*tier), full, "{tier} kernels differ from {}", full_tiers[0]);
        }
        if tiers.contains(&PtxTier::Sm75) {
            let common: Vec<_> = full.iter().filter(|(n, _)| !gated(n)).cloned().collect();
            assert_eq!(
                kernels(PtxTier::Sm75),
                common,
                "sm75 kernels differ from {} beyond {NOT_ON_SM75:?}",
                full_tiers[0]
            );
        }
    }

    /// The tier each compute capability loads, as the README states it: sm120 from 12.0 up,
    /// sm80 on every other GPU from 8.0 up, sm75 on Turing, nothing below 7.5, and the override
    /// only within that
    #[test]
    fn tier_choice_follows_compute_capability() {
        let cc = ComputeCapability::new;
        for (capability, tier) in [
            (cc(7, 5), PtxTier::Sm75),
            (cc(8, 0), PtxTier::Sm80),
            (cc(8, 6), PtxTier::Sm80),
            (cc(8, 9), PtxTier::Sm80),
            (cc(9, 0), PtxTier::Sm80),
            (cc(10, 0), PtxTier::Sm80),
            (cc(11, 0), PtxTier::Sm80),
            (cc(12, 0), PtxTier::Sm120),
            (cc(12, 1), PtxTier::Sm120),
        ] {
            assert_eq!(PROFILE.select(capability, None).unwrap(), tier, "cc {capability}");
        }

        for capability in [cc(6, 1), cc(7, 0), cc(7, 2)] {
            assert!(PROFILE.select(capability, None).is_err(), "cc {capability} must be refused");
        }

        assert_eq!(PROFILE.select(cc(12, 0), Some(PtxTier::Sm80)).unwrap(), PtxTier::Sm80);
        assert_eq!(PROFILE.select(cc(12, 0), Some(PtxTier::Sm75)).unwrap(), PtxTier::Sm75);
        assert_eq!(PROFILE.select(cc(8, 6), Some(PtxTier::Sm75)).unwrap(), PtxTier::Sm75);
        assert!(PROFILE.select(cc(7, 5), Some(PtxTier::Sm80)).is_err());
        assert!(PROFILE.select(cc(8, 6), Some(PtxTier::Sm120)).is_err());
        assert!(PROFILE.select(cc(12, 0), Some(PtxTier::Sm90)).is_err());
    }

    /// Each family loads the cubin of the tier it selects, or JITs where none fits: 8.7 runs the
    /// sm_86 cubin and 10.3 the sm_100 one, 11.x has none, and a forced lower tier has none either
    #[test]
    fn cubin_choice_follows_compute_capability() {
        let cc = ComputeCapability::new;
        for (capability, arch) in [
            (cc(7, 5), Some(cc(7, 5))),
            (cc(8, 0), Some(cc(8, 0))),
            (cc(8, 6), Some(cc(8, 6))),
            (cc(8, 7), Some(cc(8, 6))),
            (cc(8, 9), Some(cc(8, 9))),
            (cc(9, 0), Some(cc(9, 0))),
            (cc(10, 0), Some(cc(10, 0))),
            (cc(10, 3), Some(cc(10, 0))),
            (cc(11, 0), None),
            (cc(12, 0), Some(cc(12, 0))),
            (cc(12, 1), Some(cc(12, 0))),
        ] {
            let tier = PROFILE.select(capability, None).unwrap();
            assert_eq!(cubin_for(tier, capability).map(|c| c.arch), arch, "cc {capability}");
        }

        assert!(cubin_for(PtxTier::Sm80, cc(12, 0)).is_none());
        assert!(cubin_for(PtxTier::Sm75, cc(8, 6)).is_none());
    }

    /// Each cubin is a CUDA ELF image holding every kernel of the PTX it was compiled from, so a
    /// cubin left over from an older PTX can't load with kernels missing
    #[test]
    fn cubins_hold_their_tier_kernels() {
        const EM_CUDA: u16 = 190;
        for c in CUBINS {
            let name = format!("{} cubin for {}", c.tier, c.arch);
            assert_eq!(&c.bytes[..4], b"\x7fELF", "{name} is not an ELF image");
            assert_eq!(u16::from_le_bytes([c.bytes[18], c.bytes[19]]), EM_CUDA, "{name} is not for CUDA");
            let ptx = ptx_for(c.tier).unwrap();
            for line in ptx.lines() {
                let Some(kernel) = line.trim().strip_prefix(".visible .entry ").and_then(|l| l.split('(').next())
                else {
                    continue;
                };
                let symbol = format!("\0{kernel}\0");
                assert!(c.bytes.windows(symbol.len()).any(|w| w == symbol.as_bytes()), "{name} lacks {kernel}");
            }
        }
    }

    /// The embedded PTX was built from the kernel sources as committed: a kernel edited after the
    /// last build-ptx.sh run would otherwise ship with its old signature or numerics
    #[test]
    #[cfg_attr(windows, ignore = "check-ptx.sh is a bash script; the Linux build runs it")]
    fn embedded_ptx_matches_its_sources() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let out = std::process::Command::new("bash").arg(root.join("scripts/check-ptx.sh")).output().unwrap();
        assert!(
            out.status.success(),
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
