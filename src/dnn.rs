//! Channels-last cuDNN convolutions with FP32 accumulation, and the cuBLAS GEMM behind the
//! transposed convs
//!
//! cudakit's `dnn` is FP32 NCHW 2D only, so this crate drives the legacy cuDNN API through
//! cudarc's raw bindings on its own handles. The villa readers run FP16 with tensor ops; Hecate
//! runs FP32 (TF32 or not) or BF16, as its reference does ([`Dtype`]). No plan carries a bias:
//! the villa convs feed an InstanceNorm, which cancels it, and Hecate's backbone and decoder
//! convs have none

use std::ffi::{CStr, c_int, c_void};
use std::fmt;

use color_eyre::eyre::{Result, bail, ensure, eyre};
use cudarc::cublas::result as blas_result;
use cudarc::cublas::sys as blas;
use cudarc::cudnn::result as dnn_result;
use cudarc::cudnn::sys as dnn;

use crate::gpu::{Buf, Gpu};

pub mod algo_cache;
pub mod bias_act;
pub(crate) mod fused;
pub mod gemm_bf16;

fn check(status: dnn::cudnnStatus_t, what: &str) -> Result<()> {
    if status == dnn::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        return Ok(());
    }
    // SAFETY: cuDNN returns a static C string for every status
    let text = unsafe { CStr::from_ptr(dnn::cudnnGetErrorString(status)) };
    Err(eyre!("{what}: {status:?} ({})", text.to_string_lossy()))
}

/// Our own cuDNN handle on the GPU's stream
///
/// cudakit keeps its handle private to its FP32 wrappers, so the FP16 plans make another
/// after cudakit's symbol preflight has loaded the library
pub struct Dnn(dnn::cudnnHandle_t);

impl Dnn {
    /// A handle bound to `gpu`'s stream
    pub fn new(gpu: &Gpu) -> Result<Self> {
        gpu.inner.context().bind_to_thread()?;
        let handle = dnn_result::create_handle().map_err(|e| eyre!("cudnnCreate: {e:?}"))?;
        // SAFETY: a fresh handle and the GPU's live stream
        unsafe { dnn_result::set_stream(handle, gpu.cu_stream() as dnn::cudaStream_t) }
            .map_err(|e| eyre!("cudnnSetStream: {e:?}"))?;
        Ok(Self(handle))
    }

    /// `cudnnGetVersion`
    pub fn version() -> usize {
        // SAFETY: no arguments
        unsafe { dnn::cudnnGetVersion() }
    }
}

impl Drop for Dnn {
    fn drop(&mut self) {
        // SAFETY: our handle; every plan made from it has dropped first by field order
        let _ = unsafe { dnn_result::destroy_handle(self.0) };
    }
}

/// A plan's data type: storage of x, w and y, and the math cuDNN may use. Accumulation is FP32
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// FP16 data with tensor ops, the villa readers
    Half,
    /// FP32 data. `tf32` lets cuDNN round the inputs to TF32 on tensor cores, as PyTorch does
    /// by default (`torch.backends.cudnn.allow_tf32`); without it cuDNN keeps to FP32 FMAs
    Float { tf32: bool },
    /// BF16 data with tensor ops, PyTorch's bf16 autocast
    Bf16,
}

impl Dtype {
    /// Bytes per element
    pub fn bytes(self) -> usize {
        match self {
            Self::Half | Self::Bf16 => 2,
            Self::Float { .. } => 4,
        }
    }

    fn data(self) -> dnn::cudnnDataType_t {
        match self {
            Self::Half => dnn::cudnnDataType_t::CUDNN_DATA_HALF,
            Self::Float { .. } => dnn::cudnnDataType_t::CUDNN_DATA_FLOAT,
            Self::Bf16 => dnn::cudnnDataType_t::CUDNN_DATA_BFLOAT16,
        }
    }

    fn math(self) -> dnn::cudnnMathType_t {
        match self {
            Self::Half | Self::Bf16 => dnn::cudnnMathType_t::CUDNN_TENSOR_OP_MATH,
            // PyTorch's choice for FP32: default math admits TF32, FMA math forbids it
            Self::Float { tf32: true } => dnn::cudnnMathType_t::CUDNN_DEFAULT_MATH,
            Self::Float { tf32: false } => dnn::cudnnMathType_t::CUDNN_FMA_MATH,
        }
    }
}

/// How a plan chooses its algorithm
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    /// cuDNN's `_v7` heuristic ranking, first supported entry
    Heuristic,
    /// `cudnnFindConvolutionForwardAlgorithmEx`: time every algorithm, take the fastest
    Find,
}

/// A convolution shape: batch, input and output channels, input spatial size (`[D, H, W]`
/// or `[H, W]`), kernel size, padding and stride per spatial axis
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConvShape {
    pub n: i32,
    pub c: i32,
    pub k: i32,
    pub spatial: Vec<i32>,
    pub kernel: Vec<i32>,
    pub pad: Vec<i32>,
    pub stride: Vec<i32>,
}

impl ConvShape {
    /// A square or cubic kernel `size` with "same" padding, as every conv of the model uses
    pub fn same(n: i32, c: i32, k: i32, spatial: &[i32], size: i32, stride: i32) -> Self {
        let dims = spatial.len();
        Self {
            n,
            c,
            k,
            spatial: spatial.to_vec(),
            kernel: vec![size; dims],
            pad: vec![(size - 1) / 2; dims],
            stride: vec![stride; dims],
        }
    }

    /// Output spatial size
    pub fn out_spatial(&self) -> Vec<i32> {
        (0..self.spatial.len())
            .map(|i| (self.spatial[i] + 2 * self.pad[i] - self.kernel[i]) / self.stride[i] + 1)
            .collect()
    }

    /// Elements of the input, weight and output
    pub fn elements(&self) -> (usize, usize, usize) {
        let prod = |v: &[i32]| v.iter().map(|&d| d as usize).product::<usize>();
        (
            self.n as usize * self.c as usize * prod(&self.spatial),
            self.k as usize * self.c as usize * prod(&self.kernel),
            self.n as usize * self.k as usize * prod(&self.out_spatial()),
        )
    }
}

impl fmt::Display for ConvShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "conv{}d n{} {}->{} in{:?} k{:?} s{:?}",
            self.spatial.len(),
            self.n,
            self.c,
            self.k,
            self.spatial,
            self.kernel,
            self.stride
        )
    }
}

/// One algorithm's entry in cuDNN's ranking
#[derive(Clone, Copy, Debug)]
pub struct AlgoPerf {
    pub algo: dnn::cudnnConvolutionFwdAlgo_t,
    pub status: dnn::cudnnStatus_t,
    pub ms: f32,
    pub workspace: usize,
    pub math: dnn::cudnnMathType_t,
    pub deterministic: bool,
}

impl fmt::Display for AlgoPerf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let algo = format!("{:?}", self.algo);
        let math = format!("{:?}", self.math);
        write!(
            f,
            "{} {} {:.3} ms ws {} B{}{}",
            algo.trim_start_matches("CUDNN_CONVOLUTION_FWD_ALGO_"),
            math.trim_start_matches("CUDNN_"),
            self.ms,
            self.workspace,
            if self.deterministic { "" } else { " nondeterministic" },
            if self.status == dnn::cudnnStatus_t::CUDNN_STATUS_SUCCESS { "" } else { " (unsupported)" },
        )
    }
}

/// A planned NHWC/NDHWC convolution: descriptors and the chosen algorithm
pub struct ConvPlan {
    pub shape: ConvShape,
    pub dtype: Dtype,
    pub algo: AlgoPerf,
    /// Every algorithm cuDNN ranked, for the report
    pub ranking: Vec<AlgoPerf>,
    /// The [`Pick::Find`] ranking came from the [`algo_cache`], not a fresh timed find
    pub cached: bool,
    x: dnn::cudnnTensorDescriptor_t,
    w: dnn::cudnnFilterDescriptor_t,
    y: dnn::cudnnTensorDescriptor_t,
    conv: dnn::cudnnConvolutionDescriptor_t,
}

impl Drop for ConvPlan {
    fn drop(&mut self) {
        // SAFETY: descriptors this plan created and owns
        unsafe {
            dnn::cudnnDestroyTensorDescriptor(self.x);
            dnn::cudnnDestroyTensorDescriptor(self.y);
            dnn::cudnnDestroyFilterDescriptor(self.w);
            dnn::cudnnDestroyConvolutionDescriptor(self.conv);
        }
    }
}

impl ConvPlan {
    /// Describes the FP16 `shape` and picks its algorithm. [`Pick::Find`] runs every candidate
    /// on zero-filled scratch buffers, so call it only while planning
    pub fn new(gpu: &Gpu, handle: &Dnn, shape: ConvShape, pick: Pick, max_workspace: usize) -> Result<Self> {
        Self::typed(gpu, handle, shape, Dtype::Half, pick, max_workspace)
    }

    /// [`ConvPlan::new`] for any [`Dtype`]
    pub fn typed(
        gpu: &Gpu,
        handle: &Dnn,
        shape: ConvShape,
        dtype: Dtype,
        pick: Pick,
        max_workspace: usize,
    ) -> Result<Self> {
        let dims = shape.spatial.len();
        ensure!(dims == 2 || dims == 3, "{shape}: only 2D and 3D convolutions");
        ensure!(
            shape.kernel.len() == dims && shape.pad.len() == dims && shape.stride.len() == dims,
            "{shape}: rank mismatch"
        );
        gpu.inner.context().bind_to_thread()?;

        let mut plan = Self {
            shape: shape.clone(),
            dtype,
            algo: AlgoPerf {
                algo: dnn::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
                status: dnn::cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
                ms: 0.0,
                workspace: 0,
                math: dnn::cudnnMathType_t::CUDNN_DEFAULT_MATH,
                deterministic: false,
            },
            ranking: Vec::new(),
            cached: false,
            x: std::ptr::null_mut(),
            w: std::ptr::null_mut(),
            y: std::ptr::null_mut(),
            conv: std::ptr::null_mut(),
        };
        // SAFETY: plain descriptor creation and setup on arrays that outlive each call
        unsafe {
            check(dnn::cudnnCreateTensorDescriptor(&mut plan.x), "create x")?;
            check(dnn::cudnnCreateTensorDescriptor(&mut plan.y), "create y")?;
            check(dnn::cudnnCreateFilterDescriptor(&mut plan.w), "create w")?;
            check(dnn::cudnnCreateConvolutionDescriptor(&mut plan.conv), "create conv")?;

            // cuDNN takes NCHW-ordered dims and the format flag for the memory layout
            let x_dims: Vec<c_int> = [shape.n, shape.c].iter().chain(&shape.spatial).copied().collect();
            check(
                dnn::cudnnSetTensorNdDescriptorEx(
                    plan.x,
                    dnn::cudnnTensorFormat_t::CUDNN_TENSOR_NHWC,
                    dtype.data(),
                    x_dims.len() as c_int,
                    x_dims.as_ptr(),
                ),
                &format!("{shape}: x descriptor"),
            )?;
            let w_dims: Vec<c_int> = [shape.k, shape.c].iter().chain(&shape.kernel).copied().collect();
            check(
                dnn::cudnnSetFilterNdDescriptor(
                    plan.w,
                    dtype.data(),
                    dnn::cudnnTensorFormat_t::CUDNN_TENSOR_NHWC,
                    w_dims.len() as c_int,
                    w_dims.as_ptr(),
                ),
                &format!("{shape}: filter descriptor"),
            )?;
            let dilation = vec![1 as c_int; dims];
            // FLOAT compute is FP32 accumulation for every data type (pseudo-half for HALF)
            check(
                dnn::cudnnSetConvolutionNdDescriptor(
                    plan.conv,
                    dims as c_int,
                    shape.pad.as_ptr(),
                    shape.stride.as_ptr(),
                    dilation.as_ptr(),
                    dnn::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                    dnn::cudnnDataType_t::CUDNN_DATA_FLOAT,
                ),
                &format!("{shape}: conv descriptor"),
            )?;
            check(dnn::cudnnSetConvolutionMathType(plan.conv, dtype.math()), "math type")?;

            let mut out = vec![0 as c_int; dims + 2];
            check(
                dnn::cudnnGetConvolutionNdForwardOutputDim(
                    plan.conv,
                    plan.x,
                    plan.w,
                    out.len() as c_int,
                    out.as_mut_ptr(),
                ),
                "output dims",
            )?;
            let expect: Vec<c_int> = [shape.n, shape.k].iter().copied().chain(shape.out_spatial()).collect();
            ensure!(out == expect, "{shape}: cuDNN output {out:?}, expected {expect:?}");
            check(
                dnn::cudnnSetTensorNdDescriptorEx(
                    plan.y,
                    dnn::cudnnTensorFormat_t::CUDNN_TENSOR_NHWC,
                    dtype.data(),
                    out.len() as c_int,
                    out.as_ptr(),
                ),
                &format!("{shape}: y descriptor"),
            )?;
        }

        plan.ranking = plan.heuristic(handle)?;
        if pick == Pick::Find {
            plan.ranking = plan.find_cached(gpu, handle, max_workspace)?;
        }
        let ok = |p: &&AlgoPerf| p.status == dnn::cudnnStatus_t::CUDNN_STATUS_SUCCESS && p.workspace <= max_workspace;
        // FP16 plans take the best tensor-op algorithm when there is one. cudnnFind's one-shot
        // timing ranked a non-tensor-op algorithm first for reader-v2's enc.3.0.skip (1x1, 128->256
        // at 16x16), which then ran at 0.77 µs per tile against a 0.11 µs floor in the forward
        let tensor = |p: &&AlgoPerf| {
            dtype != Dtype::Half
                || matches!(
                    p.math,
                    dnn::cudnnMathType_t::CUDNN_TENSOR_OP_MATH
                        | dnn::cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
                )
        };
        let pick = plan.ranking.iter().filter(ok).find(tensor).or_else(|| plan.ranking.iter().find(ok));
        plan.algo = *pick.ok_or_else(|| {
            eyre!("{shape}: no supported cuDNN algorithm within {max_workspace} B: {:?}", plan.ranking)
        })?;
        Ok(plan)
    }

    fn heuristic(&self, handle: &Dnn) -> Result<Vec<AlgoPerf>> {
        let mut perf = [zero_perf(); 8];
        let mut count = 0;
        // SAFETY: the descriptors are set and `perf` holds the requested count
        unsafe {
            check(
                dnn::cudnnGetConvolutionForwardAlgorithm_v7(
                    handle.0,
                    self.x,
                    self.w,
                    self.conv,
                    self.y,
                    8,
                    &mut count,
                    perf.as_mut_ptr(),
                ),
                &format!("{}: heuristic", self.shape),
            )?;
        }
        Ok(perf[..count as usize].iter().map(to_perf).collect())
    }

    /// [`ConvPlan::find`] through the [`algo_cache`]: a ranking stored for this plan on this setup
    /// is replayed when cuDNN still accepts its choice, else a fresh find runs and is stored
    fn find_cached(&mut self, gpu: &Gpu, handle: &Dnn, max_workspace: usize) -> Result<Vec<AlgoPerf>> {
        let entry = algo_cache::Entry::new(gpu, &self.shape, self.dtype, max_workspace);
        if let Some(ranking) = entry.as_ref().and_then(|e| e.load()).and_then(|r| self.replay(handle, r, max_workspace))
        {
            self.cached = true;
            return Ok(ranking);
        }

        let ranking = self.find(gpu, handle, max_workspace)?;
        if let Some(entry) = entry {
            entry.store(&ranking);
        }
        Ok(ranking)
    }

    /// A stored ranking whose choice, the first supported entry within `max_workspace`, cuDNN
    /// still runs within `max_workspace`, with that choice's workspace as cuDNN now sizes it
    fn replay(&self, handle: &Dnn, mut ranking: Vec<AlgoPerf>, max_workspace: usize) -> Option<Vec<AlgoPerf>> {
        let ok = |p: &AlgoPerf| p.status == dnn::cudnnStatus_t::CUDNN_STATUS_SUCCESS && p.workspace <= max_workspace;
        let chosen = ranking.iter_mut().find(|p| ok(p))?;
        let mut bytes = 0;
        // SAFETY: the descriptors are set; cuDNN writes one size
        let status = unsafe {
            dnn::cudnnGetConvolutionForwardWorkspaceSize(
                handle.0,
                self.x,
                self.w,
                self.conv,
                self.y,
                chosen.algo,
                &mut bytes,
            )
        };
        if status != dnn::cudnnStatus_t::CUDNN_STATUS_SUCCESS || bytes > max_workspace {
            return None;
        }

        // never less than the find measured, so the run's workspace check holds either way
        chosen.workspace = chosen.workspace.max(bytes);
        Some(ranking)
    }

    fn find(&self, gpu: &Gpu, handle: &Dnn, max_workspace: usize) -> Result<Vec<AlgoPerf>> {
        let (xe, we, ye) = self.shape.elements();
        let b = self.dtype.bytes();
        let x = gpu.alloc_fill(xe * b, 0)?;
        let w = gpu.alloc_fill(we * b, 0)?;
        let y = gpu.alloc(ye * b)?;
        let ws_bytes = self
            .ranking
            .iter()
            .filter(|p| p.status == dnn::cudnnStatus_t::CUDNN_STATUS_SUCCESS)
            .map(|p| p.workspace)
            .filter(|&b| b <= max_workspace)
            .max()
            .unwrap_or(0);
        let ws = gpu.alloc(ws_bytes)?;
        gpu.sync()?;
        let mut perf = [zero_perf(); 8];
        let mut count = 0;
        // SAFETY: buffers sized for the shape, and the call is synchronous
        unsafe {
            check(
                dnn::cudnnFindConvolutionForwardAlgorithmEx(
                    handle.0,
                    self.x,
                    x.ptr() as *const c_void,
                    self.w,
                    w.ptr() as *const c_void,
                    self.conv,
                    self.y,
                    y.ptr() as *mut c_void,
                    8,
                    &mut count,
                    perf.as_mut_ptr(),
                    ws.ptr() as *mut c_void,
                    ws_bytes,
                ),
                &format!("{}: find", self.shape),
            )?;
        }
        gpu.sync()?;
        Ok(perf[..count as usize].iter().map(to_perf).collect())
    }

    /// Workspace bytes of the chosen algorithm
    pub fn workspace(&self) -> usize {
        self.algo.workspace
    }

    /// Queues `y = conv(x, w)` on the handle's stream
    ///
    /// `x`, `w` and `y` are channels-last (`[N, (D,) H, W, C]`, `[K, (D,) R, S, C]`,
    /// `[N, (D,) H, W, K]`) in the plan's [`Dtype`]; the sizes are checked against the shape
    pub fn run(&self, handle: &Dnn, x: (u64, usize), w: (u64, usize), y: (u64, usize), ws: Option<&Buf>) -> Result<()> {
        let (xe, we, ye) = self.shape.elements();
        let b = self.dtype.bytes();
        ensure!(
            x.1 >= xe * b && w.1 >= we * b && y.1 >= ye * b,
            "{}: buffers too small ({} {} {} bytes)",
            self.shape,
            x.1,
            w.1,
            y.1
        );
        let (ws_ptr, ws_bytes) = match ws {
            Some(b) => (b.ptr(), b.bytes()),
            None => (0, 0),
        };
        if ws_bytes < self.algo.workspace {
            bail!("{}: workspace {ws_bytes} B, the algorithm needs {}", self.shape, self.algo.workspace);
        }
        let (alpha, beta) = (1f32, 0f32);
        // SAFETY: buffer sizes checked above; cuDNN reads the scalars before returning
        unsafe {
            check(
                dnn::cudnnConvolutionForward(
                    handle.0,
                    (&alpha as *const f32).cast(),
                    self.x,
                    x.0 as *const c_void,
                    self.w,
                    w.0 as *const c_void,
                    self.conv,
                    self.algo.algo,
                    ws_ptr as *mut c_void,
                    ws_bytes,
                    (&beta as *const f32).cast(),
                    self.y,
                    y.0 as *mut c_void,
                ),
                &format!("{}: forward", self.shape),
            )
        }
    }
}

fn zero_perf() -> dnn::cudnnConvolutionFwdAlgoPerf_t {
    // SAFETY: a plain C struct of integers, floats and enums whose zero values are valid
    unsafe { std::mem::zeroed() }
}

fn to_perf(p: &dnn::cudnnConvolutionFwdAlgoPerf_t) -> AlgoPerf {
    AlgoPerf {
        algo: p.algo,
        status: p.status,
        ms: p.time,
        workspace: p.memory,
        math: p.mathType,
        deterministic: p.determinism == dnn::cudnnDeterminism_t::CUDNN_DETERMINISTIC,
    }
}

/// Our own cuBLAS handle on the GPU's stream, for FP16 GEMMs with FP32 accumulation
pub struct Blas(blas::cublasHandle_t);

impl Blas {
    /// A handle bound to `gpu`'s stream
    pub fn new(gpu: &Gpu) -> Result<Self> {
        gpu.inner.context().bind_to_thread()?;
        let handle = blas_result::create_handle().map_err(|e| eyre!("cublasCreate: {e:?}"))?;
        // SAFETY: a fresh handle and the GPU's live stream
        unsafe { blas_result::set_stream(handle, gpu.cu_stream() as blas::cudaStream_t) }
            .map_err(|e| eyre!("cublasSetStream: {e:?}"))?;
        Ok(Self(handle))
    }

    /// Row-major FP16 `y[m, n] = x[m, k] @ w[k, n]`, FP32 accumulation, FP16 output
    pub fn gemm_f16(
        &self,
        (m, n, k): (usize, usize, usize),
        x: (u64, usize),
        w: (u64, usize),
        y: (u64, usize),
    ) -> Result<()> {
        ensure!(x.1 >= m * k * 2 && w.1 >= k * n * 2 && y.1 >= m * n * 2, "gemm {m}x{n}x{k}: buffers too small");
        let to = |v: usize| c_int::try_from(v).map_err(|_| eyre!("gemm size {v} overflows int"));
        let (alpha, beta) = (1f32, 0f32);
        // column-major cuBLAS sees the row-major product transposed: y^T = w^T x^T
        // SAFETY: sizes checked; host scalars read before the call returns
        unsafe {
            blas::cublasGemmEx(
                self.0,
                blas::cublasOperation_t::CUBLAS_OP_N,
                blas::cublasOperation_t::CUBLAS_OP_N,
                to(n)?,
                to(m)?,
                to(k)?,
                (&alpha as *const f32).cast(),
                w.0 as *const c_void,
                blas::cudaDataType::CUDA_R_16F,
                to(n)?,
                x.0 as *const c_void,
                blas::cudaDataType::CUDA_R_16F,
                to(k)?,
                (&beta as *const f32).cast(),
                y.0 as *mut c_void,
                blas::cudaDataType::CUDA_R_16F,
                to(n)?,
                blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .result()
            .map_err(|e| eyre!("cublasGemmEx {m}x{n}x{k}: {e:?}"))
        }
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        // SAFETY: our handle
        let _ = unsafe { blas_result::destroy_handle(self.0) };
    }
}
