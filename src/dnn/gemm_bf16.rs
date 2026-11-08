//! bf16 GEMMs with float32 accumulation on [`Blas`], for the 3D U-Net's k2 s2 transposed convs

use std::ffi::{c_int, c_void};

use color_eyre::eyre::{Result, ensure, eyre};
use cudarc::cublas::sys as blas;

use super::Blas;

impl Blas {
    /// Row-major bf16 `y[m, n] = x[m, k] @ w[k, n]`, float32 accumulation, bf16 output
    pub fn gemm_bf16(
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
                blas::cudaDataType::CUDA_R_16BF,
                to(n)?,
                x.0 as *const c_void,
                blas::cudaDataType::CUDA_R_16BF,
                to(k)?,
                (&beta as *const f32).cast(),
                y.0 as *mut c_void,
                blas::cudaDataType::CUDA_R_16BF,
                to(n)?,
                blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .result()
            .map_err(|e| eyre!("cublasGemmEx bf16 {m}x{n}x{k}: {e:?}"))
        }
    }
}
