//! The Metal backend for Apple silicon: the same engine as the CUDA one, with MSL kernels in place of cuDNN, cuBLAS
//! and the PTX kernels
//!
//! - [`gpu`]: device, queue, kernel library and dispatch
//! - [`model`]: `hybrid_3d2d` planned at a fixed batch, its weights and activations
//! - [`engine`]: the resident pipeline, with the CUDA engine's public API
//! - [`coreml`]: experimental Core ML offload of reader-v2's 2D body to the Neural Engine and CPU
//! - [`hecate`]: Hecate 9.6 um, with the CUDA Hecate engine's public API
//! - [`v8in`]: v8-in (ink8um) and its A-hn fine-tunes, with the CUDA v8-in engine's public API

pub mod coreml;
pub mod engine;
pub mod gpu;
pub mod hecate;
pub mod model;
pub mod v8in;

/// The CUDA backend's algorithm choice, kept so callers build the same options on both backends; Metal has one
/// implementation per layer and ignores it
pub mod dnn {
    /// How a plan chooses its algorithm (cuDNN only)
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Pick {
        Heuristic,
        Find,
    }
}

#[cfg(test)]
mod tests {
    use super::gpu::Gpu;
    use crate::fast::ReaderPrecision;

    /// Both libraries compile
    #[test]
    fn kernels_compile() {
        for p in [ReaderPrecision::Fp16, ReaderPrecision::Fp32] {
            Gpu::open_at(0, p).unwrap().ready().unwrap();
        }
    }

    // a profiling window synchronizes mid-pass; the flush that fences a map must still return a fence, and that
    // fence must cover the work queued before the synchronize
    #[test]
    fn flush_after_synchronize_still_fences() {
        let gpu = Gpu::open(0).unwrap();
        let buf = gpu.alloc_fill(4096, 0xFF).unwrap();
        gpu.dispatch_1d("zero_u32", 1024, &[buf.arg(), 1024usize.into()]).unwrap();
        gpu.synchronize().unwrap();
        let fence = gpu.flush().unwrap().expect("a fence after synchronize");
        fence.wait().unwrap();
        let mut out = vec![1u32; 1024];
        gpu.download(&buf, 0, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0));
    }
}
