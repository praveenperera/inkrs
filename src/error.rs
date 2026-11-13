//! The one error a caller can act on: input, a model or a GPU outside what this engine implements, where a caller
//! such as villa falls back to its own PyTorch path instead of failing. Every other error is a real failure
//!
//! Limits fail through [`ensure_supported!`]; [`is_unsupported`] reads a report's chain, including cudakit's and
//! the driver's ways of saying the embedded PTX cannot run on this GPU, so callers never copy the arch list

use std::fmt;

use color_eyre::eyre::Report;
#[cfg(not(target_os = "macos"))]
use cudarc::driver::DriverError;
#[cfg(not(target_os = "macos"))]
use cudarc::driver::sys::CUresult;

/// Input, model or GPU outside what this engine implements
#[derive(Debug)]
pub struct Unsupported(pub String);

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Unsupported {}

/// A CUDA library the engine would open from a place anyone who controls the working folder can write to: the
/// current directory or a relative `PATH` entry (Windows only, see `dll`), or a non-driver file taken for the driver
#[derive(Debug)]
pub struct UntrustedLibrary(pub String);

impl UntrustedLibrary {
    /// `name` found at `file`, which would be loaded as `what`
    pub fn new(name: &str, file: &std::path::Path, what: &str) -> Self {
        Self(format!(
            "refusing to load {what}: {name} would be loaded from {}, which the default DLL search reaches before \
             the trusted copy; remove it, or run inkrs from another folder",
            file.display()
        ))
    }

    /// `name` already loaded in the process from `file`, somewhere other than where `what` belongs
    pub fn loaded(name: &str, file: &std::path::Path, what: &str) -> Self {
        Self(format!(
            "refusing to load {what}: {name} is already loaded from {}, and a load by name would reuse it",
            file.display()
        ))
    }

    /// `earlier`, which cudarc tries before the copy already loaded from `loaded`, so it would load a second copy
    /// and use that one
    pub fn shadowed(earlier: &std::path::Path, loaded: &std::path::Path) -> Self {
        Self(format!(
            "refusing to load a second copy of a CUDA library: {} would be opened before {}, which is already loaded; \
             remove one of them from the search path",
            earlier.display(),
            loaded.display()
        ))
    }
}

impl fmt::Display for UntrustedLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UntrustedLibrary {}

/// Whether `err` is a CUDA library failure: cudakit's (the driver, cuDNN or cuBLAS missing or unusable) or a
/// library refused as [`UntrustedLibrary`]
pub fn is_library_error(err: &Report) -> bool {
    err.chain().any(|cause| is_cudakit_error(cause) || cause.is::<UntrustedLibrary>())
}

#[cfg(not(target_os = "macos"))]
fn is_cudakit_error(cause: &(dyn std::error::Error + 'static)) -> bool {
    cause.is::<cudakit::Error>()
}

// macOS builds without cudakit, so no cause can be one of its errors
#[cfg(target_os = "macos")]
fn is_cudakit_error(_: &(dyn std::error::Error + 'static)) -> bool {
    false
}

/// `ensure!` that fails with [`Unsupported`]
#[macro_export]
macro_rules! ensure_supported {
    ($cond:expr, $($arg:tt)+) => {
        if !$cond {
            return Err($crate::error::Unsupported(format!($($arg)+)).into());
        }
    };
}

/// Whether `err` means "not supported here" rather than a failure: an [`Unsupported`] limit, a GPU whose compute
/// capability the embedded PTX does not serve, or a driver that cannot load that PTX
pub fn is_unsupported(err: &Report) -> bool {
    err.chain().any(|cause| cause.is::<Unsupported>() || is_unsupported_cuda_error(cause))
}

/// Whether `cause` is cudakit or the driver saying the embedded PTX cannot run on this GPU
#[cfg(not(target_os = "macos"))]
fn is_unsupported_cuda_error(cause: &(dyn std::error::Error + 'static)) -> bool {
    if let Some(e) = cause.downcast_ref::<cudakit::Error>() {
        return matches!(
            e,
            cudakit::Error::TierNotCompiledIn { .. }
                | cudakit::Error::AreaTierNotCompiledIn { .. }
                | cudakit::Error::UnsupportedDevice { .. }
                | cudakit::Error::PtxTierAboveDevice { .. }
        );
    }

    if let Some(DriverError(code)) = cause.downcast_ref::<DriverError>() {
        return matches!(code, CUresult::CUDA_ERROR_NO_BINARY_FOR_GPU | CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION);
    }

    false
}

// macOS builds without cudakit or the CUDA driver, so no cause can be one of their errors
#[cfg(target_os = "macos")]
fn is_unsupported_cuda_error(_: &(dyn std::error::Error + 'static)) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre::{WrapErr, eyre};

    #[cfg(not(target_os = "macos"))]
    use super::{CUresult, DriverError};
    use super::{Unsupported, is_unsupported};

    fn limit(ok: bool) -> color_eyre::Result<()> {
        ensure_supported!(ok, "volume {}x{} is smaller than a tile", 100, 128);
        Ok(())
    }

    #[test]
    fn unsupported_is_found_under_context() {
        let err = limit(false).wrap_err("pass 0").wrap_err("run").unwrap_err();
        assert!(is_unsupported(&err));
        assert_eq!(err.root_cause().to_string(), "volume 100x128 is smaller than a tile");
        assert!(limit(true).is_ok());
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn ptx_the_driver_cannot_load_is_unsupported() {
        for code in [CUresult::CUDA_ERROR_NO_BINARY_FOR_GPU, CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION] {
            let err = Err::<(), _>(DriverError(code)).wrap_err("load module").unwrap_err();
            assert!(is_unsupported(&err), "{code:?}");
        }
    }

    #[test]
    fn other_failures_are_not_unsupported() {
        #[cfg(not(target_os = "macos"))]
        {
            let oom = Err::<(), _>(DriverError(CUresult::CUDA_ERROR_OUT_OF_MEMORY)).wrap_err("alloc").unwrap_err();
            assert!(!is_unsupported(&oom));
        }
        assert!(!is_unsupported(&eyre!("weights file truncated")));
        assert!(is_unsupported(&Unsupported("x".into()).into()));
    }
}
