//! inkrs: GPU-resident inference for villa's ink and surface models: the flat ink readers
//! (`hybrid_3d2d-seed42`, `reader-v2`), Hecate 9.6 um ([`hecate`]), villa's 3D U-Net ink models ([`unet3d`]),
//! nnU-Net ResEnc-L surface models as `vesuvius.predict` runs them ([`nnunet`]) and the v8-in family ([`v8in`])
//!
//! The `inkrs` binary and the Python bindings (`python/`) share this library, the command line included
//! ([`cli`]). [`engine::Engine`] holds the model on the GPU; [`engine::Engine::run_pass`] runs one volume's
//! maps, writing villa's TIFFs or returning the images
//!
//! On macOS the engine runs on Metal (`metal`) instead of CUDA, behind the same [`engine`] API, for the
//! villa readers, Hecate and v8-in; the 3D ink and nnU-Net engines, 8-fold TTA and the own-conv backend are
//! CUDA-only

#[cfg(not(target_os = "macos"))]
pub mod checks;
pub mod cli;
#[cfg(not(target_os = "macos"))]
pub mod conv;
#[cfg(windows)]
pub mod dll;
#[cfg(not(target_os = "macos"))]
pub mod dnn;
#[cfg(not(target_os = "macos"))]
pub mod engine;
pub mod error;
pub mod fast;
#[cfg(not(target_os = "macos"))]
pub mod gpu;
pub mod hecate;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(target_os = "macos")]
pub use metal::{dnn, engine};
#[cfg(not(target_os = "macos"))]
pub mod model;
pub mod nnunet;
#[cfg(not(target_os = "macos"))]
pub mod probe;
pub mod sched;
pub mod tiff;
pub mod tta;
pub mod unet3d;
pub mod v8in;
pub mod weights;
pub mod zarr;
