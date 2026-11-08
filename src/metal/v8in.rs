//! v8-in on Metal: [`model`] runs the network with the kernels of `v8in.metal`; [`engine`] schedules, blends and
//! checks as the CUDA engine does, behind the same API. `crate::v8in` re-exports both on macOS

pub mod engine;
pub mod model;
