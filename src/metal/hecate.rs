//! Hecate 9.6 um on Metal: [`model`] runs the network with the kernels of `hecate.metal`; [`engine`] tiles, blends
//! and writes as the CUDA engine does, behind the same API

pub mod engine;
pub mod model;
