//! cuda-oxide device kernels for inkrs
//!
//! `scripts/build-ptx.sh` compiles this crate for sm_75 (with the `sm75` feature), sm_80 and sm_120
//! and writes `ptx/inkrs.sm{75,80,120}.ptx`, which the inkrs crate embeds and loads by kernel name.
//! Buffers are `&[T]` inputs and `DisjointSlice<T>` outputs, each two PTX params (address, then length in elements); scalars
//! are `u32`, `i32` or `f32`. The host checks every launch against the PTX param list.
//!
//! Activations are FP16 channels-last (NHWC and NDHWC), carried as `u16` bits or `u32` pairs,
//! because cuDNN runs every convolution in that layout. These kernels are the glue around the
//! cuDNN convolutions and cuBLAS GEMMs: tile gather and normalisation, InstanceNorm, the depth
//! attention pool, pooling, concatenation, the head and the blend. Hecate ([`hecate`]) instead
//! runs float32 or bf16 activations, the two precisions its reference offers. The `*384` modules
//! are the 128-edge kernels at Reader V3's 384-pixel tile edge, kept separate so the 128-edge PTX
//! does not change

#![no_std]

pub mod blend;
pub mod blend384;
pub mod common;
// the own 2D convs are built around m16n8k16 FP16 mma with sm80 tile and pipeline choices and were
// never timed or gated on Turing, so the sm_75 tier leaves them out and the host runs every 2D conv
// on cuDNN there (src/conv.rs ConvBackend::resolve)
#[cfg(not(feature = "sm75"))]
pub mod conv2d;
pub mod hecate;
pub mod inorm;
pub mod layout;
pub mod stem;
pub mod stem_conv1;
pub mod stem_conv1_384;
pub mod stem_conv2;
pub mod stem_conv2_384;
pub mod tiles;
pub mod tiles384;
pub mod tta;
// the 3D ink engine's glue rounds through bf16 for cuDNN's bf16 convs, which need compute
// capability 8.0, so the sm_75 tier leaves it out and the host refuses ink3d there
// (src/unet3d.rs check_device)
#[cfg(not(feature = "sm75"))]
pub mod unet3d;
pub mod v8in;
