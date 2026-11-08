//! Helpers shared by the kernels: libdevice math, FP16 packing and the FP16 tensor-core product
//!
//! The `sm75` feature builds the Turing tier: sm_75 has neither `cvt.rn.f16x2.f32` nor the
//! `m16n8k16` FP16 `mma` shape, so [`pack_f16`], [`narrow`] and [`mma_m16n8k16_f32_f16`] fall back
//! to `cvt.rn.f16.f32` and two `m16n8k8` products there, and [`add_rn_f32`] and [`mul_rn_f32`]
//! are written as PTX because cuda-oxide refuses its own for sm_75. Without the feature the sm80 and sm120
//! tiers compile exactly the code they did before the feature existed

#[cfg(not(feature = "sm75"))]
use cuda_device::convert::cvt_f16x2_f32;
use cuda_device::convert::{cvt_f32x2_f16x2, cvt_f32_f16x2_lo};
#[cfg(not(feature = "sm75"))]
pub use cuda_device::float::{add_rn_f32, mul_rn_f32};
#[cfg(not(feature = "sm75"))]
pub use cuda_device::wmma::mma_m16n8k16_f32_f16;

/// `add.rn.f32`: a float add the compiler may not fuse into an FMA
// cuda-oxide 8d8254869's intrinsic catalog lists its add_rn_f32 and mul_rn_f32 as sm_80+, though PTX
// has had both instructions since sm_20, so the sm_75 tier writes them itself
#[cfg(feature = "sm75")]
#[inline(always)]
pub fn add_rn_f32(a: f32, b: f32) -> f32 {
    let r: f32;
    // SAFETY: a register-only add with no memory access or warp participation
    unsafe {
        cuda_device::ptx_asm!("add.rn.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only));
    }
    r
}

/// `mul.rn.f32`: a float multiply the compiler may not fuse into an FMA, as [`add_rn_f32`]
#[cfg(feature = "sm75")]
#[inline(always)]
pub fn mul_rn_f32(a: f32, b: f32) -> f32 {
    let r: f32;
    // SAFETY: a register-only multiply with no memory access or warp participation
    unsafe {
        cuda_device::ptx_asm!("mul.rn.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only));
    }
    r
}

// the CUDA C++ math library calls these same libdevice functions, so results match what
// PyTorch's nvcc-built kernels compute with `std::exp` and `::sqrtf`
unsafe extern "C" {
    fn __nv_expf(x: f32) -> f32;
    fn __nv_sqrtf(x: f32) -> f32;
    fn __nv_fmaxf(x: f32, y: f32) -> f32;
    fn __nv_fminf(x: f32, y: f32) -> f32;
    fn __nv_dsqrt_rn(x: f64) -> f64;
}

/// Math functions under their CUDA C++ names
pub mod math {
    #[inline(always)]
    pub fn expf(x: f32) -> f32 {
        unsafe { super::__nv_expf(x) }
    }

    #[inline(always)]
    pub fn sqrtf(x: f32) -> f32 {
        unsafe { super::__nv_sqrtf(x) }
    }

    #[inline(always)]
    pub fn sqrt(x: f64) -> f64 {
        unsafe { super::__nv_dsqrt_rn(x) }
    }

    /// One `max.f32`; `f32::max` lowers to a compare and a select on some routes
    #[inline(always)]
    pub fn fmaxf(x: f32, y: f32) -> f32 {
        unsafe { super::__nv_fmaxf(x, y) }
    }

    /// One `min.f32`, as [`fmaxf`]
    #[inline(always)]
    pub fn fminf(x: f32, y: f32) -> f32 {
        unsafe { super::__nv_fminf(x, y) }
    }
}

/// Two floats rounded to nearest-even FP16, `lo` in the low half
#[cfg(not(feature = "sm75"))]
#[inline(always)]
pub fn pack_f16(lo: f32, hi: f32) -> u32 {
    cvt_f16x2_f32(lo, hi)
}

/// Two floats rounded to nearest-even FP16, `lo` in the low half
// two scalar conversions round each value exactly as the packed one does
#[cfg(feature = "sm75")]
#[inline(always)]
pub fn pack_f16(lo: f32, hi: f32) -> u32 {
    narrow(lo) as u32 | ((narrow(hi) as u32) << 16)
}

/// The two FP16 halves of a packed pair as floats, low half first
#[inline(always)]
pub fn unpack_f16(packed: u32) -> (f32, f32) {
    cvt_f32x2_f16x2(packed)
}

/// One FP16 value, as raw bits, widened to float
#[inline(always)]
pub fn widen(bits: u16) -> f32 {
    cvt_f32_f16x2_lo(bits as u32)
}

/// One float rounded to nearest-even FP16, as raw bits
#[cfg(not(feature = "sm75"))]
#[inline(always)]
pub fn narrow(x: f32) -> u16 {
    cvt_f16x2_f32(x, 0.0) as u16
}

/// One float rounded to nearest-even FP16, as raw bits
// cuda-oxide has no scalar f32 -> f16 intrinsic, and sm_75 lacks the packed one
#[cfg(feature = "sm75")]
#[inline(always)]
pub fn narrow(x: f32) -> u16 {
    let bits: u16;
    // SAFETY: a register-only conversion with no memory access or warp participation
    unsafe {
        cuda_device::ptx_asm!("cvt.rn.f16.f32 %0, %1;", out("=h") bits, in("f") x, options(register_only));
    }
    bits
}

/// `mma.m16n8k16` (FP16 in, float32 accumulate) as two `mma.m16n8k8`, the largest FP16 shape sm_75 has
///
/// The k16 A fragment's registers 0 and 1 hold columns 0..8 and registers 2 and 3 columns 8..16, each
/// pair laid out as k8's A fragment; B's register 0 holds rows 0..8 and register 1 rows 8..16, as k8's
/// B. Rows 8..16 are added to the float32 sum of rows 0..8; PTX leaves the order and rounding of an `mma`'s
/// accumulation unspecified, so sums can differ from the k16 product's in the last bits
///
/// # Safety
/// As for `cuda_device::wmma::mma_m16n8k16_f32_f16`: all 32 lanes run it with this lane's fragments
#[cfg(feature = "sm75")]
#[inline(always)]
pub unsafe fn mma_m16n8k16_f32_f16(c: [f32; 4], a: [u32; 4], b: [u32; 2]) -> [f32; 4] {
    use cuda_device::wmma::mma_m16n8k8_f32_f16;
    // SAFETY: the caller's contract, split along k
    unsafe { mma_m16n8k8_f32_f16(mma_m16n8k8_f32_f16(c, [a[0], a[1]], b[0]), [a[2], a[3]], b[1]) }
}

/// `x` rounded to FP16 and widened back: the value a PyTorch FP16 tensor would hold
#[inline(always)]
pub fn r16(x: f32) -> f32 {
    widen(narrow(x))
}

/// PyTorch's in-place `LeakyReLU(0.01)` on an FP16 value: computed in float, rounded again
#[inline(always)]
pub fn leaky16(x: f32) -> f32 {
    if x > 0.0 { x } else { r16(x * 0.01) }
}
