//! InstanceNorm over channels-last FP16 activations `[B, S, C]` (S = H*W, or D*H*W in 3D)
//!
//! PyTorch runs `InstanceNorm` under FP16 autocast on the FP16 input with FP32 statistics and
//! writes FP16; LeakyReLU and the residual add then each round to FP16 again. These kernels keep
//! that sequence: statistics in three steps ([`in_partial`], [`in_finalize`]) and one fused
//! [`in_apply`] for normalise, affine, optional residual and optional LeakyReLU.
//!
//! Partial sums are taken around a per-(sample, channel) shift, the sample's first value, so
//! `E[x^2] - E[x]^2` does not cancel when a channel's mean is large against its spread (the
//! convolutions here drop the bias that InstanceNorm removes anyway)

use core::ptr::addr_of_mut;

use cuda_device::thread::{blockDim_x, blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread};

use crate::common::{leaky16, pack_f16, r16, unpack_f16, widen};

/// Partial shifted sums of one spatial chunk for every channel
///
/// `x` is `[B, rows, cols]` FP16 as `u32` pairs, `cols` even. Grid `(chunks, B)`, block
/// `cols / 2 * lanes` threads (at most 256): thread `t` owns channel pair `t % (cols / 2)` and
/// starts at row `t / (cols / 2)` of the chunk, stepping by `lanes`. Writes
/// `partial[b, chunk, c] = [sum(x - x0), sum((x - x0)^2)]` with `x0 = x[b, 0, c]`
#[kernel]
pub fn in_partial(
    mut partial: DisjointSlice<[f32; 2]>,
    x: &[u32],
    rows: u32,
    cols: u32,
    lanes: u32,
    chunk_rows: u32,
) {
    static mut ACC: SharedArray<[f32; 4], 256> = SharedArray::UNINIT;
    let (chunk, b, tid) = (blockIdx_x(), blockIdx_y(), threadIdx_x());
    let pairs = cols / 2;
    let chunks = (rows + chunk_rows - 1) / chunk_rows.max(1);
    // block-uniform, so a block that fails it never reaches a barrier
    if pairs == 0
        || blockDim_x() != pairs * lanes
        || blockDim_x() > 256
        || ((b + 1) * rows * pairs) as usize > x.len()
        || (((b * chunks + chunk + 1) * cols) as usize) > partial.len()
    {
        return;
    }
    let (cp, lane) = (tid % pairs, tid / pairs);
    let sample = b * rows * pairs;
    let (s0, s1) = unpack_f16(x[(sample + cp) as usize]);

    let (mut a0, mut a1, mut q0, mut q1) = (0f32, 0f32, 0f32, 0f32);
    let end = ((chunk + 1) * chunk_rows).min(rows);
    let mut r = chunk * chunk_rows + lane;
    while r < end {
        let (v0, v1) = unpack_f16(x[(sample + r * pairs + cp) as usize]);
        let (d0, d1) = (v0 - s0, v1 - s1);
        a0 += d0;
        a1 += d1;
        q0 += d0 * d0;
        q1 += d1 * d1;
        r += lanes;
    }
    let acc = addr_of_mut!(ACC) as *mut [f32; 4];
    // SAFETY: each thread writes its own slot, read only after the barrier
    unsafe { *acc.add(tid as usize) = [a0, a1, q0, q1] };
    sync_threads();
    if lane == 0 {
        let (mut t0, mut t1, mut u0, mut u1) = (0f32, 0f32, 0f32, 0f32);
        let mut l = 0;
        while l < lanes {
            // SAFETY: `l * pairs + cp < blockDim <= 256`
            let [p0, p1, p2, p3] = unsafe { *acc.add((l * pairs + cp) as usize) };
            t0 += p0;
            t1 += p1;
            u0 += p2;
            u1 += p3;
            l += 1;
        }
        let o = ((b * chunks + chunk) * cols + 2 * cp) as usize;
        // SAFETY: `o + 1 < partial.len()` by the check above, and only thread `cp` of this
        // block writes channels `2cp, 2cp + 1` of this chunk
        unsafe {
            *partial.get_unchecked_mut(o) = [t0, u0];
            *partial.get_unchecked_mut(o + 1) = [t1, u1];
        }
    }
}

/// Per (sample, channel) `[mean, invstd]` from the partial sums, one thread each, in float64
///
/// `stats` is `[B, cols]`; `invstd = 1 / sqrt(var + eps)` with the biased variance, in float32
/// as PyTorch's FP32 statistics compute it
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn in_finalize(
    mut stats: DisjointSlice<[f32; 2]>,
    partial: &[[f32; 2]],
    x: &[u16],
    rows: u32,
    cols: u32,
    chunks: u32,
    eps: f32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (b, c) = (i / cols, i % cols);
    let Some(mut out) = stats.element_thread32(t) else { return };
    let shift = widen(x[(b * rows * cols + c) as usize]) as f64;
    let (mut s, mut q) = (0f64, 0f64);
    let mut k = 0;
    while k < chunks {
        let [p, pq] = partial[((b * chunks + k) * cols + c) as usize];
        s += p as f64;
        q += pq as f64;
        k += 1;
    }
    let n = rows as f64;
    let m = s / n;
    let var = q / n - m * m;
    let var = if var > 0.0 { var as f32 } else { 0.0 };
    out.write([(shift + m) as f32, 1.0 / crate::common::math::sqrtf(var + eps)]);
}

/// `x = leaky?(r16(r16(norm(x) * w + bias) + residual?))` in place, two channels per thread
///
/// `x` and `residual` are `[B, rows, cols]` FP16 pairs; an empty `residual` skips the add and
/// `leaky == 0` skips the LeakyReLU. `stats` is [`in_finalize`]'s, `w` and `bias` the affine
/// parameters in float32
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn in_apply(
    mut x: DisjointSlice<u32>,
    residual: &[u32],
    stats: &[[f32; 2]],
    w: &[f32],
    bias: &[f32],
    rows: u32,
    cols: u32,
    leaky: u32,
) {
    let t = thread::index_1d_u32(lc);
    let e = t.get();
    let pairs = cols / 2;
    let (pix, cp) = (e / pairs, e % pairs);
    let b = pix / rows;
    let has_residual = !residual.is_empty();
    let Some(mut x) = x.element_thread32(t) else { return };
    let (v0, v1) = unpack_f16(x.read());
    let c = 2 * cp;
    let [m0, i0] = stats[(b * cols + c) as usize];
    let [m1, i1] = stats[(b * cols + c + 1) as usize];
    let mut y0 = r16((v0 - m0) * i0 * w[c as usize] + bias[c as usize]);
    let mut y1 = r16((v1 - m1) * i1 * w[c as usize + 1] + bias[c as usize + 1]);
    if has_residual {
        let (r0, r1) = unpack_f16(residual[e as usize]);
        y0 = r16(y0 + r0);
        y1 = r16(y1 + r1);
    }
    if leaky != 0 {
        y0 = leaky16(y0);
        y1 = leaky16(y1);
    }
    x.write(pack_f16(y0, y1));
}

/// One normalised, affine FP16 value: `r16((v - mean) * invstd * w + bias)`, the rounding PyTorch's
/// FP16 InstanceNorm output takes. Kernels that apply a norm on load use it; it is the same
/// expression as [`in_apply`]'s, which keeps its own copy so its PTX stays unchanged
#[inline(always)]
pub fn affine16(v: f32, mean: f32, invstd: f32, w: f32, bias: f32) -> f32 {
    r16((v - mean) * invstd * w + bias)
}
