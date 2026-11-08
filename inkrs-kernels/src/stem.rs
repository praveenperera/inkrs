//! The tail of villa's `LocalDepthFusionStem`: the second InstanceNorm's apply and LeakyReLU, the
//! attention logits, the depth softmax and the two depth pools, fused into one kernel
//!
//! PyTorch, under FP16 autocast, runs the 1x1x1 `attention_logits` conv in FP16 (bias added as
//! a separate FP16 add), takes `softmax(logits.float(), dim=Z)` in float32, casts the weights to
//! FP16, multiplies them into the FP16 features, sums over Z with a float32 accumulator, and
//! takes the FP16 max over Z. Each of those roundings is kept here

use cuda_device::thread::{blockDim_x, blockIdx_x, threadIdx_x};
use cuda_device::{DisjointSlice, kernel, warp};

use crate::common::{leaky16, math, pack_f16, r16, unpack_f16};
use crate::inorm::affine16;

/// Depth slices of the stem
const Z: usize = 17;
/// Stem channels
const C: usize = 16;
/// Lanes per pixel, one channel pair each
const LANES: u32 = (C / 2) as u32;
/// Threads per block, so 32 pixels per block
const BLOCK: u32 = 256;

/// `out[b, p] = cat(sum_z w_z * f[b, z, p], max_z f[b, z, p])`, 32 FP16 channels per pixel, where
/// `f = leaky16(affine16(raw))` is the stem's second InstanceNorm and LeakyReLU applied on load
///
/// `feat` is the raw conv output `[B, Z, pixels, C]` as FP16 pairs and `out` is `[B, pixels, 2C]`
/// as FP16 pairs. `stats` is `in_finalize`'s `[B, C]` `[mean, invstd]` of `feat`; `nw` and `nb`
/// are the norm's affine parameters in float32. Applying them here rounds to FP16 exactly where
/// `in_apply` did when it wrote the normalised features back, so the output keeps its bits and the
/// features are never written normalised. `w` holds the 16 logit weights and `bias` the logit bias,
/// both already rounded to FP16 by the host, as autocast casts them
///
/// Eight lanes share a pixel, one channel pair each, so each thread holds only its pair's norm
/// and logit parameters and two accumulators of each pool (one thread per pixel needed 64 norm
/// parameters beside 17 logits and 32 accumulators, and spilled). The logit's dot product keeps
/// villa's order: lane j computes pair j's term, and every lane adds the 8 terms sequentially, as
/// the one-thread version did. Block 256 threads (32 pixels, one sample, as `pixels` is a multiple of 32), grid
/// `B * pixels / 32`
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn depth_pool(
    mut out: DisjointSlice<u32>,
    feat: &[u32],
    stats: &[[f32; 2]],
    nw: &[f32],
    nb: &[f32],
    w: &[f32],
    bias: f32,
    pixels: u32,
) {
    let first = blockIdx_x() * (BLOCK / LANES);
    let b = first / pixels.max(1);
    // block-uniform, so every lane of a warp reaches the shuffles; indexing is unchecked after it
    if (blockDim_x() != BLOCK)
        | (pixels % (BLOCK / LANES) != 0)
        | (w.len() < C)
        | (nw.len() < C)
        | (nb.len() < C)
        | (((b + 1) as usize) * C > stats.len())
        | (((b + 1) as usize) * Z * pixels as usize * LANES as usize > feat.len())
        | (((b + 1) as usize) * pixels as usize * 2 * LANES as usize > out.len())
    {
        return;
    }
    let tid = threadIdx_x();
    let k = tid % LANES;
    let p = first - b * pixels + tid / LANES;
    let group = warp::lane_id() & !(LANES - 1);
    let (c0, c1) = (2 * k as usize, 2 * k as usize + 1);
    // SAFETY (every unchecked read below): the block check bounds `b`, `p < pixels` follows from
    // `pixels` being a multiple of the block's pixel count, and `k < 8` indexes 16 channels
    let ([m0, i0], [m1, i1]) =
        unsafe { (*stats.get_unchecked(b as usize * C + c0), *stats.get_unchecked(b as usize * C + c1)) };
    let (gw0, gw1, gb0, gb1) =
        unsafe { (*nw.get_unchecked(c0), *nw.get_unchecked(c1), *nb.get_unchecked(c0), *nb.get_unchecked(c1)) };
    let (lw0, lw1) = unsafe { (*w.get_unchecked(c0), *w.get_unchecked(c1)) };
    let base = (b as usize * Z * pixels as usize + p as usize) * LANES as usize + k as usize;
    let stride = pixels as usize * LANES as usize;
    let norm = [m0, i0, gw0, gb0, m1, i1, gw1, gb1];

    // the normalised pairs, packed in FP16 (exact, as the values are FP16 already): pass 2 reuses
    // them, and LLVM would otherwise keep 34 floats live across the softmax and spill
    let mut held = [0u32; Z];
    let mut logits = [0f32; Z];
    let mut top = f32::NEG_INFINITY;
    #[unroll]
    for z in 0..Z {
        // SAFETY: as above, `z < Z`
        let (a, c) = load(unsafe { *feat.get_unchecked(base + z * stride) }, norm);
        held[z] = pack_f16(a, c);
        // villa's dot is `dot += a * w[2j] + c * w[2j + 1]` over the pairs in order: each term is
        // rounded on its own (one mul, one fma) before the add, so lane j computes term j and
        // every lane adds the 8 terms in order
        let term = a * lw0 + c * lw1;
        let mut dot = 0f32;
        #[unroll]
        for j in 0..C / 2 {
            dot += warp::shuffle_f32(term, group + j as u32);
        }
        let l = r16(r16(dot) + bias);
        logits[z] = l;
        top = math::fmaxf(top, l);
    }
    let mut sum = 0f32;
    #[unroll]
    for z in 0..Z {
        let e = math::expf(logits[z] - top);
        logits[z] = e;
        sum += e;
    }

    let (mut att0, mut att1) = (0f32, 0f32);
    let (mut mx0, mut mx1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    #[unroll]
    for z in 0..Z {
        let wz = r16(logits[z] / sum);
        let (a, c) = unpack_f16(held[z]);
        att0 += r16(a * wz);
        att1 += r16(c * wz);
        mx0 = math::fmaxf(mx0, a);
        mx1 = math::fmaxf(mx1, c);
    }
    let o = ((b * pixels + p) * 2 * LANES + k) as usize;
    // SAFETY: in bounds by the block check; lane `k` of pixel `p` alone writes pair `k` of the
    // attention pool and pair `k` of the max pool
    unsafe {
        *out.get_unchecked_mut(o) = pack_f16(att0, att1);
        *out.get_unchecked_mut(o + LANES as usize) = pack_f16(mx0, mx1);
    }
}

/// One raw FP16 pair normalised (`affine16`) and activated (`leaky16`); `n` is
/// `[mean, invstd, w, bias]` of the low channel, then of the high one
#[inline(always)]
fn load(raw: u32, n: [f32; 8]) -> (f32, f32) {
    let (a, c) = unpack_f16(raw);
    (leaky16(affine16(a, n[0], n[1], n[2], n[3])), leaky16(affine16(c, n[4], n[5], n[6], n[7])))
}
