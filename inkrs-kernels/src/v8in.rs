//! v8-in (Youssef Nader's ResNet3D-50 + 2D decoder, huggingface.co/YoussefMoNader/ink-8um-v8in, MIT): the
//! glue around its cuDNN convolutions, from the resident uint8 layers to the float32 map
//!
//! The reference runs under fp16 autocast: InstanceNorm and the trilinear upsample in float32, then every
//! conv, BatchNorm, ReLU, residual add, pool and the decoder in FP16 storage. Activations here are FP16
//! bits, channels-last. BatchNorm is folded into the convs on the host, so a conv's output already carries
//! its scale; the shift is a bias added in the epilogue.
//!
//! Tiles are 24 x 64 x 64 and use the descriptors of [`crate::tiles::DESC`]: y0, x0, the source layer of
//! depth slice 0, the layer step (+1 forward, -1 reverse) and the slot (-1 for a padding tile).

use core::ptr::addr_of_mut;

use cuda_device::thread::{blockDim_x, blockIdx_x, blockIdx_y, blockIdx_z, sync_threads, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread};

// add_rn_f32 and mul_rn_f32 from common: cuda-oxide refuses its own below sm_80, so the sm_75 tier
// writes them as PTX there and the other tiers re-export cuda-oxide's
use crate::common::{add_rn_f32, math, mul_rn_f32, pack_f16, r16, unpack_f16, widen};
use crate::tiles::DESC;

/// Tile edge, pixels per tile slice and depth slices per tile
const P: u32 = 64;
const PIX: u32 = P * P;
const D: u32 = 24;
/// Voxels per tile
const VOX: u32 = D * PIX;
/// The reference clips at 200 and scales by float32(1 / 255)
const CLIP: u8 = 200;
const INV255: f32 = 1.0 / 255.0;

/// The stem: its output is 96 x 128 x 128 x 64 before the max pool, 96 x 64 x 64 x 64 after
const SD: u32 = 96;
const SH: u32 = 128;
const PH: u32 = 64;
const SC: u32 = 64;
/// Composed kernel taps, 4 x 4 x 4 over the source tile
const TAPS: u32 = 64;
/// A stem block's pooled patch, its stem rows (2 * 8 + 1 with the pool's halo) and its source window
const PP: u32 = 8;
const SR: u32 = 2 * PP + 1;
const WIN: u32 = 12;
/// Threads of a stem block: one per (channel, row parity, column parity)
pub const V8_STEM_BLOCK: u32 = 256;

/// The InstanceNorm3d(1, affine) of every tile, as the scale and shift that map `x = min(v, 200) *
/// float32(1/255)` to `(x - mean) * invstd * gamma + beta`
///
/// One block per tile. The sums of `v` and `v^2` are exact integers (at most 98,304 * 200^2 < 2^32), so
/// the mean and the biased variance come out in float64 with no rounding beyond the final one; PyTorch's
/// float32 Welford is itself within a few ulp of that. `ab` is `[B, 2]`. A padding tile (slot -1) still
/// gets its statistics, so the network never sees NaN from it
#[kernel]
pub fn v8_stats(mut ab: DisjointSlice<[f32; 2]>, vol: &[u8], desc: &[i32], height: u32, width: u32, gamma: f32, beta: f32) {
    static mut SUMS: SharedArray<[u32; 2], 256> = SharedArray::UNINIT;
    let (b, tid) = (blockIdx_x(), threadIdx_x());
    let d = b as usize * DESC;
    // block-uniform, so a block that fails it never reaches the barrier
    if blockDim_x() != 256 || d + DESC > desc.len() || b as usize >= ab.len() {
        return;
    }
    let (y0, x0, first, step) = (desc[d] as u32, desc[d + 1] as u32, desc[d + 2], desc[d + 3]);
    let (mut s1, mut s2) = (0u32, 0u32);
    let mut i = tid;
    while i < VOX {
        let (z, y, x) = (i / PIX, y0 + (i / P) % P, x0 + i % P);
        let layer = (first + step * z as i32) as u32;
        let o = ((layer * height + y) * width + x) as usize;
        let v = if y < height && x < width && o < vol.len() { vol[o].min(CLIP) as u32 } else { 0 };
        s1 += v;
        s2 += v * v;
        i += 256;
    }
    let acc = addr_of_mut!(SUMS) as *mut [u32; 2];
    // SAFETY: each thread writes its own slot, read only after the barrier
    unsafe { *acc.add(tid as usize) = [s1, s2] };
    sync_threads();
    if tid == 0 {
        let (mut t1, mut t2) = (0u64, 0u64);
        let mut k = 0;
        while k < 256 {
            // SAFETY: k < 256
            let [a, q] = unsafe { *acc.add(k) };
            t1 += a as u64;
            t2 += q as u64;
            k += 1;
        }
        let n = VOX as u64;
        let c = INV255 as f64;
        let mean = c * t1 as f64 / n as f64;
        // n * sum(v^2) - sum(v)^2 is an exact integer, so the variance has no cancellation
        let var = c * c * (n * t2 - t1 * t1) as f64 / (n * n) as f64;
        let scale = gamma as f64 / math::sqrt(var + 1e-5);
        // SAFETY: b < ab.len() by the check above, and only thread 0 of block b writes it
        unsafe { *ab.get_unchecked_mut(b as usize) = [scale as f32, (beta as f64 - mean * scale) as f32] };
    }
}

/// Stem: InstanceNorm, the trilinear upsample to 96 x 256 x 256, conv3d 1 -> 64 (7 x 7 x 7, stride
/// (1, 2, 2), pad 3), BatchNorm, ReLU and max pool (1, 3, 3) stride (1, 2, 2) pad (0, 1, 1), in one pass
///
/// The upsample is linear, so it composes with the conv: each stem output reads a 4 x 4 x 4 window of the
/// normalised source tile, through a kernel that depends only on its class per axis (the output's phase
/// against the 4x grid, or its position near the border, where the upsample clamps or the conv's zero
/// padding cuts taps). The host builds those kernels from PyTorch's index and weight formulas in float64,
/// with the BatchNorm scale folded in: `w` is `[class, tap, 64]`, `bias` the BatchNorm shift. Per axis,
/// `dcls`/`hcls` give each output's class and `dbase`/`hbase` its window's first source index (the same
/// tables serve H and W). Neither the 12.6 MB upsampled input nor the 201 MB stem output per tile is
/// ever written, and the composed sums run in float32 where the reference rounds the upsample to FP16
/// first, so outputs differ from the reference by less than its own FP16 rounding.
///
/// Grid `(64 patches, 96 depths, B)`, [`V8_STEM_BLOCK`] threads. A block owns one depth and an 8 x 8 patch
/// of the pooled output, so 17 x 17 stem outputs with the pool's halo. Thread `t` owns channel `t % 64`
/// and the outputs whose row and column parities are `t / 64`; it holds its interior class's 64 weights
/// in registers and reads a border output's from global memory (warp-uniform: a warp shares the parity).
/// `out` is `[B, 96, 64, 64, 64]` FP16
#[kernel]
#[allow(clippy::too_many_arguments)]
pub fn v8_stem(
    mut out: DisjointSlice<u16>,
    vol: &[u8],
    desc: &[i32],
    ab: &[[f32; 2]],
    w: &[f32],
    bias: &[f32],
    dcls: &[u32],
    dbase: &[i32],
    hcls: &[u32],
    hbase: &[i32],
    height: u32,
    width: u32,
    nh: u32,
) {
    static mut SRC: SharedArray<f32, 576> = SharedArray::UNINIT;
    static mut ACT: SharedArray<u16, 18496> = SharedArray::UNINIT;
    let (patch, od, b, tid) = (blockIdx_x(), blockIdx_y(), blockIdx_z(), threadIdx_x());
    let d = b as usize * DESC;
    let nb = (b as usize + 1) * (SD * PH * PH * SC) as usize;
    // block-uniform checks before the first barrier
    if (blockDim_x() != V8_STEM_BLOCK)
        | (patch >= (PH / PP) * (PH / PP))
        | (od >= SD)
        | (d + DESC > desc.len())
        | (b as usize >= ab.len())
        | (nb > out.len())
        | (dcls.len() != SD as usize)
        | (dbase.len() != SD as usize)
        | (hcls.len() != SH as usize)
        | (hbase.len() != SH as usize)
        | (bias.len() != SC as usize)
    {
        return;
    }
    let (y0, x0, first, step) = (desc[d] as u32, desc[d + 1] as u32, desc[d + 2], desc[d + 3]);
    let [scale, shift] = ab[b as usize];
    let (ph0, pw0) = ((patch / (PH / PP)) * PP, (patch % (PH / PP)) * PP);
    // stem rows and columns 2 * p0 - 1 ..= 2 * p0 + 15; the first can be -1
    let (oh_lo, ow_lo) = (2 * ph0 as i32 - 1, 2 * pw0 as i32 - 1);
    let first_valid = |o: i32| if o < 0 { 0 } else { o as usize };
    let (r0, c0) = (hbase[first_valid(oh_lo)], hbase[first_valid(ow_lo)]);
    let z0 = dbase[od as usize];
    let src = addr_of_mut!(SRC) as *mut f32;
    let act = addr_of_mut!(ACT) as *mut u16;

    // the normalised source window, clamped into the tile: a clamped index only ever carries a zero weight
    let mut i = tid;
    while i < 4 * WIN * WIN {
        let (jd, r, c) = (i / (WIN * WIN), (i / WIN) % WIN, i % WIN);
        let z = (z0 + jd as i32).max(0).min(D as i32 - 1) as u32;
        let yy = (r0 + r as i32).max(0).min(P as i32 - 1) as u32;
        let xx = (c0 + c as i32).max(0).min(P as i32 - 1) as u32;
        let (y, x) = (y0 + yy, x0 + xx);
        let layer = (first + step * z as i32) as u32;
        let o = ((layer * height + y) * width + x) as usize;
        let v = if y < height && x < width && o < vol.len() { vol[o].min(CLIP) } else { 0 };
        let xv = mul_rn_f32(v as f32, INV255);
        // SAFETY: i < 576
        unsafe { *src.add(i as usize) = xv * scale + shift };
        i += V8_STEM_BLOCK;
    }
    sync_threads();

    let (ch, par) = (tid % SC, tid / SC);
    let (rh, rw) = (par / 2, par % 2);
    let cd = dcls[od as usize];
    // the interior class of this thread's outputs: local row i is stem row 2 * p0 - 1 + i, so local parity
    // rh holds stem rows of parity 1 - rh, whose interior representative is row 65 - rh (rows 64 and 65
    // are interior on both axes)
    let mine = (cd * nh + hcls[(65 - rh) as usize]) * nh + hcls[(65 - rw) as usize];
    let mut wr = [0f32; TAPS as usize];
    #[unroll]
    for t in 0..TAPS as usize {
        let o = (mine as usize * TAPS as usize + t) * SC as usize + ch as usize;
        wr[t] = if o < w.len() { w[o] } else { 0.0 };
    }
    let bc = bias[ch as usize];
    let mut i = rh;
    while i < SR {
        let oh = oh_lo + i as i32;
        if (0..SH as i32).contains(&oh) {
            let hb = (hbase[oh as usize] - r0).max(0).min((WIN - 4) as i32) as u32;
            let ch_cls = hcls[oh as usize];
            let mut j = rw;
            while j < SR {
                let ow = ow_lo + j as i32;
                if (0..SH as i32).contains(&ow) {
                    let wb = (hbase[ow as usize] - c0).max(0).min((WIN - 4) as i32) as u32;
                    let cls = (cd * nh + ch_cls) * nh + hcls[ow as usize];
                    let mut acc = 0f32;
                    if cls == mine {
                        #[unroll]
                        for t in 0..TAPS as usize {
                            let (jd, jh, jw) = (t as u32 / 16, (t as u32 / 4) % 4, t as u32 % 4);
                            // SAFETY: jd < 4, hb + jh < WIN and wb + jw < WIN by the clamps above
                            let s = unsafe { *src.add(((jd * WIN + hb + jh) * WIN + wb + jw) as usize) };
                            acc += wr[t] * s;
                        }
                    } else {
                        let base = cls as usize * TAPS as usize * SC as usize + ch as usize;
                        let mut t = 0u32;
                        while t < TAPS {
                            let (jd, jh, jw) = (t / 16, (t / 4) % 4, t % 4);
                            let o = base + (t * SC) as usize;
                            let wv = if o < w.len() { w[o] } else { 0.0 };
                            // SAFETY: as above
                            let s = unsafe { *src.add(((jd * WIN + hb + jh) * WIN + wb + jw) as usize) };
                            acc += wv * s;
                            t += 1;
                        }
                    }
                    let y = acc + bc;
                    // ReLU; NaN propagates, as PyTorch's does
                    let y = if y < 0.0 { 0.0 } else { y };
                    // SAFETY: i, j < 17 and ch < 64
                    unsafe { *act.add(((i * SR + j) * SC + ch) as usize) = crate::common::narrow(y) };
                }
                j += 2;
            }
        }
        i += 2;
    }
    sync_threads();

    // max pool: the window's stem rows 2p - 1 ..= 2p + 1 are local rows 2 * (p - p0) ..= + 2
    let mut k = tid;
    while k < PP * PP * SC {
        let (c, pp) = (k % SC, k / SC);
        let (pi, pj) = (pp / PP, pp % PP);
        let mut m = f32::NEG_INFINITY;
        let mut di = 0;
        while di < 3 {
            let li = 2 * pi + di;
            let oh = oh_lo + li as i32;
            let mut dj = 0;
            while dj < 3 {
                let lj = 2 * pj + dj;
                let ow = ow_lo + lj as i32;
                if (0..SH as i32).contains(&oh) && (0..SH as i32).contains(&ow) {
                    // SAFETY: li, lj < 17
                    let v = widen(unsafe { *act.add(((li * SR + lj) * SC + c) as usize) });
                    if v > m || v != v {
                        m = v;
                    }
                }
                dj += 1;
            }
            di += 1;
        }
        let o = (((b * SD + od) * PH + ph0 + pi) * PH + pw0 + pj) * SC + c;
        // SAFETY: o < nb <= out.len(), and each pooled element belongs to one (block, k)
        unsafe { *out.get_unchecked_mut(o as usize) = crate::common::narrow(m) };
        k += V8_STEM_BLOCK;
    }
}

/// `y[b, p, off + c] = max over d of x[b, d, p, c]`: a layer's depth max, written into its slot of the
/// decoder's concatenation buffer (`ystride` channels per pixel) or into a plain map (`off` 0)
///
/// One thread per (sample, pixel, channel pair), FP16 in and out, `torch.max`'s NaN propagation. `x` is
/// `[B, depth, pixels, c]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn v8_dmax(mut y: DisjointSlice<u32>, x: &[u32], depth: u32, pixels: u32, c: u32, ystride: u32, off: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (c2, ys2, off2) = (c / 2, ystride / 2, off / 2);
    if (c2 == 0) | (c % 2 != 0) | (ystride % 2 != 0) | (off % 2 != 0) | (off + c > ystride) {
        return;
    }
    let (b, p, k) = (i / (pixels * c2), (i / c2) % pixels, i % c2);
    let o = ((b * pixels + p) * ys2 + off2 + k) as usize;
    if o >= y.len() || (((b + 1) * depth * pixels * c2) as usize) > x.len() {
        return;
    }
    let (mut m0, mut m1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut z = 0;
    while z < depth {
        let (v0, v1) = unpack_f16(x[(((b * depth + z) * pixels + p) * c2 + k) as usize]);
        if v0 > m0 || v0 != v0 {
            m0 = v0;
        }
        if v1 > m1 || v1 != v1 {
            m1 = v1;
        }
        z += 1;
    }
    // SAFETY: o < y.len(); each (b, p, k) is one thread's, and every value is already FP16, so packing is exact
    unsafe { *y.get_unchecked_mut(o) = pack_f16(m0, m1) };
}

/// The decoder's `F.interpolate(low, scale_factor=2, mode="bilinear")` (`align_corners=False`) into
/// channels `off .. off + c` of the concatenation buffer `y` (`ystride` channels per pixel), FP16 in and
/// out with PyTorch's float32 index and weight formulas (`upsample_bilinear2d`'s `area_pixel` source
/// index with scale 0.5)
///
/// One thread per (sample, output pixel, channel pair). `low` is `[B, h, w, c]`, `y` `[B, 2h, 2w, ystride]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn v8_up2cat(mut y: DisjointSlice<u32>, low: &[u32], h: u32, w: u32, c: u32, ystride: u32, off: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (c2, ys2, off2) = (c / 2, ystride / 2, off / 2);
    if (c2 == 0) | (h == 0) | (w == 0) | (c % 2 != 0) | (ystride % 2 != 0) | (off % 2 != 0) | (off + c > ystride) {
        return;
    }
    let (h2, w2) = (2 * h, 2 * w);
    let (b, p, k) = (i / (h2 * w2 * c2), (i / c2) % (h2 * w2), i % c2);
    let (oy, ox) = (p / w2, p % w2);
    let o = ((b * h2 * w2 + p) * ys2 + off2 + k) as usize;
    if o >= y.len() || (((b + 1) * h * w * c2) as usize) > low.len() {
        return;
    }
    let src = |d: u32, len: u32| {
        let r = 0.5 * (d as f32 + 0.5) - 0.5;
        let r = if r < 0.0 { 0.0 } else { r };
        let i0 = r as u32;
        let step = u32::from(i0 < len - 1);
        let l1 = r - i0 as f32;
        (i0, step, 1.0 - l1, l1)
    };
    let (y0, ys, hl0, hl1) = src(oy, h);
    let (x0, xs, wl0, wl1) = src(ox, w);
    let at = |yy: u32, xx: u32| unpack_f16(low[(((b * h + yy) * w + xx) * c2 + k) as usize]);
    let (a0, a1) = at(y0, x0);
    let (b0, b1) = at(y0, x0 + xs);
    let (e0, e1) = at(y0 + ys, x0);
    let (f0, f1) = at(y0 + ys, x0 + xs);
    let v0 = hl0 * (wl0 * a0 + wl1 * b0) + hl1 * (wl0 * e0 + wl1 * f0);
    let v1 = hl0 * (wl0 * a1 + wl1 * b1) + hl1 * (wl0 * e1 + wl1 * f1);
    // SAFETY: o < y.len(), one thread per element pair
    unsafe { *y.get_unchecked_mut(o) = pack_f16(v0, v1) };
}

/// `y = relu?(x + bias[c] (+ r))`, FP16 in and out, computed in float32 and rounded once: the epilogue
/// of a folded conv when cuDNN's fused bias-activation is not used. `r` empty skips the residual. One
/// thread per channel pair
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn v8_epilogue(mut y: DisjointSlice<u32>, x: &[u32], r: &[u32], bias: &[f32], c: u32, relu: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let c2 = c / 2;
    if (c2 == 0) | (c % 2 != 0) | (bias.len() != c as usize) {
        return;
    }
    let k = (i % c2) as usize;
    let Some(mut out) = y.element_thread32(t) else { return };
    let iu = i as usize;
    if iu >= x.len() || (!r.is_empty() && iu >= r.len()) {
        return;
    }
    let (mut v0, mut v1) = unpack_f16(x[iu]);
    v0 += bias[2 * k];
    v1 += bias[2 * k + 1];
    if !r.is_empty() {
        let (r0, r1) = unpack_f16(r[iu]);
        v0 += r0;
        v1 += r1;
    }
    if relu != 0 {
        v0 = if v0 < 0.0 { 0.0 } else { v0 };
        v1 = if v1 < 0.0 { 0.0 } else { v1 };
    }
    out.write(pack_f16(v0, v1));
}

/// The head: the 1 x 1 logit conv (256 -> 1) on the decoder's last map, then sigmoid, into each real
/// tile's slot of `prob` (`[slots, 64 * 64]` float32)
///
/// As autocast runs it: the FP16 dot with float32 accumulation rounded to FP16, plus the FP16-rounded
/// bias, rounded again (cuDNN writes the conv, then PyTorch adds the bias in FP16). `w` holds the weights
/// already rounded to FP16. `sigmoid16 != 0` rounds the probability to FP16 (`torch.sigmoid` of the FP16
/// logit, as ink8um and inksweep take it); otherwise it is `sigmoid(logit.float())`, as the 306B runners
/// take it. `feat` is `[B, 64, 64, c]`; one thread per (tile, pixel)
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn v8_head(mut prob: DisjointSlice<f32>, feat: &[u16], w: &[f32], b: f32, desc: &[i32], c: u32, sigmoid16: u32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (tile, p) = (i / PIX, i % PIX);
    let d = tile as usize * DESC;
    if d + DESC > desc.len() || w.len() != c as usize || ((tile + 1) * PIX * c) as usize > feat.len() {
        return;
    }
    let slot = desc[d + 4];
    if slot < 0 {
        return;
    }
    let o = (slot as u32 * PIX + p) as usize;
    if o >= prob.len() {
        return;
    }
    let base = (i * c) as usize;
    let mut acc = 0f32;
    let mut k = 0;
    while k < c {
        acc += widen(feat[base + k as usize]) * w[k as usize];
        k += 1;
    }
    let logit = r16(r16(acc) + r16(b));
    let s = 1.0 / (1.0 + math::expf(-logit));
    let s = if sigmoid16 != 0 { r16(s) } else { s };
    // SAFETY: o < prob.len(); slot is unique per real tile, and p is this thread's pixel
    unsafe { *prob.get_unchecked_mut(o) = s };
}

/// The stitched map: per pixel, over the kept tiles that cover it in tile order (row-major on the
/// stride grid), `num += p * w` and `den += w` (`mode` 1, the 306B runners' weight sum, NaN where no
/// tile) or `den += 1` (`mode` 0, the release's tile count, 0 where no tile, clipped to 0..1), in
/// float32 with no contraction, as numpy and PyTorch add them
///
/// `grid` is `[ny, nx]` slots (-1 for a tile not kept) of the lattice `k * stride`; `out` is `[height,
/// width]`. One thread per pixel
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
#[allow(clippy::too_many_arguments)]
pub fn v8_blend(
    mut out: DisjointSlice<f32>,
    prob: &[f32],
    grid: &[i32],
    weight: &[f32],
    ny: u32,
    nx: u32,
    stride: u32,
    width: u32,
    mode: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    if (stride == 0) | (width == 0) | (weight.len() != PIX as usize) | (grid.len() != (ny * nx) as usize) {
        return;
    }
    let (y, x) = (i / width, i % width);
    let Some(mut out) = out.element_thread32(t) else { return };
    // tiles k with k * stride <= y < k * stride + 64
    let span = |v: u32, n: u32| {
        let lo = if v >= P { (v - (P - 1)).div_ceil(stride) } else { 0 };
        let hi = (v / stride + 1).min(n);
        (lo, hi)
    };
    let (ky0, ky1) = span(y, ny);
    let (kx0, kx1) = span(x, nx);
    let (mut num, mut den) = (0f32, 0f32);
    let mut ky = ky0;
    while ky < ky1 {
        let ly = y - ky * stride;
        let mut kx = kx0;
        while kx < kx1 {
            let lx = x - kx * stride;
            let slot = grid[(ky * nx + kx) as usize];
            if slot >= 0 {
                let w = weight[(ly * P + lx) as usize];
                let o = (slot as u32 * PIX + ly * P + lx) as usize;
                let p = if o < prob.len() { prob[o] } else { f32::NAN };
                num = add_rn_f32(num, mul_rn_f32(p, w));
                den = add_rn_f32(den, if mode == 0 { 1.0 } else { w });
            }
            kx += 1;
        }
        ky += 1;
    }
    let v = if mode == 0 {
        let q = if den != 0.0 { num / den } else { 0.0 };
        // np.clip(np.nan_to_num(q), 0, 1)
        if q != q {
            0.0
        } else if q < 0.0 {
            0.0
        } else if q > 1.0 {
            1.0
        } else {
            q
        }
    } else if den > 1e-9 {
        num / den
    } else {
        f32::NAN
    };
    out.write(v);
}
