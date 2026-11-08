//! Hecate 9.6 um: the glue around its cuDNN convolutions, from the resident uint8 volume to the uint8 map
//!
//! The reference is `hecate.py` at huggingface.co/scrollprize/hecate 9cb86e5 (MIT). Activations are
//! channels-last (NDHWC) in one of two storages, chosen per launch by which of a pair of slices is
//! non-empty: float32 (`--precision fp32`, where PyTorch runs the convs in TF32) or bf16 bits
//! (`--precision bf16`, PyTorch's bf16 autocast). Under autocast the convs write bf16 and BatchNorm,
//! ReLU and the residual adds of the encoder each round to bf16 again, while GroupNorm and everything
//! after it in the decoder run in float32 until the next conv casts its input. These kernels keep those
//! roundings.
//!
//! Tiles are 16 x 64 x 64 and use the descriptors of [`crate::tiles::DESC`]: y0, x0, the source layer
//! of depth slice 0, the layer step (+1 forward, -1 reverse) and the slot (-1 for a padding tile).

use core::ptr::addr_of_mut;

use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};
use cuda_device::thread::{blockDim_x, blockIdx_x, blockIdx_y, blockIdx_z, sync_threads, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread};

use crate::common::{add_rn_f32, math, mul_rn_f32};
use crate::tiles::DESC;

// the CUDA C++ math library calls these libdevice functions for `logf`, `rintf` and `rsqrtf`
unsafe extern "C" {
    fn __nv_logf(x: f32) -> f32;
    fn __nv_rintf(x: f32) -> f32;
    fn __nv_rsqrtf(x: f32) -> f32;
}

/// Tile edge
const P: u32 = 64;
/// Pixels per tile slice
const PIX: u32 = P * P;
/// Depth slices per tile
const D: u32 = 16;
/// Channels of the last decoder stage, the head's input
const HEAD_C: u32 = 64;

/// `x` as bf16 would hold it, round to nearest even
#[inline(always)]
fn rbf(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16_rne(x))
}

/// [`rbf`] when `on`, else `x`
#[inline(always)]
fn rbf_if(on: bool, x: f32) -> f32 {
    if on { rbf(x) } else { x }
}

/// Element `i` of whichever storage is in use: bf16 when `h` is non-empty, else float32
#[inline(always)]
fn load(f: &[f32], h: &[u16], i: u32) -> f32 {
    if h.is_empty() { f[i as usize] } else { bf16_to_f32(h[i as usize]) }
}

/// The 16-plane input of every tile, normalised: `uint8 / 255` in float32, then the eval BatchNorm3d(1)
/// as `((x - mean) * invstd) * gamma + beta`, plus each column's support (any nonzero voxel in its 16
/// planes)
///
/// One thread per (tile, pixel). `xf` or `xh` is `[B, 16, 64, 64, cstride]` with the value in channel
/// 0; the host zero-fills the others once, for a stem conv whose one input channel is padded for
/// cuDNN. `support` is `[B, 64, 64]`. Pixels past the volume's edge read 0, as `hecate.py`
/// zero-pads a short tile in uint8 before it normalises. The BatchNorm runs in float32 under
/// autocast too; only the stem conv casts
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn hec_gather(
    mut xf: DisjointSlice<f32>,
    mut xh: DisjointSlice<u16>,
    mut support: DisjointSlice<u8>,
    vol: &[u8],
    desc: &[i32],
    height: u32,
    width: u32,
    cstride: u32,
    mean: f32,
    invstd: f32,
    gamma: f32,
    beta: f32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (b, p) = (i / PIX, i % PIX);
    let d = b as usize * DESC;
    if d + DESC > desc.len() {
        return;
    }
    let Some(mut s) = support.element_thread32(t) else { return };
    let (y, x) = (desc[d] as u32 + p / P, desc[d + 1] as u32 + p % P);
    let (first, step) = (desc[d + 2], desc[d + 3]);
    let bf16 = !xh.is_empty();
    let n = if bf16 { xh.len() } else { xf.len() };
    let inside = y < height && x < width;
    let mut any = false;
    let mut z = 0;
    while z < D {
        let v = if inside {
            let layer = (first + step * z as i32) as u32;
            vol[((layer * height + y) * width + x) as usize]
        } else {
            0
        };
        any |= v != 0;
        let norm = v as f32 / 255.0;
        let val = (norm - mean) * invstd * gamma + beta;
        let o = ((b * D + z) * PIX + p) as usize * cstride as usize;
        if o < n {
            // SAFETY: element (b, z, p) belongs to this thread alone, and `o` is in bounds
            unsafe {
                if bf16 {
                    *xh.get_unchecked_mut(o) = f32_to_bf16_rne(val);
                } else {
                    *xf.get_unchecked_mut(o) = val;
                }
            }
        }
        z += 1;
    }
    s.write(u8::from(any));
}

/// The norm epilogue `y = relu?(round?(x * a + b) + residual?)`, out of place, channels last
///
/// `ab[k] = [a, b]` is the per-channel scale and shift: one row of `cols` for a folded eval BatchNorm
/// (`per_sample == 0`), or one row per sample for GroupNorm ([`gn_finalize`]), where a sample holds
/// `per_sample` elements. Input, residual and output each come as a float32 / bf16 pair, the empty one
/// unused; an empty pair of residuals skips the add. `round_mid != 0` rounds the affine result to bf16
/// before the add, as autocast's BatchNorm writes bf16; a bf16 output rounds the final value
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn hec_affine(
    mut yf: DisjointSlice<f32>,
    mut yh: DisjointSlice<u16>,
    xf: &[f32],
    xh: &[u16],
    rf: &[f32],
    rh: &[u16],
    ab: &[[f32; 2]],
    cols: u32,
    per_sample: u32,
    round_mid: u32,
    relu: u32,
) {
    let e = thread::index_1d_u32(lc).get();
    let n = if yh.is_empty() { yf.len() } else { yh.len() } as u32;
    if e >= n {
        return;
    }
    let c = e % cols;
    let k = if per_sample > 0 { (e / per_sample) * cols + c } else { c };
    let [a, sh] = ab[k as usize];
    let mut y = load(xf, xh, e) * a + sh;
    if round_mid != 0 {
        y = rbf(y);
    }
    if !(rf.is_empty() && rh.is_empty()) {
        y += load(rf, rh, e);
    }
    // `<=` keeps a NaN, as PyTorch's ReLU does, so it reaches the finite check
    if relu != 0 && y <= 0.0 {
        y = 0.0;
    }
    // SAFETY: element `e < n` belongs to this thread alone
    unsafe {
        if yh.is_empty() {
            *yf.get_unchecked_mut(e as usize) = y;
        } else {
            *yh.get_unchecked_mut(e as usize) = f32_to_bf16_rne(y);
        }
    }
}

/// GroupNorm's partial sums: per (sample, chunk of rows, channel), `[sum(x - x0), sum((x - x0)^2)]`
/// around the channel's first value `x0 = x[b, 0, c]`, so the sums don't cancel
///
/// `x` is `[B, rows, cols]`. Grid `(chunks, B, cols / cpb)` with `cpb = min(cols, 256)` channels per
/// block and `lanes = blockDim / cpb` threads per channel; thread `t` owns channel
/// `blockIdx.z * cpb + t % cpb` and visits rows `lane, lane + lanes, ...` of its chunk
#[kernel]
pub fn gn_partial(
    mut partial: DisjointSlice<[f32; 2]>,
    xf: &[f32],
    xh: &[u16],
    rows: u32,
    cols: u32,
    cpb: u32,
    chunk_rows: u32,
) {
    static mut ACC: SharedArray<[f32; 2], 256> = SharedArray::UNINIT;
    let (chunk, b, cb, tid) = (blockIdx_x(), blockIdx_y(), blockIdx_z(), threadIdx_x());
    let n = if xh.is_empty() { xf.len() } else { xh.len() } as u32;
    let chunks = rows.div_ceil(chunk_rows.max(1));
    // block-uniform, so a block that fails it never reaches a barrier
    if cpb == 0
        || blockDim_x() > 256
        || blockDim_x() % cpb != 0
        || (cb + 1) * cpb > cols
        || (b + 1) * rows * cols > n
        || ((b * chunks + chunk + 1) * cols) as usize > partial.len()
    {
        return;
    }
    let lanes = blockDim_x() / cpb;
    let (cl, lane) = (tid % cpb, tid / cpb);
    let c = cb * cpb + cl;
    let sample = b * rows * cols;
    let x0 = load(xf, xh, sample + c);
    let (mut s, mut q) = (0f32, 0f32);
    let end = ((chunk + 1) * chunk_rows).min(rows);
    let mut r = chunk * chunk_rows + lane;
    while r < end {
        let d = load(xf, xh, sample + r * cols + c) - x0;
        s += d;
        q += d * d;
        r += lanes;
    }
    let acc = addr_of_mut!(ACC) as *mut [f32; 2];
    // SAFETY: each thread writes its own slot, read only after the barrier
    unsafe { *acc.add(tid as usize) = [s, q] };
    sync_threads();
    if lane == 0 {
        let (mut ts, mut tq) = (0f32, 0f32);
        let mut l = 0;
        while l < lanes {
            // SAFETY: `l * cpb + cl < blockDim <= 256`
            let [ps, pq] = unsafe { *acc.add((l * cpb + cl) as usize) };
            ts += ps;
            tq += pq;
            l += 1;
        }
        let o = ((b * chunks + chunk) * cols + c) as usize;
        // SAFETY: in bounds by the check above; only lane 0 of channel `c` writes this chunk's entry
        unsafe { *partial.get_unchecked_mut(o) = [ts, tq] };
    }
}

/// GroupNorm(groups) statistics to the epilogue's scale and shift, one thread per (sample, channel)
///
/// Combines [`gn_partial`]'s sums of the channel's group in float64, then as PyTorch's GroupNorm:
/// `rstd = rsqrtf(var + eps)` with the biased variance, `a = rstd * gamma[c]`, `b = beta[c] - a * mean`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn gn_finalize(
    mut ab: DisjointSlice<[f32; 2]>,
    partial: &[[f32; 2]],
    xf: &[f32],
    xh: &[u16],
    gamma: &[f32],
    beta: &[f32],
    rows: u32,
    cols: u32,
    chunks: u32,
    groups: u32,
    eps: f32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (b, c) = (i / cols, i % cols);
    let Some(mut out) = ab.element_thread32(t) else { return };
    let cg = cols / groups.max(1);
    let g0 = (c / cg) * cg;
    let n = rows as f64;
    let (mut sum, mut sq) = (0f64, 0f64);
    let mut cc = g0;
    while cc < g0 + cg {
        let shift = load(xf, xh, b * rows * cols + cc) as f64;
        let (mut s, mut q) = (0f64, 0f64);
        let mut k = 0;
        while k < chunks {
            let [ps, pq] = partial[((b * chunks + k) * cols + cc) as usize];
            s += ps as f64;
            q += pq as f64;
            k += 1;
        }
        sum += s + n * shift;
        sq += q + 2.0 * shift * s + n * shift * shift;
        cc += 1;
    }
    let total = n * cg as f64;
    let mean = sum / total;
    let var = sq / total - mean * mean;
    // `<` keeps a NaN variance, so a NaN input stays NaN through the norm
    let var = if var < 0.0 { 0.0 } else { var as f32 };
    // SAFETY: libdevice's rsqrtf, the function `rsqrt` calls in PyTorch's GroupNorm kernels
    let rstd = unsafe { __nv_rsqrtf(var + eps) };
    let a = rstd * gamma[c as usize];
    out.write([a, beta[c as usize] - a * mean as f32]);
}

/// PyTorch's `align_corners=False` source of output index `o` at a 2x upsample of an axis of `len`:
/// the lower index, the offset to its neighbour (0 at the edge) and the two weights
#[inline(always)]
fn up_src(o: u32, len: u32) -> (u32, u32, f32, f32) {
    let r = 0.5 * (o as f32 + 0.5) - 0.5;
    let r = if r > 0.0 { r } else { 0.0 };
    let i0 = r as u32;
    let l1 = r - i0 as f32;
    (i0, u32::from(i0 + 1 < len), 1.0 - l1, l1)
}

/// `cat([trilinear_x2(low), skip])` channels last, PyTorch's `upsample_trilinear3d` with
/// `align_corners=False` at an exact 2x size
///
/// `low` is `[B, d, h, w, cl]`, `skip` `[B, 2d, 2h, 2w, cs]`, both float32 (GroupNorm and decoder
/// outputs are float32 under autocast too); the output `[B, 2d, 2h, 2w, cl + cs]` is float32 or, when
/// `yh` is non-empty, the bf16 the next conv casts it to. Source index `max(0.5 (o + 0.5) - 0.5, 0)`,
/// neighbour clamped at the edge, and PyTorch's nesting of the eight products
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn hec_up2cat(
    mut yf: DisjointSlice<f32>,
    mut yh: DisjointSlice<u16>,
    low: &[f32],
    skip: &[f32],
    d: u32,
    h: u32,
    w: u32,
    cl: u32,
    cs: u32,
) {
    let e = thread::index_1d_u32(lc).get();
    let n = if yh.is_empty() { yf.len() } else { yh.len() } as u32;
    if e >= n {
        return;
    }
    let cols = cl + cs;
    let (c, vox) = (e % cols, e / cols);
    let (ow, oh, od) = (2 * w, 2 * h, 2 * d);
    let x = vox % ow;
    let y = (vox / ow) % oh;
    let z = (vox / (ow * oh)) % od;
    let b = vox / (ow * oh * od);
    let v = if c < cl {
        let (z1, zp, t0, t1) = up_src(z, d);
        let (y1, yp, h0, h1) = up_src(y, h);
        let (x1, xp, w0, w1) = up_src(x, w);
        let row = |zz: u32, yy: u32| (b * d + zz) * h + yy;
        let at = |r: u32, xx: u32| low[((r * w + xx) * cl + c) as usize];
        let (r00, r01) = (row(z1, y1), row(z1, y1 + yp));
        let (r10, r11) = (row(z1 + zp, y1), row(z1 + zp, y1 + yp));
        t0 * (h0 * (w0 * at(r00, x1) + w1 * at(r00, x1 + xp)) + h1 * (w0 * at(r01, x1) + w1 * at(r01, x1 + xp)))
            + t1 * (h0 * (w0 * at(r10, x1) + w1 * at(r10, x1 + xp))
                + h1 * (w0 * at(r11, x1) + w1 * at(r11, x1 + xp)))
    } else {
        skip[(vox * cs + c - cl) as usize]
    };
    // SAFETY: element `e < n` belongs to this thread alone
    unsafe {
        if yh.is_empty() {
            *yf.get_unchecked_mut(e as usize) = v;
        } else {
            *yh.get_unchecked_mut(e as usize) = f32_to_bf16_rne(v);
        }
    }
}

/// The head: per column, the logit and attention 1x1 convs over the 16 depths, the masked depth
/// softmax with the depth coordinate, the weighted logit sum and `hecate.py`'s
/// `sigmoid(logit(clamp(sigmoid(.), eps, 1 - eps)))`, times the column's support
///
/// One thread per (tile, pixel). `feat` is the last decoder stage `[B, 16, 64, 64, 64]` float32; `wl`
/// and `wa` the 64 weights of the two convs, `bl` and `ba` their biases, `dcs` the depth coordinate
/// scale. With `bf16 != 0` the two convs run as autocast runs them: bf16 inputs and weights, float32
/// accumulation, a bf16 result, then the bf16 bias added and rounded again. `prob[slot, pixel]` gets
/// the probability, 0 off the support; padding tiles write nothing. Off the support `hecate.py` still
/// sums its zero weights times the logits, so a non-finite logit there gives NaN, which its finite
/// check then refuses; this kernel writes NaN in that case too
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn hec_head(
    mut prob: DisjointSlice<f32>,
    feat: &[f32],
    support: &[u8],
    wl: &[f32],
    bl: f32,
    wa: &[f32],
    ba: f32,
    dcs: f32,
    desc: &[i32],
    bf16: u32,
) {
    let i = thread::index_1d_u32(lc).get();
    let (b, p) = (i / PIX, i % PIX);
    let d = b as usize * DESC;
    if d + DESC > desc.len() || wl.len() < HEAD_C as usize || wa.len() < HEAD_C as usize {
        return;
    }
    let slot = desc[d + 4];
    if slot < 0 {
        return;
    }
    let o = slot as usize * PIX as usize + p as usize;
    if o >= prob.len() {
        return;
    }
    let on = bf16 != 0;
    let mut logit = [0f32; D as usize];
    let mut score = [0f32; D as usize];
    let mut z = 0;
    while z < D {
        let base = (((b * D + z) * PIX + p) * HEAD_C) as usize;
        let (mut l, mut a) = (0f32, 0f32);
        let mut c = 0;
        while c < HEAD_C as usize {
            let f = rbf_if(on, feat[base + c]);
            l += f * rbf_if(on, wl[c]);
            a += f * rbf_if(on, wa[c]);
            c += 1;
        }
        logit[z as usize] = rbf_if(on, rbf_if(on, l) + rbf_if(on, bl));
        // the depth coordinate (z - 8) / 8 is exact in float32
        score[z as usize] = rbf_if(on, rbf_if(on, a) + rbf_if(on, ba)) + dcs * ((z as f32 - 8.0) / 8.0);
        z += 1;
    }
    let value;
    if support[i as usize] == 0 {
        // hecate.py's zero weights times the logits: 0 when they are all finite, NaN otherwise
        let mut zero = 0f32;
        let mut k = 0;
        while k < D as usize {
            zero += 0.0 * logit[k];
            k += 1;
        }
        // -0 (all logits negative) becomes 0, NaN stays NaN
        value = if zero == 0.0 { 0.0 } else { zero };
    } else {
        let mut m = score[0];
        let mut k = 1;
        while k < D as usize {
            m = math::fmaxf(m, score[k]);
            k += 1;
        }
        let mut sum = 0f32;
        let mut e = [0f32; D as usize];
        k = 0;
        while k < D as usize {
            e[k] = math::expf(score[k] - m);
            sum += e[k];
            k += 1;
        }
        // softmax, then hecate.py's renormalisation by the weights' own sum (clamped at 1e-8)
        let mut wsum = 0f32;
        k = 0;
        while k < D as usize {
            e[k] /= sum;
            wsum += e[k];
            k += 1;
        }
        // clamps by comparison, not fmaxf/fminf, so a NaN passes through as in torch.clamp
        let wsum = if wsum < 1e-8 { 1e-8 } else { wsum };
        let mut acc = 0f32;
        k = 0;
        while k < D as usize {
            acc += (e[k] / wsum) * logit[k];
            k += 1;
        }
        let eps = 1.1920929e-7f32;
        let p = 1.0 / (1.0 + math::expf(-acc));
        let pc = if p < eps {
            eps
        } else if p > 1.0 - eps {
            1.0 - eps
        } else {
            p
        };
        // SAFETY: libdevice's logf
        let lg = unsafe { __nv_logf(pc / (1.0 - pc)) };
        value = 1.0 / (1.0 + math::expf(-lg));
    }
    // SAFETY: each real tile has its own slot, so (slot, pixel) is written by this thread only,
    // and `o` is in bounds
    unsafe { *prob.get_unchecked_mut(o) = value };
}

/// One map: `rint(255 * clip(sum(p * w) / sum(w), 0, 1))` per pixel, as `hecate.py` finishes
///
/// `grid` `[ny, nx]` holds the slot of the tile at each position, or -1 for an empty tile, which still
/// adds its window to the denominator, as `hecate.py` does. `y_span[y]` / `x_span[x]` are the
/// half-open position ranges covering row y / column x, `weight` the `[64, 64]` floored Hann window.
/// Each pixel visits its tiles in `hecate.py`'s order with a separate multiply and add, as numpy's
/// float32 `+=` does
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn hec_blend(
    mut out: DisjointSlice<u8>,
    prob: &[f32],
    grid: &[i32],
    ys: &[u32],
    xs: &[u32],
    y_span: &[[u32; 2]],
    x_span: &[[u32; 2]],
    weight: &[f32],
    width: u32,
) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (y, x) = (i / width, i % width);
    let Some(mut out) = out.element_thread32(t) else { return };
    let nx = xs.len() as u32;
    let [y_first, y_end] = y_span[y as usize];
    let [x_first, x_end] = x_span[x as usize];
    let (mut num, mut den) = (0f32, 0f32);
    let mut iy = y_first;
    while iy < y_end {
        let ly = y - ys[iy as usize];
        let mut ix = x_first;
        while ix < x_end {
            let lx = x - xs[ix as usize];
            let w = weight[(ly * P + lx) as usize];
            let slot = grid[(iy * nx + ix) as usize];
            if slot >= 0 {
                num = add_rn_f32(num, mul_rn_f32(prob[(slot as u32 * PIX + ly * P + lx) as usize], w));
            }
            den = add_rn_f32(den, w);
            ix += 1;
        }
        iy += 1;
    }
    // the host has already refused non-finite probabilities, as hecate.py does before this step;
    // the comparison clamp still keeps a NaN rather than mapping it to an endpoint
    let q = num / den;
    let p = if q < 0.0 {
        0.0
    } else if q > 1.0 {
        1.0
    } else {
        q
    };
    // SAFETY: libdevice's rintf, round half to even as numpy's `rint`
    out.write(unsafe { __nv_rintf(mul_rn_f32(p, 255.0)) } as u8);
}
