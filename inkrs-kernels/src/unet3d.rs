//! Glue kernels for villa's autoconfigured 3D ResEnc U-Net (`inkrs ink3d`), around cuDNN's
//! channels-last bf16 convolutions and cuBLAS's bf16 GEMMs
//!
//! Activations are bf16 channels-last `[S, C]` (S = D*H*W, batch 1), carried as `u32` pairs of bf16
//! bits, low half first. Every kernel repeats the rounding PyTorch's bf16 autocast forward takes in
//! villa's eager reference:
//! - a conv writes bf16, then its bf16 bias is added and rounded again (PyTorch adds a conv's bias in a
//!   separate op), so the biased value is `rbf(conv + bias)`;
//! - InstanceNorm takes float32 statistics of those bf16 values and writes
//!   `rbf(gamma * (x - mean) * invstd + beta)`, the last multiply-add fused, as ATen's
//!   `batch_norm_transform_input_kernel` compiles;
//! - the residual add and LeakyReLU each round to bf16 again;
//! - the head's logit is `rbf(rbf(dot) + bias)`, then a float32 sigmoid;
//! - the blend repeats villa's float32 steps: separate multiply and add, no FMA.
//!
//! Index math is 32-bit: the host keeps every buffer under 2^32 elements

use core::ptr::addr_of_mut;

use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicU32};
use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};
use cuda_device::float::{add_rn_f32, fma_rn_f32, mul_rn_f32};
use cuda_device::thread::{blockDim_x, blockIdx_x, sync_threads, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread};

use crate::common::math;

// the CUDA C++ math library calls this libdevice function for `rintf`, which `torch.round` uses
unsafe extern "C" {
    fn __nv_rintf(x: f32) -> f32;
}

/// `x` as bf16 holds it, round to nearest even
#[inline(always)]
fn rbf(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16_rne(x))
}

/// The low bf16 of a pair
#[inline(always)]
fn lo(p: u32) -> f32 {
    bf16_to_f32(p as u16)
}

/// The high bf16 of a pair
#[inline(always)]
fn hi(p: u32) -> f32 {
    bf16_to_f32((p >> 16) as u16)
}

/// Two floats rounded to bf16 and packed, `a` in the low half
#[inline(always)]
fn pack(a: f32, b: f32) -> u32 {
    f32_to_bf16_rne(a) as u32 | ((f32_to_bf16_rne(b) as u32) << 16)
}

/// A conv output after PyTorch's separate bias add; an empty `bias` means the conv has none
#[inline(always)]
fn biased(v: f32, bias: &[f32], c: u32) -> f32 {
    if bias.is_empty() { v } else { rbf(add_rn_f32(v, bias[c as usize])) }
}

/// ATen's InstanceNorm transform, rounded to bf16: `gamma * (x - mean) * invstd + beta`, the
/// trailing multiply-add contracted as nvcc compiles it
#[inline(always)]
fn affine(v: f32, mean: f32, invstd: f32, gamma: f32, beta: f32) -> f32 {
    rbf(fma_rn_f32(mul_rn_f32(gamma, v - mean), invstd, beta))
}

/// PyTorch's LeakyReLU on a bf16 value: computed in float, rounded to bf16
#[inline(always)]
fn leaky(v: f32, slope: f32) -> f32 {
    if v > 0.0 { v } else { rbf(mul_rn_f32(v, slope)) }
}

/// Welford partial statistics of one spatial chunk, every channel, of a biased conv output
///
/// `x` is `[rows, cols]` bf16 pairs, `cols` even. Grid `chunks`, block `cols / 2 * lanes` threads
/// (at most 256): thread `t` owns channel pair `t % (cols / 2)` and starts at row `t / (cols / 2)` of
/// the chunk, stepping by `lanes`. Writes `partial[chunk, c] = [mean, m2]` of the chunk's biased
/// values `v`, with `m2 = sum((v - mean)^2)`. Each thread runs Welford's update and the lanes merge by
/// Chan's rule, in a fixed order, as ATen's InstanceNorm statistics do: shifted sums `E[d^2] - E[d]^2`
/// cancel when the shift is an outlier, and can even go negative
#[kernel]
pub fn u3_partial(
    mut partial: DisjointSlice<[f32; 2]>,
    x: &[u32],
    bias: &[f32],
    rows: u32,
    cols: u32,
    lanes: u32,
    chunk_rows: u32,
) {
    static mut ACC: SharedArray<[f32; 6], 256> = SharedArray::UNINIT;
    let (chunk, tid) = (blockIdx_x(), threadIdx_x());
    let pairs = cols / 2;
    let chunks = (rows + chunk_rows - 1) / chunk_rows.max(1);
    // block-uniform, so a block that fails it never reaches a barrier
    if pairs == 0
        || blockDim_x() != pairs * lanes
        || blockDim_x() > 256
        || (rows * pairs) as usize > x.len()
        || (((chunk + 1) * cols) as usize) > partial.len()
        || chunk >= chunks
        || (!bias.is_empty() && bias.len() < cols as usize)
    {
        return;
    }
    let (cp, lane) = (tid % pairs, tid / pairs);

    let (mut n, mut m0, mut m1, mut q0, mut q1) = (0f32, 0f32, 0f32, 0f32, 0f32);
    let end = ((chunk + 1) * chunk_rows).min(rows);
    let mut r = chunk * chunk_rows + lane;
    while r < end {
        let p = x[(r * pairs + cp) as usize];
        let v0 = biased(lo(p), bias, 2 * cp);
        let v1 = biased(hi(p), bias, 2 * cp + 1);
        n += 1.0;
        let d0 = v0 - m0;
        let d1 = v1 - m1;
        m0 += d0 / n;
        m1 += d1 / n;
        q0 += d0 * (v0 - m0);
        q1 += d1 * (v1 - m1);
        r += lanes;
    }
    let acc = addr_of_mut!(ACC) as *mut [f32; 6];
    // SAFETY: each thread writes its own slot, read only after the barrier
    unsafe { *acc.add(tid as usize) = [n, m0, m1, q0, q1, 0.0] };
    sync_threads();
    if lane == 0 {
        let mut l = 1;
        while l < lanes {
            // SAFETY: `l * pairs + cp < blockDim <= 256`
            let [nb, b0, b1, r0, r1, _] = unsafe { *acc.add((l * pairs + cp) as usize) };
            if nb > 0.0 {
                let nt = n + nb;
                let w = nb / nt;
                let (e0, e1) = (b0 - m0, b1 - m1);
                m0 += e0 * w;
                m1 += e1 * w;
                q0 += r0 + e0 * e0 * n * w;
                q1 += r1 + e1 * e1 * n * w;
                n = nt;
            }
            l += 1;
        }
        let o = (chunk * cols + 2 * cp) as usize;
        // SAFETY: `o + 1 < partial.len()` by the check above, and only thread `cp` of this block
        // writes channels `2cp, 2cp + 1` of this chunk
        unsafe {
            *partial.get_unchecked_mut(o) = [m0, q0];
            *partial.get_unchecked_mut(o + 1) = [m1, q1];
        }
    }
}

/// Chan's merge of two `[count, mean, m2]` Welford states, in float32 as ATen's `welford_combine`
/// runs for bf16 inputs
#[inline(always)]
fn chan(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    if b[0] == 0.0 {
        return a;
    }
    if a[0] == 0.0 {
        return b;
    }
    let n = a[0] + b[0];
    let w = b[0] / n;
    let e = b[1] - a[1];
    [n, a[1] + e * w, a[2] + b[2] + e * e * a[0] * w]
}

/// Per channel `[mean, invstd]` from [`u3_partial`]'s chunks, merged by Chan's rule in float64; chunk
/// `k` holds rows `[k * chunk_rows, min((k + 1) * chunk_rows, rows))`
///
/// Grid `cols` blocks, one per channel, 256 threads: thread `t` merges chunks `t, t + 256, ...` in order,
/// then a shared-memory tree merges the threads in a fixed pairing, so the result is deterministic. Merging
/// in float32: GeForce runs float64 at 1/64 rate, and float64 merges held this kernel at 30-58 us a call
/// (2.5 ms a forward). `invstd = 1 / sqrt(var + eps)` with the biased variance, as ATen's statistics give it
#[kernel]
pub fn u3_finalize(
    mut stats: DisjointSlice<[f32; 2]>,
    partial: &[[f32; 2]],
    rows: u32,
    cols: u32,
    chunks: u32,
    chunk_rows: u32,
    eps: f32,
) {
    static mut ACC: SharedArray<[f32; 3], 256> = SharedArray::UNINIT;
    let (c, tid) = (blockIdx_x(), threadIdx_x());
    // block-uniform, so a block that fails it never reaches a barrier
    if c >= cols || blockDim_x() != 256 || (chunks * cols) as usize > partial.len() || c as usize >= stats.len() {
        return;
    }
    let mut st = [0f32; 3];
    let mut k = tid;
    while k < chunks {
        let [pm, pq] = partial[(k * cols + c) as usize];
        let nb = (rows - k * chunk_rows).min(chunk_rows) as f32;
        st = chan(st, [nb, pm, pq]);
        k += 256;
    }
    let acc = addr_of_mut!(ACC) as *mut [f32; 3];
    // SAFETY: each thread writes its own slot; reads of other slots follow a barrier
    unsafe { *acc.add(tid as usize) = st };
    sync_threads();
    let mut half = 128;
    while half > 0 {
        if tid < half {
            // SAFETY: `tid + half < 256`; slot `tid` is written only by thread `tid`, and slot
            // `tid + half` was last written before the barrier above
            unsafe { *acc.add(tid as usize) = chan(*acc.add(tid as usize), *acc.add((tid + half) as usize)) };
        }
        sync_threads();
        half /= 2;
    }
    if tid == 0 {
        // SAFETY: slot 0 holds the whole channel after the last barrier
        let [n, m, q] = unsafe { *acc };
        let var = if n > 0.0 { q / n } else { 0.0 };
        let var = if var > 0.0 { var } else { 0.0 };
        // SAFETY: `c < stats.len()` by the check above, and only thread 0 of block `c` writes it
        unsafe { *stats.get_unchecked_mut(c as usize) = [m, 1.0 / math::sqrtf(var + eps)] };
    }
}

/// InstanceNorm apply with an optional residual and LeakyReLU, two channels per thread:
/// `out = leaky?(rbf(affine(biased(x)) + residual?))`
///
/// - `x`: the conv output `[pixels, cols]` bf16 pairs, `bias` its bf16 bias or empty;
/// - `stats`, `w`, `b`: its [`u3_finalize`] statistics and float32 affine;
/// - residual, at most one of: `res`, a bf16 tensor `[pixels, cols]`; or `rraw` with `rstats`, `rw`,
///   `rb`, a bias-free conv output normalised on load (the BasicBlockD skip projection);
/// - `act != 0` applies LeakyReLU with `slope`.
///
/// The result goes to `out` at pair `pix * out_stride + out_off + cp`, so a decoder's concat buffer can
/// take it, and also to `out2` at `pix * out2_stride + out2_off + cp` when `out2` is non-empty
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn u3_apply(
    mut out: DisjointSlice<u32>,
    mut out2: DisjointSlice<u32>,
    x: &[u32],
    bias: &[f32],
    stats: &[[f32; 2]],
    w: &[f32],
    b: &[f32],
    res: &[u32],
    rraw: &[u32],
    rstats: &[[f32; 2]],
    rw: &[f32],
    rb: &[f32],
    pixels: u32,
    cols: u32,
    act: u32,
    slope: f32,
    out_stride: u32,
    out_off: u32,
    out2_stride: u32,
    out2_off: u32,
) {
    let e = thread::index_1d_u32(lc).get();
    let pairs = cols / 2;
    if pairs == 0 || e >= pixels * pairs {
        return;
    }
    let (pix, cp) = (e / pairs, e % pairs);
    let v = apply_pair(x, bias, stats, w, b, res, rraw, rstats, rw, rb, e, 2 * cp, act, slope);
    let o = (pix * out_stride + out_off + cp) as usize;
    if o < out.len() {
        // SAFETY: in bounds; threads own distinct (pixel, pair) slots, so no two write one element
        unsafe { *out.get_unchecked_mut(o) = v };
    }
    if !out2.is_empty() {
        let o2 = (pix * out2_stride + out2_off + cp) as usize;
        if o2 < out2.len() {
            // SAFETY: as for `out`
            unsafe { *out2.get_unchecked_mut(o2) = v };
        }
    }
}

/// One packed pair of [`u3_apply`]'s result: element `e` of `x` (and of the residual), channels `c, c + 1`
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn apply_pair(
    x: &[u32],
    bias: &[f32],
    stats: &[[f32; 2]],
    w: &[f32],
    b: &[f32],
    res: &[u32],
    rraw: &[u32],
    rstats: &[[f32; 2]],
    rw: &[f32],
    rb: &[f32],
    e: u32,
    c: u32,
    act: u32,
    slope: f32,
) -> u32 {
    let p = x[e as usize];
    let [m0, i0] = stats[c as usize];
    let [m1, i1] = stats[c as usize + 1];
    let mut y0 = affine(biased(lo(p), bias, c), m0, i0, w[c as usize], b[c as usize]);
    let mut y1 = affine(biased(hi(p), bias, c + 1), m1, i1, w[c as usize + 1], b[c as usize + 1]);
    if !res.is_empty() {
        let r = res[e as usize];
        y0 = rbf(add_rn_f32(y0, lo(r)));
        y1 = rbf(add_rn_f32(y1, hi(r)));
    } else if !rraw.is_empty() {
        let r = rraw[e as usize];
        let [n0, j0] = rstats[c as usize];
        let [n1, j1] = rstats[c as usize + 1];
        y0 = rbf(add_rn_f32(y0, affine(lo(r), n0, j0, rw[c as usize], rb[c as usize])));
        y1 = rbf(add_rn_f32(y1, affine(hi(r), n1, j1, rw[c as usize + 1], rb[c as usize + 1])));
    }
    if act != 0 {
        y0 = leaky(y0, slope);
        y1 = leaky(y1, slope);
    }
    pack(y0, y1)
}

/// [`u3_apply`] for an encoder stage's last block, also writing the next stage's pooled skip input
///
/// One thread per pooled pixel and channel pair: it applies the 2x2x2 block it pools, writes each pair to
/// `out` (and `out2`) exactly as [`u3_apply`] does, and writes `pool[q] = rbf(sum / 8)` of the packed
/// results summed in [`u3_pool2`]'s order, so `pool` equals `u3_pool2` of `out` bit for bit without
/// re-reading the full-resolution output (0.74 ms of a 1 GiB level-0 tensor). `d`, `h`, `wd` are the pooled
/// dims; `x`, the residual and `out` cover `[2d, 2h, 2wd]` pixels of `cols` channels
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn u3_apply_pool(
    mut out: DisjointSlice<u32>,
    mut out2: DisjointSlice<u32>,
    mut pool: DisjointSlice<u32>,
    x: &[u32],
    bias: &[f32],
    stats: &[[f32; 2]],
    w: &[f32],
    b: &[f32],
    res: &[u32],
    rraw: &[u32],
    rstats: &[[f32; 2]],
    rw: &[f32],
    rb: &[f32],
    d: u32,
    h: u32,
    wd: u32,
    cols: u32,
    act: u32,
    slope: f32,
    out_stride: u32,
    out_off: u32,
    out2_stride: u32,
    out2_off: u32,
) {
    let e = thread::index_1d_u32(lc).get();
    let pairs = cols / 2;
    if pairs == 0 || e >= d * h * wd * pairs {
        return;
    }
    let (q, cp) = (e / pairs, e % pairs);
    let (ox, oy, oz) = (q % wd, (q / wd) % h, q / (wd * h));
    let (iw, ih) = (2 * wd, 2 * h);
    let (mut s0, mut s1) = (0f32, 0f32);
    let mut dz = 0;
    while dz < 2 {
        let mut dy = 0;
        while dy < 2 {
            let mut dx = 0;
            while dx < 2 {
                let ip = ((2 * oz + dz) * ih + 2 * oy + dy) * iw + 2 * ox + dx;
                let v = apply_pair(x, bias, stats, w, b, res, rraw, rstats, rw, rb, ip * pairs + cp, 2 * cp, act, slope);
                let o = (ip * out_stride + out_off + cp) as usize;
                if o < out.len() {
                    // SAFETY: in bounds; each full-resolution (pixel, pair) belongs to one pooled thread
                    unsafe { *out.get_unchecked_mut(o) = v };
                }
                if !out2.is_empty() {
                    let o2 = (ip * out2_stride + out2_off + cp) as usize;
                    if o2 < out2.len() {
                        // SAFETY: as for `out`
                        unsafe { *out2.get_unchecked_mut(o2) = v };
                    }
                }
                s0 = add_rn_f32(s0, lo(v));
                s1 = add_rn_f32(s1, hi(v));
                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }
    if (e as usize) < pool.len() {
        // SAFETY: in bounds; thread `e` alone owns pooled pair `e`
        unsafe { *pool.get_unchecked_mut(e as usize) = pack(mul_rn_f32(s0, 0.125), mul_rn_f32(s1, 0.125)) };
    }
}

/// `AvgPool3d(2, 2)` on bf16 pairs: `out[z, y, x] = rbf(sum / 8)` of the 2x2x2 input block, summed in
/// float32 in ATen's order (z, then y, then x). `d`, `h`, `w` are the output's dims; the input is
/// `[2d, 2h, 2w, cols]`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn u3_pool2(mut out: DisjointSlice<u32>, x: &[u32], d: u32, h: u32, w: u32, cols: u32) {
    let t = thread::index_1d_u32(lc);
    let e = t.get();
    let pairs = cols / 2;
    let Some(mut o) = out.element_thread32(t) else { return };
    if pairs == 0 || e >= d * h * w * pairs {
        return;
    }
    let cp = e % pairs;
    let pix = e / pairs;
    let (ox, oy, oz) = (pix % w, (pix / w) % h, pix / (w * h));
    let (iw, ih) = (2 * w, 2 * h);
    let (mut s0, mut s1) = (0f32, 0f32);
    let mut dz = 0;
    while dz < 2 {
        let mut dy = 0;
        while dy < 2 {
            let mut dx = 0;
            while dx < 2 {
                let ip = ((2 * oz + dz) * ih + 2 * oy + dy) * iw + 2 * ox + dx;
                let v = x[(ip * pairs + cp) as usize];
                s0 = add_rn_f32(s0, lo(v));
                s1 = add_rn_f32(s1, hi(v));
                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }
    // dividing by 8 is exact as a multiply by 0.125
    o.write(pack(mul_rn_f32(s0, 0.125), mul_rn_f32(s1, 0.125)));
}

/// Scatters a k2 s2 transposed conv's GEMM output into its place, adding the bf16 bias:
/// `out[(2z + a, 2y + b, 2x + c), ch] = rbf(g[(z, y, x), tap * cols + ch] + bias[ch])`, tap `4a + 2b + c`
///
/// `g` is `[d/2 * h/2 * w/2, 8 * cols]` bf16 pairs; `d`, `h`, `w` are the output's dims. The result goes
/// to pair `pix * out_stride + out_off + cp` of `out`, the decoder stage's concat buffer
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn u3_up(
    mut out: DisjointSlice<u32>,
    g: &[u32],
    bias: &[f32],
    d: u32,
    h: u32,
    w: u32,
    cols: u32,
    out_stride: u32,
    out_off: u32,
) {
    let e = thread::index_1d_u32(lc).get();
    let pairs = cols / 2;
    if pairs == 0 || e >= d * h * w * pairs {
        return;
    }
    let cp = e % pairs;
    let pix = e / pairs;
    let (x, y, z) = (pix % w, (pix / w) % h, pix / (w * h));
    let (hw, hh) = (w / 2, h / 2);
    let src = ((z / 2) * hh + y / 2) * hw + x / 2;
    let tap = (z % 2) * 4 + (y % 2) * 2 + x % 2;
    let v = g[((src * 8 + tap) * pairs + cp) as usize];
    let c = 2 * cp;
    let r = pack(add_rn_f32(lo(v), bias[c as usize]), add_rn_f32(hi(v), bias[c as usize + 1]));
    let o = (pix * out_stride + out_off + cp) as usize;
    if o < out.len() {
        // SAFETY: in bounds; each thread writes its own (pixel, pair) slot
        unsafe { *out.get_unchecked_mut(o) = r };
    }
}

/// The 256-bin histogram of the in-volume voxels of one patch, added into `hist` (zeroed by the host)
///
/// The patch's corner is `(oz, oy, ox)` in a uint8 slab of dims `(sd, sh, sw)`, edge `p` voxels on every
/// axis; voxels with patch coordinates in `[v0, v1)` on each axis are inside the volume. Grid `p`
/// blocks, one per patch z-slice, 256 threads
#[kernel]
pub fn u3_hist(
    mut hist: DisjointSlice<u32>,
    slab: &[u8],
    sh: u32,
    sw: u32,
    oz: u32,
    oy: u32,
    ox: u32,
    p: u32,
    vz0: u32,
    vz1: u32,
    vy0: u32,
    vy1: u32,
    vx0: u32,
    vx1: u32,
) {
    static mut BINS: SharedArray<u32, 256> = SharedArray::UNINIT;
    let (lz, tid) = (blockIdx_x(), threadIdx_x());
    // block-uniform
    if blockDim_x() != 256 || hist.len() < 256 || lz >= p {
        return;
    }
    let bins = addr_of_mut!(BINS) as *mut u32;
    // SAFETY: thread `tid < 256` writes its own bin before the barrier
    unsafe { *bins.add(tid as usize) = 0 };
    sync_threads();
    if lz >= vz0 && lz < vz1 {
        let (nx, ny) = (vx1 - vx0, vy1 - vy0);
        let n = nx * ny;
        let mut i = tid;
        while i < n {
            let (ly, lx) = (vy0 + i / nx, vx0 + i % nx);
            let v = slab[(((oz + lz) * sh + oy + ly) * sw + ox + lx) as usize];
            // SAFETY: `v < 256`, a bin of this block's shared array
            unsafe { BlockAtomicU32::from_ptr(bins.add(v as usize)).fetch_add(1, AtomicOrdering::Relaxed) };
            i += 256;
        }
    }
    sync_threads();
    // SAFETY: the bins are complete after the barrier
    let count = unsafe { *bins.add(tid as usize) };
    if count != 0 {
        // SAFETY: `tid < 256 <= hist.len()`; device-scope atomics add the blocks' counts
        unsafe { DeviceAtomicU32::from_ptr(hist.as_mut_ptr().add(tid as usize)).fetch_add(count, AtomicOrdering::Relaxed) };
    }
}

/// One patch's model input: `table[code]` rounded to bf16 for in-volume voxels, 0 elsewhere (villa
/// zero-pads outside the volume without normalising). Writes channel 0 of `x`, `[p^3, cstride]` bf16;
/// the host zero-fills the other channels once. Geometry as [`u3_hist`]
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn u3_prep(
    mut x: DisjointSlice<u16>,
    slab: &[u8],
    table: &[f32],
    sh: u32,
    sw: u32,
    oz: u32,
    oy: u32,
    ox: u32,
    p: u32,
    vz0: u32,
    vz1: u32,
    vy0: u32,
    vy1: u32,
    vx0: u32,
    vx1: u32,
    cstride: u32,
) {
    let e = thread::index_1d_u32(lc).get();
    if e >= p * p * p || table.len() < 256 {
        return;
    }
    let (lx, ly, lz) = (e % p, (e / p) % p, e / (p * p));
    let inside = lz >= vz0 && lz < vz1 && ly >= vy0 && ly < vy1 && lx >= vx0 && lx < vx1;
    let v = if inside {
        let code = slab[(((oz + lz) * sh + oy + ly) * sw + ox + lx) as usize];
        f32_to_bf16_rne(table[code as usize])
    } else {
        0
    };
    let o = (e * cstride) as usize;
    if o < x.len() {
        // SAFETY: in bounds; each thread writes its own voxel's channel 0
        unsafe { *x.get_unchecked_mut(o) = v };
    }
}

/// The head, sigmoid and blend of one patch in one pass
///
/// Per patch voxel: `logit = rbf(rbf(sum_c x[c] * hw[c]) + hb)`, `p = 1 / (1 + exp(-logit))`, then into
/// the ring accumulators `acc += p * imp` and `wacc += imp`, each product and sum rounded on its own
/// as villa's float32 blend does. `x` is the last decoder stage `[p^3, cols]` bf16 pairs and `hw`, `hb`
/// the head's bf16 weight and bias as float32. `imp` is the `[p^3]` importance map.
///
/// The patch's corner is global `(pz, py, px)`; the accumulators cover the region `[rz0, rz0 + rd)` x
/// `[ry0, ry0 + rh)` x `[rx0, rx0 + rw)`, z stored modulo `ring` voxels. Voxels outside the region are
/// skipped
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn u3_head_blend(
    mut acc: DisjointSlice<f32>,
    mut wacc: DisjointSlice<f32>,
    x: &[u32],
    hw: &[f32],
    hb: f32,
    imp: &[f32],
    cols: u32,
    p: u32,
    pz: u32,
    py: u32,
    px: u32,
    rz0: u32,
    ry0: u32,
    rx0: u32,
    rd: u32,
    rh: u32,
    rw: u32,
    ring: u32,
) {
    let e = thread::index_1d_u32(lc).get();
    let pairs = cols / 2;
    if e >= p * p * p || pairs == 0 || hw.len() < cols as usize || ring == 0 {
        return;
    }
    let (lx, ly, lz) = (e % p, (e / p) % p, e / (p * p));
    let (gz, gy, gx) = (pz + lz, py + ly, px + lx);
    if gz < rz0 || gz >= rz0 + rd || gy < ry0 || gy >= ry0 + rh || gx < rx0 || gx >= rx0 + rw {
        return;
    }
    let mut dot = 0f32;
    let mut cp = 0;
    while cp < pairs {
        let v = x[(e * pairs + cp) as usize];
        dot = fma_rn_f32(lo(v), hw[(2 * cp) as usize], dot);
        dot = fma_rn_f32(hi(v), hw[(2 * cp + 1) as usize], dot);
        cp += 1;
    }
    let logit = rbf(add_rn_f32(rbf(dot), hb));
    let prob = 1.0 / (1.0 + math::expf(-logit));
    let wt = imp[e as usize];
    let o = ((((gz - rz0) % ring) * rh + gy - ry0) * rw + gx - rx0) as usize;
    if o < acc.len() && o < wacc.len() {
        // SAFETY: in bounds; the voxels of one patch map to distinct accumulator slots, and one
        // patch's launch finishes before the next queues on the stream
        unsafe {
            let a = acc.get_unchecked_mut(o);
            *a = add_rn_f32(*a, mul_rn_f32(prob, wt));
            let s = wacc.get_unchecked_mut(o);
            *s = add_rn_f32(*s, wt);
        }
    }
}

/// Finishes accumulated voxels as villa writes them: the weight is taken as 1 where it is at most
/// 1e-6, then `rint(clamp(acc / weight, 0, 1) * 255)` as uint8
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn u3_finish(mut out: DisjointSlice<u8>, acc: &[f32], wacc: &[f32]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get() as usize;
    let Some(mut o) = out.element_thread32(t) else { return };
    if i >= acc.len() || i >= wacc.len() {
        return;
    }
    let w = wacc[i];
    let w = if w <= 1e-6 { 1.0 } else { w };
    let v = math::fminf(math::fmaxf(acc[i] / w, 0.0), 1.0);
    // SAFETY: libdevice's rintf, a pure function
    o.write(unsafe { __nv_rintf(mul_rn_f32(v, 255.0)) } as u8);
}

/// villa's coverage rule for the gate: 1 where the accumulated blend weight exceeds 1e-6, the voxels
/// villa's finish divides by their weight rather than by 1
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn u3_covered(mut out: DisjointSlice<u8>, wacc: &[f32]) {
    let t = thread::index_1d_u32(lc);
    let i = t.get() as usize;
    let Some(mut o) = out.element_thread32(t) else { return };
    if i >= wacc.len() {
        return;
    }
    o.write(if wacc[i] > 1e-6 { 1 } else { 0 });
}
