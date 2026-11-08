//! Tile gather and villa's robust normalisation, straight from the resident uint8 volume
//!
//! A tile is described by 8 `i32`s (see [`DESC`]): its top-left corner, the source layer of
//! its first depth slice and the layer step (+1 forward, -1 reverse), and the slot its logits
//! go to (-1 for the padding tiles of a short last batch). The volume is `[Z, H, W]` uint8.
//!
//! Normalisation is villa `normalize_robust` on the float32 patch: clip to numpy's 1st and
//! 99th percentiles, subtract the median, divide by 1.4826 * MAD, with numpy's fallbacks.
//! The input is uint8, so a 256-bin histogram per tile gives every order statistic exactly,
//! and [`tile_stats`] replays numpy's float32 arithmetic on them

use core::ptr::addr_of_mut;

use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicU32};
use cuda_device::thread::{blockDim_x, blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread};

use crate::common::{add_rn_f32, math, mul_rn_f32, pack_f16};

/// `i32`s per tile descriptor: y0, x0, first layer, layer step, slot, the mirror4 flip bits
/// (`crate::tta`; 0 for a plain read), then 2 unused
pub const DESC: usize = 8;
/// Tile edge in pixels
pub const TILE: u32 = 128;
/// Depth slices per tile
pub const DEPTH: u32 = 17;
/// Voxels per tile
pub const TILE_VOXELS: u32 = DEPTH * TILE * TILE;

/// Histogram of every tile's voxels into `hist` (`[tiles, 256]`, zeroed by the host)
///
/// Grid `(DEPTH, tiles)`, 256 threads: block `(z, b)` counts depth slice `z` of tile `b` in
/// shared memory, then adds its counts to the tile's row. `height` and `width` are the
/// volume's; every tile lies inside it, which the host checks
#[kernel]
pub fn tile_hist(hist: &[DeviceAtomicU32], vol: &[u8], desc: &[i32], height: u32, width: u32) {
    static mut BINS: SharedArray<u32, 256> = SharedArray::UNINIT;
    let (z, b, tid) = (blockIdx_x(), blockIdx_y(), threadIdx_x());
    let d = (b as usize) * DESC;
    // block-uniform, so a block that fails it never reaches a barrier
    if d + DESC > desc.len() || ((b + 1) as usize) * 256 > hist.len() {
        return;
    }
    let (y0, x0, l0, step) = (desc[d] as u32, desc[d + 1] as u32, desc[d + 2], desc[d + 3]);
    let layer = (l0 + step * z as i32) as u32;
    let plane = layer * height * width;
    let bins = addr_of_mut!(BINS) as *mut u32;

    // SAFETY: 256 threads each clear their own bin before the barrier
    unsafe { *bins.add(tid as usize) = 0 };
    sync_threads();
    let mut i = tid;
    while i < TILE * TILE {
        let (y, x) = (i >> 7, i & 127);
        let v = vol[(plane + (y0 + y) * width + x0 + x) as usize];
        // SAFETY: `v < 256` indexes the 256-bin shared array, which between the two
        // barriers is only accessed atomically and only by this block
        unsafe { BlockAtomicU32::from_ptr(bins.add(v as usize)) }.fetch_add(1, AtomicOrdering::Relaxed);
        i += 256;
    }
    sync_threads();
    // SAFETY: after the barrier every bin is final and read by its own thread only
    let count = unsafe { *bins.add(tid as usize) };
    if count != 0 {
        hist[(b * 256 + tid) as usize].fetch_add(count, AtomicOrdering::Relaxed);
    }
}

/// The 0-based order statistic `k` of a tile from its histogram: the smallest grey value
/// whose inclusive cumulative count exceeds `k`
#[inline(always)]
fn order_stat(h: &[u32], base: usize, k: u32) -> f32 {
    let mut cum = 0u32;
    let mut v = 0u32;
    while v < 256 {
        cum += h[base + v as usize];
        if cum > k {
            return v as f32;
        }
        v += 1;
    }
    255.0
}

/// numpy's `_lerp(a, b, t)` as `np.percentile` runs it on a float32 array: float32 values and
/// weight, but `1 - t` taken in float64 before its cast (`omt`), for the `t >= 0.5` branch
/// `b - (b - a) * (1 - t)`. Checked against numpy 2.5.3 on all 32896 grey pairs for both
/// percentiles at the tile size. The product and the sum round separately, as numpy's do; a
/// plain `a + d * t` would contract to one FMA
#[inline(always)]
fn lerp32(a: f32, b: f32, t: f32, omt: f32) -> f32 {
    let d = b - a;
    if t >= 0.5 { add_rn_f32(b, -mul_rn_f32(d, omt)) } else { add_rn_f32(a, mul_rn_f32(d, t)) }
}

/// fastink's `_pct_sorted` (the PHerc. 0306B recipe's `robust_norm_gpu`, torch float32):
/// `a * (1 - t) + b * t`, both weights cast to float32 by torch and every operation its own
/// float32 rounding, since each is a separate torch kernel
#[inline(always)]
fn lerp_torch(a: f32, b: f32, t: f32, omt: f32) -> f32 {
    add_rn_f32(mul_rn_f32(a, omt), mul_rn_f32(b, t))
}

/// Per tile `[lo, hi, median, divisor]` of villa `normalize_robust`, one thread per tile
///
/// `n` is the voxel count per tile. For the lower and then the upper percentile, the host passes
/// numpy's interpolation index, its weight `t` and `1 - t` (see [`lerp32`]), computed exactly
/// as numpy computes them. `fastink` 1 replays instead the float32 torch arithmetic of the 306B
/// recipe's `robust_norm_gpu` (the TTA reads, `crate::tta`): [`lerp_torch`] with the host's
/// torch weights, and the MAD scale and the half span in float32. Its std fallback is still
/// numpy's (torch's float32 reduction order is not replayed)
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn tile_stats(
    mut stats: DisjointSlice<[f32; 4]>,
    hist: &[u32],
    n: u32,
    lo_index: u32,
    lo_t: f32,
    lo_omt: f32,
    hi_index: u32,
    hi_t: f32,
    hi_omt: f32,
    fastink: u32,
) {
    let t = thread::index_1d_u32(lc);
    let b = t.get() as usize;
    let Some(mut out) = stats.element_thread32(t) else { return };
    let base = b * 256;
    if base + 256 > hist.len() || n < 2 {
        return;
    }

    let torch = fastink != 0;
    let lerp = |a: f32, b: f32, t: f32, omt: f32| if torch { lerp_torch(a, b, t, omt) } else { lerp32(a, b, t, omt) };
    let lo = lerp(order_stat(hist, base, lo_index), order_stat(hist, base, lo_index + 1), lo_t, lo_omt);
    let hi = lerp(order_stat(hist, base, hi_index), order_stat(hist, base, hi_index + 1), hi_t, hi_omt);
    let clip = |v: u32| math::fminf(math::fmaxf(v as f32, lo), hi);

    // the median of the clipped patch is the clipped median, since clipping is monotone; numpy
    // averages the two middle values of an even count in float32
    let (k1, k2) = if n % 2 == 0 { (n / 2 - 1, n / 2) } else { (n / 2, n / 2) };
    let m1 = math::fminf(math::fmaxf(order_stat(hist, base, k1), lo), hi);
    let m2 = math::fminf(math::fmaxf(order_stat(hist, base, k2), lo), hi);
    let median = if k1 == k2 { m1 } else { (m1 + m2) / 2.0 };

    // MAD: the deviations |clip(v) - median| grow walking away from the median on either
    // side, so merging the two sides in order visits them sorted
    let mut right = 0u32;
    while right < 256 && clip(right) <= median {
        right += 1;
    }
    let mut left = right as i32 - 1;
    let (mut cum, mut d1, mut d2, mut found1) = (0u32, 0f32, 0f32, false);
    loop {
        let dl = if left >= 0 { median - clip(left as u32) } else { f32::INFINITY };
        let dr = if right < 256 { clip(right) - median } else { f32::INFINITY };
        if dl == f32::INFINITY && dr == f32::INFINITY {
            break;
        }
        let (dev, count) = if dl <= dr {
            let c = hist[base + left as usize];
            left -= 1;
            (dl, c)
        } else {
            let c = hist[base + right as usize];
            right += 1;
            (dr, c)
        };
        cum += count;
        if !found1 && cum > k1 {
            d1 = dev;
            found1 = true;
        }
        if cum > k2 {
            d2 = dev;
            break;
        }
    }
    let mad = if k1 == k2 { d2 } else { (d1 + d2) / 2.0 };
    // numpy scales in float64; torch multiplies the float32 tensor by 1.4826 cast to float32
    let scaled = if torch { mul_rn_f32(1.4826f32, mad) as f64 } else { 1.4826f64 * mad as f64 };

    // numpy's fallback is the pre-clip std, else half the percentile span, else 1. The std
    // needs every voxel in numpy's summation order, which [`tile_std`] replays; here a
    // negative divisor marks the tile for it and carries the next fallback
    let divisor = if scaled >= 1e-6 {
        scaled as f32
    } else {
        let span = if torch { (add_rn_f32(hi, -lo) / 2.0) as f64 } else { (hi as f64 - lo as f64) / 2.0 };
        -(if span >= 1e-6 { span as f32 } else { 1.0 })
    };
    out.write([lo, hi, median, divisor]);
}

/// Elements per node of numpy's pairwise-sum tree on one tile: at `n = 17 * 128 * 128` every
/// halving lands on a multiple of 8 down to 2048 nodes of 136, each split into a 64- and a
/// 72-element leaf (see `pairwise_f32` and its tree test on the host)
const NODE: u32 = 136;
const NODES: u32 = TILE_VOXELS / NODE;

/// Voxel `e` of a tile in villa's ZYX order (`e = z * 128 * 128 + y * 128 + x`) as float
#[inline(always)]
fn voxel(vol: &[u8], plane0: i32, step_plane: i32, row0: u32, width: u32, e: u32) -> f32 {
    let (z, yx) = (e >> 14, e & 16383);
    let plane = (plane0 + step_plane * z as i32) as u32;
    vol[(plane + row0 + (yx >> 7) * width + (yx & 127)) as usize] as f32
}

/// One leaf of numpy's float32 pairwise sum: 8 running sums, then
/// `((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))`; with `mean`, of `(v - mean)^2`
#[inline(always)]
fn leaf(vol: &[u8], p0: i32, sp: i32, row0: u32, width: u32, start: u32, len: u32, mean: f32, square: bool) -> f32 {
    let val = |e: u32| {
        let v = voxel(vol, p0, sp, row0, width, e);
        if square {
            let d = v - mean;
            mul_rn_f32(d, d)
        } else {
            v
        }
    };
    let mut r = [0f32; 8];
    for j in 0..8 {
        r[j] = val(start + j as u32);
    }
    let mut i = 8;
    while i < len {
            for j in 0..8 {
            r[j] = add_rn_f32(r[j], val(start + i + j as u32));
        }
        i += 8;
    }
    add_rn_f32(add_rn_f32(add_rn_f32(r[0], r[1]), add_rn_f32(r[2], r[3])), add_rn_f32(add_rn_f32(r[4], r[5]), add_rn_f32(r[6], r[7])))
}

/// numpy's float32 pairwise sum over a whole tile, by a 256-thread block; every thread gets
/// the total
#[inline(always)]
fn tile_sum(nodes: *mut f32, vol: &[u8], p0: i32, sp: i32, row0: u32, width: u32, mean: f32, square: bool) -> f32 {
    let tid = threadIdx_x();
    let mut k = tid;
    while k < NODES {
        let a = leaf(vol, p0, sp, row0, width, k * NODE, 64, mean, square);
        let b = leaf(vol, p0, sp, row0, width, k * NODE + 64, 72, mean, square);
        // SAFETY: node k is this thread's alone until the barrier
        unsafe { *nodes.add(k as usize) = add_rn_f32(a, b) };
        k += 256;
    }
    sync_threads();
    // the tree pairs adjacent nodes level by level; read every pair before any write
    let mut m = NODES;
    while m > 1 {
        let half = m / 2;
        let mut r = [0f32; 4];
            for q in 0..4 {
            let i = tid + 256 * q as u32;
            if i < half {
                // SAFETY: indices below m <= NODES, read between barriers
                r[q] = unsafe { add_rn_f32(*nodes.add(2 * i as usize), *nodes.add(2 * i as usize + 1)) };
            }
        }
        sync_threads();
            for q in 0..4 {
            let i = tid + 256 * q as u32;
            if i < half {
                // SAFETY: each output index has one writer
                unsafe { *nodes.add(i as usize) = r[q] };
            }
        }
        sync_threads();
        m = half;
    }
    // SAFETY: final after the last barrier; the caller syncs before reusing `nodes`
    unsafe { *nodes }
}

/// numpy's exact float32 `np.std` for the tiles [`tile_stats`] marked (negative divisor),
/// replacing the divisor with it, or with the carried fallback when the std is below 1e-6
///
/// Grid `(tiles)`, 256 threads. Unmarked tiles return after one read. `np.std` sums the
/// float32 patch pairwise for the mean, divides by n, then sums the rounded squared
/// deviations pairwise, divides and takes the square root, all in float32
#[kernel]
pub fn tile_std(mut stats: DisjointSlice<[f32; 4]>, vol: &[u8], desc: &[i32], height: u32, width: u32) {
    static mut TREE: SharedArray<f32, 2048> = SharedArray::UNINIT;
    static mut MARK: SharedArray<f32, 1> = SharedArray::UNINIT;
    let (b, tid) = (blockIdx_x(), threadIdx_x());
    let d = b as usize * DESC;
    // block-uniform, so a block that fails it never reaches a barrier
    if d + DESC > desc.len() || b as usize >= stats.len() || blockDim_x() != 256 || NODES != 2048 {
        return;
    }
    let mark = addr_of_mut!(MARK) as *mut f32;
    if tid == 0 {
        // SAFETY: only thread 0 touches this tile's stats; `b` is in bounds
        unsafe { *mark = (*stats.get_unchecked_mut(b as usize))[3] };
    }
    sync_threads();
    // SAFETY: written before the barrier
    let marker = unsafe { *mark };
    if marker >= 0.0 {
        return;
    }
    let (y0, x0, l0, step) = (desc[d] as u32, desc[d + 1] as u32, desc[d + 2], desc[d + 3]);
    let hw = (height * width) as i32;
    let row0 = y0 * width + x0;
    let nodes = addr_of_mut!(TREE) as *mut f32;
    let n = TILE_VOXELS as f32;
    let mean = tile_sum(nodes, vol, l0 * hw, step * hw, row0, width, 0.0, false) / n;
    sync_threads();
    let var = tile_sum(nodes, vol, l0 * hw, step * hw, row0, width, mean, true) / n;
    let std = math::sqrtf(var);
    if tid == 0 {
        // SAFETY: as above
        unsafe { (*stats.get_unchecked_mut(b as usize))[3] = if std as f64 >= 1e-6 { std } else { -marker } };
    }
}

/// The normalised FP16 input `[tiles, DEPTH, TILE, TILE]` (two voxels per `u32`), the conv3d
/// input with its single channel: `(clip(v, lo, hi) - median) / divisor` in float32, then
/// rounded to FP16 as autocast does
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn tile_norm(
    mut out: DisjointSlice<u32>,
    vol: &[u8],
    desc: &[i32],
    stats: &[[f32; 4]],
    height: u32,
    width: u32,
) {
    let t = thread::index_1d_u32(lc);
    let e = t.get();
    let per_tile = TILE_VOXELS / 2;
    let (b, r) = (e / per_tile, e % per_tile);
    let (z, yx) = (r / (TILE * TILE / 2), r % (TILE * TILE / 2));
    let (y, x) = (yx >> 6, (yx & 63) * 2);
    let d = b as usize * DESC;
    let Some(mut out) = out.element_thread32(t) else { return };
    let [lo, hi, median, divisor] = stats[b as usize];
    let (y0, x0, l0, step) = (desc[d] as u32, desc[d + 1] as u32, desc[d + 2], desc[d + 3]);
    let layer = (l0 + step * z as i32) as u32;
    // a mirror4 copy (crate::tta) writes the tile flipped by field 5's bits, 1 in y and 2 in x, so the
    // pair (x, x + 1) then reads columns (TILE - 1 - x, TILE - 2 - x); the statistics stay the tile's own
    let flip = desc[d + 5];
    let sy = if flip & 1 != 0 { TILE - 1 - y } else { y };
    let (xa, xb) = if flip & 2 != 0 { (TILE - 1 - x, TILE - 2 - x) } else { (x, x + 1) };
    let row = layer * height * width + (y0 + sy) * width + x0;
    let f = |v: u8| (math::fminf(math::fmaxf(v as f32, lo), hi) - median) / divisor;
    out.write(pack_f16(f(vol[(row + xa) as usize]), f(vol[(row + xb) as usize])));
}
