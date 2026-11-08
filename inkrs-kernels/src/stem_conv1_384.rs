//! The stem's first conv3d with its InstanceNorm fused, for 17 x 384 x 384 tiles (Reader V3)
//!
//! [`crate::stem_conv1`] at a 384 edge, in its own kernels so the 128-edge PTX stays unchanged. A block
//! covers one (sample, depth slice, band of 8 rows, strip of 128 columns): three strips per band. Its
//! halo is the same 3 x 10 x 130 FP16 array as at 128, but the two edge columns hold the neighbouring
//! strips' voxels where they lie inside the tile, and zeros only at the tile's edge. The GEMM tiling,
//! the fragment layouts, the shift and the FP16 rounding are the 128 kernels' unchanged.
//! [`stem1_finalize_384`] merges the 2448 partials per (sample, channel) with one warp each, in float64

use core::ptr::addr_of_mut;

use cuda_device::thread::{blockDim_x, blockIdx_x, blockIdx_y, blockIdx_z, sync_threads, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, warp};

use crate::common::{leaky16, math, mma_m16n8k16_f32_f16, pack_f16, r16, widen};
use crate::inorm::affine16;

/// Depth slices, rows and columns of a tile
const Z: u32 = 17;
const H: u32 = 384;
const W: u32 = 384;
/// Output channels
const K: usize = 16;
/// Kernel taps
const TAPS: usize = 27;
/// Rows and columns per block, and the halo's rows and columns
const BAND: u32 = 8;
const SW: u32 = 128;
const STRIPS: u32 = W / SW;
const HR: usize = BAND as usize + 2;
const HC: usize = SW as usize + 2;
const HALO: usize = 3 * HR * HC;
/// Threads per block: four warps, one column each while the halo loads
const BLOCK: u32 = SW;
/// GEMM tiles per block: 16 columns of one row each
const MTILES: u32 = BAND * SW / 16;
/// Blocks per depth slice, and per sample
const SLICE_BLOCKS: u32 = (H / BAND) * STRIPS;
pub const STEM1_CHUNKS_384: u32 = Z * SLICE_BLOCKS;

/// Fills the block's input halo, `[3][HR][HC]` FP16 bits, zeros outside the tile, from `x` (`[B, Z, H,
/// W]` FP16 bits): thread `tid` copies column `col0 + tid` of each of the 30 halo rows; thread 0 the
/// column left of the strip and thread 1 the column right of it
///
/// # Safety
/// `halo` points to a block-shared array of `HALO` values; `x` holds sample `b`, `tid < SW`, and
/// `col0 + SW <= W`, all checked by the caller
#[inline(always)]
unsafe fn load_halo(halo: *mut u16, x: &[u16], b: u32, z: u32, y0: u32, col0: u32, tid: u32) {
    let mut pr = 0;
    while pr < 3 * HR {
        let (plane, r) = (pr / HR, pr % HR);
        let (zz, yy) = (z as i32 + plane as i32 - 1, y0 as i32 + r as i32 - 1);
        let row_in = (zz >= 0) & (zz < Z as i32) & (yy >= 0) & (yy < H as i32);
        let row = ((b * Z + zz as u32) * H + yy as u32) * W;
        let v = if row_in {
            // SAFETY: inside the tile of sample `b`, which the caller checked `x` holds
            unsafe { *x.get_unchecked((row + col0 + tid) as usize) }
        } else {
            0
        };
        // SAFETY: pr < 3 * HR and 1 + tid < HC
        unsafe { *halo.add(pr * HC + 1 + tid as usize) = v };
        if tid < 2 {
            // thread 0: column col0 - 1; thread 1: column col0 + SW
            let xx = col0 as i32 + if tid == 0 { -1 } else { SW as i32 };
            let e = if row_in & (xx >= 0) & (xx < W as i32) {
                // SAFETY: inside the tile of sample `b`
                unsafe { *x.get_unchecked((row + xx as u32) as usize) }
            } else {
                0
            };
            // SAFETY: column 0 or HC - 1 of row pr
            unsafe { *halo.add(pr * HC + tid as usize * (HC - 1)) = e };
        }
        pr += 1;
    }
}

/// Lane `lane`'s B fragments, as [`crate::stem_conv1`]'s
///
/// # Safety
/// `w` holds `K * TAPS` weights, checked by the caller
#[inline(always)]
unsafe fn b_frags(w: &[u16], lane: u32) -> [[[u32; 2]; 2]; 2] {
    let (g, th) = ((lane / 4) as usize, (lane % 4) as usize);
    let at = |t: usize, ch: usize| -> u32 {
        // SAFETY: t < TAPS and ch < K
        if t < TAPS { unsafe { *w.get_unchecked(ch * TAPS + t) as u32 } } else { 0 }
    };
    let pair = |t: usize, ch: usize| at(t, ch) | (at(t + 1, ch) << 16);
    let f = |s: usize, n: usize| {
        let (t, ch) = (16 * s + 2 * th, 8 * n + g);
        [pair(t, ch), pair(t + 8, ch)]
    };
    [[f(0, 0), f(0, 1)], [f(1, 0), f(1, 1)]]
}

/// Lane `lane`'s A fragment tap offsets and masks, as [`crate::stem_conv1`]'s
struct Taps {
    offs: [[u32; 4]; 2],
    mask: [[u32; 2]; 2],
}

#[inline(always)]
fn taps(lane: u32) -> Taps {
    let th = (lane % 4) as usize;
    let off = |t: usize| -> u32 { if t < TAPS { ((t / 9) * HR * HC + ((t / 3) % 3) * HC + t % 3) as u32 } else { 0 } };
    let real = |t: usize| -> u32 { (if t < TAPS { 0xffff } else { 0 }) | (if t + 1 < TAPS { 0xffff_0000 } else { 0 }) };
    let f = |s: usize| {
        let t = 16 * s + 2 * th;
        [off(t), off(t + 1), off(t + 8), off(t + 9)]
    };
    let m = |s: usize| {
        let t = 16 * s + 2 * th;
        [real(t), real(t + 8)]
    };
    Taps { offs: [f(0), f(1)], mask: [m(0), m(1)] }
}

/// One GEMM tile at the 16 voxels `base + 0..16` of the halo, as [`crate::stem_conv1`]'s
///
/// # Safety
/// `halo` is [`load_halo`]'s array, filled and synchronised; `base + 15 + 2 HC + 2 + 2 HR HC < HALO`.
/// Every lane of the warp calls this together
#[inline(always)]
unsafe fn tile(halo: *const u16, base: usize, g: u32, tp: &Taps, bf: &[[[u32; 2]; 2]; 2]) -> [[f32; 4]; 2] {
    let (r0, r1) = (base + g as usize, base + g as usize + 8);
    // SAFETY: the caller's bound
    let h = |i: usize| unsafe { *halo.add(i) as u32 };
    let a = |s: usize| {
        let (o, m) = (tp.offs[s], tp.mask[s]);
        let (o0, o1, o2, o3) = (o[0] as usize, o[1] as usize, o[2] as usize, o[3] as usize);
        [
            (h(r0 + o0) | (h(r0 + o1) << 16)) & m[0],
            (h(r1 + o0) | (h(r1 + o1) << 16)) & m[0],
            (h(r0 + o2) | (h(r0 + o3) << 16)) & m[1],
            (h(r1 + o2) | (h(r1 + o3) << 16)) & m[1],
        ]
    };
    let (a0, a1) = (a(0), a(1));
    // SAFETY: the whole warp runs this, with fragments in the documented layout
    unsafe {
        [
            mma_m16n8k16_f32_f16(mma_m16n8k16_f32_f16([0.0; 4], a0, bf[0][0]), a1, bf[1][0]),
            mma_m16n8k16_f32_f16(mma_m16n8k16_f32_f16([0.0; 4], a0, bf[0][1]), a1, bf[1][1]),
        ]
    }
}

/// Statistics pass, as [`crate::stem_conv1::stem1_stats`]: grid `(48 bands x 3 strips, 17 slices, B)`,
/// block 128; block x `= band * 3 + strip`, `chunk = slice * 144 + x`
#[kernel]
pub fn stem1_stats_384(mut partial: DisjointSlice<[f32; 2]>, mut shift_out: DisjointSlice<f32>, x: &[u16], w: &[u16]) {
    static mut HALO_S: SharedArray<u16, HALO> = SharedArray::UNINIT;
    static mut RED_S: SharedArray<[f32; 2], { 4 * K }> = SharedArray::UNINIT;
    static mut SHIFT_S: SharedArray<f32, K> = SharedArray::UNINIT;
    let (bx, z, b, tid) = (blockIdx_x(), blockIdx_y(), blockIdx_z(), threadIdx_x());
    let (band, strip) = (bx / STRIPS, bx % STRIPS);
    let chunk = z * SLICE_BLOCKS + bx;
    // block-uniform, so no block leaves before a barrier, shuffle or mma that others reach
    if (blockDim_x() != BLOCK)
        | (bx >= SLICE_BLOCKS)
        | (z >= Z)
        | (w.len() < K * TAPS)
        | (((b + 1) * Z * H * W) as usize > x.len())
        | (((b + 1) * STEM1_CHUNKS_384) as usize * K > partial.len())
        | (((b + 1) as usize) * K > shift_out.len())
    {
        return;
    }
    let halo = addr_of_mut!(HALO_S) as *mut u16;
    let red = addr_of_mut!(RED_S) as *mut [f32; 2];
    let ssm = addr_of_mut!(SHIFT_S) as *mut f32;
    // SAFETY: the check above
    unsafe { load_halo(halo, x, b, z, band * BAND, strip * SW, tid) };
    if (tid as usize) < K {
        // the conv's FP16 output at voxel (0, 0, 0): the 8 taps inside the tile, in (dz, dy, dx) order
        let k = tid as usize;
        let mut acc = 0f32;
        #[unroll]
        for t in 0..8usize {
            let (dz, dy, dx) = (1 + t / 4, 1 + (t / 2) % 2, 1 + t % 2);
            let o = (((b * Z + dz as u32 - 1) * H + dy as u32 - 1) * W + dx as u32 - 1) as usize;
            // SAFETY: the check above: `x` holds sample `b` and `w` all weights
            acc += unsafe { widen(*x.get_unchecked(o)) * widen(*w.get_unchecked(k * TAPS + (dz * 3 + dy) * 3 + dx)) };
        }
        // SAFETY: k < K
        unsafe { *ssm.add(k) = r16(acc) };
    }
    let (wid, lane) = (tid / 32, tid % 32);
    let (g, th) = (lane / 4, (lane % 4) as usize);
    // SAFETY: the check above
    let bf = unsafe { b_frags(w, lane) };
    let tp = taps(lane);
    sync_threads();
    // SAFETY: written before the barrier; every index < K
    let s = unsafe { [*ssm.add(2 * th), *ssm.add(2 * th + 1), *ssm.add(8 + 2 * th), *ssm.add(9 + 2 * th)] };

    let mut sum = [0f32; 4];
    let mut sq = [0f32; 4];
    let mut mt = wid;
    while mt < MTILES {
        let base = (mt / (SW / 16)) as usize * HC + (mt % (SW / 16)) as usize * 16;
        // SAFETY: filled above; rows < BAND and columns < SW keep the taps inside the halo
        let c = unsafe { tile(halo, base, g, &tp, &bf) };
        #[unroll]
        for n in 0..2 {
            #[unroll]
            for j in 0..4 {
                let i = 2 * n + (j & 1);
                let d = r16(c[n][j]) - s[i];
                sum[i] += d;
                sq[i] += d * d;
            }
        }
        mt += 4;
    }
    #[unroll]
    for i in 0..4 {
        #[unroll]
        for k in 0..3 {
            let m = 4u32 << k;
            sum[i] += warp::shuffle_xor_f32(sum[i], m);
            sq[i] += warp::shuffle_xor_f32(sq[i], m);
        }
    }
    if g == 0 {
        #[unroll]
        for i in 0..4 {
            let ch = 8 * (i / 2) + 2 * th + (i & 1);
            // SAFETY: wid < 4, ch < K
            unsafe { *red.add(wid as usize * K + ch) = [sum[i], sq[i]] };
        }
    }
    sync_threads();
    if (tid as usize) < K {
        let k = tid as usize;
        let (mut t, mut u) = (0f32, 0f32);
        #[unroll]
        for v in 0..4 {
            // SAFETY: written before the barrier
            let [p, q] = unsafe { *red.add(v * K + k) };
            t += p;
            u += q;
        }
        let o = ((b * STEM1_CHUNKS_384 + chunk) as usize) * K + k;
        // SAFETY: in bounds by the check; only thread k of this block writes chunk `chunk`, channel k
        unsafe { *partial.get_unchecked_mut(o) = [t, u] };
        if chunk == 0 {
            // SAFETY: as above, one writer per (b, k); `ssm` written before the barriers
            unsafe { *shift_out.get_unchecked_mut(b as usize * K + k) = *ssm.add(k) };
        }
    }
}

/// Per (sample, channel) `[mean, invstd]` from [`stem1_stats_384`]'s partial sums and shifts, in
/// float64. One warp per (sample, channel): grid `B * 16 / 4`, block 128
#[kernel]
pub fn stem1_finalize_384(mut stats: DisjointSlice<[f32; 2]>, partial: &[[f32; 2]], shift: &[f32], eps: f32) {
    let (tid, lane) = (threadIdx_x(), threadIdx_x() % 32);
    let i = blockIdx_x() * 4 + tid / 32;
    let (b, k) = (i / K as u32, i % K as u32);
    // warp-uniform: `i` depends only on the warp
    if (blockDim_x() != 128)
        | (i as usize >= stats.len())
        | (i as usize >= shift.len())
        | (((b + 1) * STEM1_CHUNKS_384) as usize * K > partial.len())
    {
        return;
    }
    let (mut s, mut q) = (0f64, 0f64);
    let mut c = lane;
    while c < STEM1_CHUNKS_384 {
        // SAFETY: (b + 1) * STEM1_CHUNKS_384 partials of 16 channels, checked above
        let [p, pq] = unsafe { *partial.get_unchecked(((b * STEM1_CHUNKS_384 + c) * K as u32 + k) as usize) };
        s += p as f64;
        q += pq as f64;
        c += 32;
    }
    let mut m = 16;
    while m > 0 {
        s += warp::shuffle_xor_f64(s, m);
        q += warp::shuffle_xor_f64(q, m);
        m /= 2;
    }
    if lane == 0 {
        // SAFETY: checked above
        let s0 = unsafe { *shift.get_unchecked(i as usize) } as f64;
        let n = (Z * H * W) as f64;
        let mean = s / n;
        let var = q / n - mean * mean;
        let var = if var > 0.0 { var as f32 } else { 0.0 };
        // SAFETY: i < stats.len(); lane 0 of the one warp for (b, k) is its only writer
        unsafe { *stats.get_unchecked_mut(i as usize) = [(s0 + mean) as f32, 1.0 / math::sqrtf(var + eps)] };
    }
}

/// Apply pass, as [`crate::stem_conv1::stem1_apply`]: `out` `[B, 17, 384, 384, 16]` FP16 pairs. Grid and
/// block as [`stem1_stats_384`]
#[kernel]
pub fn stem1_apply_384(mut out: DisjointSlice<u32>, x: &[u16], w: &[u16], stats: &[[f32; 2]], nw: &[f32], nb: &[f32]) {
    static mut HALO_A: SharedArray<u16, HALO> = SharedArray::UNINIT;
    let (bx, z, b, tid) = (blockIdx_x(), blockIdx_y(), blockIdx_z(), threadIdx_x());
    let (band, strip) = (bx / STRIPS, bx % STRIPS);
    if (blockDim_x() != BLOCK)
        | (bx >= SLICE_BLOCKS)
        | (z >= Z)
        | (w.len() < K * TAPS)
        | (nw.len() < K)
        | (nb.len() < K)
        | (((b + 1) as usize) * K > stats.len())
        | (((b + 1) * Z * H * W) as usize > x.len())
        | (((b + 1) * Z * H * W) as usize * (K / 2) > out.len())
    {
        return;
    }
    let halo = addr_of_mut!(HALO_A) as *mut u16;
    // SAFETY: the check above
    unsafe { load_halo(halo, x, b, z, band * BAND, strip * SW, tid) };
    let (wid, lane) = (tid / 32, tid % 32);
    let (g, th) = (lane / 4, (lane % 4) as usize);
    // SAFETY: the check above
    let bf = unsafe { b_frags(w, lane) };
    let tp = taps(lane);
    let norm = |ch: usize| {
        // SAFETY: ch < K; `stats`, `nw`, `nb` checked above
        unsafe {
            let [m, i] = *stats.get_unchecked(b as usize * K + ch);
            [m, i, *nw.get_unchecked(ch), *nb.get_unchecked(ch)]
        }
    };
    let ns = [norm(2 * th), norm(2 * th + 1), norm(8 + 2 * th), norm(9 + 2 * th)];
    sync_threads();

    let row0 = (b * Z + z) * H + band * BAND;
    let mut mt = wid;
    while mt < MTILES {
        let (r, col0) = (mt / (SW / 16), (mt % (SW / 16)) * 16);
        let base = r as usize * HC + col0 as usize;
        // SAFETY: filled above; rows < BAND and columns < SW keep the taps inside the halo
        let c = unsafe { tile(halo, base, g, &tp, &bf) };
        let v0 = ((row0 + r) * W + strip * SW + col0 + g) as usize;
        #[unroll]
        for n in 0..2 {
            let y = |j: usize| {
                let p = ns[2 * n + (j & 1)];
                leaky16(affine16(r16(c[n][j]), p[0], p[1], p[2], p[3]))
            };
            // SAFETY: in bounds by the check; each (voxel, pair) has this lane as its only writer
            unsafe {
                *out.get_unchecked_mut(v0 * (K / 2) + 4 * n + th) = pack_f16(y(0), y(1));
                *out.get_unchecked_mut((v0 + 8) * (K / 2) + 4 * n + th) = pack_f16(y(2), y(3));
            }
        }
        mt += 4;
    }
}
