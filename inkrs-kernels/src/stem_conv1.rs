//! The stem's first conv3d (1 -> 16 channels, 3x3x3, "same" padding, no bias) with its InstanceNorm
//! fused, replacing cuDNN's conv and the `in_partial` / `in_finalize` / `in_apply` passes over its
//! 8.9 MB-per-tile output
//!
//! InstanceNorm needs each (sample, channel)'s statistics before any element is normalised, so the conv
//! runs twice: [`stem1_stats`] reduces the FP16-rounded outputs to per-block partial sums without writing
//! them, [`stem1_finalize`] turns those into `[mean, invstd]`, and [`stem1_apply`] runs the conv again
//! and writes `leaky16(affine16(r16(conv)))` once. The input is 17 x 128 x 128 FP16 per tile (0.56 MB),
//! so the second conv costs far less than a write and re-read of the output would
//!
//! The conv is an implicit GEMM on the tensor cores (`mma.m16n8k16`, FP16 in, float32 accumulate, as
//! cuDNN's tensor-op algorithms run it): M is the voxels, 16 consecutive columns of one row per tile, N
//! the 16 channels and K the 27 taps, padded to 32 with zero weights and zero inputs. Each block covers one (sample,
//! depth slice, band of 8 rows); its 3 x 10 x 130 input halo (zero padded) sits in shared memory as FP16,
//! and each lane gathers its A fragments from it at offsets fixed by its lane and tap. The products are
//! summed in the tensor core's order, not cuDNN's, so outputs can differ from cuDNN's in the last FP16 bit

use core::ptr::addr_of_mut;

use cuda_device::thread::{blockDim_x, blockIdx_x, blockIdx_y, blockIdx_z, sync_threads, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread, warp};

use crate::common::{leaky16, math, mma_m16n8k16_f32_f16, pack_f16, r16, widen};
use crate::inorm::affine16;

/// Depth slices, rows and columns of a tile
const Z: u32 = 17;
const H: u32 = 128;
const W: u32 = 128;
/// Output channels
const K: usize = 16;
/// Kernel taps
const TAPS: usize = 27;
/// Rows per block, and the halo's rows and columns
const BAND: u32 = 8;
const HR: usize = BAND as usize + 2;
const HC: usize = W as usize + 2;
const HALO: usize = 3 * HR * HC;
/// Threads per block: four warps, one column each while the halo loads
const BLOCK: u32 = W;
/// GEMM tiles per block: 16 columns of one row each
const MTILES: u32 = BAND * W / 16;
/// Blocks per sample: depth slices times row bands
pub const STEM1_CHUNKS: u32 = Z * (H / BAND);

/// Fills the block's input halo, `[3][HR][HC]` FP16 bits with zeros outside the tile, from `x`
/// (`[B, Z, H, W]` FP16 bits): thread `tid` copies column `tid` of each of the 30 halo rows, and threads
/// 0 and 1 zero the two padding columns
///
/// # Safety
/// `halo` points to a block-shared array of `HALO` values; `x` holds sample `b`, `tid < W`, both checked
/// by the caller
#[inline(always)]
unsafe fn load_halo(halo: *mut u16, x: &[u16], b: u32, z: u32, y0: u32, tid: u32) {
    let mut pr = 0;
    while pr < 3 * HR {
        let (plane, r) = (pr / HR, pr % HR);
        let (zz, yy) = (z as i32 + plane as i32 - 1, y0 as i32 + r as i32 - 1);
        let inside = (zz >= 0) & (zz < Z as i32) & (yy >= 0) & (yy < H as i32);
        let v = if inside {
            // SAFETY: inside the tile of sample `b`, which the caller checked `x` holds
            unsafe { *x.get_unchecked((((b * Z + zz as u32) * H + yy as u32) * W + tid) as usize) }
        } else {
            0
        };
        // SAFETY: pr < 3 * HR and 1 + tid < HC
        unsafe { *halo.add(pr * HC + 1 + tid as usize) = v };
        if tid < 2 {
            // SAFETY: column 0 or HC - 1 of row pr
            unsafe { *halo.add(pr * HC + tid as usize * (HC - 1)) = 0 };
        }
        pr += 1;
    }
}

/// Lane `lane`'s B fragments, `[kstep][ntile]`: weights `[16 taps][8 channels]` for k-step `s` (taps
/// `16s..16s + 16`) and n-tile `n` (channels `8n..8n + 8`), from `w` (`[K, TAPS]` FP16 bits, cuDNN's
/// channels-last filter for one input channel), zero for the padding taps 27..31
///
/// # Safety
/// `w` holds `K * TAPS` weights, checked by the caller
#[inline(always)]
unsafe fn b_frags(w: &[u16], lane: u32) -> [[[u32; 2]; 2]; 2] {
    let (g, th) = ((lane / 4) as usize, (lane % 4) as usize);
    // B element (k, n) of lane (g, th): k = 2 th + (j & 1) + 8 (j >= 2), n = g
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

/// Lane `lane`'s A fragment taps `16 s + 2 th + {0, 1, 8, 9}` for k-step `s`
struct Taps {
    /// Halo offsets, `[kstep][q]`; a padding tap (27 and up) points at offset 0, any in-bounds value
    offs: [[u32; 4]; 2],
    /// Per packed A register pair `[kstep][q / 2]`, the half-words that hold real taps. A padding tap's
    /// halo value is masked to zero rather than left to its zero weight, since an infinite input times
    /// that weight would be NaN
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

/// One GEMM tile: the conv at the 16 voxels `base + 0..16` of the halo (a row's 16 consecutive
/// columns), as this lane's two accumulator fragments, `[ntile]`: element `j` is voxel `g + 8 (j >= 2)`,
/// channel `8 ntile + 2 th + (j & 1)`
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

/// Statistics pass: per-block shifted sums of the FP16-rounded conv outputs
///
/// `x` is the tile batch `[B, 17, 128, 128]` and `w` the filter `[16, 27]`, both FP16 bits. Grid
/// `(16 row bands, 17 slices, B)`, block 128. Writes `partial[b, chunk, k] = [sum(y - s), sum((y - s)^2)]`
/// with `y = r16(conv)`, `chunk = slice * 16 + band` and `s = shift[b, k]`, the FP16 output at voxel 0,
/// which block (0, 0, b) also writes to `shift` for [`stem1_finalize`]
#[kernel]
pub fn stem1_stats(mut partial: DisjointSlice<[f32; 2]>, mut shift_out: DisjointSlice<f32>, x: &[u16], w: &[u16]) {
    static mut HALO_S: SharedArray<u16, HALO> = SharedArray::UNINIT;
    static mut RED_S: SharedArray<[f32; 2], { 4 * K }> = SharedArray::UNINIT;
    static mut SHIFT_S: SharedArray<f32, K> = SharedArray::UNINIT;
    let (band, z, b, tid) = (blockIdx_x(), blockIdx_y(), blockIdx_z(), threadIdx_x());
    let chunk = z * (H / BAND) + band;
    // block-uniform, so no block leaves before a barrier, shuffle or mma that others reach
    if (blockDim_x() != BLOCK)
        | (band >= H / BAND)
        | (z >= Z)
        | (w.len() < K * TAPS)
        | (((b + 1) * Z * H * W) as usize > x.len())
        | (((b + 1) * STEM1_CHUNKS) as usize * K > partial.len())
        | (((b + 1) as usize) * K > shift_out.len())
    {
        return;
    }
    let halo = addr_of_mut!(HALO_S) as *mut u16;
    let red = addr_of_mut!(RED_S) as *mut [f32; 2];
    let ssm = addr_of_mut!(SHIFT_S) as *mut f32;
    // SAFETY: the check above
    unsafe { load_halo(halo, x, b, z, band * BAND, tid) };
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
    // this lane's channels: 2 th + {0, 1} and 8 + 2 th + {0, 1}, as [ntile][j & 1]
    // SAFETY: written before the barrier; every index < K
    let s = unsafe { [*ssm.add(2 * th), *ssm.add(2 * th + 1), *ssm.add(8 + 2 * th), *ssm.add(9 + 2 * th)] };

    let mut sum = [0f32; 4];
    let mut sq = [0f32; 4];
    let mut mt = wid;
    while mt < MTILES {
        let base = (mt / (W / 16)) as usize * HC + (mt % (W / 16)) as usize * 16;
        // SAFETY: filled above; rows < BAND and columns < W keep the taps inside the halo
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
    // lanes with the same th hold the same channels: sum over g, then the 4 warps through shared memory
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
        let o = ((b * STEM1_CHUNKS + chunk) as usize) * K + k;
        // SAFETY: in bounds by the check; only thread k of this block writes chunk `chunk`, channel k
        unsafe { *partial.get_unchecked_mut(o) = [t, u] };
        if chunk == 0 {
            // SAFETY: as above, one writer per (b, k); `ssm` written before the barriers
            unsafe { *shift_out.get_unchecked_mut(b as usize * K + k) = *ssm.add(k) };
        }
    }
}

/// Per (sample, channel) `[mean, invstd]` from [`stem1_stats`]'s partial sums and shifts, in float64,
/// as `in_finalize` computes them for a written activation. One thread per (sample, channel)
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn stem1_finalize(mut stats: DisjointSlice<[f32; 2]>, partial: &[[f32; 2]], shift: &[f32], eps: f32) {
    let t = thread::index_1d_u32(lc);
    let i = t.get();
    let (b, k) = (i / K as u32, i % K as u32);
    let Some(mut out) = stats.element_thread32(t) else { return };
    let s0 = shift[i as usize] as f64;
    let (mut s, mut q) = (0f64, 0f64);
    let mut c = 0;
    while c < STEM1_CHUNKS {
        let [p, pq] = partial[((b * STEM1_CHUNKS + c) * K as u32 + k) as usize];
        s += p as f64;
        q += pq as f64;
        c += 1;
    }
    let n = (Z * H * W) as f64;
    let m = s / n;
    let var = q / n - m * m;
    let var = if var > 0.0 { var as f32 } else { 0.0 };
    out.write([(s0 + m) as f32, 1.0 / math::sqrtf(var + eps)]);
}

/// Apply pass: the conv again, `leaky16(affine16(r16(conv)))` with [`stem1_finalize`]'s `stats` and the
/// norm's affine `nw`, `nb` (float32), written once as `out` `[B, 17, 128, 128, 16]` FP16 pairs (eight per
/// voxel). Grid and block as [`stem1_stats`]
#[kernel]
pub fn stem1_apply(mut out: DisjointSlice<u32>, x: &[u16], w: &[u16], stats: &[[f32; 2]], nw: &[f32], nb: &[f32]) {
    static mut HALO_A: SharedArray<u16, HALO> = SharedArray::UNINIT;
    let (band, z, b, tid) = (blockIdx_x(), blockIdx_y(), blockIdx_z(), threadIdx_x());
    if (blockDim_x() != BLOCK)
        | (band >= H / BAND)
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
    unsafe { load_halo(halo, x, b, z, band * BAND, tid) };
    let (wid, lane) = (tid / 32, tid % 32);
    let (g, th) = (lane / 4, (lane % 4) as usize);
    // SAFETY: the check above
    let bf = unsafe { b_frags(w, lane) };
    let tp = taps(lane);
    // this lane's channels' norm, [ntile][j & 1]: mean, invstd, weight, bias
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
        let (r, col0) = (mt / (W / 16), (mt % (W / 16)) * 16);
        let base = r as usize * HC + col0 as usize;
        // SAFETY: filled above; rows < BAND and columns < W keep the taps inside the halo
        let c = unsafe { tile(halo, base, g, &tp, &bf) };
        let v0 = ((row0 + r) * W + col0 + g) as usize;
        #[unroll]
        for n in 0..2 {
            let y = |j: usize| {
                let p = ns[2 * n + (j & 1)];
                leaky16(affine16(r16(c[n][j]), p[0], p[1], p[2], p[3]))
            };
            // voxel g holds elements 0 and 1, voxel g + 8 elements 2 and 3; pair 4 n + th of each
            // SAFETY: in bounds by the check; each (voxel, pair) has this lane as its only writer
            unsafe {
                *out.get_unchecked_mut(v0 * (K / 2) + 4 * n + th) = pack_f16(y(0), y(1));
                *out.get_unchecked_mut((v0 + 8) * (K / 2) + 4 * n + th) = pack_f16(y(2), y(3));
            }
        }
        mt += 4;
    }
}
