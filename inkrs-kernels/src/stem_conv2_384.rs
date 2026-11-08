//! The stem's second conv3d with its InstanceNorm's statistics in the epilogue, for 17 x 384 x 384 tiles
//! (Reader V3)
//!
//! [`crate::stem_conv2`] at a 384 edge, in its own kernels so the 128-edge PTX stays unchanged. Each
//! block's work is local to its 8 x 32 patch, so only the tile edge changes: 48 x 12 patches per slice,
//! 9792 (slice, patch) partials per sample

use core::ptr::addr_of_mut;

use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::thread::{blockDim_x, blockIdx_x, blockIdx_y, blockIdx_z, sync_threads, threadIdx_x};
use cuda_device::wmma::ldmatrix_x4_shared_u32;
use cuda_device::{DisjointSlice, SharedArray, kernel, warp};

use crate::common::{math, mma_m16n8k16_f32_f16, pack_f16, r16};

/// Depth slices, rows and columns of a tile
const Z: u32 = 17;
const H: u32 = 384;
const W: u32 = 384;
/// Input and output channels
const C: usize = 16;
/// Kernel taps
const TAPS: u32 = 27;
/// A block's patch of each slice, and its halo planes' rows and columns
const BAND: u32 = 8;
const COLS: u32 = 32;
const HR: u32 = BAND + 2;
const HC: u32 = COLS + 2;
/// One halo plane in 16-byte chunks, two per voxel, and in bytes
const PLANE: u32 = HR * HC * 2;
const PLANE_BYTES: u32 = PLANE * 16;
/// The ring of three planes
const RING: usize = (3 * PLANE) as usize;
/// Bytes of one halo row
const ROW_BYTES: u32 = HC * 32;
/// The B fragments: one 16-byte chunk per (tap, lane)
const WFRAG: usize = (TAPS * 32) as usize;
/// Threads per block: four warps of two rows each
pub const STEM2_BLOCK: u32 = 128;
/// Patches per slice
const PATCHES: u32 = (H / BAND) * (W / COLS);
/// Depth slices per block, and blocks along the depth
const ZSPAN: u32 = 9;
pub const STEM2_ZCHUNKS: u32 = Z.div_ceil(ZSPAN);
/// (slice, patch) partials per sample
pub const STEM2_BLOCKS: u32 = Z * PATCHES;
/// Voxels per partial
const PATCH_VOX: u32 = BAND * COLS;

/// 16 bytes, aligned so one is a single vector load or store: eight FP16 channels of a voxel, or one
/// lane's B fragments for one tap
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct Q(pub [u32; 4]);

/// A warp's accumulators, `[tile][ntile][element]`: tile `m` is row `m / 2` of the warp's two, columns
/// `16 (m % 2)..+16` of the patch; element `j` of n-tile `n` is that tile's voxel `g + 8 (j >= 2)`,
/// channel `8 n + 2 th + (j & 1)`
type Acc = [[[f32; 4]; 2]; 4];

/// Loads halo plane `zz` (slice `zz`, rows `y0 - 1..y0 + BAND + 1`, columns `x0 - 1..x0 + COLS + 1`,
/// zero outside the tile) into ring slot `(zz + 1) % 3`, from `x` (`[B, Z, H, W, 16]` FP16 as chunks).
/// A voxel's second chunk comes first on every other group of four columns
///
/// # Safety
/// `ring` points to a block-shared array of `RING` chunks that no thread reads at slot `(zz + 1) % 3`
/// until a barrier; `x` holds sample `b`, checked by the caller; `zz >= -1`, `y0 + BAND <= H`,
/// `x0 + COLS <= W`
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn load_plane(ring: *mut Q, x: &[Q], b: u32, zz: i32, y0: u32, x0: u32, tid: u32) {
    let slot = ((zz + 1) as u32 % 3) * PLANE;
    let mut i = tid;
    while i < PLANE {
        let (half, col, row) = (i % 2, (i / 2) % HC, i / (2 * HC));
        let (yy, xx) = (y0 as i32 + row as i32 - 1, x0 as i32 + col as i32 - 1);
        let inside = (zz >= 0) & (zz < Z as i32) & (yy >= 0) & (yy < H as i32) & (xx >= 0) & (xx < W as i32);
        let v = if inside {
            let o = (((b * Z + zz as u32) * H + yy as u32) * W + xx as u32) * 2 + half;
            // SAFETY: inside the tile of sample `b`, which the caller checked `x` holds
            unsafe { *x.get_unchecked(o as usize) }
        } else {
            Q([0; 4])
        };
        // SAFETY: slot + row * HC * 2 + col * 2 + 1 < RING
        unsafe { *ring.add((slot + (row * HC + col) * 2 + (half ^ ((col >> 2) & 1))) as usize) = v };
        i += STEM2_BLOCK;
    }
}

/// Tap `t` (`(dz * 3 + dy) * 3 + dx`) of the warp's four tiles into `acc`. `bases[dz]` is the shared
/// address of the warp's first row in the plane at depth offset `dz`, `lo[dx]` this lane's `ldmatrix`
/// byte offset in a row at column offset `dx`
///
/// # Safety
/// The ring is loaded and synchronised; `wf` holds `WFRAG` chunks; every lane of the warp calls this
/// together
#[inline(always)]
unsafe fn tap(acc: &mut Acc, bases: &[u32; 3], lo: &[u32; 3], wf: &[Q], lane: u32, t: u32) {
    let (dz, dy, dx) = (t / 9, (t / 3) % 3, t % 3);
    // SAFETY: t * 32 + lane < WFRAG
    let Q(bw) = unsafe { *wf.get_unchecked((t * 32 + lane) as usize) };
    let (b0, b1) = ([bw[0], bw[1]], [bw[2], bw[3]]);
    let row = bases[dz as usize] + lo[dx as usize] + dy * ROW_BYTES;
    // tile m: the warp's row m / 2, columns 16 (m % 2).., 32 bytes a voxel
    // SAFETY: rows < 2 + 2 + 2 of HR and columns < 16 + 16 + 2 of HC keep every address in the plane
    let a = |m: u32| unsafe { ldmatrix_x4_shared_u32(row + (m / 2) * ROW_BYTES + (m % 2) * 16 * 32) };
    let (a0, a1, a2, a3) = (a(0), a(1), a(2), a(3));
    // SAFETY: the whole warp runs this, with fragments in the documented layout
    unsafe {
        acc[0][0] = mma_m16n8k16_f32_f16(acc[0][0], a0, b0);
        acc[0][1] = mma_m16n8k16_f32_f16(acc[0][1], a0, b1);
        acc[1][0] = mma_m16n8k16_f32_f16(acc[1][0], a1, b0);
        acc[1][1] = mma_m16n8k16_f32_f16(acc[1][1], a1, b1);
        acc[2][0] = mma_m16n8k16_f32_f16(acc[2][0], a2, b0);
        acc[2][1] = mma_m16n8k16_f32_f16(acc[2][1], a2, b1);
        acc[3][0] = mma_m16n8k16_f32_f16(acc[3][0], a3, b0);
        acc[3][1] = mma_m16n8k16_f32_f16(acc[3][1], a3, b1);
    }
}

/// The conv at slice `z` for warp `wid`'s four tiles, as [`Acc`], from the ring at shared address `ring`
/// holding planes `z - 1`, `z`, `z + 1`
///
/// `ldmatrix` lane `l` addresses row `l % 8` of 8x8 matrix `l / 8`; matrices 0..4 are (voxels 0..8,
/// channels 0..8), (8..16, 0..8), (0..8, 8..16), (8..16, 8..16), so the four registers are the A
/// fragment `a0..a3` of `mma.m16n8k16`
///
/// # Safety
/// As [`tap`]; `wid < 4`
#[inline(always)]
pub(crate) unsafe fn conv(ring: u32, z: u32, wf: &[Q], wid: u32, lane: u32) -> Acc {
    // plane z - 1 + dz sits in slot (z + dz) % 3
    let base = |dz: u32| ring + ((z + dz) % 3) * PLANE_BYTES + 2 * wid * ROW_BYTES;
    let bases = [base(0), base(1), base(2)];
    let (vr, half) = (lane % 8 + 8 * ((lane / 8) % 2), lane / 16);
    // this lane's voxel `vr` shifted by dx, its channel half swapped as `load_plane` stored it, in bytes
    let lo = |dx: u32| {
        let c = vr + dx;
        (c * 2 + (half ^ ((c >> 2) & 1))) * 16
    };
    let lo = [lo(0), lo(1), lo(2)];
    let mut acc = [[[0f32; 4]; 2]; 4];
    macro_rules! taps {
        ($($t:literal)*) => { $(
            // SAFETY: the caller's
            unsafe { tap(&mut acc, &bases, &lo, wf, lane, $t) };
        )* };
    }
    taps!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26);
    acc
}

/// Writes `acc` FP16-rounded into `out` (`[B, Z, H, W, 16]` FP16 pairs); `v` is the voxel of the warp's
/// first row and column, plus `g`
///
/// # Safety
/// `out` holds every voxel the warp covers; no other lane writes this lane's (voxel, pair)s
#[inline(always)]
pub(crate) unsafe fn store(out: &mut DisjointSlice<u32>, acc: &Acc, v: u32, th: u32) {
    // tiles written out one by one: a loop over them left `acc` in local memory
    macro_rules! tiles {
        ($($m:literal)*) => { $({
            let vm = (v + ($m / 2) * W + ($m % 2) * 16) as usize;
            let mut n = 0;
            while n < 2 {
                let p = 4 * n + th as usize;
                let c = acc[$m][n];
                // voxel g holds elements 0 and 1, voxel g + 8 elements 2 and 3
                // SAFETY: the caller's
                unsafe {
                    *out.get_unchecked_mut(vm * (C / 2) + p) = pack_f16(c[0], c[1]);
                    *out.get_unchecked_mut((vm + 8) * (C / 2) + p) = pack_f16(c[2], c[3]);
                }
                n += 1;
            }
        })* };
    }
    tiles!(0 1 2 3);
}

/// Sums over the lanes with this lane's `th` (its four channels), for each of `s`
#[inline(always)]
fn lanes_sum(s: &mut [f32; 4]) {
    let mut i = 0;
    while i < 4 {
        s[i] += warp::shuffle_xor_f32(s[i], 4);
        s[i] += warp::shuffle_xor_f32(s[i], 8);
        s[i] += warp::shuffle_xor_f32(s[i], 16);
        i += 1;
    }
}

/// The patch partial `[mean, M2]` per channel of the FP16-rounded `acc` into `partial[(b * STEM2_BLOCKS
/// + at) * 16 + k]`: the patch mean, then the sum of squared deviations from it. Three barriers: after
/// the sums, after every thread has read them (so the second pass may overwrite `red`), after the squares
///
/// # Safety
/// `red` is a block-shared array of `4 * C` floats no thread touches outside this call; `partial` holds
/// sample `b`'s partials; every thread of the block calls this together
#[inline(always)]
pub(crate) unsafe fn stats(partial: &mut DisjointSlice<[f32; 2]>, acc: &Acc, red: *mut f32, b: u32, at: u32, tid: u32) {
    let (wid, lane) = ((tid / 32) as usize, tid % 32);
    let (g, th) = (lane / 4, lane % 4);
    // this lane's channels, [2 n + (j & 1)]
    let ch = |i: usize| 8 * (i / 2) + 2 * th as usize + (i & 1);
    // runs `body` for every value, with `i` its channel slot `2 n + (j & 1)` and `v` its FP16 value; the
    // tiles are written out one by one, as a loop over them left `acc` in local memory
    macro_rules! each {
        (|$i:ident, $v:ident| $body:block) => {
            each!(@ $i $v $body 0 1 2 3)
        };
        (@ $i:ident $v:ident $body:block $($m:literal)*) => { $({
            let mut n = 0;
            while n < 2 {
                let mut j = 0;
                while j < 4 {
                    let ($i, $v) = (2 * n + (j & 1), r16(acc[$m][n][j]));
                    $body
                    j += 1;
                }
                n += 1;
            }
        })* };
    }
    // `red` holds one value per (warp, channel); lanes g == 0 write their four channels' warp totals
    let put = |s: &[f32; 4]| {
        if g == 0 {
            let mut i = 0;
            while i < 4 {
                // SAFETY: wid < 4, ch < C
                unsafe { *red.add(wid * C + ch(i)) = s[i] };
                i += 1;
            }
        }
    };
    // the four warps' values for channel k, summed in a fixed order so every thread gets the same total
    // SAFETY: written before the caller's barrier; 3 * C + k < 4 * C
    let total = |k: usize| unsafe { *red.add(k) + *red.add(C + k) + *red.add(2 * C + k) + *red.add(3 * C + k) };

    let mut s = [0f32; 4];
    each!(|i, v| { s[i] += v; });
    lanes_sum(&mut s);
    put(&s);
    sync_threads();
    let n = PATCH_VOX as f32;
    let mean = [total(ch(0)) / n, total(ch(1)) / n, total(ch(2)) / n, total(ch(3)) / n];
    let own = if (tid as usize) < C { total(tid as usize) / n } else { 0.0 };
    sync_threads();
    let mut q = [0f32; 4];
    each!(|i, v| {
        let d = v - mean[i];
        q[i] += d * d;
    });
    lanes_sum(&mut q);
    put(&q);
    sync_threads();
    if (tid as usize) < C {
        let k = tid as usize;
        let o = ((b * STEM2_BLOCKS + at) as usize) * C + k;
        // SAFETY: in bounds by the caller; only thread k of the block for (b, at) writes (b, at, k)
        unsafe { *partial.get_unchecked_mut(o) = [own, total(k)] };
    }
}

/// The conv with its statistics: `out` (`[B, 17, 384, 384, 16]` FP16 pairs, eight per voxel) gets
/// `r16(conv(x, w))` and `partial` (`[B, STEM2_BLOCKS, 16]`, index `slice * 576 + patch`) each slice of
/// each patch's `[mean, M2]` per channel
///
/// `x` is the stem's first activation as 16-byte chunks (two per voxel) and `wf` the B fragments,
/// `[27 taps][32 lanes]` chunks (see `stem2_frags` on the host). Grid `(576 patches, STEM2_ZCHUNKS, B)`,
/// block `STEM2_BLOCK`; patch `p` is rows `8 (p / 12)..+8`, columns `32 (p % 12)..+32`, and depth chunk
/// `c` slices `9 c..min(9 c + 9, 17)`
#[kernel]
pub fn stem2_conv_384(mut out: DisjointSlice<u32>, mut partial: DisjointSlice<[f32; 2]>, x: &[Q], wf: &[Q]) {
    static mut RING_S: SharedArray<Q, RING> = SharedArray::UNINIT;
    static mut RED_S: SharedArray<f32, { 4 * C }> = SharedArray::UNINIT;
    let (p, zc, b, tid) = (blockIdx_x(), blockIdx_y(), blockIdx_z(), threadIdx_x());
    // block-uniform, so no block leaves before a barrier, shuffle, ldmatrix or mma that others reach
    if (blockDim_x() != STEM2_BLOCK)
        | (p >= PATCHES)
        | (zc >= STEM2_ZCHUNKS)
        | (wf.len() < WFRAG)
        | (((b + 1) * Z * H * W) as usize * 2 > x.len())
        | (((b + 1) * Z * H * W) as usize * (C / 2) > out.len())
        | (((b + 1) * STEM2_BLOCKS) as usize * C > partial.len())
    {
        return;
    }
    let ring = addr_of_mut!(RING_S) as *mut Q;
    let red = addr_of_mut!(RED_S) as *mut f32;
    // SAFETY: `ring` is block-shared
    let ring_at = unsafe { cvta_generic_to_shared_u32(ring as *const u8) };
    let (y0, x0) = ((p / (W / COLS)) * BAND, (p % (W / COLS)) * COLS);
    let (wid, lane) = (tid / 32, tid % 32);
    let (z0, z1) = (zc * ZSPAN, if (zc + 1) * ZSPAN < Z { (zc + 1) * ZSPAN } else { Z });
    // SAFETY: the check above; nothing reads the ring yet
    unsafe {
        load_plane(ring, x, b, z0 as i32 - 1, y0, x0, tid);
        load_plane(ring, x, b, z0 as i32, y0, x0, tid);
    }
    let mut z = z0;
    while z < z1 {
        // slot (z + 2) % 3 last held plane z - 2, read by the previous slice's conv, which every thread
        // finished before the barriers in its `stats`
        // SAFETY: the check above
        unsafe { load_plane(ring, x, b, z as i32 + 1, y0, x0, tid) };
        sync_threads();
        // SAFETY: planes z - 1..=z + 1 loaded and synchronised; the check above; wid < 4
        let acc = unsafe { conv(ring_at, z, wf, wid, lane) };
        let v = ((b * Z + z) * H + y0 + 2 * wid) * W + x0 + lane / 4;
        // SAFETY: the check above; each (voxel, pair) of the patch has one lane as its writer
        unsafe { store(&mut out, &acc, v, lane % 4) };
        // SAFETY: the check above; every thread reaches this
        unsafe { stats(&mut partial, &acc, red, b, z * PATCHES + p, tid) };
        z += 1;
    }
}

/// Per (sample, channel) `[mean, invstd]` from [`stem2_conv`]'s partials, merged in float64: the
/// partials have equal counts, so the mean is the mean of their means and `M2 = sum(M2_i) + n
/// sum((mean_i - mean)^2)`. The variance is the population one, as `in_finalize` takes it. One warp per
/// (sample, channel): grid `B * 16 / 4`, block 128
#[kernel]
pub fn stem2_finalize_384(mut stats: DisjointSlice<[f32; 2]>, partial: &[[f32; 2]], eps: f32) {
    let (tid, lane) = (threadIdx_x(), threadIdx_x() % 32);
    let i = blockIdx_x() * 4 + tid / 32;
    let (b, k) = (i / C as u32, i % C as u32);
    // warp-uniform: `i` depends only on the warp
    if (blockDim_x() != 128) | (i as usize >= stats.len()) | (((b + 1) * STEM2_BLOCKS) as usize * C > partial.len()) {
        return;
    }
    // SAFETY: (b + 1) * STEM2_BLOCKS partials of 16 channels are in `partial`, checked above
    let at = |blk: u32| unsafe { *partial.get_unchecked(((b * STEM2_BLOCKS + blk) * C as u32 + k) as usize) };
    let warp_sum = |mut s: f64| {
        let mut m = 16;
        while m > 0 {
            s += warp::shuffle_xor_f64(s, m);
            m /= 2;
        }
        s
    };
    let mut s = 0f64;
    let mut blk = lane;
    while blk < STEM2_BLOCKS {
        s += at(blk)[0] as f64;
        blk += 32;
    }
    let mean = warp_sum(s) / STEM2_BLOCKS as f64;
    let mut m2 = 0f64;
    let mut blk = lane;
    while blk < STEM2_BLOCKS {
        let [m, q] = at(blk);
        let d = m as f64 - mean;
        m2 += q as f64 + PATCH_VOX as f64 * d * d;
        blk += 32;
    }
    let m2 = warp_sum(m2);
    if lane == 0 {
        let var = (m2 / (Z * H * W) as f64) as f32;
        let var = if var > 0.0 { var } else { 0.0 };
        // SAFETY: i < stats.len(); lane 0 of the one warp for (b, k) is its only writer
        unsafe { *stats.get_unchecked_mut(i as usize) = [mean as f32, 1.0 / math::sqrtf(var + eps)] };
    }
}
