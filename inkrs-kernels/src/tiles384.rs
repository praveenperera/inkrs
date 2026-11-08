//! Tile gather and villa's robust normalisation for 17 x 384 x 384 tiles (Reader V3)
//!
//! The same steps as [`crate::tiles`] at a 384 edge, in their own kernels so the 128-edge PTX stays
//! unchanged. `tile_stats` takes the voxel count as a parameter and serves both edges. The only
//! structural change is numpy's std for MAD-fallback tiles, which splits its pairwise-sum tree over many
//! blocks per tile ([`tile_sum_384`], [`tile_sq_384`], [`tile_std_finish_384`]), see [`NODE`]

use core::ptr::addr_of_mut;

use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32, DeviceAtomicU32};
use cuda_device::thread::{blockDim_x, blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_contract, thread, warp};

use crate::common::{add_rn_f32, math, mul_rn_f32, pack_f16};
use crate::tiles::{DEPTH, DESC};

/// Tile edge in pixels
pub const TILE: u32 = 384;
/// Pixels per depth slice
const PLANE: u32 = TILE * TILE;
/// Voxels per tile
pub const TILE_VOXELS: u32 = DEPTH * PLANE;

/// [`crate::tiles::tile_hist`] at a 384 edge: grid `(DEPTH, tiles)`, 256 threads
#[kernel]
pub fn tile_hist_384(hist: &[DeviceAtomicU32], vol: &[u8], desc: &[i32], height: u32, width: u32) {
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
    while i < PLANE {
        let (y, x) = (i / TILE, i % TILE);
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

/// Elements per node of numpy's pairwise-sum tree on one tile. At `n = 17 * 384 * 384` every halving
/// lands on a multiple of 8 down to 2048 nodes of 1224. Below that numpy splits at a multiple of 8,
/// so a node is 608 + 616, then 304 + 304 and 304 + 312, 152 + 152 and 152 + 160, and the leaves
/// are 152 = 72 + 80 and 160 = 80 + 80. A leaf sums 8 interleaved chains `start + j + 8i` in order of
/// `i`, then pairs them `((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))`.
///
/// So the whole tile sum is a perfect binary tree over [`CHAINS`] chains of 9 or 10 elements (8 per
/// leaf, 16 leaves per node, 2048 nodes), and any aligned power-of-two run of chains is one of its
/// subtrees. That is what lets the sum split over [`STD_BLOCKS`] blocks per tile and over warp
/// shuffles without moving a single addition (the host's `pairwise_tree_384` and
/// `pairwise_chains_384` tests check both views against `pairwise_f32`)
const NODE: u32 = 1224;
const NODES: u32 = TILE_VOXELS / NODE;
/// Chains per tile: 8 per leaf, 16 leaves per node
const CHAINS: u32 = NODES * 128;
/// 256-chain subtrees per block: each thread sums this many chains, 256 apart
const ROUNDS: u32 = 4;
/// Blocks per tile in each pass of the std, and so partial sums per tile per pass. The host's
/// `STD_BLOCKS_384` must match
pub const STD_BLOCKS: u32 = CHAINS / (256 * ROUNDS);

/// Voxel `e` of a tile in villa's ZYX order (`e = z * 384 * 384 + y * 384 + x`) as float
#[inline(always)]
fn voxel(vol: &[u8], plane0: i32, step_plane: i32, row0: u32, width: u32, e: u32) -> f32 {
    let (z, yx) = (e / PLANE, e % PLANE);
    let plane = (plane0 + step_plane * z as i32) as u32;
    vol[(plane + row0 + (yx / TILE) * width + (yx % TILE)) as usize] as f32
}

/// The voxel source of one tile sum: the tile's first plane offset and plane step, its first row
/// offset, the volume width, and the mean to subtract when summing squared deviations
#[derive(Clone, Copy)]
struct Src<'a> {
    vol: &'a [u8],
    p0: i32,
    sp: i32,
    row0: u32,
    width: u32,
    mean: f32,
    square: bool,
}

impl<'a> Src<'a> {
    /// Tile `b`'s source, as `desc` places it in the `height x width` volume
    #[inline(always)]
    fn new(vol: &'a [u8], desc: &[i32], b: u32, height: u32, width: u32) -> Self {
        let d = b as usize * DESC;
        let (y0, x0, l0, step) = (desc[d] as u32, desc[d + 1] as u32, desc[d + 2], desc[d + 3]);
        let hw = (height * width) as i32;
        Self { vol, p0: l0 * hw, sp: step * hw, row0: y0 * width + x0, width, mean: 0.0, square: false }
    }

    /// Element `e` as numpy sees it: the voxel, or its squared deviation from the mean
    #[inline(always)]
    fn at(self, e: u32) -> f32 {
        let v = voxel(self.vol, self.p0, self.sp, self.row0, self.width, e);
        if self.square {
            let d = v - self.mean;
            mul_rn_f32(d, d)
        } else {
            v
        }
    }

    /// Chain `c` of the tile: node `c / 128`, leaf `l = c / 8 % 16`, lane `j = c % 8`, summed in order.
    /// Leaves alternate 72 and 80 elements, except that leaf 14 is 80 (the node's last 160)
    #[inline(always)]
    fn chain(self, c: u32) -> f32 {
        let (k, l, j) = (c >> 7, (c >> 3) & 15, c & 7);
        let odd = l & 1;
        let start = k * NODE + (l >> 1) * 152 + odd * 72 + (l == 15) as u32 * 8 + j;
        let mut r = self.at(start);
        let mut i = 1;
        while i < 9 {
            r = add_rn_f32(r, self.at(start + 8 * i));
            i += 1;
        }
        if odd != 0 || l == 14 {
            r = add_rn_f32(r, self.at(start + 72));
        }
        r
    }
}

/// The pairwise sum of the warp's 32 values, lane `i` holding element `i`: a butterfly, so every lane
/// ends with the sum, and since float addition commutes each level is the tree's `left + right`
#[inline(always)]
fn warp_tree(v: f32) -> f32 {
    let mut v = v;
    let mut m = 1;
    while m < 32 {
        v = add_rn_f32(v, warp::shuffle_xor_f32(v, m));
        m *= 2;
    }
    v
}

/// The pairwise sum of 8 values at `p`
#[inline(always)]
fn tree8(p: *const f32) -> f32 {
    // SAFETY: the callers pass shared arrays of at least 8, read after the barrier that wrote them
    let v = |i: usize| unsafe { *p.add(i) };
    add_rn_f32(
        add_rn_f32(add_rn_f32(v(0), v(1)), add_rn_f32(v(2), v(3))),
        add_rn_f32(add_rn_f32(v(4), v(5)), add_rn_f32(v(6), v(7))),
    )
}

/// The pairwise sum of one 256-value row `p[0..256]` by a 256-thread block; every thread gets it.
/// `warps` is an 8-slot shared array no other code writes while this runs
#[inline(always)]
fn block_tree(p: &[f32], warps: *mut f32) -> f32 {
    let tid = threadIdx_x();
    let v = warp_tree(p[tid as usize]);
    if tid & 31 == 0 {
        // SAFETY: lane 0 of warp w is slot w's only writer
        unsafe { *warps.add((tid >> 5) as usize) = v };
    }
    sync_threads();
    tree8(warps)
}

/// One block's share of a tile sum: the sum of chains `blk * 1024 ..` `(blk + 1) * 1024`, an 8-node
/// subtree, written to `part[slot]` by thread 0. `sums` is a 32-slot shared array
#[inline(always)]
fn block_sum(part: &mut DisjointSlice<f32>, slot: u32, s: Src, blk: u32, sums: *mut f32) {
    let tid = threadIdx_x();
    let c0 = blk * 256 * ROUNDS + tid;
    let mut q = 0;
    while q < ROUNDS {
        let v = warp_tree(s.chain(c0 + 256 * q));
        if tid & 31 == 0 {
            // SAFETY: lane 0 of warp w is the only writer of slot 8q + w
            unsafe { *sums.add((8 * q + (tid >> 5)) as usize) = v };
        }
        q += 1;
    }
    sync_threads();
    if tid == 0 {
        // the four 256-chain subtrees, each a tree over its 8 warps
        let (a, b) = (
            add_rn_f32(tree8(sums), tree8(sums.wrapping_add(8))),
            add_rn_f32(tree8(sums.wrapping_add(16)), tree8(sums.wrapping_add(24))),
        );
        // SAFETY: the caller checked `slot < part.len()`; one block per slot, thread 0 its only writer
        unsafe { *part.get_unchecked_mut(slot as usize) = add_rn_f32(a, b) };
    }
}

/// The checks every std kernel makes for tile `b`, all block-uniform so a block that fails them never
/// reaches a barrier: its descriptor and stats exist, `part` holds its [`STD_BLOCKS`] partials, and
/// the block is 256 threads
#[inline(always)]
fn std_shape_ok(b: u32, blk: u32, desc: &[i32], stats_len: usize, part_len: usize) -> bool {
    (b as usize + 1) * DESC <= desc.len()
        && (b as usize) < stats_len
        && ((b + 1) * STD_BLOCKS) as usize <= part_len
        && blk < STD_BLOCKS
        && blockDim_x() == 256
        && NODES == 2048
}

/// [`crate::tiles::tile_std`]'s first pass at a 384 edge, split over [`STD_BLOCKS`] blocks per tile:
/// each marked tile's partial voxel sums, `part[tile * STD_BLOCKS + block]`. Grid `(STD_BLOCKS,
/// tiles)`, 256 threads; tiles whose MAD did not fall back (`stats[3] >= 0`) return at once
#[kernel]
pub fn tile_sum_384(
    mut part: DisjointSlice<f32>,
    stats: &[[f32; 4]],
    vol: &[u8],
    desc: &[i32],
    height: u32,
    width: u32,
) {
    static mut SUMS: SharedArray<f32, 32> = SharedArray::UNINIT;
    let (blk, b) = (blockIdx_x(), blockIdx_y());
    if !std_shape_ok(b, blk, desc, stats.len(), part.len()) || stats[b as usize][3] >= 0.0 {
        return;
    }
    let src = Src::new(vol, desc, b, height, width);
    block_sum(&mut part, b * STD_BLOCKS + blk, src, blk, addr_of_mut!(SUMS) as *mut f32);
}

/// The second pass: each block first sums the tile's [`STD_BLOCKS`] partials from [`tile_sum_384`]
/// into the mean, then its share of the squared deviations, `sq[tile * STD_BLOCKS + block]`. Grid and
/// block as [`tile_sum_384`]
#[kernel]
pub fn tile_sq_384(
    mut sq: DisjointSlice<f32>,
    part: &[f32],
    stats: &[[f32; 4]],
    vol: &[u8],
    desc: &[i32],
    height: u32,
    width: u32,
) {
    static mut WARPS: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut SUMS: SharedArray<f32, 32> = SharedArray::UNINIT;
    let (blk, b) = (blockIdx_x(), blockIdx_y());
    let ok = std_shape_ok(b, blk, desc, stats.len(), sq.len()) && ((b + 1) * STD_BLOCKS) as usize <= part.len();
    if !ok || stats[b as usize][3] >= 0.0 {
        return;
    }
    let row = (b * STD_BLOCKS) as usize;
    let mean = block_tree(&part[row..row + STD_BLOCKS as usize], addr_of_mut!(WARPS) as *mut f32) / TILE_VOXELS as f32;
    let mut src = Src::new(vol, desc, b, height, width);
    src.mean = mean;
    src.square = true;
    // `block_sum` writes SUMS, not WARPS, so it needs no barrier after the mean's reads
    block_sum(&mut sq, b * STD_BLOCKS + blk, src, blk, addr_of_mut!(SUMS) as *mut f32);
}

/// [`crate::tiles::tile_std`]'s result from [`tile_sq_384`]'s partials: numpy's exact float32 `np.std`
/// into `stats[3]` of each marked tile. Grid `(tiles)`, 256 threads
#[kernel]
pub fn tile_std_finish_384(mut stats: DisjointSlice<[f32; 4]>, sq: &[f32], desc: &[i32]) {
    static mut WARPS: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut MARK: SharedArray<f32, 1> = SharedArray::UNINIT;
    let (b, tid) = (blockIdx_x(), threadIdx_x());
    if !std_shape_ok(b, 0, desc, stats.len(), sq.len()) {
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
    let row = (b * STD_BLOCKS) as usize;
    let var = block_tree(&sq[row..row + STD_BLOCKS as usize], addr_of_mut!(WARPS) as *mut f32) / TILE_VOXELS as f32;
    let std = math::sqrtf(var);
    if tid == 0 {
        // SAFETY: as above
        unsafe { (*stats.get_unchecked_mut(b as usize))[3] = if std as f64 >= 1e-6 { std } else { -marker } };
    }
}

/// [`crate::tiles::tile_norm`] at a 384 edge: the normalised FP16 input `[tiles, DEPTH, 384, 384]`,
/// two voxels per `u32`
#[kernel(launch_context = lc)]
#[launch_contract(domain = 1, coordinates = u32)]
pub fn tile_norm_384(
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
    let (z, yx) = (r / (PLANE / 2), r % (PLANE / 2));
    let (y, x) = (yx / (TILE / 2), (yx % (TILE / 2)) * 2);
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
