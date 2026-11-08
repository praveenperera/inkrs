//! villa's tile schedule and blend window, computed on the host once per sweep
//!
//! Positions, skips and accumulation order follow villa `infer.py` (0fb38c45):
//! `_sliding_positions_1d` steps by the stride and appends the edge-aligned last position;
//! `iter_blocks` visits rows then columns and drops a tile whose footprint is empty in villa's
//! low-resolution occupancy mask (pyramid level 3 of a group, or the nearest level; none for a
//! bare array); a remaining tile runs only when its 17 selected layers hold a nonzero voxel
//! (the raw-window `.any()` skip); each pixel then sums its covering tiles in that order

use color_eyre::eyre::{Result, ensure};
use rayon::prelude::*;

use crate::zarr::{LowRes, VolumeRef};

/// Tile edge of the 128-pixel readers (`hybrid_3d2d-seed42`, reader-v2), the default
pub const PATCH: usize = 128;
/// Depth slices per tile
pub const DEPTH: usize = 17;
/// `i32`s per tile descriptor, as the kernels read them
pub const DESC: usize = 8;

/// `torch.hann_window(128, periodic=False, dtype=torch.float32)` as float32 bits, from torch
/// 2.13.0 on the CPU; recomputing it in Rust differs from torch's vectorised cosine in the last
/// bit of 6 entries, so the window is a constant
const HANN_BITS: [u32; PATCH] = [
    0x00000000, 0x3a206000, 0x3b204780, 0x3bb42180, 0x3c1fe320, 0x3c795da0, 0x3cb323f0, 0x3cf32ee0, 0x3d1e53c0,
    0x3d47afe8, 0x3d7591f8, 0x3d93ee9c, 0x3daf3914, 0x3dcc9744, 0x3debf6d0, 0x3e06a204, 0x3e18350a, 0x3e2aa974,
    0x3e3df3ba, 0x3e5207be, 0x3e66d8ee, 0x3e7c5a42, 0x3e893f20, 0x3e949b86, 0x3ea03b31, 0x3eac16dc, 0x3eb82718,
    0x3ec46456, 0x3ed0c6ec, 0x3edd471a, 0x3ee9dd07, 0x3ef680d2, 0x3f019548, 0x3f07e929, 0x3f0e3815, 0x3f147e19,
    0x3f1ab747, 0x3f20dfb8, 0x3f26f390, 0x3f2cef02, 0x3f32ce4d, 0x3f388dc5, 0x3f3e29ce, 0x3f439ee6, 0x3f48e9a0,
    0x3f4e06ae, 0x3f52f2da, 0x3f57ab0e, 0x3f5c2c57, 0x3f6073e0, 0x3f647efe, 0x3f684b25, 0x3f6bd5f7, 0x3f6f1d3a,
    0x3f721ee1, 0x3f74d90a, 0x3f7749fe, 0x3f797038, 0x3f7b4a5e, 0x3f7cd746, 0x3f7e15f8, 0x3f7f05ae, 0x3f7fa5d0,
    0x3f7ff5fa, 0x3f7ff5fa, 0x3f7fa5d0, 0x3f7f05ae, 0x3f7e15f8, 0x3f7cd746, 0x3f7b4a5d, 0x3f797038, 0x3f7749fe,
    0x3f74d909, 0x3f721ee1, 0x3f6f1d3a, 0x3f6bd5f8, 0x3f684b26, 0x3f647efd, 0x3f6073e0, 0x3f5c2c56, 0x3f57ab0e,
    0x3f52f2da, 0x3f4e06ac, 0x3f48e9a0, 0x3f439ee6, 0x3f3e29cc, 0x3f388dc4, 0x3f32ce4e, 0x3f2cef04, 0x3f26f38f,
    0x3f20dfb8, 0x3f1ab748, 0x3f147e18, 0x3f0e3815, 0x3f07e929, 0x3f019549, 0x3ef680cf, 0x3ee9dd06, 0x3edd471b,
    0x3ed0c6ea, 0x3ec46455, 0x3eb82718, 0x3eac16de, 0x3ea03b2e, 0x3e949b84, 0x3e893f22, 0x3e7c5a3c, 0x3e66d8ec,
    0x3e5207be, 0x3e3df3b2, 0x3e2aa972, 0x3e18350a, 0x3e06a208, 0x3debf6c8, 0x3dcc9744, 0x3daf3918, 0x3d93ee94,
    0x3d7591f0, 0x3d47afe8, 0x3d1e53c8, 0x3cf32ed0, 0x3cb323e0, 0x3c795da0, 0x3c1fe320, 0x3bb42140, 0x3b204780,
    0x3a206200, 0x00000000,
];

/// `torch.hann_window(384, periodic=False, dtype=torch.float32)` as float32 bits, from torch
/// 2.13.0 on the CPU, for Reader V3's 384-pixel tiles; a constant for the same reason as
/// [`HANN_BITS`]
const HANN384_BITS: [u32; 384] = [
    0x00000000, 0x388d2000, 0x398d1800, 0x3a1eb600, 0x3a8d0d00, 0x3adc5900, 0x3b1e9c80, 0x3b57d300, 0x3b8ce640,
    0x3bb24200, 0x3bdbfa40, 0x3c050600, 0x3c1e3a40, 0x3c3997e0, 0x3c571d20, 0x3c76c800, 0x3c8c4b20, 0x3c9e42c0,
    0x3cb149b0, 0x3cc55ea0, 0x3cda8020, 0x3cf0acd0, 0x3d03f190, 0x3d1010b0, 0x3d1cb308, 0x3d29d7a8, 0x3d377da8,
    0x3d45a428, 0x3d544a20, 0x3d636e88, 0x3d731068, 0x3d819750, 0x3d89e408, 0x3d926dd0, 0x3d9b3410, 0x3da43628,
    0x3dad737c, 0x3db6eb6c, 0x3dc09d4c, 0x3dca8870, 0x3dd4ac30, 0x3ddf07d8, 0x3de99aac, 0x3df463f4, 0x3dff62f8,
    0x3e054b74, 0x3e0aff86, 0x3e10cd4a, 0x3e16b45a, 0x3e1cb44e, 0x3e22ccc0, 0x3e28fd3c, 0x3e2f455c, 0x3e35a4b0,
    0x3e3c1ac6, 0x3e42a72a, 0x3e494970, 0x3e50011a, 0x3e56cdb6, 0x3e5daecc, 0x3e64a3e6, 0x3e6bac82, 0x3e72c82a,
    0x3e79f65a, 0x3e809b4e, 0x3e844436, 0x3e87f5a2, 0x3e8baf56, 0x3e8f710c, 0x3e933a82, 0x3e970b79, 0x3e9ae3aa,
    0x3e9ec2d2, 0x3ea2a8ae, 0x3ea694f8, 0x3eaa8769, 0x3eae7fc0, 0x3eb27db4, 0x3eb680fe, 0x3eba8959, 0x3ebe967c,
    0x3ec2a822, 0x3ec6be03, 0x3ecad7d3, 0x3ecef54f, 0x3ed3162c, 0x3ed73a1f, 0x3edb60e3, 0x3edf8a2d, 0x3ee3b5b1,
    0x3ee7e32b, 0x3eec124d, 0x3ef042ce, 0x3ef47465, 0x3ef8a6c9, 0x3efcd9ac, 0x3f008664, 0x3f029fe9, 0x3f04b93f,
    0x3f06d242, 0x3f08eace, 0x3f0b02ba, 0x3f0d19e6, 0x3f0f302b, 0x3f114562, 0x3f13596a, 0x3f156c1d, 0x3f177d55,
    0x3f198cf0, 0x3f1b9ac8, 0x3f1da6b8, 0x3f1fb09e, 0x3f21b856, 0x3f23bdb9, 0x3f25c0a8, 0x3f27c0fb, 0x3f29be92,
    0x3f2bb949, 0x3f2db0fc, 0x3f2fa58a, 0x3f3196cf, 0x3f3384a9, 0x3f356ef7, 0x3f375596, 0x3f393867, 0x3f3b1744,
    0x3f3cf210, 0x3f3ec8aa, 0x3f409af0, 0x3f4268c2, 0x3f443202, 0x3f45f68e, 0x3f47b648, 0x3f497112, 0x3f4b26cc,
    0x3f4cd758, 0x3f4e829b, 0x3f502873, 0x3f51c8c6, 0x3f536378, 0x3f54f868, 0x3f568780, 0x3f5810a2, 0x3f5993b0,
    0x3f5b1094, 0x3f5c8733, 0x3f5df770, 0x3f5f6134, 0x3f60c468, 0x3f6220ee, 0x3f6376b3, 0x3f64c59e, 0x3f660d97,
    0x3f674e88, 0x3f68885c, 0x3f69bafb, 0x3f6ae652, 0x3f6c0a4c, 0x3f6d26d3, 0x3f6e3bd6, 0x3f6f4940, 0x3f704f00,
    0x3f714d02, 0x3f724336, 0x3f73318c, 0x3f7417f2, 0x3f74f657, 0x3f75ccae, 0x3f769ae8, 0x3f7760f5, 0x3f781eca,
    0x3f78d456, 0x3f798190, 0x3f7a266c, 0x3f7ac2dd, 0x3f7b56d8, 0x3f7be255, 0x3f7c6548, 0x3f7cdfa9, 0x3f7d5170,
    0x3f7dba95, 0x3f7e1b10, 0x3f7e72db, 0x3f7ec1f0, 0x3f7f0848, 0x3f7f45e1, 0x3f7f7ab4, 0x3f7fa6c0, 0x3f7fca00,
    0x3f7fe472, 0x3f7ff614, 0x3f7ffee6, 0x3f7ffee6, 0x3f7ff614, 0x3f7fe472, 0x3f7fca00, 0x3f7fa6c0, 0x3f7f7ab4,
    0x3f7f45e0, 0x3f7f0848, 0x3f7ec1ef, 0x3f7e72da, 0x3f7e1b10, 0x3f7dba94, 0x3f7d5170, 0x3f7cdfaa, 0x3f7c6548,
    0x3f7be254, 0x3f7b56d8, 0x3f7ac2dc, 0x3f7a266c, 0x3f798191, 0x3f78d456, 0x3f781ec9, 0x3f7760f6, 0x3f769ae8,
    0x3f75ccae, 0x3f74f658, 0x3f7417f1, 0x3f73318c, 0x3f724336, 0x3f714d02, 0x3f704eff, 0x3f6f4940, 0x3f6e3bd4,
    0x3f6d26d2, 0x3f6c0a4b, 0x3f6ae652, 0x3f69bafa, 0x3f68885b, 0x3f674e88, 0x3f660d96, 0x3f64c59e, 0x3f6376b4,
    0x3f6220ee, 0x3f60c466, 0x3f5f6134, 0x3f5df76f, 0x3f5c8732, 0x3f5b1094, 0x3f5993b0, 0x3f5810a0, 0x3f568780,
    0x3f54f868, 0x3f536376, 0x3f51c8c6, 0x3f502874, 0x3f4e8298, 0x3f4cd757, 0x3f4b26cb, 0x3f497111, 0x3f47b648,
    0x3f45f68e, 0x3f4431ff, 0x3f4268c0, 0x3f409aee, 0x3f3ec8a9, 0x3f3cf211, 0x3f3b1745, 0x3f393867, 0x3f375595,
    0x3f356ef6, 0x3f3384a8, 0x3f3196ce, 0x3f2fa58a, 0x3f2db0fd, 0x3f2bb946, 0x3f29be91, 0x3f27c0fa, 0x3f25c0a7,
    0x3f23bdba, 0x3f21b856, 0x3f1fb09c, 0x3f1da6b7, 0x3f1b9ac7, 0x3f198cf0, 0x3f177d56, 0x3f156c1e, 0x3f135968,
    0x3f114561, 0x3f0f3029, 0x3f0d19e6, 0x3f0b02bb, 0x3f08eace, 0x3f06d240, 0x3f04b93e, 0x3f029fe8, 0x3f008664,
    0x3efcd9ad, 0x3ef8a6ca, 0x3ef47460, 0x3ef042cb, 0x3eec124a, 0x3ee7e32a, 0x3ee3b5b2, 0x3edf8a2e, 0x3edb60de,
    0x3ed73a1c, 0x3ed31629, 0x3ecef54e, 0x3ecad7d4, 0x3ec6be04, 0x3ec2a81e, 0x3ebe9679, 0x3eba8956, 0x3eb680fc,
    0x3eb27db4, 0x3eae7fc1, 0x3eaa8764, 0x3ea694f3, 0x3ea2a8aa, 0x3e9ec2d0, 0x3e9ae3aa, 0x3e970b7a, 0x3e933a7e,
    0x3e8f7108, 0x3e8baf52, 0x3e87f5a2, 0x3e844435, 0x3e809b4f, 0x3e79f660, 0x3e72c822, 0x3e6bac7c, 0x3e64a3e2,
    0x3e5daece, 0x3e56cdb8, 0x3e50011e, 0x3e494968, 0x3e42a726, 0x3e3c1ac2, 0x3e35a4ae, 0x3e2f455e, 0x3e28fd40,
    0x3e22ccb8, 0x3e1cb44a, 0x3e16b458, 0x3e10cd4a, 0x3e0aff88, 0x3e054b78, 0x3dff62e8, 0x3df463ec, 0x3de99aa8,
    0x3ddf07d8, 0x3dd4ac34, 0x3dca8878, 0x3dc09d40, 0x3db6eb60, 0x3dad7378, 0x3da43624, 0x3d9b3410, 0x3d926dd4,
    0x3d89e400, 0x3d819748, 0x3d731060, 0x3d636e88, 0x3d544a20, 0x3d45a428, 0x3d377d98, 0x3d29d798, 0x3d1cb300,
    0x3d1010b0, 0x3d03f190, 0x3cf0acd0, 0x3cda8010, 0x3cc55e90, 0x3cb149a0, 0x3c9e42c0, 0x3c8c4b20, 0x3c76c800,
    0x3c571d00, 0x3c3997c0, 0x3c1e3a20, 0x3c050600, 0x3bdbfa40, 0x3bb24200, 0x3b8ce600, 0x3b57d300, 0x3b1e9c80,
    0x3adc5900, 0x3a8d0d00, 0x3a1eb600, 0x398d1800, 0x388d1000, 0x00000000,
];

/// A tile edge the engine implements, fixed by the export's `crop_size`: 128 for
/// `hybrid_3d2d-seed42` and reader-v2, 384 for Reader V3. The architecture is the same; the
/// edge sets the schedule, the blend window and which of the edge-specific kernels run
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Edge {
    #[default]
    P128,
    P384,
}

impl Edge {
    /// The edge of a 17-layer `crop_size`, or None for a crop this engine does not run
    pub fn from_crop(crop: &[usize]) -> Option<Self> {
        match crop {
            [17, 128, 128] => Some(Self::P128),
            [17, 384, 384] => Some(Self::P384),
            _ => None,
        }
    }

    /// The edge in pixels
    pub const fn px(self) -> usize {
        match self {
            Self::P128 => 128,
            Self::P384 => 384,
        }
    }

    /// Pixels per tile slice
    pub const fn pixels(self) -> usize {
        self.px() * self.px()
    }

    /// The PTX entry for a kernel that compiles the edge in (`tile_hist`, `stem1_stats`, ...):
    /// the name itself at 128, its `_384` copy at 384. Kernels sized at run time keep one name
    pub fn kernel(self, name: &'static str) -> &'static str {
        match (self, name) {
            (Self::P128, n) => n,
            (Self::P384, "tile_hist") => "tile_hist_384",
            (Self::P384, "tile_norm") => "tile_norm_384",
            (Self::P384, "stem1_stats") => "stem1_stats_384",
            (Self::P384, "stem1_finalize") => "stem1_finalize_384",
            (Self::P384, "stem1_apply") => "stem1_apply_384",
            (Self::P384, "stem2_conv") => "stem2_conv_384",
            (Self::P384, "stem2_finalize") => "stem2_finalize_384",
            (Self::P384, "head_logits") => "head_logits_384",
            (Self::P384, "blend_finish") => "blend_finish_384",
            (Self::P384, n) => n,
        }
    }
}

/// villa's floored Hann blend window `[edge, edge]`: the outer product of the 1D window,
/// divided by its maximum and clamped below at 0.001, all in float32
pub fn hann_weights(edge: Edge) -> Vec<f32> {
    let p = edge.px();
    let bits: &[u32] = match edge {
        Edge::P128 => &HANN_BITS,
        Edge::P384 => &HANN384_BITS,
    };
    let h: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
    let mut w: Vec<f32> = (0..p * p).map(|i| h[i / p] * h[i % p]).collect();
    let max = w.iter().copied().fold(f32::MIN, f32::max).max(f32::EPSILON);
    w.iter_mut().for_each(|v| *v = (*v / max).max(0.001));
    w
}

/// villa `_sliding_positions_1d` for tiles of `patch` pixels
pub fn positions(length: usize, stride: usize, patch: usize) -> Vec<usize> {
    if length <= patch {
        return vec![0];
    }
    let last = length - patch;
    let mut p: Vec<usize> = (0..=last).step_by(stride.max(1)).collect();
    if *p.last().unwrap() != last {
        p.push(last);
    }
    p
}

/// villa `select_layer_indices` for a 17-layer reader: the first layer and the step
pub fn select_layers(depth: usize, start: Option<i64>, end: Option<i64>, reverse: bool) -> Result<(i32, i32)> {
    let d = depth as i64;
    let mut s = start.unwrap_or(0);
    let mut e = end.unwrap_or(d);
    if s < 0 {
        s += d;
    }
    if e < 0 {
        e += d;
    }
    s = s.max(0);
    e = e.min(d);
    if e <= s {
        (s, e) = (0, d);
    }
    let n = e - s;
    // villa pads short stacks only when the reader asks for it, which this one does not
    ensure!(n >= DEPTH as i64, "layers [{s}, {e}) hold {n} layers; the reader needs {DEPTH}");
    // center crop, choosing the upper center for even excess
    let first = s + n / 2 - DEPTH as i64 / 2;
    Ok(if reverse { ((first + DEPTH as i64 - 1) as i32, -1) } else { (first as i32, 1) })
}

/// For every tile position (row-major over `ys` x `xs`): whether villa's low-resolution mask
/// schedules it, and per layer whether the tile's window on that layer holds a nonzero voxel
pub struct Occupancy {
    pub ys: Vec<usize>,
    pub xs: Vec<usize>,
    /// The tile edge in pixels
    pub patch: usize,
    depth: usize,
    scheduled: Vec<bool>,
    bits: Vec<bool>,
}

impl Occupancy {
    /// `lowres` is villa's occupancy scan of the input ([`LowRes::scan`]), none for a bare array;
    /// `patch` the tile edge in pixels
    pub fn new(vol: VolumeRef, stride: usize, lowres: Option<&LowRes>, patch: usize) -> Self {
        let (h, w) = (vol.height, vol.width);
        let rect = |y0: usize, x0: usize| [y0, (y0 + patch).min(h), x0, (x0 + patch).min(w)];
        Self::over(vol, (h, w), stride, lowres, patch, rect)
    }

    /// The grid of `vol` turned a quarter, numpy `rot90(vol, 1, axes=(1, 2))`, `[depth, width, height]`,
    /// the second read of the 8-fold recipe ([`crate::tta`]), without turning the volume: rotated row `i`
    /// is column `width - 1 - i` and rotated column `j` is row `j`. No low-resolution scan, since the
    /// recipe's reads never run one
    pub fn quarter(vol: VolumeRef, stride: usize, patch: usize) -> Self {
        let (h, w) = (vol.height, vol.width);
        let rect = |y0: usize, x0: usize| [x0, (x0 + patch).min(h), w - (y0 + patch).min(w), w - y0];
        Self::over(vol, (w, h), stride, None, patch, rect)
    }

    /// The grid over a frame of `size` (height, width), where `rect` gives each tile's
    /// `[row0, row1, col0, col1]` in the volume as stored
    fn over(
        vol: VolumeRef,
        size: (usize, usize),
        stride: usize,
        lowres: Option<&LowRes>,
        patch: usize,
        rect: impl Fn(usize, usize) -> [usize; 4] + Sync,
    ) -> Self {
        let (ys, xs) = (positions(size.0, stride, patch), positions(size.1, stride, patch));
        let (h, w) = (vol.height, vol.width);
        let rect = &rect;
        let rects: Vec<[usize; 4]> = ys.iter().flat_map(|&y| xs.iter().map(move |&x| rect(y, x))).collect();
        let scheduled: Vec<bool> =
            rects.iter().map(|&[r0, r1, c0, c1]| lowres.is_none_or(|m| m.any(r0, c0, r1 - r0, c1 - c0))).collect();
        let bits: Vec<bool> = rects
            .par_iter()
            .flat_map_iter(|&[r0, r1, c0, c1]| {
                (0..vol.depth).map(move |z| {
                    let plane = &vol.data[z * h * w..(z + 1) * h * w];
                    (r0..r1).any(|y| plane[y * w + c0..y * w + c1].iter().any(|&v| v != 0))
                })
            })
            .collect();
        Self { ys, xs, patch, depth: vol.depth, scheduled, bits }
    }

    /// Whether villa runs tile `t` for the `DEPTH` layers from `first` by `step`: scheduled by
    /// the low-resolution mask, and a nonzero voxel on one of those layers
    pub fn keep(&self, t: usize, first: i32, step: i32) -> bool {
        self.scheduled[t] && (0..DEPTH as i32).any(|k| self.bits[t * self.depth + (first + step * k) as usize])
    }

    /// Tile positions the low-resolution mask schedules
    pub fn scheduled(&self) -> usize {
        self.scheduled.iter().filter(|&&s| s).count()
    }
}

/// One output map: per tile position, its grid slot or -1 when skipped; the tile's logits are slot
/// `s` of a plain read, slots `4 s..4 s + 4` of a mirror4 read
pub struct MapPlan {
    pub grid: Vec<i32>,
    pub tiles: usize,
}

/// The tiles of every map packed into fixed-size batches, and each map's blend inputs
pub struct Schedule {
    /// `[batches * batch * DESC]` descriptors: y0, x0, first layer, step, logits slot, flip bits
    /// (0, or the mirror4 copy), then unused;
    /// a short last batch is padded with copies of its first tile at slot -1
    pub desc: Vec<i32>,
    pub batches: usize,
    pub slots: usize,
    pub maps: Vec<MapPlan>,
    /// The batch after which each map's last tile has been computed
    pub map_done_after: Vec<usize>,
    pub ys: Vec<u32>,
    pub xs: Vec<u32>,
    pub y_span: Vec<[u32; 2]>,
    pub x_span: Vec<[u32; 2]>,
}

impl Schedule {
    /// Schedules `layers` (first layer and step per map) over `occ`'s grid, each kept tile as
    /// `mirrors` consecutive copies: 1, or 4 for a mirror4 read ([`crate::tta`]), whose copy `k` carries
    /// its flip bits `k` in descriptor field 5 and logits slot `4 s + k` for grid slot `s`
    pub fn new(
        occ: &Occupancy,
        height: usize,
        width: usize,
        layers: &[(i32, i32)],
        batch: usize,
        mirrors: usize,
    ) -> Result<Self> {
        ensure!(batch > 0, "batch size must be positive");
        ensure!(matches!(mirrors, 1 | 4), "{mirrors} copies per tile; 1 or 4");
        let mut desc = Vec::new();
        let mut maps = Vec::new();
        let mut map_done_after = Vec::new();
        // tiles kept so far over every map: the grid slot of the next one
        let mut tile = 0i32;
        let nx = occ.xs.len();
        for &(first, step) in layers {
            let mut grid = vec![-1; occ.ys.len() * nx];
            let mut tiles = 0;
            for (t, g) in grid.iter_mut().enumerate() {
                if occ.keep(t, first, step) {
                    *g = tile;
                    for k in 0..mirrors as i32 {
                        let (y0, x0) = (occ.ys[t / nx] as i32, occ.xs[t % nx] as i32);
                        desc.extend_from_slice(&[y0, x0, first, step, tile * mirrors as i32 + k, k, 0, 0]);
                    }
                    tile += 1;
                    tiles += 1;
                }
            }
            maps.push(MapPlan { grid, tiles });
            map_done_after.push((tile as usize * mirrors).div_ceil(batch).max(1) - 1);
        }
        let slots = tile as usize * mirrors;
        let batches = slots.div_ceil(batch);
        if batches > 0 {
            let pad = batches * batch - slots;
            let tail: Vec<i32> = desc[(batches - 1) * batch * DESC..(batches - 1) * batch * DESC + DESC].to_vec();
            for _ in 0..pad {
                desc.extend_from_slice(&tail[..4]);
                desc.extend_from_slice(&[-1, 0, 0, 0]);
            }
        }
        let span = |pos: &[usize], len: usize| -> Vec<[u32; 2]> {
            (0..len)
                .map(|p| {
                    let first = pos.iter().position(|&s| s + occ.patch > p).unwrap_or(pos.len());
                    let end = pos.iter().position(|&s| s > p).unwrap_or(pos.len());
                    [first as u32, end as u32]
                })
                .collect()
        };
        Ok(Self {
            desc,
            batches,
            slots,
            maps,
            map_done_after,
            ys: occ.ys.iter().map(|&v| v as u32).collect(),
            xs: occ.xs.iter().map(|&v| v as u32).collect(),
            y_span: span(&occ.ys, height),
            x_span: span(&occ.xs, width),
        })
    }
}

/// numpy's `percentile(..., q)` interpolation parameters on `n` sorted values, as
/// `tile_stats` takes them: the lower index, `t` and `1 - t`, both float32 from float64
pub fn percentile_params(n: usize, q: f64) -> (u32, f32, f32) {
    let index = (n - 1) as f64 * q / 100.0;
    let lower = index.floor();
    let gamma = index - lower;
    (lower as u32, gamma as f32, (1.0 - gamma) as f32)
}

/// fastink's `_pct_sorted(s, q)` parameters on `n` sorted values (the 306B recipe's torch
/// normaliser): index `q / 100 * (n - 1)` in float64 as Python computes it, then `t` and `1 - t`
/// in float64, each cast to float32 as torch casts a Python scalar for a float32 tensor
pub fn percentile_params_torch(n: usize, q: f64) -> (u32, f32, f32) {
    let index = q / 100.0 * (n - 1) as f64;
    let lower = index.floor();
    let t = index - lower;
    (lower as u32, t as f32, (1.0 - t) as f32)
}

/// CPU replicas of numpy's float32 steps, the references the normaliser tests check against
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// numpy's float32 pairwise sum (`pairwise_sum` in `loops_utils.h.src`): blocks of at most 128
    /// summed with 8 accumulators, halves split at a multiple of 8
    pub fn pairwise_f32(a: &[f32]) -> f32 {
        let n = a.len();
        if n < 8 {
            let mut r = 0f32;
            a.iter().for_each(|&v| r += v);
            r
        } else if n <= 128 {
            let mut r = [0f32; 8];
            r.copy_from_slice(&a[..8]);
            let mut i = 8;
            while i < n - n % 8 {
                (0..8).for_each(|j| r[j] += a[i + j]);
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            a[i..].iter().for_each(|&v| res += v);
            res
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            pairwise_f32(&a[..n2]) + pairwise_f32(&a[n2..])
        }
    }

    /// `np.std` of a float32 array as numpy computes it: float32 pairwise sums for the mean and
    /// the squared deviations, float32 divisions and square root
    pub fn numpy_std(v: &[u8]) -> f32 {
        let x: Vec<f32> = v.iter().map(|&b| b as f32).collect();
        let n = x.len() as f32;
        let mean = pairwise_f32(&x) / n;
        let sq: Vec<f32> = x.iter().map(|&b| (b - mean) * (b - mean)).collect();
        (pairwise_f32(&sq) / n).sqrt()
    }

    /// villa `normalize_robust` on one tile's uint8 voxels, replaying numpy's float32 steps; the
    /// CPU reference for `tile_stats`, returning `[lo, hi, median, divisor]`
    pub fn robust_stats(v: &[u8]) -> [f32; 4] {
        let n = v.len();
        let mut s: Vec<u8> = v.to_vec();
        s.sort_unstable();
        let lerp = |q: f64| {
            let (i, t, omt) = percentile_params(n, q);
            let (a, b) = (s[i as usize] as f32, s[(i as usize + 1).min(n - 1)] as f32);
            let d = b - a;
            if t >= 0.5 { b - d * omt } else { a + d * t }
        };
        let (lo, hi) = (lerp(1.0), lerp(99.0));
        let clip = |x: u8| (x as f32).max(lo).min(hi);
        let median_of = |sorted: &[f32]| {
            let m = sorted.len();
            if m % 2 == 1 { sorted[m / 2] } else { (sorted[m / 2 - 1] + sorted[m / 2]) / 2.0 }
        };
        let clipped: Vec<f32> = s.iter().map(|&x| clip(x)).collect();
        let median = median_of(&clipped);
        let mut dev: Vec<f32> = clipped.iter().map(|&x| (x - median).abs()).collect();
        dev.sort_unstable_by(f32::total_cmp);
        let mad = median_of(&dev);
        let scaled = 1.4826f64 * mad as f64;
        let divisor = if scaled >= 1e-6 {
            scaled
        } else {
            let std = numpy_std(v) as f64;
            let span = (hi as f64 - lo as f64) / 2.0;
            if std >= 1e-6 {
                std
            } else if span >= 1e-6 {
                span
            } else {
                1.0
            }
        };
        [lo, hi, median, divisor as f32]
    }

    /// fastink `robust_norm_gpu` (the 306B recipe's torch normaliser) on one tile's uint8 voxels,
    /// replaying its float32 steps; the CPU reference for `tile_stats` with `fastink` 1, for tiles
    /// whose MAD does not fall back, returning `[lo, hi, median, divisor]`
    pub fn fastink_stats(v: &[u8]) -> [f32; 4] {
        let n = v.len();
        let pct = |sorted: &[f32], q: f64| {
            let (i, t, omt) = percentile_params_torch(n, q);
            sorted[i as usize] * omt + sorted[(i as usize + 1).min(n - 1)] * t
        };
        let mut s: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        s.sort_unstable_by(f32::total_cmp);
        let (lo, hi) = (pct(&s, 1.0), pct(&s, 99.0));
        let clipped: Vec<f32> = s.iter().map(|&x| x.max(lo).min(hi)).collect();
        let median = pct(&clipped, 50.0);
        let mut dev: Vec<f32> = clipped.iter().map(|&x| (x - median).abs()).collect();
        dev.sort_unstable_by(f32::total_cmp);
        let divisor = 1.4826f32 * pct(&dev, 50.0);
        assert!(divisor >= 1e-6, "fastink_stats covers tiles whose MAD does not fall back");
        [lo, hi, median, divisor]
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{fastink_stats, pairwise_f32, robust_stats};
    use super::*;
    #[cfg(not(target_os = "macos"))]
    use crate::model::STD_BLOCKS_384;
    use crate::tta::MIRRORS;
    use crate::zarr::Volume;
    use sha2::{Digest, Sha256};

    /// The fixture directory: `INKRS_FIXTURES`, else the crate's `fixtures/` (which holds only the Hann
    /// window; the normaliser fixtures come from scripts/norm_fixture.py)
    fn fixtures() -> std::path::PathBuf {
        std::env::var_os("INKRS_FIXTURES")
            .map_or_else(|| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"), Into::into)
    }

    /// The Python with villa 0fb38c45 and tifffile installed, from `INKRS_PYTHON`; tests that need it
    /// skip without it
    fn python() -> Option<String> {
        std::env::var("INKRS_PYTHON").ok().filter(|p| !p.is_empty())
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|v| format!("{v:02x}")).collect()
    }

    /// `robust_stats` and the normalised float32 patch against villa on every tile of the
    /// fixture maps (scripts/norm_fixture.py) of the input named by the environment variable `zarr`;
    /// counts exact matches per path
    fn check(fixture: &str, zarr: &str) {
        let path = fixtures().join(fixture);
        let Some(input) = std::env::var_os(zarr) else {
            eprintln!("skipped: {zarr} is not set");
            return;
        };
        if !path.exists() {
            eprintln!("skipped: no fixture {}", path.display());
            return;
        }
        let vol = Volume::read(std::path::Path::new(&input)).unwrap();
        let (h, w) = (vol.height, vol.width);
        let (mut n, mut fb, mut bad_stats, mut bad_patch, mut bad_fb) = (0, 0, 0, 0, 0);
        for line in std::fs::read_to_string(&path).unwrap().lines() {
            let r: serde_json::Value = serde_json::from_str(line).unwrap();
            let g = |k: &str| r[k].as_i64().unwrap();
            let (y0, x0, first, step) = (g("y0") as usize, g("x0") as usize, g("first") as i32, g("step") as i32);
            let fallback = r["fallback"].as_bool().unwrap();
            let mut v = Vec::with_capacity(DEPTH * PATCH * PATCH);
            for k in 0..DEPTH as i32 {
                let z = (first + step * k) as usize;
                for y in y0..y0 + PATCH {
                    v.extend_from_slice(&vol.data[(z * h + y) * w + x0..(z * h + y) * w + x0 + PATCH]);
                }
            }
            let s = robust_stats(&v);
            let want = ["lo", "hi", "median", "divisor"].map(|k| r[k].as_f64().unwrap() as f32);
            let stats_ok = s == want;
            let out: Vec<u8> =
                v.iter().flat_map(|&x| (((x as f32).max(s[0]).min(s[1]) - s[2]) / s[3]).to_le_bytes()).collect();
            let patch_ok = hex(&Sha256::digest(&out)) == r["sha256"].as_str().unwrap();
            n += 1;
            fb += fallback as usize;
            bad_stats += !stats_ok as usize;
            bad_patch += !patch_ok as usize;
            bad_fb += (fallback && !patch_ok) as usize;
            if !stats_ok && bad_stats <= 5 {
                eprintln!("{fixture} ({y0},{x0},{first},{step}) fallback={fallback}: {s:?} != {want:?}");
            }
        }
        eprintln!(
            "{fixture}: {n} tiles ({fb} fallback): {bad_stats} stats and {bad_patch} patches differ ({bad_fb} of them fallback)"
        );
        assert_eq!((bad_stats, bad_patch), (0, 0));
    }

    /// The 306B recipe's TTA reads normalise with fastink's torch arithmetic, which rounds the
    /// upper percentile differently from numpy on this tile; the expected values are fastink's
    /// own `robust_norm_gpu` on it (torch 2.13, CPU)
    #[test]
    fn fastink_normaliser_rounds_as_torch_does() {
        let mut v = Vec::with_capacity(DEPTH * 128 * 128);
        for (g, c) in [(3u8, 92842), (4, 90057), (5, 92843), (237, 2786)] {
            v.extend(std::iter::repeat_n(g, c));
        }
        assert_eq!(v.len(), DEPTH * 128 * 128);
        let [lo, hi, median, divisor] = fastink_stats(&v);
        assert_eq!((lo, hi.to_bits(), median), (3.0, 0x432e_5c2a, 4.0));
        assert_ne!(robust_stats(&v)[1].to_bits(), hi.to_bits(), "numpy's percentile rounds this tile otherwise");
        let norm = |g: f32| half::f16::from_f32((g.max(lo).min(hi) - median) / divisor).to_f32();
        assert_eq!((norm(3.0), norm(4.0), norm(5.0), norm(237.0)), (-0.674_316_4, 0.0, 0.674_316_4, 114.9375));
    }

    #[test]
    fn normaliser_matches_villa_s1() {
        check("norm-s1.jsonl", "INKRS_S1");
    }

    #[test]
    fn normaliser_matches_villa_w048() {
        check("norm-w048.jsonl", "INKRS_W048");
    }

    /// `tile_std` sums a tile as 2048 nodes of a 64- and a 72-element leaf, paired level by
    /// level; that must be numpy's tree at this size
    #[test]
    fn pairwise_tree_is_uniform_at_tile_size() {
        let n = DEPTH * PATCH * PATCH;
        let mut seed = 1u64;
        let a: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 40) as f32 / 1024.0
            })
            .collect();
        let mut nodes: Vec<f32> = (0..2048)
            .map(|k| pairwise_f32(&a[136 * k..136 * k + 64]) + pairwise_f32(&a[136 * k + 64..136 * k + 136]))
            .collect();
        while nodes.len() > 1 {
            nodes = nodes.chunks(2).map(|p| p[0] + p[1]).collect();
        }
        assert_eq!(nodes[0].to_bits(), pairwise_f32(&a).to_bits());
    }

    /// `tile_std_384` sums a 17 x 384 x 384 tile as 2048 nodes of 1224, each
    /// `(304 + 304) + (304 + (152 + 160))` with 152 = 72 + 80 and 160 = 80 + 80 leaves, paired
    /// level by level; that must be numpy's tree at this size
    #[test]
    fn pairwise_tree_384() {
        let n = DEPTH * 384 * 384;
        let mut seed = 7u64;
        let a: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 40) as f32 / 1024.0
            })
            .collect();
        let leaf = |s: usize, len: usize| pairwise_f32(&a[s..s + len]);
        let s152 = |s: usize| leaf(s, 72) + leaf(s + 72, 80);
        let s160 = |s: usize| leaf(s, 80) + leaf(s + 80, 80);
        let s304 = |s: usize| s152(s) + s152(s + 152);
        let s312 = |s: usize| s152(s) + s160(s + 152);
        let mut nodes: Vec<f32> = (0..2048)
            .map(|k| {
                let s = 1224 * k;
                (s304(s) + s304(s + 304)) + (s304(s + 608) + s312(s + 912))
            })
            .collect();
        while nodes.len() > 1 {
            nodes = nodes.chunks(2).map(|p| p[0] + p[1]).collect();
        }
        assert_eq!(nodes[0].to_bits(), pairwise_f32(&a).to_bits());
    }

    /// `tile_sum_384` and `tile_sq_384` split the 384 tile's pairwise sum into 262144 chains (8 per
    /// leaf, `start + j + 8i`), summed over aligned power-of-two groups: 32 per warp, 8 warps, 4 rounds
    /// per block, then [`STD_BLOCKS_384`] block partials. That must be numpy's tree, on voxel-like
    /// values and on their squared deviations, where float32 rounding is heavy past 2^24. The CUDA
    /// kernels' tree; the Metal backend's normaliser is its own
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn pairwise_chains_384() {
        fn tree(v: &[f32]) -> f32 {
            if v.len() == 1 { v[0] } else { tree(&v[..v.len() / 2]) + tree(&v[v.len() / 2..]) }
        }
        let n = DEPTH * 384 * 384;
        let mut seed = 11u64;
        let vox: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) % 256) as f32
            })
            .collect();
        let mean = pairwise_f32(&vox) / n as f32;
        let sq: Vec<f32> = vox.iter().map(|&v| (v - mean) * (v - mean)).collect();
        for a in [&vox, &sq] {
            // the kernel's `Src::chain`
            let chain = |c: usize| {
                let (k, l, j) = (c >> 7, (c >> 3) & 15, c & 7);
                let start = k * 1224 + (l >> 1) * 152 + (l & 1) * 72 + usize::from(l == 15) * 8 + j;
                let len = if l & 1 == 1 || l == 14 { 10 } else { 9 };
                (1..len).fold(a[start], |r, i| r + a[start + 8 * i])
            };
            let chains: Vec<f32> = (0..2048 * 128).map(chain).collect();
            let rounds: Vec<f32> =
                chains.chunks(256).map(|r| tree(&r.chunks(32).map(tree).collect::<Vec<_>>())).collect();
            let blocks: Vec<f32> = rounds.chunks(4).map(tree).collect();
            assert_eq!(blocks.len(), STD_BLOCKS_384);
            let total = tree(&blocks.chunks(32).map(tree).collect::<Vec<_>>());
            assert_eq!(total.to_bits(), pairwise_f32(a).to_bits());
        }
    }

    #[test]
    fn hann_matches_villa() {
        // villa's compute_importance_map_2d(patch_size=(128, 128), mode="hann"), float32 little-endian
        let path = fixtures().join("hann128.f32");
        let want: Vec<f32> =
            std::fs::read(path).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        assert_eq!(hann_weights(Edge::P128), want);
    }

    #[test]
    fn hann384_matches_villa() {
        // villa's compute_importance_map_2d(patch_size=(384, 384), mode="hann"), float32 little-endian
        let path = fixtures().join("hann384.f32");
        let want: Vec<f32> =
            std::fs::read(path).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        assert_eq!(hann_weights(Edge::P384), want);
    }

    /// One pyramid level of a test group: its key, shape and voxel values by `(z, y, x)`
    type Level<'a> = (&'a str, [usize; 3], &'a dyn Fn(usize, usize, usize) -> u8);

    /// Writes a zarr v2 group at `dir` with one uncompressed single-chunk `uint8` array per level
    fn write_group(dir: &std::path::Path, levels: &[Level]) {
        if dir.exists() {
            std::fs::remove_dir_all(dir).unwrap();
        }
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(".zgroup"), r#"{"zarr_format": 2}"#).unwrap();
        for (key, [d, h, w], f) in levels {
            let a = dir.join(key);
            std::fs::create_dir_all(&a).unwrap();
            let meta = serde_json::json!({"zarr_format": 2, "shape": [d, h, w], "chunks": [d, h, w], "dtype": "|u1", "fill_value": 0,
                "order": "C", "filters": null, "compressor": null, "dimension_separator": "."});
            std::fs::write(a.join(".zarray"), meta.to_string()).unwrap();
            let data: Vec<u8> = (0..d * h * w).map(|i| f(i / (h * w), i / w % h, i % w)).collect();
            std::fs::write(a.join("0.0.0"), data).unwrap();
        }
    }

    /// The tiles `(y0, x0)` inkrs's schedule keeps for the map at layer start `s`
    fn inkrs_tiles(dir: &std::path::Path, stride: usize, s: i64, reverse: bool) -> Vec<(usize, usize)> {
        let vol = Volume::read(dir).unwrap();
        let low = LowRes::scan(dir, vol.height, vol.width).unwrap();
        let occ = Occupancy::new(vol.view(), stride, low.as_ref(), PATCH);
        let layers = select_layers(vol.depth, Some(s), Some(s + DEPTH as i64), reverse).unwrap();
        let sched = Schedule::new(&occ, vol.height, vol.width, &[layers], 4, 1).unwrap();
        let nx = occ.xs.len();
        sched.maps[0]
            .grid
            .iter()
            .enumerate()
            .filter(|(_, g)| **g >= 0)
            .map(|(t, _)| (occ.ys[t / nx], occ.xs[t % nx]))
            .collect()
    }

    /// villa's kept tiles for the same map, from scripts/villa_coverage.py
    fn villa_tiles(dir: &std::path::Path, stride: usize, s: i64, reverse: bool) -> (String, Vec<(usize, usize)>) {
        let script = concat!(env!("CARGO_MANIFEST_DIR"), "/scripts/villa_coverage.py");
        let mut cmd = std::process::Command::new(python().unwrap());
        cmd.args(["-I", script, dir.to_str().unwrap(), &stride.to_string(), &s.to_string()]);
        if reverse {
            cmd.arg("--reverse");
        }
        let out = cmd.output().unwrap();
        assert!(out.status.success(), "villa_coverage.py failed: {}", String::from_utf8_lossy(&out.stderr));
        let text = String::from_utf8(out.stdout).unwrap();
        let v: serde_json::Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
        let tiles = v["tiles"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| (t[0].as_u64().unwrap() as usize, t[1].as_u64().unwrap() as usize))
            .collect();
        (v["level"].as_str().unwrap().to_owned(), tiles)
    }

    /// The 8-fold TTA's quarter-turned grid, read through the rectangle mapping, keeps exactly the tiles
    /// a grid over the volume turned with numpy's `rot90(vol, 1, axes=(1, 2))` keeps, layer by layer
    #[test]
    fn quarter_grid_matches_a_turned_volume() {
        // a sparse volume whose nonzero voxels differ per layer, on a size neither stride divides
        let (d, h, w) = (20, 300, 430);
        let data: Vec<u8> = (0..d * h * w)
            .map(|i| {
                let (z, y, x) = (i / (h * w), (i / w) % h, i % w);
                u8::from((y * 7 + x * 13 + z * 29) % 997 == 0 && (x > 150 || y < 90))
            })
            .collect();
        let vol = VolumeRef::new(&data, d, h, w).unwrap();
        // turned[z, i, j] = vol[z, j, w - 1 - i], [d, w, h]
        let turned: Vec<u8> = (0..d * w * h)
            .map(|e| {
                let (z, i, j) = (e / (w * h), (e / h) % w, e % h);
                data[z * h * w + j * w + (w - 1 - i)]
            })
            .collect();
        let tv = VolumeRef::new(&turned, d, w, h).unwrap();
        for stride in [64, 50] {
            let quarter = Occupancy::quarter(vol, stride, PATCH);
            let direct = Occupancy::new(tv, stride, None, PATCH);
            assert_eq!((&quarter.ys, &quarter.xs), (&direct.ys, &direct.xs), "stride {stride}");
            assert_eq!(quarter.bits, direct.bits, "stride {stride}");
            assert!(quarter.bits.iter().any(|&b| b) && quarter.bits.iter().any(|&b| !b), "stride {stride}: degenerate");
        }
    }

    /// A mirror4 schedule carries each kept tile as four consecutive copies with flip bits 0..3 and slots
    /// 4 s..4 s + 4, its grid holds the tile ordinal, and its padding copies run at slot -1
    #[test]
    fn mirror4_schedule_copies_each_tile() {
        let (d, h, w) = (20, 200, 260);
        let data: Vec<u8> = (0..d * h * w).map(|i| u8::from(i % (h * w) % w > 100)).collect();
        let vol = VolumeRef::new(&data, d, h, w).unwrap();
        let occ = Occupancy::new(vol, 64, None, PATCH);
        let plain = Schedule::new(&occ, h, w, &[(0, 1), (3, 1)], 5, 1).unwrap();
        let tta = Schedule::new(&occ, h, w, &[(0, 1), (3, 1)], 5, MIRRORS).unwrap();
        assert_eq!(tta.slots, MIRRORS * plain.slots);
        assert_eq!(tta.batches, (MIRRORS * plain.slots).div_ceil(5));
        for (p, t) in plain.maps.iter().zip(&tta.maps) {
            assert_eq!((&p.grid, p.tiles), (&t.grid, t.tiles));
        }
        for s in 0..plain.slots {
            let tile = &plain.desc[s * DESC..s * DESC + 4];
            for k in 0..MIRRORS {
                let c = &tta.desc[(MIRRORS * s + k) * DESC..(MIRRORS * s + k + 1) * DESC];
                assert_eq!(&c[..4], tile);
                assert_eq!((c[4], c[5]), ((MIRRORS * s + k) as i32, k as i32));
            }
        }
        let pad = &tta.desc[tta.slots * DESC..];
        assert_eq!(pad.len(), (tta.batches * 5 - tta.slots) * DESC);
        assert!(pad.chunks(DESC).all(|c| c[4] == -1 && c[5] == 0));
        assert!(Schedule::new(&occ, h, w, &[(0, 1)], 5, 2).is_err());
    }

    /// villa's first skip on multiscale groups: tiles whose level 0 is nonzero but whose
    /// occupancy level is empty are dropped, the occupancy level is "3" when present and the
    /// nearest numeric level otherwise, and the raw-window skip still applies after it
    #[test]
    fn multiscale_skip_matches_villa() {
        if python().is_none() {
            eprintln!("skipped: INKRS_PYTHON is not set");
            return;
        }
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/test-zarr");
        let (d, h, w) = (20, 384, 448);
        // level 0: ink-bearing voxels in region A (top left) and region B (bottom right)
        let in_a = |y: usize, x: usize| y < 100 && x < 100;
        let in_b = |y: usize, x: usize| y >= 250 && x >= 300;
        let level0 = move |_z: usize, y: usize, x: usize| {
            if in_a(y, x) || in_b(y, x) { 1 + (y * 31 + x) as u8 % 200 } else { 0 }
        };
        // the occupancy level (1/8 scale): A, plus a region C at level-0 rows/cols 160..240 that
        // level 0 leaves empty, so C is scheduled and then dropped by the raw-window skip
        let low = |_z: usize, y: usize, x: usize| {
            u8::from((y < 13 && x < 13) || ((20..30).contains(&y) && (20..30).contains(&x)))
        };
        let full = |_z: usize, _y: usize, _x: usize| 1u8;
        let cases: [(&str, &str, Vec<Level>); 2] = [
            // "3" present: preferred over a level 1 that marks everything occupied
            (
                "levels-0-1-3",
                "3",
                vec![("0", [d, h, w], &level0), ("1", [10, 192, 224], &full), ("3", [3, 48, 56], &low)],
            ),
            // no "3": level 2 is nearer than 5 (here level 2 holds the 1/8 mask, so the scale is 8)
            ("levels-0-2-5", "2", vec![("0", [d, h, w], &level0), ("2", [3, 48, 56], &low), ("5", [1, 12, 14], &full)]),
        ];
        for (name, want_level, levels) in cases {
            let dir = root.join(name);
            write_group(&dir, &levels);
            let vol = Volume::read(&dir).unwrap();
            let lowres = LowRes::scan(&dir, vol.height, vol.width).unwrap().unwrap();
            assert_eq!((lowres.level.as_str(), lowres.scale_y, lowres.scale_x), (want_level, 8, 8), "{name}");
            // a level-0-only rule keeps more tiles here, so the test can tell the two apart
            let level0_only = Occupancy::new(vol.view(), 64, None, PATCH);
            for (s, reverse) in [(2, false), (2, true)] {
                let ours = inkrs_tiles(&dir, 64, s, reverse);
                let (level, theirs) = villa_tiles(&dir, 64, s, reverse);
                assert_eq!(level, want_level, "{name}: villa read level {level}");
                assert_eq!(ours, theirs, "{name} s={s} reverse={reverse}: inkrs and villa keep different tiles");
                let layers = select_layers(d, Some(s), Some(s + DEPTH as i64), reverse).unwrap();
                let n0 = (0..level0_only.ys.len() * level0_only.xs.len())
                    .filter(|&t| level0_only.keep(t, layers.0, layers.1))
                    .count();
                assert!(
                    n0 > ours.len() && !ours.is_empty(),
                    "{name}: {} tiles kept, {n0} by level 0 alone; the fixture does not exercise the skip",
                    ours.len()
                );
                eprintln!(
                    "{name} s={s} reverse={reverse}: villa and inkrs keep {} tiles (level 0 alone: {n0})",
                    ours.len()
                );
            }
        }
    }

    #[test]
    fn positions_and_layers_follow_villa() {
        assert_eq!(positions(300, 64, PATCH), vec![0, 64, 128, 172]);
        assert_eq!(positions(128, 64, PATCH), vec![0]);
        assert_eq!(select_layers(66, Some(25), Some(42), false).unwrap(), (25, 1));
        assert_eq!(select_layers(66, Some(25), Some(42), true).unwrap(), (41, -1));
        // even excess picks the upper center, as villa's center crop does
        assert_eq!(select_layers(66, Some(20), Some(40), false).unwrap(), (22, 1));
    }
}
