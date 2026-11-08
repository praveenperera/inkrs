//! v8-in's network on Metal at a fixed batch, FP16 storage as the reference's autocast runs it, with the kernels of
//! `v8in.metal`
//!
//! The structure and numerics follow the CUDA model (`v8in::model`): `v8_stats` and `v8_stem`, Bottleneck layers
//! [3, 4, 6, 3], each layer's depth max into the decoder's concatenation buffers, three decoder steps and the head.
//! Every BatchNorm is folded into its conv's weights (`w * scale`, rounded to FP16 once) and a float32 bias, and each
//! conv's epilogue computes `relu(conv + residual + bias)` in float32 and rounds once, as cuDNN's fused call does on
//! CUDA. A downsample's shift joins its block's conv3 bias, so the downsample is a plain conv
//!
//! The convs are Metal 4 tensor-op matmuls with no im2col: 1x1 convs are GEMMs (`v8_gemm`, after `v8_sub2` for a
//! stride-2 downsample), 3x3x3 and 3x3 convs run as 9 or 3 shifted K = 3c matmuls over a zero-padded input
//! (`v8_taps*`). A tap conv's input is written padded by its producer, the conv1 GEMM's epilogue or the decoder's
//! depth max and upsample, into a buffer that layout owns alone, so its zero border is written once at allocation

use std::collections::BTreeMap;

use color_eyre::eyre::{Result, ensure};
use half::f16;

use crate::metal::gpu::{Buf, Gpu};
use crate::sched::DESC;
use crate::v8in::host::{BLOCKS, DIMS, PLANES, batch_norm};
pub use crate::v8in::host::{READERS, check_contract, force_unfused, is_v8in_export};
use crate::v8in::{D, P, PH, PIX, SC, SD, StemTables};
use crate::weights::Checkpoint;

// conv epilogue flags, as v8in.metal's V8_*
const BIAS: u32 = 1;
const RES: u32 = 2;
const RELU: u32 = 4;
const PAD: u32 = 8;

/// A matmul kernel's output tile and SIMD groups (32 threads each)
#[derive(Clone, Copy, Debug)]
struct Tile {
    name: &'static str,
    bm: usize,
    bn: usize,
    sg: usize,
}

const fn tile(name: &'static str, bm: usize, bn: usize, sg: usize) -> Tile {
    Tile { name, bm, bn, sg }
}

/// How a conv runs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// 1x1x1, stride 1
    Gemm,
    /// 1x1x1, stride 2: `v8_sub2` then the GEMM
    Sub2,
    /// 3x3x3, pad 1, stride 1 or 2, over the padded input
    Taps3(usize),
    /// 3x3, pad 1, over the padded 2D input
    Taps2,
}

// the kernels v8in.metal instantiates, per conv family
const GEMM_TILES: [Tile; 4] = [
    tile("v8_gemm_64x64_sg4", 64, 64, 4),
    tile("v8_gemm_128x64_sg4", 128, 64, 4),
    tile("v8_gemm_128x64_sg8", 128, 64, 8),
    tile("v8_gemm_64x128_sg4", 64, 128, 4),
];
const TAPS3_S1_TILES: [Tile; 4] = [
    tile("v8_taps3_64x64_sg4_s1", 64, 64, 4),
    tile("v8_taps3_128x64_sg4_s1", 128, 64, 4),
    tile("v8_taps3_128x64_sg8_s1", 128, 64, 8),
    tile("v8_taps3_64x128_sg4_s1", 64, 128, 4),
];
const TAPS3_S2_TILES: [Tile; 3] = [
    tile("v8_taps3_32x64_sg4_s2", 32, 64, 4),
    tile("v8_taps3_32x128_sg4_s2", 32, 128, 4),
    tile("v8_taps3_64x64_sg4_s2", 64, 64, 4),
];
const TAPS2_TILES: [Tile; 3] = [
    tile("v8_taps2_64x64_sg4_s1", 64, 64, 4),
    tile("v8_taps2_128x64_sg4_s1", 128, 64, 4),
    tile("v8_taps2_64x128_sg4_s1", 64, 128, 4),
];

impl Kind {
    fn stride(self) -> usize {
        match self {
            Kind::Sub2 => 2,
            Kind::Taps3(s) => s,
            Kind::Gemm | Kind::Taps2 => 1,
        }
    }

    /// The kernels that can run a conv of this kind
    fn tiles(self) -> &'static [Tile] {
        match self {
            Kind::Gemm | Kind::Sub2 => &GEMM_TILES,
            Kind::Taps3(2) => &TAPS3_S2_TILES,
            Kind::Taps3(_) => &TAPS3_S1_TILES,
            Kind::Taps2 => &TAPS2_TILES,
        }
    }

    /// The kernel a conv of this kind runs with `o` outputs
    ///
    /// From `tests::tile_times` on the M4 Pro at batch 8: 64 x 64 on 4 SIMD groups is fastest, or within 2%, at
    /// every v8-in shape, by 3-30% over the wider tiles; a stride-2 tile is one output line, and 32 x 128 beats
    /// 32 x 64 by 1-4% wherever there are 128 outputs
    fn pick(self, o: usize) -> Tile {
        match self {
            Kind::Gemm | Kind::Sub2 => GEMM_TILES[0],
            Kind::Taps3(2) if o >= 128 => TAPS3_S2_TILES[1],
            Kind::Taps3(2) => TAPS3_S2_TILES[0],
            Kind::Taps3(_) => TAPS3_S1_TILES[0],
            Kind::Taps2 => TAPS2_TILES[0],
        }
    }
}

/// Refuses a tensor view whose element offsets overflow the int32 extents of Metal's tensor ops
fn i32_extent(what: &str, n: usize) -> Result<()> {
    ensure!(n < 1 << 31, "{what}: {n} elements overflow the Metal tensor ops' int32 indexing");
    Ok(())
}

/// Voxels of the zero-padded layout of `batch` volumes `[d, h, w]` (`d` 0: the 2D layout of `[h, w]` maps), plus
/// the tail the last sample's shifted tap views read past it
fn padded_voxels(batch: usize, [d, h, w]: [usize; 3]) -> usize {
    let (wp, plane) = (w + 2, (h + 2) * (w + 2));
    let per = if d == 0 { plane } else { (d + 2) * plane };
    batch * per + 2 * wp + 2
}

/// A conv with its BatchNorm folded: FP16 weights `[o, taps, c]`, the float32 bias, the input volume `din`
struct FConv {
    w: Buf,
    bias: Buf,
    c: usize,
    o: usize,
    kind: Kind,
    din: [usize; 3],
    /// the kernel it runs, one of `kind.tiles()`
    tile: Tile,
}

impl FConv {
    fn new(w: Buf, bias: Buf, (c, o): (usize, usize), kind: Kind, din: [usize; 3]) -> Self {
        Self { w, bias, c, o, kind, din, tile: kind.pick(o) }
    }

    /// `key.weight` (`[o, c, *kernel]`) times `scale[o]`, as `[o, *kernel, c]` FP16, and `shift` as the bias
    fn load(
        gpu: &Gpu,
        ck: &Checkpoint,
        key: &str,
        (c, o): (usize, usize),
        kind: Kind,
        din: [usize; 3],
        (scale, shift): (&[f64], &[f64]),
    ) -> Result<Self> {
        let kernel: &[usize] = match kind {
            Kind::Gemm | Kind::Sub2 => &[1, 1, 1],
            Kind::Taps3(_) => &[3, 3, 3],
            Kind::Taps2 => &[3, 3],
        };
        let taps: usize = kernel.iter().product();
        let mut dims = vec![o, c];
        dims.extend_from_slice(kernel);
        let w = ck.take(&format!("{key}.weight"), &dims)?;
        let mut folded = vec![f16::ZERO; o * taps * c];
        for ko in 0..o {
            for ci in 0..c {
                for t in 0..taps {
                    let v = w[(ko * c + ci) * taps + t] as f64 * scale[ko];
                    let h = f16::from_f64(v);
                    ensure!(h.is_finite(), "{key}: folded weight {v} overflows FP16");
                    folded[(ko * taps + t) * c + ci] = h;
                }
            }
        }
        let bias: Vec<f32> = shift.iter().map(|&v| v as f32).collect();
        Ok(Self::new(gpu.upload(&folded)?, gpu.upload(&bias)?, (c, o), kind, din))
    }

    fn stride(&self) -> usize {
        self.kind.stride()
    }

    fn dout(&self) -> [usize; 3] {
        self.din.map(|v| v / self.stride())
    }

    /// Output elements per sample (`d` 0 counts as 1)
    fn out_per(&self) -> usize {
        let [d, h, w] = self.dout();
        d.max(1) * h * w * self.o
    }

    fn describe(&self) -> String {
        let [_, h, w] = self.dout();
        format!("{:?} {}->{} @ {:?} -> {h}x{w} | {}", self.kind, self.c, self.o, self.din, self.tile.name)
    }
}

/// Runs [`FConv`]s at a fixed batch, with the scratch they need
struct Convs {
    batch: usize,
    /// the stride-2 downsample's subsampled input
    sub: Buf,
    /// bound to the unused residual and bias arguments
    unused: Buf,
}

impl Convs {
    /// `y = epi(conv(x))` at the runner's batch. `x` is plain for a GEMM and padded for a tap conv; `y` is plain, or
    /// padded with [`PAD`] (GEMMs only); `res` is a plain residual of `y`'s shape
    fn run(&self, gpu: &Gpu, conv: &FConv, x: &Buf, y: &Buf, res: Option<&Buf>, flags: u32) -> Result<()> {
        let b = self.batch;
        let need = |buf: &Buf, elems: usize, what: &str| -> Result<()> {
            ensure!(buf.bytes() >= elems * 2, "{what}: {} B, need {}", buf.bytes(), elems * 2);
            Ok(())
        };
        let [d, h, w] = conv.din;
        let [pd, ph, pw] = conv.dout();
        let m = b * pd.max(1) * ph * pw;
        // the GEMMs' staged epilogue moves 8 channels at a time
        ensure!(
            !matches!(conv.kind, Kind::Gemm | Kind::Sub2) || conv.o.is_multiple_of(8),
            "{:?} conv with {} outputs, not a multiple of 8",
            conv.kind,
            conv.o
        );
        // a GEMM and a tap kernel take the same argument kinds, so the dispatch check would not catch a swap
        ensure!(
            conv.kind.tiles().iter().any(|t| t.name == conv.tile.name),
            "{:?} conv planned on {}",
            conv.kind,
            conv.tile.name
        );
        ensure!(res.is_some() == (flags & RES != 0), "conv residual without the residual flag");
        if flags & PAD != 0 {
            ensure!(matches!(conv.kind, Kind::Gemm) && res.is_none(), "a padded output from a {:?} conv", conv.kind);
            need(y, padded_voxels(b, conv.dout()) * conv.o, "padded conv output")?;
        } else {
            need(y, m * conv.o, "conv output")?;
        }
        if let Some(r) = res {
            need(r, m * conv.o, "conv residual")?;
        }
        let res = res.unwrap_or(&self.unused).arg();
        let bias = if flags & BIAS != 0 { conv.bias.arg() } else { self.unused.arg() };
        let gemm = |input: &Buf| {
            let t = conv.tile;
            gpu.dispatch(
                t.name,
                [conv.o.div_ceil(t.bn), m.div_ceil(t.bm), 1],
                [32 * t.sg, 1, 1],
                &[
                    input.arg(),
                    conv.w.arg(),
                    y.arg(),
                    bias,
                    res,
                    m.into(),
                    conv.o.into(),
                    conv.c.into(),
                    flags.into(),
                    pd.into(),
                    ph.into(),
                    pw.into(),
                ],
            )
        };
        match conv.kind {
            Kind::Gemm => {
                need(x, m * conv.c, "conv input")?;
                gemm(x)
            }
            Kind::Sub2 => {
                need(x, b * d * h * w * conv.c, "conv input")?;
                need(&self.sub, m * conv.c, "subsampled conv input")?;
                let n = m * conv.c / 2;
                gpu.dispatch_1d(
                    "v8_sub2",
                    n,
                    &[self.sub.arg(), x.arg(), d.into(), h.into(), w.into(), conv.c.into(), n.into()],
                )?;
                gemm(&self.sub)
            }
            Kind::Taps3(_) | Kind::Taps2 => {
                need(x, padded_voxels(b, conv.din) * conv.c, "padded conv input")?;
                let s = conv.stride();
                let (dout, wp) = (if d == 0 { 1 } else { d / s }, w + 2);
                let t = conv.tile;
                let per_plane = if s == 1 { (ph * wp).div_ceil(t.bm) } else { ph * pw.div_ceil(t.bm) };
                gpu.dispatch(
                    t.name,
                    [conv.o.div_ceil(t.bn), b * dout * per_plane, 1],
                    [32 * t.sg, 1, 1],
                    &[
                        x.arg(),
                        conv.w.arg(),
                        y.arg(),
                        bias,
                        res,
                        b.into(),
                        d.max(1).into(),
                        h.into(),
                        w.into(),
                        conv.c.into(),
                        conv.o.into(),
                        flags.into(),
                    ],
                )
            }
        }
    }
}

/// One Bottleneck block
struct Bottleneck {
    conv1: FConv,
    conv2: FConv,
    conv3: FConv,
    downsample: Option<FConv>,
}

/// One decoder step: the 3x3 conv over `[skip, up]` and its BN-ReLU, at `hw`
struct DecStep {
    conv: FConv,
    hw: usize,
}

/// A per-stage callback of [`Model::forward`] for the P0 gate: the stage's name (`stats`, `stem`, `layer1`..
/// `layer4`), its buffer and its channels-last `[batch, depth, h, w, c]` shape (`stats` is `[batch, 1, 1, 1, 2]`
/// float32, the rest FP16); it runs after the stage is queued, before a later stage reuses the buffer
pub type StageHook<'h> = &'h mut dyn FnMut(&str, &Buf, [usize; 5]) -> Result<()>;

/// One of the last forward's decoder activations for the P0 gate: FP16, channels-last `shape` (`[batch, h, w, c]`);
/// `padded` says the buffer holds the zero-padded layout `[batch, h + 2, w + 2, c]`
pub struct Tap<'a> {
    pub name: String,
    pub buf: &'a Buf,
    pub shape: Vec<usize>,
    pub padded: bool,
}

/// Tile descriptors (`DESC` i32 per tile: y0, x0, first layer, step, slot, unused) on the GPU, checked before upload
/// against the 24-layer window, the volume and the probability slots, so the kernels, which have no bounds checks,
/// address only inside the volume and the probabilities
pub struct Descs {
    buf: Buf,
    batch: usize,
    batches: usize,
    slots: usize,
    height: usize,
    width: usize,
}

impl Descs {
    /// Checks `desc` (whole batches of `batch` tiles) for a `[24, height, width]` window and `slots` probability
    /// slots: every tile inside the volume, its 24 layers inside the window, and each slot `-1` (padding) or below
    /// `slots` and used once
    pub fn new(gpu: &Gpu, desc: &[i32], batch: usize, slots: usize, (height, width): (usize, usize)) -> Result<Self> {
        ensure!(
            batch > 0 && desc.len().is_multiple_of(batch * DESC),
            "{} descriptor words for batches of {batch}",
            desc.len()
        );
        let mut used = vec![false; slots];
        for (t, d) in desc.as_chunks::<DESC>().0.iter().enumerate() {
            let [y, x, first, step, slot] = [d[0], d[1], d[2], d[3], d[4]].map(i64::from);
            ensure!(
                y >= 0 && x >= 0 && (y as usize) + P <= height && (x as usize) + P <= width,
                "tile {t} at ({y}, {x}) past the {height} x {width} volume"
            );
            let last = first + step * (D as i64 - 1);
            ensure!(
                (0..D as i64).contains(&first) && (0..D as i64).contains(&last),
                "tile {t}: layers {first} + {step} k leave the {D}-layer window"
            );
            if slot == -1 {
                continue;
            }
            ensure!((0..slots as i64).contains(&slot), "tile {t}: slot {slot} of {slots}");
            ensure!(!std::mem::replace(&mut used[slot as usize], true), "tile {t}: slot {slot} used twice");
        }
        let batches = desc.len() / (batch * DESC);
        let buf = if desc.is_empty() { gpu.upload(&[0i32])? } else { gpu.upload(desc)? };
        Ok(Self { buf, batch, batches, slots, height, width })
    }
}

/// The network and its activations at batch `batch`
pub struct Model {
    batch: usize,
    in_norm: [f32; 2],
    nh: usize,
    stem_bufs: [Buf; 6],
    layers: Vec<Vec<Bottleneck>>,
    /// Each layer's output `[d, h, w]`
    dims: [[usize; 3]; 4],
    dec: Vec<DecStep>,
    head_w: Buf,
    head_b: f32,
    stats: Buf,
    /// the residual stream in turn, the conv2 output, the downsample
    stream: [Buf; 2],
    t2: Buf,
    ds: Buf,
    /// conv1 outputs, padded, by (volume, channels): each layout owns its buffer so its border stays zero
    t1: BTreeMap<([usize; 3], usize), Buf>,
    /// decoder: padded concatenation buffers per level 0..2, the deepest pooled map, each step's output
    cat: Vec<Buf>,
    p4: Buf,
    dout: Vec<Buf>,
    convs: Convs,
    /// Shapes that fall back to an unfused epilogue; none on Metal, kept for the report
    pub unfused: Vec<String>,
}

impl Model {
    /// Loads the weights and allocates every activation at `batch`; every checkpoint tensor must be consumed (the
    /// caller checks with [`Checkpoint::finish`])
    pub fn new(gpu: &Gpu, ck: &Checkpoint, batch: usize) -> Result<Self> {
        ensure!(batch > 0, "batch size must be positive");
        // the kernels index elements in u32 (the largest plain activation is the [B, 96, 64, 64, 256] stream) and
        // the GEMM rows in u32
        crate::hecate::u32_extent("activation", batch * SD * PH * PH * DIMS[0])?;

        let in_norm = [ck.take("normalization.weight", &[1])?[0], ck.take("normalization.bias", &[1])?[0]];
        let stem_w = ck.take("backbone.conv1.weight", &[SC, 1, 7, 7, 7])?;
        let (s, t) = batch_norm(ck, "backbone.bn1", SC)?;
        let stem = StemTables::new(&stem_w, &s, &t)?;
        let stem_bufs = [
            gpu.upload(&stem.w)?,
            gpu.upload(&stem.bias)?,
            gpu.upload(&stem.dcls)?,
            gpu.upload(&stem.dbase)?,
            gpu.upload(&stem.hcls)?,
            gpu.upload(&stem.hbase)?,
        ];

        let mut layers = Vec::new();
        let mut dims = [[0usize; 3]; 4];
        let (mut cin, mut d) = (SC, [SD, PH, PH]);
        let mut t1_layouts = Vec::new();
        let (mut t2_per, mut ds_per, mut sub_per, mut stream_per) = (0, 0, 0, SD * PH * PH * SC);
        for (l, (&planes, &nb)) in PLANES.iter().zip(&BLOCKS).enumerate() {
            let stride = if l == 0 { 1 } else { 2 };
            let (din, dout) = (d, d.map(|v| v / stride));
            let c = planes * 4;
            let mut blocks = Vec::new();
            for j in 0..nb {
                let key = format!("backbone.layer{}.{j}", l + 1);
                let (s0, bin, bdin) = if j == 0 { (stride, cin, din) } else { (1, c, dout) };
                let (s1, t1) = batch_norm(ck, &format!("{key}.bn1"), planes)?;
                let (s2, t2) = batch_norm(ck, &format!("{key}.bn2"), planes)?;
                let (s3, mut t3) = batch_norm(ck, &format!("{key}.bn3"), c)?;
                let downsample = if j == 0 {
                    let (sd, td) = batch_norm(ck, &format!("{key}.downsample.1"), c)?;
                    t3.iter_mut().zip(&td).for_each(|(a, b)| *a += b);
                    let kind = if s0 == 2 { Kind::Sub2 } else { Kind::Gemm };
                    let ds = FConv::load(
                        gpu,
                        ck,
                        &format!("{key}.downsample.0"),
                        (bin, c),
                        kind,
                        bdin,
                        (&sd, &vec![0.0; c]),
                    )?;
                    if kind == Kind::Sub2 {
                        sub_per = sub_per.max(dout.iter().product::<usize>() * bin);
                    }
                    Some(ds)
                } else {
                    None
                };
                let conv1 = FConv::load(gpu, ck, &format!("{key}.conv1"), (bin, planes), Kind::Gemm, bdin, (&s1, &t1))?;
                let conv2 =
                    FConv::load(gpu, ck, &format!("{key}.conv2"), (planes, planes), Kind::Taps3(s0), bdin, (&s2, &t2))?;
                let conv3 = FConv::load(gpu, ck, &format!("{key}.conv3"), (planes, c), Kind::Gemm, dout, (&s3, &t3))?;
                if !t1_layouts.contains(&(bdin, planes)) {
                    t1_layouts.push((bdin, planes));
                }
                t2_per = t2_per.max(conv2.out_per());
                stream_per = stream_per.max(conv3.out_per());
                if let Some(ds) = &downsample {
                    ds_per = ds_per.max(ds.out_per());
                }
                blocks.push(Bottleneck { conv1, conv2, conv3, downsample });
            }
            layers.push(blocks);
            dims[l] = dout;
            cin = c;
            d = dout;
        }

        // decoder: step i (0 = deepest) joins level 3 - i's skip with the upsampled deeper map
        let mut dec = Vec::new();
        for i in (1..4).rev() {
            let (cs, cl) = (DIMS[i - 1], DIMS[i]);
            let hw = dims[i - 1][1];
            let (s, t) = batch_norm(ck, &format!("decoder.convs.{}.1", i - 1), cs)?;
            let key = format!("decoder.convs.{}.0", i - 1);
            dec.push(DecStep {
                conv: FConv::load(gpu, ck, &key, (cs + cl, cs), Kind::Taps2, [0, hw, hw], (&s, &t))?,
                hw,
            });
        }
        // autocast casts the logit weight to FP16
        let head_w: Vec<f32> =
            ck.take("decoder.logit.weight", &[1, DIMS[0], 1, 1])?.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
        let head_b = ck.scalar("decoder.logit.bias")?;

        // NaN-filled, so a read of anything never written reaches the finite check; padded buffers are zeroed
        let nan = |elems: usize| gpu.alloc_fill(elems.max(1) * 2, 0xFF);
        let zero = |elems: usize| gpu.alloc(elems.max(1) * 2);
        let mut t1 = BTreeMap::new();
        for (din, c) in t1_layouts {
            let n = padded_voxels(batch, din) * c;
            i32_extent("padded conv input", n)?;
            t1.insert((din, c), zero(n)?);
        }
        let cat = (0..3)
            .map(|l| {
                let n = padded_voxels(batch, [0, dims[l][1], dims[l][2]]) * (DIMS[l] + DIMS[l + 1]);
                i32_extent("decoder concatenation", n)?;
                zero(n)
            })
            .collect::<Result<Vec<_>>>()?;
        let dout = dec.iter().map(|s| nan(batch * s.conv.out_per())).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            batch,
            in_norm,
            nh: stem.nh,
            stem_bufs,
            dims,
            head_w: gpu.upload(&head_w)?,
            head_b,
            stats: gpu.alloc_fill(batch * 8, 0xFF)?,
            stream: [nan(batch * stream_per)?, nan(batch * stream_per)?],
            t2: nan(batch * t2_per)?,
            ds: nan(batch * ds_per)?,
            t1,
            cat,
            p4: nan(batch * dims[3][1] * dims[3][2] * DIMS[3])?,
            dout,
            convs: Convs { batch, sub: nan(batch * sub_per)?, unused: gpu.alloc(4)? },
            layers,
            dec,
            unfused: Vec::new(),
        })
    }

    /// The distinct convs and the kernel each runs, for the report
    pub fn plans(&self) -> Vec<String> {
        let convs = self.layers.iter().flatten().flat_map(|b| {
            [Some(&b.conv1), Some(&b.conv2), Some(&b.conv3), b.downsample.as_ref()].into_iter().flatten()
        });
        let all = convs.chain(self.dec.iter().map(|s| &s.conv));
        all.map(FConv::describe).collect::<std::collections::BTreeSet<_>>().into_iter().collect()
    }

    /// `v8_dmax` of layer `l`'s output `x` (`[B, d, h, w, c]`) into channels `0..c` of its level's padded
    /// concatenation buffer, or of the plain deepest map for layer 4
    fn dmax(&self, gpu: &Gpu, l: usize, x: &Buf) -> Result<()> {
        let b = self.batch;
        let ([d, h, w], c) = (self.dims[l], DIMS[l]);
        let (y, ystride, pad) = if l < 3 { (&self.cat[l], c + DIMS[l + 1], true) } else { (&self.p4, c, false) };
        let rows = if pad { padded_voxels(b, [0, h, w]) } else { b * h * w };
        ensure!(y.bytes() >= rows * ystride * 2 && x.bytes() >= b * d * h * w * c * 2, "depth max past a buffer");
        let n = b * h * w * c / 2;
        gpu.dispatch_1d(
            "v8_dmax",
            n,
            &[
                y.arg(),
                x.arg(),
                d.into(),
                h.into(),
                w.into(),
                c.into(),
                ystride.into(),
                // channel offset 0: each level's max is the concatenation's first block (`[skip, up]`)
                0u32.into(),
                u32::from(pad).into(),
                n.into(),
            ],
        )
    }

    /// The batch the activations are allocated for; a different batch needs a new model
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Queues one forward: batch `desc.1` of the tiles `desc.0` describes, from the resident 24-layer `vol`
    /// (`[24, height, width]`), and each real tile's probability into its slot of `prob` (`[slots, 64 * 64]` float32)
    pub fn forward(
        &self,
        gpu: &Gpu,
        vol: (&Buf, usize, usize),
        (descs, bi): (&Descs, usize),
        prob: &Buf,
        sigmoid16: bool,
        mut hook: Option<StageHook<'_>>,
    ) -> Result<()> {
        let b = self.batch;
        let (vbuf, height, width) = vol;
        ensure!(vbuf.bytes() >= D * height * width, "volume buffer of {} B for {D} x {height} x {width}", vbuf.bytes());
        ensure!(
            descs.batch == b && bi < descs.batches && (descs.height, descs.width) == (height, width),
            "batch {bi} of {} descriptors (batch {}, {} x {}) for a batch-{b} forward on {height} x {width}",
            descs.batches,
            descs.batch,
            descs.height,
            descs.width
        );
        ensure!(prob.bytes() >= descs.slots * PIX * 4, "probabilities of {} B for {} slots", prob.bytes(), descs.slots);
        let dview = descs.buf.at::<i32>(bi * b * DESC);
        let [gamma, beta] = self.in_norm;
        gpu.dispatch(
            "v8_stats",
            [b, 1, 1],
            [256, 1, 1],
            &[self.stats.arg(), vbuf.arg(), dview, height.into(), width.into(), gamma.into(), beta.into()],
        )?;
        if let Some(h) = hook.as_mut() {
            h("stats", &self.stats, [b, 1, 1, 1, 2])?;
        }
        let [sw, sb, dc, db, hc, hb] = &self.stem_bufs;
        gpu.dispatch(
            "v8_stem",
            [2 * (PH / 8) * (PH / 8), SD, b],
            [128, 1, 1],
            &[
                self.stream[0].arg(),
                vbuf.arg(),
                dview,
                self.stats.arg(),
                sw.arg(),
                sb.arg(),
                dc.arg(),
                db.arg(),
                hc.arg(),
                hb.arg(),
                height.into(),
                width.into(),
                self.nh.into(),
            ],
        )?;
        if let Some(h) = hook.as_mut() {
            h("stem", &self.stream[0], [b, SD, PH, PH, SC])?;
        }

        // encoder, the residual stream alternating between the two stream buffers
        let mut cur = 0;
        for (l, blocks) in self.layers.iter().enumerate() {
            for blk in blocks {
                let (x, y) = (&self.stream[cur], &self.stream[1 - cur]);
                let residual = match &blk.downsample {
                    Some(dsc) => {
                        self.convs.run(gpu, dsc, x, &self.ds, None, 0)?;
                        &self.ds
                    }
                    None => x,
                };
                let t1 = &self.t1[&(blk.conv1.din, blk.conv1.o)];
                self.convs.run(gpu, &blk.conv1, x, t1, None, BIAS | RELU | PAD)?;
                self.convs.run(gpu, &blk.conv2, t1, &self.t2, None, BIAS | RELU)?;
                self.convs.run(gpu, &blk.conv3, &self.t2, y, Some(residual), BIAS | RELU | RES)?;
                cur = 1 - cur;
            }
            let out = &self.stream[cur];
            if let Some(h) = hook.as_mut() {
                let [d, hh, ww] = self.dims[l];
                h(&format!("layer{}", l + 1), out, [b, d, hh, ww, DIMS[l]])?;
            }
            self.dmax(gpu, l, out)?;
        }

        // decoder
        for (i, step) in self.dec.iter().enumerate() {
            let lvl = 2 - i;
            let (low, lc) = if i == 0 { (&self.p4, DIMS[3]) } else { (&self.dout[i - 1], DIMS[lvl + 1]) };
            let h = step.hw / 2;
            let ystride = DIMS[lvl] + DIMS[lvl + 1];
            let cat = &self.cat[lvl];
            ensure!(
                cat.bytes() >= padded_voxels(b, [0, step.hw, step.hw]) * ystride * 2
                    && low.bytes() >= b * h * h * lc * 2,
                "decoder upsample past a buffer"
            );
            let n = b * step.hw * step.hw * lc / 2;
            gpu.dispatch_1d(
                "v8_up2cat",
                n,
                &[cat.arg(), low.arg(), h.into(), h.into(), lc.into(), ystride.into(), DIMS[lvl].into(), n.into()],
            )?;
            self.convs.run(gpu, &step.conv, cat, &self.dout[i], None, BIAS | RELU)?;
        }

        let last = &self.dout[2];
        ensure!(last.bytes() >= b * PIX * DIMS[0] * 2, "head input past its buffer");
        gpu.dispatch_1d(
            "v8_head",
            b * PIX,
            &[
                prob.arg(),
                last.arg(),
                self.head_w.arg(),
                self.head_b.into(),
                dview,
                DIMS[0].into(),
                u32::from(sigmoid16).into(),
                (b * PIX).into(),
            ],
        )
    }

    /// The decoder-side activations of the last forward for the P0 gate: each level's concatenation (`[skip max,
    /// upsampled]`, padded), the deepest max and each decoder step's output. The encoder's buffers are reused across
    /// layers, so only these survive a forward
    pub fn taps(&self) -> Vec<Tap<'_>> {
        let b = self.batch;
        let mut v = Vec::new();
        for l in 0..3 {
            let [_, h, w] = self.dims[l];
            let shape = vec![b, h, w, DIMS[l] + DIMS[l + 1]];
            v.push(Tap { name: format!("cat{l}"), buf: &self.cat[l], shape, padded: true });
        }
        v.push(Tap { name: "max4".into(), buf: &self.p4, shape: vec![b, 8, 8, DIMS[3]], padded: false });
        for (i, s) in self.dec.iter().enumerate() {
            let shape = vec![b, s.hw, s.hw, DIMS[2 - i]];
            v.push(Tap { name: format!("dec{}", 2 - i), buf: &self.dout[i], shape, padded: false });
        }
        v
    }
}

#[cfg(test)]
mod tests {
    //! Every conv path against an f64 reference on FP16 data: odd sizes, partial row and column tiles, both tile
    //! widths, the padded output and the residual

    use super::*;

    /// Uniform in [-1, 1) from a fixed LCG, so failures reproduce
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }

        fn f16s(&mut self, n: usize, scale: f64) -> Vec<f16> {
            (0..n).map(|_| f16::from_f64(self.next() * scale)).collect()
        }
    }

    /// Kernel extent, padding and stride per axis (d, h, w) of a conv kind; a 2D conv has depth 1
    fn geometry(kind: Kind) -> ([usize; 3], [usize; 3], usize) {
        match kind {
            Kind::Gemm => ([1; 3], [0; 3], 1),
            Kind::Sub2 => ([1; 3], [0; 3], 2),
            Kind::Taps3(s) => ([3; 3], [1; 3], s),
            Kind::Taps2 => ([1, 3, 3], [0, 1, 1], 1),
        }
    }

    /// `x` [b, d, h, w, c] in the zero-padded layout with its tail (`d` 0: [b, h, w, c] in the 2D one)
    fn pad(x: &[f16], b: usize, [d, h, w]: [usize; 3], c: usize) -> Vec<f16> {
        let mut out = vec![f16::ZERO; padded_voxels(b, [d, h, w]) * c];
        let rows = b * d.max(1) * h * w;
        for row in 0..rows {
            let (xx, r) = (row % w, row / w);
            let (yy, r) = (r % h, r / h);
            // r is the sample in 2D, sample * d + z in 3D
            let plane = match r.checked_div(d) {
                Some(bi) => bi * (d + 2) + r % d + 1,
                None => r,
            };
            let p = (plane * (h + 2) + yy + 1) * (w + 2) + xx + 1;
            out[p * c..(p + 1) * c].copy_from_slice(&x[row * c..(row + 1) * c]);
        }
        out
    }

    /// Runs one conv of `kind` on random data through [`Convs::run`] and checks every output against f64
    fn check(kind: Kind, b: usize, din: [usize; 3], c: usize, o: usize, flags: u32) {
        check_on(kind, None, b, din, c, o, flags);
    }

    /// [`check`] on `tile` instead of the planned kernel
    fn check_on(kind: Kind, tile: Option<Tile>, b: usize, din: [usize; 3], c: usize, o: usize, flags: u32) {
        let gpu = Gpu::open(0).unwrap();
        let mut rng = Lcg(0x5eed ^ (c * 31 + o) as u64);
        let (k, p, s) = geometry(kind);
        let taps = k.iter().product::<usize>();
        let dd = din[0].max(1);
        let ([hd, hh, hw], dout_d) = ([dd, din[1], din[2]], if din[0] == 0 { 1 } else { dd / s });
        let (ho, wo) = (hh / s, hw / s);
        let x = rng.f16s(b * dd * hh * hw * c, 1.0);
        let w = rng.f16s(o * taps * c, 1.0 / ((taps * c) as f64).sqrt());
        let bias: Vec<f32> = (0..o).map(|_| rng.next() as f32 * 0.5).collect();
        let m = b * dout_d * ho * wo;
        let res = rng.f16s(m * o, 1.0);

        let mut conv = FConv::new(gpu.upload(&w).unwrap(), gpu.upload(&bias).unwrap(), (c, o), kind, din);
        conv.tile = tile.unwrap_or(conv.tile);
        let input = match kind {
            Kind::Gemm | Kind::Sub2 => gpu.upload(&x).unwrap(),
            _ => gpu.upload(&pad(&x, b, din, c)).unwrap(),
        };
        let dout = conv.dout();
        let out_elems = if flags & PAD != 0 { padded_voxels(b, dout) * o } else { m * o };
        // NaN wherever the conv must write (a padded output's interior; its border is zero, as Model allocates it),
        // so a missed element fails
        let nan_plain = vec![f16::NAN; m * o];
        let y =
            if flags & PAD != 0 { gpu.upload(&pad(&nan_plain, b, dout, o)) } else { gpu.upload(&nan_plain) }.unwrap();
        let rbuf = gpu.upload(&res).unwrap();
        let convs = Convs { batch: b, sub: gpu.alloc(m * c * 2).unwrap(), unused: gpu.alloc(4).unwrap() };
        let r = (flags & RES != 0).then_some(&rbuf);
        convs.run(&gpu, &conv, &input, &y, r, flags).unwrap();
        let mut got = vec![f16::ZERO; out_elems];
        gpu.download(&y, 0, &mut got).unwrap();
        if flags & PAD != 0 {
            let plain: Vec<f16> = (0..m * o).map(|_| f16::NAN).collect();
            let interior = pad(&plain, b, dout, o);
            for (i, (g, mask)) in got.iter().zip(&interior).enumerate() {
                if !mask.is_nan() {
                    assert_eq!(g.to_bits(), 0, "{kind:?}: padded border element {i} written");
                }
            }
            // gather the interior back into the plain layout
            let idx: Vec<usize> = interior.iter().enumerate().filter(|(_, v)| v.is_nan()).map(|(i, _)| i).collect();
            got = idx.iter().map(|&i| got[i]).collect();
        }

        let mut worst = 0f64;
        for bi in 0..b {
            for oz in 0..dout_d {
                for oy in 0..ho {
                    for ox in 0..wo {
                        let row = ((bi * dout_d + oz) * ho + oy) * wo + ox;
                        for oc in 0..o {
                            let mut acc = 0f64;
                            for kz in 0..k[0] {
                                let z = (s * oz + kz) as isize - p[0] as isize;
                                for ky in 0..k[1] {
                                    let yy = (s * oy + ky) as isize - p[1] as isize;
                                    for kx in 0..k[2] {
                                        let xx = (s * ox + kx) as isize - p[2] as isize;
                                        let inside = (0..hd as isize).contains(&z)
                                            && (0..hh as isize).contains(&yy)
                                            && (0..hw as isize).contains(&xx);
                                        if !inside {
                                            continue;
                                        }
                                        let t = (kz * k[1] + ky) * k[2] + kx;
                                        let xi = (((bi * hd + z as usize) * hh + yy as usize) * hw + xx as usize) * c;
                                        let wi = (oc * taps + t) * c;
                                        for ci in 0..c {
                                            acc += x[xi + ci].to_f64() * w[wi + ci].to_f64();
                                        }
                                    }
                                }
                            }
                            if flags & RES != 0 {
                                acc += res[row * o + oc].to_f64();
                            }
                            if flags & BIAS != 0 {
                                acc += bias[oc] as f64;
                            }
                            if flags & RELU != 0 {
                                acc = acc.max(0.0);
                            }
                            let g = got[row * o + oc].to_f64();
                            let err = (g - acc).abs();
                            // the FP16 rounding of the result plus float32 accumulation
                            let tol = acc.abs() * 2f64.powi(-10) + 2f64.powi(-12);
                            let on = conv.tile.name;
                            assert!(
                                err <= tol,
                                "{kind:?} {din:?} c {c} o {o} on {on}: output {row}x{oc} is {g}, f64 {acc}"
                            );
                            worst = worst.max(err);
                        }
                    }
                }
            }
        }
        eprintln!("{kind:?} {din:?} c {c} o {o} flags {flags} on {}: worst {worst:.2e}", conv.tile.name);
    }

    #[test]
    fn gemm_paths_match_f64() {
        check(Kind::Gemm, 2, [3, 5, 7], 64, 64, BIAS | RELU);
        check(Kind::Gemm, 2, [3, 5, 7], 64, 192, BIAS | RELU | RES);
        check(Kind::Gemm, 2, [3, 5, 7], 128, 128, BIAS | RELU | PAD);
        check(Kind::Sub2, 2, [4, 6, 10], 64, 128, 0);
    }

    #[test]
    fn tap_paths_match_f64() {
        check(Kind::Taps3(1), 2, [3, 5, 7], 64, 64, BIAS | RELU);
        check(Kind::Taps3(1), 2, [2, 20, 30], 64, 128, BIAS | RELU);
        check(Kind::Taps3(2), 2, [4, 6, 10], 64, 64, BIAS | RELU);
        check(Kind::Taps3(2), 1, [2, 4, 68], 64, 128, BIAS | RELU);
        check(Kind::Taps2, 2, [0, 5, 7], 128, 64, BIAS | RELU);
        check(Kind::Taps2, 2, [0, 16, 16], 192, 128, BIAS | RELU);
    }

    /// Every instantiated kernel, not only the planned ones, so a retune can pick any of them
    #[test]
    fn every_tile_matches_f64() {
        for &t in Kind::Gemm.tiles() {
            check_on(Kind::Gemm, Some(t), 2, [3, 5, 7], 64, 192, BIAS | RELU | RES);
        }
        for &t in Kind::Taps3(1).tiles() {
            check_on(Kind::Taps3(1), Some(t), 2, [2, 20, 30], 64, 192, BIAS | RELU);
        }
        for &t in Kind::Taps3(2).tiles() {
            check_on(Kind::Taps3(2), Some(t), 1, [2, 4, 68], 64, 192, BIAS | RELU);
        }
        for &t in Kind::Taps2.tiles() {
            check_on(Kind::Taps2, Some(t), 2, [0, 16, 16], 192, 192, BIAS | RELU);
        }
    }

    /// GPU milliseconds per v8-in tile of each distinct conv shape on every kernel that can run it, at batch 8,
    /// for picking [`Kind::pick`]: `libtest metal::v8in::model::tests::tile_times --ignored --nocapture` through the
    /// GPU queue
    #[test]
    #[ignore]
    fn tile_times() {
        let gpu = Gpu::open(0).unwrap();
        let b = 8;
        let mut rng = Lcg(7);
        // (kind, c, o, din, convs of this shape per tile), as Model::new lays the network out
        let mut shapes: Vec<(Kind, usize, usize, [usize; 3], usize)> = Vec::new();
        let mut add = |kind, c, o, din| match shapes.iter_mut().find(|s| (s.0, s.1, s.2, s.3) == (kind, c, o, din)) {
            Some(s) => s.4 += 1,
            None => shapes.push((kind, c, o, din, 1)),
        };
        let (mut cin, mut d) = (SC, [SD, PH, PH]);
        let mut dims = [[0usize; 3]; 4];
        for (l, (&planes, &nb)) in PLANES.iter().zip(&BLOCKS).enumerate() {
            let stride = if l == 0 { 1 } else { 2 };
            let dout = d.map(|v| v / stride);
            let c = planes * 4;
            for j in 0..nb {
                let (s0, bin, bdin) = if j == 0 { (stride, cin, d) } else { (1, c, dout) };
                if j == 0 {
                    add(if s0 == 2 { Kind::Sub2 } else { Kind::Gemm }, bin, c, bdin);
                }
                add(Kind::Gemm, bin, planes, bdin);
                add(Kind::Taps3(s0), planes, planes, bdin);
                add(Kind::Gemm, planes, c, dout);
            }
            dims[l] = dout;
            cin = c;
            d = dout;
        }
        for i in (1..4).rev() {
            let hw = dims[i - 1][1];
            add(Kind::Taps2, DIMS[i - 1] + DIMS[i], DIMS[i - 1], [0, hw, hw]);
        }

        let mut total = 0.0;
        for (kind, c, o, din, count) in shapes {
            let taps: usize = match kind {
                Kind::Gemm | Kind::Sub2 => 1,
                Kind::Taps3(_) => 27,
                Kind::Taps2 => 9,
            };
            let mut conv = FConv::new(
                gpu.upload(&rng.f16s(o * taps * c, 1.0 / ((taps * c) as f64).sqrt())).unwrap(),
                gpu.upload(&(0..o).map(|_| rng.next() as f32 * 0.1).collect::<Vec<_>>()).unwrap(),
                (c, o),
                kind,
                din,
            );
            let planned = conv.tile.name;
            let plain = b * din[0].max(1) * din[1] * din[2] * c;
            let x = match kind {
                Kind::Gemm | Kind::Sub2 => gpu.upload(&rng.f16s(plain, 1.0)).unwrap(),
                _ => gpu.alloc(padded_voxels(b, din) * c * 2).unwrap(),
            };
            let y = gpu.alloc(b * conv.out_per() * 2).unwrap();
            let convs = Convs {
                batch: b,
                sub: gpu.alloc(b * conv.out_per() / o * c * 2).unwrap(),
                unused: gpu.alloc(4).unwrap(),
            };
            let flops = 2.0 * (b * conv.out_per() * taps * c) as f64;
            let mut best = f64::INFINITY;
            for &t in kind.tiles() {
                conv.tile = t;
                let run = |n: usize| {
                    for _ in 0..n {
                        convs.run(&gpu, &conv, &x, &y, None, BIAS | RELU).unwrap();
                    }
                    // the newest buffer only: a flush fence would span the one synchronize kept from the last call
                    let f = gpu.flush_newest().unwrap().unwrap();
                    f.wait().unwrap();
                    let (a, z) = f.gpu_span();
                    (z - a) / n as f64
                };
                run(3);
                let s = (0..5).map(|_| run(10)).fold(f64::INFINITY, f64::min);
                best = best.min(s);
                let mark = if t.name == planned { "*" } else { " " };
                eprintln!(
                    "{mark} {kind:?} {c}->{o} @ {din:?} x{count} {:<26} {:8.3} ms/tile {:6.2} TF",
                    t.name,
                    s * 1e3 * count as f64 / b as f64,
                    flops / s / 1e12
                );
            }
            total += best * count as f64 / b as f64;
        }
        eprintln!("convs at the best kernel each: {:.2} ms/tile", total * 1e3);
    }

    fn download_f16(gpu: &Gpu, buf: &Buf, n: usize) -> Vec<f16> {
        let mut v = vec![f16::ZERO; n];
        gpu.download(buf, 0, &mut v).unwrap();
        v
    }

    fn same_bits(a: &[f16], b: &[f16], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: lengths");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(x.to_bits() == y.to_bits(), "{what}: element {i} is {x}, want {y}");
        }
    }

    /// Bad descriptors are refused before they reach the kernels, which have no bounds checks
    #[test]
    fn descriptors_are_checked() {
        let gpu = Gpu::open(0).unwrap();
        let tile = |y: i32, x: i32, first: i32, step: i32, slot: i32| [y, x, first, step, slot, 0, 0, 0];
        let descs = |tiles: &[[i32; 8]], slots: usize| Descs::new(&gpu, tiles.concat().as_slice(), 2, slots, (100, 90));
        assert!(descs(&[tile(0, 0, 0, 1, 0), tile(36, 26, 23, -1, -1)], 1).is_ok());
        assert!(descs(&[], 0).is_ok());
        for (bad, why) in [
            ([tile(0, 0, 1, 1, 0), tile(0, 0, 0, 1, 1)], "layers past the window"),
            ([tile(0, 0, 22, -1, 0), tile(0, 0, 0, 1, 1)], "layers below the window"),
            ([tile(0, 0, 0, 1, 2), tile(0, 0, 0, 1, 1)], "slot past the probabilities"),
            ([tile(0, 0, 0, 1, 1), tile(0, 0, 0, 1, 1)], "slot used twice"),
            ([tile(37, 0, 0, 1, 0), tile(0, 0, 0, 1, 1)], "tile past the volume"),
            ([tile(-1, 0, 0, 1, 0), tile(0, 0, 0, 1, 1)], "negative corner"),
        ] {
            assert!(descs(&bad, 2).is_err(), "{why} accepted");
        }
        assert!(Descs::new(&gpu, &tile(0, 0, 0, 1, 0), 2, 1, (100, 90)).is_err(), "half a batch accepted");
    }

    /// v8_dmax against a CPU max over depth (NaN propagating), into a padded concatenation at a channel offset and
    /// into a plain map; neither the border nor the other channels may change
    #[test]
    fn depth_max_matches_cpu() {
        let gpu = Gpu::open(0).unwrap();
        let (b, depth, h, w, c, ystride, off) = (2, 3, 5, 7, 6, 10, 4);
        let mut rng = Lcg(11);
        let mut x = rng.f16s(b * depth * h * w * c, 2.0);
        x[5 * c + 1] = f16::NAN;
        let mut want = vec![f16::ZERO; b * h * w * c];
        for (i, m) in want.iter_mut().enumerate() {
            let (bi, p, k) = (i / (h * w * c), (i / c) % (h * w), i % c);
            let vals = (0..depth).map(|z| x[((bi * depth + z) * h * w + p) * c + k]);
            // as the kernel: a NaN replaces the running max, and nothing compares greater than a NaN
            *m = vals.fold(f16::NEG_INFINITY, |a, v| if v > a || v.is_nan() { v } else { a });
        }
        let xb = gpu.upload(&x).unwrap();
        for padded in [true, false] {
            let rows = if padded { padded_voxels(b, [0, h, w]) } else { b * h * w };
            let sentinel = f16::from_f32(-7.0);
            let y = gpu.upload(&vec![sentinel; rows * ystride]).unwrap();
            let n = b * h * w * c / 2;
            let args = [y.arg(), xb.arg(), depth.into(), h.into(), w.into(), c.into(), ystride.into()];
            let tail = [off.into(), u32::from(padded).into(), n.into()];
            gpu.dispatch_1d("v8_dmax", n, &[&args[..], &tail[..]].concat()).unwrap();
            let got = download_f16(&gpu, &y, rows * ystride);
            let interior: Vec<usize> = if padded {
                let mark = pad(&vec![f16::NAN; b * h * w], b, [0, h, w], 1);
                (0..rows).filter(|&r| mark[r].is_nan()).collect()
            } else {
                (0..rows).collect()
            };
            let mut seen = vec![false; rows];
            for (pix, &r) in interior.iter().enumerate() {
                seen[r] = true;
                same_bits(&got[r * ystride + off..r * ystride + off + c], &want[pix * c..(pix + 1) * c], "depth max");
                assert!(got[r * ystride..r * ystride + off].iter().all(|v| *v == sentinel), "channels below off");
            }
            for r in (0..rows).filter(|&r| !seen[r]) {
                assert!(got[r * ystride..(r + 1) * ystride].iter().all(|v| *v == sentinel), "padded border {r}");
            }
        }
    }

    /// v8_up2cat against PyTorch's bilinear x2 (align_corners=False) in float32 on the CPU, into the padded
    /// concatenation at a channel offset
    #[test]
    fn upsample_matches_cpu() {
        let gpu = Gpu::open(0).unwrap();
        let (b, h, w, c, ystride, off) = (2, 4, 3, 4, 10, 6);
        let (h2, w2) = (2 * h, 2 * w);
        let low = Lcg(12).f16s(b * h * w * c, 3.0);
        let src = |d: usize, len: usize| {
            let r = (0.5f32 * (d as f32 + 0.5) - 0.5).max(0.0);
            let i0 = r as usize;
            let l1 = r - i0 as f32;
            (i0, usize::from(i0 < len - 1), 1.0 - l1, l1)
        };
        let mut want = vec![f16::ZERO; b * h2 * w2 * c];
        for (i, out) in want.iter_mut().enumerate() {
            let (bi, p, k) = (i / (h2 * w2 * c), (i / c) % (h2 * w2), i % c);
            let ((y0, ys, hl0, hl1), (x0, xs, wl0, wl1)) = (src(p / w2, h), src(p % w2, w));
            let at = |y: usize, x: usize| low[((bi * h + y) * w + x) * c + k].to_f32();
            let (a, e, f, g) = (at(y0, x0), at(y0, x0 + xs), at(y0 + ys, x0), at(y0 + ys, x0 + xs));
            *out = f16::from_f32(hl0 * (wl0 * a + wl1 * e) + hl1 * (wl0 * f + wl1 * g));
        }
        let rows = padded_voxels(b, [0, h2, w2]);
        let y = gpu.alloc(rows * ystride * 2).unwrap();
        let lb = gpu.upload(&low).unwrap();
        let n = b * h2 * w2 * c / 2;
        let args = [y.arg(), lb.arg(), h.into(), w.into(), c.into(), ystride.into(), off.into(), n.into()];
        gpu.dispatch_1d("v8_up2cat", n, &args).unwrap();
        let got = download_f16(&gpu, &y, rows * ystride);
        let mark = pad(&vec![f16::NAN; b * h2 * w2], b, [0, h2, w2], 1);
        let interior: Vec<usize> = (0..rows).filter(|&r| mark[r].is_nan()).collect();
        assert_eq!(interior.len(), b * h2 * w2);
        for (pix, &r) in interior.iter().enumerate() {
            same_bits(&got[r * ystride + off..r * ystride + off + c], &want[pix * c..(pix + 1) * c], "upsample");
            assert!(got[r * ystride..r * ystride + off].iter().all(|v| v.to_bits() == 0), "channels below off");
        }
        for r in (0..rows).filter(|r| !mark[*r].is_nan()) {
            assert!(got[r * ystride..(r + 1) * ystride].iter().all(|v| v.to_bits() == 0), "padded border {r}");
        }
    }

    /// v8_head in both sigmoid modes against the CPU, with a padding tile (slot -1) that must write nothing
    #[test]
    fn head_matches_cpu() {
        let gpu = Gpu::open(0).unwrap();
        let (tiles, c) = (3, 256);
        let mut rng = Lcg(13);
        let feat = rng.f16s(tiles * PIX * c, 1.0);
        let wts: Vec<f32> = rng.f16s(c, 0.1).iter().map(|v| v.to_f32()).collect();
        let bias = 0.3f32;
        // tiles 0 and 2 to slots 1 and 0; tile 1 is padding
        let desc: Vec<i32> = [[0, 0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 1, -1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]].concat();
        let (fb, wb, db) = (gpu.upload(&feat).unwrap(), gpu.upload(&wts).unwrap(), gpu.upload(&desc).unwrap());
        let r16 = |v: f32| f16::from_f32(v).to_f32();
        for sigmoid16 in [true, false] {
            let prob = gpu.alloc_fill(3 * PIX * 4, 0xFF).unwrap();
            let n = tiles * PIX;
            let args = [prob.arg(), fb.arg(), wb.arg(), bias.into(), db.arg(), c.into(), u32::from(sigmoid16).into()];
            gpu.dispatch_1d("v8_head", n, &[&args[..], &[n.into()]].concat()).unwrap();
            let mut got = vec![0f32; 3 * PIX];
            gpu.download(&prob, 0, &mut got).unwrap();
            for (tile, slot) in [(0, 1), (2, 0)] {
                for p in 0..PIX {
                    let f = &feat[(tile * PIX + p) * c..(tile * PIX + p + 1) * c];
                    let acc = f.iter().zip(&wts).fold(0f32, |a, (x, w)| a + x.to_f32() * w);
                    let logit = r16(r16(acc) + r16(bias));
                    let s = 1.0 / (1.0 + (-logit).exp());
                    let want = if sigmoid16 { r16(s) } else { s };
                    let g = got[slot * PIX + p];
                    // the GPU's exp may differ from the CPU's by an ulp, which FP16 rounding usually hides
                    let tol = if sigmoid16 { 2f32.powi(-11) } else { 4.0 * f32::EPSILON };
                    assert!((g - want).abs() <= tol, "sigmoid16 {sigmoid16} tile {tile} pixel {p}: {g}, want {want}");
                }
            }
            assert!(got[2 * PIX..].iter().all(|v| v.is_nan()), "slot 2 written by the padding tile");
        }
    }

    /// v8_blend in both modes against the same float32 sums on the CPU, with a skipped tile and uncovered pixels
    #[test]
    fn blend_matches_cpu() {
        let gpu = Gpu::open(0).unwrap();
        let (ny, nx, stride) = (3, 2, 21);
        let (height, width) = ((ny - 1) * stride + P + 5, (nx - 1) * stride + P + 3);
        let grid: Vec<i32> = vec![0, 1, -1, 2, 3, 4];
        let mut rng = Lcg(14);
        let prob: Vec<f32> = (0..5 * PIX).map(|_| (rng.next() as f32 + 1.0) * 0.5).collect();
        let weight: Vec<f32> = (0..PIX).map(|_| (rng.next() as f32 + 1.0) * 0.5 + 1e-3).collect();
        let (pb, gb, wb) = (gpu.upload(&prob).unwrap(), gpu.upload(&grid).unwrap(), gpu.upload(&weight).unwrap());
        for mode in [0u32, 1] {
            let out = gpu.alloc(height * width * 4).unwrap();
            let n = height * width;
            let args = [out.arg(), pb.arg(), gb.arg(), wb.arg(), ny.into(), nx.into(), stride.into(), width.into()];
            gpu.dispatch_1d("v8_blend", n, &[&args[..], &[mode.into(), n.into()]].concat()).unwrap();
            let mut got = vec![0f32; n];
            gpu.download(&out, 0, &mut got).unwrap();
            for (i, g) in got.iter().enumerate() {
                let (y, x) = (i / width, i % width);
                let (mut num, mut den) = (0f32, 0f32);
                for ky in 0..ny {
                    for kx in 0..nx {
                        let (ty, tx) = (ky * stride, kx * stride);
                        let slot = grid[ky * nx + kx];
                        if slot < 0 || y < ty || x < tx || y >= ty + P || x >= tx + P {
                            continue;
                        }
                        let at = (y - ty) * P + x - tx;
                        num += prob[slot as usize * PIX + at] * weight[at];
                        den += if mode == 0 { 1.0 } else { weight[at] };
                    }
                }
                let want = match mode {
                    0 if den == 0.0 => 0.0,
                    0 => (num / den).clamp(0.0, 1.0),
                    _ if den > 1e-9 => num / den,
                    _ => f32::NAN,
                };
                assert!(g.to_bits() == want.to_bits(), "mode {mode} pixel ({y}, {x}): {g}, want {want}");
            }
        }
    }

    /// GPU milliseconds per v8-in tile of the data-movement kernels at the network's shapes, at batch 8, with the
    /// bytes each moves: `libtest metal::v8in::model::tests::aux_times --ignored --nocapture` through the GPU queue
    #[test]
    #[ignore]
    fn aux_times() {
        let gpu = Gpu::open(0).unwrap();
        let b = 8;
        let time = |run: &dyn Fn()| {
            let span = |n: usize| {
                for _ in 0..n {
                    run();
                }
                let f = gpu.flush_newest().unwrap().unwrap();
                f.wait().unwrap();
                let (a, z) = f.gpu_span();
                (z - a) / n as f64
            };
            span(2);
            (0..5).map(|_| span(8)).fold(f64::INFINITY, f64::min)
        };
        let report = |what: String, s: f64, bytes: usize| {
            eprintln!("{what:<44} {:8.3} ms/tile {:7.1} GB/s", s * 1e3 / b as f64, bytes as f64 / s / 1e9);
        };
        // depth max of each layer's output into its concatenation
        for l in 0..4 {
            let ([d, h, w], c) = ([[96, 64, 64], [48, 32, 32], [24, 16, 16], [12, 8, 8]][l], DIMS[l]);
            let ystride = if l < 3 { c + DIMS[l + 1] } else { c };
            let x = gpu.alloc(b * d * h * w * c * 2).unwrap();
            let y = gpu.alloc(padded_voxels(b, [0, h, w]) * ystride * 2).unwrap();
            let n = b * h * w * c / 2;
            let pad = u32::from(l < 3);
            let run = || {
                let args = [y.arg(), x.arg(), d.into(), h.into(), w.into(), c.into(), ystride.into()];
                gpu.dispatch_1d("v8_dmax", n, &[&args[..], &[0u32.into(), pad.into(), n.into()]].concat()).unwrap();
            };
            report(format!("v8_dmax layer{} {d}x{h}x{w}x{c}", l + 1), time(&run), b * (d + 1) * h * w * c * 2);
        }
        // the decoder's upsample of its deeper map into each concatenation
        for lvl in (0..3).rev() {
            let (hw, lc, ystride) = (64 >> lvl, DIMS[lvl + 1], DIMS[lvl] + DIMS[lvl + 1]);
            let h = hw / 2;
            let low = gpu.alloc(b * h * h * lc * 2).unwrap();
            let y = gpu.alloc(padded_voxels(b, [0, hw, hw]) * ystride * 2).unwrap();
            let n = b * hw * hw * lc / 2;
            let run = || {
                let args = [y.arg(), low.arg(), h.into(), h.into(), lc.into(), ystride.into(), DIMS[lvl].into()];
                gpu.dispatch_1d("v8_up2cat", n, &[&args[..], &[n.into()]].concat()).unwrap();
            };
            report(format!("v8_up2cat cat{lvl} {h}x{h}x{lc} -> {hw}x{hw}"), time(&run), b * 5 * h * h * lc * 2);
        }
        // the stride-2 downsamples' subsampled inputs
        for (d, c) in [([96, 64, 64], 256), ([48, 32, 32], 512), ([24, 16, 16], 1024)] {
            let x = gpu.alloc(b * d.iter().product::<usize>() * c * 2).unwrap();
            let out = b * d.iter().product::<usize>() / 8 * c;
            let y = gpu.alloc(out * 2).unwrap();
            let n = out / 2;
            let run = || {
                let args = [y.arg(), x.arg(), d[0].into(), d[1].into(), d[2].into(), c.into(), n.into()];
                gpu.dispatch_1d("v8_sub2", n, &args).unwrap();
            };
            report(format!("v8_sub2 {d:?}x{c}"), time(&run), out * 2 * 2);
        }
    }
}
