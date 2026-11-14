//! villa's `Local3DStem2DUNet` for `hybrid_3d2d` on Metal, planned once at a fixed batch: the same layers, weights
//! and FP16 autocast roundings as the CUDA model (`src/model.rs`), with the convolutions on Metal 4 tensor ops
//!
//! The reader precision is the [`Gpu`]'s ([`Gpu::precision`]): at fp16 every activation, conv weight and logit is
//! stored as FP16 with autocast's roundings; at fp32 the same kernels store FP32 and round nowhere, as villa runs on
//! MPS. Accumulation, InstanceNorm statistics and the norm's affine parameters are FP32 in both
//!
//! - 3x3 stride-1 convs: the input in a zero-padded layout, then one accumulating matmul2d per row of taps, the
//!   three column taps folded into K (`conv3x3`). The padding is fused into the producer where the conv is the only
//!   reader (`in_apply_pad` after a block's first IN, `cat_up` in the decoder) and a `pad2d` pass elsewhere
//! - 3x3 stride-2 convs: `im2col_s2`, then one GEMM against the `[o, 9, c]` weights (`gemm_nt`)
//! - 1x1 skip convs and the decoder's transposed convs: one GEMM each
//! - the stem: a direct 1 -> 16 conv3d (`conv3d_c1`), then its IN writes the padded input of the 16 -> 16 conv3d,
//!   which runs as 9 matmul2d over folded taps (`conv3d_c16`)
//!
//! Every conv feeds an InstanceNorm, which removes a per-channel bias exactly, so the convs run without theirs; their
//! bias tensors are still checked and consumed

use color_eyre::eyre::{Result, WrapErr, ensure};

use super::gpu::{Arg, Buf, Gpu};
use crate::sched::{DEPTH, Edge, percentile_params};
use crate::weights::{Checkpoint, conv_nhwc_f32};

const STEM: usize = 16;
const FEATURES: [usize; 6] = [32, 64, 128, 256, 320, 320];
const BLOCKS: [usize; 6] = [1, 3, 4, 6, 6, 6];
const EPS: f32 = 1e-5;
/// Threadgroup rows of the stem's conv3d 16 -> 16
const BM: usize = 64;
/// Voxels per threadgroup of the stem's conv3d 1 -> 16, which must divide a sample at both edges
const STEM_C1_GROUP: usize = 256;
const _: () = assert!((DEPTH * Edge::P128.pixels()).is_multiple_of(STEM_C1_GROUP));
const _: () = assert!((DEPTH * Edge::P384.pixels()).is_multiple_of(STEM_C1_GROUP));

/// Rows of one sample in the stem's padded layout `[19, px + 2, px + 2]`
fn stem_padded(px: usize) -> usize {
    (DEPTH + 2) * (px + 2) * (px + 2)
}

/// Row tiles per sample of the stem's conv3d 16 -> 16: its outputs are the first 17 planes of the padded layout
fn stem_c16_tiles(px: usize) -> usize {
    (DEPTH * (px + 2) * (px + 2)).div_ceil(BM)
}

/// InstanceNorm affine parameters in float32
struct Norm {
    w: Buf,
    b: Buf,
}

impl Norm {
    fn load(gpu: &Gpu, ck: &Checkpoint, key: &str, c: usize) -> Result<Self> {
        Ok(Self {
            w: gpu.upload(&ck.take(&format!("{key}.weight"), &[c])?)?,
            b: gpu.upload(&ck.take(&format!("{key}.bias"), &[c])?)?,
        })
    }
}

/// How a 2D conv runs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// 3x3, stride 1, pad 1
    Taps,
    /// 3x3, stride 2, pad 1
    Strided,
    /// 1x1
    Pointwise,
}

/// A 2D conv: its weight `[o, k*k, c]` in the reader's storage type and its shape at the model's batch
struct Conv {
    w: Buf,
    kind: Kind,
    c: usize,
    o: usize,
    /// Input edge
    h: usize,
}

impl Conv {
    fn load(
        gpu: &Gpu,
        ck: &Checkpoint,
        key: &str,
        (c, o, h): (usize, usize, usize),
        kind: Kind,
        bias: bool,
    ) -> Result<Self> {
        let k = if kind == Kind::Pointwise { 1 } else { 3 };
        let w = ck.take(&format!("{key}.weight"), &[o, c, k, k])?;
        if bias {
            ck.take(&format!("{key}.bias"), &[o])?;
        }
        Self::from_torch(gpu, &w, (c, o, h), kind).wrap_err_with(|| key.to_string())
    }

    /// The conv of PyTorch weight `w` (`[o, c, k, k]`)
    fn from_torch(gpu: &Gpu, w: &[f32], (c, o, h): (usize, usize, usize), kind: Kind) -> Result<Self> {
        let k = if kind == Kind::Pointwise { 1 } else { 3 };
        ensure!(w.len() == o * c * k * k, "conv weight of {} values for {c} -> {o}", w.len());
        ensure!(c % 2 == 0 && o % 16 == 0, "{c} -> {o} channels: need even inputs and outputs in 16s");
        ensure!(kind != Kind::Strided || h % 2 == 0, "stride-2 conv on an odd edge {h}");
        Ok(Self { w: gpu.upload_real(&conv_nhwc_f32(w, o, c, k * k))?, kind, c, o, h })
    }

    /// Output edge
    fn out_h(&self) -> usize {
        if self.kind == Kind::Strided { self.h / 2 } else { self.h }
    }

    /// Output rows of one sample the conv's kernel tiles: the padded lines of a tap conv, the pixels of a GEMM
    fn span(&self) -> usize {
        match self.kind {
            Kind::Taps => self.h * (self.h + 2),
            Kind::Strided | Kind::Pointwise => self.out_h() * self.out_h(),
        }
    }

    /// IN partials per sample (row tiles), which `Model::norm` needs to finalise them
    fn tiles(&self) -> usize {
        let span = self.span();
        span.div_ceil(row_tile(span))
    }

    /// Scratch elements (in the storage type) this conv needs at batch `b`: the padded input or the im2col matrix
    fn scratch(&self, b: usize) -> usize {
        match self.kind {
            Kind::Taps => padded_rows(b, self.h) * self.c,
            Kind::Strided => b * self.out_h() * self.out_h() * 9 * self.c,
            Kind::Pointwise => 0,
        }
    }

    /// Runs the conv of `x` into `y` and its IN partials into `part`; returns the partials per sample
    fn run(&self, gpu: &Gpu, b: usize, scratch: &Buf, x: &Buf, y: &Buf, part: &Buf) -> Result<usize> {
        let (c, o, h) = (self.c, self.o, self.h);
        let pair = 2 * gpu.precision().bytes();
        match self.kind {
            Kind::Taps => {
                let n = padded_rows(b, h) * c / 2;
                ensure!(n * pair <= scratch.bytes(), "conv scratch too small");
                gpu.dispatch_1d(
                    "pad2d",
                    n,
                    &[scratch.arg(), x.arg(), b.into(), h.into(), h.into(), c.into(), n.into()],
                )?;
                self.run_padded(gpu, b, scratch, y, part)
            }
            Kind::Strided => {
                let ho = h / 2;
                let n = b * ho * ho * 9 * c / 2;
                ensure!(n * pair <= scratch.bytes(), "conv scratch too small");
                gpu.dispatch_1d("im2col_s2", n, &[scratch.arg(), x.arg(), h.into(), h.into(), c.into(), n.into()])?;
                gemm_in(gpu, scratch, &self.w, y, (b * ho * ho, o, 9 * c), ho * ho, part)
            }
            Kind::Pointwise => gemm_in(gpu, x, &self.w, y, (b * h * h, o, c), h * h, part),
        }
    }

    /// A tap conv on an input already in the padded layout of `padded_rows(b, h)` rows, written by a producer that
    /// fuses the padding (`in_apply_pad`, `cat_up`); returns the IN partials per sample
    fn run_padded(&self, gpu: &Gpu, b: usize, xp: &Buf, y: &Buf, part: &Buf) -> Result<usize> {
        let (c, o, h) = (self.c, self.o, self.h);
        let e = gpu.precision().bytes();
        ensure!(self.kind == Kind::Taps, "run_padded on a {:?} conv", self.kind);
        ensure!(padded_rows(b, h) * c * e <= xp.bytes(), "padded conv input too small");
        ensure!(b * h * h * o * e <= y.bytes(), "conv output too small");
        let tiles = self.tiles();
        ensure!(b * tiles * o * 16 <= part.bytes(), "IN partials too small");
        let (name, bn) = tap_kernel(o, row_tile(self.span()));
        gpu.dispatch(
            name,
            [o.div_ceil(bn), b * tiles, 1],
            [128, 1, 1],
            &[xp.arg(), self.w.arg(), y.arg(), b.into(), h.into(), h.into(), c.into(), o.into(), part.arg()],
        )?;
        Ok(tiles)
    }
}

/// Rows of the padded layout of `b` samples of edge `h`, with the tail the last sample's tap windows read
fn padded_rows(b: usize, h: usize) -> usize {
    let wp = h + 2;
    b * wp * wp + 2 * wp + 2
}

/// The tile width for `n` output channels: the widest of 64, 32 and 16 that divides it
fn col_tile(n: usize) -> usize {
    [64, 32, 16].into_iter().find(|&bn| n.is_multiple_of(bn)).unwrap_or(16)
}

/// The tile height for a run of `span` rows per sample: 32 where it rounds fewer rows up than 64 (the small layers),
/// else 64
fn row_tile(span: usize) -> usize {
    if span.div_ceil(32) * 32 < span.div_ceil(64) * 64 { 32 } else { 64 }
}

/// The tap kernel for `o` output channels at tile height `bm`, and its tile width
fn tap_kernel(o: usize, bm: usize) -> (&'static str, usize) {
    let bn = col_tile(o);
    let name = match (bm, bn) {
        (64, 64) => "conv3x3_64x64",
        (64, 32) => "conv3x3_64x32",
        (64, _) => "conv3x3_64x16",
        (_, 64) => "conv3x3_32x64",
        (_, 32) => "conv3x3_32x32",
        _ => "conv3x3_32x16",
    };
    (name, bn)
}

/// `y[m, n] = x[m, k] . w[n, k]^T` in the reader's storage type
fn gemm(gpu: &Gpu, x: &Buf, w: &Buf, y: &Buf, (m, n, k): (usize, usize, usize)) -> Result<()> {
    let e = gpu.precision().bytes();
    ensure!(
        x.bytes() >= m * k * e && w.bytes() >= n * k * e && y.bytes() >= m * n * e,
        "gemm {m}x{n}x{k}: buffers too small"
    );
    let (name, bn) = if n % 64 == 0 {
        ("gemm_nt_64x64", 64)
    } else if n % 32 == 0 {
        ("gemm_nt_64x32", 32)
    } else {
        ("gemm_nt_64x16", 16)
    };
    gpu.dispatch(
        name,
        [n.div_ceil(bn), m.div_ceil(64), 1],
        [128, 1, 1],
        &[x.arg(), w.arg(), y.arg(), m.into(), n.into(), k.into(), k.into()],
    )
}

/// `gemm` for a conv whose output feeds an InstanceNorm: `m` is whole samples of `rps` rows, and the IN partials go to
/// `part`; returns the partials per sample
fn gemm_in(
    gpu: &Gpu,
    x: &Buf,
    w: &Buf,
    y: &Buf,
    (m, n, k): (usize, usize, usize),
    rps: usize,
    part: &Buf,
) -> Result<usize> {
    let e = gpu.precision().bytes();
    ensure!(
        x.bytes() >= m * k * e && w.bytes() >= n * k * e && y.bytes() >= m * n * e,
        "gemm {m}x{n}x{k}: buffers too small"
    );
    ensure!(rps > 0 && m.is_multiple_of(rps), "gemm of {m} rows is not whole samples of {rps}");
    let bm = row_tile(rps);
    let tiles = rps.div_ceil(bm);
    ensure!(m / rps * tiles * n * 16 <= part.bytes(), "IN partials too small");
    let bn = col_tile(n);
    let name = match (bm, bn) {
        (64, 64) => "gemm_nt_in_64x64",
        (64, 32) => "gemm_nt_in_64x32",
        (64, _) => "gemm_nt_in_64x16",
        (_, 64) => "gemm_nt_in_32x64",
        (_, 32) => "gemm_nt_in_32x32",
        _ => "gemm_nt_in_32x16",
    };
    gpu.dispatch(
        name,
        [n.div_ceil(bn), m / rps * tiles, 1],
        [128, 1, 1],
        &[x.arg(), w.arg(), y.arg(), m.into(), n.into(), k.into(), k.into(), rps.into(), part.arg()],
    )?;
    Ok(tiles)
}

/// One BasicBlockD
struct Block {
    conv1: Conv,
    n1: Norm,
    conv2: Conv,
    n2: Norm,
    pooled: Option<Buf>,
    skip: Option<(Conv, Norm, Buf)>,
    t1: Buf,
    out: Buf,
    cin: usize,
    c: usize,
    h: usize,
}

/// One decoder stage
struct Up {
    /// The transposed conv as the GEMM operand `[4 * up, cin]`, row `(dy * 2 + dx) * up + co`
    w: Buf,
    bias: Buf,
    g: Buf,
    conv: Conv,
    n: Norm,
    out: Buf,
    h: usize,
    cin: usize,
    up: usize,
    skip_c: usize,
}

/// A PyTorch `ConvTranspose2d(k=2, s=2)` weight `[cin, cout, 2, 2]` as the GEMM operand `[4 * cout, cin]`
fn transp_rows(w: &[f32], cin: usize, cout: usize) -> Vec<f32> {
    let mut out = vec![0.0; w.len()];
    for ci in 0..cin {
        for co in 0..cout {
            for d in 0..4 {
                out[(d * cout + co) * cin + ci] = w[(ci * cout + co) * 4 + d];
            }
        }
    }
    out
}

/// Tile gather and villa's robust normalisation for one batch, as on CUDA (`src/model.rs` `TilePrep`)
pub struct TilePrep {
    batch: usize,
    edge: Edge,
    hist: Buf,
    tstats: Buf,
    pub x0: Buf,
    pct: [(u32, f32, f32); 2],
}

impl TilePrep {
    pub fn new(gpu: &Gpu, batch: usize, edge: Edge) -> Result<Self> {
        let vox = DEPTH * edge.pixels();
        Ok(Self {
            batch,
            edge,
            hist: gpu.alloc(batch * 256 * 4)?,
            tstats: gpu.alloc(batch * 16)?,
            x0: gpu.alloc_real_nan(batch * vox)?,
            pct: [percentile_params(vox, 1.0), percentile_params(vox, 99.0)],
        })
    }

    /// Queues the batch `desc` describes (`[batch, DESC]`) from the resident volume
    pub fn run(&self, gpu: &Gpu, vol: (&Buf, usize, usize), desc: Arg) -> Result<()> {
        let b = self.batch;
        let (vbuf, height, width) = vol;
        let px = self.edge.px();
        let vox = DEPTH * px * px;
        gpu.dispatch_1d("zero_u32", b * 256, &[self.hist.arg(), (b * 256).into()])?;
        gpu.dispatch(
            "tile_hist",
            [DEPTH, b, 1],
            [256, 1, 1],
            &[self.hist.arg(), vbuf.arg(), desc, height.into(), width.into(), b.into(), px.into()],
        )?;
        let [(li, lt, lo), (hi, ht, ho)] = self.pct;
        gpu.dispatch_1d(
            "tile_stats",
            b,
            &[
                self.tstats.arg(),
                self.hist.arg(),
                vox.into(),
                li.into(),
                lt.into(),
                lo.into(),
                hi.into(),
                ht.into(),
                ho.into(),
                b.into(),
            ],
        )?;
        gpu.dispatch(
            "tile_std",
            [b, 1, 1],
            [256, 1, 1],
            &[self.tstats.arg(), vbuf.arg(), desc, height.into(), width.into(), b.into(), px.into()],
        )?;
        let n = b * vox / 2;
        gpu.dispatch_1d(
            "tile_norm",
            n,
            &[self.x0.arg(), vbuf.arg(), desc, self.tstats.arg(), height.into(), width.into(), n.into(), px.into()],
        )
    }
}

/// The planned model and its activations at batch `batch`
pub struct Model {
    pub batch: usize,
    /// The tile edge the reader's sidecar fixes: 128 (reader-v2) or 384 (Reader V3)
    pub edge: Edge,
    stem1_w: Buf,
    stem1_n: Norm,
    stem2_w: Buf,
    stem2_n: Norm,
    att_w: Buf,
    att_b: f32,
    enc: Conv,
    enc_n: Norm,
    blocks: Vec<Vec<Block>>,
    ups: Vec<Up>,
    head_w: Buf,
    head_b: f32,
    tiles: TilePrep,
    s1: Buf,
    s2: Buf,
    pooled: Buf,
    e0: Buf,
    /// The padded input of a tap conv, the im2col matrix of a strided one, or the stem's padded `s1`
    scratch: Buf,
    partial: Buf,
    stats: Buf,
}

/// A named activation of the last forward: buffer, channels-last shape
pub struct Tap<'a> {
    pub name: String,
    pub buf: &'a Buf,
    pub shape: Vec<usize>,
}

impl Model {
    /// Uploads the weights and allocates every activation at `batch`, in the storage type of `gpu`'s reader precision;
    /// every checkpoint tensor must be consumed
    pub fn new(gpu: &Gpu, ck: &Checkpoint, batch: usize, edge: Edge) -> Result<Self> {
        let px = edge.px();
        let p = gpu.precision();
        // activations start NaN, so a layer that is never written fails the finiteness checks
        let act = |n: usize| gpu.alloc_real_nan(n);
        check_batch(batch, px)?;
        let vox = DEPTH * px * px;

        // stem
        let w = ck.take("depth_fusion.features.0.weight", &[STEM, 1, 3, 3, 3])?;
        let stem1_w = gpu.upload_real(&conv_nhwc_f32(&w, STEM, 1, 27))?;
        let stem1_n = Norm::load(gpu, ck, "depth_fusion.features.1", STEM)?;
        let w = ck.take("depth_fusion.features.3.weight", &[STEM, STEM, 3, 3, 3])?;
        let stem2_w = gpu.upload_real(&conv_nhwc_f32(&w, STEM, STEM, 27))?;
        let stem2_n = Norm::load(gpu, ck, "depth_fusion.features.4", STEM)?;
        let att_w = gpu.upload(&p.round(&ck.take("depth_fusion.attention_logits.weight", &[1, STEM, 1, 1, 1])?))?;
        let att_b = p.round(&[ck.scalar("depth_fusion.attention_logits.bias")?])[0];

        // encoder
        let enc_key = "network.shared_encoder.stem.convs.0";
        let enc = Conv::load(gpu, ck, &format!("{enc_key}.conv"), (32, 32, px), Kind::Taps, true)?;
        let enc_n = Norm::load(gpu, ck, &format!("{enc_key}.norm"), 32)?;
        let mut blocks = Vec::new();
        let (mut cin, mut h) = (32usize, px);
        for (s, (&c, &nb)) in FEATURES.iter().zip(&BLOCKS).enumerate() {
            let mut stage = Vec::new();
            for j in 0..nb {
                let key = format!("network.shared_encoder.stages.{s}.blocks.{j}");
                let strided = j == 0 && s > 0;
                let hout = if strided { h / 2 } else { h };
                let kind = if strided { Kind::Strided } else { Kind::Taps };
                let conv1 = Conv::load(gpu, ck, &format!("{key}.conv1.conv"), (cin, c, h), kind, true)?;
                let n1 = Norm::load(gpu, ck, &format!("{key}.conv1.norm"), c)?;
                let conv2 = Conv::load(gpu, ck, &format!("{key}.conv2.conv"), (c, c, hout), Kind::Taps, true)?;
                let n2 = Norm::load(gpu, ck, &format!("{key}.conv2.norm"), c)?;
                let pooled = if strided { Some(act(batch * hout * hout * cin)?) } else { None };
                let skip = if cin != c {
                    ensure!(strided, "stage {s} block {j}: a width change without a stride");
                    let conv =
                        Conv::load(gpu, ck, &format!("{key}.skip.1.conv"), (cin, c, hout), Kind::Pointwise, false)?;
                    let norm = Norm::load(gpu, ck, &format!("{key}.skip.1.norm"), c)?;
                    Some((conv, norm, act(batch * hout * hout * c)?))
                } else {
                    None
                };
                stage.push(Block {
                    conv1,
                    n1,
                    conv2,
                    n2,
                    pooled,
                    skip,
                    t1: act(batch * hout * hout * c)?,
                    out: act(batch * hout * hout * c)?,
                    cin,
                    c,
                    h: hout,
                });
                cin = c;
                h = hout;
            }
            blocks.push(stage);
        }

        // decoder: stage i upsamples from encoder stage 5 - i to the size of stage 4 - i
        let mut ups = Vec::new();
        let (mut low_c, mut low_h) = (FEATURES[5], h);
        for i in 0..5 {
            let skip_c = FEATURES[4 - i];
            let up = skip_c;
            let hh = 2 * low_h;
            let tw = ck.take(&format!("network.shared_decoder.transpconvs.{i}.weight"), &[low_c, up, 2, 2])?;
            let tb = p.round(&ck.take(&format!("network.shared_decoder.transpconvs.{i}.bias"), &[up])?);
            let key = format!("network.shared_decoder.stages.{i}.convs.0");
            let conv = Conv::load(gpu, ck, &format!("{key}.conv"), (up + skip_c, up, hh), Kind::Taps, true)?;
            let norm = Norm::load(gpu, ck, &format!("{key}.norm"), up)?;
            ups.push(Up {
                w: gpu.upload_real(&transp_rows(&tw, low_c, up))?,
                bias: gpu.upload(&tb)?,
                g: act(batch * low_h * low_h * 4 * up)?,
                conv,
                n: norm,
                out: act(batch * hh * hh * up)?,
                h: low_h,
                cin: low_c,
                up,
                skip_c,
            });
            low_c = up;
            low_h = hh;
        }
        ensure!(low_h == px && low_c == 32, "decoder ends at {low_h}x{low_h}x{low_c}");
        let head_w = gpu.upload(&p.round(&ck.take("network.task_heads.ink.weight", &[1, 32, 1, 1])?))?;
        let head_b = p.round(&[ck.scalar("network.task_heads.ink.bias")?])[0];

        // scratch sized for the largest user
        let mut scratch = (batch * stem_padded(px) + stem_tail(px)) * STEM;
        let mut convs: Vec<&Conv> = vec![&enc];
        for b in blocks.iter().flatten() {
            convs.extend([&b.conv1, &b.conv2]);
        }
        convs.extend(ups.iter().map(|u| &u.conv));
        scratch = convs.iter().map(|c| c.scratch(batch)).fold(scratch, usize::max);
        // IN partials per sample: the stem convs' and every 2D conv's (skip convs included)
        let skips = blocks.iter().flatten().filter_map(|b| b.skip.as_ref().map(|(c, ..)| c));
        let stem = [stem_c16_tiles(px) * STEM, vox / STEM_C1_GROUP * STEM];
        let partial_len = convs.iter().copied().chain(skips).map(|c| c.tiles() * c.o).chain(stem).max().unwrap();
        let max_cols = convs.iter().map(|c| c.o).chain([STEM]).max().unwrap();

        Ok(Self {
            batch,
            edge,
            stem1_w,
            stem1_n,
            stem2_w,
            stem2_n,
            att_w,
            att_b,
            enc,
            enc_n,
            blocks,
            ups,
            head_w,
            head_b,
            tiles: TilePrep::new(gpu, batch, edge)?,
            s1: act(batch * vox * STEM)?,
            s2: act(batch * vox * STEM)?,
            pooled: act(batch * px * px * 32)?,
            e0: act(batch * px * px * 32)?,
            scratch: gpu.alloc(scratch * p.bytes())?,
            partial: gpu.alloc(batch * partial_len * 16)?,
            stats: gpu.alloc(batch * max_cols * 8)?,
        })
    }

    /// InstanceNorm of `x` (`[B, rows, cols]`) in place, then the optional residual add and LeakyReLU, with the
    /// statistics `norm_stats` left in `stats`
    fn norm(
        &self,
        gpu: &Gpu,
        x: &Buf,
        (rows, cols): (usize, usize),
        n: &Norm,
        residual: Option<&Buf>,
        leaky: bool,
    ) -> Result<()> {
        let b = self.batch;
        let flags = u32::from(leaky) | if residual.is_some() { 2 } else { 0 };
        ensure!(cols.is_multiple_of(8), "in_apply over {cols} channels: needs a multiple of 8");
        let octets = b * rows * cols / 8;
        gpu.dispatch_1d(
            "in_apply",
            octets,
            &[
                x.arg(),
                residual.unwrap_or(x).arg(),
                self.stats.arg(),
                n.w.arg(),
                n.b.arg(),
                rows.into(),
                cols.into(),
                flags.into(),
                octets.into(),
            ],
        )
    }

    /// InstanceNorm, the optional residual add and LeakyReLU of `x` (`[B, depth, h, h, cols]`) written into `pad.xp`
    /// in the padded layout of the tap conv that reads it (see `Pad`), with the statistics `norm_stats` left in `stats`
    fn norm_pad(
        &self,
        gpu: &Gpu,
        x: &Buf,
        (depth, h, cols): (usize, usize, usize),
        n: &Norm,
        residual: Option<&Buf>,
        pad: Pad,
    ) -> Result<()> {
        let b = self.batch;
        let Pad { xp, dpad, pairs, keep } = pad;
        ensure!(pairs * 2 * gpu.precision().bytes() <= xp.bytes(), "padded norm output too small");
        ensure!(
            pairs * 2 >= b * (depth + 2 * dpad) * (h + 2) * (h + 2) * cols,
            "padded norm output short of the batch"
        );
        ensure!(cols.is_multiple_of(8) && pairs.is_multiple_of(4), "in_apply_pad over {cols} channels: needs 8s");
        let flags: u32 = 1 | if residual.is_some() { 2 } else { 0 } | if keep { 4 } else { 0 };
        gpu.dispatch_1d(
            "in_apply_pad",
            pairs / 4,
            &[
                xp.arg(),
                x.arg(),
                residual.unwrap_or(x).arg(),
                self.stats.arg(),
                n.w.arg(),
                n.b.arg(),
                b.into(),
                depth.into(),
                h.into(),
                h.into(),
                cols.into(),
                dpad.into(),
                flags.into(),
                (pairs / 4).into(),
            ],
        )
    }

    /// Per-(sample, channel) mean and inverse std into `stats`, from the `tiles` partials per sample the last conv left
    /// in `partial`. Call it after each conv, before the next one overwrites the partials
    fn norm_stats(&self, gpu: &Gpu, cols: usize, tiles: usize) -> Result<()> {
        let b = self.batch;
        ensure!(
            b * tiles * cols * 16 <= self.partial.bytes() && b * cols * 8 <= self.stats.bytes(),
            "IN scratch too small"
        );
        ensure!(cols.is_multiple_of(16), "in_finalize over {cols} channels: needs a multiple of 16");
        gpu.dispatch(
            "in_finalize",
            [cols / 16, b, 1],
            [256, 1, 1],
            &[self.stats.arg(), self.partial.arg(), tiles.into(), cols.into(), EPS.into()],
        )
    }

    /// Queues one forward: gather and normalise the tiles `desc` describes (`[batch, DESC]`) from the resident `vol`,
    /// run the model and write each real tile's logits into its slot of `logits`
    pub fn forward(&self, gpu: &Gpu, vol: (&Buf, usize, usize), desc: Arg, logits: &Buf) -> Result<()> {
        self.stem(gpu, vol, desc, &self.pooled)?;
        self.body(gpu, desc, logits)
    }

    /// A buffer for one batch of the stem's output, `[batch, px, px, 32]` in the storage type
    pub fn pooled_buf(&self, gpu: &Gpu) -> Result<Buf> {
        gpu.alloc_real_nan(self.batch * self.edge.pixels() * 32)
    }

    /// Queues the first half of a forward: gather and normalise the tiles, run the 3D stem and write the depth
    /// pools to `pooled` (`[batch, px, px, 32]` in the storage type, attention sums then maxima, villa's `cat` order)
    pub fn stem(&self, gpu: &Gpu, vol: (&Buf, usize, usize), desc: Arg, pooled: &Buf) -> Result<()> {
        let b = self.batch;
        let px = self.edge.px();
        let pixels = px * px;
        let vox = DEPTH * pixels;
        let scratch = &self.scratch;
        ensure!(pooled.bytes() >= b * pixels * 32 * gpu.precision().bytes(), "pooled buffer too small");

        self.tiles.run(gpu, vol, desc)?;

        // 3D stem and depth pools
        let part = &self.partial;
        self.norm_stats(gpu, STEM, stem_c1(gpu, (b, px), (&self.tiles.x0, &self.stem1_w, &self.s1), part)?)?;
        let pairs = (b * stem_padded(px) + stem_tail(px)) * STEM / 2;
        let pad = Pad { xp: scratch, dpad: 1, pairs, keep: false };
        self.norm_pad(gpu, &self.s1, (DEPTH, px, STEM), &self.stem1_n, None, pad)?;
        self.norm_stats(gpu, STEM, stem_c16(gpu, (b, px), (scratch, &self.stem2_w, &self.s2), part)?)?;
        self.norm(gpu, &self.s2, (vox, STEM), &self.stem2_n, None, true)?;
        let n = b * pixels;
        gpu.dispatch_1d(
            "depth_pool",
            n,
            &[pooled.arg(), self.s2.arg(), self.att_w.arg(), self.att_b.into(), pixels.into(), n.into()],
        )
    }

    /// Queues the 2D half of a forward on `self.pooled`, writing each real tile's logits into its slot of `logits`
    fn body(&self, gpu: &Gpu, desc: Arg, logits: &Buf) -> Result<()> {
        let b = self.batch;
        let px = self.edge.px();
        let pixels = px * px;
        let scratch = &self.scratch;
        let part = &self.partial;

        // encoder. An IN whose output feeds a stride-1 tap conv writes that conv's padded input as well (`padded`),
        // and keeps the unpadded copy for the residuals and pools that also read it
        let blocks: Vec<&Block> = self.blocks.iter().flatten().collect();
        let taps_next = |k: usize| blocks.get(k).is_some_and(|n| n.conv1.kind == Kind::Taps);
        let to_pad = |h: usize, c: usize| Pad { xp: scratch, dpad: 0, pairs: padded_rows(b, h) * c / 2, keep: true };
        self.norm_stats(gpu, 32, self.enc.run(gpu, b, scratch, &self.pooled, &self.e0, part)?)?;
        let mut padded = taps_next(0);
        if padded {
            self.norm_pad(gpu, &self.e0, (1, px, 32), &self.enc_n, None, to_pad(px, 32))?;
        } else {
            self.norm(gpu, &self.e0, (pixels, 32), &self.enc_n, None, true)?;
        }
        let mut x = &self.e0;
        for (k, blk) in blocks.iter().enumerate() {
            let rows = blk.h * blk.h;
            let tiles = if padded {
                blk.conv1.run_padded(gpu, b, scratch, &blk.t1, part)?
            } else {
                blk.conv1.run(gpu, b, scratch, x, &blk.t1, part)?
            };
            self.norm_stats(gpu, blk.c, tiles)?;
            // conv1's input in scratch is consumed by now: the encoder runs its dispatches in order
            let pad = Pad { keep: false, ..to_pad(blk.h, blk.c) };
            self.norm_pad(gpu, &blk.t1, (1, blk.h, blk.c), &blk.n1, None, pad)?;
            // the residual path runs before conv2, so the skip conv's IN partials and statistics are spent before
            // conv2 writes its own; it reads only the block input and leaves scratch (conv2's padded input) alone
            let residual = match (&blk.pooled, &blk.skip) {
                (Some(p), skip) => {
                    let n = b * rows * blk.cin / 2;
                    gpu.dispatch_1d(
                        "avgpool2",
                        n,
                        &[p.arg(), x.arg(), blk.h.into(), blk.h.into(), blk.cin.into(), n.into()],
                    )?;
                    match skip {
                        Some((conv, norm, out)) => {
                            ensure!(conv.kind == Kind::Pointwise, "a skip conv that would use scratch");
                            self.norm_stats(gpu, blk.c, conv.run(gpu, b, scratch, p, out, part)?)?;
                            self.norm(gpu, out, (rows, blk.c), norm, None, false)?;
                            out
                        }
                        None => p,
                    }
                }
                (None, _) => x,
            };
            self.norm_stats(gpu, blk.c, blk.conv2.run_padded(gpu, b, scratch, &blk.out, part)?)?;
            padded = taps_next(k + 1);
            if padded {
                self.norm_pad(gpu, &blk.out, (1, blk.h, blk.c), &blk.n2, Some(residual), to_pad(blk.h, blk.c))?;
            } else {
                self.norm(gpu, &blk.out, (rows, blk.c), &blk.n2, Some(residual), true)?;
            }
            x = &blk.out;
        }

        // decoder
        let mut low = x;
        for (i, u) in self.ups.iter().enumerate() {
            let skip = &self.blocks[4 - i].last().unwrap().out;
            let (m, hh) = (b * u.h * u.h, 2 * u.h);
            gemm(gpu, low, &u.w, &u.g, (m, 4 * u.up, u.cin))?;
            let n = padded_rows(b, hh) * (u.up + u.skip_c) / 2;
            ensure!(n * 2 * gpu.precision().bytes() <= scratch.bytes(), "decoder scratch too small");
            gpu.dispatch_1d(
                "cat_up",
                n,
                &[
                    scratch.arg(),
                    u.g.arg(),
                    u.bias.arg(),
                    skip.arg(),
                    b.into(),
                    hh.into(),
                    hh.into(),
                    u.up.into(),
                    u.skip_c.into(),
                    n.into(),
                ],
            )?;
            self.norm_stats(gpu, u.up, u.conv.run_padded(gpu, b, scratch, &u.out, part)?)?;
            self.norm(gpu, &u.out, (hh * hh, u.up), &u.n, None, true)?;
            low = &u.out;
        }

        let n = b * pixels;
        gpu.dispatch_1d(
            "head_logits",
            n,
            &[
                logits.arg(),
                low.arg(),
                self.head_w.arg(),
                self.head_b.into(),
                desc,
                n.into(),
                u32::try_from(logits.bytes() / (gpu.precision().bytes() * pixels))
                    .wrap_err("logits slots past u32")?
                    .into(),
                pixels.into(),
            ],
        )
    }

    /// The activations of the last forward, channels-last
    pub fn taps(&self) -> Vec<Tap<'_>> {
        let (b, p) = (self.batch, self.edge.px());
        let tap = |name: String, buf, shape: Vec<usize>| Tap { name, buf, shape };
        let mut v = vec![
            tap("input".into(), &self.tiles.x0, vec![b, DEPTH, p, p, 1]),
            tap("stem.block1".into(), &self.s1, vec![b, DEPTH, p, p, STEM]),
            tap("stem.block2".into(), &self.s2, vec![b, DEPTH, p, p, STEM]),
            tap("stem.out".into(), &self.pooled, vec![b, p, p, 32]),
            tap("enc.stem".into(), &self.e0, vec![b, p, p, 32]),
        ];
        for (s, stage) in self.blocks.iter().enumerate() {
            for (j, blk) in stage.iter().enumerate() {
                v.push(tap(format!("enc.{s}.{j}"), &blk.out, vec![b, blk.h, blk.h, blk.c]));
            }
        }
        for (i, u) in self.ups.iter().enumerate() {
            v.push(tap(format!("dec.{i}"), &u.out, vec![b, 2 * u.h, 2 * u.h, u.up]));
        }
        v
    }
}

/// Rejects a batch whose activations reach 2^31 elements at tile edge `px`: matmul2d tensor extents and offsets are
/// `int32`, and the glue kernels index elements with 32-bit arithmetic. The stem's padded conv3d input is the largest
/// buffer, about 5.1M elements per sample at 128 and 45M at 384, so this caps the batch at 417 and 47
fn check_batch(batch: usize, px: usize) -> Result<()> {
    let largest = (batch * stem_padded(px) + stem_tail(px)) * STEM;
    ensure!(batch > 0, "batch 0");
    ensure!(
        largest < i32::MAX as usize,
        "batch {batch} too large for the Metal backend: its stem activation is {largest} elements, past 2^31"
    );
    Ok(())
}

/// Where `norm_pad` writes: `pairs` pairs of the padded layout `[B, depth + 2 dpad, h + 2, h + 2, cols]` plus
/// zero tail rows in `xp`; `keep` also writes the result back into the unpadded input, for other readers
#[derive(Clone, Copy)]
struct Pad<'a> {
    xp: &'a Buf,
    dpad: usize,
    pairs: usize,
    keep: bool,
}

/// Zero rows after the stem's padded batch, so the last sample's 27 tap windows stay inside the buffer
fn stem_tail(px: usize) -> usize {
    let wp = px + 2;
    2 * wp * wp + 2 * wp + 2
}

/// The stem's conv3d 1 -> 16 of `x` (`[B, 17, px, px]`) with weights `w` into `y`, its IN partials into `part`;
/// returns the partials per sample
fn stem_c1(gpu: &Gpu, (b, px): (usize, usize), (x, w, y): (&Buf, &Buf, &Buf), part: &Buf) -> Result<usize> {
    let tiles = DEPTH * px * px / STEM_C1_GROUP;
    ensure!(b * tiles * STEM * 16 <= part.bytes(), "IN partials too small");
    gpu.dispatch(
        "conv3d_c1",
        [b * tiles, 1, 1],
        [STEM_C1_GROUP, 1, 1],
        &[y.arg(), x.arg(), w.arg(), part.arg(), px.into()],
    )?;
    Ok(tiles)
}

/// The stem's conv3d 16 -> 16 of the padded `xp` with weights `w` into `y`, its IN partials into `part`; returns the
/// partials per sample
fn stem_c16(gpu: &Gpu, (b, px): (usize, usize), (xp, w, y): (&Buf, &Buf, &Buf), part: &Buf) -> Result<usize> {
    let tiles = stem_c16_tiles(px);
    ensure!(b * tiles * STEM * 16 <= part.bytes(), "IN partials too small");
    gpu.dispatch(
        "conv3d_c16_64",
        [1, b * tiles, 1],
        [128, 1, 1],
        &[xp.arg(), w.arg(), y.arg(), b.into(), part.arg(), px.into()],
    )?;
    Ok(tiles)
}

/// A logits buffer of `slots` tiles of edge `edge` in the storage type, NaN-filled so an unwritten slot fails the
/// finish checks
pub fn logits_buf(gpu: &Gpu, slots: usize, edge: Edge) -> Result<Buf> {
    gpu.alloc_real_nan(slots.max(1) * edge.pixels())
}

#[cfg(test)]
mod tests {
    //! Every conv and GEMM shape class of the model against an f64 CPU reference on FP16-rounded random data, at both
    //! reader precisions: the matmul2d tile shapes are new, a shape that compiles can still return garbage, and the
    //! fp32 tolerance is tight enough to catch a matmul2d that drops FP32 operands to a narrower type

    use super::*;
    use crate::fast::ReaderPrecision;

    /// Deterministic values in [-1, 1), rounded to FP16
    fn rand16(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let v = ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
                half::f16::from_f32(v).to_f32()
            })
            .collect()
    }

    fn download(gpu: &Gpu, b: &Buf, n: usize) -> Vec<f32> {
        let mut out = vec![f32::NAN; n];
        gpu.download_real(b, 0, &mut out).unwrap();
        out
    }

    /// NHWC conv reference in f64: x [b, h, h, c], w [o, c, k, k], stride s, pad k / 2
    fn conv_ref(x: &[f32], w: &[f32], (b, h, c, o, k, s): (usize, usize, usize, usize, usize, usize)) -> Vec<f64> {
        let ho = h / s;
        let pad = k as i64 / 2;
        let mut y = vec![0f64; b * ho * ho * o];
        for n in 0..b {
            for oy in 0..ho {
                for ox in 0..ho {
                    for oc in 0..o {
                        let mut acc = 0f64;
                        for dy in 0..k {
                            for dx in 0..k {
                                let iy = (oy * s + dy) as i64 - pad;
                                let ix = (ox * s + dx) as i64 - pad;
                                if iy < 0 || ix < 0 || iy >= h as i64 || ix >= h as i64 {
                                    continue;
                                }
                                let xi = ((n * h + iy as usize) * h + ix as usize) * c;
                                for ic in 0..c {
                                    acc += x[xi + ic] as f64 * w[((oc * c + ic) * k + dy) * k + dx] as f64;
                                }
                            }
                        }
                        y[((n * ho + oy) * ho + ox) * o + oc] = acc;
                    }
                }
            }
        }
        y
    }

    /// Fails unless every output is within one rounding to the storage type of the reference, plus FP32
    /// accumulation slack
    fn check(gpu: &Gpu, name: &str, got: &[f32], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{name}: length");
        let (rel, abs) = match gpu.precision() {
            ReaderPrecision::Fp16 => (1e-3, 1e-4),
            // FP32 accumulation over K up to 5760 products drifts by about sqrt(K) 2^-24 of the running sum, near 1e-5
            // of the largest output; operands narrowed to a 10-bit mantissa would be off by about 1e-3 of it
            ReaderPrecision::Fp32 => (1e-5, 1e-5),
        };
        let name = format!("{name} {}", gpu.precision().name());
        let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-3);
        let mut worst = 0f64;
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(g.is_finite(), "{name}: non-finite at {i}");
            let err = (g as f64 - w).abs();
            let tol = w.abs() * rel + scale * abs;
            assert!(err <= tol, "{name}: element {i} got {g}, want {w} (tol {tol})");
            worst = worst.max(err / scale);
        }
        eprintln!("{name}: {} values, worst error {worst:.2e} of scale {scale:.3}", got.len());
    }

    /// Fails unless `in_finalize` over the conv's partials gives each (sample, channel) of its FP16 output `y` ([b,
    /// rows, cols]) the f64 mean and inverse std, within float rounding
    fn check_stats(gpu: &Gpu, name: &str, part: &Buf, tiles: usize, y: &[f32], (b, rows, cols): (usize, usize, usize)) {
        let stats = gpu.alloc(b * cols * 8).unwrap();
        gpu.dispatch(
            "in_finalize",
            [cols / 16, b, 1],
            [256, 1, 1],
            &[stats.arg(), part.arg(), tiles.into(), cols.into(), EPS.into()],
        )
        .unwrap();
        gpu.synchronize().unwrap();
        let mut got = vec![[0f32; 2]; b * cols];
        gpu.download(&stats, 0, &mut got).unwrap();
        for n in 0..b {
            for ch in 0..cols {
                let vals = || (0..rows).map(|r| y[(n * rows + r) * cols + ch] as f64);
                let mean = vals().sum::<f64>() / rows as f64;
                let var = vals().map(|v| (v - mean) * (v - mean)).sum::<f64>() / rows as f64;
                let inv = 1.0 / (var + EPS as f64).sqrt();
                let [gm, gi] = got[n * cols + ch];
                let sd = var.sqrt().max(1e-3);
                assert!(
                    ((gm as f64 - mean) / sd).abs() < 1e-5 && ((gi as f64 - inv) / inv).abs() < 1e-5,
                    "{name}: sample {n} channel {ch}: stats ({gm}, {gi}), want ({mean}, {inv})"
                );
            }
        }
    }

    fn conv_case(gpu: &Gpu, (b, h, c, o): (usize, usize, usize, usize), kind: Kind, seed: u64) {
        let (k, s) = match kind {
            Kind::Taps => (3, 1),
            Kind::Strided => (3, 2),
            Kind::Pointwise => (1, 1),
        };
        let x = rand16(b * h * h * c, seed);
        let w = rand16(o * c * k * k, seed + 1);
        let conv = Conv::from_torch(gpu, &w, (c, o, h), kind).unwrap();
        let ho = conv.out_h();
        let xb = gpu.upload_real(&x).unwrap();
        let y = gpu.alloc_real_nan(b * ho * ho * o).unwrap();
        let scratch = gpu.alloc(conv.scratch(b).max(1) * gpu.precision().bytes()).unwrap();
        let part = gpu.alloc(b * conv.tiles() * o * 16).unwrap();
        let tiles = conv.run(gpu, b, &scratch, &xb, &y, &part).unwrap();
        let got = download(gpu, &y, b * ho * ho * o);
        let name = format!("{kind:?} b{b} h{h} {c}->{o}");
        check(gpu, &name, &got, &conv_ref(&x, &w, (b, h, c, o, k, s)));
        check_stats(gpu, &name, &part, tiles, &got, (b, ho * ho, o));
    }

    #[test]
    fn batch_limit_follows_32_bit_indexing() {
        assert!(check_batch(1, 128).is_ok() && check_batch(417, 128).is_ok());
        assert!(check_batch(0, 128).is_err() && check_batch(418, 128).is_err() && check_batch(964, 128).is_err());
        assert!(check_batch(47, 384).is_ok() && check_batch(48, 384).is_err());
    }

    fn convs_case(gpu: &Gpu) {
        // every tap kernel width, ragged row tiles, the narrow C=32 layers and the 4x4 bottleneck
        for (i, &(b, h, c, o)) in
            [(2, 8, 64, 64), (3, 16, 32, 32), (2, 4, 320, 320), (2, 8, 640, 320), (1, 32, 48, 16), (5, 6, 96, 32)]
                .iter()
                .enumerate()
        {
            conv_case(gpu, (b, h, c, o), Kind::Taps, 10 + i as u64);
        }
        for (i, &(b, h, c, o)) in [(2, 8, 32, 64), (3, 16, 64, 128), (2, 8, 256, 320)].iter().enumerate() {
            conv_case(gpu, (b, h, c, o), Kind::Strided, 30 + i as u64);
        }
        for (i, &(b, h, c, o)) in [(2, 8, 32, 64), (3, 4, 256, 320)].iter().enumerate() {
            conv_case(gpu, (b, h, c, o), Kind::Pointwise, 50 + i as u64);
        }
    }

    // one test per precision, so a validation run can select fp16: shader validation doubles a kernel's threadgroup
    // memory, and the fp32 64-wide tiles' 18 KB then fail its 32 KB check
    #[test]
    fn convs_match_reference_fp16() {
        convs_case(&Gpu::open_at(0, ReaderPrecision::Fp16).unwrap());
    }

    #[test]
    fn convs_match_reference_fp32() {
        convs_case(&Gpu::open_at(0, ReaderPrecision::Fp32).unwrap());
    }

    #[test]
    fn transposed_gemm_matches_reference_fp16() {
        transposed_gemm_case(&Gpu::open_at(0, ReaderPrecision::Fp16).unwrap());
    }

    #[test]
    fn transposed_gemm_matches_reference_fp32() {
        transposed_gemm_case(&Gpu::open_at(0, ReaderPrecision::Fp32).unwrap());
    }

    fn transposed_gemm_case(gpu: &Gpu) {
        let (m, cin, up) = (2 * 4 * 4 + 3, 320, 320);
        let x = rand16(m * cin, 7);
        let tw = rand16(cin * up * 4, 8);
        let w = gpu.upload_real(&transp_rows(&tw, cin, up)).unwrap();
        let xb = gpu.upload_real(&x).unwrap();
        let y = gpu.alloc_real_nan(m * 4 * up).unwrap();
        gemm(gpu, &xb, &w, &y, (m, 4 * up, cin)).unwrap();
        let got = download(gpu, &y, m * 4 * up);
        let mut want = vec![0f64; m * 4 * up];
        for r in 0..m {
            for co in 0..up {
                for d in 0..4 {
                    let mut acc = 0f64;
                    for ci in 0..cin {
                        acc += x[r * cin + ci] as f64 * tw[(ci * up + co) * 4 + d] as f64;
                    }
                    want[r * 4 * up + d * up + co] = acc;
                }
            }
        }
        check(gpu, "transposed gemm", &got, &want);
    }

    #[test]
    fn stem_convs_match_reference_fp16() {
        stem_case(&Gpu::open_at(0, ReaderPrecision::Fp16).unwrap());
    }

    #[test]
    fn stem_convs_match_reference_fp32() {
        stem_case(&Gpu::open_at(0, ReaderPrecision::Fp32).unwrap());
    }

    fn stem_case(gpu: &Gpu) {
        let b = 2;
        let px = 128;
        let vox = DEPTH * px * px;
        let (d, hw) = (DEPTH as i64, px as i64);
        let x = rand16(b * vox, 3);
        let w1 = rand16(STEM * 27, 4);
        let xb = gpu.upload_real(&x).unwrap();
        let w1b = gpu.upload_real(&conv_nhwc_f32(&w1, STEM, 1, 27)).unwrap();
        let s1 = gpu.alloc_real_nan(b * vox * STEM).unwrap();
        let part = gpu.alloc(b * stem_c16_tiles(px) * STEM * 16).unwrap();
        let tiles1 = stem_c1(gpu, (b, px), (&xb, &w1b, &s1), &part).unwrap();
        let got1 = download(gpu, &s1, b * vox * STEM);
        // one reference loop for both convs: input [b, D, H, W, c], weight [o, c, 27]
        let conv3 = |x: &[f32], w: &[f32], c: usize| -> Vec<f64> {
            let mut y = vec![0f64; b * vox * STEM];
            for n in 0..b {
                for z in 0..d {
                    for yy in 0..hw {
                        for xx in 0..hw {
                            let o_at = (((n as i64 * d + z) * hw + yy) * hw + xx) as usize * STEM;
                            for t in 0..27i64 {
                                let (zz, y2, x2) = (z + t / 9 - 1, yy + (t / 3) % 3 - 1, xx + t % 3 - 1);
                                if zz < 0 || zz >= d || y2 < 0 || y2 >= hw || x2 < 0 || x2 >= hw {
                                    continue;
                                }
                                let i_at = (((n as i64 * d + zz) * hw + y2) * hw + x2) as usize * c;
                                for oc in 0..STEM {
                                    let mut acc = 0f64;
                                    for ic in 0..c {
                                        acc += x[i_at + ic] as f64 * w[(oc * c + ic) * 27 + t as usize] as f64;
                                    }
                                    y[o_at + oc] += acc;
                                }
                            }
                        }
                    }
                }
            }
            y
        };
        check(gpu, "conv3d 1->16", &got1, &conv3(&x, &w1, 1));
        check_stats(gpu, "conv3d 1->16", &part, tiles1, &got1, (b, vox, STEM));

        let x2 = rand16(b * vox * STEM, 5);
        let w2 = rand16(STEM * STEM * 27, 6);
        let x2b = gpu.upload_real(&x2).unwrap();
        let w2b = gpu.upload_real(&conv_nhwc_f32(&w2, STEM, STEM, 27)).unwrap();
        let n = (b * stem_padded(px) + stem_tail(px)) * STEM / 2;
        let scratch = gpu.alloc(n * 2 * gpu.precision().bytes()).unwrap();
        let s2 = gpu.alloc_real_nan(b * vox * STEM).unwrap();
        gpu.dispatch_1d("pad3d", n, &[scratch.arg(), x2b.arg(), b.into(), n.into(), px.into()]).unwrap();
        let tiles2 = stem_c16(gpu, (b, px), (&scratch, &w2b, &s2), &part).unwrap();
        let got2 = download(gpu, &s2, b * vox * STEM);
        check(gpu, "conv3d 16->16", &got2, &conv3(&x2, &w2, STEM));
        check_stats(gpu, "conv3d 16->16", &part, tiles2, &got2, (b, vox, STEM));
    }
}
