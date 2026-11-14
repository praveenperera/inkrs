//! villa's `Local3DStem2DUNet` for `hybrid_3d2d-seed42`, reader-v2 and Reader V3, planned once at a
//! fixed batch and tile edge and run on the GPU under FP16 autocast semantics
//!
//! The three readers share the architecture below; Reader V3 runs it on 17 x 384 x 384 tiles instead
//! of 17 x 128 x 128 ([`Edge`]). The kernels that compile the edge in have `_384` copies, and every
//! other kernel and cuDNN plan takes its sizes at run time
//!
//! - Stem: conv3d 1->16 (k3, no bias), InstanceNorm3d, LeakyReLU, conv3d 16->16, IN, LeakyReLU,
//!   then the 1x1x1 attention logits, softmax over depth and the attention and max pools,
//!   giving 32 channels at 128x128. The first conv and its IN and LeakyReLU are inkrs kernels
//!   (`stem1_*`: the conv's statistics, then the conv again with the norm in its epilogue), not
//!   cuDNN. So is the second conv (`stem2_conv`, its IN's statistics in the epilogue); that IN's
//!   apply and LeakyReLU run inside the pools' kernel (`depth_pool`), so the second conv's output is
//!   only ever stored raw
//! - Encoder: conv 32->32 + IN + LeakyReLU, then 6 stages of BasicBlockD with features
//!   [32, 64, 128, 256, 320, 320] and [1, 3, 4, 6, 6, 6] blocks; the first block of stages 1-5
//!   has stride 2 and a skip of AvgPool2d(2), plus a 1x1 conv and IN when the width changes
//! - Decoder: 5 stages of ConvTranspose2d(k2, s2) (one GEMM, `cat_up`), concatenation with the
//!   encoder skip, conv 3x3 + IN + LeakyReLU
//! - Head: 1x1 conv 32->1, into each tile's logits slot (`head_logits`)
//!
//! Every conv feeds an InstanceNorm, which removes a per-channel bias exactly, so the convs run
//! without theirs; their bias tensors are still checked and consumed
//!
//! The 2D convs run on inkrs's `conv_*` kernels ([`crate::conv`]), which write the InstanceNorm's
//! statistics from their epilogue, or with [`ConvBackend::Cudnn`] as cuDNN plans followed by the
//! `in_partial` and `in_finalize` statistics kernels

use std::cell::Cell;

use color_eyre::eyre::{Result, ensure};

use crate::conv::{Cfg, ConvBackend, OwnConv};
use crate::dnn::{Blas, ConvPlan, ConvShape, Dnn, Pick};
use crate::gpu::{Arg, Buf, Gpu};
use crate::sched::{DEPTH, DESC, Edge, percentile_params, percentile_params_torch};
use crate::weights::{Checkpoint, conv_nhwc, round16, stem2_frags, transp_gemm};

const STEM: usize = 16;
const FEATURES: [usize; 6] = [32, 64, 128, 256, 320, 320];
const BLOCKS: [usize; 6] = [1, 3, 4, 6, 6, 6];
const EPS: f32 = 1e-5;
/// `stem1_stats` blocks per depth slice: 16 bands of 8 rows at 128 (the kernel's `STEM1_CHUNKS`
/// over 17), 48 bands of 3 strips of 128 columns at 384 (`STEM1_CHUNKS_384` over 17)
const fn stem1_slice_blocks(edge: Edge) -> usize {
    match edge {
        Edge::P128 => 16,
        Edge::P384 => 48 * 3,
    }
}
/// `stem2_conv` patches of 8 x 32 per depth slice: 64 at 128, 576 at 384 (the kernels'
/// `STEM2_BLOCKS` over 17)
const fn stem2_slice_blocks(edge: Edge) -> usize {
    edge.pixels() / (8 * 32)
}
/// `stem2_conv` blocks along the depth, each running up to 9 slices of a patch (the kernel's
/// `STEM2_ZCHUNKS`)
const STEM2_ZCHUNKS: u32 = 2;

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

/// A 2D conv of the reader: inkrs's kernel with its weight as B fragments, or a cuDNN plan with its
/// FP16 channels-last weight
pub enum Conv {
    Own(OwnConv),
    Cudnn { plan: ConvPlan, w: Buf },
}

impl Conv {
    /// `key.weight` `[k, c, size, size]`, consuming `key.bias` when `bias` (it is cancelled by the
    /// InstanceNorm that follows)
    fn load(ctx: &Ctx, backend: ConvBackend, ck: &Checkpoint, key: &str, shape: ConvShape, bias: bool) -> Result<Self> {
        let ksize: usize = shape.kernel.iter().map(|&v| v as usize).product();
        let (k, c) = (shape.k as usize, shape.c as usize);
        let mut dims = vec![k, c];
        dims.extend(shape.kernel.iter().map(|&v| v as usize));
        let w = ck.take(&format!("{key}.weight"), &dims)?;
        if bias {
            ck.take(&format!("{key}.bias"), &[k])?;
        }
        Ok(match backend {
            // a shape no own kernel covers stays on cuDNN, so `--conv inkrs` runs any tile size
            ConvBackend::Inkrs if Cfg::choose(&shape).is_some() => Self::Own(OwnConv::new(ctx.gpu, shape, &w)?),
            ConvBackend::Inkrs | ConvBackend::Cudnn => {
                let w = ctx.gpu.upload(&conv_nhwc(&w, k, c, ksize))?;
                Self::Cudnn { plan: ConvPlan::new(ctx.gpu, ctx.dnn, shape, ctx.pick, ctx.max_workspace)?, w }
            }
        })
    }

    pub fn shape(&self) -> &ConvShape {
        match self {
            Self::Own(o) => &o.shape,
            Self::Cudnn { plan, .. } => &plan.shape,
        }
    }

    /// What runs the conv: the cuDNN algorithm, or `inkrs <kernel>`
    pub fn algo(&self) -> String {
        match self {
            Self::Own(o) => format!("inkrs {}", o.cfg.name),
            Self::Cudnn { plan, .. } => plan.algo.to_string(),
        }
    }

    /// Whether a cuDNN plan's ranking came from the on-disk algorithm cache
    pub fn cached(&self) -> bool {
        matches!(self, Self::Cudnn { plan, .. } if plan.cached)
    }

    fn workspace(&self) -> usize {
        match self {
            Self::Own(_) => 0,
            Self::Cudnn { plan, .. } => plan.workspace(),
        }
    }

    /// InstanceNorm partial pairs per sample the conv needs: its own epilogue's, else none
    fn partial_per_sample(&self) -> usize {
        match self {
            Self::Own(o) => o.partial_len() / o.shape.n as usize,
            Self::Cudnn { .. } => 0,
        }
    }
}

/// Planning context
pub struct Ctx<'a> {
    pub gpu: &'a Gpu,
    pub dnn: &'a Dnn,
    pub pick: Pick,
    pub max_workspace: usize,
}

/// One BasicBlockD
struct Block {
    conv1: Conv,
    n1: Norm,
    conv2: Conv,
    n2: Norm,
    /// AvgPool2d(2) of the input, for a strided block
    pooled: Option<Buf>,
    /// 1x1 conv + IN of the pooled input, when the width changes
    skip: Option<(Conv, Norm, Buf)>,
    t1: Buf,
    out: Buf,
    cin: usize,
    c: usize,
    /// Output edge
    h: usize,
}

/// One decoder stage
struct Up {
    w: Buf,
    bias: Buf,
    g: Buf,
    cat: Buf,
    conv: Conv,
    n: Norm,
    out: Buf,
    /// Input edge, channels and the transposed conv's output channels
    h: usize,
    cin: usize,
    up: usize,
    skip_c: usize,
}

/// Whose float32 arithmetic the robust normalisation replays: the same statistics, rounded
/// differently, so the FP16 input can differ in its last bit
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NormArith {
    /// villa's `normalize_robust` (numpy), for villa parity
    #[default]
    Villa,
    /// fastink's `robust_norm_gpu` (torch), which the PHerc. 0306B recipe's TTA reads run
    /// (`crate::tta`): percentiles as `a * (1 - t) + b * t` with float32 weights, and the MAD
    /// scale in float32. Its rare std fallback stays numpy's, which torch's reduction order may
    /// differ from in the last bit
    Fastink,
}

/// Tile gather and villa's robust normalisation for one batch: histograms, statistics, the
/// exact std for MAD-fallback tiles, and the normalised FP16 input `x0` (`[B, 17, edge, edge]`)
pub struct TilePrep {
    batch: usize,
    edge: Edge,
    hist: Buf,
    tstats: Buf,
    std: TileStd,
    pub x0: Buf,
    /// The lower and upper percentiles' parameters, numpy's and then torch's
    pct: [[(u32, f32, f32); 2]; 2],
}

/// Partial sums per tile in each pass of the 384 edge's split std: `tiles384::STD_BLOCKS`, the
/// kernels' block count per tile, which skip a tile whose buffer is shorter
pub(crate) const STD_BLOCKS_384: usize = 256;

/// How the prep computes numpy's exact std for MAD-fallback tiles, by edge
enum TileStd {
    /// `tile_std`: one block per tile walks the whole pairwise-sum tree (128 edge, 279K voxels)
    Block,
    /// `tile_sum_384`, `tile_sq_384`, `tile_std_finish_384`: the tree split over [`STD_BLOCKS_384`]
    /// blocks per tile and pass (384 edge, 2.5M voxels, where one block per tile took 4.6 ms a
    /// batch), with each pass's partials `[batch, STD_BLOCKS_384]`
    Split { sum: Buf, sq: Buf },
}

impl TileStd {
    fn new(gpu: &Gpu, batch: usize, edge: Edge) -> Result<Self> {
        Ok(match edge {
            Edge::P128 => Self::Block,
            Edge::P384 => {
                Self::Split { sum: gpu.alloc(batch * STD_BLOCKS_384 * 4)?, sq: gpu.alloc(batch * STD_BLOCKS_384 * 4)? }
            }
        })
    }

    /// Queues the std of the tiles `tile_stats` marked, into their `stats[3]`
    fn run(&self, gpu: &Gpu, b: usize, stats: &Buf, vol: (&Buf, usize, usize), desc: Arg) -> Result<()> {
        let (vbuf, height, width) = vol;
        let (h, w) = (Arg::U32(height as u32), Arg::U32(width as u32));
        match self {
            Self::Block => gpu.launch(
                "tile_std",
                (b as u32, 1, 1),
                (256, 1, 1),
                0,
                &[stats.arg::<[f32; 4]>(), vbuf.arg::<u8>(), desc, h, w],
            ),
            Self::Split { sum, sq } => {
                let grid = (STD_BLOCKS_384 as u32, b as u32, 1);
                let (sum, sq) = (sum.view::<f32>(0, b * STD_BLOCKS_384), sq.view::<f32>(0, b * STD_BLOCKS_384));
                gpu.launch(
                    "tile_sum_384",
                    grid,
                    (256, 1, 1),
                    0,
                    &[sum, stats.arg::<[f32; 4]>(), vbuf.arg::<u8>(), desc, h, w],
                )?;
                gpu.launch(
                    "tile_sq_384",
                    grid,
                    (256, 1, 1),
                    0,
                    &[sq, sum, stats.arg::<[f32; 4]>(), vbuf.arg::<u8>(), desc, h, w],
                )?;
                gpu.launch(
                    "tile_std_finish_384",
                    (b as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &[stats.arg::<[f32; 4]>(), sq, desc],
                )
            }
        }
    }
}

impl TilePrep {
    pub fn new(gpu: &Gpu, batch: usize, edge: Edge) -> Result<Self> {
        let vox = DEPTH * edge.pixels();
        Ok(Self {
            batch,
            edge,
            hist: gpu.alloc(batch * 256 * 4)?,
            tstats: gpu.alloc(batch * 16)?,
            std: TileStd::new(gpu, batch, edge)?,
            x0: f16_buf(gpu, batch * vox)?,
            pct: [
                [percentile_params(vox, 1.0), percentile_params(vox, 99.0)],
                [percentile_params_torch(vox, 1.0), percentile_params_torch(vox, 99.0)],
            ],
        })
    }

    /// Queues the batch `desc` describes (`[batch, DESC]`) from the resident volume, normalised
    /// with `arith`'s rounding
    pub fn run(&self, gpu: &Gpu, vol: (&Buf, usize, usize), desc: Arg, arith: NormArith) -> Result<()> {
        let b = self.batch;
        let (vbuf, height, width) = vol;
        let edge = self.edge;
        let vox = DEPTH * edge.pixels();
        gpu.fill(&self.hist, 0)?;
        gpu.launch(
            edge.kernel("tile_hist"),
            (DEPTH as u32, b as u32, 1),
            (256, 1, 1),
            0,
            &[self.hist.arg::<u32>(), vbuf.arg::<u8>(), desc, Arg::U32(height as u32), Arg::U32(width as u32)],
        )?;
        let fastink = arith == NormArith::Fastink;
        let [(li, lt, lo), (hi, ht, ho)] = self.pct[fastink as usize];
        gpu.launch_1d(
            "tile_stats",
            b,
            &[
                self.tstats.arg::<[f32; 4]>(),
                self.hist.arg::<u32>(),
                Arg::U32(vox as u32),
                Arg::U32(li),
                Arg::F32(lt),
                Arg::F32(lo),
                Arg::U32(hi),
                Arg::F32(ht),
                Arg::F32(ho),
                Arg::U32(fastink as u32),
            ],
        )?;
        // numpy's exact float32 std for the tiles whose MAD fell back
        self.std.run(gpu, b, &self.tstats, vol, desc)?;
        gpu.launch_1d(
            edge.kernel("tile_norm"),
            b * vox / 2,
            &[
                self.x0.arg::<u32>(),
                vbuf.arg::<u8>(),
                desc,
                self.tstats.arg::<[f32; 4]>(),
                Arg::U32(height as u32),
                Arg::U32(width as u32),
            ],
        )?;
        Ok(())
    }
}

/// The planned model and its activations at batch `batch`
pub struct Model {
    pub batch: usize,
    /// The tile edge, from the export's `crop_size`
    pub edge: Edge,
    /// Which library runs the 2D convs
    pub backend: ConvBackend,
    /// The first stem conv's FP16 filter `[16, 27]`: it runs as inkrs's `stem1_*` kernels with its
    /// norm fused, not as a cuDNN plan
    stem1_w: Buf,
    stem1_n: Norm,
    /// The second stem conv's B fragments (`stem2_frags`): it runs as inkrs's `stem2_conv` with its
    /// norm's statistics in the epilogue, not as a cuDNN plan
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
    // activations and scratch
    tiles: TilePrep,
    s1: Buf,
    s2: Buf,
    pooled: Buf,
    e0: Buf,
    partial: Buf,
    stats: Buf,
    /// `[B, cols]` statistics of a projection block's skip norm, apart from `stats`, which still holds
    /// the block's conv2 statistics until its norm applies them after the skip
    skip_stats: Buf,
    /// `[B, 16]` statistics of the stem's second norm, which `depth_pool` applies on load and
    /// [`Model::taps`] needs after the forward, so no later norm may reuse them
    stem2_stats: Buf,
    /// Whether `s2` still holds the raw second stem conv, so [`Model::taps`] may normalise it in
    /// place: set by each forward, cleared by `taps`, which would otherwise normalise it twice
    s2_raw: Cell<bool>,
    /// `[B, 16]` float32 shift of `stem1_stats`' partial sums, for `stem1_finalize`
    stem1_shift: Buf,
    ws: Buf,
    blas: Blas,
}

/// A named activation of the last forward, for the P0 gate: buffer, channels-last shape
pub struct Tap<'a> {
    pub name: String,
    pub buf: &'a Buf,
    pub shape: Vec<usize>,
}

fn f16_buf(gpu: &Gpu, n: usize) -> Result<Buf> {
    // NaN-filled, so a layer that is never written fails the finiteness checks
    gpu.alloc_fill(n * 2, 0xFF)
}

impl Model {
    /// Plans every layer at `batch` for `edge` tiles with its 2D convs on `backend` and uploads the
    /// weights; every checkpoint tensor must be consumed
    pub fn new(ctx: &Ctx, ck: &Checkpoint, batch: usize, edge: Edge, backend: ConvBackend) -> Result<Self> {
        let gpu = ctx.gpu;
        let n = batch as i32;
        let (patch, pixels) = (edge.px(), edge.pixels());
        let p = patch as i32;
        let vox = DEPTH * pixels;

        // stem
        let stem1_w =
            gpu.upload(&conv_nhwc(&ck.take("depth_fusion.features.0.weight", &[STEM, 1, 3, 3, 3])?, STEM, 1, 27))?;
        let stem1_n = Norm::load(gpu, ck, "depth_fusion.features.1", STEM)?;
        let stem2_w = gpu.upload(&stem2_frags(&ck.take("depth_fusion.features.3.weight", &[STEM, STEM, 3, 3, 3])?))?;
        let stem2_n = Norm::load(gpu, ck, "depth_fusion.features.4", STEM)?;
        let att_w = gpu.upload(&round16(&ck.take("depth_fusion.attention_logits.weight", &[1, STEM, 1, 1, 1])?))?;
        let att_b = round16(&[ck.scalar("depth_fusion.attention_logits.bias")?])[0];

        // encoder
        let enc_key = "network.shared_encoder.stem.convs.0";
        let enc =
            Conv::load(ctx, backend, ck, &format!("{enc_key}.conv"), ConvShape::same(n, 32, 32, &[p, p], 3, 1), true)?;
        let enc_n = Norm::load(gpu, ck, &format!("{enc_key}.norm"), 32)?;
        let mut blocks = Vec::new();
        let (mut cin, mut h) = (32usize, patch);
        for (s, (&c, &nb)) in FEATURES.iter().zip(&BLOCKS).enumerate() {
            let mut stage = Vec::new();
            for j in 0..nb {
                let key = format!("network.shared_encoder.stages.{s}.blocks.{j}");
                let strided = j == 0 && s > 0;
                let (hin, hout) = (h as i32, if strided { h / 2 } else { h });
                let conv1 = Conv::load(
                    ctx,
                    backend,
                    ck,
                    &format!("{key}.conv1.conv"),
                    ConvShape::same(n, cin as i32, c as i32, &[hin, hin], 3, if strided { 2 } else { 1 }),
                    true,
                )?;
                let n1 = Norm::load(gpu, ck, &format!("{key}.conv1.norm"), c)?;
                let conv2 = Conv::load(
                    ctx,
                    backend,
                    ck,
                    &format!("{key}.conv2.conv"),
                    ConvShape::same(n, c as i32, c as i32, &[hout as i32, hout as i32], 3, 1),
                    true,
                )?;
                let n2 = Norm::load(gpu, ck, &format!("{key}.conv2.norm"), c)?;
                let pooled = if strided { Some(f16_buf(gpu, batch * hout * hout * cin)?) } else { None };
                let skip = if cin != c {
                    ensure!(strided, "stage {s} block {j}: a width change without a stride");
                    let conv = Conv::load(
                        ctx,
                        backend,
                        ck,
                        &format!("{key}.skip.1.conv"),
                        ConvShape::same(n, cin as i32, c as i32, &[hout as i32, hout as i32], 1, 1),
                        false,
                    )?;
                    let norm = Norm::load(gpu, ck, &format!("{key}.skip.1.norm"), c)?;
                    Some((conv, norm, f16_buf(gpu, batch * hout * hout * c)?))
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
                    t1: f16_buf(gpu, batch * hout * hout * c)?,
                    out: f16_buf(gpu, batch * hout * hout * c)?,
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
            let tb = round16(&ck.take(&format!("network.shared_decoder.transpconvs.{i}.bias"), &[up])?);
            let key = format!("network.shared_decoder.stages.{i}.convs.0");
            let conv = Conv::load(
                ctx,
                backend,
                ck,
                &format!("{key}.conv"),
                ConvShape::same(n, (up + skip_c) as i32, up as i32, &[hh as i32, hh as i32], 3, 1),
                true,
            )?;
            let norm = Norm::load(gpu, ck, &format!("{key}.norm"), up)?;
            ups.push(Up {
                w: gpu.upload(&transp_gemm(&tw, low_c, up))?,
                bias: gpu.upload(&tb)?,
                g: f16_buf(gpu, batch * low_h * low_h * 4 * up)?,
                cat: f16_buf(gpu, batch * hh * hh * (up + skip_c))?,
                conv,
                n: norm,
                out: f16_buf(gpu, batch * hh * hh * up)?,
                h: low_h,
                cin: low_c,
                up,
                skip_c,
            });
            low_c = up;
            low_h = hh;
        }
        ensure!(low_h == patch && low_c == 32, "decoder ends at {low_h}x{low_h}x{low_c}");
        let head_w = gpu.upload(&round16(&ck.take("network.task_heads.ink.weight", &[1, 32, 1, 1])?))?;
        let head_b = round16(&[ck.scalar("network.task_heads.ink.bias")?])[0];

        // scratch sized for the largest layer
        let mut convs: Vec<&Conv> = vec![&enc];
        for b in blocks.iter().flatten() {
            convs.extend([&b.conv1, &b.conv2]);
            if let Some((c, _, _)) = &b.skip {
                convs.push(c);
            }
        }
        convs.extend(ups.iter().map(|u| &u.conv));
        let ws = gpu.alloc(convs.iter().map(|c| c.workspace()).max().unwrap_or(0))?;
        // (rows, cols) of every InstanceNorm, for the partial-sum scratch
        let mut norms = vec![(pixels, 32)];
        norms.extend(blocks.iter().flatten().map(|b| (b.h * b.h, b.c)));
        norms.extend(ups.iter().map(|u| (4 * u.h * u.h, u.up)));
        // and the stem norms', one partial per block of `stem1_stats` and per (slice, patch) of
        // `stem2_conv`, and the own convs', one per output patch
        let stem_partials = DEPTH * stem1_slice_blocks(edge).max(stem2_slice_blocks(edge));
        let partial_len = norms
            .iter()
            .map(|&(r, c)| in_layout(r, c).2 * c)
            .chain(convs.iter().map(|c| c.partial_per_sample()))
            .max()
            .unwrap()
            .max(stem_partials * STEM);
        let max_cols = norms.iter().map(|&(_, c)| c).max().unwrap();

        Ok(Self {
            batch,
            edge,
            backend,
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
            s1: f16_buf(gpu, batch * vox * STEM)?,
            s2: f16_buf(gpu, batch * vox * STEM)?,
            pooled: f16_buf(gpu, batch * pixels * 32)?,
            e0: f16_buf(gpu, batch * pixels * 32)?,
            partial: gpu.alloc(batch * partial_len * 8)?,
            stats: gpu.alloc(batch * max_cols * 8)?,
            skip_stats: gpu.alloc(batch * max_cols * 8)?,
            stem2_stats: gpu.alloc(batch * STEM * 8)?,
            s2_raw: Cell::new(false),
            stem1_shift: gpu.alloc(batch * STEM * 4)?,
            ws,
            blas: Blas::new(gpu)?,
        })
    }

    /// The 2D convs with their names, in the order the forward runs them, for the report
    pub fn convs(&self) -> Vec<(String, &Conv)> {
        let mut v = vec![("enc.stem".to_string(), &self.enc)];
        for (s, stage) in self.blocks.iter().enumerate() {
            for (j, b) in stage.iter().enumerate() {
                v.push((format!("enc.{s}.{j}.conv1"), &b.conv1));
                v.push((format!("enc.{s}.{j}.conv2"), &b.conv2));
                if let Some((c, _, _)) = &b.skip {
                    v.push((format!("enc.{s}.{j}.skip"), c));
                }
            }
        }
        for (i, u) in self.ups.iter().enumerate() {
            v.push((format!("dec.{i}.conv"), &u.conv));
        }
        v
    }

    /// `conv` of `x` into `y` (`[B, rows, cols]`), with the InstanceNorm `[mean, invstd]` of `y` into
    /// `stats`: from an own conv's epilogue, or from the statistics kernels after a cuDNN plan
    #[allow(clippy::too_many_arguments)]
    fn conv_stats(
        &self,
        ctx: &Ctx,
        conv: &Conv,
        x: &Buf,
        y: &Buf,
        (rows, cols): (usize, usize),
        stats: &Buf,
    ) -> Result<()> {
        match conv {
            Conv::Own(o) => {
                ensure!(
                    o.shape.k as usize == cols && o.shape.n as usize == self.batch,
                    "{}: not [B, {rows}, {cols}]",
                    o.shape
                );
                o.run(ctx.gpu, x, y, &self.partial, stats, EPS)
            }
            Conv::Cudnn { plan, w } => {
                plan.run(ctx.dnn, (x.ptr(), x.bytes()), (w.ptr(), w.bytes()), (y.ptr(), y.bytes()), Some(&self.ws))?;
                self.norm_stats(ctx.gpu, y, (rows, cols), stats)
            }
        }
    }

    /// `conv` of `x` into `y` (`[B, rows, cols]`), then its InstanceNorm in place, the optional
    /// residual add and LeakyReLU
    #[allow(clippy::too_many_arguments)]
    fn conv_norm(
        &self,
        ctx: &Ctx,
        conv: &Conv,
        x: &Buf,
        y: &Buf,
        dims: (usize, usize),
        n: &Norm,
        leaky: bool,
    ) -> Result<()> {
        self.conv_stats(ctx, conv, x, y, dims, &self.stats)?;
        self.norm_apply(ctx.gpu, y, dims, n, &self.stats, None, leaky)
    }

    /// Per (sample, channel) `[mean, invstd]` of `x` (`[B, rows, cols]`) into `stats`
    fn norm_stats(&self, gpu: &Gpu, x: &Buf, (rows, cols): (usize, usize), stats: &Buf) -> Result<()> {
        let b = self.batch;
        let pairs = cols / 2;
        let (lanes, chunk, chunks) = in_layout(rows, cols);
        let partial_len = b * chunks * cols;
        ensure!(partial_len * 8 <= self.partial.bytes() && b * cols * 8 <= stats.bytes(), "IN scratch too small");
        gpu.launch(
            "in_partial",
            (chunks as u32, b as u32, 1),
            ((pairs * lanes) as u32, 1, 1),
            0,
            &[
                self.partial.view::<[f32; 2]>(0, partial_len),
                x.arg::<u32>(),
                Arg::U32(rows as u32),
                Arg::U32(cols as u32),
                Arg::U32(lanes as u32),
                Arg::U32(chunk as u32),
            ],
        )?;
        gpu.launch_1d(
            "in_finalize",
            b * cols,
            &[
                stats.view::<[f32; 2]>(0, b * cols),
                self.partial.view::<[f32; 2]>(0, partial_len),
                x.arg::<u16>(),
                Arg::U32(rows as u32),
                Arg::U32(cols as u32),
                Arg::U32(chunks as u32),
                Arg::F32(EPS),
            ],
        )
    }

    /// Applies `stats` (from [`Model::norm_stats`] of `x`) to `x` in place, then the optional
    /// residual add and LeakyReLU
    #[allow(clippy::too_many_arguments)]
    fn norm_apply(
        &self,
        gpu: &Gpu,
        x: &Buf,
        (rows, cols): (usize, usize),
        n: &Norm,
        stats: &Buf,
        residual: Option<&Buf>,
        leaky: bool,
    ) -> Result<()> {
        let b = self.batch;
        gpu.launch_1d(
            "in_apply",
            b * rows * (cols / 2),
            &[
                x.arg::<u32>(),
                residual.map_or(Arg::EMPTY, |r| r.arg::<u32>()),
                stats.view::<[f32; 2]>(0, b * cols),
                n.w.arg::<f32>(),
                n.b.arg::<f32>(),
                Arg::U32(rows as u32),
                Arg::U32(cols as u32),
                Arg::U32(leaky as u32),
            ],
        )
    }

    /// The first stem conv with its InstanceNorm and LeakyReLU, from the tiles `x0` into `s1`: a
    /// statistics-only pass of the conv, the statistics, then the conv again writing the activation
    fn stem1(&self, gpu: &Gpu) -> Result<()> {
        let (b, edge) = (self.batch, self.edge);
        let chunks = DEPTH * stem1_slice_blocks(edge);
        ensure!(DEPTH == 17, "the stem1 kernels are built for 17-slice tiles");
        ensure!(b * chunks * STEM * 8 <= self.partial.bytes(), "IN scratch too small");
        let grid = (stem1_slice_blocks(edge) as u32, DEPTH as u32, b as u32);
        gpu.launch(
            edge.kernel("stem1_stats"),
            grid,
            (128, 1, 1),
            0,
            &[
                self.partial.view::<[f32; 2]>(0, b * chunks * STEM),
                self.stem1_shift.arg::<f32>(),
                self.tiles.x0.arg::<u16>(),
                self.stem1_w.arg::<u16>(),
            ],
        )?;
        let finalize_args = [
            self.stats.view::<[f32; 2]>(0, b * STEM),
            self.partial.view::<[f32; 2]>(0, b * chunks * STEM),
            self.stem1_shift.arg::<f32>(),
            Arg::F32(EPS),
        ];
        match edge {
            Edge::P128 => gpu.launch_1d("stem1_finalize", b * STEM, &finalize_args)?,
            // one warp per (sample, channel) over its 2448 partials
            Edge::P384 => {
                gpu.launch("stem1_finalize_384", ((b * STEM / 4) as u32, 1, 1), (128, 1, 1), 0, &finalize_args)?
            }
        }
        gpu.launch(
            edge.kernel("stem1_apply"),
            grid,
            (128, 1, 1),
            0,
            &[
                self.s1.arg::<u32>(),
                self.tiles.x0.arg::<u16>(),
                self.stem1_w.arg::<u16>(),
                self.stats.view::<[f32; 2]>(0, b * STEM),
                self.stem1_n.w.arg::<f32>(),
                self.stem1_n.b.arg::<f32>(),
            ],
        )
    }

    /// The second stem conv, raw into `s2`, with its norm's `[mean, invstd]` into `stem2_stats`: the
    /// conv writes each (slice, patch)'s per-channel mean and squared deviations, which
    /// `stem2_finalize` merges, one warp per (sample, channel)
    fn stem2(&self, gpu: &Gpu) -> Result<()> {
        let (b, edge) = (self.batch, self.edge);
        let partial_len = b * DEPTH * stem2_slice_blocks(edge) * STEM;
        ensure!(DEPTH == 17 && STEM == 16, "the stem2 kernels are built for 17-slice tiles of 16 channels");
        ensure!(partial_len * 8 <= self.partial.bytes(), "IN scratch too small");
        gpu.launch(
            edge.kernel("stem2_conv"),
            (stem2_slice_blocks(edge) as u32, STEM2_ZCHUNKS, b as u32),
            (128, 1, 1),
            0,
            &[
                self.s2.arg::<u32>(),
                self.partial.view::<[f32; 2]>(0, partial_len),
                self.s1.arg::<[u32; 4]>(),
                self.stem2_w.arg::<[u32; 4]>(),
            ],
        )?;
        gpu.launch(
            edge.kernel("stem2_finalize"),
            ((b * STEM / 4) as u32, 1, 1),
            (128, 1, 1),
            0,
            &[
                self.stem2_stats.view::<[f32; 2]>(0, b * STEM),
                self.partial.view::<[f32; 2]>(0, partial_len),
                Arg::F32(EPS),
            ],
        )
    }

    /// Queues one forward: gather and normalise the tiles `desc` describes (`[batch, DESC]`,
    /// a view of the descriptor buffer) from the resident `vol`, run the model and write each
    /// real tile's logits into its slot of `logits`
    pub fn forward(&self, ctx: &Ctx, vol: (&Buf, usize, usize), desc: Arg, logits: &Buf) -> Result<()> {
        self.forward_with(ctx, vol, desc, logits, NormArith::Villa)
    }

    /// [`Self::forward`] with the input normalised by `arith`'s arithmetic
    pub fn forward_with(
        &self,
        ctx: &Ctx,
        vol: (&Buf, usize, usize),
        desc: Arg,
        logits: &Buf,
        arith: NormArith,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let b = self.batch;
        let pixels = self.edge.pixels();

        self.tiles.run(gpu, vol, desc, arith)?;

        // 3D stem and depth pools
        self.stem1(gpu)?;
        // the second norm's apply and LeakyReLU run inside depth_pool, so s2 stays raw
        self.stem2(gpu)?;
        // 8 lanes per pixel, 256-thread blocks of 32 pixels of one sample
        ensure!(pixels.is_multiple_of(32), "depth_pool needs whole blocks per sample");
        gpu.launch(
            "depth_pool",
            ((b * pixels / 32) as u32, 1, 1),
            (256, 1, 1),
            0,
            &[
                self.pooled.arg::<u32>(),
                self.s2.arg::<u32>(),
                self.stem2_stats.view::<[f32; 2]>(0, b * STEM),
                self.stem2_n.w.arg::<f32>(),
                self.stem2_n.b.arg::<f32>(),
                self.att_w.arg::<f32>(),
                Arg::F32(self.att_b),
                Arg::U32(pixels as u32),
            ],
        )?;
        self.s2_raw.set(true);

        // encoder
        self.conv_norm(ctx, &self.enc, &self.pooled, &self.e0, (pixels, 32), &self.enc_n, true)?;
        let mut x = &self.e0;
        for stage in &self.blocks {
            for blk in stage {
                let rows = blk.h * blk.h;
                self.conv_norm(ctx, &blk.conv1, x, &blk.t1, (rows, blk.c), &blk.n1, true)?;
                // conv2's statistics wait in `stats` while the skip path norms with `skip_stats`
                self.conv_stats(ctx, &blk.conv2, &blk.t1, &blk.out, (rows, blk.c), &self.stats)?;
                let residual = match (&blk.pooled, &blk.skip) {
                    (Some(p), skip) => {
                        gpu.launch_1d(
                            "avgpool2",
                            b * rows * blk.cin / 2,
                            &[
                                p.arg::<u32>(),
                                x.arg::<u32>(),
                                Arg::U32(blk.h as u32),
                                Arg::U32(blk.h as u32),
                                Arg::U32(blk.cin as u32),
                            ],
                        )?;
                        match skip {
                            Some((conv, norm, out)) => {
                                self.conv_stats(ctx, conv, p, out, (rows, blk.c), &self.skip_stats)?;
                                self.norm_apply(gpu, out, (rows, blk.c), norm, &self.skip_stats, None, false)?;
                                out
                            }
                            None => p,
                        }
                    }
                    (None, _) => x,
                };
                self.norm_apply(gpu, &blk.out, (rows, blk.c), &blk.n2, &self.stats, Some(residual), true)?;
                x = &blk.out;
            }
        }

        // decoder
        let mut low = x;
        for (i, u) in self.ups.iter().enumerate() {
            let skip = &self.blocks[4 - i].last().unwrap().out;
            let (m, hh) = (b * u.h * u.h, 2 * u.h);
            self.blas.gemm_f16(
                (m, 4 * u.up, u.cin),
                (low.ptr(), low.bytes()),
                (u.w.ptr(), u.w.bytes()),
                (u.g.ptr(), u.g.bytes()),
            )?;
            gpu.launch_1d(
                "cat_up",
                b * hh * hh * (u.up + u.skip_c) / 2,
                &[
                    u.cat.arg::<u32>(),
                    u.g.arg::<u32>(),
                    u.bias.arg::<f32>(),
                    skip.arg::<u32>(),
                    Arg::U32(hh as u32),
                    Arg::U32(hh as u32),
                    Arg::U32(u.up as u32),
                    Arg::U32(u.skip_c as u32),
                ],
            )?;
            self.conv_norm(ctx, &u.conv, &u.cat, &u.out, (hh * hh, u.up), &u.n, true)?;
            low = &u.out;
        }

        gpu.launch_1d(
            self.edge.kernel("head_logits"),
            b * pixels,
            &[logits.arg::<u16>(), low.arg::<[u32; 16]>(), self.head_w.arg::<f32>(), Arg::F32(self.head_b), desc],
        )
    }

    /// The activations of the last forward, channels-last, for the P0 gate
    ///
    /// The forward leaves `stem.block2` raw (`depth_pool` normalises it on load), so this first
    /// queues the second stem norm's apply on it, as the unfused forward did; read the taps before
    /// the next forward, and call this once per forward
    pub fn taps(&self, gpu: &Gpu) -> Result<Vec<Tap<'_>>> {
        let b = self.batch;
        let (p, pixels) = (self.edge.px(), self.edge.pixels());
        ensure!(self.s2_raw.replace(false), "taps needs a forward first, and runs once per forward");
        self.norm_apply(gpu, &self.s2, (DEPTH * pixels, STEM), &self.stem2_n, &self.stem2_stats, None, true)?;
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
        Ok(v)
    }
}

/// `in_partial`'s launch shape for `[rows, cols]`: lanes per channel pair (the block is
/// `cols / 2 * lanes` threads, at most 256), rows per block (about 64 per lane) and blocks
/// per sample
fn in_layout(rows: usize, cols: usize) -> (usize, usize, usize) {
    let lanes = (256 / (cols / 2)).max(1);
    let chunk = (lanes * 64).min(rows).max(1);
    (lanes, chunk, rows.div_ceil(chunk))
}

/// Device descriptors and logits slots for a schedule
pub fn upload_desc(gpu: &Gpu, desc: &[i32]) -> Result<Buf> {
    ensure!(desc.len().is_multiple_of(DESC), "descriptor table not a multiple of {DESC}");
    gpu.upload(desc)
}

/// A logits buffer of `slots` tiles of `edge`, NaN-filled so an unwritten slot fails the finish checks
pub fn logits_buf(gpu: &Gpu, slots: usize, edge: Edge) -> Result<Buf> {
    gpu.alloc_fill(slots.max(1) * edge.pixels() * 2, 0xFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every tensor name and shape the architecture above implies, written out independently
    /// of `Model::new`, so a mismatched export fails on the CPU before any GPU work
    fn expected_keys() -> Vec<(String, Vec<usize>)> {
        let mut k: Vec<(String, Vec<usize>)> = Vec::new();
        let conv = |k: &mut Vec<(String, Vec<usize>)>, key: &str, o: usize, i: usize, ks: &[usize], bias: bool| {
            let mut s = vec![o, i];
            s.extend(ks);
            k.push((format!("{key}.weight"), s));
            if bias {
                k.push((format!("{key}.bias"), vec![o]));
            }
        };
        let norm = |k: &mut Vec<(String, Vec<usize>)>, key: &str, c: usize| {
            k.push((format!("{key}.weight"), vec![c]));
            k.push((format!("{key}.bias"), vec![c]));
        };
        conv(&mut k, "depth_fusion.features.0", 16, 1, &[3, 3, 3], false);
        norm(&mut k, "depth_fusion.features.1", 16);
        conv(&mut k, "depth_fusion.features.3", 16, 16, &[3, 3, 3], false);
        norm(&mut k, "depth_fusion.features.4", 16);
        conv(&mut k, "depth_fusion.attention_logits", 1, 16, &[1, 1, 1], true);
        conv(&mut k, "network.shared_encoder.stem.convs.0.conv", 32, 32, &[3, 3], true);
        norm(&mut k, "network.shared_encoder.stem.convs.0.norm", 32);
        let mut cin = 32;
        for (s, (&c, &nb)) in FEATURES.iter().zip(&BLOCKS).enumerate() {
            for j in 0..nb {
                let key = format!("network.shared_encoder.stages.{s}.blocks.{j}");
                conv(&mut k, &format!("{key}.conv1.conv"), c, cin, &[3, 3], true);
                norm(&mut k, &format!("{key}.conv1.norm"), c);
                conv(&mut k, &format!("{key}.conv2.conv"), c, c, &[3, 3], true);
                norm(&mut k, &format!("{key}.conv2.norm"), c);
                if cin != c {
                    conv(&mut k, &format!("{key}.skip.1.conv"), c, cin, &[1, 1], false);
                    norm(&mut k, &format!("{key}.skip.1.norm"), c);
                }
                cin = c;
            }
        }
        let dec = [(320, 320, 320), (320, 256, 256), (256, 128, 128), (128, 64, 64), (64, 32, 32)];
        for (i, &(low, up, skip)) in dec.iter().enumerate() {
            k.push((format!("network.shared_decoder.transpconvs.{i}.weight"), vec![low, up, 2, 2]));
            k.push((format!("network.shared_decoder.transpconvs.{i}.bias"), vec![up]));
            conv(&mut k, &format!("network.shared_decoder.stages.{i}.convs.0.conv"), up, up + skip, &[3, 3], true);
            norm(&mut k, &format!("network.shared_decoder.stages.{i}.convs.0.norm"), up);
        }
        conv(&mut k, "network.task_heads.ink", 1, 32, &[1, 1], true);
        k
    }

    /// Every supported villa reader shares this architecture: seed 42, reader-v2 and Reader V3
    #[test]
    fn exports_match_the_architecture() {
        for reader in ["hybrid_3d2d-seed42", "reader-v2", "reader-v3"] {
            let path = crate::engine::weights_dir().join(format!("{reader}.safetensors"));
            if !path.exists() {
                eprintln!("skipped: no export at {}", path.display());
                continue;
            }
            let ck = Checkpoint::load(&path, crate::fast::ReaderPrecision::Fp16).unwrap();
            for (key, shape) in expected_keys() {
                ck.take(&key, &shape).unwrap();
            }
            let (tensors, aliases, params) = ck.finish().unwrap();
            assert_eq!((tensors, aliases, params), (264, 244, 34546498), "{reader}");
            assert_eq!(crate::engine::reader_name(&path).unwrap(), reader);
        }
    }
}
