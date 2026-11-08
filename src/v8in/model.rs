//! v8-in's network on the GPU at a fixed batch, FP16 storage as the reference's autocast runs it
//!
//! - Input and stem: `v8_stats` (the per-tile InstanceNorm) and `v8_stem` (the upsample, the 7^3 conv, BN,
//!   ReLU and max pool as one pass, [`super::StemTables`])
//! - layer1..4: Bottleneck blocks [3, 4, 6, 3], planes 64/128/256/512 (x4 out), stride 2 on conv2 and the
//!   downsample in layers 2-4. Features 256 @ 96x64x64, 512 @ 48x32x32, 1024 @ 24x16x16, 2048 @ 12x8x8
//! - depth max of each layer (`v8_dmax`) straight into the decoder's concatenation buffers
//! - decoder: three steps of `v8_up2cat` (bilinear x2 of the deeper map, after the skip) and a 3x3 conv
//!   with BN and ReLU; head `v8_head`, the 1x1 logit and sigmoid
//!
//! Every BatchNorm is folded on the host into its conv's weights (`w * scale`, rounded to FP16 once) and a
//! bias (`shift`), and the conv runs as cuDNN's `y = relu(conv + z + bias)` ([`Fused`]) where cuDNN takes
//! the shape, else as the plain conv plus `v8_epilogue`. A downsample's shift joins its block's conv3
//! bias, so the downsample is a plain conv. Rounding therefore differs from the reference, which rounds
//! the conv output to FP16, then the BN output, then the sum; the real-arithmetic result is the same and
//! the parity gates bound the difference

use std::collections::HashMap;
use std::rc::Rc;

use color_eyre::eyre::{Result, ensure, eyre};
use half::f16;

use crate::dnn::bias_act::{ConvIo, Fused};
use crate::dnn::{ConvPlan, ConvShape, Dtype};
use crate::gpu::{Arg, Buf, Gpu};
use crate::hecate::u32_extent;
use crate::model::Ctx;
use crate::sched::DESC;
use crate::v8in::host::{BLOCKS, DIMS, PLANES, batch_norm};
pub use crate::v8in::host::{READERS, check_contract, force_unfused, is_v8in_export};
use crate::v8in::{PH, PIX, SC, SD, StemTables};
use crate::weights::Checkpoint;

/// Conv plans and fused plans by shape
struct Planner<'a> {
    ctx: &'a Ctx<'a>,
    plans: HashMap<ConvShape, Rc<ConvPlan>>,
    fused: HashMap<ConvShape, Option<Rc<Fused>>>,
    /// `INKRS_V8_UNFUSED=1`: every conv runs plain plus `v8_epilogue`, so the P0 gate can check the
    /// fallback path on a GPU where cuDNN fuses every shape
    no_fuse: bool,
}

impl Planner<'_> {
    fn plan(&mut self, shape: ConvShape) -> Result<Rc<ConvPlan>> {
        if let Some(p) = self.plans.get(&shape) {
            return Ok(p.clone());
        }
        let ctx = self.ctx;
        let plan = Rc::new(ConvPlan::typed(ctx.gpu, ctx.dnn, shape.clone(), Dtype::Half, ctx.pick, ctx.max_workspace)?);
        self.plans.insert(shape, plan.clone());
        Ok(plan)
    }

    /// The fused plan for `shape`, or `None` when cuDNN refuses the fused call for it with either bias
    /// type, found by running it once on scratch buffers
    fn fused(&mut self, shape: ConvShape, ws: &Buf) -> Result<Option<Rc<Fused>>> {
        if self.no_fuse {
            return Ok(None);
        }
        if let Some(f) = self.fused.get(&shape) {
            return Ok(f.clone());
        }
        let plan = self.plan(shape.clone())?;
        let gpu = self.ctx.gpu;
        let (xe, we, ye) = shape.elements();
        let (x, w, y) = (gpu.alloc_fill(xe * 2, 0)?, gpu.alloc_fill(we * 2, 0)?, gpu.alloc_fill(ye * 2, 0)?);
        let mut found = None;
        for dt in [Dtype::Float { tf32: false }, Dtype::Half] {
            let f = Fused::new(plan.clone(), dt)?;
            let bias = gpu.alloc_fill(shape.k as usize * dt.bytes(), 0)?;
            let wsb = if ws.bytes() >= plan.workspace() { Some(ws) } else { None };
            let io = ConvIo { x: (x.ptr(), x.bytes()), w: (w.ptr(), w.bytes()), z: None, y: (y.ptr(), y.bytes()) };
            let ok = f.run(self.ctx.dnn, io, &bias, wsb).is_ok() && gpu.sync().is_ok();
            if ok {
                found = Some(Rc::new(f));
                break;
            }
        }
        gpu.sync()?;
        self.fused.insert(shape, found.clone());
        Ok(found)
    }
}

/// A conv with its BatchNorm folded: FP16 channels-last weight, the bias in float32 for `v8_epilogue`
/// and in the fused plan's bias type
struct FConv {
    plan: Rc<ConvPlan>,
    fused: Option<Rc<Fused>>,
    w: Buf,
    bias32: Buf,
    bias_fused: Option<Buf>,
    k: usize,
    /// Whether this conv carries a bias and ReLU (every conv but the downsample)
    want: bool,
}

impl FConv {
    /// `key.weight` (`[k, c, *kernel]`) times `scale[k]`, as `[k, *kernel, c]` FP16, and `shift` as the
    /// bias. `fuse` asks for cuDNN's fused bias-ReLU; a plain conv (the downsample) passes `false`
    #[allow(clippy::too_many_arguments)]
    fn load(
        planner: &mut Planner,
        ck: &Checkpoint,
        key: &str,
        shape: ConvShape,
        scale: &[f64],
        shift: &[f64],
        fuse: bool,
        ws: &Buf,
    ) -> Result<Self> {
        let (k, c) = (shape.k as usize, shape.c as usize);
        let taps: usize = shape.kernel.iter().map(|&v| v as usize).product();
        let mut dims = vec![k, c];
        dims.extend(shape.kernel.iter().map(|&v| v as usize));
        let w = ck.take(&format!("{key}.weight"), &dims)?;
        let mut nhwc = vec![f16::ZERO; k * taps * c];
        for ko in 0..k {
            for ci in 0..c {
                for t in 0..taps {
                    let v = w[(ko * c + ci) * taps + t] as f64 * scale[ko];
                    let h = f16::from_f64(v);
                    ensure!(h.is_finite(), "{key}: folded weight {v} overflows FP16");
                    nhwc[(ko * taps + t) * c + ci] = h;
                }
            }
        }
        let gpu = planner.ctx.gpu;
        let plan = planner.plan(shape.clone())?;
        let fused = if fuse { planner.fused(shape, ws)? } else { None };
        let b32: Vec<f32> = shift.iter().map(|&v| v as f32).collect();
        let bias_fused = match &fused {
            Some(f) if f.bias_dtype == Dtype::Half => {
                Some(gpu.upload(&shift.iter().map(|&v| f16::from_f64(v)).collect::<Vec<_>>())?)
            }
            Some(_) => Some(gpu.upload(&b32)?),
            None => None,
        };
        Ok(Self { plan, fused, w: gpu.upload(&nhwc)?, bias32: gpu.upload(&b32)?, bias_fused, k, want: fuse })
    }

    fn out_elems(&self) -> usize {
        self.plan.shape.elements().2
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

/// The planned network and its activations at batch `batch`
pub struct Model {
    pub batch: usize,
    in_norm: [f32; 2],
    stem: StemTables,
    stem_bufs: [Buf; 6],
    layers: Vec<Vec<Bottleneck>>,
    /// Edge of each layer's output `[d, h, h]`
    dims: [[usize; 3]; 4],
    dec: Vec<DecStep>,
    head_w: Buf,
    head_b: f32,
    // activations: the residual stream in turn, the two inner convs, the downsample, the plain-conv scratch
    stats: Buf,
    stream: [Buf; 2],
    t1: Buf,
    t2: Buf,
    ds: Buf,
    raw: Option<Buf>,
    /// decoder: concatenation buffers per level 0..2, the deepest pooled map, and each step's output
    cat: Vec<Buf>,
    p4: Buf,
    dout: Vec<Buf>,
    ws: Buf,
    /// Shapes cuDNN would not fuse, for the report
    pub unfused: Vec<String>,
}

/// One of the last forward's activations for the P0 gate: buffer (FP16), channels-last shape, channels
/// A per-stage callback of [`Model::forward`] for the P0 gate: the stage's name (`stats`, `stem`,
/// `layer1`..`layer4`), its buffer and its channels-last `[batch, depth, h, w, c]` shape (`stats` is
/// `[batch, 1, 1, 1, 2]` float32, the rest FP16); it runs after the stage is queued, before a later stage
/// reuses the buffer
pub type StageHook<'h> = &'h mut dyn FnMut(&str, &Buf, [usize; 5]) -> Result<()>;

pub struct Tap<'a> {
    pub name: String,
    pub buf: &'a Buf,
    pub shape: Vec<usize>,
}

impl Model {
    /// Plans every layer at `batch` and uploads the weights; every checkpoint tensor must be consumed
    /// (the caller checks with [`Checkpoint::finish`])
    pub fn new(ctx: &Ctx, ck: &Checkpoint, batch: usize) -> Result<Self> {
        let gpu = ctx.gpu;
        ensure!(batch > 0, "batch size must be positive");
        let n = batch as i32;
        // the largest activation, the stream, is [B, 96, 64, 64, 256]
        u32_extent("activation", batch * SD * PH * PH * DIMS[0])?;
        let ws = gpu.alloc(ctx.max_workspace.max(1))?;
        let mut planner = Planner { ctx, plans: HashMap::new(), fused: HashMap::new(), no_fuse: force_unfused() };

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

        // encoder
        let mut layers = Vec::new();
        let mut dims = [[0usize; 3]; 4];
        let (mut cin, mut d) = (SC, [SD, PH, PH]);
        let to_i = |v: [usize; 3]| v.map(|x| x as i32);
        let (mut t1_len, mut t2_len, mut ds_len, mut stream_len) = (0usize, 0usize, 0usize, batch * SD * PH * PH * SC);
        for (l, (&planes, &nb)) in PLANES.iter().zip(&BLOCKS).enumerate() {
            let stride = if l == 0 { 1 } else { 2 };
            let (din, dout) = (d, d.map(|v| v / stride));
            let c = planes * 4;
            let mut blocks = Vec::new();
            for j in 0..nb {
                let key = format!("backbone.layer{}.{j}", l + 1);
                let (s0, bin, bdin) = if j == 0 { (stride as i32, cin, din) } else { (1, c, dout) };
                let (s1, t1) = batch_norm(ck, &format!("{key}.bn1"), planes)?;
                let (s2, t2) = batch_norm(ck, &format!("{key}.bn2"), planes)?;
                let (s3, mut t3) = batch_norm(ck, &format!("{key}.bn3"), c)?;
                let downsample = if j == 0 {
                    let (sd, td) = batch_norm(ck, &format!("{key}.downsample.1"), c)?;
                    t3.iter_mut().zip(&td).for_each(|(a, b)| *a += b);
                    let zero = vec![0f64; c];
                    Some(FConv::load(
                        &mut planner,
                        ck,
                        &format!("{key}.downsample.0"),
                        ConvShape::same(n, bin as i32, c as i32, &to_i(bdin), 1, s0),
                        &sd,
                        &zero,
                        false,
                        &ws,
                    )?)
                } else {
                    None
                };
                let conv1 = FConv::load(
                    &mut planner,
                    ck,
                    &format!("{key}.conv1"),
                    ConvShape::same(n, bin as i32, planes as i32, &to_i(bdin), 1, 1),
                    &s1,
                    &t1,
                    true,
                    &ws,
                )?;
                let conv2 = FConv::load(
                    &mut planner,
                    ck,
                    &format!("{key}.conv2"),
                    ConvShape::same(n, planes as i32, planes as i32, &to_i(bdin), 3, s0),
                    &s2,
                    &t2,
                    true,
                    &ws,
                )?;
                let conv3 = FConv::load(
                    &mut planner,
                    ck,
                    &format!("{key}.conv3"),
                    ConvShape::same(n, planes as i32, c as i32, &to_i(dout), 1, 1),
                    &s3,
                    &t3,
                    true,
                    &ws,
                )?;
                t1_len = t1_len.max(conv1.out_elems());
                t2_len = t2_len.max(conv2.out_elems());
                stream_len = stream_len.max(conv3.out_elems());
                if let Some(ds) = &downsample {
                    ds_len = ds_len.max(ds.out_elems());
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
            dec.push(DecStep {
                conv: FConv::load(
                    &mut planner,
                    ck,
                    &format!("decoder.convs.{}.0", i - 1),
                    ConvShape::same(n, (cs + cl) as i32, cs as i32, &[hw as i32, hw as i32], 3, 1),
                    &s,
                    &t,
                    true,
                    &ws,
                )?,
                hw,
            });
        }
        // autocast casts the logit weight to FP16
        let head_w: Vec<f32> =
            ck.take("decoder.logit.weight", &[1, DIMS[0], 1, 1])?.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
        let head_b = ck.scalar("decoder.logit.bias")?;

        let all: Vec<&FConv> = layers
            .iter()
            .flatten()
            .flat_map(|b| [Some(&b.conv1), Some(&b.conv2), Some(&b.conv3), b.downsample.as_ref()])
            .flatten()
            .chain(dec.iter().map(|s| &s.conv))
            .collect();
        // the downsample is a plain conv by design; only convs that asked for fusion and fell back need raw
        let fallback: Vec<&&FConv> = all.iter().filter(|c| c.want && c.fused.is_none()).collect();
        let unfused: Vec<String> = fallback.iter().map(|c| c.plan.shape.to_string()).collect();
        let raw_len = fallback.iter().map(|c| c.out_elems()).max().unwrap_or(0);
        let ws_need = planner.plans.values().map(|p| p.workspace()).max().unwrap_or(0);
        ensure!(ws.bytes() >= ws_need, "workspace {} B, the plans need {ws_need}", ws.bytes());

        let nan = |elems: usize| gpu.alloc_fill(elems.max(1) * 2, 0xFF);
        let cat = (0..3)
            .map(|i| nan(batch * dims[i][1] * dims[i][2] * (DIMS[i] + DIMS[i + 1])))
            .collect::<Result<Vec<_>>>()?;
        let dout = dec.iter().map(|s| nan(s.conv.out_elems())).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            batch,
            in_norm,
            stem,
            stem_bufs,
            dims,
            head_w: gpu.upload(&head_w)?,
            head_b,
            stats: gpu.alloc_fill(batch * 8, 0xFF)?,
            stream: [nan(stream_len)?, nan(stream_len)?],
            t1: nan(t1_len)?,
            t2: nan(t2_len)?,
            ds: nan(ds_len)?,
            raw: if raw_len > 0 { Some(nan(raw_len)?) } else { None },
            cat,
            p4: nan(batch * dims[3][1] * dims[3][2] * DIMS[3])?,
            dout,
            ws,
            layers,
            dec,
            unfused,
        })
    }

    /// The distinct conv plans, for the report, with whether each runs fused
    pub fn plans(&self) -> Vec<String> {
        let mut seen = std::collections::BTreeSet::new();
        for b in self.layers.iter().flatten() {
            for c in [Some(&b.conv1), Some(&b.conv2), Some(&b.conv3), b.downsample.as_ref()].into_iter().flatten() {
                seen.insert(format!("{} | {} | fused {}", c.plan.shape, c.plan.algo, c.fused.is_some()));
            }
        }
        for s in &self.dec {
            seen.insert(format!("{} | {} | fused {}", s.conv.plan.shape, s.conv.plan.algo, s.conv.fused.is_some()));
        }
        seen.into_iter().collect()
    }

    /// `y = relu(conv(x) + z + bias)` (or without the ReLU and bias for a plain conv) through cuDNN's fused
    /// call or the conv plus `v8_epilogue`
    fn conv(&self, ctx: &Ctx, c: &FConv, x: &Buf, z: Option<&Buf>, y: &Buf, act: bool) -> Result<()> {
        let mem = |b: &Buf| (b.ptr(), b.bytes());
        let ws = Some(&self.ws);
        if !act {
            return c.plan.run(ctx.dnn, mem(x), mem(&c.w), mem(y), ws);
        }
        if let (Some(f), Some(bias)) = (&c.fused, &c.bias_fused) {
            return f.run(ctx.dnn, ConvIo { x: mem(x), w: mem(&c.w), z: z.map(mem), y: mem(y) }, bias, ws);
        }
        let raw = self.raw.as_ref().ok_or_else(|| eyre!("{}: no scratch for the unfused conv", c.plan.shape))?;
        c.plan.run(ctx.dnn, mem(x), mem(&c.w), mem(raw), ws)?;
        let n2 = c.out_elems() / 2;
        ctx.gpu.launch_1d(
            "v8_epilogue",
            n2,
            &[
                y.view::<u32>(0, n2),
                raw.view::<u32>(0, n2),
                z.map_or(Arg::EMPTY, |z| z.view::<u32>(0, n2)),
                c.bias32.arg::<f32>(),
                Arg::U32(c.k as u32),
                Arg::U32(1),
            ],
        )
    }

    /// `v8_dmax` of `[B, d, h*w, c]` in `x` into channels `0..c` of `y` (`ystride` per pixel)
    fn dmax(&self, gpu: &Gpu, y: &Buf, x: &Buf, [d, h, w]: [usize; 3], c: usize, ystride: usize) -> Result<()> {
        let b = self.batch;
        gpu.launch_1d(
            "v8_dmax",
            b * h * w * c / 2,
            &[
                y.view::<u32>(0, b * h * w * ystride / 2),
                x.view::<u32>(0, b * d * h * w * c / 2),
                Arg::U32(d as u32),
                Arg::U32((h * w) as u32),
                Arg::U32(c as u32),
                Arg::U32(ystride as u32),
                // channel offset 0: each level's max is the concatenation's first block (`[skip, up]`)
                Arg::U32(0),
            ],
        )
    }

    /// Queues one forward: the tiles `desc` describes (`[batch, DESC]`) from the resident 24-layer
    /// `vol` (`[24, height, width]`), and each real tile's probability into its slot of `prob`
    /// (`[slots, 64 * 64]` float32)
    pub fn forward(
        &self,
        ctx: &Ctx,
        vol: (&Buf, usize, usize),
        desc: Arg,
        prob: &Buf,
        sigmoid16: bool,
        mut hook: Option<StageHook<'_>>,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let b = self.batch;
        ensure!(
            matches!(desc, Arg::Slice(_, len) if len as usize == b * DESC),
            "descriptor view {desc:?} for batch {b}"
        );
        let (vbuf, height, width) = vol;
        let [gamma, beta] = self.in_norm;
        gpu.launch(
            "v8_stats",
            (b as u32, 1, 1),
            (256, 1, 1),
            0,
            &[
                self.stats.view::<[f32; 2]>(0, b),
                vbuf.arg::<u8>(),
                desc,
                Arg::U32(height as u32),
                Arg::U32(width as u32),
                Arg::F32(gamma),
                Arg::F32(beta),
            ],
        )?;
        if let Some(h) = hook.as_mut() {
            h("stats", &self.stats, [b, 1, 1, 1, 2])?;
        }
        let [sw, sb, dc, db, hc, hb] = &self.stem_bufs;
        gpu.launch(
            "v8_stem",
            (((PH / 8) * (PH / 8)) as u32, SD as u32, b as u32),
            (256, 1, 1),
            0,
            &[
                self.stream[0].view::<u16>(0, b * SD * PH * PH * SC),
                vbuf.arg::<u8>(),
                desc,
                self.stats.view::<[f32; 2]>(0, b),
                sw.arg::<f32>(),
                sb.arg::<f32>(),
                dc.arg::<u32>(),
                db.arg::<i32>(),
                hc.arg::<u32>(),
                hb.arg::<i32>(),
                Arg::U32(height as u32),
                Arg::U32(width as u32),
                Arg::U32(self.stem.nh as u32),
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
                        self.conv(ctx, dsc, x, None, &self.ds, false)?;
                        &self.ds
                    }
                    None => x,
                };
                self.conv(ctx, &blk.conv1, x, None, &self.t1, true)?;
                self.conv(ctx, &blk.conv2, &self.t1, None, &self.t2, true)?;
                self.conv(ctx, &blk.conv3, &self.t2, Some(residual), y, true)?;
                cur = 1 - cur;
            }
            let out = &self.stream[cur];
            if let Some(h) = hook.as_mut() {
                let [d, hh, ww] = self.dims[l];
                h(&format!("layer{}", l + 1), out, [b, d, hh, ww, DIMS[l]])?;
            }
            if l < 3 {
                self.dmax(gpu, &self.cat[l], out, self.dims[l], DIMS[l], DIMS[l] + DIMS[l + 1])?;
            } else {
                self.dmax(gpu, &self.p4, out, self.dims[l], DIMS[l], DIMS[l])?;
            }
        }

        // decoder
        for (i, step) in self.dec.iter().enumerate() {
            let lvl = 2 - i;
            let (low, lc) = if i == 0 { (&self.p4, DIMS[3]) } else { (&self.dout[i - 1], DIMS[lvl + 1]) };
            let h = step.hw / 2;
            let ystride = DIMS[lvl] + DIMS[lvl + 1];
            gpu.launch_1d(
                "v8_up2cat",
                b * step.hw * step.hw * lc / 2,
                &[
                    self.cat[lvl].view::<u32>(0, b * step.hw * step.hw * ystride / 2),
                    low.view::<u32>(0, b * h * h * lc / 2),
                    Arg::U32(h as u32),
                    Arg::U32(h as u32),
                    Arg::U32(lc as u32),
                    Arg::U32(ystride as u32),
                    Arg::U32(DIMS[lvl] as u32),
                ],
            )?;
            self.conv(ctx, &step.conv, &self.cat[lvl], None, &self.dout[i], true)?;
        }

        gpu.launch_1d(
            "v8_head",
            b * PIX,
            &[
                prob.arg::<f32>(),
                self.dout[2].view::<u16>(0, b * PIX * DIMS[0]),
                self.head_w.arg::<f32>(),
                Arg::F32(self.head_b),
                desc,
                Arg::U32(DIMS[0] as u32),
                Arg::U32(sigmoid16 as u32),
            ],
        )
    }

    /// The decoder-side activations of the last forward, channels-last, for the P0 gate: each level's
    /// concatenation (`[skip max, upsampled]`), the deepest max and each decoder step's output. The
    /// encoder's buffers are reused across layers, so only these survive a forward
    pub fn taps(&self) -> Vec<Tap<'_>> {
        let b = self.batch;
        let mut v = Vec::new();
        for l in 0..3 {
            let [_, h, w] = self.dims[l];
            v.push(Tap { name: format!("cat{l}"), buf: &self.cat[l], shape: vec![b, h, w, DIMS[l] + DIMS[l + 1]] });
        }
        v.push(Tap { name: "max4".into(), buf: &self.p4, shape: vec![b, 8, 8, DIMS[3]] });
        for (i, s) in self.dec.iter().enumerate() {
            v.push(Tap { name: format!("dec{}", 2 - i), buf: &self.dout[i], shape: vec![b, s.hw, s.hw, DIMS[2 - i]] });
        }
        v
    }
}
