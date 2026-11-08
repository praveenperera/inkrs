//! The 3D U-Net forward at batch 1, built from an export's layer program
//!
//! Each resolution level owns its activation buffers, sized for the widest tensor at that level:
//! two for block inputs and outputs (`x`, `y`), raw conv outputs (`r`, `r2`), the normalised conv1
//! output (`h`, which also holds the transposed conv's GEMM output in the decoder), the pooled skip
//! input (`pa`) and the skip projection (`rs`) where blocks need them, and the decoder's concat
//! buffer (`cat`, `[up, skip]`). The last block of each encoder stage writes its output twice, packed
//! for the next stage and into the second half of that level's `cat`, so the concat costs no copy

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use color_eyre::eyre::{Result, WrapErr, ensure};
use half::bf16;

use super::{Contract, ConvSpec};
use crate::dnn::{Blas, ConvPlan, ConvShape, Dnn, Dtype, Pick};
use crate::gpu::{Arg, Buf, Gpu};
use crate::weights::Checkpoint;

/// The stem's input channels when cuDNN refuses a one-channel bf16 3D conv
const STEM_PAD: usize = 8;
/// Upper bound on [`crate::unet3d`]'s InstanceNorm partial-sum chunks per tensor
const MAX_CHUNKS: usize = 4096;

/// What the forward needs from the caller
pub struct Ctx<'a> {
    pub gpu: &'a Gpu,
    pub dnn: &'a Dnn,
    pub blas: &'a Blas,
    pub pick: Pick,
    pub max_workspace: usize,
}

fn bf_bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| bf16::from_f32(x).to_bits()).collect()
}

/// Values as autocast casts them: rounded to bf16, kept as float32
fn bf_round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect()
}

/// An InstanceNorm's float32 affine
struct Norm {
    w: Buf,
    b: Buf,
    eps: f32,
}

/// A planned conv: channels-last bf16 weight `[K, kd, kh, kw, C]`, bf16-rounded bias, the norm and
/// activation after it
struct Conv {
    plan: Rc<ConvPlan>,
    w: Buf,
    bias: Option<Buf>,
    norm: Option<Norm>,
    leaky: Option<f32>,
    cout: usize,
}

struct Planner<'a> {
    ctx: &'a Ctx<'a>,
    plans: HashMap<ConvShape, Rc<ConvPlan>>,
}

impl Planner<'_> {
    fn plan(&mut self, shape: ConvShape) -> Result<Rc<ConvPlan>> {
        if let Some(p) = self.plans.get(&shape) {
            return Ok(p.clone());
        }
        let c = self.ctx;
        let plan = Rc::new(ConvPlan::typed(c.gpu, c.dnn, shape.clone(), Dtype::Bf16, c.pick, c.max_workspace)?);
        self.plans.insert(shape, plan.clone());
        Ok(plan)
    }

    /// `spec` on an input of edge `edge` with `c_pad >= spec.cin` channels
    fn conv(&mut self, ck: &Checkpoint, spec: &ConvSpec, edge: usize, c_pad: usize) -> Result<Conv> {
        let gpu = self.ctx.gpu;
        let (k, c, ks) = (spec.cout, spec.cin, spec.kernel[0]);
        let shape = ConvShape::same(1, c_pad as i32, k as i32, &[edge as i32; 3], ks as i32, spec.stride[0] as i32);
        let plan = self.plan(shape)?;
        let taps = ks * ks * ks;
        let w = ck.take(&spec.w, &[k, c, ks, ks, ks])?;
        let mut ndhwc = vec![0f32; k * taps * c_pad];
        for ko in 0..k {
            for ci in 0..c {
                for t in 0..taps {
                    ndhwc[(ko * taps + t) * c_pad + ci] = w[(ko * c + ci) * taps + t];
                }
            }
        }
        let bias = match &spec.b {
            Some(key) => Some(gpu.upload(&bf_round(&ck.take(key, &[k])?))?),
            None => None,
        };
        let norm = match &spec.norm {
            Some(n) => {
                Some(Norm { w: gpu.upload(&ck.take(&n.w, &[k])?)?, b: gpu.upload(&ck.take(&n.b, &[k])?)?, eps: n.eps })
            }
            None => None,
        };
        Ok(Conv { plan, w: gpu.upload(&bf_bits(&ndhwc))?, bias, norm, leaky: spec.leaky, cout: k })
    }
}

/// One BasicBlockD
struct Block {
    conv1: Conv,
    conv2: Conv,
    pool: bool,
    proj: Option<Conv>,
    leaky: f32,
    /// Level of the block's input and output
    lin: usize,
    lout: usize,
}

/// One decoder stage, writing level `level`
struct Dec {
    /// GEMM weight `[cin, 8 * cout]` bf16, column `tap * cout + co`
    tw: Buf,
    tb: Buf,
    tcin: usize,
    convs: Vec<Conv>,
    level: usize,
}

/// One resolution level's buffers
struct Level {
    edge: usize,
    x: Buf,
    y: Buf,
    r: Buf,
    r2: Buf,
    h: Buf,
    pa: Option<Buf>,
    rs: Option<Buf>,
    cat: Option<Buf>,
    /// Channels of the encoder output at this level
    c: usize,
}

/// Where a tensor sits for a tap: its buffer, edge and channels
pub struct Tap<'a> {
    pub name: String,
    pub buf: &'a Buf,
    pub edge: usize,
    pub channels: usize,
    /// Channel stride of the buffer (the concat buffer holds two tensors)
    pub stride: usize,
    pub offset: usize,
}

/// The network on the GPU, at batch 1
pub struct Model {
    pub patch: usize,
    /// Channel stride of the input buffer [`Model::input`] (1, or [`STEM_PAD`])
    pub in_c: usize,
    stem: Vec<Conv>,
    blocks: Vec<Block>,
    decs: Vec<Dec>,
    head_w: Buf,
    head_b: f32,
    head_c: usize,
    levels: Vec<Level>,
    /// The normalised bf16 patch, `[p^3, in_c]`
    pub input: Buf,
    partial: Buf,
    stats: [Buf; 3],
    ws: Buf,
    /// (level, which buffer: 'x' or 'y') of the last decoder output, the head's input. The last
    /// stage ends in `y` after an odd number of convs and in `x` after an even one, so the head
    /// must read the buffer the forward wrote rather than assume one
    out: (usize, char),
    tap_names: Vec<(String, usize, char)>,
    /// When set, every InstanceNorm's `[mean, invstd]` in call order, downloaded as computed (`p0-ink3d`
    /// runs the forward twice and requires the two traces to be byte-identical)
    stats_trace: RefCell<Option<Vec<Vec<[f32; 2]>>>>,
}

impl Model {
    /// Plans every conv, uploads the weights and allocates the buffers; every checkpoint tensor must be
    /// taken (the caller checks with [`Checkpoint::finish`])
    pub fn new(ctx: &Ctx, ck: &Checkpoint, contract: &Contract) -> Result<Self> {
        let gpu = ctx.gpu;
        super::check_device(gpu.inner.compute_capability(), gpu.inner.ptx_tier())?;
        let prog = &contract.program;
        let p = contract.patch();
        let mut planner = Planner { ctx, plans: HashMap::new() };

        // stem: one channel, or padded when cuDNN refuses it
        let mut stem = Vec::new();
        let mut in_c = 1;
        for (i, s) in prog.stem.iter().enumerate() {
            let c_pad = if i == 0 {
                match planner.plan(ConvShape::same(1, 1, s.cout as i32, &[p as i32; 3], s.kernel[0] as i32, 1)) {
                    Ok(_) => 1,
                    Err(e) => {
                        eprintln!("ink3d: stem conv on 1 channel: {e:#}; padding to {STEM_PAD}");
                        in_c = STEM_PAD;
                        STEM_PAD
                    }
                }
            } else {
                s.cin
            };
            stem.push(planner.conv(ck, s, p, c_pad)?);
        }

        // encoder
        let mut blocks = Vec::new();
        let (mut edge, mut level) = (p, 0usize);
        let mut edges = vec![p];
        // widest tensor per level, and which levels need the skip buffers
        let mut cmax: Vec<usize> = vec![prog.stem.last().map_or(1, |s| s.cout)];
        let mut need_pa: Vec<usize> = vec![0];
        let mut need_rs = vec![false];
        for stage in &prog.stages {
            for b in stage {
                let lin = level;
                if b.conv1.stride[0] == 2 {
                    edge /= 2;
                    level += 1;
                    edges.push(edge);
                    cmax.push(0);
                    need_pa.push(0);
                    need_rs.push(false);
                }
                let lout = level;
                let conv1 = planner.conv(ck, &b.conv1, edges[lin], b.conv1.cin)?;
                let conv2 = planner.conv(ck, &b.conv2, edge, b.conv2.cin)?;
                let proj = match &b.skip.proj {
                    Some(pj) => Some(planner.conv(ck, pj, edge, pj.cin)?),
                    None => None,
                };
                if b.skip.pool == 2 {
                    need_pa[lout] = need_pa[lout].max(b.conv1.cin);
                }
                need_rs[lout] |= proj.is_some();
                cmax[lin] = cmax[lin].max(b.conv1.cin);
                cmax[lout] = cmax[lout].max(b.conv2.cout);
                blocks.push(Block { conv1, conv2, pool: b.skip.pool == 2, proj, leaky: b.leaky, lin, lout });
            }
        }
        let n_levels = level + 1;
        ensure!(prog.decoder.len() + 1 == n_levels, "{} decoder stages for {n_levels} levels", prog.decoder.len());

        // decoder
        let mut decs = Vec::new();
        for (d, ds) in prog.decoder.iter().enumerate() {
            let lv = n_levels - 2 - d;
            let t = &ds.transp;
            let (ci, co) = (t.cin, t.cout);
            let w = ck.take(&t.w, &[ci, co, 2, 2, 2])?;
            let mut g = vec![0f32; ci * 8 * co];
            for i in 0..ci {
                for o in 0..co {
                    for tap in 0..8 {
                        g[i * 8 * co + tap * co + o] = w[(i * co + o) * 8 + tap];
                    }
                }
            }
            let tb = bf_round(&ck.take(t.b.as_ref().expect("checked by the contract"), &[co])?);
            let mut convs = Vec::new();
            for cv in &ds.convs {
                convs.push(planner.conv(ck, cv, edges[lv], cv.cin)?);
                cmax[lv] = cmax[lv].max(cv.cout);
            }
            cmax[lv] = cmax[lv].max(co);
            decs.push(Dec { tw: gpu.upload(&bf_bits(&g))?, tb: gpu.upload(&tb)?, tcin: ci, convs, level: lv });
        }
        let h = &prog.head;
        let head_w = gpu.upload(&bf_round(&ck.take(&h.w, &[1, h.cin, 1, 1, 1])?))?;
        let head_b = bf_round(&ck.take(h.b.as_ref().expect("checked by the contract"), &[1])?)[0];

        // buffers
        let enc_c: Vec<usize> = {
            let mut v = vec![0; n_levels];
            for b in &blocks {
                v[b.lout] = b.conv2.cout;
            }
            v
        };
        let mut levels = Vec::new();
        for l in 0..n_levels {
            let vox = edges[l].pow(3);
            let n = vox * cmax[l];
            u32_extent("level buffer", n)?;
            let alloc = |n: usize| gpu.alloc_fill(n * 2, 0xFF);
            let cat = if l + 1 < n_levels {
                u32_extent("concat buffer", vox * 2 * enc_c[l])?;
                Some(alloc(vox * 2 * enc_c[l])?)
            } else {
                None
            };
            levels.push(Level {
                edge: edges[l],
                x: alloc(n)?,
                y: alloc(n)?,
                r: alloc(n)?,
                r2: alloc(n)?,
                h: alloc(n)?,
                pa: if need_pa[l] > 0 { Some(alloc(vox * need_pa[l])?) } else { None },
                rs: if need_rs[l] { Some(alloc(n)?) } else { None },
                cat,
                c: enc_c[l],
            });
        }
        // the decoder's GEMM output lives in `h` of the level it writes
        for d in &decs {
            ensure!(cmax[d.level] >= d.convs[0].cout.min(cmax[d.level]), "level {} too narrow", d.level);
        }
        let max_c = cmax.iter().copied().max().unwrap_or(1).max(prog.head.cin);
        ensure!(max_c <= 512, "{max_c} channels; the norm kernels take at most 512");
        let ws_bytes = planner.plans.values().map(|p| p.workspace()).max().unwrap_or(0);
        let input = gpu.alloc_fill(p.pow(3) * in_c * 2, 0)?;
        let stats = [gpu.alloc(max_c * 8)?, gpu.alloc(max_c * 8)?, gpu.alloc(max_c * 8)?];
        let mut tap_names = vec![("input".to_string(), 0, 'i')];
        if !stem.is_empty() {
            tap_names.push(("stem".into(), 0, 's'));
        }
        Ok(Self {
            patch: p,
            in_c,
            stem,
            blocks,
            decs,
            head_w,
            head_b,
            head_c: prog.head.cin,
            levels,
            input,
            partial: gpu.alloc(MAX_CHUNKS * max_c * 8)?,
            stats,
            ws: gpu.alloc(ws_bytes.max(1))?,
            out: (0, 'y'),
            tap_names,
            stats_trace: RefCell::new(None),
        })
    }

    /// Every distinct conv plan with its algorithm, for the report
    pub fn plans(&self) -> Vec<String> {
        let mut seen = Vec::new();
        let all = self
            .stem
            .iter()
            .chain(self.blocks.iter().flat_map(|b| [&b.conv1, &b.conv2].into_iter().chain(b.proj.iter())));
        for c in all.chain(self.decs.iter().flat_map(|d| d.convs.iter())) {
            let s = format!("{} | {}", c.plan.shape, c.plan.algo);
            if !seen.contains(&s) {
                seen.push(s);
            }
        }
        seen
    }

    /// The head's input: the last decoder output `[p^3, head_c]` bf16, and the head weight and bias
    pub fn head(&self) -> (&Buf, usize, &Buf, f32) {
        let (level, which) = self.out;
        let l = &self.levels[level];
        let buf = if which == 'x' { &l.x } else { &l.y };
        (buf, self.head_c, &self.head_w, self.head_b)
    }

    fn conv(&self, ctx: &Ctx, c: &Conv, x: &Buf, y: &Buf) -> Result<()> {
        c.plan.run(ctx.dnn, (x.ptr(), x.bytes()), (c.w.ptr(), c.w.bytes()), (y.ptr(), y.bytes()), Some(&self.ws))
    }

    /// InstanceNorm statistics of `x`, `[rows, cols]`, after `bias`, into `self.stats[slot]`
    fn stats(
        &self,
        gpu: &Gpu,
        x: &Buf,
        bias: Option<&Buf>,
        (rows, cols): (usize, usize),
        slot: usize,
        eps: f32,
    ) -> Result<()> {
        // u3_finalize counts rows in float32, exact only up to 2^24 (a 256^3 patch)
        ensure!(rows <= 1 << 24, "InstanceNorm over {rows} rows: float32 counts are exact only to 2^24");
        let pairs = cols / 2;
        let lanes = (256 / pairs).max(1);
        let chunk_rows = rows.div_ceil(MAX_CHUNKS).max(lanes);
        let chunks = rows.div_ceil(chunk_rows);
        let bias = bias.map_or(Arg::EMPTY, |b| b.arg::<f32>());
        gpu.launch(
            "u3_partial",
            (chunks as u32, 1, 1),
            ((pairs * lanes) as u32, 1, 1),
            0,
            &[
                self.partial.view::<[f32; 2]>(0, chunks * cols),
                x.view::<u32>(0, rows * pairs),
                bias,
                Arg::U32(rows as u32),
                Arg::U32(cols as u32),
                Arg::U32(lanes as u32),
                Arg::U32(chunk_rows as u32),
            ],
        )?;
        gpu.launch(
            "u3_finalize",
            (cols as u32, 1, 1),
            (256, 1, 1),
            0,
            &[
                self.stats[slot].view::<[f32; 2]>(0, cols),
                self.partial.view::<[f32; 2]>(0, chunks * cols),
                Arg::U32(rows as u32),
                Arg::U32(cols as u32),
                Arg::U32(chunks as u32),
                Arg::U32(chunk_rows as u32),
                Arg::F32(eps),
            ],
        )?;
        if let Some(trace) = self.stats_trace.borrow_mut().as_mut() {
            trace.push(gpu.download_range(&self.stats[slot], 0, cols)?);
        }
        Ok(())
    }

    /// Starts recording every InstanceNorm's statistics, or stops and returns what was recorded
    pub fn trace_stats(&self, on: bool) -> Option<Vec<Vec<[f32; 2]>>> {
        self.stats_trace.replace(on.then(Vec::new))
    }

    /// Statistics and apply of `c`'s output `raw` into `out` (and `out2`), with an optional residual;
    /// with `pool = (pa, edge)` it also writes the `AvgPool3d(2, 2)` of the result into `pa`, `edge^3` pixels
    #[allow(clippy::too_many_arguments)]
    fn norm_apply(
        &self,
        gpu: &Gpu,
        c: &Conv,
        raw: &Buf,
        vox: usize,
        residual: Residual,
        leaky: Option<f32>,
        out: (&Buf, usize, usize),
        out2: Option<(&Buf, usize, usize)>,
        pool: Option<(&Buf, usize)>,
    ) -> Result<()> {
        let n = c.norm.as_ref().expect("checked by the contract");
        let cols = c.cout;
        self.stats(gpu, raw, c.bias.as_ref(), (vox, cols), 0, n.eps)?;
        let pairs = cols / 2;
        let empty_stats = Arg::EMPTY;
        let (res, rraw, rstats, rw, rb) = match residual {
            Residual::None => (Arg::EMPTY, Arg::EMPTY, empty_stats, Arg::EMPTY, Arg::EMPTY),
            Residual::Plain(r) => (r.view::<u32>(0, vox * pairs), Arg::EMPTY, empty_stats, Arg::EMPTY, Arg::EMPTY),
            Residual::Normed(r, pc) => {
                let pn = pc.norm.as_ref().expect("checked by the contract");
                self.stats(gpu, r, None, (vox, cols), 1, pn.eps)?;
                (
                    Arg::EMPTY,
                    r.view::<u32>(0, vox * pairs),
                    self.stats[1].view::<[f32; 2]>(0, cols),
                    pn.w.arg::<f32>(),
                    pn.b.arg::<f32>(),
                )
            }
        };
        let (o2, s2, f2) = match out2 {
            Some((b, s, f)) => (b.arg::<u32>(), s, f),
            None => (Arg::EMPTY, 0, 0),
        };
        let tail = [
            Arg::U32(cols as u32),
            Arg::U32(leaky.is_some() as u32),
            Arg::F32(leaky.unwrap_or(0.0)),
            Arg::U32(out.1 as u32),
            Arg::U32(out.2 as u32),
            Arg::U32(s2 as u32),
            Arg::U32(f2 as u32),
        ];
        let inputs = [
            raw.view::<u32>(0, vox * pairs),
            c.bias.as_ref().map_or(Arg::EMPTY, |b| b.arg::<f32>()),
            self.stats[0].view::<[f32; 2]>(0, cols),
            n.w.arg::<f32>(),
            n.b.arg::<f32>(),
            res,
            rraw,
            rstats,
            rw,
            rb,
        ];
        if let Some((pa, edge)) = pool {
            let pv = edge.pow(3);
            ensure!(8 * pv == vox, "pooling {vox} voxels into {edge}^3");
            let e = Arg::U32(edge as u32);
            let args: Vec<Arg> = [out.0.arg::<u32>(), o2, pa.view::<u32>(0, pv * pairs)]
                .into_iter()
                .chain(inputs)
                .chain([e, e, e])
                .chain(tail)
                .collect();
            return gpu.launch_1d("u3_apply_pool", pv * pairs, &args);
        }
        let args: Vec<Arg> =
            [out.0.arg::<u32>(), o2].into_iter().chain(inputs).chain([Arg::U32(vox as u32)]).chain(tail).collect();
        gpu.launch_1d("u3_apply", vox * pairs, &args)
    }

    /// The forward on [`Model::input`]; the result is [`Model::head`]'s input
    pub fn forward(&mut self, ctx: &Ctx) -> Result<()> {
        self.forward_with(ctx, &mut |_| Ok(()))
    }

    /// [`Model::forward`], calling `on_tap` with each tap as soon as it is written: later layers reuse
    /// the per-level buffers (the decoder overwrites encoder outputs), so a tap read after the forward
    /// may hold another tensor
    pub fn forward_with(&mut self, ctx: &Ctx, on_tap: &mut dyn FnMut(Tap<'_>) -> Result<()>) -> Result<()> {
        let gpu = ctx.gpu;
        let lv = &self.levels;
        let mut tap_names = vec![("input".to_string(), 0, 'i')];
        on_tap(self.tap("input", 0, 'i'))?;

        // stem into level 0's `x`
        let v0 = lv[0].edge.pow(3);
        let mut cur_in: &Buf = &self.input;
        for (i, s) in self.stem.iter().enumerate() {
            let dst = if i % 2 == 0 { &lv[0].x } else { &lv[0].y };
            self.conv(ctx, s, cur_in, &lv[0].r)?;
            self.norm_apply(gpu, s, &lv[0].r, v0, Residual::None, s.leaky, (dst, s.cout / 2, 0), None, None)?;
            cur_in = dst;
        }
        if !self.stem.is_empty() {
            let which = if (self.stem.len() - 1).is_multiple_of(2) { 'x' } else { 'y' };
            on_tap(self.tap("stem", 0, which))?;
            tap_names.push(("stem".into(), 0, which));
        }

        // encoder
        let mut cur_name = if self.stem.len() % 2 == 1 { 'x' } else { 'y' };
        if self.stem.is_empty() {
            cur_name = 'i';
        }
        let pick = |l: usize, n: char| -> &Buf {
            match n {
                'x' => &lv[l].x,
                'y' => &lv[l].y,
                _ => &self.input,
            }
        };
        // set when the previous block's apply already wrote this block's pooled skip input
        let mut pooled = false;
        for (bi, b) in self.blocks.iter().enumerate() {
            let input = pick(b.lin, cur_name);
            let out_name = if b.lin == b.lout && cur_name == 'x' { 'y' } else { 'x' };
            let (lo, vox) = (&lv[b.lout], lv[b.lout].edge.pow(3));
            self.conv(ctx, &b.conv1, input, &lo.r)?;
            self.norm_apply(
                gpu,
                &b.conv1,
                &lo.r,
                vox,
                Residual::None,
                b.conv1.leaky,
                (&lo.h, b.conv1.cout / 2, 0),
                None,
                None,
            )?;
            self.conv(ctx, &b.conv2, &lo.h, &lo.r2)?;
            // skip source
            let src: &Buf = if b.pool && pooled {
                lo.pa.as_ref().expect("allocated for pooled skips")
            } else if b.pool {
                let pa = lo.pa.as_ref().expect("allocated for pooled skips");
                let cin = b.conv1.plan.shape.c as usize;
                gpu.launch_1d(
                    "u3_pool2",
                    vox * cin / 2,
                    &[
                        pa.view::<u32>(0, vox * cin / 2),
                        input.view::<u32>(0, vox * 8 * cin / 2),
                        Arg::U32(lo.edge as u32),
                        Arg::U32(lo.edge as u32),
                        Arg::U32(lo.edge as u32),
                        Arg::U32(cin as u32),
                    ],
                )?;
                pa
            } else {
                input
            };
            let residual = match &b.proj {
                Some(pc) => {
                    let rs = lo.rs.as_ref().expect("allocated for projected skips");
                    self.conv(ctx, pc, src, rs)?;
                    Residual::Normed(rs, pc)
                }
                None => Residual::Plain(src),
            };
            let last_in_level = self.blocks.get(bi + 1).is_none_or(|n| n.lout != b.lout);
            let c = b.conv2.cout;
            let out2 = if last_in_level { lo.cat.as_ref().map(|cat| (cat, c, c / 2)) } else { None };
            // the next stage's pooled skip is this output, so pool it while it is in registers
            let pool = match self.blocks.get(bi + 1) {
                Some(n) if n.pool && n.lin == b.lout => {
                    ensure!(
                        n.conv1.plan.shape.c as usize == c,
                        "pooled skip of {c} channels into {}",
                        n.conv1.plan.shape.c
                    );
                    let pa = lv[n.lout].pa.as_ref().expect("allocated for pooled skips");
                    Some((pa, lv[n.lout].edge))
                }
                _ => None,
            };
            pooled = pool.is_some();
            self.norm_apply(
                gpu,
                &b.conv2,
                &lo.r2,
                vox,
                residual,
                Some(b.leaky),
                (pick(b.lout, out_name), c / 2, 0),
                out2,
                pool,
            )?;
            cur_name = out_name;
            if last_in_level {
                let name = format!("enc{}", b.lout);
                on_tap(self.tap(&name, b.lout, out_name))?;
                tap_names.push((name, b.lout, out_name));
            }
        }

        // decoder: lres is the bottleneck output
        let mut lres_level = self.blocks.last().map_or(0, |b| b.lout);
        let mut lres_name = cur_name;
        for (d, dec) in self.decs.iter().enumerate() {
            let l = dec.level;
            let (lo, vox) = (&lv[l], lv[l].edge.pow(3));
            let lres = pick(lres_level, lres_name);
            let co = dec.convs[0].plan.shape.c as usize / 2;
            let vin = lv[lres_level].edge.pow(3);
            ctx.blas.gemm_bf16(
                (vin, 8 * co, dec.tcin),
                (lres.ptr(), lres.bytes()),
                (dec.tw.ptr(), dec.tw.bytes()),
                (lo.h.ptr(), lo.h.bytes()),
            )?;
            let cat = lo.cat.as_ref().expect("allocated below the bottleneck");
            gpu.launch_1d(
                "u3_up",
                vox * co / 2,
                &[
                    cat.arg::<u32>(),
                    lo.h.view::<u32>(0, vin * 8 * co / 2),
                    dec.tb.arg::<f32>(),
                    Arg::U32(lo.edge as u32),
                    Arg::U32(lo.edge as u32),
                    Arg::U32(lo.edge as u32),
                    Arg::U32(co as u32),
                    Arg::U32(co as u32),
                    Arg::U32(0),
                ],
            )?;
            let mut src: &Buf = cat;
            for (i, cv) in dec.convs.iter().enumerate() {
                let dst = if dec_dst(i) == 'y' { &lo.y } else { &lo.x };
                self.conv(ctx, cv, src, &lo.r)?;
                self.norm_apply(gpu, cv, &lo.r, vox, Residual::None, cv.leaky, (dst, cv.cout / 2, 0), None, None)?;
                src = dst;
            }
            lres_level = l;
            lres_name = dec_dst(dec.convs.len() - 1);
            let name = format!("dec{d}");
            on_tap(self.tap(&name, l, lres_name))?;
            tap_names.push((name, l, lres_name));
        }
        ensure!(lres_level == 0, "the last decoder stage must end at level 0, not {lres_level}");
        self.out = (0, lres_name);
        self.tap_names = tap_names;
        Ok(())
    }

    /// The tensors of the last forward: input, stem, each encoder level's output, each decoder stage.
    /// Only the last decoder stage is sure to still hold its tensor; [`Model::forward_with`] sees each
    /// one when it is written
    pub fn taps(&self) -> Vec<Tap<'_>> {
        self.tap_names.iter().map(|(name, l, which)| self.tap(name, *l, *which)).collect()
    }

    fn tap(&self, name: &str, l: usize, which: char) -> Tap<'_> {
        let lv = &self.levels[l];
        let buf = match which {
            'i' => &self.input,
            'x' => &lv.x,
            _ => &lv.y,
        };
        let channels = if which == 'i' {
            1
        } else if name == "stem" {
            self.stem.last().map_or(1, |s| s.cout)
        } else if let Some(d) = name.strip_prefix("dec") {
            self.decs[d.parse::<usize>().unwrap_or(0)].convs.last().map_or(0, |c| c.cout)
        } else {
            lv.c
        };
        let stride = if which == 'i' { self.in_c } else { channels };
        Tap { name: name.to_string(), buf, edge: lv.edge, channels, stride, offset: 0 }
    }

    /// Runs the InstanceNorm statistics kernels on channels that break naive or shifted sums (an outlier
    /// as the first value, a constant channel, a large common offset, spread values) and checks them
    /// against a float64 two-pass on the host. `p0-ink3d` runs it before its forward
    pub fn check_norm_stats(&self, gpu: &Gpu) -> Result<()> {
        let (rows, cols, eps) = (128usize.pow(3), 32usize, 1e-5f32);
        ensure!(self.stats[0].bytes() >= cols * 8, "stats buffer narrower than the check");
        let value = |r: usize, c: usize| -> f32 {
            match c {
                // Sol's case: shifted sums from the first value went negative and doubled invstd
                0 => {
                    if r == 0 {
                        16.0
                    } else if r % 2 == 1 {
                        1.0078125
                    } else {
                        1.0
                    }
                }
                1 => 3.0,
                2 => 100.0 + (r % 7) as f32 * 0.5,
                _ => ((r.wrapping_mul(2_654_435_761) + c * 97) % 1000) as f32 / 250.0 - 2.0,
            }
        };
        let host: Vec<bf16> = (0..rows * cols).map(|i| bf16::from_f32(value(i / cols, i % cols))).collect();
        let x = gpu.upload(&host)?;
        self.stats(gpu, &x, None, (rows, cols), 0, eps)?;
        let got: Vec<[f32; 2]> = gpu.download_range(&self.stats[0], 0, cols)?;
        // the merge order is fixed, so a second run must give the same bits
        self.stats(gpu, &x, None, (rows, cols), 0, eps)?;
        let again: Vec<[f32; 2]> = gpu.download_range(&self.stats[0], 0, cols)?;
        let bits = |v: &[[f32; 2]]| v.iter().flat_map(|p| [p[0].to_bits(), p[1].to_bits()]).collect::<Vec<_>>();
        ensure!(bits(&got) == bits(&again), "norm statistics self-check: a second run gave different bits");
        for (c, &[mean, invstd]) in got.iter().enumerate() {
            let col = || (0..rows).map(|r| host[r * cols + c].to_f32() as f64);
            let m = col().sum::<f64>() / rows as f64;
            let var = col().map(|v| (v - m) * (v - m)).sum::<f64>() / rows as f64;
            let want = 1.0 / (var + eps as f64).sqrt();
            let sd = var.sqrt();
            ensure!(
                (mean as f64 - m).abs() <= 1e-5 * (m.abs() + sd) && ((invstd as f64 - want) / want).abs() <= 1e-4,
                "norm statistics self-check, channel {c}: mean {mean} invstd {invstd}, float64 gives {m} {want}"
            );
        }
        Ok(())
    }

    /// Runs `u3_apply_pool` against `u3_apply` then `u3_pool2` on a synthetic 16^3 x 32 tensor, with a
    /// plain residual and with a normalised one, and requires equal bytes in `out`, `out2` and the pooled
    /// tensor: the fused kernel replaces the pair only if it is bit-identical. `p0-ink3d` runs it
    pub fn check_apply_pool(&self, gpu: &Gpu) -> Result<()> {
        let (edge, cols) = (16usize, 32usize);
        let (vox, pairs, pv) = (edge.pow(3), cols / 2, (edge / 2).pow(3));
        let wave = |i: usize, k: usize| ((i.wrapping_mul(2_654_435_761) + k * 7919) % 2000) as f32 / 400.0 - 2.5;
        let bf = |k: usize, n: usize| -> Vec<bf16> { (0..n).map(|i| bf16::from_f32(wave(i, k))).collect() };
        let f32s = |k: usize, n: usize, lo: f32| -> Vec<f32> { (0..n).map(|i| lo + wave(i, k).abs()).collect() };
        let stats_of =
            |k: usize| -> Vec<[f32; 2]> { (0..cols).map(|c| [wave(c, k) * 0.1, 0.5 + wave(c, k + 1).abs()]).collect() };
        let x = gpu.upload(&bf(1, vox * cols))?;
        let res = gpu.upload(&bf(2, vox * cols))?;
        let bias = gpu.upload(&bf(3, cols).iter().map(|v| v.to_f32()).collect::<Vec<f32>>())?;
        let (st, rst) = (gpu.upload(&stats_of(4))?, gpu.upload(&stats_of(6))?);
        let (w, b) = (gpu.upload(&f32s(8, cols, 0.2))?, gpu.upload(&f32s(9, cols, -1.0))?);
        let (rw, rb) = (gpu.upload(&f32s(10, cols, 0.2))?, gpu.upload(&f32s(11, cols, -1.0))?);
        for normed in [false, true] {
            let (rplain, rraw, rstats, rwa, rba) = if normed {
                (Arg::EMPTY, res.arg::<u32>(), rst.arg::<[f32; 2]>(), rw.arg::<f32>(), rb.arg::<f32>())
            } else {
                (res.arg::<u32>(), Arg::EMPTY, Arg::EMPTY, Arg::EMPTY, Arg::EMPTY)
            };
            let inputs = [
                x.arg::<u32>(),
                bias.arg::<f32>(),
                st.arg::<[f32; 2]>(),
                w.arg::<f32>(),
                b.arg::<f32>(),
                rplain,
                rraw,
                rstats,
                rwa,
                rba,
            ];
            // out packed, out2 the second half of a concat buffer, as a stage's last block writes them
            let tail = [
                Arg::U32(cols as u32),
                Arg::U32(1),
                Arg::F32(0.01),
                Arg::U32(pairs as u32),
                Arg::U32(0),
                Arg::U32(cols as u32),
                Arg::U32(pairs as u32),
            ];
            let mk = || -> Result<[Buf; 3]> {
                Ok([
                    gpu.alloc_fill(vox * cols * 2, 0xa5)?,
                    gpu.alloc_fill(vox * cols * 4, 0xa5)?,
                    gpu.alloc_fill(pv * cols * 2, 0xa5)?,
                ])
            };
            let [oa, o2a, pa] = mk()?;
            let args: Vec<Arg> = [oa.arg::<u32>(), o2a.arg::<u32>()]
                .into_iter()
                .chain(inputs)
                .chain([Arg::U32(vox as u32)])
                .chain(tail)
                .collect();
            gpu.launch_1d("u3_apply", vox * pairs, &args)?;
            let e = Arg::U32((edge / 2) as u32);
            gpu.launch_1d("u3_pool2", pv * pairs, &[pa.arg::<u32>(), oa.arg::<u32>(), e, e, e, Arg::U32(cols as u32)])?;
            let [ob, o2b, pb] = mk()?;
            let args: Vec<Arg> = [ob.arg::<u32>(), o2b.arg::<u32>(), pb.arg::<u32>()]
                .into_iter()
                .chain(inputs)
                .chain([e, e, e])
                .chain(tail)
                .collect();
            gpu.launch_1d("u3_apply_pool", pv * pairs, &args)?;
            for (what, a, bb) in [("out", &oa, &ob), ("out2", &o2a, &o2b), ("pool", &pa, &pb)] {
                let (va, vb) = (gpu.download::<u16>(a)?, gpu.download::<u16>(bb)?);
                let diff = va.iter().zip(&vb).filter(|(p, q)| p != q).count();
                ensure!(
                    va.len() == vb.len() && diff == 0,
                    "u3_apply_pool self-check (normed {normed}): {what} differs in {diff} of {} values",
                    va.len()
                );
            }
            // the sentinel must be gone everywhere apply writes, or the comparison proves nothing
            let filled = gpu.download::<u16>(&ob)?.iter().filter(|&&v| v == 0xa5a5).count();
            ensure!(filled == 0, "u3_apply_pool self-check: {filled} output values never written");
        }
        Ok(())
    }
}

/// A block's residual
enum Residual<'a> {
    None,
    /// A bf16 tensor of the output's shape
    Plain(&'a Buf),
    /// A bias-free conv output, normalised on load by that conv's norm
    Normed(&'a Buf, &'a Conv),
}

/// Fails unless `n` elements index with u32
pub fn u32_extent(what: &str, n: usize) -> Result<()> {
    u32::try_from(n).map(|_| ()).wrap_err_with(|| format!("{what}: {n} elements overflow u32 indexing"))
}

/// The buffer (`'y'` or `'x'`) decoder conv `i` of a stage writes: they alternate from `y`, so a
/// stage of n convs ends in `dec_dst(n - 1)`, which is where the head must read the last stage
fn dec_dst(i: usize) -> char {
    if i.is_multiple_of(2) { 'y' } else { 'x' }
}

#[cfg(test)]
mod dec_tests {
    use super::dec_dst;

    /// A last stage of one or three convs ends in y and of two in x, and the head reads what the
    /// forward recorded, so an even-conv last stage (which the exporter accepts) reads its output
    #[test]
    fn decoder_buffers_alternate_from_y() {
        assert_eq!([0, 1, 2].map(dec_dst), ['y', 'x', 'y']);
    }
}
