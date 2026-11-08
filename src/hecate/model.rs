//! Hecate's network on the GPU at a fixed batch, with `hecate.py`'s numerics at fp32 or bf16
//!
//! - Input: `uint8 / 255`, then the eval BatchNorm3d(1) `canonical.normalization` (`hec_gather`)
//! - Stem: conv3d 1->64 (k3, stride 1, no bias), BatchNorm, ReLU
//! - layer1..4: Bottleneck blocks [3, 8, 36, 3], planes 64/128/256/512 (x4 out); the first block of
//!   each layer has a 1x1 downsample conv + BN, with stride 2 (on conv2 and the downsample) in
//!   layers 2-4. Features 256 @ 16x64x64, 512 @ 8x32x32, 1024 @ 4x16x16, 2048 @ 2x8x8
//! - Decoder: `channel_reduce` 1x1 conv + GroupNorm + ReLU to 64/128/256/512, then three steps of
//!   trilinear x2 of the deeper map concatenated with the next reduced feature, into a
//!   ResConvBlock3D (conv3 + GN + ReLU, conv3 + GN, plus a 1x1 shortcut conv, ReLU)
//! - Head: `hec_head`, the depth-attention collapse to one probability per column
//!
//! Storage follows PyTorch. At fp32 every activation is float32. Under bf16 autocast the convs
//! write bf16, and BatchNorm, ReLU and the encoder's residual adds keep bf16, while GroupNorm
//! outputs and everything computed from them stay float32 until the next conv casts its input. So
//! here the encoder and every conv input and output are bf16, and the decoder's GroupNorm and block
//! outputs float32. BatchNorm is applied as a per-channel scale and shift after the conv rather
//! than folded into its weights, so the conv's own output rounds as PyTorch's does

use std::collections::HashMap;
use std::rc::Rc;

use color_eyre::eyre::{Result, ensure, eyre};
use half::bf16;
use serde_json::Value;

use crate::dnn::fused::{Epilogue, FusedConv};
use crate::dnn::{ConvPlan, ConvShape, Dtype};
use crate::gpu::{Arg, Buf, Gpu};
use crate::hecate::{D, P, PIX, Precision, u32_extent};
use crate::model::Ctx;
use crate::sched::DESC;
use crate::weights::Checkpoint;

/// BatchNorm and GroupNorm eps of every norm in the release, checked against the export's sidecar
const EPS: f32 = 1e-5;
/// Bottleneck blocks per layer, ResNet-152
const BLOCKS: [usize; 4] = [3, 8, 36, 3];
/// Bottleneck planes per layer; each block outputs four times as many channels
const PLANES: [usize; 4] = [64, 128, 256, 512];
/// `channel_reduce` widths, which are also the decoder widths
const DEC: [usize; 4] = [64, 128, 256, 512];
/// Channels of the stem conv's padded input when cuDNN won't take one channel channels-last
const STEM_PAD: usize = 8;

/// The fields of `hecate-9.6um.json` this engine relies on, beyond its keys
pub fn check_contract(sidecar: &Value) -> Result<()> {
    let want = [
        ("reader", Value::from("hecate-9.6um")),
        ("architecture_family", Value::from("hecate")),
        ("weight_source", Value::from("ema_model")),
        ("sampling_um", Value::from(9.6)),
        ("patch_size", serde_json::json!([D, P, P])),
        ("image_normalization", serde_json::json!({"mode": "divide", "divisor": 255.0})),
        ("bn_eps", Value::from(1e-5)),
        ("gn_eps", Value::from(1e-5)),
    ];
    for (key, value) in want {
        ensure!(sidecar.get(key) == Some(&value), "export sidecar {key} is {:?}, need {value}", sidecar.get(key));
    }
    let canonical = &sidecar["model_config"]["canonical"];
    ensure!(
        canonical["with_norm"] == true
            && canonical["depth_margin"] == 0
            && canonical["native_xy_stride"] == 1
            && canonical["native_refinement"] == false
            && canonical["input_depth"] == D,
        "export sidecar model_config.canonical {canonical} is not the 9.6 um release"
    );
    Ok(())
}

/// A device activation, float32 or bf16 bits
struct Act {
    buf: Buf,
    bf16: bool,
}

impl Act {
    /// `n` elements, NaN-filled so a value that is never written fails the finiteness checks
    fn new(gpu: &Gpu, n: usize, bf16: bool) -> Result<Self> {
        u32_extent("activation", n)?;
        Ok(Self { buf: gpu.alloc_fill(n * if bf16 { 2 } else { 4 }, 0xFF)?, bf16 })
    }

    /// The first `n` elements as the kernels' (float32, bf16) slice pair, the unused one empty
    fn pair(&self, n: usize) -> [Arg; 2] {
        if self.bf16 { [Arg::EMPTY, self.buf.view::<u16>(0, n)] } else { [self.buf.view::<f32>(0, n), Arg::EMPTY] }
    }

    /// The first `n` elements as a float32 slice, for kernels that only read float32
    fn f32(&self, n: usize) -> Result<Arg> {
        ensure!(!self.bf16, "a float32 input stored as bf16");
        Ok(self.buf.view::<f32>(0, n))
    }

    fn mem(&self) -> (u64, usize) {
        (self.buf.ptr(), self.buf.bytes())
    }
}

/// Turns the fused encoder convs off, so every conv runs unfused followed by `hec_affine`
pub(crate) const FUSED_ENV: &str = "INKRS_HECATE_FUSED";

/// The epilogue of a conv followed by BatchNorm and ReLU
const RELU: Epilogue = Epilogue { residual: false, relu: true };

/// Conv plans by shape: Hecate has 168 convs but about 30 distinct shapes
struct Planner<'a> {
    ctx: &'a Ctx<'a>,
    dtype: Dtype,
    plans: HashMap<ConvShape, Rc<ConvPlan>>,
    /// Fused plans by shape and epilogue; `None` records a shape cuDNN would not fuse
    fused: HashMap<(ConvShape, Epilogue), Option<Rc<FusedConv>>>,
    use_fused: bool,
}

impl Planner<'_> {
    fn plan(&mut self, shape: ConvShape) -> Result<Rc<ConvPlan>> {
        if let Some(p) = self.plans.get(&shape) {
            return Ok(p.clone());
        }
        let ctx = self.ctx;
        let plan = Rc::new(ConvPlan::typed(ctx.gpu, ctx.dnn, shape.clone(), self.dtype, ctx.pick, ctx.max_workspace)?);
        self.plans.insert(shape, plan.clone());
        Ok(plan)
    }

    /// The fused plan for `shape` and `ep`, or `None` (with a note on stderr) when fusion is off
    /// or cuDNN has no engine for it, so the conv falls back to the unfused path
    fn fused(&mut self, shape: &ConvShape, ep: Epilogue) -> Option<Rc<FusedConv>> {
        if !self.use_fused {
            return None;
        }
        let key = (shape.clone(), ep);
        if let Some(p) = self.fused.get(&key) {
            return p.clone();
        }
        let ctx = self.ctx;
        let plan = match FusedConv::new(ctx.gpu, ctx.dnn, shape.clone(), self.dtype, ep, ctx.pick, ctx.max_workspace) {
            Ok(p) => Some(Rc::new(p)),
            Err(e) => {
                eprintln!("hecate: {shape} {ep:?} unfused: {e:#}");
                None
            }
        };
        self.fused.insert(key, plan.clone());
        plan
    }
}

/// A planned conv and its channels-last weight in the plan's data type: fused with its BatchNorm
/// epilogue when cuDNN can, else a plain conv for a separate apply pass
struct Conv {
    plan: Option<Rc<ConvPlan>>,
    fused: Option<Rc<FusedConv>>,
    w: Buf,
}

impl Conv {
    /// `key.weight`, `[k, c, *kernel]`, as `[k, *kernel, c_pad]`, zero in the padded channels
    fn load(planner: &mut Planner, ck: &Checkpoint, key: &str, shape: ConvShape, c_real: usize) -> Result<Self> {
        Self::load_with(planner, ck, key, shape, c_real, None)
    }

    /// [`Conv::load`], fused with a BatchNorm epilogue `ep` where cuDNN offers one
    fn load_with(
        planner: &mut Planner,
        ck: &Checkpoint,
        key: &str,
        shape: ConvShape,
        c_real: usize,
        ep: Option<Epilogue>,
    ) -> Result<Self> {
        let (k, c_pad) = (shape.k as usize, shape.c as usize);
        let taps: usize = shape.kernel.iter().map(|&v| v as usize).product();
        let mut dims = vec![k, c_real];
        dims.extend(shape.kernel.iter().map(|&v| v as usize));
        let w = ck.take(&format!("{key}.weight"), &dims)?;
        let mut nhwc = vec![0f32; k * taps * c_pad];
        for ko in 0..k {
            for ci in 0..c_real {
                for t in 0..taps {
                    nhwc[(ko * taps + t) * c_pad + ci] = w[(ko * c_real + ci) * taps + t];
                }
            }
        }
        let gpu = planner.ctx.gpu;
        let w = match planner.dtype {
            Dtype::Float { .. } => gpu.upload(&nhwc)?,
            // autocast casts the float32 weight to bf16, round to nearest even
            Dtype::Bf16 => gpu.upload(&nhwc.iter().map(|&v| bf16::from_f32(v)).collect::<Vec<_>>())?,
            Dtype::Half => return Err(eyre!("Hecate has no FP16 path")),
        };
        let fused = ep.and_then(|ep| planner.fused(&shape, ep));
        let plan = if fused.is_some() { None } else { Some(planner.plan(shape)?) };
        Ok(Self { plan, fused, w })
    }

    /// The unfused conv into `y`
    fn run(&self, ctx: &Ctx, ws: &Buf, x: &Act, y: &Act) -> Result<()> {
        let plan = self.plan.as_ref().ok_or_else(|| eyre!("a fused conv run without its epilogue"))?;
        plan.run(ctx.dnn, x.mem(), (self.w.ptr(), self.w.bytes()), y.mem(), Some(ws))
    }
}

/// An eval BatchNorm3d's per-channel scale and shift: interleaved `[scale, shift]` for
/// `hec_affine`, and the two as separate float32 vectors for the fused conv epilogue
struct Bn {
    ab: Buf,
    scale: Buf,
    shift: Buf,
}

/// An eval BatchNorm3d as the epilogue's `[scale, shift]` per channel:
/// `scale = weight / sqrt(running_var + eps)`, `shift = bias - running_mean * scale`, in float64
fn batch_norm(gpu: &Gpu, ck: &Checkpoint, key: &str, c: usize) -> Result<Bn> {
    let [w, b, m, v] = ["weight", "bias", "running_mean", "running_var"].map(|f| ck.take(&format!("{key}.{f}"), &[c]));
    let (w, b, m, v) = (w?, b?, m?, v?);
    let ab: Vec<[f32; 2]> = (0..c)
        .map(|i| {
            let s = w[i] as f64 / (v[i] as f64 + EPS as f64).sqrt();
            [s as f32, (b[i] as f64 - m[i] as f64 * s) as f32]
        })
        .collect();
    Ok(Bn {
        ab: gpu.upload(&ab)?,
        scale: gpu.upload(&ab.iter().map(|v| v[0]).collect::<Vec<_>>())?,
        shift: gpu.upload(&ab.iter().map(|v| v[1]).collect::<Vec<_>>())?,
    })
}

/// A GroupNorm's affine weight and bias
struct GroupNorm {
    gamma: Buf,
    beta: Buf,
}

impl GroupNorm {
    fn load(gpu: &Gpu, ck: &Checkpoint, key: &str, c: usize) -> Result<Self> {
        Ok(Self {
            gamma: gpu.upload(&ck.take(&format!("{key}.weight"), &[c])?)?,
            beta: gpu.upload(&ck.take(&format!("{key}.bias"), &[c])?)?,
        })
    }
}

/// One Bottleneck block
struct Bottleneck {
    conv1: Conv,
    bn1: Bn,
    conv2: Conv,
    bn2: Bn,
    conv3: Conv,
    bn3: Bn,
    downsample: Option<(Conv, Bn)>,
    /// Voxels at the block's input and output resolution
    vin: usize,
    vout: usize,
    planes: usize,
}

/// One encoder layer and its buffers: two outputs used in turn, the last block's being the
/// layer's feature map
struct Layer {
    blocks: Vec<Bottleneck>,
    t1: Act,
    t2: Act,
    ds: Act,
    out: [Act; 2],
    vox: usize,
    c: usize,
    /// Edge of the layer's output `[d, h, h]`
    dims: [usize; 3],
}

impl Layer {
    fn output(&self) -> &Act {
        &self.out[(self.blocks.len() - 1) % 2]
    }
}

/// One `channel_reduce`: 1x1 conv, GroupNorm, ReLU, into float32
struct Reduce {
    conv: Conv,
    gn: GroupNorm,
    out: Act,
    vox: usize,
    c: usize,
}

/// One ResConvBlock3D after the upsample and concatenation
struct DecBlock {
    conv1: Conv,
    gn1: GroupNorm,
    conv2: Conv,
    gn2: GroupNorm,
    shortcut: Conv,
    cat: Act,
    t: Act,
    sc: Act,
    out: Act,
    /// Input `[d, h, w]` of the deeper map, its channels, the skip's and the block's output channels
    low: [usize; 3],
    cl: usize,
    cs: usize,
    c: usize,
}

/// The planned network and its activations at batch `batch`
pub struct Model {
    pub batch: usize,
    pub precision: Precision,
    /// Channels of the stem conv's input buffer: 1, or [`STEM_PAD`] when cuDNN needed padding
    pub stem_c: usize,
    norm: [f32; 4],
    stem: Conv,
    stem_bn: Bn,
    layers: Vec<Layer>,
    reduce: Vec<Reduce>,
    dec: Vec<DecBlock>,
    head_wl: Buf,
    head_bl: f32,
    head_wa: Buf,
    head_ba: f32,
    dcs: f32,
    // activations and scratch
    x0: Act,
    support: Buf,
    a0: Act,
    raw: Act,
    partial: Buf,
    ab: Buf,
    ws: Buf,
}

/// A named activation of the last forward for the P0 gate: buffer, element type, channels-last
/// shape (`[B, ...spatial, C]`) and the leading channels to keep
pub struct Tap<'a> {
    pub name: String,
    pub buf: &'a Buf,
    pub bf16: bool,
    pub shape: Vec<usize>,
    pub keep_c: usize,
}

impl Model {
    /// Plans every layer at `batch` and uploads the weights; every checkpoint tensor must be
    /// consumed (the caller checks with [`Checkpoint::finish`])
    pub fn new(ctx: &Ctx, ck: &Checkpoint, batch: usize, precision: Precision) -> Result<Self> {
        let gpu = ctx.gpu;
        precision.check_device(gpu.inner.compute_capability())?;
        let n = batch as i32;
        let bf = precision.bf16();
        let dtype = match precision {
            Precision::Fp32 { tf32 } => Dtype::Float { tf32 },
            Precision::Bf16 => Dtype::Bf16,
        };
        // the fused epilogue is off with INKRS_HECATE_FUSED=0, and for fp32 without TF32, whose
        // legacy math type is what forbids TF32
        let use_fused = std::env::var(FUSED_ENV).map_or(true, |v| v != "0") && dtype != Dtype::Float { tf32: false };
        let mut planner = Planner { ctx, dtype, plans: HashMap::new(), fused: HashMap::new(), use_fused };
        let relu = Some(RELU);
        let tile = [D as i32, P as i32, P as i32];
        let vox0 = D * PIX;

        let nk = "canonical.normalization";
        let norm = [
            ck.take(&format!("{nk}.running_mean"), &[1])?[0],
            ck.take(&format!("{nk}.running_var"), &[1])?[0],
            ck.take(&format!("{nk}.weight"), &[1])?[0],
            ck.take(&format!("{nk}.bias"), &[1])?[0],
        ];

        // the stem conv on one channel, or on STEM_PAD with zeros when cuDNN refuses one
        let stem_key = "canonical.backbone.conv1";
        let stem_shape = |c: i32| ConvShape::same(n, c, 64, &tile, 3, 1);
        let one_channel = planner.fused(&stem_shape(1), RELU).is_some()
            || planner
                .plan(stem_shape(1))
                .map_err(|e| eprintln!("hecate: stem conv on 1 channel: {e:#}; padding to {STEM_PAD}"))
                .is_ok();
        let stem_c = if one_channel { 1 } else { STEM_PAD };
        let stem = Conv::load_with(&mut planner, ck, stem_key, stem_shape(stem_c as i32), 1, Some(RELU))?;
        let stem_bn = batch_norm(gpu, ck, "canonical.backbone.bn1", 64)?;

        // encoder
        let mut layers = Vec::new();
        let (mut cin, mut dims) = (64usize, [D, P, P]);
        for (l, (&planes, &nb)) in PLANES.iter().zip(&BLOCKS).enumerate() {
            let stride = if l == 0 { 1 } else { 2 };
            let dims_in = dims;
            let dims_out = dims.map(|v| v / stride);
            let (vin, vout) = (dims_in.iter().product::<usize>(), dims_out.iter().product::<usize>());
            let to_i = |d: [usize; 3]| d.map(|v| v as i32);
            let c = planes * 4;
            let mut blocks = Vec::new();
            for j in 0..nb {
                let key = format!("canonical.backbone.layer{}.{j}", l + 1);
                let (s, din, bin) = if j == 0 { (stride as i32, dims_in, cin) } else { (1, dims_out, c) };
                let conv1 = Conv::load_with(
                    &mut planner,
                    ck,
                    &format!("{key}.conv1"),
                    ConvShape::same(n, bin as i32, planes as i32, &to_i(din), 1, 1),
                    bin,
                    relu,
                )?;
                let conv2 = Conv::load_with(
                    &mut planner,
                    ck,
                    &format!("{key}.conv2"),
                    ConvShape::same(n, planes as i32, planes as i32, &to_i(din), 3, s),
                    planes,
                    relu,
                )?;
                let conv3 = Conv::load_with(
                    &mut planner,
                    ck,
                    &format!("{key}.conv3"),
                    ConvShape::same(n, planes as i32, c as i32, &to_i(dims_out), 1, 1),
                    planes,
                    Some(Epilogue { residual: true, relu: true }),
                )?;
                let downsample = if j == 0 {
                    let conv = Conv::load_with(
                        &mut planner,
                        ck,
                        &format!("{key}.downsample.0"),
                        ConvShape::same(n, bin as i32, c as i32, &to_i(din), 1, s),
                        bin,
                        Some(Epilogue { residual: false, relu: false }),
                    )?;
                    Some((conv, batch_norm(gpu, ck, &format!("{key}.downsample.1"), c)?))
                } else {
                    None
                };
                blocks.push(Bottleneck {
                    conv1,
                    bn1: batch_norm(gpu, ck, &format!("{key}.bn1"), planes)?,
                    conv2,
                    bn2: batch_norm(gpu, ck, &format!("{key}.bn2"), planes)?,
                    conv3,
                    bn3: batch_norm(gpu, ck, &format!("{key}.bn3"), c)?,
                    downsample,
                    vin: if j == 0 { vin } else { vout },
                    vout,
                    planes,
                });
            }
            layers.push(Layer {
                blocks,
                t1: Act::new(gpu, batch * vin * planes, bf)?,
                t2: Act::new(gpu, batch * vout * planes, bf)?,
                ds: Act::new(gpu, batch * vout * c, bf)?,
                out: [Act::new(gpu, batch * vout * c, bf)?, Act::new(gpu, batch * vout * c, bf)?],
                vox: vout,
                c,
                dims: dims_out,
            });
            cin = c;
            dims = dims_out;
        }

        // decoder
        let mut reduce = Vec::new();
        for (i, layer) in layers.iter().enumerate() {
            let key = format!("canonical.decoder.channel_reduce.{i}");
            let d = layer.dims.map(|v| v as i32);
            reduce.push(Reduce {
                conv: Conv::load(
                    &mut planner,
                    ck,
                    &format!("{key}.0"),
                    ConvShape::same(n, layer.c as i32, DEC[i] as i32, &d, 1, 1),
                    layer.c,
                )?,
                gn: GroupNorm::load(gpu, ck, &format!("{key}.1"), DEC[i])?,
                out: Act::new(gpu, batch * layer.vox * DEC[i], false)?,
                vox: layer.vox,
                c: DEC[i],
            });
        }
        let mut dec = Vec::new();
        for k in 0..3 {
            // block k joins the deeper map (reduce[3] or block k-1) with reduce[2 - k]
            let (deep, skip) = (3 - k, 2 - k);
            let low = layers[deep].dims;
            let (cl, cs) = (DEC[deep], DEC[skip]);
            let c = DEC[skip];
            let d = layers[skip].dims.map(|v| v as i32);
            let vox = layers[skip].vox;
            let key = format!("canonical.decoder.decoder_blocks.{k}");
            let cat = (cl + cs) as i32;
            dec.push(DecBlock {
                conv1: Conv::load(
                    &mut planner,
                    ck,
                    &format!("{key}.conv1"),
                    ConvShape::same(n, cat, c as i32, &d, 3, 1),
                    cl + cs,
                )?,
                gn1: GroupNorm::load(gpu, ck, &format!("{key}.gn1"), c)?,
                conv2: Conv::load(
                    &mut planner,
                    ck,
                    &format!("{key}.conv2"),
                    ConvShape::same(n, c as i32, c as i32, &d, 3, 1),
                    c,
                )?,
                gn2: GroupNorm::load(gpu, ck, &format!("{key}.gn2"), c)?,
                shortcut: Conv::load(
                    &mut planner,
                    ck,
                    &format!("{key}.shortcut"),
                    ConvShape::same(n, cat, c as i32, &d, 1, 1),
                    cl + cs,
                )?,
                cat: Act::new(gpu, batch * vox * (cl + cs), bf)?,
                t: Act::new(gpu, batch * vox * c, bf)?,
                sc: Act::new(gpu, batch * vox * c, bf)?,
                out: Act::new(gpu, batch * vox * c, false)?,
                low,
                cl,
                cs,
                c,
            });
        }
        ensure!(dec[2].c == 64 && layers[0].dims == [D, P, P], "decoder ends at {} channels", dec[2].c);

        // head: Conv2d logit [1, 64, 1, 1] applied as conv3d, and the attention conv [1, 64, 1, 1, 1]
        let head_wl = gpu.upload(&ck.take("canonical.decoder.logit.weight", &[1, 64, 1, 1])?)?;
        let head_bl = ck.scalar("canonical.decoder.logit.bias")?;
        let head_wa = gpu.upload(&ck.take("canonical.decoder.depth_collapse.attn_conv.weight", &[1, 64, 1, 1, 1])?)?;
        let head_ba = ck.scalar("canonical.decoder.depth_collapse.attn_conv.bias")?;
        let dcs = ck.scalar0("depth_coordinate_scale")?;

        // scratch: the largest conv output, GroupNorm partial sums and per-sample affine
        let all: Vec<&Rc<ConvPlan>> = planner.plans.values().collect();
        let raw_len = all.iter().map(|p| p.shape.elements().2).max().unwrap_or(0);
        let fused_ws = planner.fused.values().flatten().map(|p| p.workspace).max().unwrap_or(0);
        let ws = gpu.alloc(all.iter().map(|p| p.workspace()).max().unwrap_or(0).max(fused_ws))?;
        let mut gn_shapes: Vec<(usize, usize)> = reduce.iter().map(|r| (r.vox, r.c)).collect();
        gn_shapes.extend(dec.iter().map(|b| (8 * b.low.iter().product::<usize>(), b.c)));
        let partial_len = gn_shapes.iter().map(|&(rows, cols)| gn_layout(rows, cols).2 * cols).max().unwrap_or(0);
        let max_cols = gn_shapes.iter().map(|&(_, c)| c).max().unwrap_or(0);

        Ok(Self {
            batch,
            precision,
            stem_c,
            norm,
            stem,
            stem_bn,
            reduce,
            dec,
            head_wl,
            head_bl,
            head_wa,
            head_ba,
            dcs,
            // the padded stem channels must read zero, so this one is zero-filled, not NaN
            x0: Act { buf: gpu.alloc_fill(batch * vox0 * stem_c * if bf { 2 } else { 4 }, 0)?, bf16: bf },
            support: gpu.alloc_fill(batch * PIX, 0xFF)?,
            a0: Act::new(gpu, batch * vox0 * 64, bf)?,
            raw: Act::new(gpu, raw_len, bf)?,
            partial: gpu.alloc(batch * partial_len * 8)?,
            ab: gpu.alloc(batch * max_cols * 8)?,
            ws,
            layers,
        })
    }

    /// The distinct conv plans, for the report
    pub fn plans(&self) -> Vec<&ConvPlan> {
        self.convs().iter().filter_map(|c| c.plan.as_deref()).fold(Vec::new(), |mut v, p| {
            if !v.iter().any(|q: &&ConvPlan| std::ptr::eq(*q, p)) {
                v.push(p);
            }
            v
        })
    }

    /// The distinct fused plans, for the report
    pub(crate) fn fused_plans(&self) -> Vec<&FusedConv> {
        self.convs().iter().filter_map(|c| c.fused.as_deref()).fold(Vec::new(), |mut v, p| {
            if !v.iter().any(|q: &&FusedConv| std::ptr::eq(*q, p)) {
                v.push(p);
            }
            v
        })
    }

    /// One line per fused plan for the report's `algos`, after the unfused plans' lines
    pub(crate) fn fused_algos(&self) -> impl Iterator<Item = String> + '_ {
        self.fused_plans().into_iter().map(|f| {
            format!(
                "{} | fused bn{}{} {} {:.3} ms ws {} B of {} timed{}",
                f.shape,
                if f.epilogue.residual { " +residual" } else { "" },
                if f.epilogue.relu { " relu" } else { "" },
                f.engine,
                f.ms,
                f.workspace,
                f.candidates,
                if f.cached { " (cached)" } else { "" }
            )
        })
    }

    /// Every conv in forward order
    fn convs(&self) -> Vec<&Conv> {
        let mut v = vec![&self.stem];
        for l in &self.layers {
            for b in &l.blocks {
                if let Some((c, _)) = &b.downsample {
                    v.push(c);
                }
                v.extend([&b.conv1, &b.conv2, &b.conv3]);
            }
        }
        v.extend(self.reduce.iter().map(|r| &r.conv));
        for b in &self.dec {
            v.extend([&b.shortcut, &b.conv1, &b.conv2]);
        }
        v
    }

    /// `y = relu?(round?(x * a + b) + residual?)` over the first `n` elements, `ab` per channel
    /// (`per_sample == 0`) or per sample and channel
    #[allow(clippy::too_many_arguments)]
    fn affine(
        &self,
        gpu: &Gpu,
        y: &Act,
        x: &Act,
        residual: Option<&Act>,
        ab: Arg,
        (n, cols, per_sample): (usize, usize, usize),
        relu: bool,
    ) -> Result<()> {
        let [yf, yh] = y.pair(n);
        let [xf, xh] = x.pair(n);
        let [rf, rh] = residual.map_or([Arg::EMPTY, Arg::EMPTY], |r| r.pair(n));
        // autocast's BatchNorm writes bf16 before the residual add; GroupNorm writes float32
        let round_mid = self.precision.bf16() && per_sample == 0;
        gpu.launch_1d(
            "hec_affine",
            n,
            &[
                yf,
                yh,
                xf,
                xh,
                rf,
                rh,
                ab,
                Arg::U32(cols as u32),
                Arg::U32(per_sample as u32),
                Arg::U32(round_mid as u32),
                Arg::U32(relu as u32),
            ],
        )
    }

    /// BatchNorm epilogue of the conv output in `raw` into `y`
    fn bn(
        &self,
        gpu: &Gpu,
        y: &Act,
        bn: &Bn,
        (vox, c): (usize, usize),
        residual: Option<&Act>,
        relu: bool,
    ) -> Result<()> {
        self.affine(gpu, y, &self.raw, residual, bn.ab.arg::<[f32; 2]>(), (self.batch * vox * c, c, 0), relu)
    }

    /// `y = relu?(bn(conv(x)) (+ residual))`: one fused conv where planned, else the conv into
    /// `raw` and [`Model::bn`]. `(vox, c)` is the output's voxels and channels
    #[allow(clippy::too_many_arguments)]
    fn conv_bn(
        &self,
        ctx: &Ctx,
        conv: &Conv,
        x: &Act,
        bn: &Bn,
        y: &Act,
        (vox, c): (usize, usize),
        residual: Option<&Act>,
        relu: bool,
    ) -> Result<()> {
        match &conv.fused {
            Some(f) => {
                ensure!(
                    f.epilogue == Epilogue { residual: residual.is_some(), relu },
                    "{}: fused epilogue mismatch",
                    f.shape
                );
                ensure!(x.bf16 == y.bf16 && residual.is_none_or(|r| r.bf16 == y.bf16), "fused conv storage mismatch");
                let n = self.batch * vox * c;
                ensure!(f.shape.elements().2 == n, "{}: output is not {n} elements", f.shape);
                f.run(
                    ctx.dnn,
                    x.mem(),
                    (conv.w.ptr(), conv.w.bytes()),
                    &bn.scale,
                    &bn.shift,
                    residual.map(|r| r.mem()),
                    y.mem(),
                    &self.ws,
                )
            }
            None => {
                conv.run(ctx, &self.ws, x, &self.raw)?;
                self.bn(ctx.gpu, y, bn, (vox, c), residual, relu)
            }
        }
    }

    /// GroupNorm(min(32, c)) of the conv output in `raw` (`[B, rows, cols]`) into `y`, with the
    /// optional residual and ReLU
    fn gn(
        &self,
        gpu: &Gpu,
        y: &Act,
        g: &GroupNorm,
        (rows, cols): (usize, usize),
        residual: Option<&Act>,
        relu: bool,
    ) -> Result<()> {
        let b = self.batch;
        let x = &self.raw;
        let n = b * rows * cols;
        let [xf, xh] = x.pair(n);
        let (cpb, chunk, chunks) = gn_layout(rows, cols);
        // gn_partial gives each block `256 / cpb` lanes per channel and returns early otherwise
        ensure!(256 % cpb == 0 && cols % cpb == 0, "GroupNorm over {cols} channels has no gn_partial layout");
        let partial_len = b * chunks * cols;
        ensure!(
            partial_len * 8 <= self.partial.bytes() && b * cols * 8 <= self.ab.bytes(),
            "GroupNorm scratch too small"
        );
        gpu.launch(
            "gn_partial",
            (chunks as u32, b as u32, (cols / cpb) as u32),
            (256, 1, 1),
            0,
            &[
                self.partial.view::<[f32; 2]>(0, partial_len),
                xf,
                xh,
                Arg::U32(rows as u32),
                Arg::U32(cols as u32),
                Arg::U32(cpb as u32),
                Arg::U32(chunk as u32),
            ],
        )?;
        let [xf, xh] = x.pair(n);
        gpu.launch_1d(
            "gn_finalize",
            b * cols,
            &[
                self.ab.view::<[f32; 2]>(0, b * cols),
                self.partial.view::<[f32; 2]>(0, partial_len),
                xf,
                xh,
                g.gamma.arg::<f32>(),
                g.beta.arg::<f32>(),
                Arg::U32(rows as u32),
                Arg::U32(cols as u32),
                Arg::U32(chunks as u32),
                Arg::U32(cols.min(32) as u32),
                Arg::F32(EPS),
            ],
        )?;
        self.affine(gpu, y, x, residual, self.ab.view::<[f32; 2]>(0, b * cols), (n, cols, rows * cols), relu)
    }

    /// Queues one forward: gather and normalise the tiles `desc` describes (`[batch, DESC]`) from
    /// the resident `vol`, run the network, and write each real tile's probability times its
    /// support into its slot of `prob` (`[slots, 64 * 64]` float32)
    pub fn forward(&self, ctx: &Ctx, vol: (&Buf, usize, usize), desc: Arg, prob: &Buf) -> Result<()> {
        let gpu = ctx.gpu;
        let b = self.batch;
        let ws = &self.ws;
        let vox0 = D * PIX;
        ensure!(
            matches!(desc, Arg::Slice(_, len) if len as usize == b * DESC),
            "descriptor view {desc:?} for batch {b}"
        );
        let (vbuf, height, width) = vol;

        let [mean, var, gamma, beta] = self.norm;
        // PyTorch's eval BatchNorm: (x - mean) * invstd * weight + bias, invstd = 1 / sqrt(var + eps)
        let invstd = 1.0 / (var + EPS).sqrt();
        let [xf, xh] = self.x0.pair(b * vox0 * self.stem_c);
        gpu.launch_1d(
            "hec_gather",
            b * PIX,
            &[
                xf,
                xh,
                self.support.arg::<u8>(),
                vbuf.arg::<u8>(),
                desc,
                Arg::U32(height as u32),
                Arg::U32(width as u32),
                Arg::U32(self.stem_c as u32),
                Arg::F32(mean),
                Arg::F32(invstd),
                Arg::F32(gamma),
                Arg::F32(beta),
            ],
        )?;
        self.conv_bn(ctx, &self.stem, &self.x0, &self.stem_bn, &self.a0, (vox0, 64), None, true)?;

        // encoder
        let mut x = &self.a0;
        for layer in &self.layers {
            for (j, blk) in layer.blocks.iter().enumerate() {
                let out = &layer.out[j % 2];
                let residual = match &blk.downsample {
                    Some((conv, bn)) => {
                        self.conv_bn(ctx, conv, x, bn, &layer.ds, (blk.vout, layer.c), None, false)?;
                        &layer.ds
                    }
                    None => x,
                };
                self.conv_bn(ctx, &blk.conv1, x, &blk.bn1, &layer.t1, (blk.vin, blk.planes), None, true)?;
                self.conv_bn(ctx, &blk.conv2, &layer.t1, &blk.bn2, &layer.t2, (blk.vout, blk.planes), None, true)?;
                self.conv_bn(ctx, &blk.conv3, &layer.t2, &blk.bn3, out, (blk.vout, layer.c), Some(residual), true)?;
                x = out;
            }
        }

        // decoder
        for (r, layer) in self.reduce.iter().zip(&self.layers) {
            r.conv.run(ctx, ws, layer.output(), &self.raw)?;
            self.gn(gpu, &r.out, &r.gn, (r.vox, r.c), None, true)?;
        }
        for (k, blk) in self.dec.iter().enumerate() {
            let low = if k == 0 { &self.reduce[3].out } else { &self.dec[k - 1].out };
            let skip = &self.reduce[2 - k].out;
            let [d, h, w] = blk.low;
            let vox = 8 * d * h * w;
            let (nl, ns, ncat) = (b * d * h * w * blk.cl, b * vox * blk.cs, b * vox * (blk.cl + blk.cs));
            let [yf, yh] = blk.cat.pair(ncat);
            gpu.launch_1d(
                "hec_up2cat",
                ncat,
                &[
                    yf,
                    yh,
                    low.f32(nl)?,
                    skip.f32(ns)?,
                    Arg::U32(d as u32),
                    Arg::U32(h as u32),
                    Arg::U32(w as u32),
                    Arg::U32(blk.cl as u32),
                    Arg::U32(blk.cs as u32),
                ],
            )?;
            blk.shortcut.run(ctx, ws, &blk.cat, &blk.sc)?;
            blk.conv1.run(ctx, ws, &blk.cat, &self.raw)?;
            self.gn(gpu, &blk.t, &blk.gn1, (vox, blk.c), None, true)?;
            blk.conv2.run(ctx, ws, &blk.t, &self.raw)?;
            self.gn(gpu, &blk.out, &blk.gn2, (vox, blk.c), Some(&blk.sc), true)?;
        }

        let feat = &self.dec[2].out;
        gpu.launch_1d(
            "hec_head",
            b * PIX,
            &[
                prob.arg::<f32>(),
                feat.f32(b * vox0 * 64)?,
                self.support.arg::<u8>(),
                self.head_wl.arg::<f32>(),
                Arg::F32(self.head_bl),
                self.head_wa.arg::<f32>(),
                Arg::F32(self.head_ba),
                Arg::F32(self.dcs),
                desc,
                Arg::U32(self.precision.bf16() as u32),
            ],
        )
    }

    /// The activations of the last forward, channels-last, for the P0 gate
    pub fn taps(&self) -> Vec<Tap<'_>> {
        let b = self.batch;
        fn tap(name: String, a: &Act, shape: [usize; 5], keep_c: usize) -> Tap<'_> {
            Tap { name, buf: &a.buf, bf16: a.bf16, shape: shape.to_vec(), keep_c }
        }
        let at = |[d, h, w]: [usize; 3], c: usize| [b, d, h, w, c];
        let mut v = vec![
            tap("input".into(), &self.x0, at([D, P, P], self.stem_c), 1),
            tap("stem".into(), &self.a0, at([D, P, P], 64), 64),
        ];
        for (i, l) in self.layers.iter().enumerate() {
            v.push(tap(format!("layer{}", i + 1), l.output(), at(l.dims, l.c), l.c));
        }
        for (i, (r, l)) in self.reduce.iter().zip(&self.layers).enumerate() {
            v.push(tap(format!("reduce.{i}"), &r.out, at(l.dims, r.c), r.c));
        }
        for (k, blk) in self.dec.iter().enumerate() {
            v.push(tap(format!("block.{k}"), &blk.out, at(self.layers[2 - k].dims, blk.c), blk.c));
        }
        v
    }
}

/// `gn_partial`'s launch shape for `[rows, cols]`: channels per block (`cpb`, with
/// `256 / cpb` lanes per channel), rows per block (64 per lane) and blocks per sample
fn gn_layout(rows: usize, cols: usize) -> (usize, usize, usize) {
    let cpb = cols.min(256);
    let chunk = (256 / cpb * 64).min(rows).max(1);
    (cpb, chunk, rows.div_ceil(chunk))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every tensor name and shape the architecture above implies, written out independently of
    /// `Model::new`, so a mismatched export fails on the CPU before any GPU work
    fn expected_keys() -> Vec<(String, Vec<usize>)> {
        let mut k: Vec<(String, Vec<usize>)> = Vec::new();
        let bn = |k: &mut Vec<(String, Vec<usize>)>, key: &str, c: usize| {
            for f in ["weight", "bias", "running_mean", "running_var"] {
                k.push((format!("{key}.{f}"), vec![c]));
            }
        };
        let gn = |k: &mut Vec<(String, Vec<usize>)>, key: &str, c: usize| {
            k.push((format!("{key}.weight"), vec![c]));
            k.push((format!("{key}.bias"), vec![c]));
        };
        let conv = |k: &mut Vec<(String, Vec<usize>)>, key: &str, o: usize, i: usize, size: usize| {
            k.push((format!("{key}.weight"), vec![o, i, size, size, size]));
        };
        k.push(("depth_coordinate_scale".into(), vec![]));
        bn(&mut k, "canonical.normalization", 1);
        conv(&mut k, "canonical.backbone.conv1", 64, 1, 3);
        bn(&mut k, "canonical.backbone.bn1", 64);
        let mut cin = 64;
        for (l, (&planes, &nb)) in PLANES.iter().zip(&BLOCKS).enumerate() {
            for j in 0..nb {
                let key = format!("canonical.backbone.layer{}.{j}", l + 1);
                conv(&mut k, &format!("{key}.conv1"), planes, cin, 1);
                bn(&mut k, &format!("{key}.bn1"), planes);
                conv(&mut k, &format!("{key}.conv2"), planes, planes, 3);
                bn(&mut k, &format!("{key}.bn2"), planes);
                conv(&mut k, &format!("{key}.conv3"), 4 * planes, planes, 1);
                bn(&mut k, &format!("{key}.bn3"), 4 * planes);
                if j == 0 {
                    conv(&mut k, &format!("{key}.downsample.0"), 4 * planes, cin, 1);
                    bn(&mut k, &format!("{key}.downsample.1"), 4 * planes);
                }
                cin = 4 * planes;
            }
        }
        for (i, (enc, dec)) in [256, 512, 1024, 2048].into_iter().zip(DEC).enumerate() {
            conv(&mut k, &format!("canonical.decoder.channel_reduce.{i}.0"), dec, enc, 1);
            gn(&mut k, &format!("canonical.decoder.channel_reduce.{i}.1"), dec);
        }
        for (b, (cat, out)) in [(768, 256), (384, 128), (192, 64)].into_iter().enumerate() {
            let key = format!("canonical.decoder.decoder_blocks.{b}");
            conv(&mut k, &format!("{key}.conv1"), out, cat, 3);
            gn(&mut k, &format!("{key}.gn1"), out);
            conv(&mut k, &format!("{key}.conv2"), out, out, 3);
            gn(&mut k, &format!("{key}.gn2"), out);
            conv(&mut k, &format!("{key}.shortcut"), out, cat, 1);
        }
        conv(&mut k, "canonical.decoder.depth_collapse.attn_conv", 1, 64, 1);
        k.push(("canonical.decoder.depth_collapse.attn_conv.bias".into(), vec![1]));
        k.push(("canonical.decoder.logit.weight".into(), vec![1, 64, 1, 1]));
        k.push(("canonical.decoder.logit.bias".into(), vec![1]));
        k
    }

    #[test]
    fn export_matches_the_architecture() {
        let path = crate::engine::weights_dir().join("hecate-9.6um.safetensors");
        if !path.exists() {
            eprintln!("skipped: no export at {}", path.display());
            return;
        }
        let (ck, sidecar) = Checkpoint::read(&path).unwrap();
        check_contract(&sidecar).unwrap();
        for (key, shape) in expected_keys() {
            ck.take(&key, &shape).unwrap();
        }
        assert_eq!(ck.finish().unwrap(), (817, 0, 128_435_399));
        assert_eq!(crate::engine::reader_name(&path).unwrap(), "hecate-9.6um");
    }
}
