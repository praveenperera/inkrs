//! Hecate's network on Metal at a fixed batch, with `hecate.py`'s numerics at fp32 or bf16: the same layers, storage
//! types and roundings as the CUDA model (`src/hecate/model.rs`), with the MSL kernels of `hecate.metal` in place of
//! cuDNN
//!
//! Convs by shape:
//! - 1x1 stride 1: one GEMM over the channels-last activation (`hec_gemm`)
//! - 1x1 stride 2 (downsample): the even voxels (`hec_sub2`), then the GEMM
//! - 3x3x3 stride 1: zero-pad the input (`hec_pad`), then nine K = 3c matmuls over shifted row views (`hec_taps`)
//! - 3x3x3 stride 2: im2col (`hec_im2col_s2`), then one K = 27c GEMM
//!
//! BatchNorm, the residual add and the ReLU run in the conv's epilogue with `hec_affine`'s arithmetic. GroupNorm
//! convs write their raw output, which `hec_gn_partial`, `hec_gn_finalize` and `hec_gn_apply` normalise

use color_eyre::eyre::{Result, ensure, eyre};
use half::bf16;
use serde_json::Value;

use crate::hecate::{D, P, PIX, Precision, u32_extent};
use crate::metal::gpu::{Buf, Gpu};
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
/// Rows and columns of one GEMM threadgroup's output tile
const TILE: usize = 64;

// epilogue flags, as `hecate.metal`'s EPI_*
const AFFINE: u32 = 1;
const RESIDUAL: u32 = 2;
const RELU: u32 = 4;
const ROUND_MID: u32 = 8;
const PAD: u32 = 16;

/// The fields of `hecate-9.6um.json` this engine relies on, beyond its keys; the CUDA model's check, which lives in a
/// CUDA-only module
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

/// The kernels of one activation type
struct Kernels {
    gemm: &'static str,
    taps: &'static str,
    /// The tap conv with 32-row tiles, for the narrow layers whose per-plane spans 64-row tiles round up badly
    taps32: &'static str,
    pad: &'static str,
    im2col: &'static str,
    sub2: &'static str,
    gather: &'static str,
    stem: &'static str,
    gn_partial: &'static str,
    gn_finalize: &'static str,
    /// GroupNorm apply into float32, and into the activation type
    gn_apply_f32: &'static str,
    gn_apply_t: &'static str,
    up2cat: &'static str,
}

const F32: Kernels = Kernels {
    gemm: "hec_gemm_f32",
    taps: "hec_taps_f32",
    taps32: "hec_taps32_f32",
    pad: "hec_pad_f32",
    im2col: "hec_im2col_s2_f32",
    sub2: "hec_sub2_f32",
    gather: "hec_gather_f32",
    stem: "hec_stem_f32",
    gn_partial: "hec_gn_partial_f32",
    gn_finalize: "hec_gn_finalize_f32",
    gn_apply_f32: "hec_gn_apply_f32_f32",
    gn_apply_t: "hec_gn_apply_f32_f32",
    up2cat: "hec_up2cat_f32",
};

const BF16: Kernels = Kernels {
    gemm: "hec_gemm_bf16",
    taps: "hec_taps_bf16",
    taps32: "hec_taps32_bf16",
    pad: "hec_pad_bf16",
    im2col: "hec_im2col_s2_bf16",
    sub2: "hec_sub2_bf16",
    gather: "hec_gather_bf16",
    stem: "hec_stem_bf16",
    gn_partial: "hec_gn_partial_bf16",
    gn_finalize: "hec_gn_finalize_bf16",
    gn_apply_f32: "hec_gn_apply_bf16_f32",
    gn_apply_t: "hec_gn_apply_bf16_bf16",
    up2cat: "hec_up2cat_bf16",
};

/// Refuses a tensor view whose element offsets overflow the int32 extents of Metal's tensor ops
fn i32_extent(what: &str, n: usize) -> Result<()> {
    ensure!(n < 1 << 31, "{what}: {n} elements overflow the Metal tensor ops' int32 indexing; lower --batch-size");
    Ok(())
}

/// Padded rows of a `[batch, d, h, w]` volume for `hec_taps`: the zero-padded volume plus the tail its last shifted
/// views read
fn padded_rows(batch: usize, [d, h, w]: [usize; 3]) -> usize {
    let (wp, plane) = (w + 2, (h + 2) * (w + 2));
    batch * (d + 2) * plane + 2 * plane + 2 * wp + 2
}

/// A conv's epilogue: flags, the BatchNorm's scale and shift (with [`AFFINE`]) and the residual (with [`RESIDUAL`])
#[derive(Clone, Copy)]
struct Epi<'a> {
    ab: Option<&'a Buf>,
    res: Option<&'a Buf>,
    flags: u32,
}

impl Epi<'_> {
    /// The conv's own output, rounded to the activation type
    const RAW: Self = Self { ab: None, res: None, flags: 0 };

    /// The same epilogue writing the zero-padded layout a tap conv reads, so it needs no `hec_pad`
    fn padded(self) -> Self {
        Self { flags: self.flags | PAD, ..self }
    }
}

/// A conv input: a plain activation, or one already in the zero-padded layout of a tap conv
#[derive(Clone, Copy)]
enum In<'a> {
    Plain(&'a Buf),
    Padded(&'a Buf),
}

impl<'a> In<'a> {
    /// The plain activation, refusing a padded one for a conv that can't read that layout
    fn plain(self, kind: Kind) -> Result<&'a Buf> {
        match self {
            In::Plain(x) => Ok(x),
            In::Padded(_) => Err(eyre!("a padded input for a {kind:?} conv")),
        }
    }
}

/// Where a GroupNorm writes: float32 or the activation type, plain or padded for the next tap conv
struct GnOut<'a> {
    buf: &'a Buf,
    f32: bool,
    /// The `[d, h, w]` of the padded layout to write
    pad: Option<[usize; 3]>,
}

/// How a conv runs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Gemm,
    Sub2,
    Taps,
    Im2col,
}

/// One conv: its kind, channels-last weight `[o, taps, c]` in the activation type, and input `[d, h, w]`
struct Conv {
    kind: Kind,
    w: Buf,
    c: usize,
    o: usize,
    din: [usize; 3],
}

impl Conv {
    /// `key.weight` (`[o, c, k, k, k]`) as `[o, k^3, c]`, rounded to bf16 under autocast
    fn load(
        gpu: &Gpu,
        ck: &Checkpoint,
        key: &str,
        bf: bool,
        (c, o, k, stride): (usize, usize, usize, usize),
        din: [usize; 3],
    ) -> Result<Self> {
        let taps = k * k * k;
        let w = ck.take(&format!("{key}.weight"), &[o, c, k, k, k])?;
        let mut nhwc = vec![0f32; o * taps * c];
        for ko in 0..o {
            for ci in 0..c {
                for t in 0..taps {
                    nhwc[(ko * taps + t) * c + ci] = w[(ko * c + ci) * taps + t];
                }
            }
        }
        let w = if bf {
            gpu.upload(&nhwc.iter().map(|&v| bf16::from_f32(v)).collect::<Vec<_>>())?
        } else {
            gpu.upload(&nhwc)?
        };
        let kind = match (k, stride) {
            (1, 1) => Kind::Gemm,
            (1, 2) => Kind::Sub2,
            (3, 1) => Kind::Taps,
            (3, 2) => Kind::Im2col,
            _ => return Err(eyre!("{key}: no Metal conv for kernel {k} stride {stride}")),
        };
        ensure!(kind != Kind::Taps || c % 8 == 0, "{key}: {c} input channels; the tap conv needs a multiple of 8");
        Ok(Self { kind, w, c, o, din })
    }

    fn dout(&self) -> [usize; 3] {
        match self.kind {
            Kind::Gemm | Kind::Taps => self.din,
            Kind::Sub2 | Kind::Im2col => self.din.map(|v| v / 2),
        }
    }

    /// Elements of the scratch input it needs at `batch`: the padded volume, the im2col matrix or the subsampled
    /// input
    fn scratch(&self, batch: usize) -> usize {
        let vout: usize = self.dout().iter().product();
        match self.kind {
            Kind::Gemm => 0,
            Kind::Sub2 => batch * vout * self.c,
            Kind::Taps => padded_rows(batch, self.din) * self.c,
            Kind::Im2col => batch * vout * 27 * self.c,
        }
    }

    /// Checks every index space of the conv at `batch` against the kernels' u32 and the tensor ops' int32 indexing
    fn check(&self, batch: usize) -> Result<()> {
        let vout: usize = self.dout().iter().product();
        let m = batch * vout;
        i32_extent("conv input", self.scratch(batch).max(batch * self.din.iter().product::<usize>() * self.c))?;
        i32_extent("conv output", m * self.o)?;
        u32_extent("conv output", m * self.o)
    }
}

/// An eval BatchNorm3d as the epilogue's `[scale, shift]` per channel: `scale = weight / sqrt(running_var + eps)`,
/// `shift = bias - running_mean * scale`, in float64, as the CUDA model computes it
fn batch_norm(gpu: &Gpu, ck: &Checkpoint, key: &str, c: usize) -> Result<Buf> {
    let [w, b, m, v] = ["weight", "bias", "running_mean", "running_var"].map(|f| ck.take(&format!("{key}.{f}"), &[c]));
    let (w, b, m, v) = (w?, b?, m?, v?);
    let ab: Vec<[f32; 2]> = (0..c)
        .map(|i| {
            let s = w[i] as f64 / (v[i] as f64 + EPS as f64).sqrt();
            [s as f32, (b[i] as f64 - m[i] as f64 * s) as f32]
        })
        .collect();
    gpu.upload(&ab)
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
    bn1: Buf,
    conv2: Conv,
    bn2: Buf,
    conv3: Conv,
    bn3: Buf,
    downsample: Option<(Conv, Buf)>,
}

/// One encoder layer and its buffers: two outputs used in turn, the last block's being the layer's feature map
struct Layer {
    blocks: Vec<Bottleneck>,
    /// conv1's output for a stride-2 conv2 (im2col), and padded for a tap conv2, zeroed once so its borders stay
    /// zero
    t1: Buf,
    t1p: Buf,
    t2: Buf,
    ds: Buf,
    out: [Buf; 2],
    vox: usize,
    c: usize,
    dims: [usize; 3],
}

impl Layer {
    fn output(&self) -> &Buf {
        &self.out[(self.blocks.len() - 1) % 2]
    }
}

/// One `channel_reduce`: 1x1 conv, GroupNorm, ReLU, into float32
struct Reduce {
    conv: Conv,
    gn: GroupNorm,
    out: Buf,
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
    cat: Buf,
    /// gn1's output, padded for conv2 and zeroed once
    tp: Buf,
    sc: Buf,
    out: Buf,
    /// Input `[d, h, w]` of the deeper map, its channels, the skip's and the block's output channels
    low: [usize; 3],
    cl: usize,
    cs: usize,
    c: usize,
}

/// The network and its activations at batch `batch`
pub struct Model {
    pub batch: usize,
    pub precision: Precision,
    k: &'static Kernels,
    /// Bytes per activation element
    size: usize,
    norm: [f32; 4],
    stem_w: Buf,
    stem_bn: Buf,
    layers: Vec<Layer>,
    reduce: Vec<Reduce>,
    dec: Vec<DecBlock>,
    head_wl: Buf,
    head_bl: f32,
    head_wa: Buf,
    head_ba: f32,
    dcs: f32,
    x0: Buf,
    support: Buf,
    a0: Buf,
    /// A conv's raw output before its GroupNorm
    raw: Buf,
    /// A conv's padded, im2col or subsampled input
    scratch: Buf,
    partial: Buf,
    ab: Buf,
    /// Bound to the epilogue's unused residual and scale parameters
    unused: Buf,
}

impl Model {
    /// Loads every layer at `batch`; every checkpoint tensor must be consumed (the caller checks with
    /// [`Checkpoint::finish`])
    pub fn new(gpu: &Gpu, ck: &Checkpoint, batch: usize, precision: Precision) -> Result<Self> {
        ensure!(batch > 0, "batch size must be positive");
        let bf = precision.bf16();
        let size = if bf { 2 } else { 4 };
        // NaN-filled, so an element no kernel writes fails the finite check
        let act = |n: usize, elem: usize| -> Result<Buf> {
            u32_extent("activation", n)?;
            gpu.alloc_fill(n * elem, 0xFF)
        };
        let vox0 = D * PIX;

        let nk = "canonical.normalization";
        let norm = [
            ck.take(&format!("{nk}.running_mean"), &[1])?[0],
            ck.take(&format!("{nk}.running_var"), &[1])?[0],
            ck.take(&format!("{nk}.weight"), &[1])?[0],
            ck.take(&format!("{nk}.bias"), &[1])?[0],
        ];

        // the stem's weights stay float32 for its SIMT kernel, holding the values autocast's bf16 cast gives
        let stem = ck.take("canonical.backbone.conv1.weight", &[64, 1, 3, 3, 3])?;
        let stem: Vec<f32> = if bf { stem.iter().map(|&v| bf16::from_f32(v).to_f32()).collect() } else { stem };
        let stem_w = gpu.upload(&stem)?;
        let stem_bn = batch_norm(gpu, ck, "canonical.backbone.bn1", 64)?;

        let mut convs_scratch = 0usize;
        let mut raw_len = 0usize;
        let mut layers = Vec::new();
        let (mut cin, mut dims) = (64usize, [D, P, P]);
        for (l, (&planes, &nb)) in PLANES.iter().zip(&BLOCKS).enumerate() {
            let stride = if l == 0 { 1 } else { 2 };
            let dims_in = dims;
            let dims_out = dims.map(|v| v / stride);
            let (vin, vout) = (dims_in.iter().product::<usize>(), dims_out.iter().product::<usize>());
            let c = planes * 4;
            let mut blocks = Vec::new();
            for j in 0..nb {
                let key = format!("canonical.backbone.layer{}.{j}", l + 1);
                let (s, din, bin) = if j == 0 { (stride, dims_in, cin) } else { (1, dims_out, c) };
                let conv1 = Conv::load(gpu, ck, &format!("{key}.conv1"), bf, (bin, planes, 1, 1), din)?;
                let conv2 = Conv::load(gpu, ck, &format!("{key}.conv2"), bf, (planes, planes, 3, s), din)?;
                let conv3 = Conv::load(gpu, ck, &format!("{key}.conv3"), bf, (planes, c, 1, 1), dims_out)?;
                let downsample = if j == 0 {
                    let conv = Conv::load(gpu, ck, &format!("{key}.downsample.0"), bf, (bin, c, 1, s), din)?;
                    Some((conv, batch_norm(gpu, ck, &format!("{key}.downsample.1"), c)?))
                } else {
                    None
                };
                for conv in
                    [Some(&conv1), Some(&conv2), Some(&conv3), downsample.as_ref().map(|d| &d.0)].into_iter().flatten()
                {
                    conv.check(batch)?;
                    convs_scratch = convs_scratch.max(conv.scratch(batch));
                }
                blocks.push(Bottleneck {
                    conv1,
                    bn1: batch_norm(gpu, ck, &format!("{key}.bn1"), planes)?,
                    conv2,
                    bn2: batch_norm(gpu, ck, &format!("{key}.bn2"), planes)?,
                    conv3,
                    bn3: batch_norm(gpu, ck, &format!("{key}.bn3"), c)?,
                    downsample,
                });
            }
            layers.push(Layer {
                blocks,
                t1: act(batch * vin * planes, size)?,
                t1p: gpu.alloc(padded_rows(batch, dims_out) * planes * size)?,
                t2: act(batch * vout * planes, size)?,
                ds: act(batch * vout * c, size)?,
                out: [act(batch * vout * c, size)?, act(batch * vout * c, size)?],
                vox: vout,
                c,
                dims: dims_out,
            });
            cin = c;
            dims = dims_out;
        }

        let mut reduce = Vec::new();
        for (i, layer) in layers.iter().enumerate() {
            let key = format!("canonical.decoder.channel_reduce.{i}");
            let conv = Conv::load(gpu, ck, &format!("{key}.0"), bf, (layer.c, DEC[i], 1, 1), layer.dims)?;
            conv.check(batch)?;
            raw_len = raw_len.max(batch * layer.vox * DEC[i]);
            reduce.push(Reduce {
                conv,
                gn: GroupNorm::load(gpu, ck, &format!("{key}.1"), DEC[i])?,
                out: act(batch * layer.vox * DEC[i], 4)?,
                vox: layer.vox,
                c: DEC[i],
            });
        }
        let mut dec = Vec::new();
        for k in 0..3 {
            // block k joins the deeper map (reduce[3] or block k-1) with reduce[2 - k]
            let (deep, skip) = (3 - k, 2 - k);
            let (cl, cs, c) = (DEC[deep], DEC[skip], DEC[skip]);
            let d = layers[skip].dims;
            let vox = layers[skip].vox;
            let key = format!("canonical.decoder.decoder_blocks.{k}");
            let conv1 = Conv::load(gpu, ck, &format!("{key}.conv1"), bf, (cl + cs, c, 3, 1), d)?;
            let conv2 = Conv::load(gpu, ck, &format!("{key}.conv2"), bf, (c, c, 3, 1), d)?;
            let shortcut = Conv::load(gpu, ck, &format!("{key}.shortcut"), bf, (cl + cs, c, 1, 1), d)?;
            for conv in [&conv1, &conv2, &shortcut] {
                conv.check(batch)?;
                convs_scratch = convs_scratch.max(conv.scratch(batch));
            }
            raw_len = raw_len.max(batch * vox * c);
            dec.push(DecBlock {
                conv1,
                gn1: GroupNorm::load(gpu, ck, &format!("{key}.gn1"), c)?,
                conv2,
                gn2: GroupNorm::load(gpu, ck, &format!("{key}.gn2"), c)?,
                shortcut,
                cat: act(batch * vox * (cl + cs), size)?,
                tp: gpu.alloc(padded_rows(batch, d) * c * size)?,
                sc: act(batch * vox * c, size)?,
                out: act(batch * vox * c, 4)?,
                low: layers[deep].dims,
                cl,
                cs,
                c,
            });
        }
        ensure!(dec[2].c == 64 && layers[0].dims == [D, P, P], "decoder ends at {} channels", dec[2].c);

        let head_wl = gpu.upload(&ck.take("canonical.decoder.logit.weight", &[1, 64, 1, 1])?)?;
        let head_bl = ck.scalar("canonical.decoder.logit.bias")?;
        let head_wa = gpu.upload(&ck.take("canonical.decoder.depth_collapse.attn_conv.weight", &[1, 64, 1, 1, 1])?)?;
        let head_ba = ck.scalar("canonical.decoder.depth_collapse.attn_conv.bias")?;
        let dcs = ck.scalar0("depth_coordinate_scale")?;

        let mut gn_shapes: Vec<(usize, usize)> = reduce.iter().map(|r| (r.vox, r.c)).collect();
        gn_shapes.extend(dec.iter().map(|b| (8 * b.low.iter().product::<usize>(), b.c)));
        let partial_len = gn_shapes.iter().map(|&(rows, cols)| gn_layout(rows, cols).2 * cols).max().unwrap_or(0);
        let max_cols = gn_shapes.iter().map(|&(_, c)| c).max().unwrap_or(0);
        u32_extent("conv scratch", convs_scratch)?;

        Ok(Self {
            batch,
            precision,
            k: if bf { &BF16 } else { &F32 },
            size,
            norm,
            stem_w,
            stem_bn,
            reduce,
            dec,
            head_wl,
            head_bl,
            head_wa,
            head_ba,
            dcs,
            x0: act(batch * vox0, size)?,
            support: gpu.alloc_fill(batch * PIX, 0xFF)?,
            a0: act(batch * vox0 * 64, size)?,
            raw: act(raw_len.max(1), size)?,
            scratch: gpu.alloc_fill(convs_scratch.max(1) * size, 0xFF)?,
            partial: gpu.alloc(batch * partial_len * 8)?,
            ab: gpu.alloc(batch * max_cols * 8)?,
            unused: gpu.alloc(16)?,
            layers,
        })
    }

    /// `conv(x)` into `y` with the epilogue `epi`
    fn conv(&self, gpu: &Gpu, conv: &Conv, x: In, y: &Buf, Epi { ab, res, flags }: Epi) -> Result<()> {
        let b = self.batch;
        let [d, h, w] = conv.din;
        let vout: usize = conv.dout().iter().product();
        let m = b * vout;
        let need = |buf: &Buf, n: usize, what: &str| {
            ensure!(buf.bytes() >= n * self.size, "{what}: {} bytes for {n} elements", buf.bytes());
            Ok(())
        };
        match x {
            In::Plain(x) => need(x, b * d * h * w * conv.c, "conv input")?,
            In::Padded(xp) => {
                ensure!(conv.kind == Kind::Taps, "a padded input for a {:?} conv", conv.kind);
                need(xp, padded_rows(b, conv.din) * conv.c, "padded conv input")?;
            }
        }
        let [pd, ph, pw] = conv.dout();
        if flags & PAD != 0 {
            // only the GEMM epilogue writes the padded layout
            ensure!(conv.kind == Kind::Gemm && res.is_none(), "a padded output from a {:?} conv", conv.kind);
            need(y, padded_rows(b, conv.dout()) * conv.o, "padded conv output")?;
        } else {
            need(y, m * conv.o, "conv output")?;
        }
        if let Some(r) = res {
            need(r, m * conv.o, "conv residual")?;
        }
        ensure!(ab.is_some() == (flags & AFFINE != 0), "conv epilogue scale without the affine flag");
        ensure!(res.is_some() == (flags & RESIDUAL != 0), "conv epilogue residual without the residual flag");
        let ab = ab.unwrap_or(&self.unused).arg();
        let res = res.unwrap_or(&self.unused).arg();
        let gemm = |input: &Buf, k: usize| {
            gpu.dispatch(
                self.k.gemm,
                [conv.o.div_ceil(TILE), m.div_ceil(TILE), 1],
                [128, 1, 1],
                &[
                    input.arg(),
                    conv.w.arg(),
                    y.arg(),
                    ab,
                    res,
                    m.into(),
                    conv.o.into(),
                    k.into(),
                    k.into(),
                    flags.into(),
                    pd.into(),
                    ph.into(),
                    pw.into(),
                ],
            )
        };
        match conv.kind {
            Kind::Gemm => gemm(x.plain(conv.kind)?, conv.c),
            Kind::Sub2 => {
                let x = x.plain(conv.kind)?;
                gpu.dispatch_1d(
                    self.k.sub2,
                    m * conv.c,
                    &[self.scratch.arg(), x.arg(), d.into(), h.into(), w.into(), conv.c.into(), (m * conv.c).into()],
                )?;
                gemm(&self.scratch, conv.c)
            }
            Kind::Im2col => {
                let x = x.plain(conv.kind)?;
                let n = m * 27 * conv.c;
                gpu.dispatch_1d(
                    self.k.im2col,
                    n,
                    &[self.scratch.arg(), x.arg(), d.into(), h.into(), w.into(), conv.c.into(), n.into()],
                )?;
                gemm(&self.scratch, 27 * conv.c)
            }
            Kind::Taps => {
                let xp = match x {
                    In::Padded(xp) => xp,
                    In::Plain(x) => {
                        let n = padded_rows(b, conv.din) * conv.c;
                        ensure!(n * self.size <= self.scratch.bytes(), "padded conv input past the scratch buffer");
                        gpu.dispatch_1d(
                            self.k.pad,
                            n,
                            &[
                                self.scratch.arg(),
                                x.arg(),
                                b.into(),
                                d.into(),
                                h.into(),
                                w.into(),
                                conv.c.into(),
                                n.into(),
                            ],
                        )?;
                        &self.scratch
                    }
                };
                // one run of row tiles per output (sample, z), as hec_taps maps them, in whichever tile height wastes
                // fewer rows (64 on a tie)
                let span = h * (w + 2);
                let (name, bm) = if span.div_ceil(32) * 32 < span.div_ceil(TILE) * TILE {
                    (self.k.taps32, 32)
                } else {
                    (self.k.taps, TILE)
                };
                gpu.dispatch(
                    name,
                    [conv.o.div_ceil(TILE), b * d * span.div_ceil(bm), 1],
                    [128, 1, 1],
                    &[
                        xp.arg(),
                        conv.w.arg(),
                        y.arg(),
                        ab,
                        res,
                        b.into(),
                        d.into(),
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

    /// The BatchNorm epilogue, with the optional residual and ReLU; autocast's BatchNorm writes bf16 before the
    /// residual add
    fn bn<'a>(&self, ab: &'a Buf, res: Option<&'a Buf>, relu: bool) -> Epi<'a> {
        let flags = AFFINE
            | if self.precision.bf16() { ROUND_MID } else { 0 }
            | if res.is_some() { RESIDUAL } else { 0 }
            | if relu { RELU } else { 0 };
        Epi { ab: Some(ab), res, flags }
    }

    /// GroupNorm(min(32, c)) of the conv output in `raw` (`[B, rows, cols]`) into `y`, with the optional residual and
    /// a ReLU
    fn gn(
        &self,
        gpu: &Gpu,
        y: GnOut,
        g: &GroupNorm,
        (rows, cols): (usize, usize),
        residual: Option<&Buf>,
    ) -> Result<()> {
        let b = self.batch;
        let n = b * rows * cols;
        let (cpb, chunk, chunks) = gn_layout(rows, cols);
        // hec_gn_partial gives each threadgroup `256 / cpb` lanes per channel and returns early otherwise
        ensure!(256 % cpb == 0 && cols % cpb == 0, "GroupNorm over {cols} channels has no hec_gn_partial layout");
        ensure!(
            b * chunks * cols * 8 <= self.partial.bytes()
                && b * cols * 8 <= self.ab.bytes()
                && n * self.size <= self.raw.bytes()
                && y.pad.map_or(n, |dims| padded_rows(b, dims) * cols) * if y.f32 { 4 } else { self.size }
                    <= y.buf.bytes(),
            "GroupNorm buffers too small"
        );
        let [pd, ph, pw] = y.pad.unwrap_or([1, 1, 1]);
        ensure!(y.pad.is_none_or(|_| pd * ph * pw == rows), "GroupNorm padded output of {:?} for {rows} rows", y.pad);
        gpu.dispatch(
            self.k.gn_partial,
            [chunks, b, cols / cpb],
            [256, 1, 1],
            &[self.partial.arg(), self.raw.arg(), rows.into(), cols.into(), cpb.into(), chunk.into()],
        )?;
        gpu.dispatch_1d(
            self.k.gn_finalize,
            b * cols,
            &[
                self.ab.arg(),
                self.partial.arg(),
                self.raw.arg(),
                g.gamma.arg(),
                g.beta.arg(),
                rows.into(),
                cols.into(),
                chunks.into(),
                cols.min(32).into(),
                EPS.into(),
                (b * cols).into(),
            ],
        )?;
        let flags = RELU | if residual.is_some() { RESIDUAL } else { 0 } | if y.pad.is_some() { PAD } else { 0 };
        gpu.dispatch_1d(
            if y.f32 { self.k.gn_apply_f32 } else { self.k.gn_apply_t },
            n,
            &[
                y.buf.arg(),
                self.raw.arg(),
                residual.unwrap_or(&self.unused).arg(),
                self.ab.arg(),
                cols.into(),
                (rows * cols).into(),
                flags.into(),
                n.into(),
                pd.into(),
                ph.into(),
                pw.into(),
            ],
        )
    }

    /// Queues one forward: gather and normalise the `batch` tiles of `desc` (the descriptors from element `desc_at`)
    /// from the resident `vol`, run the network, and write each real tile's probability times its support into its
    /// slot of `prob` (`[slots, 64 * 64]` float32)
    pub fn forward(
        &self,
        gpu: &Gpu,
        (vol, height, width): (&Buf, usize, usize),
        desc: (&Buf, usize),
        prob: &Buf,
    ) -> Result<()> {
        let b = self.batch;
        let vox0 = D * PIX;
        let (desc, at) = desc;
        ensure!((at + b * DESC) * 4 <= desc.bytes(), "descriptors {at}.. past the end of the buffer for batch {b}");
        let desc = desc.at::<i32>(at);
        let slots = u32::try_from(prob.bytes() / (4 * PIX)).map_err(|_| eyre!("too many probability slots"))?;

        let [mean, var, gamma, beta] = self.norm;
        // PyTorch's eval BatchNorm: (x - mean) * invstd * weight + bias, invstd = 1 / sqrt(var + eps)
        let invstd = 1.0 / (var + EPS).sqrt();
        gpu.dispatch_1d(
            self.k.gather,
            b * PIX,
            &[
                self.x0.arg(),
                self.support.arg(),
                vol.arg(),
                desc,
                height.into(),
                width.into(),
                mean.into(),
                invstd.into(),
                gamma.into(),
                beta.into(),
                (b * PIX).into(),
            ],
        )?;
        let stem_flags = self.bn(&self.stem_bn, None, true).flags;
        gpu.dispatch_1d(
            self.k.stem,
            b * vox0 * 4,
            &[
                self.a0.arg(),
                self.x0.arg(),
                self.stem_w.arg(),
                self.stem_bn.arg(),
                stem_flags.into(),
                (b * vox0 * 4).into(),
            ],
        )?;

        let mut x = &self.a0;
        for layer in &self.layers {
            for (j, blk) in layer.blocks.iter().enumerate() {
                let out = &layer.out[j % 2];
                let residual = match &blk.downsample {
                    Some((conv, bn)) => {
                        self.conv(gpu, conv, In::Plain(x), &layer.ds, self.bn(bn, None, false))?;
                        &layer.ds
                    }
                    None => x,
                };
                // a tap conv2 reads conv1's output padded, written so by conv1's epilogue
                let bn1 = self.bn(&blk.bn1, None, true);
                let t1 = if blk.conv2.kind == Kind::Taps {
                    self.conv(gpu, &blk.conv1, In::Plain(x), &layer.t1p, bn1.padded())?;
                    In::Padded(&layer.t1p)
                } else {
                    self.conv(gpu, &blk.conv1, In::Plain(x), &layer.t1, bn1)?;
                    In::Plain(&layer.t1)
                };
                self.conv(gpu, &blk.conv2, t1, &layer.t2, self.bn(&blk.bn2, None, true))?;
                self.conv(gpu, &blk.conv3, In::Plain(&layer.t2), out, self.bn(&blk.bn3, Some(residual), true))?;
                x = out;
            }
        }

        for (r, layer) in self.reduce.iter().zip(&self.layers) {
            self.conv(gpu, &r.conv, In::Plain(layer.output()), &self.raw, Epi::RAW)?;
            self.gn(gpu, GnOut { buf: &r.out, f32: true, pad: None }, &r.gn, (r.vox, r.c), None)?;
        }
        for (k, blk) in self.dec.iter().enumerate() {
            let low = if k == 0 { &self.reduce[3].out } else { &self.dec[k - 1].out };
            let skip = &self.reduce[2 - k].out;
            let [d, h, w] = blk.low;
            let vox = 8 * d * h * w;
            let ncat = b * vox * (blk.cl + blk.cs);
            ensure!(
                low.bytes() >= b * d * h * w * blk.cl * 4 && skip.bytes() >= b * vox * blk.cs * 4,
                "decoder block {k}: upsample inputs too small"
            );
            gpu.dispatch_1d(
                self.k.up2cat,
                ncat,
                &[
                    blk.cat.arg(),
                    low.arg(),
                    skip.arg(),
                    d.into(),
                    h.into(),
                    w.into(),
                    blk.cl.into(),
                    blk.cs.into(),
                    ncat.into(),
                ],
            )?;
            self.conv(gpu, &blk.shortcut, In::Plain(&blk.cat), &blk.sc, Epi::RAW)?;
            self.conv(gpu, &blk.conv1, In::Plain(&blk.cat), &self.raw, Epi::RAW)?;
            // gn1 writes conv2's padded input directly
            let tp = GnOut { buf: &blk.tp, f32: false, pad: Some(blk.conv2.din) };
            self.gn(gpu, tp, &blk.gn1, (vox, blk.c), None)?;
            self.conv(gpu, &blk.conv2, In::Padded(&blk.tp), &self.raw, Epi::RAW)?;
            self.gn(gpu, GnOut { buf: &blk.out, f32: true, pad: None }, &blk.gn2, (vox, blk.c), Some(&blk.sc))?;
        }

        let feat = &self.dec[2].out;
        ensure!(feat.bytes() >= b * vox0 * 64 * 4, "head features too small");
        gpu.dispatch_1d(
            "hec_head",
            b * PIX,
            &[
                prob.arg(),
                feat.arg(),
                self.support.arg(),
                self.head_wl.arg(),
                self.head_bl.into(),
                self.head_wa.arg(),
                self.head_ba.into(),
                self.dcs.into(),
                desc,
                u32::from(self.precision.bf16()).into(),
                (b * PIX).into(),
                slots.into(),
            ],
        )
    }
}

/// `hec_gn_partial`'s launch shape for `[rows, cols]`: channels per threadgroup (`cpb`, with `256 / cpb` lanes per
/// channel), rows per threadgroup (64 per lane) and threadgroups per sample, as the CUDA model's
fn gn_layout(rows: usize, cols: usize) -> (usize, usize, usize) {
    let cpb = cols.min(256);
    let chunk = (256 / cpb * 64).min(rows).max(1);
    (cpb, chunk, rows.div_ceil(chunk))
}

#[cfg(test)]
mod tests {
    //! GroupNorm statistics against an f64 reference, on inputs where float32 moments lose the variance or overflow

    use super::*;

    /// Runs hec_gn_partial + hec_gn_finalize (fp32) over x [1, rows, cols] with unit gamma and zero beta; returns the
    /// per-channel (a, b) of y = a x + b
    fn gn_ab(gpu: &Gpu, x: &[f32], rows: usize, cols: usize) -> Vec<[f32; 2]> {
        let (cpb, chunk, chunks) = gn_layout(rows, cols);
        // the layout Model::gn requires; hec_gn_partial skips channels otherwise
        assert!(256 % cpb == 0 && cols.is_multiple_of(cpb), "no hec_gn_partial layout for {cols} channels");
        let partial = gpu.alloc(chunks * cols * 8).unwrap();
        let ab = gpu.alloc(cols * 8).unwrap();
        let raw = gpu.upload(x).unwrap();
        let gamma = gpu.upload(&vec![1f32; cols]).unwrap();
        let beta = gpu.upload(&vec![0f32; cols]).unwrap();
        gpu.dispatch(
            F32.gn_partial,
            [chunks, 1, cols / cpb],
            [256, 1, 1],
            &[partial.arg(), raw.arg(), rows.into(), cols.into(), cpb.into(), chunk.into()],
        )
        .unwrap();
        gpu.dispatch_1d(
            F32.gn_finalize,
            cols,
            &[
                ab.arg(),
                partial.arg(),
                raw.arg(),
                gamma.arg(),
                beta.arg(),
                rows.into(),
                cols.into(),
                chunks.into(),
                cols.min(32).into(),
                EPS.into(),
                cols.into(),
            ],
        )
        .unwrap();
        gpu.synchronize().unwrap();

        let mut out = vec![0f32; cols * 2];
        gpu.download(&ab, 0, &mut out).unwrap();
        out.chunks(2).map(|p| [p[0], p[1]]).collect()
    }

    /// The f64 (a, b) of every channel, groups of cols / 32 channels as CUDA's float64 finalize computes them
    fn gn_ref(x: &[f32], rows: usize, cols: usize) -> Vec<[f64; 2]> {
        let cg = cols / cols.min(32);
        (0..cols)
            .map(|c| {
                let g0 = c / cg * cg;
                let vals = || (0..rows).flat_map(move |r| (g0..g0 + cg).map(move |cc| x[r * cols + cc] as f64));
                let n = (rows * cg) as f64;
                let mean = vals().sum::<f64>() / n;
                let var = vals().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
                let a = 1.0 / (var + EPS as f64).sqrt();
                [a, -a * mean]
            })
            .collect()
    }

    fn check(name: &str, x: &[f32], rows: usize, cols: usize) {
        let gpu = Gpu::open(0).unwrap();
        let got = gn_ab(&gpu, x, rows, cols);
        let want = gn_ref(x, rows, cols);
        for (c, (g, w)) in got.iter().zip(&want).enumerate() {
            for k in 0..2 {
                let err = (g[k] as f64 - w[k]).abs() / w[k].abs().max(1.0);
                assert!(g[k].is_finite() && err < 1e-5, "{name}: channel {c} term {k}: {} vs {}", g[k], w[k]);
            }
        }
    }

    #[test]
    fn group_norm_keeps_the_low_square_of_the_mean() {
        // channel pairs one ulp apart at 65536: the variance sits in the low word of the double-float mean
        let (rows, cols) = (65536, 64);
        let next = f32::from_bits(65536f32.to_bits() + 1);
        let x: Vec<f32> = (0..rows * cols).map(|i| if i % 2 == 0 { 65536.0 } else { next }).collect();
        check("near-constant", &x, rows, cols);
    }

    #[test]
    fn group_norm_stays_finite_on_large_constant_input() {
        let (rows, cols) = (65536, 64);
        check("constant 1e17", &vec![1e17f32; rows * cols], rows, cols);
    }

    #[test]
    fn group_norm_matches_reference_on_offset_random_input() {
        // per-channel offsets so the group's channels are centred on different shifts
        let (rows, cols) = (4096, 128);
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0 + (i % cols) as f32 * 37.0
            })
            .collect();
        check("offset random", &x, rows, cols);
    }
}
