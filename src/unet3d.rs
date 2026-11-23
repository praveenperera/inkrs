//! villa's autoconfigured 3D ResEnc U-Net (`NetworkFromConfig`, nnU-Net style) for whole-volume 3D ink
//! (`inkrs ink3d`), first for `ink_3d_dino_guided` v3
//!
//! `scripts/export_unet3d.py` exports a checkpoint's weights and its layer program: the stem, the
//! BasicBlockD encoder stages, the transposed-conv and ConvBlock decoder stages and the 1x1 head, each
//! conv with its tensor names, channels, kernel and stride. [`model::Model`] builds the network from
//! that program, so any checkpoint inside the op set the export accepts runs without code changes.
//! The forward is bf16 channels-last at batch 1: cuDNN runs the convs, cuBLAS the transposed convs as
//! GEMMs, and the `u3_*` kernels the InstanceNorm, LeakyReLU, residuals, pooling, concat, input
//! normalisation, head and blend. [`engine`] runs villa's patch plan over a zarr region with the
//! accumulators on the GPU and writes villa's uint8 output zarr

use color_eyre::eyre::{Result, bail};
use cudakit::{ComputeCapability, PtxTier};
use serde::Deserialize;

use crate::ensure_supported;
use crate::error::Unsupported;

pub mod engine;
pub mod model;
pub mod p0;
pub mod zio;

/// The sidecar format this engine reads
pub const FORMAT: &str = "inkrs-unet3d-v1";

/// The precision autocast runs the network in: the activations' storage type, the weights' rounding, the
/// cuDNN and cuBLAS data type and the glue kernels (`u3_*` for bf16, `u3h_*` for fp16)
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    Fp16,
    Bf16,
}

impl Precision {
    /// `v` rounded to the storage type, as raw bits
    pub fn bits(self, v: f32) -> u16 {
        match self {
            Self::Bf16 => half::bf16::from_f32(v).to_bits(),
            Self::Fp16 => half::f16::from_f32(v).to_bits(),
        }
    }

    /// Raw storage bits as float32
    pub fn value(self, bits: u16) -> f32 {
        match self {
            Self::Bf16 => half::bf16::from_bits(bits).to_f32(),
            Self::Fp16 => half::f16::from_bits(bits).to_f32(),
        }
    }

    /// `v` as autocast casts it: rounded to the storage type, kept as float32
    pub fn round(self, v: f32) -> f32 {
        self.value(self.bits(v))
    }
}

/// An InstanceNorm's affine tensors and eps
#[derive(Clone, Debug, Deserialize)]
pub struct NormSpec {
    pub w: String,
    pub b: String,
    pub eps: f32,
}

/// One conv, with the norm and activation that follow it when it is a ConvDropoutNormReLU
#[derive(Clone, Debug, Deserialize)]
pub struct ConvSpec {
    pub w: String,
    pub b: Option<String>,
    pub cin: usize,
    pub cout: usize,
    /// z, y, x: a sidecar giving another axis count fails to parse
    pub kernel: [usize; 3],
    pub stride: [usize; 3],
    pub norm: Option<NormSpec>,
    /// LeakyReLU's negative slope, or `None` for no activation
    pub leaky: Option<f32>,
}

/// A BasicBlockD skip: optional average pool (kernel = stride), then optional bias-free 1x1 conv + norm
#[derive(Clone, Debug, Deserialize)]
pub struct SkipSpec {
    pub pool: Pool,
    pub proj: Option<ConvSpec>,
}

/// A skip's average-pool stride: one value for every axis (`inkrs-unet3d-v1`) or one per axis
/// (`inkrs-nnunet-v1`); 1 means no pooling on that axis
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum Pool {
    Iso(usize),
    Axes([usize; 3]),
}

impl Pool {
    /// The stride per axis, z, y, x
    pub fn axes(self) -> [usize; 3] {
        match self {
            Self::Iso(v) => [v; 3],
            Self::Axes(a) => a,
        }
    }
}

/// One BasicBlockD: `leaky(norm(conv2(leaky(norm(conv1(x))))) + skip(x))`
#[derive(Clone, Debug, Deserialize)]
pub struct BlockSpec {
    pub conv1: ConvSpec,
    pub conv2: ConvSpec,
    pub skip: SkipSpec,
    pub leaky: f32,
}

/// One decoder stage: k2 s2 transposed conv, concat `(up, skip)`, then its convs
#[derive(Clone, Debug, Deserialize)]
pub struct DecSpec {
    pub transp: ConvSpec,
    pub convs: Vec<ConvSpec>,
}

/// The whole layer program
#[derive(Clone, Debug, Deserialize)]
pub struct Program {
    pub stem: Vec<ConvSpec>,
    pub stages: Vec<Vec<BlockSpec>>,
    pub decoder: Vec<DecSpec>,
    pub head: ConvSpec,
}

/// villa's `percentile_minmax` bounds, in percent
#[derive(Clone, Debug, Deserialize)]
pub struct NormalizationSpec {
    pub mode: String,
    pub lower: f64,
    pub upper: f64,
}

/// villa's Gaussian importance map as its three float32 factors
#[derive(Clone, Debug, Deserialize)]
pub struct BlendSpec {
    pub mode: String,
    pub sigma_scale: f64,
    pub axes_f32_bits: Vec<Vec<u32>>,
}

/// The export's contract
#[derive(Clone, Debug, Deserialize)]
pub struct Contract {
    pub format: String,
    pub patch_size: Vec<usize>,
    pub in_channels: usize,
    pub amp_dtype: Option<String>,
    pub normalization: NormalizationSpec,
    pub program: Program,
    pub blend: BlendSpec,
}

/// Fails with [`Unsupported`] on a device the engine can't run on: its convs are cuDNN bf16, which
/// needs compute capability 8.0, and the sm75 PTX tier (also forced with `INKRS_PTX_TIER=sm75`) leaves
/// its kernels out
pub fn check_device(cc: ComputeCapability, tier: PtxTier) -> Result<()> {
    ensure_supported!(
        cc >= ComputeCapability::new(8, 0) && tier != PtxTier::Sm75,
        "the 3D U-Net engines (ink3d, nnunet) need a GPU of compute capability 8.0 or above and the sm80 or sm120 \
         PTX tier, for their cuDNN bf16 and fp16 convs; this GPU is {cc} on the {tier} tier"
    );
    Ok(())
}

impl Contract {
    /// Parses and checks the sidecar: the format, a cubic patch, one channel, bf16 autocast,
    /// `percentile_minmax`, a Gaussian blend, and a program whose channels chain stage to stage
    pub fn from_sidecar(sidecar: &serde_json::Value) -> Result<Self> {
        let c: Contract =
            serde_json::from_value(sidecar.clone()).map_err(|e| Unsupported(format!("not an {FORMAT} export: {e}")))?;
        ensure_supported!(c.format == FORMAT, "sidecar format {:?}, expected {FORMAT}", c.format);
        let p = c.patch_size.first().copied().unwrap_or(0);
        ensure_supported!(
            c.patch_size.len() == 3 && c.patch_size.iter().all(|&v| v == p) && p > 0,
            "patch size {:?}, only cubic",
            c.patch_size
        );
        ensure_supported!(c.in_channels == 1, "{} input channels, only 1", c.in_channels);
        ensure_supported!(c.amp_dtype.as_deref() == Some("bf16"), "amp dtype {:?}, only bf16", c.amp_dtype);
        ensure_supported!(c.normalization.mode == "percentile_minmax", "normalization {:?}", c.normalization.mode);
        ensure_supported!(
            c.blend.mode == "gaussian"
                && c.blend.axes_f32_bits.len() == 3
                && c.blend.axes_f32_bits.iter().all(|a| a.len() == p),
            "blend {:?} with {} axes",
            c.blend.mode,
            c.blend.axes_f32_bits.len()
        );
        c.program.check(p)?;
        Ok(c)
    }

    /// Patch edge in voxels
    pub fn patch(&self) -> usize {
        self.patch_size[0]
    }

    /// villa's importance map `[p, p, p]`: `max((a0 * a1) * a2 / peak, eps)` in float32, the export
    /// having checked this rebuild against villa's own map
    pub fn importance_map(&self) -> Vec<f32> {
        let p = self.patch();
        let ax: Vec<Vec<f32>> =
            self.blend.axes_f32_bits.iter().map(|a| a.iter().map(|&b| f32::from_bits(b)).collect()).collect();
        let mut m = Vec::with_capacity(p * p * p);
        for &az in &ax[0] {
            for &ay in &ax[1] {
                let zy = az * ay;
                m.extend(ax[2].iter().map(|&ax2| zy * ax2));
            }
        }
        let peak = m.iter().copied().fold(f32::MIN, f32::max);
        for v in &mut m {
            *v = (*v / peak).max(f32::EPSILON);
        }
        m
    }
}

impl ConvSpec {
    /// The kernel and stride as three axes, each kernel odd "same" and each stride 1 or 2 when `odd`
    fn check(&self, what: &str, cin: usize, odd: bool) -> Result<()> {
        ensure_supported!(self.cin == cin, "{what}: {} input channels, the previous layer gives {cin}", self.cin);
        if odd {
            ensure_supported!(
                self.kernel.iter().all(|&k| k % 2 == 1) && self.stride.iter().all(|&s| s == 1 || s == 2),
                "{what}: kernel {:?} stride {:?}",
                self.kernel,
                self.stride
            );
        }
        ensure_supported!(
            self.cout.is_multiple_of(2) || self.norm.is_none(),
            "{what}: odd channel count {}",
            self.cout
        );
        if let Some(n) = &self.norm {
            ensure_supported!(n.eps > 0.0 && n.eps.is_finite(), "{what}: eps {}", n.eps);
        }
        Ok(())
    }

    /// Whether kernel and stride are the same on every axis
    fn isotropic(&self) -> bool {
        self.kernel.iter().all(|&v| v == self.kernel[0]) && self.stride.iter().all(|&v| v == self.stride[0])
    }

    /// Multiply-accumulates on an input of `shape`, and the output shape ("same" padding)
    fn macs(&self, shape: [usize; 3]) -> (u64, [usize; 3]) {
        let out: [usize; 3] = std::array::from_fn(|a| shape[a].div_ceil(self.stride[a]));
        let taps: usize = self.kernel.iter().product();
        let vox: usize = out.iter().product();
        ((vox * self.cout * self.cin * taps) as u64, out)
    }
}

impl Program {
    /// Every layer's channels chain from `in_channels`, each stage's stride divides its input evenly, the
    /// decoder mirrors the encoder (a transposed conv of kernel = stride undoing each stage's stride, with
    /// bias) and the head is a 1x1 conv with bias. Returns each stage's output shape and channels, for a
    /// patch of `patch` voxels (z, y, x)
    pub fn levels(&self, patch: [usize; 3], in_channels: usize) -> Result<Vec<([usize; 3], usize)>> {
        ensure_supported!(!self.stages.is_empty() && self.stages.iter().all(|s| !s.is_empty()), "empty stage");
        ensure_supported!(patch.iter().all(|&v| v > 0 && v < 1 << 10), "patch {patch:?}");
        let mut cin = in_channels;
        for (i, s) in self.stem.iter().enumerate() {
            s.check(&format!("stem {i}"), cin, true)?;
            ensure_supported!(
                s.stride.iter().all(|&v| v == 1) && s.norm.is_some() && s.leaky.is_some(),
                "stem {i}: stride or activation"
            );
            cin = s.cout;
        }
        let mut shape = patch;
        let mut levels = Vec::new();
        let mut strides = Vec::new();
        for (si, stage) in self.stages.iter().enumerate() {
            let stride = stage[0].conv1.stride;
            for (bi, b) in stage.iter().enumerate() {
                let what = format!("stage {si} block {bi}");
                b.conv1.check(&format!("{what} conv1"), cin, true)?;
                b.conv2.check(&format!("{what} conv2"), b.conv1.cout, true)?;
                ensure_supported!(b.conv2.stride.iter().all(|&v| v == 1), "{what}: conv2 stride {:?}", b.conv2.stride);
                ensure_supported!(b.conv1.norm.is_some() && b.conv1.leaky.is_some(), "{what}: conv1 norm/act");
                ensure_supported!(b.conv2.norm.is_some() && b.conv2.leaky.is_none(), "{what}: conv2 norm/act");
                let s = b.conv1.stride;
                ensure_supported!(b.skip.pool.axes() == s, "{what}: skip pool {:?} with stride {s:?}", b.skip.pool);
                ensure_supported!(bi == 0 || s == [1; 3], "{what}: stride {s:?} past a stage's first block");
                for a in 0..3 {
                    ensure_supported!(
                        shape[a].is_multiple_of(s[a]),
                        "{what}: shape {shape:?} does not divide by {s:?}"
                    );
                    shape[a] /= s[a];
                }
                match &b.skip.proj {
                    Some(pj) => {
                        pj.check(&format!("{what} skip"), cin, true)?;
                        ensure_supported!(
                            pj.kernel.iter().all(|&v| v == 1)
                                && pj.stride.iter().all(|&v| v == 1)
                                && pj.b.is_none()
                                && pj.norm.is_some()
                                && pj.leaky.is_none(),
                            "{what}: skip projection must be a bias-free 1x1 conv + norm"
                        );
                        ensure_supported!(pj.cout == b.conv2.cout, "{what}: skip gives {} channels", pj.cout);
                    }
                    None => {
                        ensure_supported!(cin == b.conv2.cout, "{what}: identity skip across {cin} -> {}", b.conv2.cout)
                    }
                }
                cin = b.conv2.cout;
            }
            levels.push((shape, cin));
            strides.push(stride);
        }
        ensure_supported!(self.decoder.len() + 1 == self.stages.len(), "{} decoder stages", self.decoder.len());
        for (d, dec) in self.decoder.iter().enumerate() {
            let level = self.stages.len() - 2 - d;
            let (_, skip_c) = levels[level];
            let t = &dec.transp;
            t.check(&format!("decoder {d} transp"), cin, false)?;
            let undo = strides[level + 1];
            ensure_supported!(
                t.kernel == undo && t.stride == undo && t.b.is_some(),
                "decoder {d}: transp k{:?} s{:?} must undo stride {undo:?}, with bias",
                t.kernel,
                t.stride
            );
            ensure_supported!(
                t.cout == skip_c && t.cout % 2 == 0,
                "decoder {d}: transp to {} channels, skip has {skip_c}",
                t.cout
            );
            ensure_supported!(!dec.convs.is_empty(), "decoder {d}: no convs");
            let mut c = 2 * skip_c;
            for (i, cv) in dec.convs.iter().enumerate() {
                cv.check(&format!("decoder {d} conv {i}"), c, true)?;
                ensure_supported!(
                    cv.stride.iter().all(|&v| v == 1) && cv.norm.is_some() && cv.leaky.is_some(),
                    "decoder {d} conv {i}"
                );
                c = cv.cout;
            }
            cin = c;
        }
        let h = &self.head;
        h.check("head", cin, true)?;
        ensure_supported!(
            h.kernel.iter().all(|&v| v == 1) && h.cout >= 1 && h.b.is_some() && h.norm.is_none() && h.leaky.is_none(),
            "head must be a 1x1 conv with bias"
        );
        Ok(levels)
    }

    /// v0.2's contract on top of [`Program::levels`]: one input channel, a cubic patch of edge `p`, cubic
    /// kernels and isotropic strides, k2 s2 transposed convs and a 1-channel head
    fn check(&self, p: usize) -> Result<()> {
        self.levels([p; 3], 1)?;
        self.check_isotropic()?;
        ensure_supported!(self.head.cout == 1, "head must be a 1x1 conv to 1 channel with bias");
        if p >= 1 << 10 {
            bail!("patch edge {p} too large");
        }
        Ok(())
    }

    /// The layers [`model::Model`] runs: cubic kernels, isotropic strides and pools, k2 s2 transposed convs
    pub fn check_isotropic(&self) -> Result<()> {
        let convs = self
            .stem
            .iter()
            .chain(self.stages.iter().flatten().flat_map(|b| [&b.conv1, &b.conv2].into_iter().chain(&b.skip.proj)))
            .chain(self.decoder.iter().flat_map(|d| d.convs.iter()))
            .chain([&self.head]);
        for c in convs {
            ensure_supported!(c.isotropic(), "{}: kernel {:?} stride {:?}, only isotropic", c.w, c.kernel, c.stride);
        }
        for d in &self.decoder {
            let t = &d.transp;
            ensure_supported!(
                t.kernel.iter().all(|&v| v == 2) && t.stride.iter().all(|&v| v == 2),
                "{}: transp k{:?} s{:?}, only k2 s2",
                t.w,
                t.kernel,
                t.stride
            );
        }
        for b in self.stages.iter().flatten() {
            let pool = b.skip.pool.axes();
            ensure_supported!(pool.iter().all(|&v| v == pool[0]), "skip pool {pool:?}, only isotropic");
        }
        Ok(())
    }

    /// Multiply-accumulates of one forward on a `patch` input: every conv (output voxels x Cout x Cin x
    /// taps), every transposed conv (input voxels x Cin x Cout x taps) and the head
    pub fn macs(&self, patch: [usize; 3]) -> u64 {
        let mut shape = patch;
        let mut total = 0u64;
        let conv = |total: &mut u64, c: &ConvSpec, s: [usize; 3]| -> [usize; 3] {
            let (m, out) = c.macs(s);
            *total += m;
            out
        };
        for s in &self.stem {
            shape = conv(&mut total, s, shape);
        }
        let mut shapes = Vec::new();
        for stage in &self.stages {
            for b in stage {
                if let Some(pj) = &b.skip.proj {
                    let pool = b.skip.pool.axes();
                    conv(&mut total, pj, std::array::from_fn(|a| shape[a] / pool[a]));
                }
                shape = conv(&mut total, &b.conv1, shape);
                shape = conv(&mut total, &b.conv2, shape);
            }
            shapes.push(shape);
        }
        for (d, dec) in self.decoder.iter().enumerate() {
            let t = &dec.transp;
            let taps: usize = t.kernel.iter().product();
            total += (shape.iter().product::<usize>() * t.cin * t.cout * taps) as u64;
            shape = shapes[shapes.len() - 2 - d];
            for c in &dec.convs {
                shape = conv(&mut total, c, shape);
            }
        }
        conv(&mut total, &self.head, shape);
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ink3d is refused as unsupported below compute capability 8.0 and on the sm75 tier, forced or
    /// not, so callers fall back instead of failing at a missing kernel or a cuDNN bf16 plan
    #[test]
    fn ink3d_needs_sm80() {
        let cc = ComputeCapability::new;
        for (c, tier) in [(cc(7, 5), PtxTier::Sm75), (cc(12, 0), PtxTier::Sm75)] {
            let err = check_device(c, tier).unwrap_err();
            assert!(crate::error::is_unsupported(&err), "{err}");
        }
        assert!(check_device(cc(8, 6), PtxTier::Sm80).is_ok());
        assert!(check_device(cc(12, 0), PtxTier::Sm80).is_ok());
        assert!(check_device(cc(12, 0), PtxTier::Sm120).is_ok());
    }
}
