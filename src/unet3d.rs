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
    pub kernel: Vec<usize>,
    pub stride: Vec<usize>,
    pub norm: Option<NormSpec>,
    /// LeakyReLU's negative slope, or `None` for no activation
    pub leaky: Option<f32>,
}

/// A BasicBlockD skip: optional 2x2x2 average pool, then optional bias-free 1x1 conv + norm
#[derive(Clone, Debug, Deserialize)]
pub struct SkipSpec {
    pub pool: usize,
    pub proj: Option<ConvSpec>,
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
        "ink3d needs a GPU of compute capability 8.0 or above and the sm80 or sm120 PTX tier, for its bf16 convs; \
         this GPU is {cc} on the {tier} tier"
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
    fn check(&self, what: &str, cin: usize, odd: bool) -> Result<()> {
        ensure_supported!(self.cin == cin, "{what}: {} input channels, the previous layer gives {cin}", self.cin);
        ensure_supported!(self.kernel.len() == 3 && self.stride.len() == 3, "{what}: not 3D");
        let k = self.kernel[0];
        ensure_supported!(self.kernel.iter().all(|&v| v == k), "{what}: kernel {:?}, only cubic", self.kernel);
        let s = self.stride[0];
        ensure_supported!(self.stride.iter().all(|&v| v == s), "{what}: stride {:?}, only isotropic", self.stride);
        if odd {
            ensure_supported!(k % 2 == 1 && (s == 1 || s == 2), "{what}: kernel {k} stride {s}");
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
}

impl Program {
    /// Every layer's channels chain, strides halve the edge evenly, and the decoder mirrors the
    /// encoder, for a patch of edge `p`
    fn check(&self, p: usize) -> Result<()> {
        ensure_supported!(!self.stages.is_empty() && self.stages.iter().all(|s| !s.is_empty()), "empty stage");
        let mut cin = 1;
        for (i, s) in self.stem.iter().enumerate() {
            s.check(&format!("stem {i}"), cin, true)?;
            ensure_supported!(
                s.stride[0] == 1 && s.norm.is_some() && s.leaky.is_some(),
                "stem {i}: stride or activation"
            );
            cin = s.cout;
        }
        let mut edge = p;
        let mut chans = Vec::new();
        for (si, stage) in self.stages.iter().enumerate() {
            for (bi, b) in stage.iter().enumerate() {
                let what = format!("stage {si} block {bi}");
                b.conv1.check(&format!("{what} conv1"), cin, true)?;
                b.conv2.check(&format!("{what} conv2"), b.conv1.cout, true)?;
                ensure_supported!(b.conv2.stride[0] == 1, "{what}: conv2 stride {}", b.conv2.stride[0]);
                ensure_supported!(b.conv1.norm.is_some() && b.conv1.leaky.is_some(), "{what}: conv1 norm/act");
                ensure_supported!(b.conv2.norm.is_some() && b.conv2.leaky.is_none(), "{what}: conv2 norm/act");
                let s = b.conv1.stride[0];
                ensure_supported!(b.skip.pool == s, "{what}: skip pool {} with stride {s}", b.skip.pool);
                ensure_supported!(bi == 0 || s == 1, "{what}: stride {s} past a stage's first block");
                if s == 2 {
                    ensure_supported!(edge.is_multiple_of(2), "{what}: edge {edge} does not halve");
                    edge /= 2;
                }
                match &b.skip.proj {
                    Some(pj) => {
                        pj.check(&format!("{what} skip"), cin, true)?;
                        ensure_supported!(
                            pj.kernel[0] == 1
                                && pj.stride[0] == 1
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
            chans.push((cin, edge));
        }
        ensure_supported!(self.decoder.len() + 1 == self.stages.len(), "{} decoder stages", self.decoder.len());
        for (d, dec) in self.decoder.iter().enumerate() {
            let level = self.stages.len() - 2 - d;
            let (skip_c, _) = chans[level];
            let t = &dec.transp;
            t.check(&format!("decoder {d} transp"), cin, false)?;
            ensure_supported!(
                t.kernel[0] == 2 && t.stride[0] == 2 && t.b.is_some(),
                "decoder {d}: transp must be k2 s2 with bias"
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
                ensure_supported!(cv.stride[0] == 1 && cv.norm.is_some() && cv.leaky.is_some(), "decoder {d} conv {i}");
                c = cv.cout;
            }
            cin = c;
        }
        let h = &self.head;
        h.check("head", cin, true)?;
        ensure_supported!(
            h.kernel[0] == 1 && h.cout == 1 && h.b.is_some() && h.norm.is_none(),
            "head must be a 1x1 conv to 1 channel with bias"
        );
        if p >= 1 << 10 {
            bail!("patch edge {p} too large");
        }
        Ok(())
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
