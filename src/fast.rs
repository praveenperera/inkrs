//! Fast mode: per-model settings that give up villa's and `hecate.py`'s map parity for speed
//!
//! A preset is admitted only when it keeps label AUC within 0.005 of parity mode on every labelled
//! segment of the accuracy grid, with paired bootstrap CIs (`docs/FAST.md`, from a grid registered before
//! its first run). A model with no passing setting has no preset, and `--fast` refuses to run it
//! rather than quietly running parity mode or an unmeasured setting

use color_eyre::eyre::{Result, bail, ensure};
use serde::Serialize;

use crate::hecate::{self, Precision};

/// Reader numerics, chosen by the caller and never read from the checkpoint
///
/// - `Fp16`, the default: villa's FP16 autocast on CUDA, with every autocast rounding
/// - `Fp32`: every op in float32, as villa runs on MPS (it autocasts on CUDA only), so a Mac run can
///   match villa's own MPS maps. Metal only: the CUDA engine's kernels are FP16 ([`FP32_METAL_ONLY`])
///
/// bf16 or fp8 would need new kernels
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ReaderPrecision {
    #[default]
    Fp16,
    Fp32,
}

/// Why the CUDA engine refuses fp32 reader numerics
pub const FP32_METAL_ONLY: &str = "reader precision fp32 runs on the Metal backend only: the CUDA engine's reader \
     kernels are FP16, as villa autocasts to FP16 on CUDA";

impl ReaderPrecision {
    /// The name the CLI, the Python API and the run report use
    pub fn name(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
        }
    }

    /// The precision named `name` (`fp16` or `fp32`)
    pub fn parse(name: &str) -> Option<Self> {
        [Self::Fp16, Self::Fp32].into_iter().find(|p| p.name() == name)
    }

    /// Bytes of one stored activation, conv weight or logit
    pub fn bytes(self) -> usize {
        match self {
            Self::Fp16 => 2,
            Self::Fp32 => 4,
        }
    }

    /// `w` as this precision stores the weights autocast would cast: rounded through FP16 at fp16
    /// ([`crate::weights::round16`]), unchanged at fp32
    pub fn round(self, w: &[f32]) -> Vec<f32> {
        match self {
            Self::Fp16 => crate::weights::round16(w),
            Self::Fp32 => w.to_vec(),
        }
    }
}

/// A reader's fast settings
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ReaderPreset {
    pub stride: usize,
    pub precision: ReaderPrecision,
}

/// Hecate's fast settings
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct HecatePreset {
    pub stride: usize,
    #[serde(serialize_with = "precision_name")]
    pub precision: Precision,
    pub batch: usize,
}

/// `hecate.py`'s name for the precision ("fp32" or "bf16"), as the run report writes it
fn precision_name<S: serde::Serializer>(p: &Precision, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(p.name())
}

/// Readers with a fast preset, by the reader name their export's sidecar records. None passed at any
/// stride the grids tried above parity's 64: reader-v2 lost up to 0.007 AUC at 72, 0.025 at 80, 0.0145
/// at 96 and 0.047 at 128, and seed42 failed every one, on segments clean for it too
const READERS: &[(&str, ReaderPreset)] = &[];

/// Hecate 9.6 um: bf16 at hecate.py's stride 32 kept every segment within the rule (worst paired
/// ΔAUC -0.0006, lowest CI end -0.0010) at 2.21x the fp32 grid speed; stride 40 lost 0.017, 48 lost 0.023
/// and 64 lost 0.041, so the stride stays. Batch 32 is inkrs's fastest bf16 batch (MODELS.md, Hecate timing) and
/// changes no tile's arithmetic
pub const HECATE: HecatePreset = HecatePreset { stride: hecate::P / 2, precision: Precision::Bf16, batch: 32 };

/// The fast preset of the reader named `reader` (the export sidecar's name) on `patch` px tiles, or
/// why it has none
pub fn reader(reader: Option<&str>, patch: usize) -> Result<ReaderPreset> {
    let name = reader.unwrap_or("this checkpoint");
    match READERS.iter().find(|(n, _)| Some(*n) == reader) {
        Some((_, p)) => Ok(*p),
        None => bail!(
            "--fast: {name} has no fast preset. No reader stride tested above parity mode's {} kept label AUC \
             within 0.005 on the labelled segments (reader-v2 lost up to 0.007 at stride 72, 0.025 at 80, 0.0145 \
             at 96 and 0.047 at 128), so run without --fast. See \
             https://github.com/praveenperera/inkrs/blob/master/docs/FAST.md",
            patch / 2
        ),
    }
}

/// A reader's tile stride on `patch` px tiles (128, or 384 for Reader V3): its fast preset's with
/// `fast`, else `stride`, else villa's from `overlap` (default 0.5); refuses a stride outside
/// 1..=patch, a preset with explicit tiling, or a reader with no preset, before any GPU work. The CLI
/// and the Python module both resolve through this
pub fn reader_stride(
    reader: Option<&str>,
    patch: usize,
    fast: bool,
    stride: Option<usize>,
    overlap: Option<f64>,
) -> Result<usize> {
    ensure!(stride.is_none() || overlap.is_none(), "pass a stride or an overlap, not both");
    let s = if fast {
        ensure!(stride.is_none() && overlap.is_none(), "--fast sets the stride; drop --stride and --overlap");
        self::reader(reader, patch)?.stride
    } else {
        match (stride, overlap) {
            (Some(s), _) => s,
            (None, o) => stride_from_overlap(patch, o.unwrap_or(0.5))?,
        }
    };
    ensure!((1..=patch).contains(&s), "stride {s} is outside 1..={patch}");
    Ok(s)
}

/// Hecate's stride, precision and batch: [`HECATE`] with `fast` (a given `batch` still wins, since
/// it changes no tile's arithmetic), else `hecate.py`'s defaults (stride 32, fp32 with TF32, and
/// inkrs's batch 8) under whatever is given; refuses a preset with explicit tiling or precision
pub fn hecate_settings(
    fast: bool,
    stride: Option<usize>,
    overlap: Option<f64>,
    precision: Option<Precision>,
    batch: Option<usize>,
) -> Result<HecatePreset> {
    ensure!(stride.is_none() || overlap.is_none(), "pass a stride or an overlap, not both");
    let p = if fast {
        ensure!(
            stride.is_none() && overlap.is_none() && precision.is_none(),
            "--fast sets the stride and precision; drop --stride, --overlap, --precision and --no-tf32"
        );
        HecatePreset { batch: batch.unwrap_or(HECATE.batch), ..HECATE }
    } else {
        let stride = match (stride, overlap) {
            (Some(s), _) => s,
            (None, Some(o)) => stride_from_overlap(hecate::P, o)?,
            (None, None) => hecate::P / 2,
        };
        HecatePreset {
            stride,
            precision: precision.unwrap_or(Precision::Fp32 { tf32: true }),
            batch: batch.unwrap_or(8),
        }
    };
    ensure!((1..=hecate::P).contains(&p.stride), "stride {} is outside 1..={}", p.stride, hecate::P);
    ensure!(p.batch > 0, "batch size must be positive");
    Ok(p)
}

/// Every preset, for `inkrs fast-presets` and the Python module
#[derive(Debug, Serialize)]
pub struct Presets {
    pub readers: Vec<(&'static str, ReaderPreset)>,
    pub hecate: HecatePreset,
}

pub fn presets() -> Presets {
    Presets { readers: READERS.to_vec(), hecate: HECATE }
}

/// `--overlap` for a tile of `patch` px: `round(patch * (1 - overlap))`, half to even, at least 1,
/// as villa's `resolve_patch_stride` does for its 128 px tile
pub fn stride_from_overlap(patch: usize, overlap: f64) -> Result<usize> {
    if !(0.0..1.0).contains(&overlap) {
        bail!("--overlap must be in [0, 1), got {overlap}");
    }
    Ok(((patch as f64 * (1.0 - overlap)).round_ties_even() as usize).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readers_refuse_and_hecate_is_bf16_at_parity_stride() {
        for name in [Some("reader-v2"), Some("hybrid_3d2d-seed42"), None] {
            let err = format!("{:#}", reader(name, 128).unwrap_err());
            // the refusal claims only what the grid measured: tested strides, not every possible one
            assert!(err.contains("no fast preset") && err.contains("tested"), "{err}");
            assert!(!err.contains("fastest accurate"), "{err}");
        }
        assert_eq!(HECATE, HecatePreset { stride: 32, precision: Precision::Bf16, batch: 32 });
    }

    #[test]
    fn settings_resolve_and_refuse() {
        assert_eq!(reader_stride(Some("reader-v2"), 128, false, None, None).unwrap(), 64);
        assert_eq!(reader_stride(None, 128, false, None, Some(0.25)).unwrap(), 96);
        assert_eq!(reader_stride(None, 128, false, Some(128), None).unwrap(), 128);
        assert!(reader_stride(None, 128, false, Some(129), None).is_err());
        assert!(reader_stride(None, 128, false, Some(0), None).is_err());
        assert!(reader_stride(None, 128, false, Some(64), Some(0.5)).is_err());
        assert!(reader_stride(Some("reader-v2"), 128, true, None, None).is_err());
        // Reader V3's 384 px tiles: villa's overlap 0.5 is stride 192, and strides run up to the tile edge
        assert_eq!(reader_stride(Some("reader-v3"), 384, false, None, None).unwrap(), 192);
        assert_eq!(reader_stride(None, 384, false, None, Some(0.25)).unwrap(), 288);
        assert_eq!(reader_stride(None, 384, false, Some(384), None).unwrap(), 384);
        assert!(reader_stride(None, 384, false, Some(385), None).is_err());
        assert!(reader_stride(Some("reader-v3"), 384, true, None, None).is_err());

        let parity = hecate_settings(false, None, None, None, None).unwrap();
        assert_eq!(parity, HecatePreset { stride: 32, precision: Precision::Fp32 { tf32: true }, batch: 8 });
        assert_eq!(hecate_settings(true, None, None, None, None).unwrap(), HECATE);
        assert_eq!(hecate_settings(true, None, None, None, Some(8)).unwrap().batch, 8);
        assert_eq!(hecate_settings(false, None, Some(0.0), Some(Precision::Bf16), None).unwrap().stride, 64);
        assert!(hecate_settings(true, Some(32), None, None, None).is_err());
        assert!(hecate_settings(true, None, None, Some(Precision::Bf16), None).is_err());
        assert!(hecate_settings(false, Some(65), None, None, None).is_err());
        assert!(hecate_settings(false, None, None, None, Some(0)).is_err());
    }

    #[test]
    fn overlap_rounds_as_villa() {
        assert_eq!(stride_from_overlap(128, 0.5).unwrap(), 64);
        assert_eq!(stride_from_overlap(128, 0.25).unwrap(), 96);
        assert_eq!(stride_from_overlap(128, 0.0).unwrap(), 128);
        assert_eq!(stride_from_overlap(64, 0.5).unwrap(), 32);
        assert_eq!(stride_from_overlap(64, 0.25).unwrap(), 48);
        // 128 * (1 - 0.24609375) = 96.5 exactly, which rounds half to even; the stride is at least 1
        assert_eq!(stride_from_overlap(128, 0.24609375).unwrap(), 96);
        assert_eq!(stride_from_overlap(2, 0.75).unwrap(), 1);
        assert!(stride_from_overlap(64, 1.0).is_err() && stride_from_overlap(64, -0.1).is_err());
    }
}
