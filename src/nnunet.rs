//! nnU-Net ResEnc-L models (`inkrs nnunet`): the scroll surface model m7, and the contract for HercUNet and
//! Socratic F0
//!
//! `scripts/export_nnunet.py` exports a checkpoint's weights, its layer program (the same program
//! [`crate::unet3d`] builds its forward from, which records kernels, strides and pools per axis but runs only
//! isotropic ones) and the recipe of the pipeline the weights run in: precision, input normalisation, test-time augmentation, blend and
//! finalisation. [`Contract`] reads and checks that sidecar on the CPU; [`grid`] is villa's sliding-window
//! patch plan. [`engine`] runs the `vesuvius-predict` recipe (villa `vesuvius.predict`, m7's pipeline); the
//! others are refused as unsupported

use std::collections::{BTreeMap, BTreeSet};

use color_eyre::eyre::Result;
use serde::Deserialize;

use crate::ensure_supported;
use crate::error::Unsupported;
pub use crate::unet3d::Precision;
use crate::unet3d::{ConvSpec, Program};

#[cfg(not(target_os = "macos"))]
pub mod engine;
pub mod grid;
pub mod host;
pub mod out;

#[cfg(target_os = "macos")]
pub use crate::metal::nnunet as engine;

/// The sidecar format this engine reads
pub const FORMAT: &str = "inkrs-nnunet-v1";

/// nnU-Net's CTNormalization as villa computes it: `(clip(x, low, high) - center) / scale` in float64,
/// stored as float32, one value per uint8 code
#[derive(Clone, Debug, Deserialize)]
pub struct CtNorm {
    pub center: f64,
    pub scale: f64,
    pub low: f64,
    pub high: f64,
    pub table_f32_bits: Vec<u32>,
}

impl CtNorm {
    /// The 256-entry table as float32
    pub fn table(&self) -> Vec<f32> {
        self.table_f32_bits.iter().map(|&b| f32::from_bits(b)).collect()
    }

    /// Fails unless the table is the float32 rounding of the float64 formula for every code, so the
    /// recorded parameters and the table cannot disagree
    fn check(&self) -> Result<()> {
        ensure_supported!(self.table_f32_bits.len() == 256, "CT table of {} entries", self.table_f32_bits.len());
        ensure_supported!(
            self.scale > 0.0 && self.low <= self.high,
            "CT scale {} range {}..{}",
            self.scale,
            self.low,
            self.high
        );
        for (code, &bits) in self.table_f32_bits.iter().enumerate() {
            let want = ((code as f64).clamp(self.low, self.high) - self.center) / self.scale;
            ensure_supported!(
                (want as f32).to_bits() == bits,
                "CT table entry {code}: {} for {want}",
                f32::from_bits(bits)
            );
        }
        Ok(())
    }
}

/// How the per-voxel input is normalised
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Normalization {
    Ct(CtNorm),
}

/// Test-time augmentation
#[derive(Clone, Debug, Deserialize)]
pub struct Tta {
    /// `mirror8` (every flip of z, y, x, as one batch) or `none`
    pub mode: String,
    /// `mean_logits`: the flips' fp16 logits averaged
    pub combine: String,
}

/// The patch blend
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Blend {
    /// villa's `generate_gaussian_map`, stored as the float32 tensor `tensor`
    GaussianMap { tensor: String, sigma_scale: f64 },
}

/// The output finalisation
#[derive(Clone, Debug, Deserialize)]
pub struct Finalize {
    /// `softmax_u8`: villa's `finalize_outputs` (binary: softmax[1] x 255 truncated, or a logit cutoff)
    pub mode: String,
}

/// The pipeline the weights run in
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "pipeline", rename_all = "kebab-case")]
pub enum Recipe {
    /// villa `vesuvius.predict` with `--blend_in_memory` (m7)
    VesuviusPredict {
        precision: Precision,
        normalization: Normalization,
        tta: Tta,
        blend: Blend,
        finalize: Finalize,
        overlap: f64,
        skip_constant_patches: bool,
    },
    /// hercUNet's `jacobi_refine` (HercUNet v0)
    Hercunet {
        precision: Precision,
        normalization: Normalization,
        inputs: Vec<String>,
        tta: Tta,
        blend: Blend,
        finalize: Finalize,
        overlap: f64,
        passes: usize,
    },
    /// The network only, for a pipeline inkrs does not implement (Socratic F0)
    ProgramOnly,
}

impl Recipe {
    /// The pipeline's name, as the sidecar spells it
    pub fn name(&self) -> &'static str {
        match self {
            Self::VesuviusPredict { .. } => "vesuvius-predict",
            Self::Hercunet { .. } => "hercunet",
            Self::ProgramOnly => "program-only",
        }
    }
}

/// The export's contract
#[derive(Clone, Debug, Deserialize)]
pub struct Contract {
    pub format: String,
    pub name: String,
    pub patch_size: [usize; 3],
    pub in_channels: usize,
    pub num_classes: usize,
    pub recipe: Recipe,
    pub program: Program,
    /// Every exported tensor and its shape
    pub keys: BTreeMap<String, Vec<usize>>,
    pub aliases: BTreeMap<String, String>,
    /// Checkpoint tensors the inference graph never reads, with the reason
    pub dropped: BTreeMap<String, String>,
    /// The exporter's count, `2 x` [`Program::macs`]
    pub flops_per_patch: u64,
}

/// The `vesuvius-predict` recipe the engine runs, checked
#[derive(Clone, Debug)]
pub struct Predict {
    pub precision: Precision,
    pub ct: CtNorm,
    /// Flips run per patch: 8 for `mirror8`, 1 for none
    pub flips: usize,
    pub gaussian_key: String,
    pub overlap: f64,
    pub skip_constant_patches: bool,
}

impl Contract {
    /// Parses and checks the sidecar: the format, the program's geometry and channels for this patch and
    /// input, the head's classes, the tensors the program reads against the exported keys (each exactly
    /// once, nothing else but the recipe's tensors), the recipe's own tables, and the FLOP count
    pub fn from_sidecar(sidecar: &serde_json::Value) -> Result<Self> {
        let c: Contract =
            serde_json::from_value(sidecar.clone()).map_err(|e| Unsupported(format!("not an {FORMAT} export: {e}")))?;
        ensure_supported!(c.format == FORMAT, "sidecar format {:?}, expected {FORMAT}", c.format);
        c.program.levels(c.patch_size, c.in_channels)?;
        ensure_supported!(
            c.program.head.cout == c.num_classes && c.num_classes >= 2,
            "head of {} channels for {} classes",
            c.program.head.cout,
            c.num_classes
        );
        let macs = c.program.macs(c.patch_size);
        ensure_supported!(
            2 * macs == c.flops_per_patch,
            "the exporter counts {} FLOPs a patch, the program {}",
            c.flops_per_patch,
            2 * macs
        );
        c.check_keys()?;
        match &c.recipe {
            Recipe::VesuviusPredict { normalization: Normalization::Ct(ct), .. } => ct.check()?,
            Recipe::Hercunet { normalization: Normalization::Ct(ct), .. } => ct.check()?,
            _ => {}
        }
        Ok(c)
    }

    /// The tensor names the program reads, in program order, each once
    pub fn program_tensors(&self) -> Vec<&str> {
        fn conv<'a>(out: &mut Vec<&'a str>, c: &'a ConvSpec) {
            out.push(c.w.as_str());
            out.extend(c.b.as_deref());
            if let Some(n) = &c.norm {
                out.extend([n.w.as_str(), n.b.as_str()]);
            }
        }
        let p = &self.program;
        let mut out = Vec::new();
        for s in &p.stem {
            conv(&mut out, s);
        }
        for b in p.stages.iter().flatten() {
            conv(&mut out, &b.conv1);
            conv(&mut out, &b.conv2);
            if let Some(pj) = &b.skip.proj {
                conv(&mut out, pj);
            }
        }
        for d in &p.decoder {
            conv(&mut out, &d.transp);
            for c in &d.convs {
                conv(&mut out, c);
            }
        }
        conv(&mut out, &p.head);
        out
    }

    /// The recipe's own tensors
    fn recipe_tensors(&self) -> Vec<&str> {
        match &self.recipe {
            Recipe::VesuviusPredict { blend: Blend::GaussianMap { tensor, .. }, .. }
            | Recipe::Hercunet { blend: Blend::GaussianMap { tensor, .. }, .. } => vec![tensor.as_str()],
            _ => Vec::new(),
        }
    }

    /// Fails unless the program reads every exported tensor but the recipe's exactly once, and reads nothing
    /// else; aliases must point at exported tensors
    fn check_keys(&self) -> Result<()> {
        let mut seen = BTreeSet::new();
        for t in self.program_tensors() {
            ensure_supported!(seen.insert(t), "the program reads {t} twice");
            ensure_supported!(self.keys.contains_key(t), "the program reads {t}, which the export does not hold");
        }
        for t in self.recipe_tensors() {
            ensure_supported!(seen.insert(t), "recipe tensor {t} is also a network tensor");
            ensure_supported!(self.keys.contains_key(t), "recipe tensor {t} missing from the export");
        }
        let unread: Vec<&String> = self.keys.keys().filter(|k| !seen.contains(k.as_str())).collect();
        ensure_supported!(unread.is_empty(), "exported tensors nothing reads: {unread:?}");
        for (alias, target) in &self.aliases {
            ensure_supported!(self.keys.contains_key(target), "alias {alias} of {target}, which is not exported");
        }
        Ok(())
    }

    /// The `vesuvius-predict` recipe with what this engine implements: CT normalisation, 8-flip mirroring or
    /// none, mean logits, villa's Gaussian map of the patch's shape, softmax finalisation, one input channel,
    /// an isotropic cubic program; anything else is [`Unsupported`]
    pub fn predict(&self) -> Result<Predict> {
        let Recipe::VesuviusPredict { precision, normalization, tta, blend, finalize, overlap, skip_constant_patches } =
            &self.recipe
        else {
            return Err(
                Unsupported(format!("recipe {}, inkrs nnunet runs vesuvius-predict", self.recipe.name())).into()
            );
        };
        let Normalization::Ct(ct) = normalization;
        let Blend::GaussianMap { tensor, .. } = blend;
        let flips = match tta.mode.as_str() {
            "mirror8" => 8,
            "none" => 1,
            m => return Err(Unsupported(format!("tta {m}")).into()),
        };
        ensure_supported!(tta.combine == "mean_logits", "tta combine {}", tta.combine);
        ensure_supported!(finalize.mode == "softmax_u8", "finalize {}", finalize.mode);
        ensure_supported!((0.0..1.0).contains(overlap), "overlap {overlap}");
        ensure_supported!(*precision == Precision::Fp16, "precision {precision:?}, the engine runs fp16");
        ensure_supported!(self.in_channels == 1, "{} input channels, the engine takes 1", self.in_channels);
        ensure_supported!(
            self.num_classes == 2,
            "{} classes; the engine finalises binary models (multiclass is phase 2)",
            self.num_classes
        );
        let p = self.patch_size[0];
        ensure_supported!(self.patch_size.iter().all(|&v| v == p), "patch {:?}, only cubic", self.patch_size);
        self.program.check_isotropic()?;
        ensure_supported!(
            self.keys.get(tensor).is_some_and(|s| s == &self.patch_size.to_vec()),
            "blend tensor {tensor} of shape {:?}, the patch is {:?}",
            self.keys.get(tensor),
            self.patch_size
        );
        Ok(Predict {
            precision: *precision,
            ct: ct.clone(),
            flips,
            gaussian_key: tensor.clone(),
            overlap: *overlap,
            skip_constant_patches: *skip_constant_patches,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> serde_json::Value {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/nnunet").join(format!("{name}.json"));
        serde_json::from_slice(&std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
            .expect("fixture JSON")
    }

    /// All three exports parse and pass the CPU checks: geometry, channels, every tensor read exactly once,
    /// FLOPs, CT tables. m7 is the 6-stage 192^3 network at 4.842 TFLOP a patch, torch's `FlopCounterMode`
    /// count to the FLOP (the design's 5.22 counted each transposed conv 8 times)
    #[test]
    fn three_exports_check() {
        let m7 = Contract::from_sidecar(&fixture("m7")).expect("m7");
        assert_eq!(m7.patch_size, [192; 3]);
        assert_eq!((m7.in_channels, m7.num_classes, m7.program.stages.len()), (1, 2, 6));
        let levels = m7.program.levels(m7.patch_size, 1).expect("levels");
        assert_eq!(levels.last(), Some(&([6, 6, 6], 320)));
        assert_eq!(m7.flops_per_patch, 4_842_471_555_072);
        let predict = m7.predict().expect("m7 runs");
        assert_eq!((predict.flips, predict.precision), (8, Precision::Fp16));
        assert_eq!(predict.ct.table()[0], ((0.0 - 87.54424285888672f64) / 47.74376678466797) as f32);

        let herc = Contract::from_sidecar(&fixture("herc")).expect("herc");
        assert_eq!(herc.in_channels, 8);
        assert!(herc.dropped.keys().any(|k| k.starts_with("base.aff_head.")));
        let soc = Contract::from_sidecar(&fixture("socratic")).expect("socratic");
        assert_eq!(soc.flops_per_patch, m7.flops_per_patch);
        for (c, recipe) in [(&herc, "hercunet"), (&soc, "program-only")] {
            let err = c.predict().unwrap_err();
            assert!(crate::error::is_unsupported(&err), "{err}");
            assert_eq!(c.recipe.name(), recipe);
        }
    }

    /// The sidecar records strides per axis, so a program whose last stage strides (2, 2, 1) parses and passes
    /// the contract checks, but the forward runs only isotropic programs: `check_isotropic`, which ink3d's
    /// contract and `Model::new` run, and `predict` refuse it as unsupported
    #[test]
    fn anisotropic_programs_are_unsupported() {
        let mut v = fixture("m7");
        let axes = serde_json::json!([2, 2, 1]);
        v["program"]["stages"][5][0]["conv1"]["stride"] = axes.clone();
        v["program"]["stages"][5][0]["skip"]["pool"] = axes.clone();
        v["program"]["decoder"][0]["transp"]["kernel"] = axes.clone();
        v["program"]["decoder"][0]["transp"]["stride"] = axes;
        let program: Program = serde_json::from_value(v["program"].clone()).expect("program");
        v["flops_per_patch"] = serde_json::json!(2 * program.macs([192; 3]));

        let c = Contract::from_sidecar(&v).expect("an anisotropic program parses");
        assert_eq!(c.program.levels(c.patch_size, 1).expect("levels").last().map(|l| l.0), Some([6, 6, 12]));
        assert!(crate::error::is_unsupported(&c.program.check_isotropic().unwrap_err()));
        assert!(crate::error::is_unsupported(&c.predict().unwrap_err()));
    }

    /// A region too big for the kernels' u32 indexing is unsupported, which villa's `--engine inkrs` falls back on
    #[test]
    fn u32_overflow_is_unsupported() {
        let e = crate::unet3d::host::u32_extent("accumulator", 1 << 33).unwrap_err();
        assert!(crate::error::is_unsupported(&e), "{e:#}");
        crate::unet3d::host::u32_extent("accumulator", u32::MAX as usize).unwrap();
    }

    /// A conv whose kernel or stride is not 3D fails to parse rather than panicking where the program reads its axes
    #[test]
    fn short_strides_are_refused() {
        for field in ["stride", "kernel"] {
            let mut v = fixture("m7");
            v["program"]["stages"][0][0]["conv1"][field] = serde_json::json!([1, 1]);
            let e = Contract::from_sidecar(&v).expect_err(field);
            assert!(format!("{e:#}").contains("expected an array of length 3"), "{field}: {e:#}");
        }
    }

    /// The real exports load as the engine loads them: every tensor in each `.safetensors` is taken exactly
    /// once by the fp16 host net or the recipe, at the shape the program says. Needs `m7`, `herc` and `socratic`
    /// exported by `scripts/export_nnunet.py` into the directory `INKRS_NNUNET_EXPORTS` names
    #[test]
    #[ignore = "needs the three exports on disk"]
    fn exports_load_every_tensor_once() {
        let dir = std::env::var("INKRS_NNUNET_EXPORTS").expect("INKRS_NNUNET_EXPORTS names the exports' directory");
        for name in ["m7", "herc", "socratic"] {
            let path = std::path::Path::new(&dir).join(format!("{name}.safetensors"));
            let (ck, sidecar) = crate::weights::Checkpoint::read(&path).unwrap_or_else(|e| panic!("{name}: {e:#}"));
            let contract = Contract::from_sidecar(&sidecar).unwrap_or_else(|e| panic!("{name}: {e:#}"));
            for t in contract.recipe_tensors() {
                ck.take(t, &contract.keys[t]).unwrap_or_else(|e| panic!("{name} {t}: {e:#}"));
            }
            crate::unet3d::host::HostNet::take(&ck, &contract.program, Precision::Fp16)
                .unwrap_or_else(|e| panic!("{name}: {e:#}"));
            let (tensors, _, params) = ck.finish().unwrap_or_else(|e| panic!("{name}: {e:#}"));
            assert_eq!(tensors, contract.keys.len(), "{name}");
            assert!(params > 1_000_000, "{name}: {params} parameters");
        }
    }

    /// A sidecar whose program reads a tensor the export lacks, or leaves an exported tensor unread, or whose
    /// CT table disagrees with its parameters, is refused before any GPU work
    #[test]
    fn tensor_and_table_mismatches_refused() {
        let base = fixture("m7");
        let mut extra = base.clone();
        extra["keys"]["stray.weight"] = serde_json::json!([3]);
        let err = Contract::from_sidecar(&extra).unwrap_err();
        assert!(err.to_string().contains("stray.weight"), "{err}");

        let mut missing = base.clone();
        let head = missing["program"]["head"]["w"].as_str().expect("head").to_string();
        missing["keys"].as_object_mut().expect("keys").remove(&head);
        let err = Contract::from_sidecar(&missing).unwrap_err();
        assert!(err.to_string().contains(&head), "{err}");

        let mut table = base;
        table["recipe"]["normalization"]["table_f32_bits"][100] = serde_json::json!(0);
        let err = Contract::from_sidecar(&table).unwrap_err();
        assert!(err.to_string().contains("entry 100"), "{err}");
    }
}
