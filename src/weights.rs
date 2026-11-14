//! The reader's weights from the safetensors export of `scripts/export_weights.py`, with a
//! key-for-key schema check
//!
//! The export holds villa's inference state dict after `load_state_dict(strict=True)`, one
//! tensor per storage; names that alias another tensor (`all_modules.*`) are listed in the
//! metadata. The model takes every tensor it needs by name and shape through
//! [`Checkpoint::take`]; [`Checkpoint::finish`] then fails unless every exported tensor was
//! taken exactly once and every alias points at one of them
//!
//! villa's `.pth` checkpoint (zip + pickle) is not read directly; export it first. Hecate's export
//! (`scripts/export_hecate.py`) has the same form with a contract of its own, checked in
//! [`crate::hecate::model`]

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use color_eyre::eyre::{Result, WrapErr, bail, ensure};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::ensure_supported;
use crate::error::Unsupported;
use crate::fast::ReaderPrecision;
use crate::sched::Edge;

/// The fields of the export's JSON sidecar that fix the inference contract
#[derive(Debug, Deserialize)]
pub struct Sidecar {
    pub crop_size: Vec<usize>,
    pub input_pad_depth_to: Option<usize>,
    pub preprocessing: String,
    /// `null` when the checkpoint ran without autocast. The FP16 engine needs `float16`; at fp32 every
    /// value runs as villa runs on MPS, which ignores it. The field itself is required, so a sidecar
    /// missing it is corrupt rather than unsupported
    #[serde(deserialize_with = "present")]
    pub amp_dtype: Option<String>,
    pub targets: BTreeMap<String, serde_json::Value>,
    /// villa's `model_type` and `model_config`: the tensor names and shapes do not pin every option of the
    /// network (a depth projection or block type can change with the same weights), so these are checked too
    pub model_type: String,
    #[serde(deserialize_with = "present_value")]
    pub model_config: Option<serde_json::Value>,
}

/// The `model_config` options of the `vesuvius_unet_3d_stem_2d` network this engine implements; other keys
/// may only spell out what autoconfiguration derives, which the strict tensor check then confirms
const MODEL_CONFIG: [(&str, &str); 4] = [
    ("basic_encoder_block", "BasicBlockD"),
    ("basic_decoder_block", "ConvBlock"),
    ("z_projection_mode", "none"),
    ("stem_channels", "16"),
];

/// The `model_config` keys allowed beside [`MODEL_CONFIG`]: autoconfiguration's own outputs for this network,
/// which a checkpoint may spell out (Reader V3 and Rectina M4 do) but only with these values. A changed stride or
/// pooling keeps every tensor shape, so [`Checkpoint::finish`] cannot catch it; each value is checked here
const MODEL_CONFIG_DERIVED: [(&str, &str); 6] = [
    ("autoconfigure", "true"),
    ("features_per_stage", "[32,64,128,256,320,320]"),
    ("n_blocks_per_stage", "[1,3,4,6,6,6]"),
    ("n_conv_per_stage_decoder", "[1,1,1,1,1]"),
    ("strides", "[[1,1],[2,2],[2,2],[2,2],[2,2],[2,2]]"),
    ("pool_op_kernel_sizes", "[[1,1],[2,2],[2,2],[2,2],[2,2],[2,2]]"),
];

/// Whether a checkpoint whose sidecar records autocast dtype `amp` runs at `precision`: the FP16 engine
/// copies villa's FP16 autocast and needs it recorded, while fp32 runs every checkpoint as villa does
/// on MPS
fn amp_supported(amp: Option<&str>, precision: ReaderPrecision) -> bool {
    match precision {
        ReaderPrecision::Fp16 => amp == Some("float16"),
        ReaderPrecision::Fp32 => matches!(amp, None | Some("float16" | "bfloat16")),
    }
}

/// A required field that may be `null`, as [`present`] for a JSON value
fn present_value<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Option<serde_json::Value>, D::Error> {
    Option::<serde_json::Value>::deserialize(d)
}

/// Whether villa's `model_config` builds the network this engine implements
fn supported_model_config(config: Option<&serde_json::Value>) -> bool {
    let Some(map) = config.and_then(|c| c.as_object()) else { return false };
    let text = |v: &serde_json::Value| v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string());
    MODEL_CONFIG.iter().all(|(k, want)| map.get(*k).map(text).as_deref() == Some(*want))
        && map.keys().all(|k| {
            MODEL_CONFIG.iter().any(|(m, _)| m == k)
                || MODEL_CONFIG_DERIVED.iter().any(|(m, want)| {
                    m == k && serde_json::from_str::<serde_json::Value>(want).is_ok_and(|w| w == map[k])
                })
        })
}

/// A required field that may be `null`: serde would otherwise read a missing `Option` as `None`
fn present<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Option<String>, D::Error> {
    Option::<String>::deserialize(d)
}

/// The exported tensors, all float32
pub struct Checkpoint {
    /// The tile edge the sidecar's `crop_size` fixes; 128 for an export read without the model
    /// contract ([`Checkpoint::read`])
    pub edge: Edge,
    tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    aliases: BTreeMap<String, String>,
    taken: RefCell<BTreeSet<String>>,
}

impl Checkpoint {
    /// Loads `path` (`.safetensors`) and its `.json` sidecar and checks the contract this
    /// engine implements at `precision`: a 17x128x128 or 17x384x384 crop, robust normalisation, one
    /// `ink` target with no activation in the model and, at fp16, FP16 autocast
    ///
    /// The precision is the caller's choice and is never read from the sidecar: villa runs a
    /// checkpoint that records FP16 autocast in float32 on MPS, since it autocasts on CUDA only, so
    /// an fp32 run takes any recorded autocast dtype (`null`, `float16`, `bfloat16`)
    ///
    /// The contract is checked from the sidecar before the tensors are read, so a checkpoint this
    /// engine does not implement is reported as unsupported (a fallback) whatever its tensor file holds
    pub fn load(path: &Path, precision: ReaderPrecision) -> Result<Self> {
        let sidecar_path = path.with_extension("json");
        let sidecar: Sidecar = serde_json::from_slice(
            &std::fs::read(&sidecar_path).wrap_err_with(|| format!("read {}", sidecar_path.display()))?,
        )
        .wrap_err_with(|| format!("parse {}", sidecar_path.display()))?;

        let edge = Edge::from_crop(&sidecar.crop_size);
        ensure_supported!(
            edge.is_some(),
            "crop {:?}, this engine needs [17, 128, 128] or [17, 384, 384]",
            sidecar.crop_size
        );
        ensure_supported!(
            sidecar.input_pad_depth_to.is_none(),
            "input_pad_depth_to {:?} is not supported",
            sidecar.input_pad_depth_to
        );
        ensure_supported!(
            sidecar.preprocessing == "tifxyz_robust",
            "preprocessing {:?} is not supported",
            sidecar.preprocessing
        );
        ensure_supported!(
            amp_supported(sidecar.amp_dtype.as_deref(), precision),
            "amp dtype {:?} is not supported at reader precision {}",
            sidecar.amp_dtype,
            precision.name()
        );
        ensure_supported!(
            sidecar.model_type == "vesuvius_unet_3d_stem_2d",
            "model type {:?}, this engine needs vesuvius_unet_3d_stem_2d",
            sidecar.model_type
        );
        ensure_supported!(
            supported_model_config(sidecar.model_config.as_ref()),
            "model_config {:?} is not the stem-16 BasicBlockD / ConvBlock network this engine implements",
            sidecar.model_config
        );
        let targets: Vec<&String> = sidecar.targets.keys().collect();
        ensure_supported!(targets == ["ink"], "targets {targets:?}, this engine needs exactly `ink`");
        let ink = &sidecar.targets["ink"];
        ensure_supported!(
            ink.get("activation").and_then(|v| v.as_str()) == Some("none")
                && ink.get("out_channels").and_then(|v| v.as_u64()) == Some(1),
            "ink target {ink} is not a one-channel logit"
        );
        let (mut ck, _) = Self::read(path)?;
        ck.edge = edge.unwrap_or_default();
        Ok(ck)
    }

    /// Loads `path` (`.safetensors`) and its `.json` sidecar with no model contract: every tensor
    /// float32 and finite, and the sidecar's `keys` and `aliases` equal to the file's. Returns the
    /// sidecar for the caller's contract checks
    pub fn read(path: &Path) -> Result<(Self, serde_json::Value)> {
        // mapped rather than read: the tensors are decoded straight from the page cache, without a
        // 138 MB copy first (about 65 ms of the load)
        let file = std::fs::File::open(path).wrap_err_with(|| format!("open {}", path.display()))?;
        // SAFETY: the map lives only for this call, and every byte is decoded into owned tensors
        // before it ends; a file truncated meanwhile is the same failure as reading it mid-write
        let bytes = unsafe { memmap2::Mmap::map(&file) }.wrap_err_with(|| format!("map {}", path.display()))?;
        let sidecar_path = path.with_extension("json");
        let sidecar: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&sidecar_path).wrap_err_with(|| format!("read {}", sidecar_path.display()))?,
        )
        .wrap_err_with(|| format!("parse {}", sidecar_path.display()))?;
        let side_keys: BTreeMap<String, Vec<usize>> = serde_json::from_value(sidecar["keys"].clone())
            .wrap_err_with(|| format!("{}: keys", sidecar_path.display()))?;
        let side_aliases: BTreeMap<String, String> = serde_json::from_value(sidecar["aliases"].clone())
            .wrap_err_with(|| format!("{}: aliases", sidecar_path.display()))?;

        let st = SafeTensors::deserialize(&bytes).wrap_err("parse the safetensors header")?;
        let mut tensors = BTreeMap::new();
        for (name, view) in st.tensors() {
            ensure_supported!(view.dtype() == Dtype::F32, "{name}: dtype {:?}, expected F32", view.dtype());
            let data: Vec<f32> = view.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
            ensure!(data.iter().all(|v| v.is_finite()), "{name}: non-finite weights");
            tensors.insert(name, (view.shape().to_vec(), data));
        }

        let (_, header) = SafeTensors::read_metadata(&bytes).wrap_err("read the safetensors metadata")?;
        let aliases: BTreeMap<String, String> = match header.metadata().as_ref().and_then(|m| m.get("aliases")) {
            Some(json) => serde_json::from_str(json).wrap_err("parse the alias metadata")?,
            None => bail!("{}: no alias metadata; re-export with scripts/export_weights.py", path.display()),
        };

        for (alias, target) in &aliases {
            ensure!(tensors.contains_key(target), "alias {alias} points at missing {target}");
            ensure!(!tensors.contains_key(alias), "alias {alias} is also a tensor");
        }
        let ck = Self { edge: Edge::P128, tensors, aliases, taken: RefCell::new(BTreeSet::new()) };
        // the sidecar and the tensor file must describe the same tensors
        ensure!(ck.keys() == side_keys, "the sidecar's keys and shapes differ from the tensor file's");
        ensure!(ck.aliases == side_aliases, "the sidecar's aliases differ from the tensor file's");
        Ok((ck, sidecar))
    }

    fn keys(&self) -> BTreeMap<String, Vec<usize>> {
        self.tensors.iter().map(|(k, (s, _))| (k.clone(), s.clone())).collect()
    }

    /// The tensor `key`, which must have `shape` and not have been taken before
    pub fn take(&self, key: &str, shape: &[usize]) -> Result<Vec<f32>> {
        // a missing or reshaped tensor means another architecture than the one this engine builds
        let Some((have, data)) = self.tensors.get(key) else {
            return Err(Unsupported(format!("checkpoint has no tensor {key}; not the hybrid_3d2d architecture")).into());
        };
        ensure_supported!(have == shape, "{key}: shape {have:?}, the model needs {shape:?}");
        ensure!(self.taken.borrow_mut().insert(key.to_owned()), "{key} taken twice");
        Ok(data.clone())
    }

    /// A one-element tensor as a scalar
    pub fn scalar(&self, key: &str) -> Result<f32> {
        Ok(self.take(key, &[1])?[0])
    }

    /// A zero-dimensional tensor (shape `[]`) as a scalar
    pub fn scalar0(&self, key: &str) -> Result<f32> {
        Ok(self.take(key, &[])?[0])
    }

    /// Fails unless every tensor was taken; returns (tensors, aliases, parameters)
    pub fn finish(&self) -> Result<(usize, usize, usize)> {
        let taken = self.taken.borrow();
        let unused: Vec<&String> = self.tensors.keys().filter(|k| !taken.contains(*k)).collect();
        ensure_supported!(unused.is_empty(), "{} checkpoint tensors unused by the model: {unused:?}", unused.len());
        let params = self.tensors.values().map(|(_, d)| d.len()).sum();
        Ok((self.tensors.len(), self.aliases.len(), params))
    }
}

/// `w` rounded through FP16, as autocast casts weights for FP16 convs
pub fn round16(w: &[f32]) -> Vec<f32> {
    w.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect()
}

/// A PyTorch 16 -> 16 channel 3x3x3 conv weight `[16, 16, 27]` as the `stem2_conv` kernel's B fragments:
/// for tap `t` and lane `l` (`g = l / 4`, `th = l % 4`) four FP16 pairs, `[n-tile 0: k-half 0, 1, n-tile 1:
/// k-half 0, 1]`, each pair the weights from input channels `2 th + 8 half + {0, 1}` (low half first) to
/// output channel `8 ntile + g`, the `mma.m16n8k16` B layout with K the input channels of one tap
pub fn stem2_frags(w: &[f32]) -> Vec<u32> {
    const C: usize = 16;
    const TAPS: usize = 27;
    assert_eq!(w.len(), C * C * TAPS, "stem2_frags takes a [16, 16, 27] weight");
    let at = |k: usize, c: usize, t: usize| half::f16::from_f32(w[(k * C + c) * TAPS + t]).to_bits() as u32;
    let mut out = Vec::with_capacity(TAPS * 32 * 4);
    for t in 0..TAPS {
        for lane in 0..32 {
            let (g, th) = (lane / 4, lane % 4);
            for nt in 0..2 {
                for half in 0..2 {
                    let (k, c) = (8 * nt + g, 2 * th + 8 * half);
                    out.push(at(k, c, t) | (at(k, c + 1, t) << 16));
                }
            }
        }
    }
    out
}

/// A PyTorch conv weight `[K, C, *kernel]` as FP16 channels-last `[K, *kernel, C]`
pub fn conv_nhwc(w: &[f32], k: usize, c: usize, kernel: usize) -> Vec<half::f16> {
    conv_nhwc_f32(w, k, c, kernel).into_iter().map(half::f16::from_f32).collect()
}

/// [`conv_nhwc`] without the FP16 rounding
pub fn conv_nhwc_f32(w: &[f32], k: usize, c: usize, kernel: usize) -> Vec<f32> {
    let mut out = vec![0.0; w.len()];
    for ko in 0..k {
        for ci in 0..c {
            for s in 0..kernel {
                out[(ko * kernel + s) * c + ci] = w[(ko * c + ci) * kernel + s];
            }
        }
    }
    out
}

/// A PyTorch `ConvTranspose2d(k=2, s=2)` weight `[Cin, Cout, 2, 2]` as the FP16 GEMM operand
/// `[Cin, 4 * Cout]`, column `(dy * 2 + dx) * Cout + co`
pub fn transp_gemm(w: &[f32], cin: usize, cout: usize) -> Vec<half::f16> {
    let mut out = vec![half::f16::ZERO; w.len()];
    for ci in 0..cin {
        for co in 0..cout {
            for d in 0..4 {
                out[ci * 4 * cout + d * cout + co] = half::f16::from_f32(w[(ci * cout + co) * 4 + d]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_move_channels_last() {
        // w[k][c][s] = 100k + 10c + s
        let (k, c, s) = (2, 3, 4);
        let w: Vec<f32> = (0..k * c * s).map(|i| (100 * (i / (c * s)) + 10 * (i / s % c) + i % s) as f32).collect();
        let n = conv_nhwc(&w, k, c, s);
        assert_eq!(n[(s + 2) * c + 1].to_f32(), 112.0);
        let t = transp_gemm(&(0..2 * 3 * 4).map(|i| i as f32).collect::<Vec<_>>(), 2, 3);
        // w[ci=1][co=2][d=3] = (1*3+2)*4+3 = 23 lands at column 3*3+2 of row 1
        assert_eq!(t[12 + 3 * 3 + 2].to_f32(), 23.0);
    }
}

#[cfg(test)]
mod contract_tests {
    use std::path::PathBuf;

    use super::Checkpoint;
    use crate::error::is_unsupported;
    use crate::fast::ReaderPrecision;

    /// villa's `model_config` for `hybrid_3d2d-seed42`, reader-v2, dense_native and TRACE v3B
    const SEED42_CONFIG: &str = r#"{"autoconfigure": true, "basic_decoder_block": "ConvBlock",
        "basic_encoder_block": "BasicBlockD", "stem_channels": 16, "z_projection_mode": "none"}"#;

    fn export_with(amp: &str) -> PathBuf {
        export_full(amp, SEED42_CONFIG)
    }

    fn export_full(amp: &str, config: &str) -> PathBuf {
        // one directory per call: the tests run in parallel and each removes its own
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let tag = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("inkrs-weights-test-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let st = dir.join("w.safetensors");
        std::fs::write(&st, b"").unwrap();
        let side = format!(
            r#"{{"crop_size": [17, 128, 128], "input_pad_depth_to": null, "preprocessing": "tifxyz_robust",
                "amp_dtype": {amp}, "targets": {{"ink": {{"activation": "none", "out_channels": 1}}}},
                "keys": {{}}, "aliases": {{}},
                "model_type": "vesuvius_unet_3d_stem_2d", "model_config": {config}}}"#
        );
        std::fs::write(dir.join("w.json"), side).unwrap();
        st
    }

    #[test]
    fn a_checkpoint_without_fp16_autocast_is_unsupported_at_fp16_only() {
        for amp in ["null", "\"bfloat16\""] {
            let st = export_with(amp);
            let err = Checkpoint::load(&st, ReaderPrecision::Fp16).err().expect("loads");
            assert!(is_unsupported(&err) && format!("{err:#}").contains("amp dtype"), "{amp}: {err:#}");
            // fp32 runs it as villa does on MPS: the contract passes and only the empty tensor file fails
            let err = Checkpoint::load(&st, ReaderPrecision::Fp32).err().expect("loads");
            assert!(!is_unsupported(&err), "{amp} at fp32: {err:#}");
            std::fs::remove_dir_all(st.parent().unwrap()).unwrap();
        }
        let st = export_with("\"float32\"");
        let err = Checkpoint::load(&st, ReaderPrecision::Fp32).err().expect("loads");
        assert!(is_unsupported(&err) && format!("{err:#}").contains("amp dtype"), "{err:#}");
        std::fs::remove_dir_all(st.parent().unwrap()).unwrap();
    }

    #[test]
    fn another_network_config_is_unsupported() {
        // the supported config passes the contract and fails only on the empty tensor file
        // and so does Reader V3's and Rectina M4's, which spells out the autoconfigured values
        let v3 = SEED42_CONFIG.replace(
            "}",
            r#", "features_per_stage": [32, 64, 128, 256, 320, 320], "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1], "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            "pool_op_kernel_sizes": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]}"#,
        );
        for config in [SEED42_CONFIG, v3.as_str()] {
            let st = export_full("\"float16\"", config);
            let err = Checkpoint::load(&st, ReaderPrecision::Fp16).err().expect("loads");
            assert!(!is_unsupported(&err), "{err:#}");
            std::fs::remove_dir_all(st.parent().unwrap()).unwrap();
        }
        for config in [
            "null",
            "{}",
            &SEED42_CONFIG.replace(r#""z_projection_mode": "none""#, r#""z_projection_mode": "max""#),
            &SEED42_CONFIG.replace("16", "32"),
            &SEED42_CONFIG.replace(r#""autoconfigure": true"#, r#""autoconfigure": false"#),
            &SEED42_CONFIG.replace("}", r#", "squeeze_excitation": true}"#),
            // a stride that keeps every tensor shape
            &SEED42_CONFIG.replace("}", r#", "strides": [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]}"#),
            &SEED42_CONFIG.replace("}", r#", "features_per_stage": [32, 64, 128, 256, 320]}"#),
        ] {
            let st = export_full("\"float16\"", config);
            let err = Checkpoint::load(&st, ReaderPrecision::Fp16).err().expect("loads");
            assert!(is_unsupported(&err) && format!("{err:#}").contains("model_config"), "{config}: {err:#}");
            std::fs::remove_dir_all(st.parent().unwrap()).unwrap();
        }
    }

    #[test]
    fn a_sidecar_without_amp_dtype_is_corrupt_not_unsupported() {
        let st = export_with("\"float16\"");
        let side = std::fs::read_to_string(st.with_extension("json")).unwrap();
        std::fs::write(st.with_extension("json"), side.replace(r#""amp_dtype": "float16","#, "")).unwrap();
        let err = Checkpoint::load(&st, ReaderPrecision::Fp16).err().expect("loads");
        assert!(!is_unsupported(&err) && format!("{err:#}").contains("amp_dtype"), "{err:#}");
        std::fs::remove_dir_all(st.parent().unwrap()).unwrap();
    }
}
