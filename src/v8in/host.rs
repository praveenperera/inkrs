//! The backend-independent half of the v8-in engine: the export's contract, the run settings and report, the
//! 24-layer window, the convention's coverage and tile schedule, the BatchNorm fold and the `.npy` writers.
//! The CUDA engine (`v8in::engine`, `v8in::model`) and the Metal one (`metal::v8in`) both build on it and
//! re-export it, so the CLI and the Python binding see one API on either backend

use std::path::Path;

use color_eyre::eyre::{Result, WrapErr, ensure, eyre};
use serde::Serialize;
use serde_json::Value;

use crate::dnn::Pick;
use crate::hecate::u32_extent;
use crate::sched::DESC;
use crate::v8in::{Convention, D, P, PIX, admit, close7};
use crate::weights::Checkpoint;
use crate::zarr::VolumeRef;

/// BatchNorm and InstanceNorm eps of every norm in the release, checked against the export's sidecar
pub const EPS: f64 = 1e-5;
/// Bottleneck blocks per layer, ResNet-50
pub const BLOCKS: [usize; 4] = [3, 4, 6, 3];
/// Bottleneck planes per layer; each block outputs four times as many channels
pub const PLANES: [usize; 4] = [64, 128, 256, 512];
/// Encoder output channels per level, which are also the decoder widths
pub const DIMS: [usize; 4] = [256, 512, 1024, 2048];
/// The checkpoint names [`check_contract`] accepts
pub const READERS: [&str; 4] = ["v8in", "v8in-ahn-s0", "v8in-ahn-s1", "v8in-ahn-s2"];

/// The fields of the export's sidecar this engine relies on, beyond its keys: the reader name and the
/// map convention its maps default to
///
/// The checkpoint's own `config.reverse_layers` is never read. A-hn's sets it to true, but the runner that
/// made the A-hn maps ignores it and reads stored order, so the depth direction comes from the caller only
pub fn check_contract(sidecar: &Value) -> Result<(String, Convention)> {
    let reader = sidecar.get("reader").and_then(Value::as_str).unwrap_or_default().to_string();
    ensure!(READERS.contains(&reader.as_str()), "export sidecar reader {reader:?} is not one of {READERS:?}");
    ensure!(sidecar.get("architecture_family") == Some(&Value::from("v8in")), "not a v8-in export");
    ensure!(sidecar.get("eps") == Some(&Value::from(1e-5)), "export sidecar eps {:?}", sidecar.get("eps"));
    let cfg = &sidecar["config"];
    let want = serde_json::json!({"in_depth": D, "tile_size": P, "input_upsample_depth": 96, "input_upsample_hw": 256,
        "encoder_depth": 50, "clip_max": 200});
    for (k, v) in want.as_object().into_iter().flatten() {
        ensure!(cfg.get(k) == Some(v), "export sidecar config.{k} is {:?}, need {v}", cfg.get(k));
    }
    let mc =
        sidecar.get("map_convention").ok_or_else(|| eyre!("export sidecar has no map_convention; re-export it"))?;
    let conv = Convention::from_name(mc.get("name").and_then(Value::as_str).unwrap_or_default())?;
    ensure!(*mc == conv.spec(), "export sidecar map_convention {mc} differs from inkrs's {}", conv.spec());
    Ok((reader, conv))
}

/// Whether the sidecar at `export.with_extension("json")` is a v8-in family export, for the commands and
/// the Python `Engine` that pick an engine by the export's metadata
pub fn is_v8in_export(export: &Path) -> bool {
    std::fs::read(export.with_extension("json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .is_some_and(|v| v.get("architecture_family") == Some(&Value::from("v8in")))
}

/// An eval BatchNorm as float64 `(scale, shift)` per channel: `scale = weight / sqrt(running_var +
/// eps)`, `shift = bias - running_mean * scale`
pub fn batch_norm(ck: &Checkpoint, key: &str, c: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let [w, b, m, v] = ["weight", "bias", "running_mean", "running_var"].map(|f| ck.take(&format!("{key}.{f}"), &[c]));
    let (w, b, m, v) = (w?, b?, m?, v?);
    let scale: Vec<f64> = (0..c).map(|i| w[i] as f64 / (v[i] as f64 + EPS).sqrt()).collect();
    let shift = (0..c).map(|i| b[i] as f64 - m[i] as f64 * scale[i]).collect();
    Ok((scale, shift))
}

/// Whether `INKRS_V8_UNFUSED=1` forces the unfused path (CUDA; the Metal convs always fuse their epilogue)
pub fn force_unfused() -> bool {
    std::env::var("INKRS_V8_UNFUSED").is_ok_and(|v| v == "1")
}

/// Run settings
#[derive(Clone, Debug)]
pub struct Options {
    /// The map convention; `None` takes the export's own (its sidecar's `map_convention`)
    pub convention: Option<Convention>,
    /// First stored layer of the 24; `None` takes the central 24, `(depth - 24) / 2`, as ink8um does and
    /// as the 306B runner's `--layer0 8` does on its 41-layer render
    pub layer0: Option<usize>,
    /// Flip the 24 layers' depth order before the model
    pub reverse: bool,
    pub batch: usize,
    pub pick: Pick,
    pub max_workspace: usize,
}

/// Timings and choices of a run
#[derive(Debug, Default, Serialize)]
pub struct Report {
    pub reader: String,
    pub convention: String,
    pub layer0: usize,
    pub reverse: bool,
    pub batch: usize,
    /// cuDNN's version; 0 on Metal
    pub cudnn: usize,
    /// The CUDA compute capability, or the Metal device's name
    pub compute_capability: String,
    /// The PTX tier, or `metal`
    pub ptx_tier: String,
    pub load_weights_s: f64,
    pub open_gpu_s: f64,
    pub plan_s: f64,
    pub read_volume_s: f64,
    pub schedule_s: f64,
    pub upload_s: f64,
    /// Queueing every batch and the blend through the map's download
    pub run_s: f64,
    /// GPU time from the first batch to the blend, by events (CUDA) or command-buffer timestamps (Metal)
    pub gpu_s: f64,
    pub finite_check_s: f64,
    pub total_s: f64,
    pub shape: [usize; 3],
    pub positions: usize,
    pub tiles: usize,
    pub batches: usize,
    pub tiles_per_s: f64,
    pub ms_per_tile: f64,
    pub plans: Vec<String>,
    pub unfused: Vec<String>,
    pub schema: (usize, usize, usize),
}

/// The map and its report
pub struct Output {
    pub map: Vec<f32>,
    pub height: usize,
    pub width: usize,
    /// The admitted tiles' top-left `[y, x]`, in run order
    pub tiles: Vec<[u32; 2]>,
    pub report: Report,
}

impl Output {
    /// 1 where at least one admitted tile covers the pixel: the map's coverage, independent of its values
    /// (a release map is 0 both where no tile ran and where a tile said 0), for the parity gate
    pub fn coverage(&self) -> Vec<u8> {
        let mut m = vec![0u8; self.height * self.width];
        for &[y, x] in &self.tiles {
            for row in y as usize..y as usize + P {
                m[row * self.width + x as usize..row * self.width + x as usize + P].fill(1);
            }
        }
        m
    }
}

/// The 24 layers the model reads, `[24, H, W]` in the model's order, and the stored index of layer 0
pub fn window_layers(vol: VolumeRef<'_>, layer0: Option<usize>, reverse: bool) -> Result<(Vec<u8>, usize)> {
    ensure!(vol.depth >= D, "the volume has {} layers; v8-in needs {D}", vol.depth);
    let l0 = layer0.unwrap_or((vol.depth - D) / 2);
    ensure!(l0 + D <= vol.depth, "layers {l0}..{} past the volume's {}", l0 + D, vol.depth);
    let plane = vol.height * vol.width;
    let mut out = Vec::with_capacity(D * plane);
    for z in 0..D {
        let src = if reverse { l0 + D - 1 - z } else { l0 + z };
        out.extend_from_slice(&vol.data[src * plane..(src + 1) * plane]);
    }
    Ok((out, l0))
}

/// The convention's coverage mask on the host: the release's closed `max over the window > 0`, or the 306B
/// rule's stored layer `layer0 + 12`
pub fn coverage(vol: VolumeRef<'_>, layer0: usize, convention: Convention) -> Vec<bool> {
    use rayon::prelude::*;

    let plane = vol.height * vol.width;
    match convention {
        Convention::Release => {
            let any: Vec<bool> =
                (0..plane).into_par_iter().map(|i| (0..D).any(|z| vol.data[(layer0 + z) * plane + i] > 0)).collect();
            close7(&any, vol.height, vol.width)
        }
        Convention::Ahn306b => vol.data[(layer0 + 12) * plane..(layer0 + 13) * plane].iter().map(|&v| v > 0).collect(),
    }
}

/// The tiles of one map in run order and the descriptors the GPU reads them by
pub struct Schedule {
    /// The 24-layer window, `[24, H, W]`, in the model's order
    pub layers: Vec<u8>,
    pub layer0: usize,
    /// The lattice's rows and columns
    pub ys: Vec<usize>,
    pub xs: Vec<usize>,
    /// `[ys, xs]`: each lattice position's slot, or -1 where no tile was admitted, for the blend
    pub grid: Vec<i32>,
    /// `[batches * batch * DESC]`: y0, x0, first layer, step, slot (-1 for a padding tile), then unused
    pub desc: Vec<i32>,
    /// The admitted tiles' top-left `[y, x]`, slot order
    pub tiles: Vec<[u32; 2]>,
    pub batches: usize,
}

impl Schedule {
    /// Admits `vol`'s tiles by `conv` and pads the last batch of `batch` with copies of the first tile
    /// marked slot -1, which the network runs and the head discards
    pub fn new(vol: VolumeRef<'_>, conv: Convention, opts: &Options, batch: usize) -> Result<Self> {
        let (height, width) = (vol.height, vol.width);
        let (layers, layer0) = window_layers(vol, opts.layer0, opts.reverse)?;
        let cov = coverage(vol, layer0, conv);
        let (ys, xs, keep) = admit(&cov, height, width, conv);
        drop(cov);
        let mut grid = vec![-1i32; keep.len()];
        let mut desc: Vec<i32> = Vec::new();
        let mut tiles = Vec::new();
        for (i, &k) in keep.iter().enumerate() {
            if k {
                let (y, x) = (ys[i / xs.len()], xs[i % xs.len()]);
                let slot = i32::try_from(tiles.len()).map_err(|_| eyre!("more than 2^31 tiles"))?;
                grid[i] = slot;
                desc.extend_from_slice(&[y as i32, x as i32, 0, 1, slot, 0, 0, 0]);
                tiles.push([y as u32, x as u32]);
            }
        }
        let slots = tiles.len();
        let batches = slots.div_ceil(batch);
        let pad = if slots == 0 { [0, 0, 0, 1, -1, 0, 0, 0] } else { [desc[0], desc[1], 0, 1, -1, 0, 0, 0] };
        while desc.len() < batches * batch * DESC {
            desc.extend_from_slice(&pad);
        }
        u32_extent("tile probabilities", slots.saturating_mul(PIX))?;
        u32_extent("window", layers.len())?;
        Ok(Self { layers, layer0, ys, xs, grid, desc, tiles, batches })
    }
}

/// The descriptors of P0's batches: `yx` (top-left corners) in chunks of `batch`, slot = the tile's index,
/// the last chunk padded with copies of its first tile marked slot -1
pub fn p0_desc(yx: &[[usize; 2]], batch: usize, height: usize, width: usize) -> Result<Vec<Vec<i32>>> {
    yx.chunks(batch)
        .enumerate()
        .map(|(bi, chunk)| {
            let mut desc = Vec::with_capacity(batch * DESC);
            for (k, &[y, x]) in chunk.iter().enumerate() {
                ensure!(y + P <= height && x + P <= width, "tile ({y}, {x}) past the volume");
                desc.extend_from_slice(&[y as i32, x as i32, 0, 1, (bi * batch + k) as i32, 0, 0, 0]);
            }
            while desc.len() < batch * DESC {
                desc.extend_from_slice(&[desc[0], desc[1], 0, 1, -1, 0, 0, 0]);
            }
            Ok(desc)
        })
        .collect()
}

/// What `Engine::p0` returns
pub struct P0Out {
    /// `[n, 64, 64]`
    pub probs: Vec<f32>,
    /// (name, channels-last shape, FP16 bits)
    pub taps: Vec<(String, Vec<usize>, Vec<u16>)>,
    /// `[n, 2]`: each tile's InstanceNorm as the scale and shift of `min(v, 200) / 255`; empty without stages
    pub stats: Vec<f32>,
    /// The run's settings, for the compare script to check against the reference's
    pub meta: serde_json::Value,
}

/// The depth indices a P0 stage tap keeps of a stage `d` deep: both borders, their neighbours and the
/// middle, where the stem's border classes and the stride-2 layers' edges sit
pub fn stage_depths(d: usize) -> Vec<usize> {
    let mut z = vec![0, 1, d / 2, d - 2, d - 1];
    z.dedup();
    z
}

/// Writes `[h, w]` float32 as a C-order `.npy` (version 1), via `.partial` and a rename
pub fn write_npy_f32(path: &Path, data: &[f32], shape: &[usize]) -> Result<()> {
    write_npy(path, bytemuck::cast_slice(data), "<f4", shape)
}

/// Writes raw little-endian `bytes` of dtype `descr` and `shape` as a C-order `.npy`
pub fn write_npy(path: &Path, bytes: &[u8], descr: &str, shape: &[usize]) -> Result<()> {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let tuple = if dims.len() == 1 { format!("({},)", dims[0]) } else { format!("({})", dims.join(", ")) };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {tuple}, }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(bytes);
    let partial = path.with_extension("partial");
    std::fs::write(&partial, &out).wrap_err_with(|| format!("write {}", partial.display()))?;
    std::fs::rename(&partial, path).wrap_err_with(|| format!("rename to {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A-hn's `config.json` says `reverse_layers: true`, but the runner that made its maps ignores it: the
    /// A-hn exports resolve to the 306b convention with stored order, and with the default `reverse = false`
    /// the model reads stored layer `layer0 + z` at depth `z`
    #[test]
    fn ahn_reverse_layers_is_ignored() {
        let (depth, h, w) = (41, 2, 3);
        let data: Vec<u8> = (0..depth).flat_map(|z| std::iter::repeat_n(z as u8, h * w)).collect();
        let vol = VolumeRef::new(&data, depth, h, w).unwrap();
        let (layers, l0) = window_layers(vol, None, false).unwrap();
        assert_eq!(l0, 8);
        let order: Vec<u8> = layers.chunks(h * w).map(|c| c[0]).collect();
        assert_eq!(order, (8..32).collect::<Vec<u8>>());
        let (rev, _) = window_layers(vol, None, true).unwrap();
        assert_eq!(rev.chunks(h * w).map(|c| c[0]).collect::<Vec<u8>>(), (8..32).rev().collect::<Vec<u8>>());

        let mut seen = 0;
        for name in ["v8in-ahn-s0", "v8in-ahn-s1", "v8in-ahn-s2"] {
            let sidecar = crate::engine::weights_dir().join(format!("{name}.json"));
            let Ok(bytes) = std::fs::read(&sidecar) else {
                eprintln!("skipped: no export at {}", sidecar.display());
                continue;
            };
            let side: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(side["config"]["reverse_layers"], serde_json::Value::Bool(true), "{name}: upstream flag");
            assert_eq!(side["upstream_reverse_layers"], serde_json::Value::Bool(true), "{name}");
            let (_, conv) = check_contract(&side).unwrap();
            assert_eq!(conv, Convention::Ahn306b, "{name}");
            assert_eq!(conv.spec()["reverse"], serde_json::Value::Bool(false), "{name}");
            seen += 1;
        }
        eprintln!("{seen} A-hn sidecars checked");
    }

    /// Every tensor name and shape of the release, written out independently of the models, so a
    /// mismatched export fails on the CPU before any GPU work
    #[test]
    fn exports_match_the_architecture() {
        for name in READERS {
            let path = crate::engine::weights_dir().join(format!("{name}.safetensors"));
            if !path.exists() {
                eprintln!("skipped: no export at {}", path.display());
                continue;
            }
            let (ck, sidecar) = Checkpoint::read(&path).unwrap();
            let want = if name == "v8in" { Convention::Release } else { Convention::Ahn306b };
            assert_eq!(check_contract(&sidecar).unwrap(), (name.to_string(), want));
            let bn = |key: &str, c: usize| {
                for f in ["weight", "bias", "running_mean", "running_var"] {
                    ck.take(&format!("{key}.{f}"), &[c]).unwrap();
                }
            };
            ck.take("normalization.weight", &[1]).unwrap();
            ck.take("normalization.bias", &[1]).unwrap();
            ck.take("backbone.conv1.weight", &[64, 1, 7, 7, 7]).unwrap();
            bn("backbone.bn1", 64);
            let mut cin = 64;
            for (l, (&planes, &nb)) in PLANES.iter().zip(&BLOCKS).enumerate() {
                for j in 0..nb {
                    let key = format!("backbone.layer{}.{j}", l + 1);
                    ck.take(&format!("{key}.conv1.weight"), &[planes, cin, 1, 1, 1]).unwrap();
                    ck.take(&format!("{key}.conv2.weight"), &[planes, planes, 3, 3, 3]).unwrap();
                    ck.take(&format!("{key}.conv3.weight"), &[4 * planes, planes, 1, 1, 1]).unwrap();
                    for (bk, c) in [("bn1", planes), ("bn2", planes), ("bn3", 4 * planes)] {
                        bn(&format!("{key}.{bk}"), c);
                    }
                    if j == 0 {
                        ck.take(&format!("{key}.downsample.0.weight"), &[4 * planes, cin, 1, 1, 1]).unwrap();
                        bn(&format!("{key}.downsample.1"), 4 * planes);
                    }
                    cin = 4 * planes;
                }
            }
            for (i, (cat, out)) in [(768, 256), (1536, 512), (3072, 1024)].into_iter().enumerate() {
                ck.take(&format!("decoder.convs.{i}.0.weight"), &[out, cat, 3, 3]).unwrap();
                bn(&format!("decoder.convs.{i}.1"), out);
            }
            ck.take("decoder.logit.weight", &[1, 256, 1, 1]).unwrap();
            ck.take("decoder.logit.bias", &[1]).unwrap();
            assert_eq!(ck.finish().unwrap().2, 83_374_531, "{name}");
        }
    }
}
