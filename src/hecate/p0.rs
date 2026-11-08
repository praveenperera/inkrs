//! P0-H: one Hecate forward on listed tiles, every tapped activation dumped as channels-last
//! float32 `.npy` for `scripts/p0_hecate.py compare`

use std::path::Path;

use color_eyre::eyre::{Result, WrapErr, ensure};
use half::bf16;
use serde::{Deserialize, Serialize};

use crate::checks::write_npy;
use crate::dnn::{Dnn, Pick};
use crate::engine::{resolve_weights, weights_dir};
use crate::gpu::Gpu;
use crate::hecate::model::{Model, check_contract};
use crate::hecate::{D, P, PIX, Precision, central_planes, read_volume, u32_extent};
use crate::model::Ctx;
use crate::sched::DESC;
use crate::weights::Checkpoint;

/// One tile: its corner and direction
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct TileSpec {
    y0: usize,
    x0: usize,
    reverse: bool,
}

/// Runs the tiles in `tiles` (a JSON list of `{y0, x0, reverse}`) as one batch and writes each
/// tap, plus `prob` (`[B, 64, 64]`, the probability times the support), to `out` with a
/// `manifest.json`; fails on any non-finite value
pub fn run(input: &Path, ckpt: &Path, tiles: &Path, out: &Path, precision: Precision, pick: Pick) -> Result<()> {
    let tiles: Vec<TileSpec> =
        serde_json::from_slice(&std::fs::read(tiles).wrap_err_with(|| format!("read {}", tiles.display()))?)?;
    ensure!(!tiles.is_empty(), "no tiles");
    let batch = tiles.len();
    u32_extent("tile probabilities", batch.saturating_mul(PIX))?;
    let vol = read_volume(input)?;
    let mut desc = Vec::with_capacity(batch * DESC);
    for (i, t) in tiles.iter().enumerate() {
        ensure!(t.y0 + P <= vol.height && t.x0 + P <= vol.width, "tile ({}, {}) not inside the volume", t.y0, t.x0);
        let (first, step) = central_planes(vol.depth, t.reverse)?;
        desc.extend_from_slice(&[t.y0 as i32, t.x0 as i32, first, step, i as i32, 0, 0, 0]);
    }

    let (ck, sidecar) = Checkpoint::read(&resolve_weights(ckpt, &weights_dir())?)?;
    check_contract(&sidecar)?;
    let gpu = Gpu::open()?;
    let dnn = Dnn::new(&gpu)?;
    let ctx = Ctx { gpu: &gpu, dnn: &dnn, pick, max_workspace: 1 << 30 };
    let model = Model::new(&ctx, &ck, batch, precision)?;
    let schema = ck.finish()?;
    let vbuf = gpu.upload(&vol.data)?;
    let dbuf = gpu.upload(&desc)?;
    let prob = gpu.alloc_fill(batch * PIX * 4, 0xFF)?;
    model.forward(&ctx, (&vbuf, vol.height, vol.width), dbuf.arg::<i32>(), &prob)?;
    gpu.sync()?;

    std::fs::create_dir_all(out)?;
    let mut manifest = Vec::new();
    let mut taps: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    for t in model.taps() {
        let all: Vec<f32> = if t.bf16 {
            gpu.download::<bf16>(t.buf)?.iter().map(|v| v.to_f32()).collect()
        } else {
            gpu.download::<f32>(t.buf)?
        };
        let n: usize = t.shape.iter().product();
        ensure!(all.len() >= n, "tap {}: {} values for shape {:?}", t.name, all.len(), t.shape);
        let c = t.shape[t.shape.len() - 1];
        let data: Vec<f32> = all[..n].chunks(c).flat_map(|px| px[..t.keep_c].iter().copied()).collect();
        let mut shape = t.shape.clone();
        *shape.last_mut().unwrap() = t.keep_c;
        taps.push((t.name, shape, data));
    }
    let p: Vec<f32> = gpu.download(&prob)?;
    taps.push(("prob".into(), vec![batch, P, P], p));
    for (name, shape, data) in &taps {
        let bad = data.iter().filter(|v| !v.is_finite()).count();
        ensure!(bad == 0, "tap {name}: {bad} non-finite values");
        write_npy(&out.join(format!("{name}.npy")), shape, data)?;
        manifest.push(serde_json::json!({"name": name, "shape": shape, "layout": "channels_last"}));
    }
    let algos: Vec<String> =
        model.plans().iter().map(|p| format!("{} | {}", p.shape, p.algo)).chain(model.fused_algos()).collect();
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "tiles": batch,
            // the reference's own list, so the compare refuses dumps of different tiles
            "tile_list": tiles,
            "depth": D,
            "precision": precision,
            "stem_channels": model.stem_c,
            "taps": manifest,
            "algos": algos,
            "schema": schema,
            "cudnn": Dnn::version(),
            "compute_capability": gpu.inner.compute_capability().to_string(),
            "ptx_tier": gpu.inner.ptx_tier().to_string(),
        }))?,
    )?;
    println!("p0-hecate: {} taps for {batch} tiles in {}", taps.len(), out.display());
    Ok(())
}
