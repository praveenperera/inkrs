//! `hecate.py --output` on the GPU: tile the central 16 planes at `stride`, skip the tiles whose
//! planes are all zero, run the network in fixed batches with no host synchronisation between
//! them, blend each map as `predict` does and write its PNG and JSON
//!
//! `predict` adds every tile position's window to the denominator, kept or not, and each kept
//! tile's `probability * support * window` to the numerator in tile order, in float32. The blend
//! kernel gathers per pixel in that order, so both sums match `hecate.py`'s for the same tile
//! probabilities

use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use color_eyre::eyre::{Result, WrapErr, ensure, eyre};
use cudarc::driver::result as cur;
use cudarc::driver::sys as cu;
use rayon::prelude::*;
use serde::Serialize;

use crate::dnn::{Dnn, Pick, algo_cache};
use crate::engine::resolve_weights;
use crate::gpu::{Arg, Gpu};
use crate::hecate::model::{Model, check_contract};
use crate::hecate::{D, P, PIX, Precision, SAMPLING_UM, axis_starts, central_planes, read_volume, u32_extent, window};
use crate::model::Ctx;
use crate::sched::DESC;
use crate::weights::Checkpoint;
use crate::zarr::Volume;

/// One output map: the direction and the PNG path; its JSON goes next to it
#[derive(Clone, Debug)]
pub struct MapSpec {
    pub reverse: bool,
    pub png: PathBuf,
}

/// Run settings
pub struct Options {
    pub stride: usize,
    /// Whether the settings came from the `--fast` preset ([`crate::fast::HECATE`]); recorded only
    pub fast: bool,
    pub batch: usize,
    pub precision: Precision,
    pub pick: Pick,
    pub max_workspace: usize,
    /// The render's voxel size as given, checked against the model's and written to the JSON
    pub spacing_um: f64,
    /// The `--checkpoint` argument, whose file name the JSON records as `hecate.py` does
    pub checkpoint: PathBuf,
    /// Where a `.pth` checkpoint's export is looked up ([`crate::engine::weights_dir`] for the CLI)
    pub weights_dir: PathBuf,
    /// The CUDA device ordinal
    pub device: usize,
    pub started: Instant,
}

/// Timings and choices of a run
#[derive(Debug, Default, Serialize)]
pub struct Report {
    pub precision: String,
    /// Whether the run used the `--fast` preset, which gives up map parity with `hecate.py`'s defaults
    pub fast: bool,
    pub stride: usize,
    pub batch: usize,
    pub cudnn: usize,
    /// The device's compute capability and the PTX tier loaded on it
    pub compute_capability: String,
    pub ptx_tier: String,
    /// How the kernels were loaded: a cubin compiled ahead (`sm_120 SASS`) or the PTX through the
    /// driver's JIT (`PTX, JIT-compiled`); `metal` on Metal
    pub kernels: String,
    pub stem_channels: usize,
    pub load_weights_s: f64,
    pub open_gpu_s: f64,
    pub plan_s: f64,
    /// Plans (unfused and fused, as `algos` lists them) whose `--algo find` choice came from the
    /// on-disk algorithm cache
    pub algo_cache_hits: usize,
    /// The algorithm cache's state, and why it is off when it is; None only before planning
    pub algo_cache: Option<algo_cache::Status>,
    pub read_volume_s: f64,
    pub schedule_s: f64,
    pub upload_s: f64,
    /// Queueing every batch and blend through the last map's download
    pub run_s: f64,
    /// GPU time from the first batch to the last blend, by events
    pub gpu_s: f64,
    /// Downloading every tile probability and refusing a non-finite one
    pub finite_check_s: f64,
    pub write_s: f64,
    pub total_s: f64,
    pub shape: [usize; 3],
    /// Tile positions per map, and the kept (non-empty) ones per map
    pub positions: usize,
    pub map_tiles: Vec<(String, usize)>,
    pub tiles: usize,
    pub batches: usize,
    pub tiles_per_s: f64,
    pub algos: Vec<String>,
    pub schema: (usize, usize, usize),
}

/// Refuses a request [`run`] would refuse before touching the GPU: no maps, a bad batch or stride, a
/// spacing hecate.py rejects, or an output that exists or is not a .png. The Python module calls it
/// before it loads any CUDA library, so these stay `ValueError`s on a machine without one
pub fn check_request(maps: &[MapSpec], opts: &Options) -> Result<()> {
    ensure!(!maps.is_empty(), "no maps to run");
    ensure!(opts.batch > 0, "batch size must be positive");
    ensure!((1..=P).contains(&opts.stride), "stride {} outside 1..={P}", opts.stride);
    // hecate.py: `abs(spacing / sampling - 1) > .02` is an error
    ensure!(
        (opts.spacing_um / SAMPLING_UM - 1.0).abs() <= 0.02,
        "--spacing-um {}: resample the render to {SAMPLING_UM} um first (no automatic resampling)",
        opts.spacing_um
    );
    for m in maps {
        ensure!(
            m.png.extension().is_some_and(|e| e.eq_ignore_ascii_case("png")) && !m.png.exists(),
            "choose a new output file name ending in .png: {}",
            m.png.display()
        );
    }
    Ok(())
}

/// Runs `maps` over the volume at `input` with the Hecate export for `ckpt`
pub fn run(input: &Path, ckpt: &Path, maps: &[MapSpec], opts: &Options) -> Result<Report> {
    check_request(maps, opts)?;
    let mut rep = Report {
        precision: opts.precision.name().into(),
        fast: opts.fast,
        stride: opts.stride,
        batch: opts.batch,
        ..Default::default()
    };

    let t = Instant::now();
    let (ck, sidecar) = Checkpoint::read(&resolve_weights(ckpt, &opts.weights_dir)?)?;
    check_contract(&sidecar)?;
    rep.load_weights_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let gpu = Gpu::open_device(opts.device)?;
    let dnn = Dnn::new(&gpu)?;
    rep.open_gpu_s = t.elapsed().as_secs_f64();
    rep.cudnn = Dnn::version();
    rep.compute_capability = gpu.inner.compute_capability().to_string();
    rep.ptx_tier = gpu.inner.ptx_tier().to_string();
    rep.kernels = gpu.kernels().to_string();

    let t = Instant::now();
    let ctx = Ctx { gpu: &gpu, dnn: &dnn, pick: opts.pick, max_workspace: opts.max_workspace };
    let model = Model::new(&ctx, &ck, opts.batch, opts.precision)?;
    rep.schema = ck.finish()?;
    drop(ck);
    rep.stem_channels = model.stem_c;
    rep.algos = model.plans().iter().map(|p| format!("{} | {}", p.shape, p.algo)).chain(model.fused_algos()).collect();
    rep.algo_cache_hits =
        model.plans().iter().filter(|p| p.cached).count() + model.fused_plans().iter().filter(|f| f.cached).count();
    rep.algo_cache = Some(algo_cache::status());
    gpu.sync()?;
    rep.plan_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let vol = read_volume(input)?;
    let (height, width) = (vol.height, vol.width);
    rep.shape = [vol.depth, height, width];
    rep.read_volume_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let sched = Schedule::new(&vol, maps, opts.stride, opts.batch)?;
    u32_extent("tile probabilities", sched.slots.saturating_mul(PIX))?;
    rep.positions = sched.ys.len() * sched.xs.len();
    rep.map_tiles = maps.iter().zip(&sched.kept).map(|(m, k)| (m.png.display().to_string(), *k)).collect();
    rep.tiles = sched.slots;
    rep.batches = sched.batches;
    rep.schedule_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let vbuf = gpu.upload(&vol.data)?;
    drop(vol);
    let desc = gpu.upload(&sched.desc)?;
    let grids = sched.grids.iter().map(|g| gpu.upload(g)).collect::<Result<Vec<_>>>()?;
    let ys = gpu.upload(&sched.ys)?;
    let xs = gpu.upload(&sched.xs)?;
    let y_span = gpu.upload(&spans(&sched.ys, height))?;
    let x_span = gpu.upload(&spans(&sched.xs, width))?;
    let weight = gpu.upload(&window())?;
    // NaN-filled: a kept slot that is never written fails the finite check and the blend's
    let prob = gpu.alloc_fill(sched.slots.max(1) * PIX * 4, 0xFF)?;
    let outs = maps.iter().map(|_| gpu.alloc(height * width)).collect::<Result<Vec<_>>>()?;
    gpu.sync()?;
    rep.upload_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let (start_ev, end_ev) = (timing_event()?, timing_event()?);
    // SAFETY: our stream; the events outlive the records
    unsafe { cur::event::record(start_ev, gpu.cu_stream())? };
    for i in 0..sched.batches {
        let view = desc.view::<i32>(i * opts.batch * DESC, opts.batch * DESC);
        model.forward(&ctx, (&vbuf, height, width), view, &prob)?;
    }
    for (m, out) in outs.iter().enumerate() {
        gpu.launch_1d(
            "hec_blend",
            height * width,
            &[
                out.arg::<u8>(),
                prob.arg::<f32>(),
                grids[m].arg::<i32>(),
                ys.arg::<u32>(),
                xs.arg::<u32>(),
                y_span.arg::<[u32; 2]>(),
                x_span.arg::<[u32; 2]>(),
                weight.arg::<f32>(),
                Arg::U32(width as u32),
            ],
        )?;
    }
    // SAFETY: as above
    unsafe { cur::event::record(end_ev, gpu.cu_stream())? };
    let images = outs.iter().map(|o| gpu.download::<u8>(o)).collect::<Result<Vec<_>>>()?;
    gpu.sync()?;
    rep.run_s = t.elapsed().as_secs_f64();
    // SAFETY: both events recorded and complete after the sync
    rep.gpu_s = unsafe { cur::event::elapsed(start_ev, end_ev)? } as f64 / 1e3;
    // SAFETY: done with them
    unsafe {
        cur::event::destroy(start_ev)?;
        cur::event::destroy(end_ev)?;
    }
    rep.tiles_per_s = sched.slots as f64 / rep.gpu_s.max(1e-9);

    // always, as hecate.py raises on a non-finite probability before writing: the blend's clamp and
    // uint8 cast would otherwise turn a NaN into a plausible pixel
    let t = Instant::now();
    let all: Vec<f32> = gpu.download_range(&prob, 0, sched.slots * PIX)?;
    let bad = all.iter().filter(|v| !v.is_finite()).count();
    ensure!(bad == 0, "{bad} non-finite probabilities among {} tiles; no map written", sched.slots);
    drop(all);
    rep.finite_check_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    for (m, image) in maps.iter().zip(&images) {
        write_png(&m.png, image, height, width)?;
        let json = serde_json::json!({
            "checkpoint": opts.checkpoint.file_name().map(|n| n.to_string_lossy()),
            "sampling_um": opts.spacing_um,
            "shape_yx": [height, width],
            "reverse_z": m.reverse,
            "central_depth": D,
            "precision": opts.precision.name(),
            "stride": opts.stride,
            "probabilities": "uint8 = round(255 * probability); no min-max scaling",
        });
        let path = m.png.with_extension("json");
        std::fs::write(&path, serde_json::to_string_pretty(&json)? + "\n")
            .wrap_err_with(|| format!("write {}", path.display()))?;
    }
    rep.write_s = t.elapsed().as_secs_f64();
    rep.total_s = opts.started.elapsed().as_secs_f64();
    Ok(rep)
}

/// Refuses inputs whose index spaces overflow the kernels' u32 offsets: the volume (`hec_gather`
/// reads `z * h * w + y * w + x`) and the tile probabilities (`hec_blend` reads `slot * 4096 + pixel`)
/// Every map's tile grid and the batched descriptors of its kept tiles
pub(crate) struct Schedule {
    pub(crate) ys: Vec<u32>,
    pub(crate) xs: Vec<u32>,
    /// Per map, `[ny, nx]` slots, -1 for a tile `predict` skips
    pub(crate) grids: Vec<Vec<i32>>,
    kept: Vec<usize>,
    desc: Vec<i32>,
    pub(crate) slots: usize,
    batches: usize,
}

impl Schedule {
    pub(crate) fn new(vol: &Volume, maps: &[MapSpec], stride: usize, batch: usize) -> Result<Self> {
        let (h, w) = (vol.height, vol.width);
        let ys = axis_starts(h, stride);
        let xs = axis_starts(w, stride);
        let origins: Vec<(usize, usize)> = ys.iter().flat_map(|&y| xs.iter().map(move |&x| (y, x))).collect();
        let mut grids = Vec::new();
        let mut kept = Vec::new();
        let mut desc = Vec::new();
        let mut slots = 0usize;
        for m in maps {
            let (first, step) = central_planes(vol.depth, m.reverse)?;
            // predict skips a tile whose 16 planes are all zero inside the volume
            let planes: Vec<usize> = (0..D as i32).map(|z| (first + step * z) as usize).collect();
            let keep: Vec<bool> = origins
                .par_iter()
                .map(|&(y, x)| {
                    let (ph, pw) = (P.min(h - y), P.min(w - x));
                    planes
                        .iter()
                        .any(|&z| (y..y + ph).any(|yy| vol.data[(z * h + yy) * w + x..][..pw].iter().any(|&v| v != 0)))
                })
                .collect();
            let mut grid = vec![-1i32; origins.len()];
            for (i, (&(y, x), &k)) in origins.iter().zip(&keep).enumerate() {
                if k {
                    grid[i] = i32::try_from(slots).map_err(|_| eyre!("more than 2^31 tiles"))?;
                    desc.extend_from_slice(&[y as i32, x as i32, first, step, slots as i32, 0, 0, 0]);
                    slots += 1;
                }
            }
            kept.push(keep.iter().filter(|&&k| k).count());
            grids.push(grid);
        }
        // pad the last batch with copies of a real tile that write no probabilities
        let batches = slots.div_ceil(batch);
        let pad =
            if slots == 0 { [0, 0, 0, 1, -1, 0, 0, 0] } else { [desc[0], desc[1], desc[2], desc[3], -1, 0, 0, 0] };
        while desc.len() < batches * batch * DESC {
            desc.extend_from_slice(&pad);
        }
        Ok(Self {
            ys: ys.iter().map(|&v| v as u32).collect(),
            xs: xs.iter().map(|&v| v as u32).collect(),
            grids,
            kept,
            desc,
            slots,
            batches,
        })
    }
}

/// For each pixel along an axis, the half-open range of tile indices whose 64 pixels cover it
pub(crate) fn spans(starts: &[u32], length: usize) -> Vec<[u32; 2]> {
    (0..length as u32)
        .map(|p| {
            let first = starts.iter().position(|&s| s + P as u32 > p).unwrap_or(starts.len());
            let end = starts.iter().rposition(|&s| s <= p).map_or(0, |i| i + 1);
            [first as u32, end.max(first) as u32]
        })
        .collect()
}

fn timing_event() -> Result<cu::CUevent> {
    Ok(cur::event::create(cu::CUevent_flags::CU_EVENT_DEFAULT)?)
}

/// An 8-bit grayscale PNG with `hecate.py`'s Sub filter on every row and the best zlib
/// compression, written to `.partial` and renamed. The pixels match `hecate.py`'s file; the
/// compressed bytes may not, since the deflate encoder differs
fn write_png(path: &Path, image: &[u8], height: usize, width: usize) -> Result<()> {
    ensure!(image.len() == height * width, "image of {} bytes for {height}x{width}", image.len());
    let partial = path.with_extension("partial");
    let file = std::fs::File::create(&partial).wrap_err_with(|| format!("create {}", partial.display()))?;
    let mut enc = png::Encoder::new(BufWriter::new(file), width as u32, height as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::Best);
    enc.set_filter(png::FilterType::Sub);
    enc.set_adaptive_filter(png::AdaptiveFilterType::NonAdaptive);
    let mut writer = enc.write_header()?;
    writer.write_image_data(image)?;
    writer.finish()?;
    std::fs::rename(&partial, path).wrap_err_with(|| format!("rename {} to {}", partial.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hecate::check_volume;

    #[test]
    fn spans_cover_each_pixel_with_its_tiles() {
        let starts: Vec<u32> = axis_starts(150, 32).iter().map(|&v| v as u32).collect();
        assert_eq!(starts, [0, 32, 64, 86]);
        let s = spans(&starts, 150);
        assert_eq!(s[0], [0, 1]);
        assert_eq!(s[63], [0, 2]);
        assert_eq!(s[64], [1, 3]);
        assert_eq!(s[149], [3, 4]);
        for (p, &[a, b]) in s.iter().enumerate() {
            for (i, &st) in starts.iter().enumerate() {
                let covers = st as usize <= p && p < st as usize + P;
                assert_eq!(covers, (a as usize..b as usize).contains(&i), "pixel {p} tile {i}");
            }
        }
    }

    #[test]
    fn extents_past_u32_are_refused() {
        let slots = |n: usize| u32_extent("tile probabilities", n.saturating_mul(PIX));
        // the r96 gate input and the largest volume and probability spaces the kernels can index
        check_volume(28, 2682, 2916).unwrap();
        check_volume(1, 1, u32::MAX as usize).unwrap();
        slots(7_700).unwrap();
        slots(u32::MAX as usize / PIX).unwrap();
        // one past, and the review's cases: plane 64 of 128 x 8192 x 8192 starts at byte 2^32; a
        // fully occupied 1200 x 1200 map at stride 1 has 1,292,769 tiles
        assert!(check_volume(1, 1, u32::MAX as usize + 1).is_err());
        assert!(check_volume(128, 8192, 8192).is_err());
        assert!(slots(1_292_769).is_err());
        assert!(slots((1 << 32) / PIX).is_err());
        // an activation of 8 x 64^3 x 2048 is exactly 2^32 elements
        assert!(u32_extent("activation", 8 * 64 * 64 * 64 * 2048).is_err());
    }
}
