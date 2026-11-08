//! `hecate.py --output` on Metal, with the CUDA engine's API (`src/hecate/engine.rs`): tile the central 16 planes at
//! `stride`, skip the tiles whose planes are all zero, run the network in fixed batches with no host synchronisation
//! between them, blend each map as `predict` does and write its PNG and JSON
//!
//! The schedule, the blend spans and the PNG writer repeat the CUDA engine's, which lives in a CUDA-only module; the
//! tests below pin them to the same cases

use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use color_eyre::eyre::{Result, WrapErr, ensure, eyre};
use rayon::prelude::*;
use serde::Serialize;

use crate::hecate::{D, P, PIX, Precision, SAMPLING_UM, axis_starts, central_planes, read_volume, u32_extent, window};
use crate::metal::dnn::Pick;
use crate::metal::engine::resolve_weights;
use crate::metal::gpu::Gpu;
use crate::metal::hecate::model::{Model, check_contract};
use crate::sched::DESC;
use crate::weights::Checkpoint;
use crate::zarr::Volume;

/// One output map: the direction and the PNG path; its JSON goes next to it
#[derive(Clone, Debug)]
pub struct MapSpec {
    pub reverse: bool,
    pub png: PathBuf,
}

/// Run settings, the CUDA engine's; Metal ignores `pick` and `max_workspace`, which choose cuDNN algorithms
pub struct Options {
    pub stride: usize,
    /// Whether the settings came from the `--fast` preset ([`crate::fast`]); recorded in the report only
    pub fast: bool,
    pub batch: usize,
    pub precision: Precision,
    pub pick: Pick,
    pub max_workspace: usize,
    /// The render's voxel size as given, checked against the model's and written to the JSON
    pub spacing_um: f64,
    /// The `--checkpoint` argument, whose file name the JSON records as `hecate.py` does
    pub checkpoint: PathBuf,
    /// Where a `.pth` checkpoint's export is looked up
    pub weights_dir: PathBuf,
    /// The GPU; Apple silicon has only device 0
    pub device: usize,
    pub started: Instant,
}

/// Timings of a run
#[derive(Debug, Default, Serialize)]
pub struct Report {
    pub precision: String,
    /// Whether the run used the `--fast` preset, which gives up map parity with `hecate.py`
    pub fast: bool,
    pub stride: usize,
    pub batch: usize,
    pub device: String,
    pub load_weights_s: f64,
    pub open_gpu_s: f64,
    pub plan_s: f64,
    pub read_volume_s: f64,
    pub schedule_s: f64,
    pub upload_s: f64,
    /// Queueing every batch and blend through the last map's download
    pub run_s: f64,
    /// GPU time from the first batch's command buffer to the last blend's
    pub gpu_s: f64,
    /// Reading every tile probability and refusing a non-finite one
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
    /// The CUDA report's cuDNN fields, so one CLI log line serves both backends: Metal plans no cuDNN algorithms,
    /// so these stay 0 and empty
    pub algo_cache_hits: usize,
    pub algos: Vec<String>,
    pub schema: (usize, usize, usize),
}

/// Refuses, before the GPU or any file is touched, every request the engine would refuse: the CUDA engine's
/// checks, so the Python module raises the same errors on either backend
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

    // the kernel library compiles on Gpu::open's thread while the weights load
    let t = Instant::now();
    let gpu = Gpu::open(opts.device)?;
    rep.device = gpu.name();
    rep.open_gpu_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let (ck, sidecar) = Checkpoint::read(&resolve_weights(ckpt, &opts.weights_dir)?)?;
    check_contract(&sidecar)?;
    rep.load_weights_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let model = Model::new(&gpu, &ck, opts.batch, opts.precision)?;
    rep.schema = ck.finish()?;
    drop(ck);
    gpu.ready()?;
    rep.plan_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let vol = read_volume(input)?;
    let (height, width) = (vol.height, vol.width);
    rep.shape = [vol.depth, height, width];
    rep.read_volume_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let sched = Schedule::new(&vol, maps, opts.stride, opts.batch)?;
    u32_extent("tile probabilities", sched.slots.saturating_mul(PIX))?;
    u32_extent("map", height.saturating_mul(width))?;
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
    rep.upload_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let mut first = None;
    for i in 0..sched.batches {
        model.forward(&gpu, (&vbuf, height, width), (&desc, i * opts.batch * DESC), &prob)?;
        if i == 0 {
            first = gpu.flush()?;
        }
    }
    for (m, out) in outs.iter().enumerate() {
        gpu.dispatch_1d(
            "hec_blend",
            height * width,
            &[
                out.arg(),
                prob.arg(),
                grids[m].arg(),
                ys.arg(),
                xs.arg(),
                y_span.arg(),
                x_span.arg(),
                weight.arg(),
                width.into(),
                sched.xs.len().into(),
                (height * width).into(),
            ],
        )?;
    }
    let last = gpu.flush()?;
    if let Some(f) = &last {
        f.wait()?;
    }
    gpu.synchronize()?;
    let mut images = Vec::with_capacity(outs.len());
    for o in &outs {
        let mut image = vec![0u8; height * width];
        gpu.download(o, 0, &mut image)?;
        images.push(image);
    }
    rep.run_s = t.elapsed().as_secs_f64();
    if let (Some(a), Some(b)) = (first.as_ref(), last.as_ref()) {
        rep.gpu_s = (b.gpu_span().1 - a.gpu_span().0).max(0.0);
    }
    rep.tiles_per_s = sched.slots as f64 / rep.gpu_s.max(1e-9);

    // always, as hecate.py raises on a non-finite probability before writing: the blend's clamp and uint8 cast would
    // otherwise turn a NaN into a plausible pixel
    let t = Instant::now();
    let mut all = vec![0f32; sched.slots * PIX];
    gpu.download(&prob, 0, &mut all)?;
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

    gpu.print_profile()?;
    rep.total_s = opts.started.elapsed().as_secs_f64();
    Ok(rep)
}

/// Every map's tile grid and the batched descriptors of its kept tiles
struct Schedule {
    ys: Vec<u32>,
    xs: Vec<u32>,
    /// Per map, `[ny, nx]` slots, -1 for a tile `predict` skips
    grids: Vec<Vec<i32>>,
    kept: Vec<usize>,
    desc: Vec<i32>,
    slots: usize,
    batches: usize,
}

impl Schedule {
    fn new(vol: &Volume, maps: &[MapSpec], stride: usize, batch: usize) -> Result<Self> {
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
fn spans(starts: &[u32], length: usize) -> Vec<[u32; 2]> {
    (0..length as u32)
        .map(|p| {
            let first = starts.iter().position(|&s| s + P as u32 > p).unwrap_or(starts.len());
            let end = starts.iter().rposition(|&s| s <= p).map_or(0, |i| i + 1);
            [first as u32, end.max(first) as u32]
        })
        .collect()
}

/// An 8-bit grayscale PNG with `hecate.py`'s Sub filter on every row and the best zlib compression, written to
/// `.partial` and renamed. The pixels match `hecate.py`'s file; the compressed bytes may not, since the deflate
/// encoder differs
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
}
