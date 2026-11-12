//! A v8-in map on Metal: read the 24-layer window, admit tiles by the [`Convention`], run the network in fixed
//! batches with no host synchronisation between them, stitch on the GPU and return the float32 map, behind the CUDA
//! engine's API (`v8in::engine`)
//!
//! Both reference runners add each kept tile's `probability * window` to the numerator in tile order (row-major on
//! the stride-21 grid), in float32; the blend kernel gathers per pixel in that order

use std::path::Path;
use std::time::Instant;

use color_eyre::eyre::{Result, ensure};

use crate::dnn::Pick;
use crate::hecate::{read_volume, u32_extent};
use crate::metal::gpu::{Buf, Gpu};
use crate::metal::v8in::model::{Descs, Model, StageHook, check_contract, force_unfused};
pub use crate::v8in::host::{
    Options, Output, P0Out, Report, coverage, stage_depths, window_layers, write_npy, write_npy_f32,
};
use crate::v8in::host::{Schedule, p0_desc};
use crate::v8in::{Convention, PIX, STRIDE, window};
use crate::weights::Checkpoint;
use crate::zarr::VolumeRef;

/// The resident model and GPU, for several maps in one process
pub struct Engine {
    gpu: Gpu,
    model: Model,
    reader: String,
    convention: Convention,
    report: Report,
}

impl Engine {
    /// Loads the export at `ckpt` and allocates the network at `batch`; Metal has one kernel per layer shape, so
    /// `pick` and `max_workspace` (cuDNN's) are ignored
    pub fn new(ckpt: &Path, batch: usize, _pick: Pick, _max_workspace: usize) -> Result<Self> {
        let mut report =
            Report { batch, cudnn: 0, ptx_tier: "metal".into(), kernels: "metal".into(), ..Default::default() };
        // the kernel library compiles on Gpu::open's thread while the weights load
        let t = Instant::now();
        let gpu = Gpu::open(0)?;
        report.compute_capability = gpu.name();
        report.open_gpu_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let (ck, sidecar) = Checkpoint::read(ckpt)?;
        let (reader, convention) = check_contract(&sidecar)?;
        report.load_weights_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let model = Model::new(&gpu, &ck, batch)?;
        report.schema = ck.finish()?;
        report.plans = model.plans();
        report.unfused = model.unfused.clone();
        gpu.ready()?;
        report.plan_s = t.elapsed().as_secs_f64();
        report.reader = reader.clone();
        Ok(Self { gpu, model, reader, convention, report })
    }

    pub fn reader(&self) -> &str {
        &self.reader
    }

    /// The export's default map convention
    pub fn convention(&self) -> Convention {
        self.convention
    }

    /// One map of the volume at `input` (zarr v2 or `.npy`, `[Z, Y, X]` uint8)
    pub fn run_path(&self, input: &Path, opts: &Options) -> Result<Output> {
        let t = Instant::now();
        let vol = read_volume(input)?;
        let read_s = t.elapsed().as_secs_f64();
        let mut out = self.run(vol.view(), opts)?;
        out.report.read_volume_s = read_s;
        Ok(out)
    }

    /// One map of `vol`
    pub fn run(&self, vol: VolumeRef<'_>, opts: &Options) -> Result<Output> {
        let started = Instant::now();
        ensure!(
            opts.batch == self.model.batch(),
            "batch {} but the model is planned at {}",
            opts.batch,
            self.model.batch()
        );
        let (height, width) = (vol.height, vol.width);
        let conv = opts.convention.unwrap_or(self.convention);
        let mut rep = Report {
            convention: conv.name().into(),
            reverse: opts.reverse,
            shape: [vol.depth, height, width],
            ..Report::default()
        };
        let t = Instant::now();
        let b = self.model.batch();
        let Schedule { layers, layer0, ys, xs, grid, desc, tiles, batches } = Schedule::new(vol, conv, opts, b)?;
        let slots = tiles.len();
        u32_extent("tile probabilities", slots.saturating_mul(PIX))?;
        u32_extent("map", height.saturating_mul(width))?;
        rep.layer0 = layer0;
        rep.positions = grid.len();
        rep.tiles = slots;
        rep.batches = batches;
        rep.schedule_s = t.elapsed().as_secs_f64();

        let gpu = &self.gpu;
        let t = Instant::now();
        let vbuf = gpu.upload(&layers)?;
        drop(layers);
        let descs = Descs::new(gpu, &desc, b, slots, (height, width))?;
        let gbuf = gpu.upload(&grid)?;
        let wbuf = gpu.upload(&window(conv.nsig())?)?;
        // NaN-filled: a kept slot never written fails the finite check and poisons the blend
        let prob = gpu.alloc_fill(slots.max(1) * PIX * 4, 0xFF)?;
        let map = gpu.alloc(height * width * 4)?;
        rep.upload_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut first = None;
        for i in 0..batches {
            self.model.forward(gpu, (&vbuf, height, width), (&descs, i), &prob, conv.sigmoid16(), None)?;
            if i == 0 {
                first = gpu.flush()?;
            }
        }
        gpu.dispatch_1d(
            "v8_blend",
            height * width,
            &[
                map.arg(),
                prob.arg(),
                gbuf.arg(),
                wbuf.arg(),
                ys.len().into(),
                xs.len().into(),
                STRIDE.into(),
                width.into(),
                conv.blend_mode().into(),
                (height * width).into(),
            ],
        )?;
        let last = gpu.flush()?;
        let mut image = vec![0f32; height * width];
        gpu.download(&map, 0, &mut image)?;
        rep.run_s = t.elapsed().as_secs_f64();
        if let (Some(a), Some(z)) = (first.as_ref(), last.as_ref()) {
            rep.gpu_s = (z.gpu_span().1 - a.gpu_span().0).max(0.0);
        }
        rep.tiles_per_s = slots as f64 / rep.gpu_s.max(1e-9);
        rep.ms_per_tile = rep.gpu_s * 1e3 / slots.max(1) as f64;

        // the clamp in the release's blend would turn a NaN probability into a plausible pixel
        let t = Instant::now();
        let mut all = vec![0f32; slots * PIX];
        gpu.download(&prob, 0, &mut all)?;
        let bad = all.iter().filter(|v| !v.is_finite()).count();
        ensure!(bad == 0, "{bad} non-finite probabilities among {slots} tiles; no map returned");
        rep.finite_check_s = t.elapsed().as_secs_f64();
        gpu.print_profile()?;
        rep.total_s = started.elapsed().as_secs_f64();
        let base = &self.report;
        rep.reader = base.reader.clone();
        rep.batch = base.batch;
        rep.cudnn = base.cudnn;
        rep.compute_capability = base.compute_capability.clone();
        rep.ptx_tier = base.ptx_tier.clone();
        rep.kernels = base.kernels.clone();
        rep.load_weights_s = base.load_weights_s;
        rep.open_gpu_s = base.open_gpu_s;
        rep.plan_s = base.plan_s;
        rep.plans = base.plans.clone();
        rep.unfused = base.unfused.clone();
        rep.schema = base.schema;
        Ok(Output { map: image, height, width, tiles, report: rep })
    }

    /// P0: the network on the tiles at `yx` (top-left corners) of `vol`, in batches of the model's (the window's
    /// layers, convention and direction as [`Options`] picks them), returning each tile's probability `[n, 64, 64]`,
    /// the decoder taps of every batch (name, channels-last shape, FP16 bits), and with `stages` the encoder stages
    /// too: `stats` (`[n, 2]` float32 scale and shift) and `stem`, `layer1`..`layer4` at the depth indices
    /// [`stage_depths`] picks, `[n, depths, h, w, c]`
    pub fn p0(&self, vol: VolumeRef<'_>, yx: &[[usize; 2]], opts: &Options, stages: bool) -> Result<P0Out> {
        let (layers, layer0) = window_layers(vol, opts.layer0, opts.reverse)?;
        let conv = opts.convention.unwrap_or(self.convention);
        let (height, width) = (vol.height, vol.width);
        let b = self.model.batch();
        let gpu = &self.gpu;
        let vbuf = gpu.upload(&layers)?;
        let n = yx.len();
        let prob = gpu.alloc_fill(n.max(1) * PIX * 4, 0xFF)?;
        let mut taps: Vec<(String, Vec<usize>, Vec<u16>)> = Vec::new();
        let mut stats: Vec<f32> = Vec::new();
        for (bi, (chunk, desc)) in yx.chunks(b).zip(p0_desc(yx, b, height, width)?).enumerate() {
            let descs = Descs::new(gpu, &desc, b, n, (height, width))?;
            let real = chunk.len();
            let mut stage_taps: Vec<(String, Vec<usize>, Vec<u16>)> = Vec::new();
            let mut hook = |name: &str, buf: &Buf, [_, d, h, w, c]: [usize; 5]| -> Result<()> {
                if name == "stats" {
                    let mut ab = vec![0f32; real * 2];
                    gpu.download(buf, 0, &mut ab)?;
                    stats.extend(ab);
                    return Ok(());
                }
                let zs = stage_depths(d);
                let per = h * w * c;
                let mut bits = vec![0u16; real * zs.len() * per];
                for t in 0..real {
                    for (k, &z) in zs.iter().enumerate() {
                        let at = (t * zs.len() + k) * per;
                        gpu.download(buf, (t * d + z) * per, &mut bits[at..at + per])?;
                    }
                }
                stage_taps.push((name.to_string(), vec![real, zs.len(), h, w, c], bits));
                Ok(())
            };
            let hk: Option<StageHook<'_>> = if stages { Some(&mut hook) } else { None };
            self.model.forward(gpu, (&vbuf, height, width), (&descs, 0), &prob, conv.sigmoid16(), hk)?;
            let mut all = Vec::new();
            for tap in self.model.taps() {
                let [_, h, w, c] = tap.shape[..] else { unreachable!("decoder taps are [b, h, w, c]") };
                let mut bits = vec![0u16; real * h * w * c];
                if tap.padded {
                    // the interior of each padded row of the zero-padded [b, h + 2, w + 2, c] layout
                    let mut row = vec![0u16; w * c];
                    for t in 0..real {
                        for y in 0..h {
                            gpu.download(tap.buf, ((t * (h + 2) + y + 1) * (w + 2) + 1) * c, &mut row)?;
                            let at = (t * h + y) * w * c;
                            bits[at..at + w * c].copy_from_slice(&row);
                        }
                    }
                } else {
                    gpu.download(tap.buf, 0, &mut bits)?;
                }
                all.push((tap.name, vec![real, h, w, c], bits));
            }
            all.extend(stage_taps);
            if bi == 0 {
                taps = all;
            } else {
                ensure!(all.len() == taps.len(), "batch {bi}: {} taps, batch 0 had {}", all.len(), taps.len());
                for (acc, (name, shape, bits)) in taps.iter_mut().zip(all) {
                    ensure!(acc.0 == name, "batch {bi}: tap {name} where batch 0 had {}", acc.0);
                    acc.1[0] += shape[0];
                    acc.2.extend_from_slice(&bits);
                }
            }
        }
        let mut probs = vec![0f32; n * PIX];
        gpu.download(&prob, 0, &mut probs)?;
        let meta = serde_json::json!({
            "reader": self.reader, "convention": conv.name(), "layer0": layer0, "reverse": opts.reverse,
            "tiles": yx, "stage_depths": ([96, 96, 48, 24, 12].map(stage_depths)), "unfused": self.model.unfused,
            "force_unfused": force_unfused(), "backend": "metal",
        });
        Ok(P0Out { probs, taps, stats, meta })
    }
}
