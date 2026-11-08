//! A v8-in map on the GPU: read the 24-layer window, admit tiles by the [`Convention`], run the network in
//! fixed batches with no host synchronisation between them, stitch on the GPU and return the float32 map
//!
//! Both reference runners add each kept tile's `probability * window` to the numerator in tile order
//! (row-major on the stride-21 grid), in float32; the blend kernel gathers per pixel in that order

use std::path::Path;
use std::time::Instant;

use color_eyre::eyre::{Result, ensure};
use cudarc::driver::result as cur;
use cudarc::driver::sys as cu;

use crate::dnn::{Dnn, Pick};
use crate::gpu::{Arg, Gpu};
use crate::hecate::read_volume;
use crate::model::Ctx;
use crate::sched::DESC;
pub use crate::v8in::host::{
    Options, Output, P0Out, Report, coverage, stage_depths, window_layers, write_npy, write_npy_f32,
};
use crate::v8in::host::{Schedule, p0_desc};
use crate::v8in::model::{Model, check_contract};
use crate::v8in::{Convention, PIX, window};
use crate::weights::Checkpoint;
use crate::zarr::VolumeRef;

/// The resident model and GPU, for several maps in one process
pub struct Engine {
    gpu: Gpu,
    dnn: Dnn,
    model: Model,
    pick: Pick,
    max_workspace: usize,
    reader: String,
    convention: Convention,
    report: Report,
}

impl Engine {
    /// Loads the export at `ckpt` and plans the network at `batch`
    pub fn new(ckpt: &Path, batch: usize, pick: Pick, max_workspace: usize) -> Result<Self> {
        let mut report = Report { batch, ..Default::default() };
        let t = Instant::now();
        let (ck, sidecar) = Checkpoint::read(ckpt)?;
        let (reader, convention) = check_contract(&sidecar)?;
        report.load_weights_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let gpu = Gpu::open()?;
        let dnn = Dnn::new(&gpu)?;
        report.open_gpu_s = t.elapsed().as_secs_f64();
        report.cudnn = Dnn::version();
        report.compute_capability = gpu.inner.compute_capability().to_string();
        report.ptx_tier = gpu.inner.ptx_tier().to_string();

        let t = Instant::now();
        let model = {
            let ctx = Ctx { gpu: &gpu, dnn: &dnn, pick, max_workspace };
            Model::new(&ctx, &ck, batch)?
        };
        report.schema = ck.finish()?;
        report.plans = model.plans();
        report.unfused = model.unfused.clone();
        gpu.sync()?;
        report.plan_s = t.elapsed().as_secs_f64();
        report.reader = reader.clone();
        Ok(Self { gpu, dnn, model, pick, max_workspace, reader, convention, report })
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
            opts.batch == self.model.batch,
            "batch {} but the model is planned at {}",
            opts.batch,
            self.model.batch
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
        let b = self.model.batch;
        let Schedule { layers, layer0, ys, xs, grid, desc, tiles, batches } = Schedule::new(vol, conv, opts, b)?;
        let slots = tiles.len();
        rep.layer0 = layer0;
        rep.positions = grid.len();
        rep.tiles = slots;
        rep.batches = batches;
        rep.schedule_s = t.elapsed().as_secs_f64();

        let gpu = &self.gpu;
        let t = Instant::now();
        let vbuf = gpu.upload(&layers)?;
        drop(layers);
        let dbuf = gpu.upload(&desc)?;
        let gbuf = gpu.upload(&grid)?;
        let wbuf = gpu.upload(&window(conv.nsig())?)?;
        // NaN-filled: a kept slot never written fails the finite check and poisons the blend
        let prob = gpu.alloc_fill(slots.max(1) * PIX * 4, 0xFF)?;
        let map = gpu.alloc(height * width * 4)?;
        gpu.sync()?;
        rep.upload_s = t.elapsed().as_secs_f64();

        let ctx = Ctx { gpu, dnn: &self.dnn, pick: self.pick, max_workspace: self.max_workspace };
        let t = Instant::now();
        let (start_ev, end_ev) = (timing_event()?, timing_event()?);
        // SAFETY: our stream; the events outlive the records
        unsafe { cur::event::record(start_ev, gpu.cu_stream())? };
        for i in 0..batches {
            let view = dbuf.view::<i32>(i * b * DESC, b * DESC);
            self.model.forward(&ctx, (&vbuf, height, width), view, &prob, conv.sigmoid16(), None)?;
        }
        gpu.launch_1d(
            "v8_blend",
            height * width,
            &[
                map.arg::<f32>(),
                prob.arg::<f32>(),
                gbuf.arg::<i32>(),
                wbuf.arg::<f32>(),
                Arg::U32(ys.len() as u32),
                Arg::U32(xs.len() as u32),
                Arg::U32(crate::v8in::STRIDE as u32),
                Arg::U32(width as u32),
                Arg::U32(conv.blend_mode()),
            ],
        )?;
        // SAFETY: as above
        unsafe { cur::event::record(end_ev, gpu.cu_stream())? };
        let image = gpu.download::<f32>(&map)?;
        gpu.sync()?;
        rep.run_s = t.elapsed().as_secs_f64();
        // SAFETY: both events recorded and complete after the sync
        rep.gpu_s = unsafe { cur::event::elapsed(start_ev, end_ev)? } as f64 / 1e3;
        // SAFETY: done with them
        unsafe {
            cur::event::destroy(start_ev)?;
            cur::event::destroy(end_ev)?;
        }
        rep.tiles_per_s = slots as f64 / rep.gpu_s.max(1e-9);
        rep.ms_per_tile = rep.gpu_s * 1e3 / slots.max(1) as f64;

        // the clamp in the release's blend would turn a NaN probability into a plausible pixel
        let t = Instant::now();
        let all: Vec<f32> = gpu.download_range(&prob, 0, slots * PIX)?;
        let bad = all.iter().filter(|v| !v.is_finite()).count();
        ensure!(bad == 0, "{bad} non-finite probabilities among {slots} tiles; no map returned");
        rep.finite_check_s = t.elapsed().as_secs_f64();
        rep.total_s = started.elapsed().as_secs_f64();
        let base = &self.report;
        rep.reader = base.reader.clone();
        rep.batch = base.batch;
        rep.cudnn = base.cudnn;
        rep.compute_capability = base.compute_capability.clone();
        rep.ptx_tier = base.ptx_tier.clone();
        rep.load_weights_s = base.load_weights_s;
        rep.open_gpu_s = base.open_gpu_s;
        rep.plan_s = base.plan_s;
        rep.plans = base.plans.clone();
        rep.unfused = base.unfused.clone();
        rep.schema = base.schema;
        Ok(Output { map: image, height, width, tiles, report: rep })
    }

    /// P0: the network on the tiles at `yx` (top-left corners) of `vol`, in batches of the model's (the
    /// window's layers, convention and direction as [`Options`] picks them), returning each tile's
    /// probability `[n, 64, 64]`, the decoder taps of every batch (name, channels-last shape, FP16 bits),
    /// and with `stages` the encoder stages too: `stats` (`[n, 2]` float32 scale and shift) and `stem`,
    /// `layer1`..`layer4` at the depth indices [`stage_depths`] picks, `[n, depths, h, w, c]`
    pub fn p0(&self, vol: VolumeRef<'_>, yx: &[[usize; 2]], opts: &Options, stages: bool) -> Result<P0Out> {
        let (layers, layer0) = window_layers(vol, opts.layer0, opts.reverse)?;
        let conv = opts.convention.unwrap_or(self.convention);
        let (height, width) = (vol.height, vol.width);
        let b = self.model.batch;
        let gpu = &self.gpu;
        let vbuf = gpu.upload(&layers)?;
        let n = yx.len();
        let prob = gpu.alloc_fill(n.max(1) * PIX * 4, 0xFF)?;
        let ctx = Ctx { gpu, dnn: &self.dnn, pick: self.pick, max_workspace: self.max_workspace };
        let mut taps: Vec<(String, Vec<usize>, Vec<u16>)> = Vec::new();
        let mut stats: Vec<f32> = Vec::new();
        for (bi, (chunk, desc)) in yx.chunks(b).zip(p0_desc(yx, b, height, width)?).enumerate() {
            let dbuf = gpu.upload(&desc)?;
            let real = chunk.len();
            let mut stage_taps: Vec<(String, Vec<usize>, Vec<u16>)> = Vec::new();
            let mut hook = |name: &str, buf: &crate::gpu::Buf, [_, d, h, w, c]: [usize; 5]| -> Result<()> {
                if name == "stats" {
                    stats.extend(gpu.download_range::<[f32; 2]>(buf, 0, real)?.into_iter().flatten());
                    return Ok(());
                }
                let zs = stage_depths(d);
                let per = h * w * c;
                let mut bits = Vec::with_capacity(real * zs.len() * per);
                for t in 0..real {
                    for &z in &zs {
                        bits.extend(gpu.download_range::<u16>(buf, (t * d + z) * per, per)?);
                    }
                }
                stage_taps.push((name.to_string(), vec![real, zs.len(), h, w, c], bits));
                Ok(())
            };
            let hk: Option<crate::v8in::model::StageHook<'_>> = if stages { Some(&mut hook) } else { None };
            self.model.forward(&ctx, (&vbuf, height, width), dbuf.arg::<i32>(), &prob, conv.sigmoid16(), hk)?;
            let decoder = self.model.taps().into_iter().map(|tap| -> Result<(String, Vec<usize>, Vec<u16>)> {
                let per: usize = tap.shape[1..].iter().product();
                let mut shape = tap.shape.clone();
                shape[0] = real;
                Ok((tap.name, shape, gpu.download_range(tap.buf, 0, real * per)?))
            });
            let all: Vec<_> = decoder.collect::<Result<Vec<_>>>()?.into_iter().chain(stage_taps).collect();
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
        let probs = gpu.download_range::<f32>(&prob, 0, n * PIX)?;
        gpu.sync()?;
        let meta = serde_json::json!({
            "reader": self.reader, "convention": conv.name(), "layer0": layer0, "reverse": opts.reverse,
            "tiles": yx, "stage_depths": ([96, 96, 48, 24, 12].map(stage_depths)), "unfused": self.model.unfused,
            "force_unfused": crate::v8in::model::force_unfused(),
        });
        Ok(P0Out { probs, taps, stats, meta })
    }
}

fn timing_event() -> Result<cu::CUevent> {
    Ok(cur::event::create(cu::CUevent_flags::CU_EVENT_DEFAULT)?)
}
