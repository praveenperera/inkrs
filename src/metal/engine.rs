//! The resident pipeline on Metal, with the CUDA engine's public API (`src/engine.rs`): the volume and model stay in
//! GPU memory, every map's tiles run in fixed-size batches with no host wait between them, each map is blended and
//! finished on the GPU as soon as its last tile is done, and writer threads encode the TIFFs while later maps compute
//!
//! Memory is unified, so a finished map is read straight out of its buffer once the command buffer that wrote it has
//! completed; there is no pinned staging copy

use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::coreml::{Job, Lane, Offload, Progress, Ran, Unit, work};
use super::dnn::Pick;
use super::gpu::{Arg, Buf, Fence, Gpu};
use super::model::{Model, logits_buf};
use crate::ensure_supported;
use crate::fast::ReaderPrecision;
use crate::sched::{DESC, Edge, Occupancy, Schedule, hann_weights};
use crate::tta::Tta;
use crate::weights::Checkpoint;
use crate::zarr::{LowRes, MAX_VOLUME_BYTES, Volume, VolumeRef};

/// One output map of the CLI: its layers and where each pass writes it
#[derive(Clone, Debug)]
pub struct MapSpec {
    pub first: i32,
    pub step: i32,
    /// One output path per pass
    pub paths: Vec<PathBuf>,
    /// With `--coverage`, one path per pass for the engine's coverage mask; empty otherwise
    pub coverage: Vec<PathBuf>,
}

/// Engine options shared by `infer` and `sweep`
#[derive(Clone, Debug)]
pub struct Options {
    pub stride: usize,
    pub batch: usize,
    /// The reader's numerics: fp16 (villa's CUDA autocast) or fp32 (villa on MPS)
    pub precision: ReaderPrecision,
    /// Whether the settings came from the `--fast` preset ([`crate::fast`]) rather than parity mode;
    /// recorded in the report only
    pub fast: bool,
    /// cuDNN's algorithm choice; Metal has one implementation per layer and ignores it
    pub pick: Pick,
    /// cuDNN's workspace cap; ignored on Metal
    pub max_workspace: usize,
    pub check_finite: bool,
    pub coverage: bool,
    pub writers: usize,
    /// Test-time augmentation; Metal has no TTA kernels, so only off runs ([`TTA_CUDA_ONLY`])
    pub tta: Tta,
    pub started: Instant,
}

/// Why the Metal backend refuses `--tta 8`: the recipe's mirror4 reads and their reduction are CUDA kernels only
pub const TTA_CUDA_ONLY: &str = "--tta 8 runs on CUDA only: the Metal backend has no TTA kernels";

/// The CUDA engine's 8-fold TTA result, so the Python binding has one shape on both backends; Metal never makes one
pub struct TtaOutput {
    pub report: PassReport,
    pub maps: Vec<Option<Vec<f32>>>,
}

/// Timings and counts of one run, written next to the outputs; the CUDA fields that have no Metal meaning (cuDNN
/// version, compute capability, PTX tier, algorithms) hold the Metal equivalents or stay empty
#[derive(Debug, Default, Serialize)]
pub struct Report {
    pub stride: usize,
    pub batch: usize,
    /// The reader's numerics, `fp16` or `fp32` ([`ReaderPrecision`])
    pub precision: String,
    /// Whether the run used the `--fast` preset, which gives up map parity with villa
    pub fast: bool,
    pub cudnn: usize,
    /// The GPU's name, such as "Apple M4 Pro"
    pub compute_capability: String,
    /// `metal`
    pub ptx_tier: String,
    /// How the kernels were loaded: a cubin compiled ahead (`sm_120 SASS`) or the PTX through the
    /// driver's JIT (`PTX, JIT-compiled`); `metal` on Metal
    pub kernels: String,
    /// The library that ran the reader's 2D convs: always [`CONV`]; the CUDA engine reports `inkrs` or `cudnn`
    pub conv: &'static str,
    pub conv_reason: String,
    pub load_weights_s: f64,
    pub open_gpu_s: f64,
    /// Weight upload and the wait for the kernel library build
    pub plan_s: f64,
    /// Always 0: Metal has no algorithm search or cache; kept for the CUDA engine's report shape
    pub algo_cache_hits: usize,
    /// Always None (JSON null): Metal has no algorithm cache; kept for the CUDA engine's report shape
    pub algo_cache: Option<()>,
    pub cold_s: f64,
    pub passes: Vec<PassReport>,
    pub algos: Vec<(String, String)>,
    pub schema: (usize, usize, usize),
}

/// One pass over the volume with the model resident
#[derive(Debug, Default, Serialize)]
pub struct PassReport {
    pub volume: String,
    pub shape: [usize; 3],
    pub maps: usize,
    pub tiles: usize,
    pub batches: usize,
    pub padded_tiles: usize,
    pub read_volume_s: f64,
    pub schedule_s: f64,
    pub upload_s: f64,
    pub run_s: f64,
    pub total_s: f64,
    /// GPU time from the first batch's command buffer start to the last one's end
    pub gpu_s: f64,
    pub tiles_per_s: f64,
    pub map_tiles: Vec<(String, usize)>,
    pub occupancy: Option<OccupancyReport>,
    pub scheduled_positions: usize,
}

/// The pyramid level villa's first empty-tile skip read
#[derive(Debug, Default, Serialize)]
pub struct OccupancyReport {
    pub level: String,
    pub shape: [usize; 2],
    pub scale: [usize; 2],
    pub nonempty: usize,
}

/// Where one output image goes
#[derive(Clone, Debug)]
pub enum Dest {
    File(PathBuf),
    Memory,
}

/// One map of a pass: its layers and where its map and, optionally, its coverage mask go
#[derive(Clone, Debug)]
pub struct Target {
    pub first: i32,
    pub step: i32,
    pub map: Dest,
    pub coverage: Option<Dest>,
}

/// The input of a pass
pub enum Source<'a> {
    Zarr { path: &'a Path, scan: Scan<'a> },
    Array { volume: VolumeRef<'a>, scan: Option<&'a LowRes>, label: String },
}

/// villa's first empty-tile skip, its low-resolution occupancy scan, for a zarr input
#[derive(Clone, Copy)]
pub enum Scan<'a> {
    Villa,
    Given(Option<&'a LowRes>),
}

/// What a pass returns: its report, its logits and the in-memory images
pub struct PassOutput {
    pub report: PassReport,
    pub logits: Buf,
    pub maps: Vec<Option<Vec<u8>>>,
    pub coverage: Vec<Option<Vec<u8>>>,
}

/// Per-pass settings
#[derive(Clone, Copy, Debug)]
pub struct PassOptions {
    pub stride: usize,
    pub writers: usize,
}

/// How the model is planned
#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub weights_dir: PathBuf,
    pub device: usize,
    pub batch: usize,
    /// The reader's numerics; the caller's choice, never read from the checkpoint ([`ReaderPrecision`])
    pub precision: ReaderPrecision,
    pub pick: Pick,
    pub max_workspace: usize,
}

/// The conv backend name the Metal engine reports, in the field where the CUDA engine names `inkrs` or `cudnn`
pub const CONV: &str = "metal";

/// The variable that picks the CUDA engine's conv backend (`ConvBackend::ENV`, which is not built on macOS)
const CONV_ENV: &str = "INKRS_CONV";

/// Refuses an explicit conv backend request: both backends `INKRS_CONV` names (`inkrs`, `cudnn`) are CUDA-only,
/// and running Metal instead would let a caller believe the backend it asked for ran
pub fn check_conv_request(requested: Option<&std::ffi::OsStr>) -> Result<()> {
    ensure_supported!(
        requested.is_none(),
        "{CONV_ENV}={:?} asks for a CUDA conv backend; the Metal backend has only its own convs, so unset it",
        requested.unwrap_or_default()
    );
    Ok(())
}

/// What loading the engine cost
#[derive(Debug, Default)]
pub struct LoadReport {
    pub reader: Option<String>,
    pub cudnn: usize,
    pub compute_capability: String,
    pub ptx_tier: String,
    /// How the kernels were loaded: a cubin compiled ahead (`sm_120 SASS`) or the PTX through the
    /// driver's JIT (`PTX, JIT-compiled`); `metal` on Metal
    pub kernels: String,
    /// Always [`CONV`], as in [`Report::conv`]
    pub conv: &'static str,
    pub conv_reason: String,
    pub load_weights_s: f64,
    pub open_gpu_s: f64,
    pub plan_s: f64,
    /// Always 0: Metal has no conv plan cache, the field mirrors the CUDA report
    pub algo_cache_hits: usize,
    /// Always None (JSON null): Metal has no algorithm cache, the field mirrors the CUDA report
    pub algo_cache: Option<()>,
    pub algos: Vec<(String, String)>,
    pub schema: (usize, usize, usize),
}

/// A finished map handed to a writer: the fence of the command buffer that finished it, the target, and the output
/// and coverage buffer indices
struct WriteJob {
    fence: Fence,
    index: usize,
    map: Dest,
    coverage: Option<Dest>,
}

/// An image the writers kept in memory: target index, whether it is the coverage mask, bytes
type Kept = (usize, bool, Vec<u8>);

/// The model resident on the GPU, planned at one batch size, reusable across volumes
pub struct Engine {
    model: Model,
    gpu: Gpu,
    offload: Option<Offload>,
    pub load: LoadReport,
}

impl Engine {
    /// Loads the safetensors export `ckpt` (or the export of a villa `.pth` in `opts.weights_dir`), opens the GPU
    /// and builds the model at `opts.batch`
    pub fn open(ckpt: &Path, opts: &EngineOptions) -> Result<Self> {
        ensure!(opts.batch > 0, "batch size must be positive");
        check_conv_request(std::env::var_os(CONV_ENV).as_deref())?;
        let mut load = LoadReport::default();
        let t = Instant::now();
        // the kernel library compiles on a background thread while the weights load
        let gpu = Gpu::open_at(opts.device, opts.precision)?;
        load.open_gpu_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let export = resolve_weights(ckpt, &opts.weights_dir)?;
        let ck = Checkpoint::load(&export, opts.precision)?;
        load.reader = export_reader(&export).ok();
        load.load_weights_s = t.elapsed().as_secs_f64();
        load.compute_capability = gpu.name();
        load.ptx_tier = "metal".into();
        load.kernels = "metal".into();
        load.conv = CONV;
        load.conv_reason = "the Metal backend's own kernels".into();

        let t = Instant::now();
        let model = Model::new(&gpu, &ck, opts.batch, ck.edge)?;
        load.schema = ck.finish()?;
        drop(ck);
        gpu.ready()?;
        let offload = Offload::from_env(&gpu, &model, || export_checkpoint_sha(&export))?;
        load.plan_s = t.elapsed().as_secs_f64();
        Ok(Self { model, gpu, offload, load })
    }

    /// (free, total) in bytes, as CUDA reports them. Metal has no device-wide free figure, so total is the
    /// recommended working set and free is that minus this process's Metal allocations
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        let (used, total) = self.gpu.memory();
        Ok((total.saturating_sub(used), total))
    }

    /// The tile edge of the loaded reader
    pub fn edge(&self) -> Edge {
        self.model.edge
    }

    /// The reader precision the engine runs at
    pub fn precision(&self) -> ReaderPrecision {
        self.gpu.precision()
    }

    /// The batch size the model was planned at
    pub fn batch(&self) -> usize {
        self.model.batch
    }

    /// The GPU layer, for tests and profiling
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// Downloads a pass's logits and fails on any non-finite value
    pub fn check_finite(&mut self, out: &PassOutput) -> Result<()> {
        let mut all = vec![0.0f32; out.report.tiles * self.model.edge.pixels()];
        self.gpu.download_real(&out.logits, 0, &mut all)?;
        let bad = all.iter().filter(|v| !v.is_finite()).count();
        ensure!(bad == 0, "{bad} non-finite logits among {} slots", out.report.tiles);
        Ok(())
    }

    /// One pass over `src`: every target's map, written to files or returned
    pub fn run_pass(&mut self, src: Source, targets: &[Target], opts: &PassOptions) -> Result<PassOutput> {
        ensure!(!targets.is_empty(), "no maps to run");
        ensure!(opts.writers > 0, "writer count must be positive");
        let patch = self.model.edge.px();
        ensure!((1..=patch).contains(&opts.stride), "stride must be in [1, {patch}]");
        run_pass(&self.gpu, &self.model, self.offload.as_ref(), src, targets, opts)
    }

    /// Refuses: 8-fold TTA runs on CUDA only ([`TTA_CUDA_ONLY`])
    pub fn run_tta(&mut self, _src: Source, _targets: &[Target], _stride: usize) -> Result<TtaOutput> {
        Err(crate::error::Unsupported(TTA_CUDA_ONLY.into()).into())
    }
}

/// Runs `maps` over the volume at `zarr` with the weights at `ckpt`, once per output path each map carries
pub fn run(zarr: &Path, ckpt: &Path, maps: &[MapSpec], opts: &Options) -> Result<Report> {
    ensure!(!maps.is_empty(), "no maps to run");
    ensure_supported!(opts.tta == Tta::Off, "{TTA_CUDA_ONLY}");
    ensure!(opts.batch > 0 && opts.writers > 0, "batch size and writer count must be positive");
    let passes = maps[0].paths.len();
    ensure!(passes > 0 && maps.iter().all(|m| m.paths.len() == passes), "every map needs one path per pass");
    ensure!(
        maps.iter().all(|m| m.coverage.len() == if opts.coverage { passes } else { 0 }),
        "every map needs one coverage path per pass with --coverage, and none without"
    );
    let mut engine = Engine::open(
        ckpt,
        &EngineOptions {
            weights_dir: weights_dir(),
            device: 0,
            batch: opts.batch,
            precision: opts.precision,
            pick: opts.pick,
            max_workspace: opts.max_workspace,
        },
    )?;
    let load = &engine.load;
    let mut rep = Report {
        stride: opts.stride,
        batch: opts.batch,
        precision: engine.precision().name().into(),
        fast: opts.fast,
        cudnn: load.cudnn,
        compute_capability: load.compute_capability.clone(),
        ptx_tier: load.ptx_tier.clone(),
        kernels: load.kernels.clone(),
        conv: load.conv,
        conv_reason: load.conv_reason.clone(),
        load_weights_s: load.load_weights_s,
        open_gpu_s: load.open_gpu_s,
        plan_s: load.plan_s,
        algo_cache_hits: load.algo_cache_hits,
        algo_cache: load.algo_cache,
        algos: load.algos.clone(),
        schema: load.schema,
        ..Default::default()
    };

    let pass_opts = PassOptions { stride: opts.stride, writers: opts.writers };
    for pass in 0..passes {
        let targets: Vec<Target> = maps
            .iter()
            .map(|m| Target {
                first: m.first,
                step: m.step,
                map: Dest::File(m.paths[pass].clone()),
                coverage: m.coverage.get(pass).cloned().map(Dest::File),
            })
            .collect();
        let out = engine.run_pass(Source::Zarr { path: zarr, scan: Scan::Villa }, &targets, &pass_opts)?;
        if pass == 0 {
            rep.cold_s = opts.started.elapsed().as_secs_f64();
        }
        if opts.check_finite {
            engine.check_finite(&out).wrap_err_with(|| format!("pass {}", pass + 1))?;
        }
        rep.passes.push(out.report);
    }
    engine.gpu.print_profile()?;
    Ok(rep)
}

/// One pass of `model` over `src`
fn run_pass(
    gpu: &Gpu,
    model: &Model,
    offload: Option<&Offload>,
    src: Source,
    targets: &[Target],
    opts: &PassOptions,
) -> Result<PassOutput> {
    let patch = model.edge.px();
    let mut rep = PassReport { maps: targets.len(), ..Default::default() };
    let t_pass = Instant::now();

    // the host volume lives only until it is in a GPU buffer
    let (vbuf, sched, height, width) = {
        let t = Instant::now();
        let read;
        let vol = match &src {
            Source::Zarr { path, .. } => {
                read = Volume::read(path)?;
                rep.volume = read.array.display().to_string();
                read.view()
            }
            Source::Array { volume, label, .. } => {
                rep.volume = label.clone();
                *volume
            }
        };
        rep.read_volume_s = t.elapsed().as_secs_f64();
        rep.shape = [vol.depth, vol.height, vol.width];
        ensure_supported!(
            vol.height >= patch && vol.width >= patch,
            "volume {}x{} is smaller than a tile; villa would pad it, this engine does not",
            vol.height,
            vol.width
        );
        ensure_supported!(
            vol.data.len() < MAX_VOLUME_BYTES,
            "volume of {} bytes is over 4 GiB; the kernels index it with u32",
            vol.data.len()
        );
        for m in targets {
            let last = m.first + m.step * 16;
            ensure!(
                m.first >= 0 && last >= 0 && (m.first.max(last) as usize) < vol.depth,
                "layers {}..{} outside the volume's {}",
                m.first,
                last,
                vol.depth
            );
        }

        let t = Instant::now();
        let scanned = match &src {
            Source::Zarr { path, scan: Scan::Villa } => LowRes::scan(path, vol.height, vol.width)?,
            _ => None,
        };
        let lowres = match &src {
            Source::Zarr { scan: Scan::Villa, .. } => scanned.as_ref(),
            Source::Zarr { scan: Scan::Given(given), .. } | Source::Array { scan: given, .. } => *given,
        };
        rep.occupancy = lowres.map(|m| OccupancyReport {
            level: m.level.clone(),
            shape: [m.height, m.width],
            scale: [m.scale_y, m.scale_x],
            nonempty: m.mask.iter().filter(|&&v| v).count(),
        });
        let occ = Occupancy::new(vol, opts.stride, lowres, patch);
        rep.scheduled_positions = occ.scheduled();
        let layers: Vec<(i32, i32)> = targets.iter().map(|m| (m.first, m.step)).collect();
        let sched = Schedule::new(&occ, vol.height, vol.width, &layers, model.batch, 1)?;
        rep.schedule_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let vbuf = gpu.upload(vol.data)?;
        rep.upload_s = t.elapsed().as_secs_f64();
        (vbuf, sched, vol.height, vol.width)
    };
    rep.tiles = sched.slots;
    rep.batches = sched.batches;
    rep.padded_tiles = sched.batches * model.batch - sched.slots;
    rep.map_tiles = targets
        .iter()
        .zip(&sched.maps)
        .enumerate()
        .map(|(i, (m, p))| {
            let name = match &m.map {
                Dest::File(path) => path.display().to_string(),
                Dest::Memory => format!("map {i}"),
            };
            (name, p.tiles)
        })
        .collect();

    let t = Instant::now();
    ensure!(sched.desc.len().is_multiple_of(DESC), "descriptor table not a multiple of {DESC}");
    let desc = gpu.upload(&sched.desc)?;
    let logits = logits_buf(gpu, sched.slots, model.edge)?;
    let weight = gpu.upload(&hann_weights(model.edge))?;
    let ys = gpu.upload(&sched.ys)?;
    let xs = gpu.upload(&sched.xs)?;
    let y_span = gpu.upload(&sched.y_span)?;
    let x_span = gpu.upload(&sched.x_span)?;
    let grids: Vec<Buf> = sched.maps.iter().map(|m| gpu.upload(&m.grid)).collect::<Result<_>>()?;
    let outs: Vec<Buf> = (0..targets.len()).map(|_| gpu.alloc(height * width)).collect::<Result<_>>()?;
    let covs: Vec<Option<Buf>> = targets
        .iter()
        .map(|m| m.coverage.as_ref().map(|_| gpu.alloc(height * width)).transpose())
        .collect::<Result<_>>()?;
    rep.upload_s += t.elapsed().as_secs_f64();

    let (tx, rx) = mpsc::channel::<WriteJob>();
    let rx = std::sync::Mutex::new(rx);
    let (kept_tx, kept_rx) = mpsc::channel::<Kept>();
    let t = Instant::now();
    let mut first_fence: Option<Fence> = None;
    let mut last_fence: Option<Fence> = None;
    let (outs_ref, covs_ref) = (&outs, &covs);
    // Core ML workers, when offloading: each runs the 2D body of batches whose stems the GPU wrote to the ring
    let pixels = model.edge.pixels();
    let progress = Progress::new(sched.batches);
    let (ane_tx, ane_rx) = mpsc::channel::<Job>();
    let (cpu_tx, cpu_rx) = mpsc::channel::<Job>();
    let (ane_rx, cpu_rx) = (std::sync::Mutex::new(ane_rx), std::sync::Mutex::new(cpu_rx));
    let (free_tx, free_rx) = mpsc::channel::<usize>();
    let written: Result<()> = std::thread::scope(|scope| {
        let writers: Vec<_> = (0..opts.writers)
            .map(|_| {
                let rx = &rx;
                let kept_tx = kept_tx.clone();
                scope.spawn(move || -> Result<()> {
                    let emit = |buf: &Buf, index: usize, coverage: bool, dest: &Dest| -> Result<()> {
                        // SAFETY: the job's fence covers the command buffer that wrote this map, and nothing
                        // queued later writes it
                        let image = &unsafe { buf.host::<u8>() }[..height * width];
                        match dest {
                            Dest::File(path) => crate::tiff::write(path, image, height, width)
                                .wrap_err_with(|| format!("write {}", path.display())),
                            Dest::Memory => {
                                kept_tx.send((index, coverage, image.to_vec())).ok();
                                Ok(())
                            }
                        }
                    };
                    loop {
                        let job = { rx.lock().map_err(|_| eyre!("writer queue poisoned"))?.recv() };
                        let Ok(job) = job else { return Ok(()) };
                        job.fence.wait()?;
                        emit(&outs_ref[job.index], job.index, false, &job.map)?;
                        if let (Some(dest), Some(buf)) = (&job.coverage, &covs_ref[job.index]) {
                            emit(buf, job.index, true, dest)?;
                        }
                    }
                })
            })
            .collect();
        drop(kept_tx);

        let ring = offload.map_or(&[][..], |o| o.ring());
        for slot in 0..ring.len() {
            free_tx.send(slot).ok();
        }
        let (logits_ref, sched_ref, progress_ref) = (&logits, &sched, &progress);
        // only the offload scatters, and it runs with fp16 logits only (`coreml::OFFLOAD_FP16_ONLY`)
        let scatter = move |bi: usize, t: usize, row: &[f16]| {
            let slot = sched_ref.desc[(bi * model.batch + t) * DESC + 4];
            if slot < 0 || slot as usize >= sched_ref.slots || row.len() != pixels {
                return;
            }
            // SAFETY: a tile's slot is its own: no GPU forward writes it (this batch's body runs here, not on the
            // GPU), no other tile writes it, and the blends that read it are encoded only after `progress` says
            // this batch is done
            unsafe {
                let dst = logits_ref.ptr().cast::<f16>().add(slot as usize * pixels);
                dst.copy_from_nonoverlapping(NonNull::from(row).cast(), pixels);
            }
        };
        let bodies = offload.map_or(&[][..], |o| o.bodies());
        let ran: Vec<_> = bodies
            .iter()
            .map(|body| {
                let job_rx = if body.unit == Unit::Ane { &ane_rx } else { &cpu_rx };
                let free_tx = free_tx.clone();
                scope.spawn(move || work(body, ring, job_rx, &free_tx, progress_ref, scatter))
            })
            .collect();
        drop(free_tx);

        // seconds the encoding thread spent waiting for a free ring slot: the workers set the pace while it waits
        let mut slot_wait_s = 0.0f64;
        let queued = (|| -> Result<()> {
            let mut next_map = 0;
            let finish = |m: usize| -> Result<()> {
                finish_map(
                    gpu,
                    &sched,
                    m,
                    &logits,
                    &grids,
                    &ys,
                    &xs,
                    &y_span,
                    &x_span,
                    &weight,
                    &outs[m],
                    covs[m].as_ref(),
                    (width, model.edge),
                )?;
                let fence = gpu.flush()?.ok_or_else(|| eyre!("map {m}: nothing committed"))?;
                ensure!(covs[m].is_some() == targets[m].coverage.is_some(), "map {m}: coverage buffers out of step");
                tx.send(WriteJob {
                    fence,
                    index: m,
                    map: targets[m].map.clone(),
                    coverage: targets[m].coverage.clone(),
                })
                .map_err(|_| eyre!("writers stopped"))
            };
            for bi in 0..sched.batches {
                let view: Arg = desc.at::<i32>(bi * model.batch * DESC);
                let vol = (&vbuf, height, width);
                match offload {
                    None => {
                        model.forward(gpu, vol, view, &logits)?;
                        progress.finish(bi, None);
                    }
                    Some(off) => {
                        let lane = off.lane(bi);
                        let queue = match lane {
                            Lane::Gpu => {
                                model.forward(gpu, vol, view, &logits)?;
                                progress.finish(bi, None);
                                None
                            }
                            Lane::Ane => Some(&ane_tx),
                            Lane::Cpu => Some(&cpu_tx),
                        };
                        if let Some(queue) = queue {
                            // the ring bounds how far the stems run ahead of the workers
                            let t = Instant::now();
                            let slot = loop {
                                progress.check()?;
                                match free_rx.recv_timeout(Duration::from_millis(5)) {
                                    Ok(slot) => break slot,
                                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                                    Err(mpsc::RecvTimeoutError::Disconnected) => bail!("every Core ML worker stopped"),
                                }
                            };
                            slot_wait_s += t.elapsed().as_secs_f64();
                            model.stem(gpu, vol, view, &ring[slot])?;
                            let fence = gpu.flush_newest()?.ok_or_else(|| eyre!("batch {bi}: nothing committed"))?;
                            queue
                                .send(Job { batch: bi, slot, fence })
                                .map_err(|_| eyre!("{lane:?} workers stopped"))?;
                        }
                    }
                }
                if bi == 0 {
                    first_fence = gpu.flush()?;
                }
                // a map's blend reads every earlier batch's logits, so it waits for the workers' writes
                while next_map < targets.len()
                    && sched.map_done_after[next_map] <= bi
                    && progress.ready(sched.map_done_after[next_map])?
                {
                    finish(next_map)?;
                    next_map += 1;
                }
            }
            while next_map < targets.len() {
                progress.wait(sched.map_done_after[next_map])?;
                finish(next_map)?;
                next_map += 1;
            }
            last_fence = gpu.flush()?;
            Ok(())
        })();
        drop(tx);
        drop((ane_tx, cpu_tx));
        let ran: Vec<Ran> = ran
            .into_iter()
            .map(|w| w.join().map_err(|_| eyre!("a Core ML worker thread panicked")))
            .collect::<Result<_>>()?;
        if let Some(off) = offload {
            let by = |u: Unit| {
                off.bodies().iter().zip(&ran).filter(|(b, _)| b.unit == u).map(|(_, r)| r.batches).sum::<usize>()
            };
            let (ane, cpu) = (by(Unit::Ane), by(Unit::Cpu));
            eprintln!(
                "coreml: {ane} batches on the Neural Engine, {cpu} on the CPU, {} on the GPU; encoder waited {slot_wait_s:.2}s for ring slots",
                sched.batches - ane - cpu
            );
            for (body, r) in off.bodies().iter().zip(&ran) {
                eprintln!(
                    "coreml:   {:?} worker {} batches, predict {:.2}s, fence {:.2}s, idle {:.2}s",
                    body.unit, r.batches, r.predict_s, r.fence_s, r.idle_s
                );
            }
        }
        let mut result = queued;
        for w in writers {
            let r = w.join().map_err(|_| eyre!("a writer thread panicked"))?;
            if result.is_ok() {
                result = r;
            }
        }
        result
    });
    gpu.synchronize()?;
    written?;
    rep.run_s = t.elapsed().as_secs_f64();
    rep.total_s = t_pass.elapsed().as_secs_f64();
    if let (Some(a), Some(b)) = (&first_fence, &last_fence) {
        rep.gpu_s = (b.gpu_span().1 - a.gpu_span().0).max(0.0);
    }
    rep.tiles_per_s = rep.tiles as f64 / rep.run_s;

    let mut maps: Vec<Option<Vec<u8>>> = vec![None; targets.len()];
    let mut coverage: Vec<Option<Vec<u8>>> = vec![None; targets.len()];
    for (index, is_cov, image) in kept_rx.try_iter() {
        let kept = if is_cov { &mut coverage } else { &mut maps };
        kept[index] = Some(image);
    }
    for (i, m) in targets.iter().enumerate() {
        let want = |d: Option<&Dest>| matches!(d, Some(Dest::Memory));
        ensure!(
            maps[i].is_some() == want(Some(&m.map)) && coverage[i].is_some() == want(m.coverage.as_ref()),
            "map {i}: in-memory outputs out of step"
        );
    }
    Ok(PassOutput { report: rep, logits, maps, coverage })
}

#[allow(clippy::too_many_arguments)]
fn finish_map(
    gpu: &Gpu,
    sched: &Schedule,
    m: usize,
    logits: &Buf,
    grids: &[Buf],
    ys: &Buf,
    xs: &Buf,
    y_span: &Buf,
    x_span: &Buf,
    weight: &Buf,
    out: &Buf,
    cov: Option<&Buf>,
    (width, edge): (usize, Edge),
) -> Result<()> {
    ensure!(grids[m].bytes() == sched.ys.len() * sched.xs.len() * 4, "map {m}: grid size");
    let n = out.bytes();
    gpu.dispatch_1d(
        "blend_finish",
        n,
        &[
            out.arg(),
            cov.unwrap_or(out).arg(),
            logits.arg(),
            grids[m].arg(),
            ys.arg(),
            xs.arg(),
            y_span.arg(),
            x_span.arg(),
            weight.arg(),
            width.into(),
            n.into(),
            sched.xs.len().into(),
            u32::from(cov.is_some()).into(),
            edge.px().into(),
        ],
    )
}

/// The reader name the export for `ckpt` records, for output file names (`hybrid_3d2d-seed42`, `reader-v2`)
pub fn reader_name(ckpt: &Path) -> Result<String> {
    export_reader(&resolve_weights(ckpt, &weights_dir())?)
}

/// The tile edge the sidecar of the safetensors export `export` fixes, as the CUDA engine's `export_edge`
pub fn export_edge(export: &Path) -> Result<Edge> {
    let sidecar = export.with_extension("json");
    let side: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&sidecar).wrap_err_with(|| format!("read {}", sidecar.display()))?)?;
    let crop: Vec<usize> = serde_json::from_value(side.get("crop_size").cloned().unwrap_or_default())
        .wrap_err_with(|| format!("{}: crop_size", sidecar.display()))?;
    let edge = Edge::from_crop(&crop);
    ensure_supported!(edge.is_some(), "crop {crop:?}, this engine needs [17, 128, 128] or [17, 384, 384]");
    Ok(edge.unwrap_or_default())
}

/// The SHA-256 of the checkpoint the safetensors export `export` was made from, as its sidecar records it
fn export_checkpoint_sha(export: &Path) -> Result<String> {
    let sidecar = export.with_extension("json");
    let side: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&sidecar).wrap_err_with(|| format!("read {}", sidecar.display()))?)?;
    let sha = side.get("checkpoint_sha256").and_then(|v| v.as_str()).unwrap_or_default();
    ensure!(sha.len() == 64, "{}: no checkpoint_sha256", sidecar.display());
    Ok(sha.to_owned())
}

/// The reader name the sidecar of the safetensors export `export` records
pub fn export_reader(export: &Path) -> Result<String> {
    let sidecar = export.with_extension("json");
    let side: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&sidecar).wrap_err_with(|| format!("read {}", sidecar.display()))?)?;
    let name =
        side.get("reader").and_then(|v| v.as_str()).ok_or_else(|| eyre!("{}: no reader name", sidecar.display()))?;
    ensure!(
        !name.is_empty()
            && name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
            && !name.contains(".."),
        "{}: reader name {name:?} is not a plain file-name token",
        sidecar.display()
    );
    Ok(name.to_owned())
}

/// The safetensors export for `ckpt`: the export itself, or for a `.pth`, the export in the weights directory whose
/// sidecar names that checkpoint
pub fn resolve_weights(ckpt: &Path, dir: &Path) -> Result<PathBuf> {
    match ckpt.extension().and_then(|e| e.to_str()) {
        Some("safetensors") => Ok(ckpt.to_path_buf()),
        Some("pth") => {
            let want = std::fs::canonicalize(ckpt).wrap_err_with(|| format!("checkpoint {}", ckpt.display()))?;
            for entry in std::fs::read_dir(dir).wrap_err_with(|| format!("read {}", dir.display()))? {
                let path = entry?.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                let side: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
                let named = side.get("checkpoint").and_then(|v| v.as_str()).map(Path::new);
                if named.and_then(|p| std::fs::canonicalize(p).ok()).as_deref() == Some(want.as_path()) {
                    let sha = side.get("checkpoint_sha256").and_then(|v| v.as_str()).unwrap_or_default();
                    let bytes = std::fs::read(&want).wrap_err_with(|| format!("read {}", want.display()))?;
                    let got: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
                    ensure!(
                        got == sha,
                        "{} has sha256 {got}, but {} was exported from {sha}; re-export",
                        want.display(),
                        path.display()
                    );
                    return Ok(path.with_extension("safetensors"));
                }
            }
            bail!(
                "no safetensors export of {} in {} (set INKRS_WEIGHTS_DIR); run `python -m inkrs.export_weights CKPT DIR` first",
                ckpt.display(),
                dir.display()
            )
        }
        _ => bail!("checkpoint {} is neither a .pth nor a .safetensors export", ckpt.display()),
    }
}

/// Where a `.pth` checkpoint's safetensors export is looked up: `INKRS_WEIGHTS_DIR`, else `weights/`
pub fn weights_dir() -> PathBuf {
    std::env::var_os("INKRS_WEIGHTS_DIR").map_or_else(|| PathBuf::from("weights"), PathBuf::from)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::check_conv_request;
    use crate::error::is_unsupported;

    #[test]
    fn a_cuda_conv_request_is_refused_as_unsupported() {
        assert!(check_conv_request(None).is_ok());

        for v in ["inkrs", "cudnn", ""] {
            let err = check_conv_request(Some(OsStr::new(v))).unwrap_err();
            assert!(is_unsupported(&err), "{v}: {err}");
        }
    }
}
