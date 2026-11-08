//! The resident pipeline: the volume and model stay on the GPU, every map's tiles run in
//! fixed-size batches with no host synchronisation between them, each map is blended and
//! finished on the GPU as soon as its last tile is done, and writer threads encode the TIFFs
//! while later maps compute

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

use color_eyre::eyre::{Result, WrapErr, bail, ensure, eyre};
use cudarc::driver::result as cur;
use cudarc::driver::sys as cu;
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::conv::ConvBackend;
use crate::dnn::{Dnn, Pick, algo_cache};
use crate::ensure_supported;
use crate::gpu::{Arg, Buf, Gpu, Pinned};
use crate::model::{Ctx, Model, logits_buf, upload_desc};
use crate::sched::{DESC, Edge, Occupancy, Schedule, hann_weights};
use crate::tta::{MIRRORS, Tta};
use crate::weights::Checkpoint;
use crate::zarr::{LowRes, MAX_VOLUME_BYTES, Volume, VolumeRef};

mod tta;
pub use tta::TtaOutput;

/// One output map of the CLI: its layers and where each pass writes it
#[derive(Clone, Debug)]
pub struct MapSpec {
    pub first: i32,
    pub step: i32,
    /// One output path per pass
    pub paths: Vec<PathBuf>,
    /// With `--coverage`, one path per pass for the engine's coverage mask (uint8 TIFF, 1 where
    /// the blend's weight sum is above villa's 1e-6 cut); empty otherwise
    pub coverage: Vec<PathBuf>,
}

/// Engine options shared by `infer` and `sweep`
#[derive(Clone, Debug)]
pub struct Options {
    pub stride: usize,
    pub batch: usize,
    /// Whether the settings came from the `--fast` preset ([`crate::fast`]) rather than parity mode;
    /// recorded in the report only
    pub fast: bool,
    pub pick: Pick,
    pub max_workspace: usize,
    /// Which library runs the reader's 2D convs; None for the GPU's default ([`ConvBackend::resolve`])
    pub conv: Option<ConvBackend>,
    /// Download every logits slot after each pass and fail on a non-finite value
    pub check_finite: bool,
    /// Export each map's coverage mask from the blend kernel to [`MapSpec::coverage`]
    pub coverage: bool,
    pub writers: usize,
    /// Test-time augmentation: off writes villa's uint8 TIFFs, 8-fold the recipe's float32 `.npy` maps
    /// ([`crate::tta`])
    pub tta: Tta,
    /// When the process started, for the cold-start time
    pub started: Instant,
}

/// Timings and counts of one run, written next to the outputs
#[derive(Debug, Default, Serialize)]
pub struct Report {
    pub stride: usize,
    pub batch: usize,
    /// The reader's numerics, always fp16 ([`crate::fast::ReaderPrecision`])
    pub precision: String,
    /// Whether the run used the `--fast` preset, which gives up map parity with villa
    pub fast: bool,
    /// Test-time augmentation variants per tile position, 8 with `--tta 8`; absent when off
    #[serde(skip_serializing_if = "is_one")]
    pub tta: u32,
    pub cudnn: usize,
    /// The device's compute capability and the PTX tier loaded on it
    pub compute_capability: String,
    pub ptx_tier: String,
    /// The library that ran the reader's 2D convs, `inkrs` or `cudnn` ([`ConvBackend::resolve`])
    pub conv: &'static str,
    /// Why that library: requested, the default for this GPU's compute capability, or a tier without
    /// the own kernels
    pub conv_reason: String,
    pub load_weights_s: f64,
    pub open_gpu_s: f64,
    /// Model construction: weight upload and cuDNN plans (with `--algo find`, the timed search)
    pub plan_s: f64,
    /// Plans whose `--algo find` ranking came from the on-disk algorithm cache
    pub algo_cache_hits: usize,
    /// The algorithm cache's state, and why it is off when it is; None only before planning
    pub algo_cache: Option<algo_cache::Status>,
    /// Process start to the first pass's last TIFF on disk
    pub cold_s: f64,
    pub passes: Vec<PassReport>,
    pub algos: Vec<(String, String)>,
    pub schema: (usize, usize, usize),
}

/// One pass over the volume with the model resident: everything a further segment would cost
#[derive(Debug, Default, Serialize)]
pub struct PassReport {
    pub volume: String,
    pub shape: [usize; 3],
    pub maps: usize,
    pub tiles: usize,
    pub batches: usize,
    pub padded_tiles: usize,
    /// For the CLI's first pass, the read and the schedule run while the engine loads
    pub read_volume_s: f64,
    pub schedule_s: f64,
    pub upload_s: f64,
    /// From the first batch queued to the last TIFF written
    pub run_s: f64,
    /// From the volume read to the last TIFF written: the pass alone
    pub total_s: f64,
    /// GPU time from the first batch to the last map's finish, by events
    pub gpu_s: f64,
    /// Tiles over `run_s`
    pub tiles_per_s: f64,
    /// Per map: output path and the number of tiles villa's skip rule keeps
    pub map_tiles: Vec<(String, usize)>,
    /// villa's low-resolution occupancy scan, none for a bare array
    pub occupancy: Option<OccupancyReport>,
    /// Tile positions that scan schedules, before the per-map raw-window skip
    pub scheduled_positions: usize,
    /// The 8-fold TTA's second read, turned a quarter; `tiles` and `batches` above count both reads'
    /// network runs. Absent without TTA
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tta_quarter: Option<QuarterReport>,
}

/// The quarter-turned read of an 8-fold TTA pass
#[derive(Debug, Default, Serialize)]
pub struct QuarterReport {
    /// Tile positions (each run as four mirrored copies) the upright and the quarter-turned read kept
    pub upright_positions: usize,
    pub quarter_positions: usize,
}

fn is_one(n: &u32) -> bool {
    *n == 1
}

/// The pyramid level villa's first empty-tile skip read
#[derive(Debug, Default, Serialize)]
pub struct OccupancyReport {
    pub level: String,
    pub shape: [usize; 2],
    pub scale: [usize; 2],
    /// Nonzero mask pixels
    pub nonempty: usize,
}

/// Where one output image goes
#[derive(Clone, Debug)]
pub enum Dest {
    /// villa's uint8 TIFF at this path
    File(PathBuf),
    /// Returned in [`PassOutput`]
    Memory,
}

/// One map of a pass: its layers (the first of the 17 and the step, -1 for the reverse
/// direction) and where its map and, optionally, its coverage mask go
#[derive(Clone, Debug)]
pub struct Target {
    pub first: i32,
    pub step: i32,
    pub map: Dest,
    /// The coverage mask (1 where the blend's weight sum is above villa's 1e-6 cut, else 0)
    pub coverage: Option<Dest>,
}

/// The input of a pass
pub enum Source<'a> {
    /// A zarr v2 array, or a group whose array `0` is read
    Zarr { path: &'a Path, scan: Scan<'a> },
    /// A volume already in memory; `label` names it in the report
    Array { volume: VolumeRef<'a>, scan: Option<&'a LowRes>, label: String },
}

/// villa's first empty-tile skip, its low-resolution occupancy scan, for a zarr input
#[derive(Clone, Copy)]
pub enum Scan<'a> {
    /// As villa scans the input: its group's pyramid, or no scan for a bare array
    Villa,
    /// This scan instead (none skips only by the raw window)
    Given(Option<&'a LowRes>),
}

/// What a pass returns: its report, its logits (one FP16 128x128 slot per tile, for the
/// finite check), and per target the images sent to [`Dest::Memory`], `height * width` bytes
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
    /// Where a `.pth` checkpoint's export is looked up ([`weights_dir`] for the CLI)
    pub weights_dir: PathBuf,
    /// The CUDA device ordinal
    pub device: usize,
    pub batch: usize,
    pub pick: Pick,
    pub max_workspace: usize,
    /// Which library runs the reader's 2D convs; None for the GPU's default ([`ConvBackend::resolve`])
    pub conv: Option<ConvBackend>,
}

/// What loading the engine cost, and what it chose
#[derive(Debug, Default)]
pub struct LoadReport {
    /// The reader name the export's sidecar records, for output file names; None without a sidecar
    pub reader: Option<String>,
    pub cudnn: usize,
    /// The device's compute capability and the PTX tier loaded on it
    pub compute_capability: String,
    pub ptx_tier: String,
    /// The library that runs the reader's 2D convs, as resolved for this GPU
    pub conv: &'static str,
    /// Why that library ([`crate::conv::ConvReason`])
    pub conv_reason: String,
    pub load_weights_s: f64,
    pub open_gpu_s: f64,
    pub plan_s: f64,
    /// Plans whose [`Pick::Find`] ranking came from the on-disk algorithm cache
    pub algo_cache_hits: usize,
    /// The algorithm cache's state, and why it is off when it is; None only before planning
    pub algo_cache: Option<algo_cache::Status>,
    pub algos: Vec<(String, String)>,
    pub schema: (usize, usize, usize),
}

/// A recorded event, sent to a writer thread
struct Event(cu::CUevent);

// SAFETY: a driver event handle may be synchronised and destroyed from any thread
unsafe impl Send for Event {}

struct WriteJob {
    event: Event,
    host: Pinned,
    /// The target's index and where its map goes
    map: (usize, Dest),
    /// The coverage mask's buffer and destination, downloaded before the same event
    coverage: Option<(Pinned, Dest)>,
}

/// An image the writers kept in memory: target index, whether it is the coverage mask, bytes
type Kept = (usize, bool, Vec<u8>);

/// The model resident on the GPU, planned at one batch size, reusable across volumes
///
/// Every method binds the engine's CUDA context to the calling thread, so the engine may move
/// between threads, but it must not run on two at once
pub struct Engine {
    // field order is drop order: the pinned buffers, the model's buffers and plans, then the
    // cuDNN handle, then the GPU's context; `Drop` binds the context first
    /// Pinned map buffers kept between passes; pinning is slow
    hosts: Vec<Pinned>,
    model: Model,
    dnn: Dnn,
    gpu: Gpu,
    pick: Pick,
    max_workspace: usize,
    pub load: LoadReport,
}

impl Drop for Engine {
    fn drop(&mut self) {
        // the last reference can go on a thread that never ran the engine (a Python finalizer, say),
        // and the buffers', plans' and handles' destructors need the engine's context current there;
        // they ignore errors, so without this they would leak silently
        let _ = self.gpu.inner.context().bind_to_thread();
    }
}

// SAFETY: the driver, cuDNN and cuBLAS handles inside may be used from any thread once the
// engine's context is bound to it, which every method does first; `&mut self` on the methods
// that touch the GPU keeps them from running on two threads at once, and the kernel cache
// behind `Gpu`'s RefCell is only reached through them
unsafe impl Send for Engine {}

impl Engine {
    /// Loads the safetensors export `ckpt` (or the export of a villa `.pth` in
    /// `opts.weights_dir`), opens device 0 and plans the model at `opts.batch`
    pub fn open(ckpt: &Path, opts: &EngineOptions) -> Result<Self> {
        ensure!(opts.batch > 0, "batch size must be positive");
        let mut load = LoadReport::default();
        let t = Instant::now();
        let export = resolve_weights(ckpt, &opts.weights_dir)?;
        let ck = Checkpoint::load(&export)?;
        // `inkrs run` takes an export without a sidecar, so only callers that name files need the reader
        load.reader = export_reader(&export).ok();
        load.load_weights_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let gpu = Gpu::open_device(opts.device)?;
        let dnn = Dnn::new(&gpu)?;
        load.open_gpu_s = t.elapsed().as_secs_f64();
        load.cudnn = Dnn::version();
        load.compute_capability = gpu.inner.compute_capability().to_string();
        load.ptx_tier = gpu.inner.ptx_tier().to_string();

        let t = Instant::now();
        let ctx = Ctx { gpu: &gpu, dnn: &dnn, pick: opts.pick, max_workspace: opts.max_workspace };
        let (conv, why) = ConvBackend::resolve(opts.conv, gpu.inner.compute_capability(), gpu.inner.ptx_tier())?;
        load.conv = conv.name();
        load.conv_reason = why.to_string();
        let model = Model::new(&ctx, &ck, opts.batch, ck.edge, conv)?;
        load.schema = ck.finish()?;
        drop(ck);
        load.algos = model.convs().into_iter().map(|(n, c)| (n, format!("{} | {}", c.shape(), c.algo()))).collect();
        load.algo_cache_hits = model.convs().iter().filter(|(_, c)| c.cached()).count();
        load.algo_cache = Some(algo_cache::status());
        gpu.sync()?;
        load.plan_s = t.elapsed().as_secs_f64();
        Ok(Self { hosts: Vec::new(), model, dnn, gpu, pick: opts.pick, max_workspace: opts.max_workspace, load })
    }

    /// The device's free and total memory in bytes, as the driver reports it
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        self.gpu.inner.context().bind_to_thread()?;
        Ok(cudarc::driver::result::mem_get_info()?)
    }

    /// The batch size the model was planned at
    pub fn batch(&self) -> usize {
        self.model.batch
    }

    /// The tile edge of the loaded reader
    pub fn edge(&self) -> Edge {
        self.model.edge
    }

    /// Downloads a pass's logits and fails on any non-finite value
    pub fn check_finite(&mut self, out: &PassOutput) -> Result<()> {
        self.gpu.inner.context().bind_to_thread()?;
        let all: Vec<f16> = self.gpu.download_range(&out.logits, 0, out.report.tiles * self.model.edge.pixels())?;
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
        let input = Input::prepare(src, targets, opts.stride, self.model.batch, self.model.edge, Tta::Off)?;
        self.run_prepared(input, targets, opts)
    }

    /// One 8-fold TTA pass over `src` ([`crate::tta`]): every target's float32 map, written as `.npy` or
    /// returned. `src` must carry no low-resolution scan, and no target a coverage mask
    pub fn run_tta(&mut self, src: Source, targets: &[Target], stride: usize) -> Result<TtaOutput> {
        ensure!(!targets.is_empty(), "no maps to run");
        let patch = self.model.edge.px();
        ensure!((1..=patch).contains(&stride), "stride must be in [1, {patch}]");
        let input = Input::prepare(src, targets, stride, self.model.batch, self.model.edge, Tta::Dihedral8)?;
        self.run_prepared_tta(input, targets)
    }

    fn run_prepared_tta(&mut self, input: Input, targets: &[Target]) -> Result<TtaOutput> {
        ensure!(
            input.batch == self.model.batch && input.edge == self.model.edge && input.quarter.is_some(),
            "input not scheduled for this engine's 8-fold TTA"
        );
        self.gpu.inner.context().bind_to_thread()?;
        let Self { model, dnn, gpu, pick, max_workspace, .. } = self;
        let ctx = Ctx { gpu, dnn, pick: *pick, max_workspace: *max_workspace };
        tta::run(&ctx, model, input, targets)
    }

    /// [`Engine::run_pass`] over an input already read and scheduled for `targets` at this
    /// engine's batch size
    fn run_prepared(&mut self, input: Input, targets: &[Target], opts: &PassOptions) -> Result<PassOutput> {
        ensure!(opts.writers > 0, "writer count must be positive");
        ensure!(
            input.batch == self.model.batch && input.edge == self.model.edge,
            "input scheduled at batch {} and edge {}, the model's are {} and {}",
            input.batch,
            input.edge.px(),
            self.model.batch,
            self.model.edge.px()
        );
        self.gpu.inner.context().bind_to_thread()?;
        let Self { model, dnn, gpu, pick, max_workspace, hosts, .. } = self;
        let ctx = Ctx { gpu, dnn, pick: *pick, max_workspace: *max_workspace };
        run_pass(&ctx, model, input, targets, hosts, opts)
    }
}

/// Runs `maps` over the volume at `zarr` with the weights at `ckpt`, once per output path
/// each map carries; the model and its plans persist across passes, while each pass reads,
/// schedules and uploads the volume again as a new segment would. The first pass's read and
/// schedule run on the CPU while the engine loads
pub fn run(zarr: &Path, ckpt: &Path, maps: &[MapSpec], opts: &Options) -> Result<Report> {
    ensure!(!maps.is_empty(), "no maps to run");
    ensure!(opts.batch > 0 && opts.writers > 0, "batch size and writer count must be positive");
    let passes = maps[0].paths.len();
    ensure!(passes > 0 && maps.iter().all(|m| m.paths.len() == passes), "every map needs one path per pass");
    ensure!(
        maps.iter().all(|m| m.coverage.len() == if opts.coverage { passes } else { 0 }),
        "every map needs one coverage path per pass with --coverage, and none without"
    );
    ensure_supported!(
        opts.tta == Tta::Off || !(opts.coverage || opts.check_finite),
        "--tta 8 writes no coverage mask and runs no finite check"
    );
    // the edge comes from the export's sidecar, so the schedule below can run beside the load
    let export = resolve_weights(ckpt, &weights_dir())?;
    let edge = export_edge(&export)?;
    let patch = edge.px();
    ensure!((1..=patch).contains(&opts.stride), "stride must be in [1, {patch}]");
    let targets = |pass: usize| -> Vec<Target> {
        maps.iter()
            .map(|m| Target {
                first: m.first,
                step: m.step,
                map: Dest::File(m.paths[pass].clone()),
                coverage: m.coverage.get(pass).cloned().map(Dest::File),
            })
            .collect()
    };
    // the 8-fold recipe never runs villa's low-resolution scan ([`crate::tta`])
    let source =
        || Source::Zarr { path: zarr, scan: if opts.tta == Tta::Off { Scan::Villa } else { Scan::Given(None) } };
    // the volume read and the tile schedule need neither the weights nor the GPU, so they run
    // beside the load; a failed load is reported before a failed read, as when they ran in turn
    let first = targets(0);
    let (engine, input) = std::thread::scope(|scope| {
        let reader = scope.spawn(|| Input::prepare(source(), &first, opts.stride, opts.batch, edge, opts.tta));
        let engine = Engine::open(
            &export,
            &EngineOptions {
                weights_dir: weights_dir(),
                device: 0,
                batch: opts.batch,
                pick: opts.pick,
                max_workspace: opts.max_workspace,
                conv: opts.conv,
            },
        );
        let input = reader.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic));
        (engine, input)
    });
    let mut engine = engine?;
    let mut input = Some(input?);
    let load = &engine.load;
    let mut rep = Report {
        stride: opts.stride,
        batch: opts.batch,
        precision: crate::fast::ReaderPrecision::Fp16.name().into(),
        fast: opts.fast,
        tta: opts.tta.count(),
        cudnn: load.cudnn,
        compute_capability: load.compute_capability.clone(),
        ptx_tier: load.ptx_tier.clone(),
        conv: load.conv,
        conv_reason: load.conv_reason.clone(),
        load_weights_s: load.load_weights_s,
        open_gpu_s: load.open_gpu_s,
        plan_s: load.plan_s,
        algo_cache_hits: load.algo_cache_hits,
        algo_cache: load.algo_cache.clone(),
        algos: load.algos.clone(),
        schema: load.schema,
        ..Default::default()
    };

    let pass_opts = PassOptions { stride: opts.stride, writers: opts.writers };
    for pass in 0..passes {
        let targets = targets(pass);
        if opts.tta != Tta::Off {
            let out = match input.take() {
                Some(input) => engine.run_prepared_tta(input, &targets)?,
                None => engine.run_tta(source(), &targets, opts.stride)?,
            };
            if pass == 0 {
                rep.cold_s = opts.started.elapsed().as_secs_f64();
            }
            rep.passes.push(out.report);
            continue;
        }
        let out = match input.take() {
            Some(input) => engine.run_prepared(input, &targets, &pass_opts)?,
            None => engine.run_pass(source(), &targets, &pass_opts)?,
        };
        if pass == 0 {
            rep.cold_s = opts.started.elapsed().as_secs_f64();
        }
        if opts.check_finite {
            // after the timings: every slot of the pass is downloaded and scanned
            engine.check_finite(&out).wrap_err_with(|| format!("pass {}", pass + 1))?;
        }
        rep.passes.push(out.report);
    }
    Ok(rep)
}

/// A pass's input read and scheduled on the CPU, before anything reaches the GPU
struct Input<'a> {
    /// The pass report so far: the volume, its read and schedule times and the occupancy scan
    rep: PassReport,
    /// When the read started, for the pass total
    started: Instant,
    /// The volume read from zarr, or None for a caller's array
    read: Option<Volume>,
    array: Option<VolumeRef<'a>>,
    sched: Schedule,
    /// The 8-fold TTA's quarter-turned read ([`Occupancy::quarter`]); `sched` is then the upright
    /// read's, both of mirror4 copies. None without TTA
    quarter: Option<Schedule>,
    /// The batch size the tiles were packed for
    batch: usize,
    /// The tile edge they were scheduled at
    edge: Edge,
}

impl<'a> Input<'a> {
    /// Reads `src`, checks it, runs villa's occupancy scan and packs `targets`' tiles of `edge` into
    /// batches of `batch` at `stride`; with 8-fold TTA, both reads' mirror4 copies
    fn prepare(src: Source<'a>, targets: &[Target], stride: usize, batch: usize, edge: Edge, tta: Tta) -> Result<Self> {
        let patch = edge.px();
        let mut rep = PassReport { maps: targets.len(), ..Default::default() };
        let started = Instant::now();
        let t = Instant::now();
        let (read, array) = match &src {
            Source::Zarr { path, .. } => {
                let read = Volume::read(path)?;
                rep.volume = read.array.display().to_string();
                (Some(read), None)
            }
            Source::Array { volume, label, .. } => {
                rep.volume = label.clone();
                (None, Some(*volume))
            }
        };
        let vol = read.as_ref().map_or_else(|| array.expect("an array source"), Volume::view);
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
        let occ = Occupancy::new(vol, stride, lowres, patch);
        rep.scheduled_positions = occ.scheduled();
        let layers: Vec<(i32, i32)> = targets.iter().map(|m| (m.first, m.step)).collect();
        let (sched, quarter) = match tta {
            Tta::Off => (Schedule::new(&occ, vol.height, vol.width, &layers, batch, 1)?, None),
            Tta::Dihedral8 => {
                ensure_supported!(lowres.is_none(), "8-fold TTA runs no low-resolution occupancy scan, as its recipe");
                ensure_supported!(targets.iter().all(|m| m.coverage.is_none()), "8-fold TTA writes no coverage mask");
                let upright = Schedule::new(&occ, vol.height, vol.width, &layers, batch, MIRRORS)?;
                let occ = Occupancy::quarter(vol, stride, patch);
                let quarter = Schedule::new(&occ, vol.width, vol.height, &layers, batch, MIRRORS)?;
                rep.tta_quarter = Some(QuarterReport {
                    upright_positions: upright.slots / MIRRORS,
                    quarter_positions: quarter.slots / MIRRORS,
                });
                (upright, Some(quarter))
            }
        };
        rep.schedule_s = t.elapsed().as_secs_f64();
        Ok(Self { rep, started, read, array, sched, quarter, batch, edge })
    }

    fn volume(&self) -> VolumeRef<'_> {
        self.read.as_ref().map_or_else(|| self.array.expect("an array source"), Volume::view)
    }
}

/// One pass. `hosts` holds pinned map buffers that later passes reuse when the size still fits
fn run_pass(
    ctx: &Ctx,
    model: &Model,
    input: Input,
    targets: &[Target],
    hosts: &mut Vec<Pinned>,
    opts: &PassOptions,
) -> Result<PassOutput> {
    let gpu = ctx.gpu;
    let t_pass = input.started;

    // the host volume lives only until it is on the GPU: a `..` pattern would leave `read` in
    // `input` until the pass returns, holding up to a GiB of host memory through it
    let (height, width) = (input.volume().height, input.volume().width);
    let Input { mut rep, read, array, sched, .. } = input;
    let t = Instant::now();
    let vbuf = gpu.upload(read.as_ref().map_or_else(|| array.expect("an array source"), Volume::view).data)?;
    drop(read);
    rep.upload_s = t.elapsed().as_secs_f64();
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
    let desc = upload_desc(gpu, &sched.desc)?;
    let logits = logits_buf(gpu, sched.slots, model.edge)?;
    let weight = gpu.upload(&hann_weights(model.edge))?;
    let ys = gpu.upload(&sched.ys)?;
    let xs = gpu.upload(&sched.xs)?;
    let y_span = gpu.upload(&sched.y_span)?;
    let x_span = gpu.upload(&sched.x_span)?;
    let grids: Vec<Buf> = sched.maps.iter().map(|m| gpu.upload(&m.grid)).collect::<Result<_>>()?;
    let outs: Vec<Buf> = (0..targets.len()).map(|_| gpu.alloc(height * width)).collect::<Result<_>>()?;
    // one coverage buffer per target that asks for one
    let covs: Vec<Option<Buf>> = targets
        .iter()
        .map(|m| m.coverage.as_ref().map(|_| gpu.alloc(height * width)).transpose())
        .collect::<Result<_>>()?;
    // pinned buffers from an earlier pass are kept when large enough; pinning is slow
    let needed = targets.len() + covs.iter().flatten().count();
    hosts.retain(|h| h.as_slice().len() >= height * width);
    while hosts.len() < needed {
        hosts.push(gpu.pinned(height * width)?);
    }
    let mut pool = std::mem::take(hosts).into_iter();
    let mut slots: Vec<Option<(Pinned, Option<Pinned>)>> = covs
        .iter()
        .map(|cov| {
            let main = pool.next()?;
            let cov = if cov.is_some() { Some(pool.next()?) } else { None };
            Some((main, cov))
        })
        .collect();
    hosts.extend(pool);
    gpu.sync()?;
    rep.upload_s += t.elapsed().as_secs_f64();

    // writers wait on each map's event, encode or copy, and hand the pinned buffer back
    let (tx, rx) = mpsc::channel::<WriteJob>();
    let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
    let (back_tx, back_rx) = mpsc::channel::<Pinned>();
    let (kept_tx, kept_rx) = mpsc::channel::<Kept>();
    let context = gpu.inner.context().clone();
    let start_ev = timing_event()?;
    let end_ev = timing_event()?;
    let t = Instant::now();
    let written: Result<()> = std::thread::scope(|scope| {
        let writers: Vec<_> = (0..opts.writers)
            .map(|_| {
                let rx = rx.clone();
                let back_tx = back_tx.clone();
                let kept_tx = kept_tx.clone();
                let context = context.clone();
                scope.spawn(move || -> Result<()> {
                    context.bind_to_thread()?;
                    let emit = |host: &Pinned, index: usize, coverage: bool, dest: &Dest| -> Result<()> {
                        let image = &host.as_slice()[..height * width];
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
                        // SAFETY: the event was recorded after this map's copy into `host`
                        unsafe {
                            cur::event::synchronize(job.event.0)?;
                            cur::event::destroy(job.event.0)?;
                        }
                        let (index, dest) = &job.map;
                        emit(&job.host, *index, false, dest)?;
                        back_tx.send(job.host).ok();
                        if let Some((host, dest)) = job.coverage {
                            emit(&host, *index, true, &dest)?;
                            back_tx.send(host).ok();
                        }
                    }
                })
            })
            .collect();
        drop(back_tx);
        drop(kept_tx);

        let queued = (|| -> Result<()> {
            // SAFETY: our stream; the event outlives the record
            unsafe { cur::event::record(start_ev, gpu.cu_stream())? };
            let mut next_map = 0;
            let mut finish = |m: usize| -> Result<()> {
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
                    &outs,
                    covs[m].as_ref(),
                    width,
                    model.edge,
                )?;
                let (host, cov_host) = slots[m].take().ok_or_else(|| eyre!("map {m} finished twice"))?;
                // the coverage copy is queued first, so the map's event covers both
                let coverage = match (cov_host, covs[m].as_ref(), targets[m].coverage.as_ref()) {
                    (Some(mut h), Some(c), Some(dest)) => {
                        // SAFETY: `h` moves to the writer, which synchronises on the map's event,
                        // recorded after this copy, before reading it
                        unsafe { gpu.download_async(&mut h, c, 0, c.bytes())? };
                        Some((h, dest.clone()))
                    }
                    (None, None, None) => None,
                    _ => bail!("map {m}: coverage buffers out of step"),
                };
                let (event, host) = queue_download(gpu, host, &outs[m])?;
                tx.send(WriteJob { event, host, map: (m, targets[m].map.clone()), coverage })
                    .map_err(|_| eyre!("writers stopped"))
            };
            for bi in 0..sched.batches {
                let view = desc.view::<i32>(bi * model.batch * DESC, model.batch * DESC);
                model.forward(ctx, (&vbuf, height, width), view, &logits)?;
                while next_map < targets.len() && sched.map_done_after[next_map] <= bi {
                    finish(next_map)?;
                    next_map += 1;
                }
            }
            // maps with no tiles at all, after the last batch
            while next_map < targets.len() {
                finish(next_map)?;
                next_map += 1;
            }
            // SAFETY: as above
            unsafe { cur::event::record(end_ev, gpu.cu_stream())? };
            Ok(())
        })();
        drop(tx);
        let mut result = queued;
        for w in writers {
            let r = w.join().map_err(|_| eyre!("a writer thread panicked"))?;
            if result.is_ok() {
                result = r;
            }
        }
        result
    });
    gpu.sync()?;
    written?;
    rep.run_s = t.elapsed().as_secs_f64();
    rep.total_s = t_pass.elapsed().as_secs_f64();
    // SAFETY: both events were recorded and the stream is idle
    unsafe {
        rep.gpu_s = cur::event::elapsed(start_ev, end_ev)? as f64 / 1e3;
        cur::event::destroy(start_ev)?;
        cur::event::destroy(end_ev)?;
    }
    rep.tiles_per_s = rep.tiles as f64 / rep.run_s;
    hosts.extend(back_rx.try_iter());

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

fn timing_event() -> Result<cu::CUevent> {
    Ok(cur::event::create(cu::CUevent_flags::CU_EVENT_DEFAULT)?)
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
    outs: &[Buf],
    cov: Option<&Buf>,
    width: usize,
    edge: Edge,
) -> Result<()> {
    ensure!(grids[m].bytes() == sched.ys.len() * sched.xs.len() * 4, "map {m}: grid size");
    ensure!(weight.bytes() == edge.pixels() * 4, "map {m}: blend window size");
    gpu.launch_1d(
        edge.kernel("blend_finish"),
        outs[m].bytes(),
        &[
            outs[m].arg::<u8>(),
            cov.map_or(Arg::EMPTY, |c| c.arg::<u8>()),
            logits.arg::<u16>(),
            grids[m].arg::<i32>(),
            ys.arg::<u32>(),
            xs.arg::<u32>(),
            y_span.arg::<[u32; 2]>(),
            x_span.arg::<[u32; 2]>(),
            weight.arg::<f32>(),
            Arg::U32(width as u32),
        ],
    )
}

/// Queues the D2H copy of a finished map into pinned memory and records its event
fn queue_download(gpu: &Gpu, mut host: Pinned, out: &Buf) -> Result<(Event, Pinned)> {
    // SAFETY: `host` moves to the writer, which synchronises on the event before reading it
    unsafe { gpu.download_async(&mut host, out, 0, out.bytes())? };
    let event = cur::event::create(cu::CUevent_flags::CU_EVENT_DISABLE_TIMING)?;
    // SAFETY: our stream and a fresh event
    unsafe { cur::event::record(event, gpu.cu_stream())? };
    Ok((Event(event), host))
}

/// The reader name the export for `ckpt` records, for output file names (`hybrid_3d2d-seed42`, `reader-v2`)
pub fn reader_name(ckpt: &Path) -> Result<String> {
    export_reader(&resolve_weights(ckpt, &weights_dir())?)
}

/// The tile edge the sidecar of the safetensors export `export` fixes by its `crop_size`;
/// unsupported for a crop this engine does not run
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

/// The reader name the sidecar of the safetensors export `export` records
pub fn export_reader(export: &Path) -> Result<String> {
    let sidecar = export.with_extension("json");
    let side: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&sidecar).wrap_err_with(|| format!("read {}", sidecar.display()))?)?;
    let name =
        side.get("reader").and_then(|v| v.as_str()).ok_or_else(|| eyre!("{}: no reader name", sidecar.display()))?;
    // the name goes into file names, so it stays a plain token
    ensure!(
        !name.is_empty()
            && name.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c))
            && !name.contains(".."),
        "{}: reader name {name:?} is not a plain file-name token",
        sidecar.display()
    );
    Ok(name.to_owned())
}

/// The safetensors export for `ckpt`: the export itself, or for a `.pth` (villa or Hecate), the export in
/// the weights directory whose sidecar names that checkpoint
pub fn resolve_weights(ckpt: &Path, dir: &Path) -> Result<PathBuf> {
    match ckpt.extension().and_then(|e| e.to_str()) {
        Some("safetensors") => Ok(ckpt.to_path_buf()),
        Some("pth") => {
            let want = canonicalize(ckpt).wrap_err_with(|| format!("checkpoint {}", ckpt.display()))?;
            for entry in std::fs::read_dir(dir).wrap_err_with(|| format!("read {}", dir.display()))? {
                let path = entry?.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                let side: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
                let named = side.get("checkpoint").and_then(|v| v.as_str()).map(Path::new);
                if named.and_then(|p| canonicalize(p).ok()).as_deref() == Some(want.as_path()) {
                    // the export must come from these exact checkpoint bytes, not an older file at the same path
                    let sha = side.get("checkpoint_sha256").and_then(|v| v.as_str()).unwrap_or_default();
                    let got = checkpoint_sha256(&want)?;
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
                "no safetensors export of {} in {} (set INKRS_WEIGHTS_DIR); run `python -m inkrs.export_weights CKPT DIR` (or scripts/export_weights.py, or scripts/export_hecate.py for Hecate) first, since inkrs does not read .pth files itself",
                ckpt.display(),
                dir.display()
            )
        }
        _ => bail!("checkpoint {} is neither a .pth nor a .safetensors export", ckpt.display()),
    }
}

/// `std::fs::canonicalize`, without the `\\?\` prefix Windows adds when the path reads the same without it
///
/// The digest cache keys its entries on the checkpoint's path, which Python's `Path.resolve()` gives without the
/// prefix, so both must spell it the same way to share entries. A path that needs the prefix, such as one with a
/// trailing dot, keeps it
pub fn canonicalize(path: &Path) -> std::io::Result<PathBuf> {
    let full = std::fs::canonicalize(path)?;
    #[cfg(windows)]
    let full = without_verbatim_prefix(full);
    Ok(full)
}

/// `full` (a canonical path) as `C:\...` or `\\server\share\...` when that names the same file
#[cfg(windows)]
fn without_verbatim_prefix(full: PathBuf) -> PathBuf {
    let Some(text) = full.to_str() else {
        return full;
    };
    let plain = if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = text.strip_prefix(r"\\?\")
        && rest.as_bytes().get(1) == Some(&b':')
    {
        rest.to_owned()
    } else {
        return full;
    };
    // the plain form goes through Windows' path normalisation, which the verbatim one skips
    match std::fs::canonicalize(&plain) {
        Ok(again) if again == full => PathBuf::from(plain),
        _ => full,
    }
}

/// The SHA-256 of the checkpoint at `path` (canonical), from the digest cache when an entry records the
/// file at its current size and modification time, else read and hashed, and stored for the next start.
/// The entry is `<cache>/checkpoint-sha256/<first 16 bytes of sha256(path), hex>.json`, which the Python
/// package reads and writes too.
/// Hashing the 138 MB `.pth` cost about 0.1 s of every load; a rewrite changes the modification time,
/// so a cached digest is never taken for other bytes at the same path
pub fn checkpoint_sha256(path: &Path) -> Result<String> {
    #[derive(serde::Deserialize, Serialize, PartialEq)]
    struct Digest256 {
        path: PathBuf,
        size: u64,
        mtime_ns: u128,
        sha256: String,
    }
    let meta = std::fs::metadata(path).wrap_err_with(|| format!("stat {}", path.display()))?;
    let mtime_ns = meta.modified()?.duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    let entry = cache_root().map(|root| {
        let name: String =
            Sha256::digest(path.as_os_str().as_encoded_bytes()).iter().take(16).map(|b| format!("{b:02x}")).collect();
        root.join("checkpoint-sha256").join(format!("{name}.json"))
    });
    let hit = |d: &Digest256| d.path == path && d.size == meta.len() && d.mtime_ns == mtime_ns && mtime_ns > 0;
    if let Some(d) =
        entry.as_ref().and_then(|e| std::fs::read(e).ok()).and_then(|b| serde_json::from_slice::<Digest256>(&b).ok())
        && hit(&d)
    {
        return Ok(d.sha256);
    }
    let bytes = std::fs::read(path).wrap_err_with(|| format!("read {}", path.display()))?;
    let sha256: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
    let d = Digest256 { path: path.to_path_buf(), size: meta.len(), mtime_ns, sha256 };
    // best effort: a cache that cannot be written only costs the next start the hash again
    if let Some(e) = entry
        && hit(&d)
        && let Some(dir) = e.parent()
        && std::fs::create_dir_all(dir).is_ok()
    {
        let tmp = e.with_extension(format!("tmp{}", std::process::id()));
        if let Ok(json) = serde_json::to_vec(&d)
            && std::fs::write(&tmp, json).is_ok()
        {
            let _ = std::fs::rename(&tmp, &e);
        }
    }
    Ok(d.sha256)
}

/// inkrs's cache directory: `$INKRS_CACHE_DIR`, else `$XDG_CACHE_HOME/inkrs`, else `~/.cache/inkrs`
/// (`%LOCALAPPDATA%\inkrs` on Windows), as the Python package resolves it; None when no directory can be named
pub fn cache_root() -> Option<PathBuf> {
    cache_root_in(&|name| std::env::var_os(name))
}

/// [`cache_root`] as `env` names it, so a caller can resolve it from an environment of its own
pub fn cache_root_in(env: &dyn Fn(&str) -> Option<std::ffi::OsString>) -> Option<PathBuf> {
    let nonempty = |name: &str| env(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    nonempty("INKRS_CACHE_DIR")
        .or_else(|| nonempty("XDG_CACHE_HOME").map(|p| p.join("inkrs")))
        .or_else(|| default_cache_home(&nonempty).map(|p| p.join("inkrs")))
}

/// The per-user cache directory when `XDG_CACHE_HOME` is unset: `~/.cache`
#[cfg(not(windows))]
fn default_cache_home(nonempty: &dyn Fn(&str) -> Option<PathBuf>) -> Option<PathBuf> {
    nonempty("HOME").map(|home| home.join(".cache"))
}

/// The per-user cache directory when `XDG_CACHE_HOME` is unset
///
/// Windows sets no `HOME`. cuDNN plans and compiled kernels belong to this machine's GPU, so they go in the local
/// profile, `%LOCALAPPDATA%`, not the one that roams to other machines
#[cfg(windows)]
fn default_cache_home(nonempty: &dyn Fn(&str) -> Option<PathBuf>) -> Option<PathBuf> {
    nonempty("LOCALAPPDATA")
}

/// Where a `.pth` checkpoint's safetensors export is looked up: `INKRS_WEIGHTS_DIR`, else `weights/`
/// in the working directory (`scripts/export_weights.py`'s OUT_DIR)
pub fn weights_dir() -> PathBuf {
    std::env::var_os("INKRS_WEIGHTS_DIR").map_or_else(|| PathBuf::from("weights"), PathBuf::from)
}
