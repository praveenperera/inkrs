//! Experimental: reader-v2's 2D body (everything after the depth pool) through Core ML, so the Neural Engine and the
//! CPU run some batches' bodies while the GPU runs the 3D stems and the other batches
//!
//! A package takes inkrs's pooled activations as they are, `x` = `[batch, px, px, 32]` f16 channels-last, and
//! returns `logits` = `[batch, px, px]` f16, the same values `head_logits` writes. `metal/coreml_body.py`
//! builds them: fp16 compute for the Neural Engine, fp32 for the CPU (fp16 on the CPU misses parity)
//!
//! The offload runs with the fp16 reader only ([`OFFLOAD_FP16_ONLY`]): both bodies take and return f16, so even the
//! CPU's fp32-compute body rounds its input and its logits through FP16, and the Neural Engine computes in FP16

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

use block2::RcBlock;
use color_eyre::eyre::{Result, bail, ensure, eyre};
use half::f16;
use objc2::AnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel, MLModelConfiguration,
    MLModelCreatorDefinedKey, MLMultiArray, MLMultiArrayDataType, MLPredictionOptions,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};

use super::gpu::{Buf, Fence, Gpu};
use super::model::Model;
use crate::fast::ReaderPrecision;
use crate::sched::Edge;

/// Where Core ML runs a body
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// The Neural Engine, with the CPU for any op it refuses
    Ane,
    /// The CPU alone
    Cpu,
}

impl Unit {
    fn compute_units(self) -> MLComputeUnits {
        match self {
            Unit::Ane => MLComputeUnits::CPUAndNeuralEngine,
            Unit::Cpu => MLComputeUnits::CPUOnly,
        }
    }
}

/// One loaded body. Each worker thread owns its own, so no instance runs two predictions at once
pub struct Body {
    model: Retained<MLModel>,
    pub unit: Unit,
    batch: usize,
    px: usize,
}

// SAFETY: an MLModel may move between threads; each `Body` is used by one worker at a time (the engine hands every
// worker its own instance and shares none), so its predictions never overlap
unsafe impl Send for Body {}
unsafe impl Sync for Body {}

fn ns_err(e: Retained<NSError>) -> color_eyre::Report {
    eyre!("{}", e.localizedDescription())
}

fn numbers(v: &[usize]) -> Retained<NSArray<NSNumber>> {
    let v: Vec<Retained<NSNumber>> = v.iter().map(|&n| NSNumber::numberWithInteger(n as isize)).collect();
    NSArray::from_retained_slice(&v)
}

/// Why the offload refuses the fp32 reader
pub const OFFLOAD_FP16_ONLY: &str = "the Core ML offload (INKRS_COREML_ANE, INKRS_COREML_CPU) runs with reader \
     precision fp16 only: the Neural Engine computes in FP16, and every body takes and returns f16, so an fp32 run \
     would round its pooled activations and logits through FP16; unset them for --precision fp32";

/// The layout `metal/coreml_body.py` stamps on the bodies it builds: input `x` = the engine's pooled buffer, output
/// `logits` = one slot per tile. Bump both together when either side changes
const LAYOUT: &str = "pooled-nhwc32-logits-v1";

/// The creator-defined metadata of a loaded model, as strings
fn creator_metadata(model: &MLModel) -> HashMap<String, String> {
    // SAFETY: getters on a live model; the key is a framework constant
    let (meta, key) = unsafe { (model.modelDescription().metadata(), MLModelCreatorDefinedKey) };
    let dict = key.and_then(|k| meta.objectForKey(k)).and_then(|o| o.downcast::<NSDictionary>().ok());
    let Some(dict) = dict else { return HashMap::new() };
    let (keys, values) = dict.to_vecs();
    let text = |o: &AnyObject| o.downcast_ref::<NSString>().map(|s| s.to_string());
    keys.iter().zip(&values).filter_map(|(k, v)| Some((text(k)?, text(v)?))).collect()
}

impl Body {
    /// Loads the compiled model (`.mlmodelc`, from `metal/coreml_body.py`) at `compiled` for `unit`, planned for
    /// `batch` tiles of `px` x `px`. The model's stamp must name the checkpoint whose SHA-256 is `checkpoint`, the
    /// engine's layout, this batch and the unit's precision: a body from another reader would otherwise run
    /// beside the loaded reader's stems and GPU bodies and return plausible but wrong maps
    pub fn load(compiled: &Path, unit: Unit, (batch, px): (usize, usize), checkpoint: &str) -> Result<Self> {
        ensure!(
            compiled.extension().is_some_and(|e| e == "mlmodelc"),
            "{} is not a compiled .mlmodelc; run `xcrun coremlcompiler compile` on the package",
            compiled.display()
        );
        autoreleasepool(|_| {
            let url = NSURL::from_file_path(compiled).ok_or_else(|| eyre!("bad path {}", compiled.display()))?;
            // SAFETY: plain Objective-C object creation and a setter with a valid enum value
            let config = unsafe { MLModelConfiguration::new() };
            unsafe { config.setComputeUnits(unit.compute_units()) };
            // SAFETY: a valid URL and configuration
            let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
                .map_err(|e| ns_err(e).wrap_err(format!("load {}", compiled.display())))?;
            let meta = creator_metadata(&model);
            let precision = match unit {
                Unit::Ane => "fp16",
                Unit::Cpu => "fp32",
            };
            let want = [
                ("inkrs.layout", LAYOUT.to_owned()),
                ("inkrs.checkpoint_sha256", checkpoint.to_owned()),
                ("inkrs.batch", batch.to_string()),
                ("inkrs.edge", px.to_string()),
                ("inkrs.precision", precision.to_owned()),
            ];
            for (key, value) in want {
                let got = meta.get(key).map_or("(none)", String::as_str);
                ensure!(
                    got == value,
                    "{}: {key} is {got}, the engine needs {value}; rebuild it with metal/coreml_body.py",
                    compiled.display()
                );
            }
            Ok(Self { model, unit, batch, px })
        })
    }

    /// Runs one batch, reading `[batch, px, px, 32]` f16 at `x`, and calls `emit(t, logits)` with tile `t`'s
    /// `px * px` logits in row-major order
    ///
    /// # Safety
    /// `x` must point at `batch * px * px * 32` f16 values that nothing writes until this returns
    pub unsafe fn predict(&self, x: NonNull<u8>, mut emit: impl FnMut(usize, &[f16])) -> Result<()> {
        let (b, px) = (self.batch, self.px);
        autoreleasepool(|_| {
            let shape = numbers(&[b, px, px, 32]);
            let strides = numbers(&[px * px * 32, px * 32, 32, 1]);
            // SAFETY: the caller keeps `x` valid and unwritten for the prediction; with no deallocator Core ML
            // never frees it, and the array dies inside this pool, before the call returns
            let input = unsafe {
                MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
                    MLMultiArray::alloc(),
                    x.cast::<c_void>(),
                    &shape,
                    MLMultiArrayDataType::Float16,
                    &strides,
                    None,
                )
            }
            .map_err(ns_err)?;
            // SAFETY: plain Objective-C object creation from a valid array
            let value = unsafe { MLFeatureValue::featureValueWithMultiArray(&input) };
            let key = NSString::from_str("x");
            let value: &AnyObject = &value;
            let dict = NSDictionary::from_slices(&[&*key], &[value]);
            // SAFETY: a dictionary of feature values keyed by name
            let features = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(MLDictionaryFeatureProvider::alloc(), &dict)
            }
            .map_err(ns_err)?;
            let options = unsafe { MLPredictionOptions::new() };
            // SAFETY: valid provider and options; the model's input `x` has this shape and type, which Core ML checks
            let out = unsafe {
                self.model.predictionFromFeatures_options_error(ProtocolObject::from_ref(&*features), &options)
            }
            .map_err(|e| ns_err(e).wrap_err("Core ML prediction"))?;
            // SAFETY: plain lookups on the returned provider
            let logits = unsafe { out.featureValueForName(&NSString::from_str("logits")) }
                .and_then(|v| unsafe { v.multiArrayValue() })
                .ok_or_else(|| eyre!("the Core ML body has no `logits` array output"))?;
            self.emit(&logits, &mut emit)
        })
    }

    /// Hands each tile's logits from a `[batch, px, px]` f16 output to `emit`, honouring Core ML's strides
    fn emit(&self, logits: &MLMultiArray, emit: &mut dyn FnMut(usize, &[f16])) -> Result<()> {
        let (b, px) = (self.batch, self.px);
        // SAFETY: getters on a live array
        let (dtype, shape, strides) = unsafe { (logits.dataType(), logits.shape(), logits.strides()) };
        let dims: Vec<usize> = shape.iter().map(|n| n.integerValue() as usize).collect();
        let st: Vec<usize> = strides.iter().map(|n| n.integerValue() as usize).collect();
        ensure!(dtype == MLMultiArrayDataType::Float16, "Core ML logits are {dtype:?}, not f16");
        ensure!(dims == [b, px, px] && st.len() == 3 && st[2] == 1, "Core ML logits {dims:?} strides {st:?}");
        // the block is `Fn`, so its mutable state lives in a cell
        let state = RefCell::new((vec![f16::ZERO; px * px], emit, None));
        let handler = RcBlock::new(|ptr: NonNull<c_void>, size: isize| {
            let (row, emit, failed) = &mut *state.borrow_mut();
            let need = (b - 1) * st[0] + (px - 1) * st[1] + px;
            if (size as usize) < need * 2 {
                *failed = Some(eyre!("Core ML logits hold {size} bytes, need {}", need * 2));
                return;
            }
            // SAFETY: Core ML lends `size` bytes of f16 at `ptr` for the handler's duration, checked above
            let all = unsafe { std::slice::from_raw_parts(ptr.as_ptr().cast::<f16>(), size as usize / 2) };
            for t in 0..b {
                for y in 0..px {
                    let at = t * st[0] + y * st[1];
                    row[y * px..(y + 1) * px].copy_from_slice(&all[at..at + px]);
                }
                emit(t, row);
            }
        });
        // SAFETY: the handler runs synchronously, before this returns
        unsafe { logits.getBytesWithHandler(&handler) };
        drop(handler);
        let (_, _, failed) = state.into_inner();
        failed.map_or(Ok(()), Err)
    }
}

/// What an engine keeps across passes when `INKRS_COREML_ANE` or `INKRS_COREML_CPU` names a compiled body: one
/// body per worker thread and the ring of stem outputs the GPU writes and the workers read
///
/// - `INKRS_COREML_ANE`, `INKRS_COREML_ANE_N` (default 2): the fp16 `.mlmodelc` for the Neural Engine and how many
///   workers run it; two keep the Neural Engine busy while one worker copies its output
/// - `INKRS_COREML_CPU`, `INKRS_COREML_CPU_N` (default 1): the fp32 `.mlmodelc` for the CPU, only with a split
///   that gives it batches. Off by default: on S1 on an M4 Pro, 28:3:1 was 5% faster than the default at best, but
///   the CPU body slows from 12 to 19 ms a tile when the host is busy, and the fixed split then waits on it, so
///   passes ranged from 21 to 32 s
/// - `INKRS_COREML_RING` (default 8): stem outputs in flight, which bounds how far the GPU's stems run ahead
/// - `INKRS_COREML_SPLIT=a:c:g`: of every `a + c + g` batches, how many run their body on the Neural Engine, the
///   CPU and the GPU. The default, 12:0:1, was the fastest of 1:0:0 to 6:0:1 on S1 on an M4 Pro (22.6 s against
///   23.8 s with no GPU bodies and 24.7 s at 6:0:1, where the GPU falls behind). The Neural Engine sets the pace
///   at about 1.6 ms a tile
///
/// The split is fixed rather than chosen at run time so a batch always runs on the same unit: the three units
/// round differently, and a run-time choice made two passes over the same volume differ by a few 1/255 steps
pub struct Offload {
    bodies: Vec<Body>,
    ring: Vec<Buf>,
    /// Which unit runs each batch's body, repeating
    pattern: Vec<Lane>,
}

/// Where a batch's 2D body runs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    Ane,
    Cpu,
    /// The whole forward on the GPU, as without an offload
    Gpu,
}

/// Interleaves `weights[i]` batches of lane `i` per cycle as evenly as it can (smooth weighted round robin), so no
/// lane gets a long run of consecutive batches
fn interleave(weights: [usize; 3]) -> Vec<Lane> {
    let lanes = [Lane::Ane, Lane::Cpu, Lane::Gpu];
    let total: usize = weights.iter().sum();
    let mut credit = [0isize; 3];
    (0..total)
        .map(|_| {
            for (c, w) in credit.iter_mut().zip(weights) {
                *c += w as isize;
            }
            let pick = (0..3).max_by_key(|&i| (credit[i], std::cmp::Reverse(i))).expect("three lanes");
            credit[pick] -= total as isize;
            lanes[pick]
        })
        .collect()
}

impl Offload {
    /// The offload the environment asks for, or `None`
    /// `checkpoint` gives the SHA-256 of the loaded reader's `.pth`, which every body must have been built from; it
    /// runs only when the environment asks for an offload
    pub fn from_env(gpu: &Gpu, model: &Model, checkpoint: impl FnOnce() -> Result<String>) -> Result<Option<Self>> {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        if var("INKRS_COREML_ANE").is_none() && var("INKRS_COREML_CPU").is_none() {
            return Ok(None);
        }
        crate::ensure_supported!(gpu.precision() == ReaderPrecision::Fp16, "{OFFLOAD_FP16_ONLY}");
        let checkpoint = checkpoint()?;
        let count = |k: &str, default: usize| -> Result<usize> {
            var(k).map_or(Ok(default), |v| v.parse().map_err(|_| eyre!("{k}={v} is not a count")))
        };
        let mut bodies = Vec::new();
        let kinds = [
            ("INKRS_COREML_ANE", Unit::Ane, "INKRS_COREML_ANE_N", 2),
            ("INKRS_COREML_CPU", Unit::Cpu, "INKRS_COREML_CPU_N", 1),
        ];
        for (key, unit, n_key, n) in kinds {
            let Some(path) = var(key) else { continue };
            for _ in 0..count(n_key, n)? {
                bodies.push(Body::load(Path::new(&path), unit, (model.batch, model.edge.px()), &checkpoint)?);
            }
        }
        if bodies.is_empty() {
            return Ok(None);
        }

        // the packages hold reader-v2's 2D body; nothing here can check the weights match the loaded reader
        ensure!(model.edge == Edge::P128, "the Core ML offload runs reader-v2's 128 px body only");
        let ring: Vec<Buf> =
            (0..count("INKRS_COREML_RING", 8)?).map(|_| model.pooled_buf(gpu)).collect::<Result<_>>()?;
        ensure!(!ring.is_empty(), "INKRS_COREML_RING must be positive");

        let has = |u: Unit| bodies.iter().any(|b| b.unit == u);
        let split = match var("INKRS_COREML_SPLIT") {
            None => {
                ensure!(!has(Unit::Cpu), "INKRS_COREML_CPU needs an INKRS_COREML_SPLIT that gives the CPU batches");
                [12, 0, 1]
            }
            Some(v) => {
                let parts: Vec<usize> = v.split(':').map(str::parse).collect::<Result<_, _>>().unwrap_or_default();
                let split: [usize; 3] = parts.try_into().map_err(|_| eyre!("INKRS_COREML_SPLIT={v} is not a:c:g"))?;
                split
            }
        };
        ensure!(split.iter().sum::<usize>() > 0, "INKRS_COREML_SPLIT gives no batch a lane");
        ensure!(split[0] == 0 || has(Unit::Ane), "INKRS_COREML_SPLIT sends batches to the Neural Engine with no body");
        ensure!(split[1] == 0 || has(Unit::Cpu), "INKRS_COREML_SPLIT sends batches to the CPU with no body");
        // a loaded body the split never uses only costs its load and memory, and is almost surely a mistake
        ensure!(split[0] > 0 || !has(Unit::Ane), "INKRS_COREML_ANE is set but INKRS_COREML_SPLIT gives it no batches");
        ensure!(split[1] > 0 || !has(Unit::Cpu), "INKRS_COREML_CPU is set but INKRS_COREML_SPLIT gives it no batches");

        // the first prediction plans the model for its device, which takes seconds on the Neural Engine
        for body in &bodies {
            // SAFETY: a fresh ring buffer no kernel has touched
            unsafe { body.predict(ring[0].ptr(), |_, _| {}) }?;
        }
        Ok(Some(Self { bodies, ring, pattern: interleave(split) }))
    }

    /// The lane batch `b` of a pass runs on
    pub fn lane(&self, b: usize) -> Lane {
        self.pattern[b % self.pattern.len()]
    }

    /// The bodies, one per worker
    pub fn bodies(&self) -> &[Body] {
        &self.bodies
    }

    /// The stem output ring
    pub fn ring(&self) -> &[Buf] {
        &self.ring
    }
}

/// A batch whose stem the GPU has queued into ring slot `slot`, to run once `fence` passes
pub struct Job {
    pub batch: usize,
    pub slot: usize,
    pub fence: Fence,
}

/// Which batches have their logits written, for the encoding thread to know when a map's blend may run
pub struct Progress {
    state: Mutex<ProgressState>,
    cv: Condvar,
}

impl ProgressState {
    fn covers(&self, b: usize) -> bool {
        self.prefix > b.min(self.done.len().saturating_sub(1)) || self.prefix == self.done.len()
    }
}

struct ProgressState {
    done: Vec<bool>,
    /// Batches `0..prefix` are all done
    prefix: usize,
    failure: Option<String>,
}

impl Progress {
    pub fn new(batches: usize) -> Self {
        let state = ProgressState { done: vec![false; batches], prefix: 0, failure: None };
        Self { state: Mutex::new(state), cv: Condvar::new() }
    }

    /// Marks batch `b` done, recording `failure` if its worker failed
    pub fn finish(&self, b: usize, failure: Option<String>) {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        s.done[b] = true;
        if s.failure.is_none() {
            s.failure = failure;
        }
        while s.prefix < s.done.len() && s.done[s.prefix] {
            s.prefix += 1;
        }
        self.cv.notify_all();
    }

    /// Fails if any worker has failed
    pub fn check(&self) -> Result<()> {
        let s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        s.failure.as_ref().map_or(Ok(()), |f| Err(eyre!("Core ML worker: {f}")))
    }

    /// Whether batches `0..=b` are done, failing on any worker failure. Past the last batch means all of them, so a
    /// pass with no batches (an empty volume) is ready at once
    pub fn ready(&self, b: usize) -> Result<bool> {
        let s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(f) = &s.failure {
            bail!("Core ML worker: {f}");
        }
        Ok(s.covers(b))
    }

    /// Waits until batches `0..=b` are done, as `ready`
    pub fn wait(&self, b: usize) -> Result<()> {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(f) = &s.failure {
                bail!("Core ML worker: {f}");
            }
            if s.covers(b) {
                return Ok(());
            }
            s = self.cv.wait(s).unwrap_or_else(|p| p.into_inner());
        }
    }
}

/// What one worker did in a pass, for the report that says which lane set the pace
#[derive(Clone, Copy, Debug, Default)]
pub struct Ran {
    pub batches: usize,
    /// Seconds waiting for a job, for its stem's fence, and in the prediction and scatter
    pub idle_s: f64,
    pub fence_s: f64,
    pub predict_s: f64,
}

/// One worker: runs each job's body on its ring slot, hands the logits to `write(batch, tile, logits)`, frees the
/// slot and marks the batch done, until the job queue closes
pub fn work(
    body: &Body,
    ring: &[Buf],
    jobs: &Mutex<Receiver<Job>>,
    free: &Sender<usize>,
    progress: &Progress,
    write: impl Fn(usize, usize, &[f16]),
) -> Ran {
    let mut ran = Ran::default();
    loop {
        let t = Instant::now();
        let job = { jobs.lock().unwrap_or_else(|p| p.into_inner()).recv() };
        ran.idle_s += t.elapsed().as_secs_f64();
        let Ok(job) = job else { return ran };
        // a panic becomes this batch's failure, so the slot comes back and the encoding thread stops instead of
        // waiting on a batch no worker will finish
        let result = catch_unwind(AssertUnwindSafe(|| {
            let t = Instant::now();
            job.fence.wait()?;
            ran.fence_s += t.elapsed().as_secs_f64();
            let t = Instant::now();
            // SAFETY: the fence passed, so the stem that wrote this slot is done, and the slot is not handed to
            // another stem until this worker frees it below
            let r = unsafe { body.predict(ring[job.slot].ptr(), |t, logits| write(job.batch, t, logits)) };
            ran.predict_s += t.elapsed().as_secs_f64();
            r
        }));
        let failure = match result {
            Ok(Ok(())) => None,
            Ok(Err(e)) => Some(format!("{e:#}")),
            Err(_) => Some(format!("{:?} worker panicked on batch {}", body.unit, job.batch)),
        };
        free.send(job.slot).ok();
        progress.finish(job.batch, failure);
        ran.batches += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{Lane, Progress, interleave};

    // a pass over an empty volume has no batches, yet its maps still finish; their wait must not block
    #[test]
    fn progress_with_no_batches_is_ready() {
        let p = Progress::new(0);
        assert!(p.ready(0).unwrap());
        p.wait(0).unwrap();
    }

    // the blend of a map waits for every batch up to its last, in order, and a failure stops the wait
    #[test]
    fn progress_tracks_the_done_prefix_and_failures() {
        let p = Progress::new(3);
        p.finish(1, None);
        assert!(!p.ready(0).unwrap() && !p.ready(1).unwrap());
        p.finish(0, None);
        assert!(p.ready(1).unwrap() && !p.ready(2).unwrap());
        p.finish(2, Some("boom".into()));
        assert!(p.wait(2).unwrap_err().to_string().contains("boom"));
    }

    #[test]
    fn interleave_spreads_each_lane_evenly() {
        let p = interleave([28, 4, 1]);
        let count = |l: Lane| p.iter().filter(|&&x| x == l).count();
        assert_eq!((p.len(), count(Lane::Ane), count(Lane::Cpu), count(Lane::Gpu)), (33, 28, 4, 1));
        // the four CPU batches sit about a quarter of the cycle apart, not back to back
        let cpu: Vec<usize> = (0..p.len()).filter(|&i| p[i] == Lane::Cpu).collect();
        let wrap = cpu[0] + p.len() - cpu[3];
        assert!(cpu.windows(2).map(|w| w[1] - w[0]).chain([wrap]).all(|g| (7..=11).contains(&g)), "{cpu:?}");
        assert_eq!(interleave([0, 0, 1]), vec![Lane::Gpu]);
    }
}
