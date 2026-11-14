//! `inkrs._inkrs`: the native half of the `inkrs` Python package
//!
//! The Python layer (`python/inkrs/__init__.py`) converts volumes and options and loads the CUDA
//! libraries; this module holds the engine. Every call that touches the GPU or the disk runs with
//! the GIL released

use std::path::PathBuf;
use std::sync::Mutex;

use color_eyre::eyre::{Report, Result, eyre};
#[cfg(not(target_os = "macos"))]
use inkrs::conv::ConvBackend;
use inkrs::dnn::Pick;
use inkrs::engine::{
    Dest, Engine, EngineOptions, PassOptions, PassOutput, PassReport, Scan, Source, Target, TtaOutput,
};
use inkrs::fast::ReaderPrecision;
use inkrs::hecate::engine::MapSpec as HecateMap;
use inkrs::zarr::{LowRes, VolumeRef};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray3, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

mod v8in;
#[cfg(target_os = "macos")]
use cuda_only::ink3d;

create_exception!(_inkrs, InkrsError, PyRuntimeError, "An inkrs engine failure; the message holds the cause chain");
create_exception!(
    _inkrs,
    UnsupportedError,
    InkrsError,
    "Input, a model or a GPU this engine does not implement; a caller can fall back to its own path"
);
create_exception!(
    _inkrs,
    CudaLibraryError,
    InkrsError,
    "The NVIDIA driver, cuDNN 9 or cuBLAS could not be loaded or used"
);

/// An eyre report as a Python exception, with every cause in the message: [`UnsupportedError`] for
/// a limit of the engine (see `inkrs::error::is_unsupported`), [`CudaLibraryError`] for any other
/// cudakit failure (driver and library loading) or a library refused on Windows (see
/// `inkrs::error::is_library_error`), else [`InkrsError`]
fn py_err(err: Report) -> PyErr {
    let text = format!("{err:#}");
    if inkrs::error::is_unsupported(&err) {
        UnsupportedError::new_err(text)
    } else if inkrs::error::is_library_error(&err) {
        CudaLibraryError::new_err(text)
    } else {
        InkrsError::new_err(text)
    }
}

/// Per window, the map and the coverage mask that came back in memory
type Maps<'py> = Vec<(Option<Bound<'py, PyArray2<u8>>>, Option<Bound<'py, PyArray2<u8>>>)>;

/// One output request from Python: a window and where its map and coverage go
type WindowSpec = (i32, i32, Option<PathBuf>, bool, Option<PathBuf>);

/// One map of an 8-fold TTA pass: first layer, layer step, and the `.npy` path or None to return it
type TtaWindow = (i32, i32, Option<PathBuf>);

/// Per window of an 8-fold TTA pass, its float32 map, or None where it was written to a file
type FloatMaps<'py> = Vec<Option<Bound<'py, PyArray2<f32>>>>;

/// The engine, behind a lock so one call runs at a time
#[pyclass(module = "inkrs._inkrs", name = "_Engine")]
struct PyEngine {
    engine: Mutex<Engine>,
    /// JSON of the last pass's report
    last: Mutex<Option<String>>,
}

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (checkpoint, device, batch_size, algo, max_workspace_mib, weights_dir, precision))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        checkpoint: PathBuf,
        device: usize,
        batch_size: usize,
        algo: &str,
        max_workspace_mib: usize,
        weights_dir: PathBuf,
        precision: &str,
    ) -> PyResult<Self> {
        let precision = ReaderPrecision::parse(precision).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("precision {precision:?}: need fp16 or fp32"))
        })?;
        let pick = match algo {
            "find" => Pick::Find,
            "heuristic" => Pick::Heuristic,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("algo {other:?}: need find or heuristic")));
            }
        };
        // the conv backend has no Python option; `INKRS_CONV` overrides the default. The Metal backend has
        // one conv implementation, and its engine refuses `INKRS_CONV` as unsupported when it opens
        #[cfg(not(target_os = "macos"))]
        let conv = ConvBackend::from_env().map_err(py_err)?;
        let opts = EngineOptions {
            weights_dir,
            device,
            batch: batch_size,
            precision,
            pick,
            max_workspace: max_workspace_mib << 20,
            #[cfg(not(target_os = "macos"))]
            conv,
        };
        let engine = py.detach(|| Engine::open(&checkpoint, &opts)).map_err(py_err)?;
        Ok(Self { engine: Mutex::new(engine), last: Mutex::new(None) })
    }

    /// The load report as JSON
    fn load_report(&self) -> PyResult<String> {
        let engine = self.lock()?;
        let l = &engine.load;
        Ok(serde_json::json!({
            "reader": l.reader,
            "cudnn": l.cudnn,
            "compute_capability": l.compute_capability,
            "ptx_tier": l.ptx_tier,
            "kernels": l.kernels,
            "conv": l.conv,
            "conv_reason": l.conv_reason,
            "load_weights_s": l.load_weights_s,
            "open_gpu_s": l.open_gpu_s,
            "plan_s": l.plan_s,
            "algo_cache_hits": l.algo_cache_hits,
            "algo_cache": l.algo_cache,
            "algos": l.algos,
            "schema": l.schema,
        })
        .to_string())
    }

    /// The loaded reader's tile edge in pixels: 128, or 384 for Reader V3
    fn patch(&self) -> PyResult<usize> {
        Ok(self.lock()?.edge().px())
    }

    /// The device's (free, total) memory in bytes
    fn memory_info(&self, py: Python<'_>) -> PyResult<(usize, usize)> {
        py.detach(|| self.lock_engine()?.memory_info()).map_err(py_err)
    }

    /// The last pass's report as JSON, or None before the first call
    fn last_report(&self) -> PyResult<Option<String>> {
        Ok(self.last.lock().map_err(|_| InkrsError::new_err("report lock poisoned"))?.clone())
    }

    /// One pass over the zarr at `path`; see [`scan`] for the occupancy arguments
    #[pyo3(signature = (path, windows, stride, occupancy_mode, occupancy, occupancy_level, writers))]
    #[allow(clippy::too_many_arguments)]
    fn run_zarr<'py>(
        &self,
        py: Python<'py>,
        path: PathBuf,
        windows: Vec<WindowSpec>,
        stride: usize,
        occupancy_mode: &str,
        occupancy: Option<PathBuf>,
        occupancy_level: Option<PyReadonlyArrayDyn<'py, u8>>,
        writers: usize,
    ) -> PyResult<Maps<'py>> {
        let level = level_bytes(occupancy_level.as_ref())?;
        let out = py
            .detach(|| -> Result<PassOutput> {
                let mut engine = self.lock_engine()?;
                // a given scan is scaled to level 0, whose size only the metadata knows before the read
                let given = match occupancy_mode {
                    "given" => {
                        let (height, width) = zarr_hw(&path)?;
                        Some(lowres(occupancy.as_deref(), level.as_ref(), height, width)?)
                    }
                    _ => None,
                };
                let scan = match (occupancy_mode, &given) {
                    ("villa", _) => Scan::Villa,
                    ("none", _) => Scan::Given(None),
                    ("given", Some(g)) => Scan::Given(Some(g)),
                    (other, _) => return Err(eyre!("occupancy mode {other:?}")),
                };
                engine.run_pass(
                    Source::Zarr { path: &path, scan },
                    &targets(&windows),
                    &PassOptions { stride, writers },
                )
            })
            .map_err(py_err)?;
        self.finish(py, out)
    }

    /// One pass over an in-memory `[Z, H, W]` uint8 volume, read in place
    #[pyo3(signature = (volume, windows, stride, occupancy_mode, occupancy, occupancy_level, writers, label))]
    #[allow(clippy::too_many_arguments)]
    fn run_array<'py>(
        &self,
        py: Python<'py>,
        volume: PyReadonlyArray3<'py, u8>,
        windows: Vec<WindowSpec>,
        stride: usize,
        occupancy_mode: &str,
        occupancy: Option<PathBuf>,
        occupancy_level: Option<PyReadonlyArrayDyn<'py, u8>>,
        writers: usize,
        label: String,
    ) -> PyResult<Maps<'py>> {
        // numpy's as_slice accepts Fortran order too; the engine needs C order
        if !volume.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err("volume must be C-contiguous"));
        }
        let shape = volume.shape().to_vec();
        let data: &[u8] = volume.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let level = level_bytes(occupancy_level.as_ref())?;
        let out = py
            .detach(|| -> Result<PassOutput> {
                let mut engine = self.lock_engine()?;
                let vol = VolumeRef::new(data, shape[0], shape[1], shape[2])?;
                // an in-memory volume has no pyramid of its own, so villa's choice is no scan
                let given = match occupancy_mode {
                    "given" => Some(lowres(occupancy.as_deref(), level.as_ref(), vol.height, vol.width)?),
                    "villa" | "none" => None,
                    other => return Err(eyre!("occupancy mode {other:?}")),
                };
                engine.run_pass(
                    Source::Array { volume: vol, scan: given.as_ref(), label },
                    &targets(&windows),
                    &PassOptions { stride, writers },
                )
            })
            .map_err(py_err)?;
        self.finish(py, out)
    }

    /// One 8-fold TTA pass over the zarr at `path` ([`inkrs::tta`]); no occupancy scan, as the recipe
    #[pyo3(signature = (path, windows, stride))]
    fn run_zarr_tta<'py>(
        &self,
        py: Python<'py>,
        path: PathBuf,
        windows: Vec<TtaWindow>,
        stride: usize,
    ) -> PyResult<FloatMaps<'py>> {
        let out = py
            .detach(|| -> Result<TtaOutput> {
                let src = Source::Zarr { path: &path, scan: Scan::Given(None) };
                self.lock_engine()?.run_tta(src, &tta_targets(&windows), stride)
            })
            .map_err(py_err)?;
        self.finish_tta(py, out)
    }

    /// One 8-fold TTA pass over an in-memory `[Z, H, W]` uint8 volume, read in place
    #[pyo3(signature = (volume, windows, stride, label))]
    fn run_array_tta<'py>(
        &self,
        py: Python<'py>,
        volume: PyReadonlyArray3<'py, u8>,
        windows: Vec<TtaWindow>,
        stride: usize,
        label: String,
    ) -> PyResult<FloatMaps<'py>> {
        if !volume.is_c_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err("volume must be C-contiguous"));
        }
        let shape = volume.shape().to_vec();
        let data: &[u8] = volume.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let out = py
            .detach(|| -> Result<TtaOutput> {
                let vol = VolumeRef::new(data, shape[0], shape[1], shape[2])?;
                let src = Source::Array { volume: vol, scan: None, label };
                self.lock_engine()?.run_tta(src, &tta_targets(&windows), stride)
            })
            .map_err(py_err)?;
        self.finish_tta(py, out)
    }
}

impl PyEngine {
    /// Stores the report and hands the in-memory float32 maps to numpy without copying
    fn finish_tta<'py>(&self, py: Python<'py>, out: TtaOutput) -> PyResult<FloatMaps<'py>> {
        let [_, height, width] = out.report.shape;
        let json = report_json(&out.report)?;
        *self.last.lock().map_err(|_| InkrsError::new_err("report lock poisoned"))? = Some(json);
        out.maps
            .into_iter()
            .map(|m| {
                m.map(|v| {
                    Array2::from_shape_vec((height, width), v)
                        .map(|a| a.into_pyarray(py))
                        .map_err(|e| InkrsError::new_err(e.to_string()))
                })
                .transpose()
            })
            .collect()
    }

    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, Engine>> {
        self.engine.lock().map_err(|_| InkrsError::new_err("engine lock poisoned by an earlier panic"))
    }

    fn lock_engine(&self) -> Result<std::sync::MutexGuard<'_, Engine>> {
        self.engine.lock().map_err(|_| eyre!("engine lock poisoned by an earlier panic"))
    }

    /// Stores the report and hands the in-memory maps to numpy without copying
    fn finish<'py>(&self, py: Python<'py>, out: PassOutput) -> PyResult<Maps<'py>> {
        let [_, height, width] = out.report.shape;
        let json = report_json(&out.report)?;
        *self.last.lock().map_err(|_| InkrsError::new_err("report lock poisoned"))? = Some(json);
        let array = |v: Option<Vec<u8>>| -> PyResult<Option<Bound<'py, PyArray2<u8>>>> {
            v.map(|v| {
                Array2::from_shape_vec((height, width), v)
                    .map(|a| a.into_pyarray(py))
                    .map_err(|e| InkrsError::new_err(e.to_string()))
            })
            .transpose()
        };
        out.maps.into_iter().zip(out.coverage).map(|(m, c)| Ok((array(m)?, array(c)?))).collect()
    }
}

fn report_json(rep: &PassReport) -> PyResult<String> {
    serde_json::to_string(rep).map_err(|e| InkrsError::new_err(e.to_string()))
}

/// Each window's [`Target`]: a path writes villa's TIFF there, none returns the image
fn targets(windows: &[WindowSpec]) -> Vec<Target> {
    let dest = |path: &Option<PathBuf>| path.clone().map_or(Dest::Memory, Dest::File);
    windows
        .iter()
        .map(|(first, step, map, coverage, coverage_path)| Target {
            first: *first,
            step: *step,
            map: dest(map),
            coverage: coverage.then(|| dest(coverage_path)),
        })
        .collect()
}

/// Each TTA window's [`Target`]: a path writes the float32 `.npy` there, none returns the map
fn tta_targets(windows: &[TtaWindow]) -> Vec<Target> {
    windows
        .iter()
        .map(|(first, step, map)| Target {
            first: *first,
            step: *step,
            map: map.clone().map_or(Dest::Memory, Dest::File),
            coverage: None,
        })
        .collect()
}

/// A pyramid level from Python: its shape and C-order bytes
struct Level {
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn level_bytes(level: Option<&PyReadonlyArrayDyn<'_, u8>>) -> PyResult<Option<Level>> {
    let Some(level) = level else { return Ok(None) };
    let shape = level.shape().to_vec();
    if !matches!(shape.len(), 2 | 3) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "occupancy level of shape {shape:?}: need 2D or 3D"
        )));
    }
    // copied: a pyramid level is small, and the copy lets it cross the GIL release
    let data = level.as_array().iter().copied().collect();
    Ok(Some(Level { shape, data }))
}

/// A given occupancy scan, from a zarr group's pyramid or a level in memory, for a
/// `height x width` level 0
///
/// The Python layer passes an occupancy mode: "villa" (a zarr input's own scan, or none for an
/// in-memory volume), "none", or "given" with exactly one of the group path and the level
fn lowres(group: Option<&std::path::Path>, level: Option<&Level>, height: usize, width: usize) -> Result<LowRes> {
    match (group, level) {
        (Some(group), None) => LowRes::scan(group, height, width)?
            .ok_or_else(|| eyre!("occupancy {}: a bare array, not a group with a pyramid", group.display())),
        (None, Some(l)) => LowRes::from_level("array".into(), &l.shape, &l.data, height, width),
        _ => Err(eyre!("a given occupancy needs exactly one of a zarr group and a level")),
    }
}

/// Level 0's height and width from a zarr's metadata
fn zarr_hw(path: &std::path::Path) -> Result<(usize, usize)> {
    let meta = [path.join(".zarray"), path.join("0/.zarray")]
        .into_iter()
        .find(|p| p.exists())
        .ok_or_else(|| eyre!("{}: no zarr array", path.display()))?;
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&meta)?)?;
    let dim = |i: usize| v.get("shape").and_then(|s| s.get(i)).and_then(|d| d.as_u64()).map(|d| d as usize);
    match (dim(1), dim(2)) {
        (Some(h), Some(w)) => Ok((h, w)),
        _ => Err(eyre!("{}: no 3D shape", meta.display())),
    }
}

/// Writes villa's TIFF for a uint8 `[H, W]` map
#[pyfunction]
fn write_tiff(py: Python<'_>, path: PathBuf, image: numpy::PyReadonlyArray2<'_, u8>) -> PyResult<()> {
    if !image.is_c_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err("image must be C-contiguous"));
    }
    let [height, width] = [image.shape()[0], image.shape()[1]];
    let data = image.as_slice().map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    py.detach(|| inkrs::tiff::write(&path, data, height, width)).map_err(py_err)
}

/// Every model's `--fast` preset as JSON ([`inkrs::fast::presets`])
#[pyfunction]
fn fast_presets() -> PyResult<String> {
    serde_json::to_string(&inkrs::fast::presets()).map_err(|e| InkrsError::new_err(e.to_string()))
}

/// The reader's tile stride, resolved as the CLI resolves it ([`inkrs::fast::reader_stride`]); a bad
/// combination or a reader with no fast preset is a `ValueError`
#[pyfunction]
#[pyo3(signature = (reader, patch, fast, stride, overlap))]
fn reader_stride(
    reader: Option<&str>,
    patch: usize,
    fast: bool,
    stride: Option<usize>,
    overlap: Option<f64>,
) -> PyResult<usize> {
    inkrs::fast::reader_stride(reader, patch, fast, stride, overlap)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e:#}")))
}

/// `inkrs hecate`: writes each `(reverse, png)` map of the volume at `input` and returns the run report
/// as JSON. Settings resolve as the CLI's do ([`inkrs::fast::hecate_settings`]); a bad combination is a
/// `ValueError`
#[pyfunction]
#[pyo3(signature = (
    input, checkpoint, maps, spacing_um, *, fast, stride, overlap, precision, tf32, batch_size, algo,
    max_workspace_mib, weights_dir, device, check_only
))]
#[allow(clippy::too_many_arguments)]
fn hecate(
    py: Python<'_>,
    input: PathBuf,
    checkpoint: PathBuf,
    maps: Vec<(bool, PathBuf)>,
    spacing_um: f64,
    fast: bool,
    stride: Option<usize>,
    overlap: Option<f64>,
    precision: Option<&str>,
    tf32: bool,
    batch_size: Option<usize>,
    algo: &str,
    max_workspace_mib: usize,
    weights_dir: PathBuf,
    device: usize,
    check_only: bool,
) -> PyResult<String> {
    use inkrs::hecate::Precision;

    let value_err = |m: String| pyo3::exceptions::PyValueError::new_err(m);
    let precision = match (precision, tf32) {
        (None, true) => None,
        (None | Some("fp32"), false) => Some(Precision::Fp32 { tf32: false }),
        (Some("fp32"), true) => Some(Precision::Fp32 { tf32: true }),
        (Some("bf16"), true) => Some(Precision::Bf16),
        (Some("bf16"), false) => return Err(value_err("tf32=False applies to fp32 only".into())),
        (Some(other), _) => return Err(value_err(format!("precision {other:?}: need fp32 or bf16"))),
    };
    let pick = match algo {
        "find" => Pick::Find,
        "heuristic" => Pick::Heuristic,
        other => return Err(value_err(format!("algo {other:?}: need find or heuristic"))),
    };
    let settings = inkrs::fast::hecate_settings(fast, stride, overlap, precision, batch_size)
        .map_err(|e| value_err(format!("{e:#}")))?;
    let specs: Vec<HecateMap> = maps.into_iter().map(|(reverse, png)| HecateMap { reverse, png }).collect();
    let opts = inkrs::hecate::engine::Options {
        stride: settings.stride,
        fast,
        batch: settings.batch,
        precision: settings.precision,
        pick,
        max_workspace: max_workspace_mib << 20,
        spacing_um,
        checkpoint: checkpoint.clone(),
        weights_dir,
        device,
        started: std::time::Instant::now(),
    };
    // everything the engine refuses before the GPU, as ValueError; with check_only that is all, so the
    // module can refuse a bad request before it loads any CUDA library
    inkrs::hecate::engine::check_request(&specs, &opts).map_err(|e| value_err(format!("{e:#}")))?;
    if check_only {
        return Ok(String::new());
    }
    let rep = py.detach(|| inkrs::hecate::engine::run(&input, &checkpoint, &specs, &opts)).map_err(py_err)?;
    serde_json::to_string(&rep).map_err(|e| InkrsError::new_err(e.to_string()))
}

#[cfg(not(target_os = "macos"))]
/// One 3D U-Net ink run over a zarr volume into a zarr prediction (`inkrs ink3d`); returns the
/// run report as JSON
#[pyfunction]
#[pyo3(signature = (checkpoint, input, output, device, bbox, overlap, chunk, max_accum_gb, algo, overwrite, coverage))]
#[allow(clippy::too_many_arguments)]
fn ink3d(
    py: Python<'_>,
    checkpoint: PathBuf,
    input: PathBuf,
    output: PathBuf,
    device: usize,
    bbox: Option<[usize; 6]>,
    overlap: f64,
    chunk: usize,
    max_accum_gb: f64,
    algo: &str,
    overwrite: bool,
    coverage: bool,
) -> PyResult<String> {
    let pick = match algo {
        "find" => Pick::Find,
        "heuristic" => Pick::Heuristic,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("algo {other:?}: need find or heuristic")));
        }
    };
    let opts = inkrs::unet3d::engine::Options {
        overlap,
        bbox,
        chunk,
        pick,
        max_accum_gb,
        overwrite,
        forward_bench: 0,
        coverage,
        device,
    };
    let report = py.detach(|| inkrs::unet3d::engine::run(&input, &checkpoint, &output, &opts)).map_err(py_err)?;
    serde_json::to_string(&report).map_err(|e| InkrsError::new_err(e.to_string()))
}

/// The embedded PTX: its targets and ISA versions, as `(target, version)` pairs
#[cfg(not(target_os = "macos"))]
#[pyfunction]
fn ptx_targets() -> Vec<(String, String)> {
    inkrs::gpu::ptx_targets()
}

/// The embedded PTX: none on macOS, where the kernels are Metal shaders compiled at load
#[cfg(target_os = "macos")]
#[pyfunction]
fn ptx_targets() -> Vec<(String, String)> {
    Vec::new()
}

/// The embedded cubins: the PTX target each was compiled from and the GPU family it runs on, as
/// `(target, arch)` pairs
#[cfg(not(target_os = "macos"))]
#[pyfunction]
fn cubin_targets() -> Vec<(String, String)> {
    inkrs::gpu::cubin_targets()
}

/// The embedded cubins: none on macOS, where the kernels are Metal shaders compiled at load
#[cfg(target_os = "macos")]
#[pyfunction]
fn cubin_targets() -> Vec<(String, String)> {
    Vec::new()
}

/// The 3D ink engine runs on CUDA only: on macOS its entry point raises [`UnsupportedError`], so the
/// Python package imports the same names on every platform. v8-in runs on both backends
#[cfg(target_os = "macos")]
mod cuda_only {
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyTuple};

    use super::UnsupportedError;

    #[pyfunction]
    #[pyo3(signature = (*_args, **_kwargs))]
    pub fn ink3d(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        Err(UnsupportedError::new_err("ink3d runs on CUDA only; the Metal backend has no 3D ink engine"))
    }
}

#[pymodule]
fn _inkrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<v8in::PyV8in>()?;
    m.add_function(wrap_pyfunction!(write_tiff, m)?)?;
    m.add_function(wrap_pyfunction!(ptx_targets, m)?)?;
    m.add_function(wrap_pyfunction!(cubin_targets, m)?)?;
    m.add_function(wrap_pyfunction!(fast_presets, m)?)?;
    m.add_function(wrap_pyfunction!(reader_stride, m)?)?;
    m.add_function(wrap_pyfunction!(hecate, m)?)?;
    m.add_function(wrap_pyfunction!(ink3d, m)?)?;
    m.add("InkrsError", m.py().get_type::<InkrsError>())?;
    m.add("CudaLibraryError", m.py().get_type::<CudaLibraryError>())?;
    m.add("UnsupportedError", m.py().get_type::<UnsupportedError>())?;
    // the package's one version: pyproject.toml takes it from this crate, and CI checks the crate matches inkrs
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
