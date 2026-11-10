//! `inkrs._inkrs._V8in`: the v8-in family engine (v8-in, A-hn seeds 0-2), one export resident, behind
//! `inkrs.Engine` when the export's sidecar names the v8-in architecture
//!
//! The engine holds reference-counted plans, so it stays on the thread that made it (`unsendable`) and
//! keeps the GIL while it runs

use std::path::PathBuf;

use inkrs::dnn::Pick;
use inkrs::v8in::Convention;
use inkrs::v8in::engine::{Engine, Options, Output};
use inkrs::zarr::VolumeRef;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::py_err;

#[pyclass(module = "inkrs._inkrs", name = "_V8in", unsendable)]
pub struct PyV8in {
    engine: Engine,
    batch: usize,
    pick: Pick,
    max_workspace: usize,
    last: Option<String>,
}

/// `None` keeps the export's own convention
fn parse_convention(name: Option<&str>) -> PyResult<Option<Convention>> {
    name.map(|n| Convention::from_name(n).map_err(|e| PyValueError::new_err(e.to_string()))).transpose()
}

#[pymethods]
impl PyV8in {
    #[new]
    #[pyo3(signature = (checkpoint, batch_size, algo, max_workspace_mib))]
    fn new(checkpoint: PathBuf, batch_size: usize, algo: &str, max_workspace_mib: usize) -> PyResult<Self> {
        let pick = match algo {
            "find" => Pick::Find,
            "heuristic" => Pick::Heuristic,
            other => return Err(PyValueError::new_err(format!("algo {other:?}: need find or heuristic"))),
        };
        let max_workspace = max_workspace_mib << 20;
        let engine = Engine::new(&checkpoint, batch_size, pick, max_workspace).map_err(py_err)?;
        Ok(Self { engine, batch: batch_size, pick, max_workspace, last: None })
    }

    /// The export's reader name (`v8in`, `v8in-ahn-s0`, ...)
    fn reader(&self) -> String {
        self.engine.reader().to_string()
    }

    /// The export's default map convention (`release` or `306b`)
    fn convention(&self) -> String {
        self.engine.convention().name().to_string()
    }

    /// The last run's report as JSON, or None before the first run
    fn last_report(&self) -> Option<String> {
        self.last.clone()
    }

    /// The float32 map of a C-order uint8 `[Z, H, W]` volume
    #[pyo3(signature = (volume, convention_name, layer0, reverse))]
    fn run_array<'py>(
        &mut self,
        py: Python<'py>,
        volume: PyReadonlyArray3<'py, u8>,
        convention_name: Option<&str>,
        layer0: Option<usize>,
        reverse: bool,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if !volume.is_c_contiguous() {
            return Err(PyValueError::new_err("volume must be C-contiguous"));
        }
        let shape = volume.shape().to_vec();
        let data: &[u8] = volume.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let vol = VolumeRef::new(data, shape[0], shape[1], shape[2]).map_err(py_err)?;
        let out = self.engine.run(vol, &self.options(convention_name, layer0, reverse)?).map_err(py_err)?;
        self.finish(py, out)
    }

    /// The float32 map of a zarr v2 array or group (array `0`) or a uint8 `.npy` at `path`
    #[pyo3(signature = (path, convention_name, layer0, reverse))]
    fn run_path<'py>(
        &mut self,
        py: Python<'py>,
        path: PathBuf,
        convention_name: Option<&str>,
        layer0: Option<usize>,
        reverse: bool,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let out = self.engine.run_path(&path, &self.options(convention_name, layer0, reverse)?).map_err(py_err)?;
        self.finish(py, out)
    }
}

impl PyV8in {
    fn options(&self, convention_name: Option<&str>, layer0: Option<usize>, reverse: bool) -> PyResult<Options> {
        Ok(Options {
            convention: parse_convention(convention_name)?,
            layer0,
            reverse,
            batch: self.batch,
            pick: self.pick,
            max_workspace: self.max_workspace,
        })
    }

    fn finish<'py>(&mut self, py: Python<'py>, out: Output) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.last = Some(serde_json::to_string(&out.report).map_err(|e| PyValueError::new_err(e.to_string()))?);
        let a = Array2::from_shape_vec((out.height, out.width), out.map)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(a.into_pyarray(py))
    }
}
