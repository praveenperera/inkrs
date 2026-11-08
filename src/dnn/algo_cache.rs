//! An on-disk cache of cuDNN's timed algorithm choice ([`Pick::Find`](super::Pick::Find))
//!
//! `cudnnFindConvolutionForwardAlgorithmEx` times every algorithm of every plan on each load,
//! about 0.6 s for a villa reader. The ranking it returns is kept per plan in
//! `<cache dir>/cudnn-algos/`, keyed on everything that can change it: the inkrs version, the PTX
//! tier, the GPU's name and compute capability, the driver release and cuDNN version, and the plan's
//! full descriptor and workspace limit. A later load on the same setup replays that ranking
//! instead of timing it again, so it runs the same algorithms as the run that wrote it. Fused
//! convs ([`super::fused`]) keep their timed engine choice in the same directory, keyed on the
//! same setup and the fused graph.
//!
//! The cache never fails a load: an unreadable, corrupt or mismatched entry, or a cached
//! algorithm cuDNN no longer accepts, falls back to a fresh find, which overwrites the entry, and
//! a failed write is ignored. [`ENV`]`=0` turns the cache off
//!
//! The driver release comes from `/proc/driver/nvidia/version`, else from NVML, which containers
//! (Modal, Docker with the NVIDIA runtime) expose when they hide that file. Where neither names it,
//! the cache stays off, the first plan says so once on stderr, and every run report carries the
//! reason ([`status`])

use std::ffi::{CStr, OsString, c_char, c_int, c_uint};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{AlgoPerf, ConvShape, Dnn, Dtype, dnn};
use crate::gpu::Gpu;

/// Set to `0`, `off` or `false` to plan every load with a fresh timed find and leave the cache alone
pub const ENV: &str = "INKRS_ALGO_CACHE";

/// Bumped whenever the entry format or the key's meaning changes, so old entries miss
const SCHEMA: u32 = 1;

/// Every forward algorithm, to map a stored number back without trusting it
const ALGOS: [dnn::cudnnConvolutionFwdAlgo_t; 8] = {
    use dnn::cudnnConvolutionFwdAlgo_t as A;
    [
        A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        A::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        A::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        A::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        A::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    ]
};

/// Every math type, likewise
const MATHS: [dnn::cudnnMathType_t; 4] = {
    use dnn::cudnnMathType_t as M;
    [M::CUDNN_DEFAULT_MATH, M::CUDNN_TENSOR_OP_MATH, M::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION, M::CUDNN_FMA_MATH]
};

/// The cache directory: `$INKRS_CACHE_DIR`, else `$XDG_CACHE_HOME/inkrs`, else `~/.cache/inkrs`,
/// as the Python package resolves it, then `cudnn-algos/`. None when the cache is off or no
/// directory can be named
pub fn dir() -> Option<PathBuf> {
    root(&|name| std::env::var_os(name)).map(|r| r.join("cudnn-algos"))
}

/// The directory to give the CUDA driver as `CUDA_CACHE_PATH` for its compiled-PTX cache:
/// `nv-compute/` in the same cache root as [`dir`], read through `env`. None when the user set
/// `CUDA_CACHE_PATH` themselves (to any value), the cache is off, or no root can be named; the
/// driver's own default applies then
///
/// The fused convs' kernels are generated as sm80 PTX that the driver compiles for the GPU on
/// first use, about 9 s on an RTX 5090. The driver's default cache is shared by every CUDA program
/// and evicts its oldest entries at 1 GiB, so other programs can push those kernels out and bring
/// the compile back; a directory of our own keeps them. Only the CLI sets it, before its first CUDA
/// call: the driver reads the variable once, at initialisation
pub fn cuda_cache_path(env: &dyn Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    if env("CUDA_CACHE_PATH").is_some() {
        return None;
    }
    root(env).map(|r| r.join("nv-compute"))
}

/// Whether [`ENV`], as `env` reads it, turns the cache off
fn disabled(env: &dyn Fn(&str) -> Option<OsString>) -> bool {
    env(ENV)
        .and_then(|v| v.into_string().ok())
        .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false"))
}

/// The cache root as `env` names it, or None when [`ENV`] turns the cache off or nothing names one
fn root(env: &dyn Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    if disabled(env) {
        return None;
    }

    crate::engine::cache_root_in(env)
}

/// Where the driver release in the cache key came from
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DriverSource {
    /// The kernel module's version line in `/proc/driver/nvidia/version`
    Proc,
    /// NVML's `nvmlSystemGetDriverVersion`
    Nvml,
}

/// The algorithm cache's state in this process, as the run reports carry it
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Status {
    /// `--algo find` plans load and store entries in `dir`, keyed on the driver release `driver`
    /// read from `source`
    On { dir: PathBuf, source: DriverSource, driver: String },
    /// [`ENV`] turned the cache off
    Disabled,
    /// No cache directory could be named: no `INKRS_CACHE_DIR`, `XDG_CACHE_HOME` or `HOME`
    NoDirectory,
    /// No source named the driver release, so every load times cuDNN afresh; `tried` says why each
    /// source failed
    NoDriverRelease { tried: Vec<String> },
    /// Keyed as [`Status::On`], but a write into `dir` failed in this process (`error`, the first
    /// failure), so entries this run timed are not kept and later runs time them again
    Unwritable { dir: PathBuf, source: DriverSource, driver: String, error: String },
}

/// The cache's state in this process, whether or not this run planned with `--algo find`
pub fn status() -> Status {
    status_in(&|name| std::env::var_os(name), || driver_release().clone(), WRITE_ERROR.get().cloned())
}

/// The first failed entry write in this process, for [`status`]
static WRITE_ERROR: OnceLock<String> = OnceLock::new();

/// Keeps `error` as the process's first failed write and notes it on stderr; later failures stay
/// quiet, since the first already says the cache is not kept
fn note_write_error(error: String) {
    let mut first = false;
    WRITE_ERROR.get_or_init(|| {
        first = true;
        error
    });
    if first {
        eprintln!(
            "inkrs: cuDNN algorithm cache not kept: {}; later runs time cuDNN afresh",
            WRITE_ERROR.get().map_or("", String::as_str)
        );
    }
}

/// [`status`] with the environment read through `env`, the driver release from `release`, which
/// runs only when the cache is otherwise on, and the process's first failed write
fn status_in(
    env: &dyn Fn(&str) -> Option<OsString>,
    release: impl FnOnce() -> Result<Release, Vec<String>>,
    write_error: Option<String>,
) -> Status {
    if disabled(env) {
        return Status::Disabled;
    }
    let Some(dir) = root(env).map(|r| r.join("cudnn-algos")) else {
        return Status::NoDirectory;
    };

    match (release(), write_error) {
        (Ok(r), None) => Status::On { dir, source: r.source, driver: r.release },
        (Ok(r), Some(error)) => Status::Unwritable { dir, source: r.source, driver: r.release, error },
        (Err(tried), _) => Status::NoDriverRelease { tried },
    }
}

/// A driver release and where it came from
#[derive(Clone, Debug, PartialEq, Eq)]
struct Release {
    source: DriverSource,
    release: String,
}

/// One way to read the driver release: where it reads, and a reader that returns the release or
/// why not
type Reader<'a> = (DriverSource, &'a dyn Fn() -> Result<String, String>);

/// The first release `readers` name, in order, or every reader's failure
fn resolve(readers: &[Reader<'_>]) -> Result<Release, Vec<String>> {
    let mut tried = Vec::new();
    for (source, read) in readers {
        match read() {
            Ok(release) => return Ok(Release { source: *source, release }),
            Err(why) => tried.push(why),
        }
    }
    Err(tried)
}

/// The driver release, resolved once per process: `/proc` first, so a host that has it keeps the
/// keys (and so the entries) it wrote before NVML was a source, then NVML. cuDriverGetVersion is
/// no source: it names only the CUDA API level, which many driver releases share, and a release
/// can change cuDNN's timings and so its choice. Windows has no `/proc`, so there NVML names it
fn driver_release() -> &'static Result<Release, Vec<String>> {
    static RELEASE: OnceLock<Result<Release, Vec<String>>> = OnceLock::new();
    RELEASE.get_or_init(|| resolve(&[(DriverSource::Proc, &proc_release), (DriverSource::Nvml, &nvml_release)]))
}

/// The driver release for a plan's key. Its first failure prints one note on stderr, since the
/// cache then stays off for the whole process; only planning calls it, so a run that never plans
/// with `--algo find` stays quiet
fn planning_release() -> Option<&'static Release> {
    static NOTED: OnceLock<()> = OnceLock::new();
    match driver_release() {
        Ok(r) => Some(r),
        Err(tried) => {
            NOTED.get_or_init(|| {
                eprintln!(
                    "inkrs: cuDNN algorithm cache off: no driver release to key it on ({}); every load times cuDNN afresh",
                    tried.join("; ")
                );
            });
            None
        }
    }
}

/// The kernel module's version line, such as `NVRM version: NVIDIA UNIX x86_64 Kernel Module
/// 580.95.05 ...`, with its whitespace collapsed
fn proc_release() -> Result<String, String> {
    const PATH: &str = "/proc/driver/nvidia/version";
    let text = std::fs::read_to_string(PATH).map_err(|e| format!("{PATH}: {e}"))?;
    let line = text.lines().next().unwrap_or_default().split_whitespace().collect::<Vec<_>>().join(" ");
    if line.is_empty() {
        return Err(format!("{PATH}: empty"));
    }
    Ok(line)
}

/// NVML's library names, tried in order: the versioned name the driver installs first
#[cfg(not(windows))]
const NVML_LIBS: [&str; 2] = ["libnvidia-ml.so.1", "libnvidia-ml.so"];

/// NVML from the driver's install: the first of [`NVML_LIBS`] the dynamic loader finds, which never
/// searches the current directory on Linux
#[cfg(not(windows))]
fn open_nvml() -> Result<libloading::Library, String> {
    // SAFETY: loading NVML runs only its own initialisers
    NVML_LIBS
        .iter()
        .find_map(|name| unsafe { libloading::Library::new(*name) }.ok())
        .ok_or_else(|| format!("NVML: none of {NVML_LIBS:?} loads"))
}

/// NVML from System32 by full path, where the driver installs it ([`crate::dll::open_nvml`]); a
/// bare name would search the current directory and `PATH`
#[cfg(windows)]
fn open_nvml() -> Result<libloading::Library, String> {
    crate::dll::open_nvml()
}

/// NVML's driver release, such as `580.95.05`. NVML ships with the driver, and the NVIDIA container
/// runtime mounts it where it hides `/proc/driver/nvidia/version`. It loads at run time like the
/// CUDA libraries, so a machine without it still builds and runs
fn nvml_release() -> Result<String, String> {
    type Init = unsafe extern "C" fn() -> c_int;
    type Version = unsafe extern "C" fn(*mut c_char, c_uint) -> c_int;
    type Shutdown = unsafe extern "C" fn() -> c_int;

    let lib = open_nvml()?;
    // SAFETY: these are NVML's signatures (nvml.h), whose nvmlReturn_t is a C enum with 0 for
    // success, and `lib` outlives every call below
    let (init, version, shutdown) = unsafe {
        (
            symbol::<Init>(&lib, b"nvmlInit_v2\0")?,
            symbol::<Version>(&lib, b"nvmlSystemGetDriverVersion\0")?,
            symbol::<Shutdown>(&lib, b"nvmlShutdown\0")?,
        )
    };

    // SAFETY: no arguments; NVML counts inits, so another NVML user in the process is unaffected
    let rc = unsafe { init() };
    if rc != 0 {
        return Err(format!("NVML: nvmlInit_v2 returned {rc}"));
    }

    // NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE is 80, and the call writes a NUL-terminated string
    let mut buf = [0 as c_char; 96];
    // SAFETY: the buffer holds the length passed, and NVML stays initialised until the shutdown
    let rc = unsafe { version(buf.as_mut_ptr(), buf.len() as c_uint) };
    // SAFETY: balances the init above
    unsafe { shutdown() };
    if rc != 0 {
        return Err(format!("NVML: nvmlSystemGetDriverVersion returned {rc}"));
    }

    // the last byte stays NUL whatever NVML wrote, so the string ends inside the buffer
    buf[buf.len() - 1] = 0;
    // SAFETY: NUL-terminated within `buf`, as just ensured
    let release = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().trim().to_string();
    if release.is_empty() {
        return Err("NVML: empty driver version".into());
    }
    Ok(release)
}

/// The function `name` (NUL-terminated) in `lib`, as a plain function pointer
///
/// # Safety
/// `T` must be the symbol's true signature, and the pointer must not be called after `lib` drops
unsafe fn symbol<T: Copy>(lib: &libloading::Library, name: &[u8]) -> Result<T, String> {
    // SAFETY: the caller vouches for `T`
    unsafe { lib.get::<T>(name) }
        .map(|s| *s)
        .map_err(|e| format!("NVML: {}: {e}", String::from_utf8_lossy(name.strip_suffix(b"\0").unwrap_or(name))))
}

/// The driver part of a key. The `/proc` line stands alone, as it did before NVML was a source,
/// so existing entries still hit; NVML's release carries a prefix, so an entry written under one
/// source never passes for the other's
fn driver_key(r: &Release) -> String {
    match r.source {
        DriverSource::Proc => r.release.clone(),
        DriverSource::Nvml => format!("nvml {}", r.release),
    }
}

/// One plan's place in the cache: its full key and the file that key hashes to
pub(super) struct Entry {
    key: String,
    path: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct Stored {
    schema: u32,
    key: String,
    ranking: Vec<StoredPerf>,
}

/// A fused conv's timed engine choice: where it sat in cuDNN's candidate list for the same graph
/// on the same setup, and the engine tag it had there, which a replay checks before trusting the
/// position
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct FusedChoice {
    pub position: usize,
    pub engine: String,
    pub ms: f32,
    pub candidates: usize,
}

#[derive(Serialize, Deserialize)]
struct StoredFused {
    schema: u32,
    key: String,
    choice: FusedChoice,
}

#[derive(Serialize, Deserialize)]
struct StoredPerf {
    algo: u32,
    ok: bool,
    ms: f32,
    workspace: usize,
    math: u32,
    deterministic: bool,
}

/// Everything about this setup that can change cuDNN's choice, as the start of a key: the entry
/// schema, the inkrs version, the PTX tier, the GPU's name and compute capability, the driver
/// release and cuDNN version. None when the device or driver can't be described
fn setup_key(gpu: &Gpu) -> Option<String> {
    let ctx = gpu.inner.context();
    let name = ctx.name().ok()?;
    let driver = driver_key(planning_release()?);
    Some(format!(
        "schema {SCHEMA}; inkrs {}; tier {}; gpu {name}; sm {}; driver {driver}; cudnn {}",
        env!("CARGO_PKG_VERSION"),
        gpu.inner.ptx_tier(),
        gpu.inner.compute_capability(),
        Dnn::version(),
    ))
}

impl Entry {
    /// The entry for a plan of `shape` and `dtype` under `max_workspace`, or None when the cache
    /// is off or the device can't be described
    pub(super) fn new(gpu: &Gpu, shape: &ConvShape, dtype: Dtype, max_workspace: usize) -> Option<Self> {
        let dir = dir()?;
        let mut key = setup_key(gpu)?;
        // every input to cuDNN's choice; the layout, dilation, mode and compute type are fixed by
        // ConvPlan::typed but named here so that changing them there must change the key too
        let _ = write!(
            key,
            "; conv{}d NHWC cross-correlation dilation 1 compute FLOAT; n {} c {} k {} spatial {:?} kernel {:?} \
             pad {:?} stride {:?}; data {:?}; math {:?}; max workspace {max_workspace}",
            shape.spatial.len(),
            shape.n,
            shape.c,
            shape.k,
            shape.spatial,
            shape.kernel,
            shape.pad,
            shape.stride,
            dtype.data(),
            dtype.math(),
        );
        Some(Self::at(dir, key))
    }

    /// The entry for a fused conv's engine choice. `graph` names everything the caller's graph and
    /// selection depend on beyond this setup, so a change to either must change it
    pub(super) fn fused(gpu: &Gpu, graph: &str) -> Option<Self> {
        let dir = dir()?;
        let key = format!("{}; fused {graph}", setup_key(gpu)?);
        Some(Self::at(dir, key))
    }

    fn at(dir: PathBuf, key: String) -> Self {
        let hash = Sha256::digest(key.as_bytes());
        let file: String = hash.iter().map(|b| format!("{b:02x}")).collect();
        Self { key, path: dir.join(format!("{file}.json")) }
    }

    /// The stored ranking, or None for a missing, unreadable, corrupt or mismatched entry
    pub(super) fn load(&self) -> Option<Vec<AlgoPerf>> {
        let text = std::fs::read(&self.path).ok()?;
        let stored: Stored = serde_json::from_slice(&text).ok()?;
        // the file name is a hash, so the full key guards against a collision or a stray file
        if stored.schema != SCHEMA || stored.key != self.key || stored.ranking.is_empty() {
            return None;
        }

        stored
            .ranking
            .iter()
            .map(|p| {
                Some(AlgoPerf {
                    algo: *ALGOS.iter().find(|a| **a as u32 == p.algo)?,
                    // the plan only tells supported from unsupported, so that is all an entry keeps
                    status: if p.ok {
                        dnn::cudnnStatus_t::CUDNN_STATUS_SUCCESS
                    } else {
                        dnn::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
                    },
                    ms: p.ms,
                    workspace: p.workspace,
                    math: *MATHS.iter().find(|m| **m as u32 == p.math)?,
                    deterministic: p.deterministic,
                })
            })
            .collect()
    }

    /// Writes `ranking` atomically (a temporary file, then a rename); a failure only costs the
    /// next load a fresh find
    pub(super) fn store(&self, ranking: &[AlgoPerf]) {
        let stored = Stored {
            schema: SCHEMA,
            key: self.key.clone(),
            ranking: ranking
                .iter()
                .map(|p| StoredPerf {
                    algo: p.algo as u32,
                    ok: p.status == dnn::cudnnStatus_t::CUDNN_STATUS_SUCCESS,
                    ms: p.ms,
                    workspace: p.workspace,
                    math: p.math as u32,
                    deterministic: p.deterministic,
                })
                .collect(),
        };
        self.write(&stored);
    }

    /// The stored fused engine choice, or None for a missing, unreadable, corrupt or mismatched entry
    pub(super) fn load_fused(&self) -> Option<FusedChoice> {
        let text = std::fs::read(&self.path).ok()?;
        let stored: StoredFused = serde_json::from_slice(&text).ok()?;
        (stored.schema == SCHEMA && stored.key == self.key).then_some(stored.choice)
    }

    /// Writes a fused engine choice, as [`Entry::store`] does a ranking
    pub(super) fn store_fused(&self, choice: &FusedChoice) {
        self.write(&StoredFused { schema: SCHEMA, key: self.key.clone(), choice: choice.clone() });
    }

    /// Writes `value` atomically (a temporary file, then a rename). A failure never fails the load,
    /// but the first one in the process is noted once on stderr and turns [`status`] to
    /// [`Status::Unwritable`], so an unwritable cache directory is never silent
    fn write(&self, value: &impl Serialize) {
        let written = (|| -> std::io::Result<()> {
            let dir = self.path.parent().ok_or(std::io::ErrorKind::NotFound)?;
            std::fs::create_dir_all(dir)?;
            let bytes = serde_json::to_vec(value)?;
            // unique per process and call, so concurrent writers never share a temporary file
            let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
            let tmp = dir.join(format!(".{}.{}.{nanos}.tmp", std::process::id(), self.file_stem()));
            std::fs::write(&tmp, bytes)?;
            std::fs::rename(&tmp, &self.path).inspect_err(|_| {
                let _ = std::fs::remove_file(&tmp);
            })
        })();
        if let Err(e) = written {
            note_write_error(format!("{}: {e}", self.path.display()));
        }
    }

    fn file_stem(&self) -> String {
        self.path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(dir: &std::path::Path) -> Entry {
        Entry { key: "schema 1; test key".into(), path: dir.join("abc.json") }
    }

    fn perf(algo: dnn::cudnnConvolutionFwdAlgo_t, ok: bool, workspace: usize) -> AlgoPerf {
        AlgoPerf {
            algo,
            status: if ok {
                dnn::cudnnStatus_t::CUDNN_STATUS_SUCCESS
            } else {
                dnn::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED
            },
            ms: 0.25,
            workspace,
            math: dnn::cudnnMathType_t::CUDNN_TENSOR_OP_MATH,
            deterministic: true,
        }
    }

    #[test]
    fn a_stored_ranking_loads_back_and_a_bad_entry_misses() {
        let dir = std::env::temp_dir().join(format!("inkrs-algo-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let e = entry(&dir);
        assert!(e.load().is_none(), "no file yet");

        use dnn::cudnnConvolutionFwdAlgo_t as A;
        let ranking = vec![
            perf(A::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED, false, 0),
            perf(A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, true, 4096),
        ];
        e.store(&ranking);
        let back = e.load().expect("stored entry loads");
        assert_eq!(back.len(), 2);
        assert_eq!(back[1].algo, A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
        assert_eq!(back[1].workspace, 4096);
        assert_eq!(back[0].status, dnn::cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED);
        assert_eq!(back[1].math, dnn::cudnnMathType_t::CUDNN_TENSOR_OP_MATH);

        // another key hashing to the same file, a corrupt file and an unknown algorithm all miss
        let other = Entry { key: "schema 1; another key".into(), path: e.path.clone() };
        assert!(other.load().is_none());
        std::fs::write(&e.path, b"{ not json").unwrap();
        assert!(e.load().is_none());
        let text = serde_json::json!({"schema": SCHEMA, "key": e.key, "ranking": [
            {"algo": 99, "ok": true, "ms": 0.1, "workspace": 0, "math": 0, "deterministic": true}
        ]});
        std::fs::write(&e.path, text.to_string()).unwrap();
        assert!(e.load().is_none());

        // a rewrite replaces the bad entry and leaves no temporary file behind
        e.store(&ranking);
        assert!(e.load().is_some());
        let names: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|f| f.unwrap().file_name()).collect();
        assert_eq!(names, vec![std::ffi::OsString::from("abc.json")]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_fused_choice_loads_back_and_a_ranking_entry_does_not_pass_for_one() {
        let dir = std::env::temp_dir().join(format!("inkrs-fused-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let e = entry(&dir);
        assert!(e.load_fused().is_none(), "no file yet");

        let choice =
            FusedChoice { position: 3, engine: "engine 7 knobs [(1, 2)] notes [0]".into(), ms: 0.5, candidates: 6 };
        e.store_fused(&choice);
        assert_eq!(e.load_fused(), Some(choice.clone()));
        assert!(e.load().is_none(), "a fused entry is no ranking");
        let other = Entry { key: "schema 1; another key".into(), path: e.path.clone() };
        assert!(other.load_fused().is_none());

        use dnn::cudnnConvolutionFwdAlgo_t as A;
        e.store(&[perf(A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, true, 0)]);
        assert!(e.load_fused().is_none(), "a ranking is no fused choice");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_driver_cache_follows_the_inkrs_cache_root_unless_the_user_set_one() {
        let pick = |vars: &[(&str, &str)]| {
            let env = |name: &str| vars.iter().find(|(k, _)| *k == name).map(|(_, v)| OsString::from(v));
            cuda_cache_path(&env)
        };
        // Windows sets no HOME, so its per-user default is %LOCALAPPDATA%\inkrs
        let (home, under_home) = if cfg!(windows) {
            (("LOCALAPPDATA", "/h"), "/h/inkrs/nv-compute")
        } else {
            (("HOME", "/h"), "/h/.cache/inkrs/nv-compute")
        };
        let all = [("INKRS_CACHE_DIR", "/c"), ("XDG_CACHE_HOME", "/x"), home];
        assert_eq!(pick(&all), Some(PathBuf::from("/c/nv-compute")));
        assert_eq!(pick(&all[1..]), Some(PathBuf::from("/x/inkrs/nv-compute")));
        assert_eq!(pick(&all[2..]), Some(PathBuf::from(under_home)));
        assert_eq!(pick(&[("INKRS_CACHE_DIR", ""), home]), Some(PathBuf::from(under_home)));
        assert_eq!(pick(&[]), None, "no root, so the driver's default");

        // the user's own setting wins, even an empty one, and so does turning the inkrs cache off
        assert_eq!(pick(&[("CUDA_CACHE_PATH", "/mine"), all[0]]), None);
        assert_eq!(pick(&[("CUDA_CACHE_PATH", ""), all[0]]), None);
        assert_eq!(pick(&[(ENV, "off"), all[0]]), None);
        assert_eq!(pick(&[(ENV, "1"), all[0]]), Some(PathBuf::from("/c/nv-compute")));
    }

    const LINE: &str = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64 595.99.02 Release Build";

    #[test]
    fn the_driver_release_falls_back_from_proc_to_nvml_and_reports_every_failure() {
        use std::cell::Cell;

        let nvml_calls = Cell::new(0);
        let proc_ok = || Ok::<_, String>(LINE.to_string());
        let proc_gone = || Err::<String, _>("/proc/driver/nvidia/version: No such file or directory".to_string());
        let nvml_ok = || {
            nvml_calls.set(nvml_calls.get() + 1);
            Ok::<_, String>("595.99.02".to_string())
        };
        let nvml_gone = || Err::<String, _>("NVML: none of [\"libnvidia-ml.so.1\"] loads".to_string());

        // /proc wins where it reads, and NVML is never loaded then
        let r = resolve(&[(DriverSource::Proc, &proc_ok), (DriverSource::Nvml, &nvml_ok)]).unwrap();
        assert_eq!(r, Release { source: DriverSource::Proc, release: LINE.into() });
        assert_eq!(nvml_calls.get(), 0);

        // a container without the /proc file keys on NVML's release
        let r = resolve(&[(DriverSource::Proc, &proc_gone), (DriverSource::Nvml, &nvml_ok)]).unwrap();
        assert_eq!(r, Release { source: DriverSource::Nvml, release: "595.99.02".into() });
        assert_eq!(nvml_calls.get(), 1);

        // neither: every reason, in order, for the stderr note and the report
        let tried = resolve(&[(DriverSource::Proc, &proc_gone), (DriverSource::Nvml, &nvml_gone)]).unwrap_err();
        assert_eq!(tried.len(), 2);
        assert!(tried[0].starts_with("/proc/driver/nvidia/version:") && tried[1].starts_with("NVML:"), "{tried:?}");
    }

    #[test]
    fn a_proc_key_is_unchanged_and_an_nvml_key_never_passes_for_one() {
        // the /proc line alone is what keys held before NVML was a source, so warm caches still hit
        let from_proc = driver_key(&Release { source: DriverSource::Proc, release: LINE.into() });
        assert_eq!(from_proc, LINE);
        let from_nvml = driver_key(&Release { source: DriverSource::Nvml, release: "595.99.02".into() });
        assert_eq!(from_nvml, "nvml 595.99.02");
        assert_ne!(from_proc, from_nvml);
    }

    #[test]
    fn the_status_names_why_the_cache_is_off() {
        let env_of = |vars: &'static [(&'static str, &'static str)]| {
            move |name: &str| vars.iter().find(|(k, _)| *k == name).map(|(_, v)| OsString::from(v))
        };
        let nvml = || Ok(Release { source: DriverSource::Nvml, release: "595.99.02".into() });
        let never =
            || -> Result<Release, Vec<String>> { panic!("the driver release is read only when the cache is wanted") };

        assert_eq!(status_in(&env_of(&[(ENV, "0"), ("HOME", "/h")]), never, None), Status::Disabled);
        assert_eq!(status_in(&env_of(&[]), never, None), Status::NoDirectory);
        assert_eq!(
            status_in(&env_of(&[("INKRS_CACHE_DIR", "/c")]), nvml, None),
            Status::On { dir: PathBuf::from("/c/cudnn-algos"), source: DriverSource::Nvml, driver: "595.99.02".into() }
        );
        // INKRS_CACHE_DIR names the root on every platform; Windows has no HOME default
        let off = status_in(&env_of(&[("INKRS_CACHE_DIR", "/c")]), || Err(vec!["a".into(), "b".into()]), None);
        assert_eq!(off, Status::NoDriverRelease { tried: vec!["a".into(), "b".into()] });

        // the report field's JSON names the state, so a script can check it without parsing prose
        let json = serde_json::to_value(&off).unwrap();
        assert_eq!(json, serde_json::json!({"state": "no_driver_release", "tried": ["a", "b"]}));
        // a failed write keeps the key's description and names the failure
        let unwritable =
            status_in(&env_of(&[("INKRS_CACHE_DIR", "/c")]), nvml, Some("/c/cudnn-algos/x.json: denied".into()));
        assert_eq!(
            unwritable,
            Status::Unwritable {
                dir: PathBuf::from("/c/cudnn-algos"),
                source: DriverSource::Nvml,
                driver: "595.99.02".into(),
                error: "/c/cudnn-algos/x.json: denied".into()
            }
        );

        let json = serde_json::to_value(status_in(&env_of(&[("INKRS_CACHE_DIR", "/c")]), nvml, None)).unwrap();
        assert_eq!(json["state"], "on");
        assert_eq!(json["source"], "nvml");
    }

    #[test]
    fn a_failed_write_is_kept_for_the_status() {
        // a regular file where the entry's directory should be, so create_dir_all fails
        let blocker = std::env::temp_dir().join(format!("inkrs-algo-cache-blocker-{}", std::process::id()));
        std::fs::write(&blocker, b"").unwrap();
        let e = Entry { key: "schema 1; unwritable".into(), path: blocker.join("cudnn-algos").join("abc.json") };
        use dnn::cudnnConvolutionFwdAlgo_t as A;
        e.store(&[perf(A::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, true, 0)]);
        let error = WRITE_ERROR.get().expect("the failed write is recorded");
        assert!(error.contains("inkrs-algo-cache-blocker"), "{error}");
        let _ = std::fs::remove_file(&blocker);
    }

    /// Needs an NVIDIA driver: `/proc` and NVML name the same release, so either source keys one
    /// driver the same way (run with `--ignored` on a GPU host)
    #[test]
    #[ignore = "needs an NVIDIA driver"]
    fn proc_and_nvml_name_the_same_release() {
        let line = proc_release().expect("/proc/driver/nvidia/version");
        let nvml = nvml_release().expect("NVML");
        assert!(line.split_whitespace().any(|w| w == nvml), "/proc says {line:?}, NVML says {nvml:?}");
        eprintln!("proc: {line}\nnvml: {nvml}");
    }
}
