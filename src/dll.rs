//! Windows: where the NVIDIA driver, cuBLAS, cuDNN and NVRTC load from
//!
//! cudakit opens every CUDA library through cudarc 0.19.10, which loads each by bare name with Windows' default
//! DLL search, trying a list of candidate names in order (`cudarc::get_lib_name_candidates`): `cuda.dll` before
//! `nvcuda.dll`, and `cudnn.dll`, `cudnn64.dll` and eight more before `cudnn64_9.dll`. That search looks in the
//! current directory and in relative `PATH` entries, so a DLL planted in a folder inkrs runs from would load and run
//! its code. cuDNN then loads its sublibraries, and NVRTC for its runtime-compiled engines, by bare name with the
//! same search. Linux has no such step: its dynamic loader never searches the current directory
//!
//! [`pin`] runs before cudakit opens anything. For each library, in cudarc's order of names, it loads the first
//! name it finds in a trusted place by full path, which a bare-name load of that name then reuses wherever it came
//! from, and holds a reference to it for the life of the process. It refuses to go on, with [`UntrustedLibrary`],
//! when a name that would be tried first sits in the current directory or a relative `PATH` entry, since cudarc
//! would open that file instead, and when a name cudarc tries first would open a second copy of a library already
//! loaded under a later name. The trusted places follow sam3rs's order:
//!
//! - the driver, `nvcuda.dll`, only from System32, by full path, where the driver installs it; a driver module
//!   already loaded from anywhere else is refused;
//! - every other library from a module already loaded under the name (such as the Python package's pip copy), the
//!   executable's directory, System32 or an `AddDllDirectory` directory, then each absolute `PATH` entry in order,
//!   then `%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin`. A load by full path also searches the DLL's own directory
//!   for its imports, so `cublas64_12.dll` binds the `cublasLt64_12.dll` beside it;
//! - cuDNN's sublibraries and NVRTC's builtins from the directory their parent loaded from.
//!
//! A library found nowhere is left for cudakit to report as missing, as on Linux
//!
//! The folders are checked once, when [`pin`] runs. The CLI covers what comes after: [`restrict_search`], first
//! thing in its `main`, takes the current directory and `PATH` out of the process's default search, so a file
//! written there later, under a name cudarc tries before the pinned one, is never opened. The executable's own
//! directory stays in the search, as [`pin`] trusts it: run from there, or with `PATH` naming it, a file written into
//! it later is opened, but whoever can write there can already replace the executable. The Python package can't
//! set a search mode for the user's whole interpreter, so there it relies on [`pin`] and the loaded-copy checks
//! alone: a file written into the current directory between [`pin`] and cudarc's load would still be opened

use std::env;
use std::ffi::{OsStr, OsString};
use std::os::windows::ffi::OsStringExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use color_eyre::eyre::{Report, Result, eyre};
use libloading::os::windows::{
    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, Library,
};
use windows_sys::Win32::System::LibraryLoader::SetDefaultDllDirectories;

use crate::error::UntrustedLibrary;

/// cuDNN 9's sublibraries, dependencies first, as the 9.28 Windows wheel ships them; older releases lack some, and
/// those are skipped. `cudnn64_9.dll` and `cudnn_graph64_9.dll` load them by bare name when they need them
pub const CUDNN_SUBLIBRARIES: [&str; 9] = [
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_engines_tensor_ir64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_ext64_9.dll",
];

/// CUDA 12's NVRTC, which cuDNN's runtime-compiled engines load by bare name for the fused Hecate convs
pub const NVRTC: &str = "nvrtc64_120_0.dll";

/// NVRTC loads its builtins, `nvrtc-builtins64_<major><minor>.dll`, from beside itself by bare name
const NVRTC_BUILTINS_PREFIX: &str = "nvrtc-builtins64_";

/// References to the modules this file chose, held for the life of the process so that each stays loaded, and a
/// later load by bare name reuses it, even if whoever loaded it first unloads it; cudarc's own handles do the same
static PINNED: Mutex<Vec<Library>> = Mutex::new(Vec::new());

/// Loads the driver, cuBLAS, cuDNN with its sublibraries and NVRTC from trusted places before cudakit opens them;
/// fails with [`UntrustedLibrary`] when cudarc or cuDNN would open a file from the current directory or a relative
/// `PATH` entry instead. Cheap to call again: a library already loaded is not searched for twice
pub fn pin() -> Result<()> {
    let search = Search::from_env()?;

    search.driver()?;
    // cuBLAS before cuDNN, whose sublibraries import cuBLAS by name
    search.library("cublas")?;
    if let Some(dir) = search.library("cudnn")? {
        search.companions(&dir, CUDNN_SUBLIBRARIES.iter().map(|name| name.to_string()))?;
    }

    // NVRTC is optional: cuDNN only falls back to other engines without it
    if let Some(dir) = search.named(NVRTC)? {
        let builtins = files_with_prefix(&dir, NVRTC_BUILTINS_PREFIX);
        search.companions(&dir, builtins)?;
    }

    // a builtins file is never found by name in the current directory unless it is listed there
    search.refuse_prefix(NVRTC_BUILTINS_PREFIX)
}

/// Takes the current directory and `PATH` out of this process's default DLL search, leaving the executable's
/// directory, System32 and `AddDllDirectory` directories, so that a load by bare name (cudarc's, cuDNN's) after
/// [`pin`] never opens a file written to either after [`pin`] checked them. The CLI calls it first thing in `main`;
/// a library must not, since the mode is the whole process's
pub fn restrict_search() -> Result<()> {
    // SAFETY: takes a flags value and touches no memory of ours
    if unsafe { SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) } == 0 {
        let err = std::io::Error::last_os_error();
        return Err(eyre!(
            "could not take the current directory and PATH out of the DLL search (SetDefaultDllDirectories): {err}"
        ));
    }

    Ok(())
}

/// NVML, for the cuDNN plan cache's driver-release key (`dnn::algo_cache`), since Windows has no
/// `/proc/driver/nvidia/version`
///
/// The driver installs `nvml.dll` in System32, so it loads from there by full path: a bare name would search the
/// current directory and `PATH`, or reuse an `nvml.dll` loaded from another directory
pub fn open_nvml() -> Result<libloading::Library, String> {
    let file = system32().map_err(|e| format!("NVML: {e}"))?.join("nvml.dll");
    // SAFETY: loading NVML runs only its own initialisers
    unsafe { Library::load_with_flags(&file, LOAD_LIBRARY_SEARCH_SYSTEM32) }
        .map(libloading::Library::from)
        .map_err(|e| format!("NVML: {}: {e}", file.display()))
}

/// The directories one search looks in, read once from the environment
struct Search {
    /// The current directory, and relative or empty `PATH` entries, resolved against it: the places only the
    /// default search reaches, where anyone who controls the folder inkrs runs from can put a file
    untrusted: Vec<PathBuf>,
    /// Absolute `PATH` entries in order, then `%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin`
    trusted: Vec<PathBuf>,
    /// The executable's directory, which the default search tries before System32
    exe_dir: Option<PathBuf>,
    system32: PathBuf,
}

impl Search {
    fn from_env() -> Result<Self> {
        let cwd = env::current_dir().map_err(|e| eyre!("read the current directory: {e}"))?;
        let path = env::var_os("PATH").unwrap_or_default();
        Ok(Self::new(&cwd, &path, env::var_os("CUDA_PATH").as_deref(), system32()?))
    }

    /// A search from these values rather than the process's, so a test never changes the working directory or
    /// `PATH` that other tests read
    fn new(cwd: &Path, path: &OsStr, cuda_path: Option<&OsStr>, system32: PathBuf) -> Self {
        let (trusted, untrusted) = split_dirs(cwd, path, cuda_path);
        let exe_dir = env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_path_buf));
        Self { untrusted, trusted, exe_dir, system32 }
    }

    /// The driver: the first of cudarc's names for it, which must be in System32, failing with a cudakit library
    /// error that names nvcuda.dll when there is none
    fn driver(&self) -> Result<()> {
        self.driver_from(&candidates(&["cuda", "nvcuda"]))
    }

    /// [`Search::driver`] over `names`, in order. A name already loaded must have come from System32, since
    /// cudarc's load by name would reuse it; one not loaded is loaded from System32 by full path; and an earlier name
    /// the default search would find anywhere else is refused
    fn driver_from(&self, names: &[String]) -> Result<()> {
        for name in names {
            if let Some((lib, file)) = loaded(name) {
                if !in_dir(&file, &self.system32) {
                    return Err(UntrustedLibrary::loaded(name, &file, "the CUDA driver").into());
                }
                keep(lib);
                return Ok(());
            }

            let file = self.system32.join(name);
            // SAFETY: the driver's initialisers have no caller preconditions
            if file.is_file()
                && let Ok(lib) = unsafe { Library::load_with_flags(&file, LOAD_LIBRARY_SEARCH_SYSTEM32) }
            {
                keep(lib);
                return Ok(());
            }

            // a copy anywhere the default search looks would be opened as the driver, even from an absolute PATH
            // entry or the executable's directory
            let elsewhere = self.untrusted.iter().chain(&self.trusted).chain(&self.exe_dir);
            if let Some(found) = elsewhere.map(|dir| dir.join(name)).find(|file| file.is_file()) {
                return Err(UntrustedLibrary::new(name, &found, "the CUDA driver").into());
            }
        }

        // cudakit would name the Linux driver, libcuda; the class stays cudakit's, so callers see a library error
        let missing = cudakit::Error::LibraryUnavailable { library: cudakit::Library::Driver };
        Err(Report::new(missing).wrap_err(format!(
            "no NVIDIA driver: nvcuda.dll is not in {}; inkrs needs an NVIDIA GPU and its driver",
            self.system32.display()
        )))
    }

    /// The first of cudarc's names for `lib` found in a trusted place, loaded; returns its directory. None when no
    /// name is found, which cudakit reports
    fn library(&self, lib: &str) -> Result<Option<PathBuf>> {
        self.first_of(&candidates(&[lib]))
    }

    /// [`Search::library`] over `names`, in cudarc's order. When a later name is already loaded, by a host such as
    /// the Python package, an earlier name the default search can find is refused: cudarc would open it as a second
    /// copy and use that one
    fn first_of(&self, names: &[String]) -> Result<Option<PathBuf>> {
        let held = names.iter().enumerate().find_map(|(at, name)| Some((at, loaded(name)?.1)));
        if let Some((at, file)) = held
            && let Some(earlier) = names[..at].iter().find_map(|name| self.findable(name))
        {
            return Err(UntrustedLibrary::shadowed(&earlier, &file).into());
        }

        for name in names {
            if let Some(dir) = self.named(name)? {
                return Ok(Some(dir));
            }
        }
        Ok(None)
    }

    /// `name` loaded from a trusted place, returning its directory; None when no trusted place has it. Fails when
    /// the default search would find it in an untrusted one
    fn named(&self, name: &str) -> Result<Option<PathBuf>> {
        if let Some(dir) = self.open_trusted(name) {
            return Ok(Some(dir));
        }
        self.refuse(name)?;
        Ok(None)
    }

    /// Opens `name` by sam3rs's order: a module already loaded under the name, the bare name from the executable's
    /// directory, System32 or an `AddDllDirectory` directory, then each trusted directory by full path
    fn open_trusted(&self, name: &str) -> Option<PathBuf> {
        if let Some(dir) = hold_loaded(name) {
            return Some(dir);
        }

        // SAFETY (all loads here): CUDA libraries' initialisers have no caller preconditions
        if let Ok(lib) = unsafe { Library::load_with_flags(name, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) } {
            let (lib, file) = with_file(lib);
            keep(lib);
            return file?.parent().map(Path::to_path_buf);
        }

        // the DLL's own directory resolves its imports, again without the current directory
        let flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
        for dir in &self.trusted {
            let file = dir.join(name);
            if !file.is_file() {
                continue;
            }
            // a broken copy, such as one for another architecture, leaves later entries to try
            if let Ok(lib) = unsafe { Library::load_with_flags(&file, flags) } {
                keep(lib);
                return Some(dir.clone());
            }
        }
        None
    }

    /// Loads each of `names` from `dir`, the directory of the library that loads them by bare name, so that load
    /// reuses it; refuses one found only in an untrusted place
    fn companions(&self, dir: &Path, names: impl IntoIterator<Item = String>) -> Result<()> {
        let flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
        for name in names {
            if hold_loaded(&name).is_some() {
                continue;
            }
            let file = dir.join(&name);
            // SAFETY: as in open_trusted
            if file.is_file()
                && let Ok(lib) = unsafe { Library::load_with_flags(&file, flags) }
            {
                keep(lib);
                continue;
            }
            self.refuse(&name)?;
        }
        Ok(())
    }

    /// Fails when the default search would find `name` in the current directory or a relative `PATH` entry
    fn refuse(&self, name: &str) -> Result<()> {
        match self.untrusted.iter().map(|dir| dir.join(name)).find(|file| file.is_file()) {
            Some(file) => Err(UntrustedLibrary::new(name, &file, "a CUDA library").into()),
            None => Ok(()),
        }
    }

    /// [`Search::refuse`] for every file in an untrusted directory whose name starts with `prefix`, unless a module
    /// of that name is already loaded
    fn refuse_prefix(&self, prefix: &str) -> Result<()> {
        for dir in &self.untrusted {
            for name in files_with_prefix(dir, prefix) {
                if loaded(&name).is_none() {
                    return Err(UntrustedLibrary::new(&name, &dir.join(&name), "a CUDA library").into());
                }
            }
        }
        Ok(())
    }

    /// Where the default search would find `name` as a file, if anywhere this search knows: the executable's
    /// directory, System32, the untrusted directories, then the trusted ones
    fn findable(&self, name: &str) -> Option<PathBuf> {
        let dirs = self.exe_dir.iter().chain([&self.system32]).chain(&self.untrusted).chain(&self.trusted);
        dirs.map(|dir| dir.join(name)).find(|file| file.is_file())
    }
}

/// cudarc's candidate names for each of `libs`, in the order its loader tries them
fn candidates(libs: &[&str]) -> Vec<String> {
    libs.iter().flat_map(|lib| cudarc::get_lib_name_candidates(lib)).collect()
}

/// Splits the directories the default search looks in after System32 into trusted ones (absolute `PATH` entries,
/// then `%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin`) and untrusted ones (the current directory, and relative or
/// empty `PATH` entries resolved against it)
fn split_dirs(cwd: &Path, path: &OsStr, cuda_path: Option<&OsStr>) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut trusted = Vec::new();
    let mut untrusted = vec![cwd.to_path_buf()];
    for dir in env::split_paths(path) {
        if dir.is_absolute() {
            trusted.push(dir);
        } else {
            untrusted.push(cwd.join(dir));
        }
    }

    if let Some(root) = cuda_path.map(PathBuf::from).filter(|root| root.is_absolute()) {
        let bin = root.join("bin");
        trusted.extend([bin.join("x64"), bin]);
    }
    (trusted, untrusted)
}

/// The names of the files in `dir` that start with `prefix` (ASCII case-insensitively, as Windows matches names)
/// and end in `.dll`, sorted
fn files_with_prefix(dir: &Path, prefix: &str) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .filter(|name| {
            let lower = name.to_ascii_lowercase();
            lower.starts_with(prefix) && lower.ends_with(".dll")
        })
        .collect();
    names.sort();
    names
}

fn keep(lib: Library) {
    PINNED.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(lib);
}

/// The module loaded under `name`, if one is: a new reference to it, which keeps it loaded while held, and its file
fn loaded(name: &str) -> Option<(Library, PathBuf)> {
    let (lib, file) = with_file(Library::open_already_loaded(name).ok()?);
    Some((lib, file?))
}

/// The directory of the module loaded under `name`, holding a reference to it for the life of the process
fn hold_loaded(name: &str) -> Option<PathBuf> {
    let (lib, file) = loaded(name)?;
    keep(lib);
    file.parent().map(Path::to_path_buf)
}

/// Whether `file` is directly in `dir`. Windows paths compare without regard to case, and System32 is reported as
/// `C:\WINDOWS\system32` by one API and `C:\Windows\System32` by another
fn in_dir(file: &Path, dir: &Path) -> bool {
    let fold = |path: &Path| path.as_os_str().to_string_lossy().trim_end_matches('\\').to_lowercase();
    file.parent().is_some_and(|parent| fold(parent) == fold(dir))
}

/// `lib` with the file it was loaded from, which is None when Windows can't say
fn with_file(lib: Library) -> (Library, Option<PathBuf>) {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleFileNameW(module: isize, name: *mut u16, size: u32) -> u32;
    }

    let raw = lib.into_raw();
    // long paths need more than MAX_PATH; 32767 UTF-16 units is Windows' limit for any path
    let mut buffer = vec![0u16; 32768];
    // SAFETY: the handle is live, and the buffer is writable for the length passed
    let len = unsafe { GetModuleFileNameW(raw, buffer.as_mut_ptr(), buffer.len() as u32) };
    // SAFETY: the handle came from into_raw above, so this takes back the same reference
    let lib = unsafe { Library::from_raw(raw) };
    let file = usize::try_from(len)
        .ok()
        .filter(|len| (1..buffer.len()).contains(len))
        .map(|len| PathBuf::from(OsString::from_wide(&buffer[..len])));
    (lib, file)
}

/// The System32 directory, from `GetSystemDirectoryW` rather than the environment, which a caller can change
fn system32() -> Result<PathBuf> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetSystemDirectoryW(buffer: *mut u16, size: u32) -> u32;
    }

    // the system directory is never longer than MAX_PATH
    let mut buffer = [0u16; 260];
    // SAFETY: the buffer is writable for the length passed
    let len = unsafe { GetSystemDirectoryW(buffer.as_mut_ptr(), buffer.len() as u32) };
    let len = usize::try_from(len)
        .ok()
        .filter(|len| (1..buffer.len()).contains(len))
        .ok_or_else(|| eyre!("GetSystemDirectoryW failed"))?;
    Ok(OsString::from_wide(&buffer[..len]).into())
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::ffi::OsString;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use libloading::os::windows::Library;

    use super::{Search, candidates, in_dir, loaded, restrict_search, split_dirs, system32};

    /// The probe's name, for the child half of [`the_process_search_skips_the_current_directory_and_relative_path`]
    const CHILD_PROBE: &str = "INKRS_DLL_CHILD_PROBE";

    /// Set for the child half of [`after_startup_a_dll_planted_in_the_current_directory_is_not_opened`]
    const CHILD_STARTUP: &str = "INKRS_DLL_CHILD_STARTUP";

    /// The temporary directory in long form, as Windows reports a loaded module's file; a runner's `TEMP` can be an
    /// 8.3 short name such as `C:\Users\RUNNER~1\...`
    fn temp() -> PathBuf {
        crate::engine::canonicalize(&env::temp_dir()).unwrap()
    }

    /// A copy of a system DLL under a name no Windows directory has, in a directory of its own; `tag` keeps tests
    /// that run at once apart, since a module loaded under a name is reused from then on
    fn probe(tag: &str) -> (PathBuf, String) {
        let id = std::process::id();
        let name = format!("inkrs_probe_{id}_{tag}.dll");
        let dir = temp().join(format!("inkrs-dll-{id}-{tag}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(system32().unwrap().join("version.dll"), dir.join(&name)).unwrap();
        (dir, name)
    }

    fn no_path() -> OsString {
        env::join_paths::<[&str; 0], _>([]).unwrap()
    }

    #[test]
    fn relative_and_empty_path_entries_are_untrusted() {
        let cwd = Path::new(r"C:\work");
        let path = env::join_paths([r"C:\cuda\bin", "rel", r"D:\tools"]).unwrap();
        let (trusted, untrusted) = split_dirs(cwd, &path, Some(r"C:\toolkit".as_ref()));
        let want: Vec<PathBuf> =
            [r"C:\cuda\bin", r"D:\tools", r"C:\toolkit\bin\x64", r"C:\toolkit\bin"].iter().map(PathBuf::from).collect();
        assert_eq!(trusted, want);
        assert_eq!(untrusted, [PathBuf::from(r"C:\work"), PathBuf::from(r"C:\work\rel")]);
    }

    #[test]
    fn cudnn9_is_among_cudarcs_names_and_cuda_comes_before_nvcuda() {
        let cudnn = candidates(&["cudnn"]);
        assert!(cudnn.iter().any(|name| name == "cudnn64_9.dll"), "{cudnn:?}");
        let driver = candidates(&["cuda", "nvcuda"]);
        let at = |name: &str| driver.iter().position(|n| n == name).unwrap();
        assert!(at("cuda.dll") < at("nvcuda.dll"));
    }

    #[test]
    fn paths_compare_without_case() {
        assert!(in_dir(Path::new(r"C:\Windows\System32\nvcuda.dll"), Path::new(r"C:\WINDOWS\system32")));
        assert!(!in_dir(Path::new(r"C:\Windows\System32\sub\nvcuda.dll"), Path::new(r"C:\Windows\System32")));
    }

    /// A DLL in the current directory or behind a relative PATH entry is refused, and one in an absolute PATH entry
    /// loads. Searches are built from explicit values, so the process's working directory and PATH stay as they are
    #[test]
    fn the_search_refuses_the_current_directory_and_loads_from_path() {
        let (dir, name) = probe("search");
        let system = system32().unwrap();

        let in_cwd = Search::new(&dir, &no_path(), None, system.clone());
        assert!(in_cwd.open_trusted(&name).is_none(), "loaded from the current directory");
        assert!(in_cwd.named(&name).is_err(), "a copy in the current directory was not refused");

        // the relative entry names the probe's directory from its parent
        let relative = Search::new(dir.parent().unwrap(), dir.file_name().unwrap(), None, system.clone());
        assert!(relative.named(&name).is_err(), "a relative PATH entry was not refused");

        // last, since a module loaded under the name is reused from then on
        let absolute = Search::new(&env::temp_dir().join("elsewhere"), dir.as_os_str(), None, system);
        assert_eq!(absolute.named(&name).unwrap(), Some(dir.clone()));
    }

    /// The process's own working directory and PATH, in a child process run from the probe's directory and then with
    /// a relative PATH entry naming it: the loads themselves must not search either, which a search built from
    /// explicit values can't show
    #[test]
    fn the_process_search_skips_the_current_directory_and_relative_path() {
        let (dir, name) = probe("process");
        let system = system32().unwrap();
        let relative = env::join_paths([PathBuf::from(dir.file_name().unwrap()), system.clone()]).unwrap();
        let runs = [(dir.as_path(), system.into_os_string()), (dir.parent().unwrap(), relative)];

        for (cwd, path) in runs {
            let out = Command::new(env::current_exe().unwrap())
                .args(["dll::tests::child_searches_the_process_environment", "--exact", "--ignored"])
                .current_dir(cwd)
                .env("PATH", &path)
                .env(CHILD_PROBE, &name)
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(
                out.status.success() && stdout.contains("1 passed"),
                "from {} with PATH={path:?}:\n{stdout}\n{stderr}",
                cwd.display()
            );
        }
    }

    #[test]
    #[ignore = "run by the_process_search_skips_the_current_directory_and_relative_path in a child process"]
    fn child_searches_the_process_environment() {
        let name = env::var(CHILD_PROBE).expect("run from its parent test");
        let search = Search::from_env().unwrap();
        assert!(search.named(&name).is_err(), "the probe was not refused");
        assert!(loaded(&name).is_none(), "a load searched the current directory or a relative PATH entry");
    }

    /// What the CLI's `main` does first: after [`restrict_search`], a DLL written into the current directory under
    /// a name cudarc tries first is not opened by a load by bare name, which is how cudarc loads. In a child process
    /// run from an empty directory, since the search mode is the whole process's
    #[test]
    fn after_startup_a_dll_planted_in_the_current_directory_is_not_opened() {
        let dir = temp().join(format!("inkrs-dll-{}-startup", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = Command::new(env::current_exe().unwrap())
            .args(["dll::tests::child_restricts_the_search", "--exact", "--ignored"])
            .current_dir(&dir)
            .env(CHILD_STARTUP, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success() && stdout.contains("1 passed"), "{stdout}\n{stderr}");
    }

    #[test]
    #[ignore = "run by after_startup_a_dll_planted_in_the_current_directory_is_not_opened in a child process"]
    fn child_restricts_the_search() {
        env::var(CHILD_STARTUP).expect("run from its parent test");
        let cwd = env::current_dir().unwrap();
        let plant = |name: &str| std::fs::copy(system32().unwrap().join("version.dll"), cwd.join(name)).unwrap();
        // SAFETY: copies of a system DLL; each handle is dropped at once, unloading the module, so the next load of
        // the name searches again
        let opens = |name: &str| unsafe { Library::new(name) }.is_ok();

        // the control: before the call, the default search opens a file in the current directory
        plant("cudnn.dll");
        assert!(opens("cudnn.dll"), "the default search did not open a DLL in the current directory");

        restrict_search().unwrap();
        // the first name cudarc tries for the driver, written after startup
        plant("cuda.dll");
        assert!(!opens("cuda.dll"), "a DLL written into the current directory after startup was opened");
        assert!(!opens("cudnn.dll"), "a DLL in the current directory was opened after startup");
    }

    /// cudarc's load by name reuses a driver someone else already loaded, so it must be System32's
    #[test]
    fn a_driver_already_loaded_from_elsewhere_is_refused() {
        let (dir, name) = probe("driver");
        let other = temp().join(format!("inkrs-dll-{}-driver-system", std::process::id()));
        std::fs::create_dir_all(&other).unwrap();
        std::fs::copy(dir.join(&name), other.join(&name)).unwrap();
        // SAFETY: a copy of a system DLL
        let held = unsafe { Library::new(dir.join(&name)) }.unwrap();
        let names = [name.clone()];

        let elsewhere = Search::new(&env::temp_dir(), &no_path(), None, other);
        let err = elsewhere.driver_from(&names).unwrap_err();
        assert!(err.to_string().contains("already loaded"), "{err}");

        let here = Search::new(&env::temp_dir(), &no_path(), None, dir);
        here.driver_from(&names).unwrap();
        drop(held);
    }

    /// A module reused because it is already loaded stays loaded after its first owner lets it go
    #[test]
    fn a_module_already_loaded_is_held() {
        let (dir, name) = probe("held");
        // SAFETY: a copy of a system DLL
        let owner = unsafe { Library::new(dir.join(&name)) }.unwrap();
        let search = Search::new(&env::temp_dir(), &no_path(), None, system32().unwrap());
        assert_eq!(search.open_trusted(&name), Some(dir));
        drop(owner);
        assert!(loaded(&name).is_some(), "unloaded with its first owner");
    }

    /// With a later name already loaded, an earlier one cudarc can find would be a second copy, which it would use
    #[test]
    fn an_earlier_name_cannot_shadow_one_already_loaded() {
        let (first_dir, first) = probe("earlier");
        let (later_dir, later) = probe("later");
        // SAFETY: a copy of a system DLL
        let held = unsafe { Library::new(later_dir.join(&later)) }.unwrap();
        let names = [first.clone(), later];

        let shadowed = Search::new(&env::temp_dir(), first_dir.as_os_str(), None, system32().unwrap());
        let err = shadowed.first_of(&names).unwrap_err();
        assert!(err.to_string().contains("second copy"), "{err}");
        assert!(loaded(&first).is_none(), "the earlier name was loaded before it was refused");

        let clear = Search::new(&env::temp_dir(), &no_path(), None, system32().unwrap());
        assert_eq!(clear.first_of(&names).unwrap(), Some(later_dir));
        drop(held);
    }
}
