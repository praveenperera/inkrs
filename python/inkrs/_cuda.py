"""Finding cuDNN 9 and cuBLAS 12 before the engine opens the GPU

The engine (through cudarc) dlopens each library by name, trying a fixed list of names in order (`libcudnn.so`, ...,
`libcudnn.so.9`; `libcublas.so`, ..., `libcublas.so.12`). The first name that opens wins. A name matches a library
already loaded into the process by its soname, before the linker's search path is tried.

So this module does nothing when one of those names already opens: from the system, `LD_LIBRARY_PATH`, or a copy
torch has loaded. In that case the engine uses the same files the `inkrs` CLI would. Otherwise it loads the copy from
the `nvidia-cudnn-cu12` / `nvidia-cublas-cu12` pip packages by absolute path, with RTLD_GLOBAL, so the engine's
`libcudnn.so.9` and `libcublas.so.12` resolve to it.

Loading a pip copy cannot stop an earlier name in the engine's list (a system `libcudnn.so`, say) from opening a
second copy, which the engine would then use. So after loading, this module opens the names in the engine's order
itself and reports the file that defines each library's entry point, which is the file the engine will call.

On Windows the engine tries the same list of names as `.dll` files (`cudnn.dll`, ..., `cudnn64_9.dll`), and for each
name reuses a module already loaded under it before it searches (the executable's directory, System32, then absolute
`PATH` entries and `%CUDA_PATH%`, never the current directory); cuDNN loads its sublibraries, and NVRTC, by bare name.
So the pip copies, in each package's `bin` directory, are loaded here whenever they are installed, and the engine
reuses them over a CUDA toolkit on `PATH`. The engine refuses to open a second copy of a library under a name it
tries before the one already loaded, so neither does this module: it raises CudaLibraryMissing naming that file.
Without pip copies the engine uses the system copy, which this module names after the same search.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# cudarc 0.19 get_lib_name_candidates for CUDA 12, in its order; the engine stops at the first that opens
_SUFFIXES = ["", "64", "64_12", "64_124", "64_124_0", "64_120_4", "64_10", "64_11", "64_12", "64_120_0", "64_9"]


def _candidates(name: str) -> list[str]:
    names = [f"lib{name}{s}.so" for s in _SUFFIXES]
    return names + [f"lib{name}.so.{v}" for v in ("12", "12", "11", "10", "9", "1")]


# (engine name, pip files to load, the pip packages' library directories under site-packages, pip package to install,
# a symbol the library defines, to find which file the engine's name resolves to)
_LIBRARIES = [
    ("cublas", ["libcublasLt.so.12", "libcublas.so.12"], ["nvidia/cublas/lib"], "nvidia-cublas-cu12", "cublasCreate_v2"),
    ("cudnn", ["libcudnn.so.9"], ["nvidia/cudnn/lib"], "nvidia-cudnn-cu12", "cudnnGetVersion"),
]

_WINDOWS = sys.platform == "win32"


def _windows_candidates(name: str) -> list[str]:
    """cudarc's names for `name` on Windows, in its order"""
    names = [f"{name}{s}.dll" for s in _SUFFIXES]
    return names + [f"{name}.dll.{v}" for v in ("12", "12", "11", "10", "9", "1")]


# Windows: (engine name, pip DLLs to load, dependencies first, the pip package's DLL directory under site-packages, pip
# package to install). cuDNN loads its sublibraries by bare name, so they load here too, from beside cudnn64_9.dll
_WINDOWS_LIBRARIES = [
    ("cublas", ["cublasLt64_12.dll", "cublas64_12.dll"], "nvidia/cublas/bin", "nvidia-cublas-cu12"),
    (
        "cudnn",
        [
            "cudnn_graph64_9.dll",
            "cudnn_ops64_9.dll",
            "cudnn_heuristic64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll",
            "cudnn_engines_tensor_ir64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_adv64_9.dll",
            "cudnn_ext64_9.dll",
            "cudnn64_9.dll",
        ],
        "nvidia/cudnn/bin",
        "nvidia-cudnn-cu12",
    ),
]

# Windows: NVRTC, which cuDNN loads by bare name for its runtime-compiled engines, and NVRTC its builtins from beside
# itself. Optional: loaded from its pip package (directory, file, package) only when that is installed
_WINDOWS_NVRTC = ("nvidia/cuda_nvrtc/bin", "nvrtc64_120_0.dll", "nvidia-cuda-nvrtc-cu12")

# LoadLibraryExW's LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: the executable's directory, System32 and the AddDllDirectory (and
# os.add_dll_directory) directories, never the current directory or PATH
_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000

_loaded: dict[str, str] | None = None


class CudaLibraryMissing(OSError):
    """A CUDA library the engine needs is neither on the system nor in a pip package"""


def _opens(name: str) -> bool:
    try:
        ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return False
    return True


class _DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def _defining_file(lib: ctypes.CDLL, symbol: str) -> str | None:
    """The file that defines `symbol` as `lib` resolves it, through dladdr"""
    try:
        addr = ctypes.cast(getattr(lib, symbol), ctypes.c_void_p).value
    except AttributeError:
        return None
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
    info = _DlInfo()
    if not addr or not dladdr(addr, ctypes.byref(info)) or not info.dli_fname:
        return None
    return os.path.realpath(info.dli_fname.decode())


def _engine_file(name: str, symbol: str) -> str | None:
    """The file the engine will use for `name`: the first of its names that opens, as cudarc tries them"""
    for candidate in _candidates(name):
        try:
            lib = ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        return _defining_file(lib, symbol) or candidate
    return None


def _site_dirs() -> list[Path]:
    """The sys.path directories a pip package can be installed in: absolute entries other than the current directory.
    An empty or relative entry (the current directory, in a REPL or under `python -c`) and the current directory itself
    (under `python -m`) are skipped, since a library found by this search is loaded by full path, and the engine never
    loads one from the current directory"""
    cwd = os.path.normcase(os.path.abspath(os.getcwd()))
    seen: list[Path] = []
    for entry in sys.path:
        if not entry or not os.path.isabs(entry) or os.path.normcase(os.path.abspath(entry)) == cwd:
            continue
        p = Path(entry)
        if p.is_dir() and p not in seen:
            seen.append(p)
    return seen


def _from_pip(setting: str) -> set[str]:
    """The libraries INKRS_CUDA_LIBS forces from pip: "pip" for all, or a comma list such as "cudnn" """
    names = {entry[0] for entry in _LIBRARIES}
    if setting in ("", "auto"):
        return set()
    if setting == "pip":
        return names
    asked = {part.strip() for part in setting.split(",") if part.strip()}
    if not asked <= names:
        raise CudaLibraryMissing(
            f"INKRS_CUDA_LIBS={setting!r}: use auto, pip, or a comma list of {', '.join(sorted(names))}"
        )
    return asked


def load_cuda_libraries() -> dict[str, str]:
    """Makes cuDNN 9 and cuBLAS 12 loadable for the engine; returns, per library, the file the engine will call

    Raises CudaLibraryMissing naming what was searched when a library is found nowhere, or when a pip copy fails to
    load. `INKRS_CUDA_LIBS=pip` loads the pip copies even when the system has its own (`INKRS_CUDA_LIBS=cudnn` only
    cuDNN's), and raises CudaLibraryMissing if a system copy would still be the one the engine opens.
    """
    global _loaded
    if _loaded is not None:
        return dict(_loaded)
    from_pip = _from_pip(os.environ.get("INKRS_CUDA_LIBS", "auto"))
    if _WINDOWS:
        _loaded = _load_windows(from_pip)
        return dict(_loaded)
    found: dict[str, str] = {}
    for name, files, subdirs, package, symbol in _LIBRARIES:
        prefer_pip = name in from_pip
        if not prefer_pip and any(_opens(c) for c in _candidates(name)):
            found[name] = "system"
        else:
            for site in _site_dirs():
                for sub in subdirs:
                    paths = [site / sub / f for f in files]
                    if all(p.is_file() for p in paths):
                        for p in paths:
                            try:
                                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                            except OSError as e:
                                raise CudaLibraryMissing(
                                    f"inkrs found {p} but could not load it: {e}. Reinstall it with "
                                    f"`pip install --force-reinstall {package}`, or put a system copy on LD_LIBRARY_PATH"
                                ) from e
                        found[name] = os.path.realpath(paths[-1])
                        break
                if name in found:
                    break
        if name in found:
            used = _engine_file(name, symbol)
            if prefer_pip and used != found[name]:
                raise CudaLibraryMissing(
                    f"INKRS_CUDA_LIBS forced {found[name]} from pip, but the engine would open {used} instead, because it "
                    f"tries {', '.join(_candidates(name)[:3])}, ... first and one of them opens a system copy. Unset "
                    f"INKRS_CUDA_LIBS to use that copy, or remove it from the library path"
                )
            found[name] = used or found[name]
        else:
            raise CudaLibraryMissing(
                f"inkrs needs {files[-1]} and found it neither on the system library path nor in a pip package "
                f"(looked for {', '.join(subdirs)} under {len(_site_dirs())} sys.path entries). "
                f"Install it with `pip install {package}` (or `pip install inkrs[cu12]`), or put the system copy on "
                f"LD_LIBRARY_PATH"
            )
    _loaded = found
    return dict(found)


def _windows_system_dirs() -> list[Path]:
    """Where the engine looks for a DLL not already loaded, in its order: the executable's directory, System32, each
    absolute PATH entry, then %CUDA_PATH%\\bin\\x64 and %CUDA_PATH%\\bin. Never the current directory"""
    dirs = [Path(p).parent for p in [_windows_module_file(None)] if p] + [_windows_system32()]
    dirs += [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p and Path(p).is_absolute()]
    cuda = os.environ.get("CUDA_PATH")
    if cuda and Path(cuda).is_absolute():
        dirs += [Path(cuda) / "bin" / "x64", Path(cuda) / "bin"]
    return dirs


def _windows_system32() -> Path:
    """System32 from GetSystemDirectoryW, as the engine reads it, rather than from the environment"""
    buffer = ctypes.create_unicode_buffer(260)
    length = ctypes.WinDLL("kernel32").GetSystemDirectoryW(buffer, len(buffer))
    return Path(buffer.value if 0 < length < len(buffer) else r"C:\Windows\System32")


def _windows_module_file(name: str | None) -> str | None:
    """The file of the module loaded under `name` (the executable for None), without loading or holding it"""
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    module = kernel32.GetModuleHandleW(name)
    if not module:
        return None
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel32.GetModuleFileNameW(module, buffer, len(buffer))
    return buffer.value if 0 < length < len(buffer) else None


def _windows_lookup(name: str) -> str | None:
    """The file the engine would use for `name`, without loading anything: a module already loaded under the name,
    else the first of the engine's directories holding it"""
    loaded = _windows_module_file(name)
    if loaded:
        return loaded
    return next((str(d / name) for d in _windows_system_dirs() if (d / name).is_file()), None)


def _windows_load_default_dirs(name: str) -> str | None:
    """Loads `name` by the engine's first search, which also reaches AddDllDirectory directories that can't be listed;
    returns its file, or None when it isn't there"""
    try:
        ctypes.WinDLL(name, winmode=_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)
    except OSError:
        return None
    return _windows_module_file(name)


def _windows_engine_file(name: str) -> str | None:
    """The file the engine will use for `name`: the first of cudarc's names it finds, in cudarc's order"""
    names = _windows_candidates(name)
    found = next((f for f in map(_windows_lookup, names) if f), None)
    return found or next((f for f in map(_windows_load_default_dirs, names) if f), None)


def _windows_pip_nvrtc(sites: list[Path]) -> list[Path]:
    """NVRTC's builtins and then NVRTC itself from its pip package, if installed: the files to load, in order"""
    subdir, file, _ = _WINDOWS_NVRTC
    bin_dir = next((site / subdir for site in sites if (site / subdir / file).is_file()), None)
    if bin_dir is None:
        return []
    builtins = sorted(p for p in bin_dir.glob("nvrtc-builtins64_*.dll") if p.is_file())
    return builtins + [bin_dir / file]


def _load_pip_dll(path: Path, package: str) -> None:
    try:
        # an absolute path also searches the DLL's own directory for its imports
        ctypes.CDLL(str(path))
    except OSError as e:
        raise CudaLibraryMissing(
            f"inkrs found {path} but could not load it: {e}. Reinstall it with "
            f"`pip install --force-reinstall {package}`, or remove it to use a CUDA toolkit on PATH"
        ) from e


def _same_file(a: str, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _load_windows(from_pip: set[str]) -> dict[str, str]:
    """Windows: loads each library's pip copy when installed, so the engine reuses it; returns the file the engine
    will use for each. Raises CudaLibraryMissing when a library is found nowhere, a pip copy forced or found will not
    load, or the engine would open a second copy before the pip one"""
    for path in _windows_pip_nvrtc(_site_dirs()):
        _load_pip_dll(path, _WINDOWS_NVRTC[2])

    found: dict[str, str] = {}
    for name, files, subdir, package in _WINDOWS_LIBRARIES:
        pip_dir = next((site / subdir for site in _site_dirs() if (site / subdir / files[-1]).is_file()), None)
        if pip_dir is None and name in from_pip:
            raise CudaLibraryMissing(
                f"INKRS_CUDA_LIBS forced {name} from pip, but {package} is not installed (looked for {subdir} under "
                f"{len(_site_dirs())} sys.path entries). Install it with `pip install {package}`"
            )
        if pip_dir is not None:
            for file in files:
                if (pip_dir / file).is_file():
                    _load_pip_dll(pip_dir / file, package)

        used = _windows_engine_file(name)
        if used is None:
            raise CudaLibraryMissing(
                f"inkrs needs {files[-1]} and found it neither in a pip package nor in the executable's directory, "
                f"System32, an absolute PATH entry or %CUDA_PATH%. Install it with `pip install {package}` (or "
                f"`pip install inkrs[cu12]`), or put the directory holding it on PATH"
            )
        if pip_dir is not None and not _same_file(used, pip_dir / files[-1]):
            raise CudaLibraryMissing(
                f"the engine would open {used} as a second copy of {name}, before the pip copy in {pip_dir}, because it "
                f"tries {', '.join(_windows_candidates(name)[:3])}, ... first; it refuses to. Remove that file from "
                f"the search path, or uninstall {package} to use it"
            )
        found[name] = used
    return found


def _windows_modules() -> list[str]:
    """The files of every module loaded in this process (Windows), through EnumProcessModules"""
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.K32EnumProcessModules.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
    ]  # fmt: skip
    kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    process = kernel32.GetCurrentProcess()
    count = 1024
    while True:
        modules = (wintypes.HMODULE * count)()
        needed = wintypes.DWORD()
        if not kernel32.K32EnumProcessModules(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
            return []
        if needed.value <= ctypes.sizeof(modules):
            break
        count = needed.value // ctypes.sizeof(wintypes.HMODULE) + 16
    out: list[str] = []
    buffer = ctypes.create_unicode_buffer(32768)
    for module in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        length = kernel32.GetModuleFileNameW(module, buffer, len(buffer))
        if length:
            out.append(buffer.value)
    return out


def mapped_cuda_libraries() -> list[str]:
    """The cuDNN, cuBLAS and driver files mapped into this process, from /proc/self/maps (the loaded modules on
    Windows)"""
    if _WINDOWS:
        names = ("cudnn", "cublas", "nvcuda.dll", "nvrtc")
        return sorted({f for f in _windows_modules() if any(n in Path(f).name.lower() for n in names)})
    keep = ("libcudnn", "libcublas", "libcuda.so")
    out: list[str] = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6 and any(k in Path(parts[5]).name for k in keep) and parts[5] not in out:
                    out.append(parts[5])
    except OSError:
        pass
    return sorted(out)
