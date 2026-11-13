"""CPU-side tests of the installed package: no GPU, cuDNN or checkpoint needed

    pip install inkrs pytest tifffile && pytest python/tests/test_api.py
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import inkrs
from inkrs import _cuda

#: the PTX, device and CUDA library tests; a macOS wheel runs on Metal and has none of them
cuda_only = pytest.mark.skipif(sys.platform == "darwin", reason="the macOS wheel runs on Metal")
metal_only = pytest.mark.skipif(sys.platform != "darwin", reason="Metal is the macOS backend")


@cuda_only
def test_build_info_reads_the_embedded_ptx():
    info = inkrs.build_info()
    assert info["version"] == inkrs.__version__
    # every tier the engine chooses between, lowest first
    assert [(e["target"], e["isa"]) for e in info["ptx"]] == [("sm_75", "6.5"), ("sm_80", "7.0"), ("sm_120", "8.7")]
    assert [e["min_driver"] for e in info["ptx"]] == [440, 450, 570]
    # the families compiled ahead, each from the tier the engine loads there
    assert [(e["ptx"], e["arch"]) for e in info["cubins"]] == [
        ("sm_75", "sm_75"),
        ("sm_80", "sm_80"),
        ("sm_80", "sm_86"),
        ("sm_80", "sm_89"),
        ("sm_80", "sm_90"),
        ("sm_80", "sm_100"),
        ("sm_120", "sm_120"),
    ]
    # the sm75 tier JITs on r440, but the CUDA 12 cuDNN and cuBLAS need r525
    assert info["min_driver"] == 525


def test_write_tiff_is_villas_lzw_bigtiff(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("imagecodecs")  # tifffile decodes LZW through it
    rng = np.random.default_rng(0)
    # an edge that is not a multiple of the 128 tile, so the padded edge tiles are exercised
    image = rng.integers(0, 256, size=(300, 200), dtype=np.uint8)
    image[:64] = 0
    path = tmp_path / "map.tif"
    inkrs.write_tiff(path, image)
    with tifffile.TiffFile(path) as tif:
        assert tif.is_bigtiff
        page = tif.pages[0]
        assert page.compression == tifffile.COMPRESSION.LZW
        assert (page.tilelength, page.tilewidth) == (128, 128)
        assert page.software == "vesuvius.ink_detection.inference.infer"
        np.testing.assert_array_equal(page.asarray(), image)


def test_write_tiff_rejects_non_uint8(tmp_path):
    with pytest.raises(TypeError):
        inkrs.write_tiff(tmp_path / "x.tif", np.zeros((128, 128), dtype=np.float32))


@pytest.mark.parametrize(
    "depth, start, end, reverse, want",
    [
        # the PHerc. 343 recipe: S = 25 + d over 66 layers, window [S, S + 17)
        (66, 25, 42, False, (25, 1)),
        (66, 21, 38, True, (37, -1)),
        # no window: the full depth, centre-cropped to 17 (upper centre for even excess)
        (66, None, None, False, (25, 1)),
        (65, None, None, False, (24, 1)),
        # negatives wrap, an empty window falls back to the whole depth
        (66, -20, -3, False, (46, 1)),
        (66, 40, 10, False, (25, 1)),
    ],
)
def test_layer_selection_follows_villa(depth, start, end, reverse, want):
    assert inkrs._select_layers(depth, start, end, reverse) == want


def test_layer_selection_refuses_short_windows():
    with pytest.raises(ValueError, match="needs 17"):
        inkrs._select_layers(66, 10, 20, False)


@pytest.mark.parametrize("overlap, want", [(0.5, 64), (0.25, 96), (0.0, 128), (0.75, 32), (None, 64)])
def test_stride_from_overlap(overlap, want):
    assert inkrs._stride("reader-v2", 128, overlap, None) == want


@pytest.mark.parametrize("overlap, want", [(0.5, 192), (0.25, 288), (0.0, 384), (None, 192)])
def test_stride_from_overlap_on_384_tiles(overlap, want):
    # Reader V3's tiles are 384 px, so villa's overlap 0.5 is stride 192 and a stride may reach 384
    assert inkrs._stride("reader-v3", 384, overlap, None) == want
    assert inkrs._stride(None, 384, None, 384) == 384
    with pytest.raises(ValueError, match="outside"):
        inkrs._stride(None, 384, None, 385)


def test_stride_refusals_come_from_the_engine():
    assert inkrs._stride(None, 128, None, 96) == 96
    for bad in (0, 129):
        with pytest.raises(ValueError, match="outside"):
            inkrs._stride(None, 128, None, bad)
    with pytest.raises(ValueError, match="not both"):
        inkrs._stride(None, 128, 0.5, 64)
    # no reader has a fast preset: fast mode refuses rather than running parity or an unmeasured stride
    for reader in ("reader-v2", "hybrid_3d2d-seed42", None):
        with pytest.raises(ValueError, match="no fast preset"):
            inkrs._stride(reader, 128, None, None, fast=True)


def test_fast_presets():
    p = inkrs.fast_presets()
    assert p["readers"] == []
    assert p["hecate"] == {"stride": 32, "precision": "bf16", "batch": 32}


def test_hecate_refuses_bad_requests_before_any_cuda_library_loads(tmp_path, monkeypatch):
    # every refusal comes before the CUDA libraries load or the volume is read, so it is a ValueError even on a
    # machine without them; the loader stub fails the test if a bad request ever reaches it
    def no_load():
        raise AssertionError("load_cuda_libraries ran for a request the engine refuses")

    monkeypatch.setattr(inkrs, "load_cuda_libraries", no_load)
    (tmp_path / "exists.png").write_bytes(b"")
    cases = [
        ({"fast": True, "stride": 48}, "--fast sets"),
        ({"fast": True, "precision": "fp32"}, "--fast sets"),
        ({"fast": True, "overlap": 0.5}, "--fast sets"),
        ({"stride": 32, "overlap": 0.5}, "not both"),
        ({"stride": 65}, "outside"),
        ({"stride": 0}, "outside"),
        ({"stride": -1}, "negative"),
        ({"overlap": 1.0}, "overlap"),
        ({"batch_size": 0}, "batch size"),
        ({"precision": "bf16", "tf32": False}, "fp32 only"),
        ({"precision": "fp8"}, "need fp32 or bf16"),
        ({"algo": "fastest"}, "need find or heuristic"),
        ({"direction": "sideways"}, "direction"),
        ({"spacing_um": 8.64}, "resample"),
        ({"spacing_um": float("nan")}, "resample"),
        ({"output": tmp_path / "o.tif"}, "ending in .png"),
        ({"output": tmp_path / "exists.png"}, "ending in .png"),
        ({"output": tmp_path / "exists.png", "direction": "reverse"}, "ending in .png"),
        ({"device": -1}, "negative"),
    ]
    for kwargs, match in cases:
        kwargs = {"spacing_um": 9.6, "output": tmp_path / "o.png"} | kwargs
        output = kwargs.pop("output")
        with pytest.raises(ValueError, match=match):
            inkrs.hecate(tmp_path / "missing.npy", tmp_path / "missing.safetensors", output, **kwargs)


def test_engine_precision_is_fp16_only():
    with pytest.raises(ValueError, match="fp16 only"):
        inkrs.Engine("missing.safetensors", precision="bf16")


@metal_only
def test_build_info_on_macos_names_metal_and_no_ptx():
    info = inkrs.build_info()
    assert info["backend"] == "metal"
    assert info["ptx"] == [] and info["cubins"] == [] and inkrs.load_cuda_libraries() == {}


@metal_only
def test_device_forms_on_macos():
    assert inkrs._device_index(0) == 0
    assert inkrs._device_index("mps") == 0
    assert inkrs._device_index("mps:0") == 0
    for bad in (1, "cuda", "mps:1", "cpu"):
        with pytest.raises(ValueError):
            inkrs._device_index(bad)


@pytest.mark.parametrize("metal", [False, True])
def test_a_negative_device_is_refused_alike_on_both_backends(monkeypatch, metal):
    # macOS once answered device=-1 with the Metal device list, so test_api's hecate case failed there only; the
    # backend switch is a module flag, so both branches run on any host
    monkeypatch.setattr(inkrs, "_METAL", metal)
    with pytest.raises(ValueError, match="must not be negative"):
        inkrs._device_index(-1)


@cuda_only
def test_device_forms():
    assert inkrs._device_index(1) == 1
    assert inkrs._device_index("cuda") == 0
    assert inkrs._device_index("cuda:2") == 2
    with pytest.raises(ValueError):
        inkrs._device_index("cpu")


def test_volume_conversion_is_zero_copy_for_c_uint8():
    vol = np.zeros((17, 128, 130), dtype=np.uint8)
    assert np.shares_memory(inkrs._as_volume(vol, 128), vol)
    fortran = np.asfortranarray(vol)
    out = inkrs._as_volume(fortran, 128)
    assert out.flags.c_contiguous and not np.shares_memory(out, fortran)
    with pytest.raises(ValueError, match=r"\[Z, H, W\]"):
        inkrs._as_volume(vol[0], 128)


@pytest.mark.parametrize(
    "volume, match",
    [
        (np.zeros((17, 128, 128), np.uint16), "uint8"),
        (np.zeros((17, 128, 128), np.float32), "uint8"),
        (np.zeros((17, 100, 128), np.uint8), "smaller than"),
        (np.zeros((17, 128, 127), np.uint8), "smaller than"),
    ],
)
def test_unsupported_input_raises_the_fallback_error(volume, match):
    # villa falls back to PyTorch on UnsupportedError and re-raises any other InkrsError
    assert issubclass(inkrs.UnsupportedError, inkrs.InkrsError)
    with pytest.raises(inkrs.UnsupportedError, match=match):
        inkrs._as_volume(volume, 128)


class _Lazy:
    """A lazy array whose data must never be read"""

    dtype = np.dtype(np.uint8)

    def __init__(self, shape):
        self.shape = shape

    def __array__(self, *args, **kwargs):
        raise AssertionError("read before the limits were checked")


@pytest.mark.parametrize("shape", [(17, 65536, 65536), (17, 100, 4096)])
def test_lazy_arrays_are_refused_from_their_metadata(shape):
    with pytest.raises(inkrs.UnsupportedError):
        inkrs._as_volume(_Lazy(shape), 128)


def test_a_384_reader_refuses_a_volume_smaller_than_its_tile_before_reading():
    # Reader V3's engine has 384 px tiles: a 256 px lazy volume passes the 128 check but must not be read
    with pytest.raises(inkrs.UnsupportedError, match="smaller than a 384 px tile"):
        inkrs._as_volume(_Lazy((17, 256, 256)), 384)
    assert inkrs._as_volume(np.zeros((17, 384, 384), np.uint8), 384).shape == (17, 384, 384)


def test_a_lazy_array_of_unknown_extent_is_checked_after_reading():
    # dask reports nan for an extent it knows only after computing, e.g. after boolean selection
    class Unknown(_Lazy):
        def __init__(self, shape, data):
            super().__init__(shape)
            self.data = data

        def __array__(self, *args, **kwargs):
            return self.data

    full = Unknown((float("nan"), 128, 128), np.zeros((17, 128, 128), np.uint8))
    assert inkrs._as_volume(full, 128).shape == (17, 128, 128)
    with pytest.raises(inkrs.UnsupportedError):
        inkrs._as_volume(Unknown((float("nan"), 100, 128), np.zeros((17, 100, 128), np.uint8)), 128)


def test_over_4_gib_is_unsupported(monkeypatch):
    monkeypatch.setattr(inkrs, "_MAX_VOLUME_BYTES", 17 * 128 * 128)
    inkrs._as_volume(np.zeros((16, 128, 128), np.uint8), 128)
    with pytest.raises(inkrs.UnsupportedError, match="4 GiB"):
        inkrs._as_volume(np.zeros((17, 128, 128), np.uint8), 128)


def test_torch_cpu_tensor_is_read_in_place():
    torch = pytest.importorskip("torch")
    t = torch.zeros((17, 128, 128), dtype=torch.uint8)
    assert np.shares_memory(inkrs._as_volume(t, 128), t.numpy())
    with pytest.raises(inkrs.UnsupportedError, match="uint8"):
        inkrs._as_volume(t.float(), 128)


@cuda_only
@pytest.mark.skipif(sys.platform == "win32", reason="the Linux soname search; see the Windows tests below")
def test_missing_cudnn_names_the_pip_package(monkeypatch):
    monkeypatch.setattr(_cuda, "_loaded", None)
    monkeypatch.setattr(_cuda, "_candidates", lambda name: [f"libinkrs-test-missing-{name}.so"])
    monkeypatch.setattr(_cuda, "_site_dirs", list)
    with pytest.raises(inkrs.CudaLibraryError, match="pip install nvidia-cublas-cu12"):
        inkrs.load_cuda_libraries()


def _pip_cudnn_only(monkeypatch, tmp_path, contents):
    lib = tmp_path / "site" / "nvidia" / "cudnn" / "lib"
    lib.mkdir(parents=True)
    (lib / "libcudnn.so.9").write_bytes(contents)
    monkeypatch.setenv("INKRS_CUDA_LIBS", "pip")
    monkeypatch.setattr(_cuda, "_loaded", None)
    monkeypatch.setattr(_cuda, "_LIBRARIES", [entry for entry in _cuda._LIBRARIES if entry[0] == "cudnn"])
    monkeypatch.setattr(_cuda, "_site_dirs", lambda: [tmp_path / "site"])
    return lib / "libcudnn.so.9"


@cuda_only
@pytest.mark.skipif(sys.platform == "win32", reason="ELF libraries and /proc/self/maps")
def test_a_pip_library_that_will_not_load_is_a_cuda_library_error(monkeypatch, tmp_path):
    path = _pip_cudnn_only(monkeypatch, tmp_path, b"not an ELF file")
    with pytest.raises(inkrs.CudaLibraryError, match=f"found {path} but could not load it"):
        inkrs.load_cuda_libraries()


def _libm_path():
    ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
    with open("/proc/self/maps") as f:
        for line in f:
            path = line.split()[-1]
            if os.path.basename(path).startswith("libm.so"):
                return os.path.realpath(path)
    pytest.skip("libm is not mapped")


@cuda_only
@pytest.mark.skipif(sys.platform == "win32", reason="ELF libraries and /proc/self/maps")
def test_forced_pip_fails_when_the_engine_would_open_another_copy(monkeypatch, tmp_path):
    # libm stands in for the pip cuDNN: a real shared object that loads; the engine's lookup is made to find another
    stand_in = _pip_cudnn_only(monkeypatch, tmp_path, b"")
    stand_in.unlink()
    stand_in.symlink_to(_libm_path())
    monkeypatch.setattr(_cuda, "_engine_file", lambda name, symbol: "/usr/lib/libcudnn.so.9")
    with pytest.raises(inkrs.CudaLibraryError, match="but the engine would open /usr/lib/libcudnn.so.9"):
        inkrs.load_cuda_libraries()
    monkeypatch.setattr(_cuda, "_engine_file", lambda name, symbol: os.path.realpath(stand_in))
    assert inkrs.load_cuda_libraries() == {"cudnn": os.path.realpath(stand_in)}


def test_cuda_libs_setting():
    assert _cuda._from_pip("auto") == set()
    assert _cuda._from_pip("pip") == {"cudnn", "cublas"}
    assert _cuda._from_pip("cudnn") == {"cudnn"}
    with pytest.raises(_cuda.CudaLibraryMissing, match="comma list"):
        _cuda._from_pip("cudnn9")


@cuda_only
@pytest.mark.skipif(sys.platform == "win32", reason="ELF libraries and /proc/self/maps")
def test_engine_file_names_the_file_that_defines_the_symbol(monkeypatch):
    libm = _libm_path()
    monkeypatch.setattr(_cuda, "_candidates", lambda name: ["libinkrs-test-missing.so", libm])
    assert _cuda._engine_file("m", "cos") == libm


def test_windows_cache_dir_is_localappdata(monkeypatch, tmp_path):
    # the engine resolves the same root (engine::cache_root), so both share digests and exports
    for name in ("INKRS_CACHE_DIR", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(sys, "platform", "win32")
    assert inkrs._cache_dir() == tmp_path / "local" / "inkrs"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert inkrs._cache_dir() == tmp_path / "xdg" / "inkrs"


def _windows_search(monkeypatch, tmp_path, system_dirs, loaded=None):
    # the engine's search with no Win32 call: `loaded` maps a module name to the file it is loaded from
    loaded = loaded or {}
    monkeypatch.setattr(_cuda, "_site_dirs", lambda: [tmp_path / "site"])
    monkeypatch.setattr(_cuda, "_windows_system_dirs", lambda: system_dirs)
    monkeypatch.setattr(_cuda, "_windows_module_file", lambda name: loaded.get(name))
    monkeypatch.setattr(_cuda, "_windows_load_default_dirs", lambda name: None)


def test_site_dirs_skip_the_current_directory(monkeypatch, tmp_path):
    # a pip copy is loaded by full path, so one planted under the current directory must not be found
    cwd, site = tmp_path / "cwd", tmp_path / "site"
    (cwd / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    site.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, "path", ["", "rel", str(cwd), str(site)])
    assert _cuda._site_dirs() == [site]


def test_windows_loads_nvrtc_from_pip_when_installed(monkeypatch, tmp_path):
    # selection only, so it runs on any OS
    bin_dir = tmp_path / "site" / "nvidia" / "cuda_nvrtc" / "bin"
    bin_dir.mkdir(parents=True)
    assert _cuda._windows_pip_nvrtc([tmp_path / "site"]) == []
    for name in ("nvrtc64_120_0.dll", "nvrtc-builtins64_128.dll"):
        (bin_dir / name).write_bytes(b"")
    want = [bin_dir / "nvrtc-builtins64_128.dll", bin_dir / "nvrtc64_120_0.dll"]
    assert _cuda._windows_pip_nvrtc([tmp_path / "empty", tmp_path / "site"]) == want

    toolkit = tmp_path / "toolkit"
    toolkit.mkdir()
    for name in ("cublas64_12.dll", "cudnn64_9.dll"):
        (toolkit / name).write_bytes(b"")
    _windows_search(monkeypatch, tmp_path, [toolkit])
    loads = []
    monkeypatch.setattr(_cuda, "_load_pip_dll", lambda path, package: loads.append(path))
    _cuda._load_windows(set())
    assert loads == want


def test_windows_refuses_a_second_copy_before_the_pip_one(monkeypatch, tmp_path):
    # the engine tries cudnn.dll before cudnn64_9.dll and would open it beside the loaded pip copy (src/dll.rs refuses)
    pip = tmp_path / "site" / "nvidia" / "cudnn" / "bin"
    pip.mkdir(parents=True)
    (pip / "cudnn64_9.dll").write_bytes(b"")
    toolkit = tmp_path / "toolkit"
    toolkit.mkdir()
    monkeypatch.setattr(_cuda, "_WINDOWS_LIBRARIES", [e for e in _cuda._WINDOWS_LIBRARIES if e[0] == "cudnn"])
    monkeypatch.setattr(_cuda, "_load_pip_dll", lambda path, package: None)
    _windows_search(monkeypatch, tmp_path, [toolkit], loaded={"cudnn64_9.dll": str(pip / "cudnn64_9.dll")})
    assert _cuda._load_windows(set()) == {"cudnn": str(pip / "cudnn64_9.dll")}
    (toolkit / "cudnn.dll").write_bytes(b"")
    with pytest.raises(_cuda.CudaLibraryMissing, match="second copy"):
        _cuda._load_windows(set())


def test_windows_names_the_system_copy_or_the_package_to_install(monkeypatch, tmp_path):
    # selection only: no file here is loaded, so it runs on any OS
    toolkit = tmp_path / "toolkit"
    toolkit.mkdir()
    _windows_search(monkeypatch, tmp_path, [tmp_path / "empty", toolkit])
    with pytest.raises(_cuda.CudaLibraryMissing, match="pip install nvidia-cublas-cu12"):
        _cuda._load_windows(set())
    for name in ("cublas64_12.dll", "cudnn64_9.dll"):
        (toolkit / name).write_bytes(b"")
    assert _cuda._load_windows(set()) == {"cublas": str(toolkit / "cublas64_12.dll"), "cudnn": str(toolkit / "cudnn64_9.dll")}
    # a forced pip copy that is not installed is an error even with a system copy
    with pytest.raises(_cuda.CudaLibraryMissing, match="INKRS_CUDA_LIBS forced cudnn from pip"):
        _cuda._load_windows({"cudnn"})


@pytest.mark.skipif(sys.platform != "win32", reason="loads a DLL")
def test_a_pip_dll_that_will_not_load_is_a_cuda_library_error(monkeypatch, tmp_path):
    bin_dir = tmp_path / "site" / "nvidia" / "cudnn" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "cudnn64_9.dll").write_bytes(b"not a PE file")
    monkeypatch.setenv("INKRS_CUDA_LIBS", "cudnn")
    monkeypatch.setattr(_cuda, "_loaded", None)
    monkeypatch.setattr(_cuda, "_WINDOWS_LIBRARIES", [e for e in _cuda._WINDOWS_LIBRARIES if e[0] == "cudnn"])
    _windows_search(monkeypatch, tmp_path, [])
    with pytest.raises(inkrs.CudaLibraryError, match="cudnn64_9.dll but could not load it"):
        inkrs.load_cuda_libraries()


def test_engine_reports_a_missing_checkpoint(tmp_path):
    # InkrsError, or its subclass CudaLibraryError on a machine without cuDNN
    with pytest.raises(inkrs.InkrsError):
        inkrs.Engine(tmp_path / "missing.safetensors")


def _fake_export(directory, sha, name="hybrid_3d2d-seed42"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.safetensors").write_bytes(b"")
    (directory / f"{name}.json").write_text(f'{{"checkpoint_sha256": "{sha}"}}')
    return directory / f"{name}.safetensors"


def test_pth_resolves_to_its_export_by_sha256(tmp_path, monkeypatch):
    monkeypatch.setenv("INKRS_CACHE_DIR", str(tmp_path / "cache"))
    pth = tmp_path / "step.pth"
    pth.write_bytes(b"a checkpoint")
    sha = inkrs._sha256(pth)
    weights = tmp_path / "weights"
    _fake_export(weights / "other", "0" * 64)
    # an export of another checkpoint in weights_dir is not taken
    _fake_export(weights, "f" * 64, name="stale")
    cached = _fake_export(tmp_path / "cache" / "weights" / sha, sha)
    assert inkrs._resolve_checkpoint(pth, weights) == cached
    # weights_dir wins over the cache
    assert inkrs._resolve_checkpoint(pth, tmp_path / "weights2") == cached
    local = _fake_export(tmp_path / "weights2", sha)
    assert inkrs._resolve_checkpoint(pth, tmp_path / "weights2") == local
    st = tmp_path / "x.safetensors"
    assert inkrs._resolve_checkpoint(st, weights) == st


def test_checkpoint_digest_is_cached_until_the_file_changes(tmp_path, monkeypatch):
    import hashlib
    import json

    monkeypatch.setenv("INKRS_CACHE_DIR", str(tmp_path / "cache"))
    pth = tmp_path / "step.pth"
    pth.write_bytes(b"a checkpoint")
    real = hashlib.sha256(b"a checkpoint").hexdigest()
    assert inkrs._sha256(pth) == real
    (entry,) = (tmp_path / "cache" / "checkpoint-sha256").glob("*.json")
    # a planted digest for the same size and mtime is taken, so the file was not hashed again
    meta = json.loads(entry.read_text())
    entry.write_text(json.dumps({**meta, "sha256": "0" * 64}))
    assert inkrs._sha256(pth) == "0" * 64
    # a rewrite of the same size moves the mtime, so the bytes are hashed again
    st = pth.stat()
    pth.write_bytes(b"b checkpoint")
    os.utime(pth, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert inkrs._sha256(pth) == hashlib.sha256(b"b checkpoint").hexdigest()


def test_pth_without_export_or_torch_is_unsupported(tmp_path, monkeypatch):
    monkeypatch.setenv("INKRS_CACHE_DIR", str(tmp_path / "cache"))
    pth = tmp_path / "step.pth"
    pth.write_bytes(b"not a checkpoint")
    real = inkrs.importlib.util.find_spec
    monkeypatch.setattr(inkrs.importlib.util, "find_spec", lambda name: None if name == "torch" else real(name))
    with pytest.raises(inkrs.UnsupportedError, match="python -m inkrs.export_weights"):
        inkrs.Engine(pth, weights_dir=tmp_path / "weights")


@pytest.mark.parametrize(
    "error, kind",
    [
        ("unsupported", "UnsupportedError"),
        ("import", "UnsupportedError"),
        (OSError(28, "No space left on device"), "InkrsError"),
        (MemoryError(), "InkrsError"),
        (RuntimeError("PytorchStreamReader failed reading zip archive"), "InkrsError"),
    ],
)
def test_only_an_unsupported_checkpoint_makes_a_failed_export_unsupported(tmp_path, monkeypatch, error, kind):
    # villa falls back on UnsupportedError, so a disk-full, OOM or corrupt-file export must stay a plain failure
    monkeypatch.setenv("INKRS_CACHE_DIR", str(tmp_path / "cache"))
    pth = tmp_path / "step.pth"
    pth.write_bytes(b"not a checkpoint")
    monkeypatch.setattr(inkrs.importlib.util, "find_spec", lambda name: object())
    from inkrs import export_weights

    def boom(checkpoint, out_dir, name="hybrid_3d2d-seed42"):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "half.safetensors").write_bytes(b"x")
        if error == "unsupported":
            raise export_weights.ExportUnsupported("mode 'surface' is not flat")
        if error == "import":
            raise ImportError("cannot import name 'flat_preprocessing_from_config'")
        raise error

    monkeypatch.setattr(export_weights, "export", boom)
    with pytest.raises(inkrs.InkrsError) as caught:
        inkrs._resolve_checkpoint(pth, tmp_path / "weights")
    assert type(caught.value).__name__ == kind, caught.value
    assert list((tmp_path / "cache" / "weights").iterdir()) == []


def test_concurrent_auto_exports_publish_one_complete_entry(tmp_path, monkeypatch):
    import threading
    import time

    monkeypatch.setenv("INKRS_CACHE_DIR", str(tmp_path / "cache"))
    pth = tmp_path / "step.pth"
    pth.write_bytes(b"a checkpoint")
    sha = inkrs._sha256(pth)
    monkeypatch.setattr(inkrs.importlib.util, "find_spec", lambda name: object())
    from inkrs import export_weights

    staged = []
    together = threading.Barrier(4, timeout=10)

    def slow(checkpoint, out_dir, name="hybrid_3d2d-seed42"):
        staged.append(out_dir)
        together.wait()  # every export is in flight at once
        out = _fake_export(out_dir, sha)
        out.write_bytes(b"x" * 1024)
        time.sleep(0.05)
        return out

    monkeypatch.setattr(export_weights, "export", slow)
    got, errors = [], []

    def run():
        try:
            got.append(inkrs._resolve_checkpoint(pth, tmp_path / "weights"))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(set(staged)) == 4, "two exports shared a staging directory"
    assert len(set(got)) == 1 and got[0].read_bytes() == b"x" * 1024
    assert [p.name for p in (tmp_path / "cache" / "weights").iterdir()] == [sha], "a staging directory was left"


def test_auto_export_lands_in_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("INKRS_CACHE_DIR", str(tmp_path / "cache"))
    pth = tmp_path / "step.pth"
    pth.write_bytes(b"a checkpoint")
    sha = inkrs._sha256(pth)
    monkeypatch.setattr(inkrs.importlib.util, "find_spec", lambda name: object())
    from inkrs import export_weights

    calls = []

    def fake(checkpoint, out_dir, name="hybrid_3d2d-seed42"):
        calls.append(checkpoint)
        return _fake_export(out_dir, sha)

    monkeypatch.setattr(export_weights, "export", fake)
    got = inkrs._resolve_checkpoint(pth, tmp_path / "weights")
    assert got == tmp_path / "cache" / "weights" / sha / "hybrid_3d2d-seed42.safetensors"
    assert inkrs._resolve_checkpoint(pth, tmp_path / "weights") == got
    assert len(calls) == 1, "the second call must reuse the cache"


def test_occupancy_modes():
    e = inkrs.Engine.__new__(inkrs.Engine)
    assert e._occupancy("seg.zarr", "auto") == ("villa", None, None)
    assert e._occupancy(np.zeros((17, 128, 128), np.uint8), "auto") == ("none", None, None)
    assert e._occupancy("seg.zarr", None) == ("none", None, None)
    assert e._occupancy("seg.zarr", "group.zarr") == ("given", "group.zarr", None)
    mode, path, level = e._occupancy("seg.zarr", np.ones((4, 4), dtype=bool))
    assert (mode, path, level.dtype, level.shape) == ("given", None, np.uint8, (4, 4))


def test_sweep_needs_a_name_for_an_in_memory_volume(tmp_path):
    e = inkrs.Engine.__new__(inkrs.Engine)
    # Engine.__init__ sets both from the loaded export: a villa reader, so no v8-in engine
    e.patch = 128
    e._v8 = None
    e.reader = "reader-v2"
    with pytest.raises(ValueError, match="name is needed"):
        e.sweep(np.zeros((66, 128, 128), np.uint8), range(1), out_dir=tmp_path)


# two real villa checkpoints of one architecture; the test exports both, so it runs only where villa, torch and the
# files are present: INKRS_TEST_PTH_A and INKRS_TEST_PTH_B name them (for example hybrid_3d2d-seed43 and TRACE v3B)
_PTH_PAIR = (
    Path(os.environ.get("INKRS_TEST_PTH_A", "")),
    Path(os.environ.get("INKRS_TEST_PTH_B", "")),
)


def test_auto_export_names_each_checkpoint_after_itself(tmp_path, monkeypatch):
    # the auto-export once named every .pth `hybrid_3d2d-seed42`, so two readers' maps took one name
    pytest.importorskip("torch")
    pytest.importorskip("vesuvius")
    missing = [p for p in _PTH_PAIR if not p.is_file()]
    if missing:
        pytest.skip(f"checkpoints not present: {missing}")
    monkeypatch.setenv("INKRS_CACHE_DIR", str(tmp_path / "cache"))
    names = []
    for pth in _PTH_PAIR:
        export = inkrs._resolve_checkpoint(pth, tmp_path / "weights")
        meta = json.loads(export.with_suffix(".json").read_text())
        assert meta["checkpoint_sha256"] == inkrs._sha256(pth)
        assert meta["reader"] == export.stem
        names.append(meta["reader"])
    assert names[0] != names[1], names
    assert names[0] == pytest.importorskip("inkrs.export_weights").default_name(_PTH_PAIR[0])
