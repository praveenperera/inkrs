"""inkrs: GPU-resident ink inference for villa's flat ink readers (`hybrid_3d2d-seed42`, `reader-v2`, Reader V3)

    import inkrs
    eng = inkrs.Engine("weights/hybrid_3d2d-seed42.safetensors")
    maps = eng.infer("segment.zarr", layer_start=25, layer_end=42, direction="both")
    maps["forward"], maps["reverse"]   # uint8 [H, W], villa's encoding

The same `Engine` runs the v8-in family (v8-in, A-hn seeds 0-2) when the export's sidecar names it:

    eng = inkrs.Engine("weights/v8in-ahn-s0.safetensors")
    prob = eng.infer(render_volume)["forward"]   # float32 [H, W], the export's map convention ("306b")

See API.md in the source tree for the full contract.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import _cuda
from ._inkrs import (
    CudaLibraryError,
    InkrsError,
    UnsupportedError,
    __version__,
    _Engine,
    _V8in,
    cubin_targets,
    ptx_targets,
)
from ._inkrs import fast_presets as _fast_presets
from ._inkrs import hecate as _hecate
from ._inkrs import ink3d as _ink3d
from ._inkrs import reader_stride as _reader_stride
from ._inkrs import write_tiff as _write_tiff

__all__ = [
    "CudaLibraryError",
    "Engine",
    "InkrsError",
    "UnsupportedError",
    "__version__",
    "build_info",
    "fast_presets",
    "hecate",
    "ink3d",
    "load_cuda_libraries",
    "write_tiff",
]

#: macOS builds run the engine on Metal: one GPU (device 0, or "mps"), no PTX and no CUDA libraries
_METAL = sys.platform == "darwin"

#: the engine indexes the volume passed to a call with u32 (Rust's `vol.data.len() < u32::MAX`)
_MAX_VOLUME_BYTES = 2**32 - 1

#: the tile edge of the 128-pixel readers (hybrid_3d2d-seed42, reader-v2) and the reader's depth; Reader V3's
#: tiles are 384 pixels, and `Engine.patch` gives the loaded reader's edge
PATCH = 128
DEPTH = 17
#: the default batch per tile edge, about the same GPU memory at either edge
_DEFAULT_BATCH = {128: 64, 384: 8}

def ink3d(
    checkpoint: str | os.PathLike,
    volume: str | os.PathLike,
    output: str | os.PathLike,
    *,
    device: Any = 0,
    bbox: Sequence[int] | None = None,
    overlap: float = 0.5,
    chunk: int = 128,
    max_accum_gb: float = 8.0,
    algo: str = "find",
    overwrite: bool = False,
    coverage: bool = False,
) -> dict[str, Any]:
    """Runs a villa 3D U-Net ink model (`ink_3d_dino_guided` v3 and its family) over a uint8 zarr volume

    `checkpoint` is the `.safetensors` export from `scripts/export_unet3d.py`, with its `.json` sidecar beside it;
    `volume` a zarr v2 array or a group whose array `0` is read; `output` the new zarr group, whose array `0` gets
    uint8 probabilities times 255 at level 0 only. `bbox` is `(z0, y0, x0, z1, y1, x1)` on the `chunk` grid, the
    whole volume when None. `coverage` also writes villa's coverage mask as group `coverage` inside `output`.
    Returns the run report (patches, timings, cuDNN plans). The GIL is released while it runs
    """
    if bbox is not None and len(bbox) != 6:
        raise ValueError("bbox takes six numbers: z0, y0, x0, z1, y1, x1")
    load_cuda_libraries()
    report = _ink3d(
        os.fspath(checkpoint), os.fspath(volume), os.fspath(output), _device_index(device),
        None if bbox is None else tuple(int(v) for v in bbox), float(overlap), int(chunk), float(max_accum_gb),
        algo, bool(overwrite), bool(coverage),
    )  # fmt: skip
    return json.loads(report)


# the oldest driver branch that JITs each PTX ISA version (CUDA toolkit release notes)
_PTX_ISA_DRIVER = {
    "6.5": 440, "7.0": 450, "8.0": 525, "8.1": 530, "8.2": 535, "8.3": 545, "8.4": 550, "8.5": 555, "8.6": 560, "8.7": 570,
    "8.8": 575, "9.0": 580, "9.1": 590,
}  # fmt: skip
#: the oldest driver branch the CUDA 12 builds of cuDNN 9 and cuBLAS run on, whatever the PTX needs
_CUDA12_LIBRARY_DRIVER = 525


def build_info() -> dict[str, Any]:
    """The embedded PTX tiers (target, ISA version and the driver each needs), the compiled kernels, the minimum driver, and the libraries loaded at run time

    The engine loads the highest tier the GPU runs (or the one ``INKRS_PTX_TIER`` names), so the minimum
    driver is the lowest any tier needs, raised to what the CUDA 12 libraries need; a GPU whose own tier
    needs more (sm_120 at ISA 8.7 needs r570) gets that from its own entry in ``ptx``. ``cubins`` lists the
    tiers compiled ahead for a GPU family (``arch``, which also covers later minor versions of its major) from
    the tier's PTX (``ptx``); a GPU with none JITs the PTX on its first run. On macOS the engine runs on Metal,
    so ``ptx`` and ``cubins`` are empty and ``min_driver`` and ``cuda_libraries`` are None
    """
    if _METAL:
        return {
            "version": __version__,
            "backend": "metal",
            "ptx": [],
            "cubins": [],
            "min_driver": None,
            "cuda_libraries": None,
        }
    targets = ptx_targets()
    drivers = [_PTX_ISA_DRIVER.get(version) for _, version in targets]
    return {
        "version": __version__,
        "ptx": [{"target": t, "isa": v, "min_driver": d} for (t, v), d in zip(targets, drivers)],
        "cubins": [{"arch": arch, "ptx": target} for target, arch in cubin_targets()],
        # None when an ISA version is newer than this table
        "min_driver": None if None in drivers else max(min(drivers), _CUDA12_LIBRARY_DRIVER),
        "backend": "cuda",
        "cuda_libraries": (
            {"cudnn": "cudnn64_9.dll", "cublas": "cublas64_12.dll"}
            if sys.platform == "win32"
            else {"cudnn": "libcudnn.so.9", "cublas": "libcublas.so.12 (or the system libcublas.so)"}
        ),
    }


def load_cuda_libraries() -> dict[str, str]:
    """Finds cuDNN 9 and cuBLAS 12 (system first, then the nvidia-* pip packages); returns the files the engine will call

    Raises CudaLibraryError when one is missing or will not load. On macOS there is nothing to load, and it
    returns an empty dict
    """
    if _METAL:
        return {}
    try:
        return _cuda.load_cuda_libraries()
    except _cuda.CudaLibraryMissing as e:
        raise CudaLibraryError(str(e)) from None


def write_tiff(path: str | os.PathLike, image: np.ndarray) -> None:
    """Writes a uint8 [H, W] map as villa's LZW BigTIFF, byte-identical to the CLI's"""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 2:
        raise TypeError(f"need a uint8 [H, W] map, got {image.dtype} {image.shape}")
    _write_tiff(os.fspath(path), np.ascontiguousarray(image))


def _device_index(device: Any) -> int:
    # a negative index is wrong on every backend, so it gets one message before the backend's own forms
    if isinstance(device, int) and device < 0:
        raise ValueError(f"device {device!r}: must not be negative")
    if _METAL:
        return _metal_device_index(device)
    if isinstance(device, int):
        return device
    if type(device).__name__ == "device" and hasattr(device, "type"):  # torch.device, without importing torch
        if device.type != "cuda":
            raise ValueError(f"inkrs runs on CUDA devices only, got {device}")
        return device.index or 0
    m = re.fullmatch(r"cuda(?::(\d+))?", str(device))
    if not m:
        raise ValueError(f"device {device!r}: need an int, 'cuda', 'cuda:N' or a torch CUDA device")
    return int(m.group(1) or 0)


def _metal_device_index(device: Any) -> int:
    # the Metal engine opens the system default GPU, so only device 0 means anything
    if type(device).__name__ == "device" and hasattr(device, "type"):  # torch.device, without importing torch
        if device.type != "mps" or (device.index or 0) != 0:
            raise ValueError(f"inkrs on macOS runs on the MPS device only, got {device}")
        return 0
    if device == 0 or (isinstance(device, str) and re.fullmatch(r"mps(?::0)?", device)):
        return 0
    raise ValueError(f"device {device!r}: on macOS need 0, 'mps', 'mps:0' or a torch MPS device")



def fast_presets() -> dict[str, Any]:
    """Every model's fast preset: {"readers": [[name, {"stride", "precision"}], ...], "hecate": {"stride", "precision", "batch"}}

    A reader that is not listed has no fast preset (docs/FAST.md in the source tree has the measured accuracy behind each)
    """
    return json.loads(_fast_presets())


def _stride(reader: str | None, patch: int, overlap: float | None, stride: int | None, fast: bool = False) -> int:
    """The stride on `patch` px tiles, resolved in the engine as the CLI resolves it (src/fast.rs reader_stride)"""
    return _reader_stride(reader, int(patch), bool(fast), None if stride is None else int(stride),
                          None if overlap is None else float(overlap))  # fmt: skip


def _select_layers(depth: int, start: int | None, end: int | None, reverse: bool) -> tuple[int, int]:
    """villa select_layer_indices for a 17-layer reader: the first layer and the step (src/sched.rs select_layers)"""
    s = 0 if start is None else start
    e = depth if end is None else end
    if s < 0:
        s += depth
    if e < 0:
        e += depth
    s, e = max(s, 0), min(e, depth)
    if e <= s:
        s, e = 0, depth
    n = e - s
    if n < DEPTH:
        raise ValueError(f"layers [{s}, {e}) hold {n} layers; the reader needs {DEPTH}")
    first = s + n // 2 - DEPTH // 2
    return (first + DEPTH - 1, -1) if reverse else (first, 1)


def _directions(direction: str) -> list[str]:
    if direction not in ("forward", "reverse", "both"):
        raise ValueError(f"direction {direction!r}: need forward, reverse or both")
    return ["forward", "reverse"] if direction == "both" else [direction]


def _suffixed(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _is_path(x: Any) -> bool:
    return isinstance(x, (str, os.PathLike))


def _check_shape(shape: tuple[int, ...], patch: int) -> None:
    """The volume limits for the engine's tile edge `patch`, from a uint8 shape alone"""
    if len(shape) != 3:
        raise ValueError(f"volume of shape {shape}: need [Z, H, W]")
    if shape[1] < patch or shape[2] < patch:
        raise UnsupportedError(f"volume {shape[1]}x{shape[2]} is smaller than a {patch} px tile")
    nbytes = int(np.prod(shape, dtype=object))
    if nbytes >= _MAX_VOLUME_BYTES:
        raise UnsupportedError(f"volume of {nbytes} bytes is over 4 GiB; the kernels index it with u32")


def _as_volume(volume: Any, patch: int) -> np.ndarray:
    """A C-contiguous uint8 [Z, H, W] numpy view of `volume`, copying only when it must

    Input the engine cannot run at tile edge `patch` (the engine's, 128 or 384; no default, so a 384 reader is
    never checked at 128) raises UnsupportedError here, before a CUDA tensor or a lazy array is read; the engine
    checks the same limits again
    """
    if type(volume).__module__.startswith("torch"):
        if str(volume.dtype) != "torch.uint8":
            raise UnsupportedError(f"volume dtype {volume.dtype}: inkrs reads uint8 only")
        _check_shape(tuple(volume.shape), patch)
        t = volume.detach()
        if t.device.type != "cpu":
            t = t.cpu()  # the skip schedule is built on the host
        volume = t.numpy()
    elif not isinstance(volume, np.ndarray):
        # a lazy array (zarr, dask): its metadata, before np.asarray reads it whole
        if hasattr(volume, "dtype") and np.dtype(volume.dtype) != np.uint8:
            raise UnsupportedError(f"volume dtype {volume.dtype}: inkrs reads uint8 only")
        shape = tuple(getattr(volume, "shape", ()))
        # dask reports an unknown extent as nan; such a shape is checked after the read instead
        if shape and all(isinstance(n, (int, np.integer)) for n in shape):
            _check_shape(shape, patch)
    arr = np.asarray(volume)
    if arr.dtype != np.uint8:
        raise UnsupportedError(f"volume dtype {arr.dtype}: inkrs reads uint8 only")
    _check_shape(arr.shape, patch)
    return np.ascontiguousarray(arr)


def _volume_depth(volume: Any) -> int:
    if _is_path(volume):
        root = Path(volume)
        for meta in (root / ".zarray", root / "0" / ".zarray"):
            if meta.is_file():
                return int(json.loads(meta.read_text())["shape"][0])
        raise InkrsError(f"{root}: no zarr v2 array")
    return int(volume.shape[0])


def _cache_dir() -> Path:
    """Where exports of .pth checkpoints (and the engine's cuDNN algorithm choices) are kept: $INKRS_CACHE_DIR, else
    $XDG_CACHE_HOME/inkrs, else ~/.cache/inkrs (%LOCALAPPDATA%\\inkrs on Windows, as the engine resolves it)"""
    if os.environ.get("INKRS_CACHE_DIR"):
        return Path(os.environ["INKRS_CACHE_DIR"])
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]) / "inkrs"
    # Windows sets no HOME; plans and compiled kernels belong to this machine's GPU, so not the roaming profile
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "inkrs"
    return Path.home() / ".cache" / "inkrs"


def _sha256(path: Path) -> str:
    """The sha256 of the checkpoint at `path`, from the digest cache when an entry records the file at its current size
    and modification time, else hashed and stored for the next start. The entry,
    `<cache>/checkpoint-sha256/<first 16 bytes of sha256(path), hex>.json`, is the one the engine reads and writes for a
    .pth passed to it, so the two share digests; a rewrite changes the modification time, so a cached digest never
    stands for other bytes at the same path"""
    path = path.resolve()
    st = path.stat()
    entry = _cache_dir() / "checkpoint-sha256" / f"{hashlib.sha256(os.fsencode(path)).hexdigest()[:32]}.json"
    want = {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    try:
        cached = json.loads(entry.read_text())
        if st.st_mtime_ns > 0 and isinstance(cached, dict) and {k: cached.get(k) for k in want} == want:
            return str(cached["sha256"])
    except (OSError, ValueError, KeyError):
        pass
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    digest = h.hexdigest()
    # best effort: a cache that cannot be written only costs the next start the hash again
    if st.st_mtime_ns > 0:
        try:
            entry.parent.mkdir(parents=True, exist_ok=True)
            tmp = entry.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps({**want, "sha256": digest}))
            os.replace(tmp, entry)
        except (OSError, ValueError):
            pass
    return digest


def _export_in(directory: Path, sha: str) -> Path | None:
    """The .safetensors export in `directory` whose sidecar names a checkpoint with this sha256"""
    if not directory.is_dir():
        return None
    for side in sorted(directory.glob("*.json")):
        try:
            meta = json.loads(side.read_text())
        except (OSError, ValueError):
            continue
        tensors = side.with_suffix(".safetensors")
        if isinstance(meta, dict) and meta.get("checkpoint_sha256") == sha and tensors.is_file():
            return tensors
    return None


def _export_patch(export: Path) -> int | None:
    """The tile edge the export's sidecar `crop_size` records, or None without a readable sidecar"""
    try:
        crop = json.loads(export.with_suffix(".json").read_text())["crop_size"]
        return int(crop[-1])
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def _resolve_checkpoint(checkpoint: Path, weights_dir: Path) -> Path:
    """The safetensors export to load for `checkpoint`

    A .pth is matched by its sha256 against the exports in `weights_dir`, then in the cache. When neither has it,
    it is exported into the cache (keyed by that sha256) if torch and villa import here, and UnsupportedError
    names the manual export otherwise.
    """
    if checkpoint.suffix != ".pth":
        return checkpoint
    if not checkpoint.is_file():
        raise InkrsError(f"checkpoint {checkpoint} does not exist")
    sha = _sha256(checkpoint)
    cache = _cache_dir() / "weights" / sha
    for directory in (weights_dir, cache):
        found = _export_in(directory, sha)
        if found is not None:
            return found
    manual = (
        f"export it once where villa and torch are installed, `python -m inkrs.export_weights {checkpoint} DIR`, "
        f"then pass weights_dir=DIR (or the .safetensors file)"
    )
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("vesuvius") is None:
        raise UnsupportedError(
            f"{checkpoint} has no safetensors export, and torch and villa are not importable to make one; {manual}"
        )
    from . import export_weights

    # export into a staging directory of its own beside the cache entry and rename it in, so a concurrent export (in
    # this process or another) or a crash never leaves a partial entry, and each export cleans up only its own
    cache.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{sha}.", suffix=".partial", dir=cache.parent))
    try:
        try:
            export_weights.export(checkpoint, staging)
        except export_weights.ExportUnsupported as e:
            # a checkpoint inkrs does not implement (not flat, keys or shapes that do not match its model, not
            # float32); a manual export would refuse it the same way, so no export is suggested
            raise UnsupportedError(f"cannot export {checkpoint} for inkrs: {e}") from e
        except ImportError as e:
            # a villa without the exporter's entry points
            raise UnsupportedError(f"cannot export {checkpoint} for inkrs: {type(e).__name__}: {e}; {manual}") from e
        except Exception as e:
            # disk full, out of memory, a corrupt checkpoint: real failures, which must not become a silent fallback
            raise InkrsError(f"exporting {checkpoint} for inkrs failed: {type(e).__name__}: {e}") from e
        try:
            os.replace(staging, cache)
        except OSError:
            # another export of the same checkpoint was renamed in first; keep that one
            if _export_in(cache, sha) is None:
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    found = _export_in(cache, sha)
    if found is None:
        raise InkrsError(f"no export of {checkpoint} in {cache} after exporting it")
    return found


#: tiles per batch by default for the v8-in family (24 x 64 x 64 tiles upsampled to 96 x 256 x 256, about
#: 0.4 GB of activations each); villa's readers take `_DEFAULT_BATCH` by tile edge
_V8IN_BATCH = 16


def _check_tta(tta: int) -> None:
    if isinstance(tta, bool) or tta not in (1, 8):
        raise ValueError(f"tta={tta!r}: 1 (off) or 8 (the 306B recipe's dihedral TTA)")


def _tta_defaults(occupancy: Any, coverage: bool) -> None:
    """tta=8 runs the recipe's reads, which skip only all-zero windows and write no coverage"""
    if not (isinstance(occupancy, str) and occupancy == "auto") or coverage:
        raise ValueError("tta=8 runs no occupancy scan and writes no coverage; leave occupancy and coverage at their defaults")


def _architecture(export: Path) -> str:
    """`v8in` when the export's sidecar names the v8-in architecture, else `villa`"""
    try:
        side = json.loads(export.with_suffix(".json").read_text())
    except (OSError, ValueError):
        return "villa"
    return "v8in" if side.get("architecture_family") == "v8in" else "villa"


class Engine:
    """The reader resident on one GPU, reusable across volumes, depths and directions

    The export's sidecar picks the network: villa's readers, or the v8-in family (v8-in, A-hn seeds 0-2),
    whose maps are float32 probabilities stitched by the export's map convention (see `infer`).

    One call runs at a time; calls from other threads wait. The GIL is released while a villa reader loads
    and runs; a v8-in engine runs on the first visible GPU and keeps the GIL.
    """

    def __init__(
        self,
        checkpoint: str | os.PathLike,
        *,
        device: Any = 0,
        batch_size: int | None = None,
        algo: str = "find",
        max_workspace_mib: int = 1024,
        weights_dir: str | os.PathLike | None = None,
        precision: str = "fp16",
    ) -> None:
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        if precision not in ("fp16", "fp32"):
            raise ValueError(f"precision {precision!r}: need fp16 or fp32")
        if precision == "fp32" and not _METAL:
            # refused before any CUDA library loads, so a caller that falls back on UnsupportedError always can
            raise UnsupportedError(
                "reader precision fp32 runs on the Metal backend only: the CUDA engine's reader kernels are FP16, "
                "as villa autocasts to FP16 on CUDA"
            )
        #: the reader's numerics: "fp16" (the default, villa's FP16 autocast) or "fp32" (Metal only, villa's
        #: float32 run on MPS); never read from the checkpoint
        self.precision = precision
        if weights_dir is None:
            weights_dir = os.environ.get("INKRS_WEIGHTS_DIR", "weights")
        self.device = _device_index(device)
        #: the safetensors export loaded: `checkpoint` itself, or the export found or made for a .pth
        self.weights = _resolve_checkpoint(Path(checkpoint), Path(weights_dir))
        #: `villa` or `v8in`, from the export's sidecar
        self.architecture = _architecture(self.weights)
        if batch_size is None:
            batch_size = _V8IN_BATCH if self.architecture == "v8in" else _DEFAULT_BATCH.get(_export_patch(self.weights), 64)
        self.batch_size = int(batch_size)
        #: the cuDNN and cuBLAS files the engine calls
        self.cuda_libraries = load_cuda_libraries()
        self._v8 = None
        #: v8-in only: the export's default map convention, "release" or "306b"; None for villa's readers
        self.convention: str | None = None
        if self.architecture == "v8in":
            if self.device != 0:
                raise UnsupportedError("a v8-in engine runs on the first visible GPU; set CUDA_VISIBLE_DEVICES instead")
            if precision != "fp16":
                raise UnsupportedError("precision applies to villa's readers only; v8-in runs its own FP16 numerics")
            self._v8 = _V8in(os.fspath(self.weights), self.batch_size, algo, int(max_workspace_mib))
            self.reader = self._v8.reader()
            self.convention = self._v8.convention()
            self.load_report = {
                "reader": self.reader, "convention": self.convention, "weights": os.fspath(self.weights),
                "cuda_libraries": self.cuda_libraries, "mapped_libraries": _cuda.mapped_cuda_libraries(),
            }  # fmt: skip
            return
        self._engine = _Engine(
            os.fspath(self.weights),
            self.device,
            self.batch_size,
            algo,
            int(max_workspace_mib),
            os.fspath(weights_dir),
            precision,
        )
        self.load_report: dict[str, Any] = json.loads(self._engine.load_report())
        self.load_report["weights"] = os.fspath(self.weights)
        #: the reader name the export's sidecar records (`hybrid_3d2d-seed42`, `reader-v2`), for output file names;
        #: None for an export without a sidecar
        self.reader: str | None = self.load_report["reader"]
        #: the loaded reader's tile edge in pixels: 128, or 384 for Reader V3
        self.patch: int = int(self._engine.patch())
        self.load_report["cuda_libraries"] = self.cuda_libraries
        self.load_report["mapped_libraries"] = _cuda.mapped_cuda_libraries()

    def _villa_only(self, what: str) -> None:
        if self._v8 is not None:
            raise UnsupportedError(f"{what} runs villa's readers; a v8-in export maps through Engine.infer")

    def _memory_info(self) -> tuple[int, int]:
        """The device's (free, total) memory in bytes"""
        self._villa_only("Engine._memory_info")
        return self._engine.memory_info()

    @property
    def last_report(self) -> dict[str, Any] | None:
        """The last call's report: for villa's readers the pass report (timings, tile counts per map, the
        occupancy scan); for v8-in the last map's (tiles admitted, GPU and wall time, the conv plans)"""
        text = (self._v8 or self._engine).last_report()
        return None if text is None else json.loads(text)

    def run(
        self,
        volume: Any,
        windows: Sequence[tuple[int, int]],
        *,
        stride: int | None = None,
        occupancy: Any = "auto",
        coverage: bool = False,
        outputs: Sequence[str | os.PathLike] | None = None,
        coverage_outputs: Sequence[str | os.PathLike] | None = None,
        writers: int = 4,
    ) -> list[Any]:
        """Runs each (first_layer, step) window over `volume` in one pass

        Returns one entry per window: the uint8 [H, W] map, or (map, coverage) with coverage=True, or with
        `outputs` the path written (and with `coverage_outputs`, the pair of paths). The stride defaults to
        half the reader's tile edge (villa's overlap 0.5).
        """
        self._villa_only("Engine.run")
        stride = _stride(self.reader, self.patch, None, stride)
        windows = [(int(f), int(s)) for f, s in windows]
        if not windows:
            raise ValueError("no windows")
        for f, s in windows:
            if s not in (1, -1):
                raise ValueError(f"window ({f}, {s}): step must be 1 or -1")
        if outputs is not None and len(outputs) != len(windows):
            raise ValueError("outputs needs one path per window")
        if coverage_outputs is not None and (not coverage or len(coverage_outputs) != len(windows)):
            raise ValueError("coverage_outputs needs coverage=True and one path per window")
        if writers < 1:
            raise ValueError("writers must be positive")
        specs = [
            (
                f,
                s,
                None if outputs is None else os.fspath(outputs[i]),
                bool(coverage),
                None if coverage_outputs is None else os.fspath(coverage_outputs[i]),
            )
            for i, (f, s) in enumerate(windows)
        ]
        occ = self._occupancy(volume, occupancy)
        if _is_path(volume):
            got = self._engine.run_zarr(os.fspath(volume), specs, int(stride), *occ, writers)
        else:
            arr = _as_volume(volume, self.patch)
            label = f"array {arr.shape[0]}x{arr.shape[1]}x{arr.shape[2]}"
            got = self._engine.run_array(arr, specs, int(stride), *occ, writers, label)
        out: list[Any] = []
        for i, (m, c) in enumerate(got):
            m = Path(outputs[i]) if outputs is not None else m
            if coverage:
                c = Path(coverage_outputs[i]) if coverage_outputs is not None else c
                out.append((m, c))
            else:
                out.append(m)
        return out

    @staticmethod
    def _occupancy(volume: Any, occupancy: Any) -> tuple[str, str | None, np.ndarray | None]:
        """(mode, group path, level array) for the native call; mode is 'villa', 'none' or 'given'"""
        if isinstance(occupancy, str) and occupancy == "auto":
            return ("villa" if _is_path(volume) else "none"), None, None
        if occupancy is None:
            return "none", None, None
        if _is_path(occupancy):
            return "given", os.fspath(occupancy), None
        level = np.asarray(occupancy)
        if level.dtype == np.bool_:
            level = level.view(np.uint8)
        if level.dtype != np.uint8 or level.ndim not in (2, 3):
            raise TypeError(f"occupancy level {level.dtype} {level.shape}: need a uint8 or bool [h, w] or [d, h, w]")
        return "given", None, np.ascontiguousarray(level)

    def infer(
        self,
        volume: Any,
        *,
        layer_start: int | None = None,
        layer_end: int | None = None,
        direction: str = "forward",
        overlap: float | None = None,
        stride: int | None = None,
        fast: bool = False,
        blend_mode: str = "hann",
        occupancy: Any = "auto",
        coverage: bool = False,
        output_tiff: str | os.PathLike | None = None,
        writers: int = 4,
        convention: str | None = None,
        tta: int = 1,
    ) -> dict[str, Any]:
        """One villa `infer` call: {"forward": map, "reverse": map} for the directions asked for

        With coverage=True also "<direction>_coverage". With output_tiff, writes villa's files (OUT.tif, and
        OUT_reverse.tif for the reverse map of a "both" run) and returns their paths instead. The stride wins over
        the overlap (default 0.5); fast=True takes the reader's fast preset and raises ValueError for a reader with
        none, which is every reader so far (FAST.md).

        A v8-in export returns float32 [H, W] probability maps instead, stitched by `convention`: "release"
        (ink8um's `predict_stack`) or "306b" (the 306B A-hn runner), default the export's own
        (`Engine.convention`). It reads 24 layers from `layer_start`, default the central 24; `layer_end`
        may only repeat `layer_start + 24`; the convention fixes the tiling, window and coverage, so
        overlap, stride, fast, blend_mode, occupancy and coverage keep their defaults. `output_tiff` must then
        end in `.npy` (float32).

        tta=8 runs the PHerc. 0306B recipe's 8-fold dihedral test-time augmentation: each map is read twice,
        as stored and turned a quarter, every tile runs as four mirrored copies, and the result is a float32
        [H, W] map in [0, 1] (`output_tiff` must then end in `.npy`). It runs no occupancy scan, as the recipe,
        so occupancy must stay "auto", and it takes no coverage. Default 1, villa's single read.
        """
        _check_tta(tta)
        if self._v8 is not None:
            return self._infer_v8(
                volume, layer_start, layer_end, direction, convention, output_tiff,
                overlap is None and stride is None and not fast and blend_mode == "hann" and not coverage
                and isinstance(occupancy, str) and occupancy == "auto" and tta == 1,
            )  # fmt: skip
        if convention is not None:
            raise ValueError("convention applies to v8-in exports only")
        step = _stride(self.reader, self.patch, None if stride is not None else overlap, stride, fast)
        blend = blend_mode
        if blend == "auto":
            blend = "constant" if step >= self.patch else "hann"
        if blend != "hann":
            raise ValueError(f"blend_mode {blend_mode!r} resolves to {blend}; only hann is supported")
        dirs = _directions(direction)
        depth = _volume_depth(volume)
        windows = [_select_layers(depth, layer_start, layer_end, d == "reverse") for d in dirs]
        outputs = cov_outputs = None
        if tta == 8 and output_tiff is not None and Path(output_tiff).suffix != ".npy":
            raise ValueError("tta=8 maps are float32; give an output path ending in .npy")
        if output_tiff is not None:
            base = Path(output_tiff)
            base.parent.mkdir(parents=True, exist_ok=True)
            # villa suffixes only the reverse map of a two-direction run
            outputs = [_suffixed(base, "_reverse") if d == "reverse" and direction == "both" else base for d in dirs]
            if coverage:
                cov_outputs = [_suffixed(p, ".coverage") for p in outputs]
        if tta == 8:
            _tta_defaults(occupancy, coverage)
            return dict(zip(dirs, self._run_tta(volume, windows, step, outputs)))
        got = self.run(
            volume, windows, stride=step, occupancy=occupancy, coverage=coverage, outputs=outputs,
            coverage_outputs=cov_outputs, writers=writers,
        )  # fmt: skip
        result: dict[str, Any] = {}
        for d, g in zip(dirs, got):
            if coverage:
                result[d], result[f"{d}_coverage"] = g
            else:
                result[d] = g
        return result

    def _run_tta(
        self, volume: Any, windows: Sequence[tuple[int, int]], stride: int, outputs: Sequence[Path] | None
    ) -> list[Any]:
        """One 8-fold TTA pass: per window the float32 [H, W] map, or with `outputs` the .npy path written"""
        specs = [(int(f), int(s), None if outputs is None else os.fspath(outputs[i])) for i, (f, s) in enumerate(windows)]
        if _is_path(volume):
            got = self._engine.run_zarr_tta(os.fspath(volume), specs, int(stride))
        else:
            arr = _as_volume(volume, self.patch)
            label = f"array {arr.shape[0]}x{arr.shape[1]}x{arr.shape[2]}"
            got = self._engine.run_array_tta(arr, specs, int(stride), label)
        return [Path(outputs[i]) if outputs is not None else m for i, m in enumerate(got)]

    def _infer_v8(self, volume, layer_start, layer_end, direction, convention, output, defaults) -> dict[str, Any]:
        if not defaults:
            raise ValueError("v8-in's map convention fixes the tiling, window and coverage; leave overlap, stride, "
                             "fast, blend_mode, occupancy, coverage and tta at their defaults")  # fmt: skip
        if convention not in (None, "release", "306b"):
            raise ValueError(f"convention {convention!r}: need 'release' or '306b'")
        if layer_start is not None and layer_start < 0:
            raise ValueError("layer_start must be >= 0 for v8-in")
        if layer_end is not None and (layer_start is None or layer_end != layer_start + 24):
            raise ValueError("v8-in reads 24 layers: give layer_start only, or layer_end = layer_start + 24")
        if output is not None and Path(output).suffix != ".npy":
            raise ValueError("v8-in maps are float32; give an output path ending in .npy")
        if not _is_path(volume):
            volume = np.asarray(volume)
            if volume.dtype != np.uint8 or volume.ndim != 3:
                raise UnsupportedError(f"volume {volume.dtype} {volume.shape}: need uint8 [Z, H, W]")
            volume = np.ascontiguousarray(volume)
        result: dict[str, Any] = {}
        for d in _directions(direction):
            reverse = d == "reverse"
            if _is_path(volume):
                m = self._v8.run_path(os.fspath(volume), convention, layer_start, reverse)
            else:
                m = self._v8.run_array(volume, convention, layer_start, reverse)
            if output is not None:
                base = Path(output)
                base.parent.mkdir(parents=True, exist_ok=True)
                path = _suffixed(base, "_reverse") if reverse and direction == "both" else base
                np.save(path, m)
                m = path
            result[d] = m
        return result

    def sweep(
        self,
        volume: Any,
        depths: Iterable[int],
        *,
        direction: str = "both",
        stride: int | None = None,
        overlap: float | None = None,
        fast: bool = False,
        occupancy: Any = "auto",
        coverage: bool = False,
        out_dir: str | os.PathLike | None = None,
        name: str | None = None,
        writers: int = 4,
        tta: int = 1,
    ) -> dict[tuple[int, str], Any]:
        """Every depth and direction in one pass: {(d, "forward"): map, (d, "reverse"): map, ...}

        Depth d reads the 17 layers from Z//2 + d - 8. With out_dir, writes the CLI's
        OUT_DIR/<name>__<reader>__L<Z>s<S>[_reverse].tif (coverage under the sibling coverage/) and returns
        the paths. tta=8 is `infer`'s: float32 maps, written as `.npy` in place of `.tif`.
        """
        self._villa_only("Engine.sweep")
        _check_tta(tta)
        if tta == 8:
            _tta_defaults(occupancy, coverage)
        step = _stride(self.reader, self.patch, overlap, stride, fast)
        dirs = _directions(direction)
        depth = _volume_depth(volume)
        keys: list[tuple[int, str]] = []
        windows: list[tuple[int, int]] = []
        files: list[str] = []
        for d in depths:
            s = depth // 2 + int(d) - 8
            if s < 0 or s + DEPTH > depth:
                raise ValueError(f"depth {d}: layers {s}..{s + DEPTH} outside the volume's {depth}")
            for direc in dirs:
                keys.append((int(d), direc))
                windows.append(_select_layers(depth, s, s + DEPTH, direc == "reverse"))
                if out_dir is not None:
                    if name is None:
                        if not _is_path(volume):
                            raise ValueError("name is needed with out_dir for an in-memory volume")
                        name = Path(volume).stem
                    if self.reader is None:
                        raise InkrsError(f"{self.weights} has no sidecar naming its reader, which out_dir names files by")
                    ext = "npy" if tta == 8 else "tif"
                    files.append(f"{name}__{self.reader}__L{depth}s{s}{'_reverse' if direc == 'reverse' else ''}.{ext}")
        if not keys:
            raise ValueError("no depths")
        outputs = cov_outputs = None
        if out_dir is not None:
            ink = Path(out_dir)
            ink.mkdir(parents=True, exist_ok=True)
            outputs = [ink / f for f in files]
            if coverage:
                cov = ink.with_name("coverage")
                cov.mkdir(parents=True, exist_ok=True)
                cov_outputs = [cov / f for f in files]
        if tta == 8:
            return dict(zip(keys, self._run_tta(volume, windows, step, outputs)))
        got = self.run(
            volume, windows, stride=step, occupancy=occupancy, coverage=coverage, outputs=outputs,
            coverage_outputs=cov_outputs, writers=writers,
        )  # fmt: skip
        return dict(zip(keys, got))


def hecate(
    volume: str | os.PathLike,
    checkpoint: str | os.PathLike,
    output: str | os.PathLike,
    *,
    spacing_um: float,
    direction: str = "forward",
    stride: int | None = None,
    overlap: float | None = None,
    precision: str | None = None,
    tf32: bool = True,
    batch_size: int | None = None,
    fast: bool = False,
    algo: str = "find",
    max_workspace_mib: int = 1024,
    weights_dir: str | os.PathLike | None = None,
    device: Any = 0,
) -> dict[str, Any]:
    """`inkrs hecate`: Hecate 9.6 um's 2D ink map of the central 16 planes, as `hecate.py --output` writes it

    `volume` is a uint8 [Z, Y, X] `.npy` or a local zarr v2 array or group (array 0), at `spacing_um` within 2% of
    9.6. Writes OUT.png and OUT.json (`output` must end in .png and not exist), and with direction="both" also
    OUT_reverse.png and its JSON. The defaults are hecate.py's: stride 32 (or `overlap`), fp32 with TF32 convs
    (`tf32=False` for strict fp32, `precision="bf16"` for its bf16 autocast), and inkrs's batch 8. fast=True takes
    Hecate's fast preset, bf16 at stride 32 and batch 32 (FAST.md), and refuses an explicit stride, overlap or
    precision. Returns the run report (the CLI's `--report` JSON) with "outputs": {"forward": Path, ...}
    """
    if direction not in ("forward", "reverse", "both"):
        raise ValueError(f"direction {direction!r}: need forward, reverse or both")
    if weights_dir is None:
        weights_dir = os.environ.get("INKRS_WEIGHTS_DIR", "weights")
    out = Path(output)
    outputs = {"reverse": out} if direction == "reverse" else {"forward": out}
    if direction == "both":
        outputs["reverse"] = _suffixed(out, "_reverse")
    for name, value in (("stride", stride), ("batch_size", batch_size), ("max_workspace_mib", max_workspace_mib)):
        if value is not None and int(value) < 0:
            raise ValueError(f"{name} {value}: must not be negative")
    device_index = _device_index(device)

    def run(check_only: bool) -> str:
        return _hecate(
            os.fspath(volume), os.fspath(checkpoint), [(d == "reverse", os.fspath(p)) for d, p in outputs.items()],
            float(spacing_um), fast=bool(fast), stride=None if stride is None else int(stride),
            overlap=None if overlap is None else float(overlap), precision=precision, tf32=bool(tf32),
            batch_size=None if batch_size is None else int(batch_size), algo=algo,
            max_workspace_mib=int(max_workspace_mib), weights_dir=os.fspath(weights_dir), device=device_index,
            check_only=check_only,
        )  # fmt: skip

    # every refusal the engine makes before the GPU, raised as ValueError before any CUDA library loads, so a
    # bad request reads the same on a machine without them
    run(check_only=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    load_cuda_libraries()
    report = json.loads(run(check_only=False))
    report["outputs"] = outputs
    return report
