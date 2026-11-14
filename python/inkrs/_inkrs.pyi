"""Type stub for the ``inkrs._inkrs`` Rust extension module (python/src/lib.rs and python/src/v8in.rs)

The public API is the ``inkrs`` package, which converts volumes and options before calling these
"""

from collections.abc import Sequence
from os import PathLike
from typing import final

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

_Path: TypeAlias = str | PathLike[str]
#: one output window: first layer, layer step (1 or -1), map path or None to return it, whether to make the
#: coverage mask, and the coverage path or None to return it
_WindowSpec: TypeAlias = tuple[int, int, _Path | None, bool, _Path | None]
#: one 8-fold TTA window: first layer, layer step, and the float32 `.npy` path or None to return the map
_TtaWindow: TypeAlias = tuple[int, int, _Path | None]

__all__ = [
    "CudaLibraryError",
    "InkrsError",
    "UnsupportedError",
    "_Engine",
    "_V8in",
    "__version__",
    "cubin_targets",
    "fast_presets",
    "hecate",
    "ink3d",
    "ptx_targets",
    "reader_stride",
    "write_tiff",
]

__version__: str

class InkrsError(RuntimeError):
    """An inkrs engine failure; the message holds the cause chain"""

class UnsupportedError(InkrsError):
    """Input, a model or a GPU this engine does not implement; a caller can fall back to its own path"""

class CudaLibraryError(InkrsError):
    """The NVIDIA driver, cuDNN 9 or cuBLAS could not be loaded or used"""

@final
class _Engine:
    """The flat-reader engine, behind a lock so one call runs at a time"""

    def __new__(
        cls,
        checkpoint: _Path,
        device: int,
        batch_size: int,
        algo: str,
        max_workspace_mib: int,
        weights_dir: _Path,
        precision: str,
    ) -> _Engine: ...
    def load_report(self) -> str: ...
    def patch(self) -> int: ...
    def memory_info(self) -> tuple[int, int]: ...
    def last_report(self) -> str | None: ...
    def run_zarr(
        self,
        path: _Path,
        windows: Sequence[_WindowSpec],
        stride: int,
        occupancy_mode: str,
        occupancy: _Path | None,
        occupancy_level: npt.NDArray[np.uint8] | None,
        writers: int,
    ) -> list[tuple[npt.NDArray[np.uint8] | None, npt.NDArray[np.uint8] | None]]: ...
    def run_array(
        self,
        volume: npt.NDArray[np.uint8],
        windows: Sequence[_WindowSpec],
        stride: int,
        occupancy_mode: str,
        occupancy: _Path | None,
        occupancy_level: npt.NDArray[np.uint8] | None,
        writers: int,
        label: str,
    ) -> list[tuple[npt.NDArray[np.uint8] | None, npt.NDArray[np.uint8] | None]]: ...
    def run_zarr_tta(
        self, path: _Path, windows: Sequence[_TtaWindow], stride: int
    ) -> list[npt.NDArray[np.float32] | None]: ...
    def run_array_tta(
        self, volume: npt.NDArray[np.uint8], windows: Sequence[_TtaWindow], stride: int, label: str
    ) -> list[npt.NDArray[np.float32] | None]: ...

@final
class _V8in:
    """The v8-in family engine; it stays on the thread that made it"""

    def __new__(cls, checkpoint: _Path, batch_size: int, algo: str, max_workspace_mib: int) -> _V8in: ...
    def reader(self) -> str: ...
    def convention(self) -> str: ...
    def last_report(self) -> str | None: ...
    def run_array(
        self, volume: npt.NDArray[np.uint8], convention_name: str | None, layer0: int | None, reverse: bool
    ) -> npt.NDArray[np.float32]: ...
    def run_path(
        self, path: _Path, convention_name: str | None, layer0: int | None, reverse: bool
    ) -> npt.NDArray[np.float32]: ...

def write_tiff(path: _Path, image: npt.NDArray[np.uint8]) -> None: ...
def ptx_targets() -> list[tuple[str, str]]: ...
def cubin_targets() -> list[tuple[str, str]]: ...
def fast_presets() -> str: ...
def reader_stride(reader: str | None, patch: int, fast: bool, stride: int | None, overlap: float | None) -> int: ...
def hecate(
    input: _Path,
    checkpoint: _Path,
    maps: Sequence[tuple[bool, _Path]],
    spacing_um: float,
    *,
    fast: bool,
    stride: int | None,
    overlap: float | None,
    precision: str | None,
    tf32: bool,
    batch_size: int | None,
    algo: str,
    max_workspace_mib: int,
    weights_dir: _Path,
    device: int,
    check_only: bool,
) -> str: ...

# on macOS this takes any arguments and raises UnsupportedError
def ink3d(
    checkpoint: _Path,
    input: _Path,
    output: _Path,
    device: int,
    # six numbers: z0, y0, x0, z1, y1, x1
    bbox: Sequence[int] | None,
    overlap: float,
    chunk: int,
    max_accum_gb: float,
    algo: str,
    overwrite: bool,
    coverage: bool,
) -> str: ...
