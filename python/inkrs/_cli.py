"""The wheel's `inkrs` console script: the same command line as the `inkrs` binary, run in this process

    inkrs nnunet predict --model weights/m7.safetensors --input VOLUME.zarr --output PRED.zarr

It runs the binary's own code (`inkrs::cli::run`) after the start-up the binary does itself: the CUDA driver's JIT
cache in the inkrs cache, and cuDNN and cuBLAS found the way the Python API finds them, so a `cu12` install's pip
copies serve the commands too. The cuDNN and cuBLAS the engine calls are the first its names open (see `_cuda`)
"""
from __future__ import annotations

import os
import signal
import sys


def main() -> None:
    # the commands run in Rust with the GIL released and never return to Python's SIGINT check; Ctrl-C ends the
    # process at once, as it does the binary
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    # imported after the signal so an early Ctrl-C ends it the same way
    import inkrs
    from inkrs import _inkrs

    # the binary's first step (src/main.rs): set before any CUDA call, since the driver reads it once
    cache = _inkrs.cuda_cache_path()
    if cache is not None:
        # a failure leaves it to the driver, as in the binary
        try:
            os.makedirs(cache, exist_ok=True)
        except OSError:
            pass
        os.environ["CUDA_CACHE_PATH"] = os.fspath(cache)

    # a command that needs cuDNN or cuBLAS fails in the engine when they are missing; the loader's message names
    # what it searched, so it follows the engine's error
    args = sys.argv[1:]
    missing = None
    try:
        inkrs.load_cuda_libraries()
    except inkrs.CudaLibraryError as e:
        missing = str(e)

    # INKRS_CUDA_LIBS picks the copies to run, so a command that cannot have them stops instead of running on
    # another copy, as the Python API does; help and version open no library
    forced = os.environ.get("INKRS_CUDA_LIBS", "auto") not in ("", "auto")
    if missing is not None and forced and not _opens_no_library(args):
        print(f"inkrs: {missing}", file=sys.stderr)
        sys.exit(1)

    status = _inkrs.cli(["inkrs", *args])
    if status != 0 and missing is not None:
        print(f"note: {missing}", file=sys.stderr)
    sys.exit(status)


def _opens_no_library(args: list[str]) -> bool:
    """Whether the command line only asks for help or the version, which run without cuDNN or cuBLAS"""
    return not args or args[0] == "help" or any(a in ("-h", "--help", "-V", "--version") for a in args)


if __name__ == "__main__":
    main()
