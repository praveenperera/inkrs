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
    missing = None
    try:
        inkrs.load_cuda_libraries()
    except inkrs.CudaLibraryError as e:
        missing = str(e)

    status = _inkrs.cli(["inkrs", *sys.argv[1:]])
    if status != 0 and missing is not None:
        print(f"note: {missing}", file=sys.stderr)
    sys.exit(status)


if __name__ == "__main__":
    main()
