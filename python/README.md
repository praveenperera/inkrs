# inkrs (Python)

inkrs runs villa's flat ink readers on the GPU, from Python: `hybrid_3d2d-seed42` and the readers with its architecture (`reader-v2`, Reader V3, ink_9um seed 43, dense_native, TRACE v3B and Rectina M1, M2 and M4). Give it the same input, layers, direction and stride as villa's `vesuvius.ink_detection.inference.infer` and it returns the uint8 maps that `infer` writes, or writes villa's TIFFs itself. `inkrs.Engine` also runs the v8-in exports, `inkrs.hecate` runs Hecate 9.6 µm and `inkrs.ink3d` runs villa's 3D ink models, each with the CLI's code. The model stays loaded on the GPU between calls, and `Engine.sweep` runs every depth and both directions of a segment in one pass.

## Install

On Linux x86-64:

```
pip install "inkrs[cu12]"
```

The wheel is for CPython 3.9 and newer, on glibc 2.28 or newer. It doesn't need Rust or a CUDA toolkit, because the kernels ship inside it as PTX. You do need:

- **An NVIDIA GPU of compute capability 7.5 or above** (T4 and RTX 20, RTX 30, 40 and 50, A100, H100) **and a driver of r525 or newer**.
- **cuDNN 9 and cuBLAS 12.** The `cu12` extra installs them from pip (`nvidia-cudnn-cu12`, `nvidia-cublas-cu12`). Without it, inkrs uses the copies already on the system, or the ones a CUDA 12 build of PyTorch installed. A CUDA 13 PyTorch ships cuBLAS 13 under a name inkrs doesn't look for, so there add `pip install nvidia-cublas-cu12`.

The wheel has three PTX tiers, which the driver compiles on first use: sm_75 for Turing, sm_80 for every GPU from 8.0 up, and sm_120 (r570 or newer) for the RTX 50 series. The engine loads the highest tier the GPU runs, and `INKRS_PTX_TIER=sm80` or `sm75` forces a lower one. `inkrs.build_info()` reports each tier and its minimum driver. Hecate in bf16 and the 3D ink U-Net need 8.0 or above, so they're refused on Turing. It's tested on an RTX 5090 (every tier), a Tesla T4 (sm75), and an A100, an A10, an L4 and an RTX 4090 (sm80, the readers).

### Which cuDNN and cuBLAS it loads

inkrs takes the system copies first (the linker path, `LD_LIBRARY_PATH`, or a copy already loaded, such as PyTorch's), then the pip packages. `INKRS_CUDA_LIBS=pip` loads the pip copies even when the system has its own, and `INKRS_CUDA_LIBS=cudnn` (or `cublas`) does that for one of them. The engine tries the unversioned name (`libcublas.so`) first, so a system copy under that name still wins; when it would, the forced load raises instead of running on the other copy. If a library is found nowhere, or a pip copy fails to load, `inkrs.Engine` raises `inkrs.CudaLibraryError` naming the package to install. `inkrs.load_cuda_libraries()` and `eng.load_report["cuda_libraries"]` show the file the engine calls for each.

### Windows

There's an x86-64 Windows wheel too, built with `maturin build --release` in the repo's `python/` directory. It builds, and it imports only Python's stable ABI DLL and Windows system DLLs, but it hasn't been installed or run on Windows.

There the libraries are `cudnn64_9.dll` and `cublas64_12.dll`, and the `cu12` extra installs them as well. When the pip packages are installed, inkrs loads them from their `bin` directories before the engine starts, so they win over a CUDA toolkit on `PATH`, and `INKRS_CUDA_LIBS=pip` only makes them required. NVRTC, which cuDNN uses for some of its engines, loads from the `nvidia-cuda-nvrtc-cu12` package's `bin` directory when that's installed. Without the pip packages, the engine looks in the executable's directory, System32, each absolute `PATH` entry, then `%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin`.

It never loads a library from the current directory. It raises `inkrs.CudaLibraryError` when a copy there or behind a relative `PATH` entry would load instead, or when a file it tries first, such as a `cudnn.dll` on `PATH`, would load a second copy beside the pip one. The CLI also takes the current directory and `PATH` out of its own DLL search at startup. The Python package can't do that for your interpreter, so a DLL written into the current directory after the engine's checks, under a name it tries first, could still load.

### macOS

Build the wheel on a Mac with `python/build-wheel.sh`, then:

```
pip install inkrs-0.2.0-cp39-abi3-macosx_26_0_arm64.whl
```

On Apple silicon with macOS 26 or newer the engine runs on Metal. It needs no CUDA libraries and no Xcode, because the Metal shaders ship as source and compile when the engine opens. Pass `device=0`, `"mps"`, `"mps:0"` or a torch MPS device; the engine uses the system's default GPU. `algo` does nothing there, and `inkrs.build_info()` reports `"backend": "metal"` and no PTX. Only the M4 Pro has been tested.

Experimental, and reader-v2 only: the engine can run most batches' 2D network on the Neural Engine through Core ML while the GPU runs every batch's 3D stem. Build the Core ML bodies once with `metal/coreml_body.py`, set `INKRS_COREML_ANE` to the `fp16` `.mlmodelc` it prints, and run at `--batch-size 8` (`batch_size=8`). The engine refuses a body built from another checkpoint, batch or precision. The maps stay within villa's parity limits and are the same on every run, but differ from the GPU-only maps by up to a few 1/255 steps. `src/metal/coreml.rs` lists the tuning variables, including an opt-in CPU lane.

## Weights

inkrs reads a safetensors export of villa's checkpoint. Make it once, on the CPU, where villa and torch are installed:

```
python -m inkrs.export_weights step-075000.pth weights/
```

That writes `weights/hybrid_3d2d-seed42.safetensors` and its `.json` sidecar, and `inkrs.Engine` takes the `.safetensors` file directly.

It takes the `.pth` too. inkrs then looks for an export of those exact bytes (by SHA-256) in `weights_dir` (default `$INKRS_WEIGHTS_DIR`, else `./weights`), then in `~/.cache/inkrs/weights/<sha256>/` (or under `$XDG_CACHE_HOME`, or `$INKRS_CACHE_DIR`; `%LOCALAPPDATA%\inkrs\weights\<sha256>\` on Windows). If neither has one and torch and villa are installed, as in a villa environment, it exports the checkpoint into that cache once and reuses it from then on. Without them it raises `inkrs.UnsupportedError` with the command above.

## Use

```python
import inkrs

eng = inkrs.Engine("weights/hybrid_3d2d-seed42.safetensors", device=0)

# one villa `infer` call: --layer-start 25 --layer-end 42 --direction both
maps = eng.infer("segment.zarr", layer_start=25, layer_end=42, direction="both")
maps["forward"], maps["reverse"]          # uint8 [H, W]; forward is the blind side, reverse the reading side

# or write villa's files: out.tif and out_reverse.tif
eng.infer("segment.zarr", layer_start=25, layer_end=42, direction="both", output_tiff="out.tif")

# every depth d = -4..4, both directions, in one pass (layers from Z//2 + d - 8)
sweep = eng.sweep("segment.zarr", range(-4, 5))
sweep[(0, "reverse")]

# an in-memory volume: numpy or torch, uint8 [Z, H, W]; a C-contiguous array is read in place
maps = eng.infer(volume, layer_start=25, layer_end=42, occupancy="segment.zarr")
```

`occupancy` controls villa's low-resolution empty-tile skip. For a zarr path the default does what villa does. For an in-memory volume, pass the zarr group it came from (or one pyramid level as an array) to get villa's covered pixels. Pass nothing and only the per-tile skip runs, as villa does for a bare array.

`eng.load_report` and `eng.last_report` hold the load and per-call timings. The GIL is released while the engine loads and runs, and one call runs at a time per engine.

[API.md](API.md) has the full contract, including how it maps onto villa's `infer`.

## Limits

Input, a model or a GPU outside these limits raises `inkrs.UnsupportedError`, a subclass of `inkrs.InkrsError`. So a caller with a PyTorch path can fall back on `UnsupportedError` and treat any other `InkrsError` as a failure.

- In `Engine`, the `hybrid_3d2d-seed42` architecture and contract (its own weights or another reader's of the same architecture) or a v8-in export. Hecate runs through `inkrs.hecate` and 3D ink through `inkrs.ink3d`.
- On macOS, the villa readers, Hecate and the v8-in exports only. `inkrs.ink3d` and `tta=8` raise there.
- uint8 input only.
- The Hann blend only.
- The unpadded tile phase only.
- A volume under 4 GiB, with both spatial dimensions at least 128. That's measured on the array passed to the call, so a 17-layer window can cover up to about 15,800 px square.
- A GPU the embedded PTX serves, with a driver new enough to load it, or on macOS a GPU with Metal 4.

The maps match villa eager's to within 2/255, with the same covered pixels. Two processes can differ from each other by 1/255 on a few pixels when `algo="find"` (the default) times cuDNN's algorithms and picks differently. The timed picks are cached on disk (see [API.md](API.md)), so later processes on the same setup reuse them; `algo="heuristic"` makes the choice repeatable without the cache.

Either the Apache License, Version 2.0 or the MIT licence, at your option. The embedded PTX includes NVIDIA libdevice code; see `THIRD_PARTY_NOTICES`.
