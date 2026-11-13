# inkrs Python API

The `inkrs` module runs the inkrs engine in-process. It runs villa's `hybrid_3d2d-seed42` ink reader and the readers with its architecture (`reader-v2`, Reader V3, ink_9um seed 43, dense_native, TRACE v3B, Rectina M4) with the model resident on the GPU, and returns or writes the maps villa's `infer` writes. It also runs the v8-in exports, Hecate 9.6 µm (`inkrs.hecate`) and villa's 3D ink models (`inkrs.ink3d`). The Linux wheel embeds the kernels compiled for the common NVIDIA GPUs, with PTX as the fallback for any other, so it needs no Rust or CUDA toolkit, only an NVIDIA driver plus cuDNN 9 and cuBLAS 12 at run time ([Install](README.md#install)). The macOS arm64 wheel runs the villa readers, Hecate and the v8-in exports on Metal; there `inkrs.ink3d` and `tta=8` raise `UnsupportedError`.

## Where it plugs into villa

villa has no in-process API for flat ink inference. Everything goes through `vesuvius.ink_detection.inference.infer`. The line numbers below are from villa `0fb38c45b488e192566ea83ae36369cb6db92652`, the parity reference, with `I` = `vesuvius/src/vesuvius/ink_detection/inference/infer.py`; main's are from `740f0103b9574e9e88bbb24a8438dbe1d5a1a0c0`.

- `main(argv)` (`I:1362`) parses the flags, `configure_model(args)` (`I:894`) loads the `.pth` and builds the torch model, and `infer_single_zarr(*, args, input_zarr, configured_model, device, output_tiff, layer_direction)` (`I:955`) runs one direction of one zarr and writes one TIFF. Both take an `argparse.Namespace`. main has the same shape (`I:1146`, `configure_single_model` at `I:1181`); it adds `--roi`, `--device`, `--loader-threads` and `--ensemble-checkpoint`, and has no engine or backend switch.
- Layers: `select_layer_indices(depth, *, layer_start, layer_end, output_depth, direction)` (`I:217-244`). `layer_end` is exclusive, negatives wrap, an empty window falls back to the full depth, the window is centre-cropped to 17, and reverse reverses the 17 layers' order (`I:242`).
- Output: `trunc(255 · clip(Σpw / Σw, 0, 1))` as uint8 `[H, W]` (`I:803-812`), written by `write_output_tiff` (`I:815`) as an LZW BigTIFF with 128 × 128 tiles. With `--direction both` the reverse map goes to `<stem>_reverse<suffix>` (`I:1150-1162`).
- The PHerc. 343 recipe (`Nieuwlaar/pherc343-first-letters@e8370f1f`, `ink/scripts/run_ink.sh:52-62`, villa pinned at `0fb38c45` by `ink/Dockerfile:35`) runs `infer ZARR CKPT OUT.tif --overlap 0.5 --blend-mode hann --batch-size 4 --direction both --layer-start S --layer-end S+17` with S = 66/2 + d − 8 = 25 + d, once per depth.
- reader-v2 (`DomRusso2/reader-v2@02c7b469`) is a checkpoint of the same architecture and config format. Its README says only the checkpoint changes, and the engine loads its export. The Python API is tested on `hybrid_3d2d-seed42` and, through a patched villa, on reader-v2; the CLI's reader-v2 checks are in [MODELS.md](../docs/MODELS.md). Hecate is a separate script and model, which `inkrs hecate` runs and `inkrs.hecate` wraps.

The least glue is an engine switch inside `infer_single_zarr` that hands the same inputs to `Engine.infer` and gets villa's uint8 maps back, or has inkrs write villa's TIFFs. One `Engine` per process keeps the model and its cuDNN plans loaded across directions, depths and segments. `Engine.sweep` runs every depth and both directions of a segment in one pass, which is where most of inkrs's speed comes from.

## Module

```python
import inkrs

inkrs.__version__          # "0.2.1"
inkrs.build_info()         # dict: backend ("cuda" or "metal"), embedded PTX targets and ISA version, the GPU families
                           # with compiled kernels, minimum driver, wanted cuDNN/cuBLAS sonames; on macOS no PTX or
                           # cubins and None for the driver and libraries
inkrs.load_cuda_libraries()  # dict: the cuDNN / cuBLAS file the engine calls (libcudnn.so.9, or cudnn64_9.dll on
                             # Windows); Engine() calls it first; {} on macOS
inkrs.write_tiff(path, image)  # villa's TIFF for a uint8 [H, W] map, byte-identical to the CLI's
inkrs.fast_presets()       # dict: each model's fast preset (see "Fast mode"); a reader not listed has none
inkrs.hecate(...)          # Hecate 9.6 um, as `inkrs hecate` (see "inkrs.hecate")

inkrs.ink3d(checkpoint, volume, output, *, device=0, bbox=None, overlap=0.5, chunk=128, max_accum_gb=8.0,
            algo="find", overwrite=False, coverage=False)  # 3D ink: see "3D ink models" below

class inkrs.InkrsError(RuntimeError): ...            # any engine failure; the message holds the full cause chain
class inkrs.UnsupportedError(inkrs.InkrsError): ...  # input, model or GPU inkrs does not implement; fall back
class inkrs.CudaLibraryError(inkrs.InkrsError): ...  # no driver, cuDNN 9 or cuBLAS 12 found, with what was searched, or (Windows) one refused in the current directory or as a second copy
```

`import inkrs` needs no GPU. The CUDA libraries load when the first `Engine` is made.

### `UnsupportedError`: when to fall back

`UnsupportedError` means inkrs doesn't implement this case, and a caller with its own PyTorch path should run that instead. Every other `InkrsError` is a real failure: out of GPU memory, a corrupt or truncated file, a missing library. inkrs raises `UnsupportedError` for:

- a volume that isn't uint8 (numpy, torch, zarr or anything with `__array__`);
- a volume with a spatial side under the reader's tile edge (`Engine.patch`: 128 px, or 384 for Reader V3);
- a volume of 4 GiB or more (see "Volumes"), checked from its shape before a zarr, lazy array or CUDA tensor is read;
- a valid zarr that inkrs can't read: not uint8, not 3D, not `[Z, H, W]`, Fortran order, filters, or a compressor other than blosc-lz4, zstd or none (a zarr with broken metadata or chunks is an `InkrsError`);
- a checkpoint that isn't the `hybrid_3d2d-seed42` contract: another architecture, crop, padding, preprocessing, amp dtype, target set or weight dtype;
- a `.pth` with no export, where torch and villa aren't importable to make one, or a `.pth` the exporter refuses (see "Weights"): not a flat model, a state whose keys or shapes don't match the model its config builds (the message names the missing and unexpected keys and the mismatched shapes), or non-float32 weights. villa's PyTorch path loads such a state with `strict=False`, but the export never runs from a partial one;
- a GPU the embedded PTX doesn't serve, or a driver too old to load it.

The engine classifies its own errors, so callers don't need to copy these limits. The GPU case reads cudakit's and the driver's own errors (`CUDA_ERROR_NO_BINARY_FOR_GPU`, `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`), so it follows whatever PTX set the wheel embeds. Input checks run before any GPU work. The GPU check raises when the `Engine` is made and a volume limit raises from the call, so catch it in both places.

## `inkrs.Engine`

```python
eng = inkrs.Engine(
    checkpoint,               # str | PathLike
    *,
    device=0,                 # int | "cuda" | "cuda:N" | torch.device("cuda", N); on macOS 0 | "mps" | "mps:0" |
                              # torch.device("mps")
    batch_size=None,          # tiles per forward: default 64, or 16 for a v8-in export; not villa's --batch-size
    algo="find",              # "find" (time every cuDNN algorithm once) | "heuristic"
    max_workspace_mib=1024,
    weights_dir=None,         # where a .pth's export is looked up first; default $INKRS_WEIGHTS_DIR, else ./weights
    precision="fp16",         # the reader's kernels are FP16-only; anything else raises ValueError
)
eng.weights                   # Path: the safetensors export loaded
eng.reader                    # str: the reader name its sidecar records ("hybrid_3d2d-seed42", "reader-v2"),
                              # or None without a sidecar
eng.load_report               # dict: reader, cudnn, compute_capability, ptx_tier (the tier loaded, "sm80" or
                              # "sm120"), kernels ("sm_120 SASS" when a cubin compiled ahead loaded, "PTX,
                              # JIT-compiled" otherwise), load_weights_s, open_gpu_s, plan_s, algo_cache_hits, algo_cache, algos,
                              # schema, weights, cuda_libraries (the files the engine calls), mapped_libraries; on
                              # macOS compute_capability is the GPU's name, ptx_tier and kernels are "metal", cudnn 0 and algo_cache null
eng.batch_size, eng.device
```

- `checkpoint` is the safetensors export from `python -m inkrs.export_weights CKPT OUT_DIR`, or villa's `.pth` (see "Weights"). `eng.weights` is the export that was loaded.
- Construction loads the weights, opens the device and plans every layer (with `"find"`, a few seconds of cuDNN timing). The GIL is released throughout.
- Which library runs the reader's 2D convs depends on the GPU. On compute capability 12.x it's inkrs's own kernels, which need no cuDNN timing and give the same bytes in every process. Other GPUs use cuDNN plans, because on the four other GPUs measured no compute-capability rule picked the own kernels ([results/README.md](../results/README.md)). So the same run on a 12.x GPU and on another GPU can differ by cuDNN's rounding, within the parity contract. `INKRS_CONV=inkrs` or `INKRS_CONV=cudnn` overrides the default on the sm80 and sm120 tiers. The sm75 tier (Turing) has no own convs and always uses cuDNN. On macOS the Metal backend runs its own convs, and `Engine()` raises `UnsupportedError` when `INKRS_CONV` is set. `load_report["conv"]` names the library that ran them and `load_report["conv_reason"]` says why.
- With `"find"`, each layer's timed ranking is kept in `<cache>/cudnn-algos/` (the cache directory below), keyed on the inkrs version, PTX tier, GPU, driver release, cuDNN and the layer's descriptor. Later engines on the same setup replay it instead of timing again (`load_report["algo_cache_hits"]` counts them); `INKRS_ALGO_CACHE=0` turns it off. The driver release comes from `/proc/driver/nvidia/version`, else from NVML, which containers provide where they hide that file (on Windows, always NVML: System32's `nvml.dll`). Where neither names it, the cache stays off and the first plan says so once on stderr. `load_report["algo_cache"]` gives the state: `{"state": "on", "dir", "source", "driver"}`, or `"disabled"`, `"no_directory"`, `"no_driver_release"` (with `"tried"`, each source's error), or `"unwritable"` (with `"error"`, the first failed write, also noted once on stderr). The `hecate()` and `ink3d()` reports carry the same field, and `hecate()`'s also has `algo_cache_hits`.
- The module leaves the CUDA driver's kernel cache (`CUDA_CACHE_PATH`) as the process has it, since torch may have started CUDA first. The `inkrs` CLI moves it into the inkrs cache directory.
- One call runs at a time per engine; a second thread's call waits for the first. Engines on different devices can run in parallel threads.

### Weights

inkrs reads a safetensors export, never a `.pth` itself. For a `.pth`, `Engine` hashes it (SHA-256) and takes the first export whose `.json` sidecar has that `checkpoint_sha256`, looking:

1. in `weights_dir`;
2. in the cache, `$INKRS_CACHE_DIR/weights/<sha256>/`, else `$XDG_CACHE_HOME/inkrs/weights/<sha256>/`, else `~/.cache/inkrs/weights/<sha256>/` (`%LOCALAPPDATA%\inkrs\weights\<sha256>\` on Windows);
3. otherwise, if torch and villa (`vesuvius`) are importable, `Engine` exports the `.pth` into that cache entry on the CPU and loads it. A villa process has both, so it pays for one export per checkpoint, ever. The export is written to a staging directory and renamed into place, so a crash or a concurrent `Engine` never leaves a partial entry. A checkpoint the exporter refuses, including one whose keys or tensor shapes don't match its model, or a villa without the exporter's entry points, raises `UnsupportedError`. Any other export failure (disk full, out of memory, a corrupt `.pth`) is an `InkrsError`;
4. without torch or villa, `UnsupportedError` names the manual command, `python -m inkrs.export_weights CKPT DIR`.

Matching is by content, so a moved or renamed `.pth` still finds its export, and a changed one never takes a stale export. The digest is kept in `<cache>/checkpoint-sha256/`, keyed on the `.pth`'s path, size and modification time, so later starts skip reading the checkpoint; a rewritten `.pth` gets a new modification time and is hashed again. The CLI keeps the same entries for a `.pth` passed to it.

### `Engine.infer`: one villa `infer` call

```python
maps = eng.infer(
    volume,                   # see "Volumes"
    *,
    layer_start=None, layer_end=None,   # villa's --layer-start/--layer-end, same rules (end exclusive, negatives wrap)
    direction="forward",      # "forward" | "reverse" | "both"
    overlap=None, stride=None, # stride wins when given; villa's rounding of Engine.patch * (1 - overlap) otherwise, default 0.5
    fast=False,               # the reader's fast preset; ValueError for a reader with none (every reader so far)
    blend_mode="hann",        # "hann" or "auto" (which picks hann for stride < Engine.patch); others raise
    occupancy="auto",         # see "Empty-tile skips"
    coverage=False,           # also return each map's coverage mask
    output_tiff=None,         # str | PathLike: write villa's TIFFs instead of returning arrays
    tta=1,                    # 8: the 306B recipe's 8-fold dihedral TTA (docs/MODELS.md), float32 maps
)
# -> {"forward": ndarray[uint8, (H, W)], "reverse": ...}   only the directions asked for
#    with coverage=True also "forward_coverage" / "reverse_coverage" (uint8, 1 where the blend weight > 1e-6)
#    with output_tiff: {"forward": Path, "reverse": Path} instead, named as villa names them
#    (OUT.tif, and OUT_reverse.tif for the reverse map of a "both" run)
```

The maps are villa's: `trunc(255 · clip(Σpw / Σw, 0, 1))`, and 0 where the weight sum is ≤ 1e-6. "forward" is the blind side and "reverse" the reading side. The recipe's ensemble then reads `clip((v − 64) / 128, 0, 1)` from them, so the caller applies its own post-processing as it does to villa's output.

With `tta=8` (CUDA only; the Metal engine raises `UnsupportedError`), each map is the PHerc. 0306B recipe's 8-fold dihedral TTA map (the CLI's `--tta 8`): float32 `[H, W]` in [0, 1], the mean of two villa reads (as stored and turned a quarter) whose tiles each ran as four mirrored copies. `output_tiff` must then end in `.npy`, `occupancy` must stay `"auto"` (the recipe runs no occupancy scan) and `coverage` must stay False. v8-in exports refuse it.

### `Engine.sweep`: every depth and direction of a segment in one pass

```python
maps = eng.sweep(
    volume,
    depths,                   # iterable of int, e.g. range(-4, 5); depth d reads layers Z//2 + d - 8 .. +17
    *,
    direction="both",
    stride=None,              # or overlap=; default overlap 0.5 (stride 64, or 192 for Reader V3); both at once raise ValueError
    fast=False,               # as infer
    occupancy="auto", coverage=False,
    out_dir=None, name=None,  # write OUT_DIR/<name>__<reader>__L<Z>s<S>[_reverse].tif instead,
                              # <reader> from the export sidecar (Engine.reader)
    tta=1,                    # 8: as infer, float32 maps written as .npy in place of .tif
)
# -> {(d, "forward"): ndarray, (d, "reverse"): ndarray, ...}, or the paths with out_dir
```

`sweep` is the CLI's `inkrs sweep`. It reads, schedules and uploads the volume once, and finishes each map on the GPU as soon as its last tile is done.

### `Engine.run`: the layer windows directly

```python
maps = eng.run(volume, windows, *, stride=None, occupancy="auto", coverage=False, outputs=None)  # stride: half the tile edge
# windows: [(first_layer, step), ...] with step +1 (forward) or -1 (reverse, first = the window's top layer)
# outputs: None (return arrays) or one path per window (write TIFFs)
# -> list, one entry per window: ndarray, or (map, coverage) with coverage=True, or the path
```

`infer` and `sweep` are thin layers over `run`. After every call, `eng.last_report` holds the pass's timings and per-map tile counts, the same dict as the `passes[]` entries of the CLI's `inkrs.json`.

## Fast mode

The defaults are parity mode: villa's maps for the readers and `hecate.py`'s for Hecate. Fast mode trades that for speed. A model gets a preset only when it kept ink-label AUC within 0.005 of parity mode on every labelled segment of the accuracy grid ([FAST.md](../docs/FAST.md) has the per-segment numbers). `inkrs.fast_presets()` returns the presets, and the CLI's `inkrs fast-presets` prints the same JSON.

- **Readers: none.** They stay at stride 64 (overlap 0.5), fp16. reader-v2 and seed42 lost too much AUC at every stride above 64 that was tried, so `fast=True` raises `ValueError`.
- **Hecate: bf16 at stride 32, batch 32.** 2.21× the fp32 default's GPU speed on the labelled crops, with a worst paired ΔAUC of −0.0006.
- **The ensemble** (reader-v2 + Hecate) is reader-v2 at stride 64 with Hecate at its fast preset.

`stride`, `overlap` and `precision` stay as expert settings, off by default. Any value other than the default leaves parity, and FAST.md lists each one's measured accuracy cost.

## `inkrs.hecate`: Hecate 9.6 um

```python
report = inkrs.hecate(
    volume,                   # str | PathLike: a uint8 [Z, Y, X] .npy, or a local zarr v2 array or group (array 0)
    checkpoint,               # the Hecate .safetensors export, or hecate.py's .pth with an export in weights_dir
    output,                   # OUT.png; must not exist. OUT.json is written beside it, as hecate.py writes it
    *,
    spacing_um,               # the render's voxel size; must be within 2% of 9.6
    direction="forward",      # "forward" | "reverse" (hecate.py --reverse) | "both" (also OUT_reverse.png)
    stride=None, overlap=None, # default stride 32, hecate.py's; overlap o means round(64 * (1 - o))
    precision=None,           # "fp32" (default, TF32 convs as PyTorch allows) | "bf16" (hecate.py's autocast)
    tf32=True,                # False: strict fp32 convs
    batch_size=None,          # default 8, or 32 with fast=True
    fast=False,               # Hecate's fast preset; refuses an explicit stride, overlap or precision
    algo="find", max_workspace_mib=1024, weights_dir=None, device=0,
)
# -> the CLI's --report JSON as a dict, plus "outputs": {"forward": Path, "reverse": Path}
```

It runs the same code as `inkrs hecate`, whose checks against `hecate.py` are in [MODELS.md](../docs/MODELS.md). Any request the engine would refuse before touching the GPU raises `ValueError` before the CUDA libraries load or the volume is read: bad or conflicting settings, a spacing more than 2% from 9.6, and an output that exists or doesn't end in .png. Engine failures raise `InkrsError`, as `Engine` does. There's no in-memory volume input and no `UnsupportedError` classification of Hecate's limits.

## v8-in exports

`inkrs.Engine` loads the v8-in family (v8-in, and the A-hn fine-tunes seeds 0-2) when the export's sidecar says `"architecture_family": "v8in"` (exports from `scripts/export_v8in.py`). Only `infer` runs them; `run`, `sweep` and `_memory_info` raise `UnsupportedError`.

```python
eng = inkrs.Engine("weights/v8in-ahn-s0.safetensors")   # batch_size defaults to 16 here
eng.architecture              # "v8in" ("villa" for villa's readers)
eng.reader                    # "v8in", "v8in-ahn-s0", "v8in-ahn-s1" or "v8in-ahn-s2"
eng.convention                # the export's default map convention: "release" (v8-in) or "306b" (A-hn)
maps = eng.infer(volume, layer_start=None, direction="forward", convention=None, output_tiff=None)
maps["forward"]               # float32 [H, W] probabilities (or the .npy path written with output_tiff)
eng.last_report               # dict: tiles admitted, GPU and wall time, the conv plans
```

- `volume` is a uint8 `[Z, H, W]` array, or a path to a zarr v2 array, a group with array `0`, or a `.npy`.
- `layer_start` is the first of the 24 layers read, by default the central 24, `(Z - 24) // 2` (8 on a 41-layer render). `layer_end` may only repeat `layer_start + 24`. `direction="reverse"` feeds the 24 layers deepest first.
- The map convention decides how tiles are admitted and stitched. The sidecar's `map_convention` records it field by field, and the engine refuses an export whose record differs from the code. `convention=` overrides it per call:
  - `"release"` reproduces ink8um's `predict_stack`. Coverage is any of the 24 layers above 0, closed with a 7 × 7 ellipse, and the tiles on the 21 px lattice entirely inside the coverage run. A ±1 sigma window; the sum is divided by the tile count, then NaN is set to 0 and the map clipped to [0, 1]. The sigmoid is of the FP16 logit.
  - `"306b"` reproduces the A-hn runner (`aclick/bench/infer.py --tta 1`). Tiles at least a quarter covered on stored layer `layer_start + 12` run. A ±3 sigma window; the sum is divided by the weight sum, NaN where no tile reached. The sigmoid is of the float32 logit.
- The A-hn `config.json` says `reverse_layers: true`, but its runner never reads it, and the A-hn maps were made in stored order. inkrs doesn't read the flag either; the sidecar keeps it as `upstream_reverse_layers`.
- overlap, stride, fast, blend_mode, occupancy and coverage must keep their defaults, because the convention fixes them. `output_tiff` must end in `.npy`.
- The engine runs on the first visible GPU (`device` must be 0; use `CUDA_VISIBLE_DEVICES`) and keeps the GIL while it runs.

The CLI is the same: `inkrs infer VOLUME EXPORT OUT.npy [--layer-start L] [--direction ...] [--convention release|306b] [--batch-size 16] [--passes N] [--report R.json]`, where `--passes` repeats the map with the model resident, for timing.

## Volumes

`volume` is one of:

| Type | Handling |
|---|---|
| `str` / `PathLike` | a local zarr v2: a group (its array `0` is read) or a bare array, `[Z, H, W]` uint8. Rust reads it exactly as the CLI does |
| `numpy.ndarray` | `[Z, H, W]` uint8. A C-contiguous array is read in place with no copy. Any other layout is copied once to C order |
| `torch.Tensor` | `[Z, H, W]` uint8. A CPU tensor is read in place through `.numpy()`; a CUDA tensor is copied to the host first |
| anything with `__array__` | e.g. a `zarr.Array`; converted with `numpy.asarray` (which reads it whole) |

Other dtypes raise `UnsupportedError`, before a lazy array is read or a CUDA tensor is copied. villa's robust normalisation would cast them to float32, but inkrs supports uint8 only. Depth must be the first axis (another rank raises `ValueError`), and both spatial dimensions must be at least 128. The array must not change while a call runs, because the GIL is released.

The 4 GiB limit applies to the `volume` passed to the call, as Z × H × W bytes (under 2³² − 1), because the kernels index it with u32. It isn't measured on the zarr the volume came from. A villa backend that passes only the 17-layer window `[17, H, W]` can therefore take segments up to H × W < 252.6 Mpx, about 15,800 px square. A zarr path passed as `volume` is read whole, so there the limit applies to its level-0 array. GPU memory is a separate limit: a segment that fits under 4 GiB but not on the card raises a plain `InkrsError`, not `UnsupportedError`.

## Empty-tile skips

villa skips tiles in two steps, and inkrs makes the same two. They decide which pixels are covered.

1. A low-resolution occupancy scan: pyramid level `3` of a zarr group, or the nearest numeric level. villa makes no scan for a bare array.
2. Each remaining tile runs only if its 17 selected layers hold a nonzero voxel.

`occupancy` controls the first skip:

| Value | Meaning |
|---|---|
| `"auto"` (default) | for a zarr path, villa's own choice (scan the group, or no scan for a bare array); for an in-memory volume, no scan |
| `None` | no scan |
| `str` / `PathLike` | scan this zarr group's pyramid, as villa would scan the group the array came from |
| array `[h, w]` or `[d, h, w]` uint8 / bool | a pyramid level already in memory, collapsed and scaled to the volume as villa does (`round(H / h)`) |

A villa backend that passes an in-memory level 0 should pass that group's level `3` array (or its path) to keep villa's covered pixels.

## Equivalence

For the same volume, layers, stride, batch size and `algo`, the Python API runs the same code as the CLI. Its maps are byte-identical to `inkrs infer` and `inkrs sweep`, and its `output_tiff` / `out_dir` files are byte-identical to the CLI's TIFFs. Against villa eager the CLI's checks apply: each map within 2/255, with the same covered pixels ([README](../README.md)).

## Not covered

- Models other than `hybrid_3d2d-seed42`'s architecture and the v8-in family in `Engine` (Hecate has `inkrs.hecate`, 3D ink `inkrs.ink3d`), villa's zero-padded tile phase, non-Hann blends, and non-uint8 input. These are the CLI's limits; the input and model ones raise `UnsupportedError`.
- Float probability maps. The engine finishes each map on the GPU in villa's uint8 encoding.
- Zero-copy input from GPU memory. A CUDA tensor makes one host round trip, because the skip schedule is computed on the host.

## 3D ink models: `inkrs.ink3d`

`inkrs.ink3d` runs villa's autoconfigured 3D U-Net ink models (the `ink_3d_dino_guided` v3 family, villa's `infer_full3d_tifxyz` native path) over a uint8 zarr volume and writes a uint8 zarr prediction, as the CLI's `inkrs ink3d` does. The checkpoint is the `.safetensors` export from `scripts/export_unet3d.py` (run with torch and villa importable); `.pth` checkpoints aren't exported on the fly here. It plans every chunk of `bbox` (default: the whole volume) with villa's patch grid and `overlap`, normalises each patch as villa's `percentile_minmax` does, and blends with villa's Gaussian. Its checks against villa eager are in [MODELS.md](../docs/MODELS.md).

```python
report = inkrs.ink3d("v3-78k.safetensors", "volume.zarr", "pred.zarr", bbox=(0, 0, 0, 768, 1024, 1024))
# -> {"patches": 245, "chunks_written": ..., "run_s": ..., "s_per_patch": ..., "plans": [...], ...}
```

Differences from villa: only level 0 is written (villa also writes five downsampled levels), chunks are raw rather than zstd, and the GPU accumulators span the whole region's Y × X, so `max_accum_gb` bounds the region's area. It needs compute capability 8.0 or above, because its convs are cuDNN bf16, so on a Turing GPU, or with the sm75 tier forced, it raises `UnsupportedError`.
