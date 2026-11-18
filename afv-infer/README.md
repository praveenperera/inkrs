# inkrs-afv: faster inference for the AFV fibre model V2

`inkrs-afv` runs Qual's [AFV fibre model V2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2) over a CT volume
and writes the probability zarr that villa's `python -m lasagna.qualzz_inference` writes: `semantic`
(2 × Z × Y × X, uint8 V and H fibre probabilities) and `tracing` (the same at half resolution), the input that
[AVF_SPLINE_GENERATOR](https://github.com/Qualzz/AVF_SPLINE_GENERATOR) and
[inkrs-afv-trace](https://github.com/praveenperera/inkrs/tree/master/afv-trace) trace.

The model, its weights and its output are Qual's, and the tiling, blending and quantisation are villa's
lasagna inference. This package changes how fast that runs, not what it computes. It keeps the GPU busy by reading
tiles on several threads ahead of the network, and by blending and finalising on the GPU instead of the host.

## Modes

| Mode | What it computes | Checked against |
|---|---|---|
| `--mode villa` | the whole array, as villa main's `lasagna.qualzz_inference` computes it: one tile per forward, and villa's rule for which tiles run (from the input's stored chunks) | villa main `97fb9336c`, voxel for voxel |
| band mode (the default) | a z band `[z0, z1)`, reflected at the band's own edges, 4 tiles per forward, tiles whose input is all zero skipped | batch-4 tiling of the same band, voxel for voxel |

Use `--mode villa` when you want villa's output. Band mode is for splitting a scroll into z bands that run
separately and resume: it does not reproduce villa's output, because villa runs the whole volume, forwards one tile
at a time (convolutions round slightly differently at other batch sizes) and also runs tiles of stored zeros.

## Install

```sh
pip install inkrs-afv
```

It needs torch 2.14.1 to 2.15 (with CUDA for GPU runs), zarr 2 and `dynamic-network-architectures` 0.4.4, the version the model's repository pins.
Download the model snapshot (`model.py`, `plans.json`, `checkpoint_final.pth`) from
[Qualzz20/afv_fiber_9um_v2](https://huggingface.co/Qualzz20/afv_fiber_9um_v2); `--model-dir` is that directory.

## Use

```sh
# villa's output, for a whole CT array
inkrs-afv --mode villa --ct scroll.zarr/0 --model-dir afv_fiber_9um_v2 --out pred.zarr

# one z band, resumable
inkrs-afv --ct scroll.zarr/0 --z0 12032 --z1 12544 --model-dir afv_fiber_9um_v2 --out band.zarr --max-minutes 60
```

`--ct` is a level-0 uint8 zarr v2 array (the `/0` directory of an OME-Zarr). `--device` defaults to `cuda`;
`cpu` and, on Apple silicon, `mps` also run. In band mode, run the same command again on the same `--out` to
resume a capped (`--max-minutes`) or interrupted run at the first unfinished column. `--expect-sha256` refuses a
checkpoint with another hash.

`--readers` (tile reader threads, default 4) changes the speed, not the output. `--column` (the size of the y–x
columns a run is split into, default 480) does the same in villa mode; in band mode the 4-tile batches are formed
within each column, so the column size is part of what a band-mode output is.

## Same output

In villa mode, inkrs-afv and villa main's `python -m lasagna.qualzz_inference` (97fb9336c, `--device cpu`) wrote
the same `semantic` and `tracing` arrays, every value, on:

- a 160 × 300 × 300 crop of the PHerc0826 CT, every chunk stored;
- a 159 × 301 × 299 crop of the same CT copied chunk for chunk, with 9 of its 18 chunks absent, so that villa's
  skip rule decides which tiles run;
- a synthetic volume whose shape is odd on every axis, with absent chunks, stored all-zero regions and input
  chunks that are not the output chunks (the package's test suite).

On CUDA, villa mode's output has not been compared with villa main's, because villa main's GPU run did not
finish on our test host (see Speed). The check was planned to compare CUDA output as well; it was narrowed to the
CPU after those GPU runs failed, so this is an amended check.

Band mode writes the same arrays as a batch-4 tiling of the band that runs the network on every non-air tile in
order and blends on the host, on the CPU, CUDA and Apple MPS.

## Speed

On PHerc0826, z 12032–12544, y 5120–5632, all x (512 × 512 × 8169, with 561 of its 1,024 input chunks stored),
`inkrs-afv --mode villa --device cuda` took 89–106 s on an NVIDIA RTX 5090, over six runs (89–93 s in five; the
106 s run started with the host's load average at 26). That is the wall time of
the whole command: loading the model, reading the CT, the network and writing the zarr.

Villa main's GPU run did not finish on our test host: on the same slab and GPU, its GPU manager stopped answering
partway through each of the four attempts in which its GPU run started, so we give our time alone.

In the last run (93 s), both tools ran pinned to the same 4 CPU cores at nice 10, with `NUMPY_MADVISE_HUGEPAGE=0`. Villa main ran with
`--flush-workers 2 --accumulator-workers 2`, because with its default of 16 workers on 4 cores its GPU helper
stalled; its GPU manager ran outside the pin at nice 0. Earlier runs had only some of these settings. In the last run, memory "full" pressure averaged 3.4% (at
most 9.2%) on the host while inkrs-afv ran, and 4.2% (at most 8.5%) while villa main ran until it stopped.

## Limits

- The output's `voxel_size_um` attribute is always 9.362, the PHerc0826 CT's voxel size. If your CT has another
  voxel size, pass the real one to the tracer (`--voxel-size-um`), which reads its own flag, not the attribute.
- `--precision bf16` and `fp16` (autocast for the network) are faster on some GPUs but change the output;
  `fp32` (torch's defaults, which run convolutions in TF32 on CUDA) is the default and the checked mode.
- Outputs from different devices (CPU, CUDA, MPS) differ from each other slightly, as the model's own outputs do.
- Villa mode reads which chunks are stored from the input's directory listing, so the input must be a zarr v2
  directory store (`.` or `/` chunk separator).

## Build and test

```sh
uv build --wheel                                    # dist/inkrs_afv-<version>-py3-none-any.whl
uv venv && uv pip install dist/inkrs_afv-*.whl pytest
.venv/bin/python -m pytest tests                    # CPU only, a stub network, under a minute
```

`tests/test_villa.py` also runs villa main's `python -m lasagna.qualzz_inference` on the synthetic volume above and
compares the arrays when `INKRS_AFV_VILLA_PYTHON` (a Python with villa's requirements) and `INKRS_AFV_VILLA_SRC` (a
villa checkout) are set; otherwise it skips that test.

## Licences

MIT or Apache-2.0. `NOTICE` lists what comes from elsewhere: `blend_ramp` is copied from villa (MIT), whose
licence ships in `licenses/`, and villa mode follows villa's tile rule.
