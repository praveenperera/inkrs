# Results

The JSON behind the numbers in the README and the docs. To check the main ones yourself, see [Rerun it](#rerun-it) at the end.

The inputs: **R343** is the PHerc. 343 `concat_w047-w048_R5B2_z9500-11000` render (66 × 2560 × 6720, level 0 only), **R343 multiscale** is the same level 0 with our 2 × 2 × 2 mean pyramid, and **S1** is its 2048 × 2048 crop. The PHerc. 1447 segment is `20250702235910-auto_grown_20250702235910292`. Everything ran on one RTX 5090 unless the line names another GPU.

The README's timings are in `release-timing.json`, `public-ports.json` and `tta-306b-timing.json`. The other timing files hold older measurements, so their numbers differ from the README's.

## The README's numbers

| Number | File and key |
|---|---|
| R343: inkrs 8.50 s, inksweep 64.5 s, 7.57× | `release-timing.json` `runs.r343` |
| villa main with #2428's flags on R343: 161.7 s, 19.0× | `release-timing.json` `r343_villa_fast_flags` |
| Reader V3, Hecate `--fast`, and villa's `infer` with the inkrs adapter | `release-timing.json` `runs.v3`, `runs.hec`, `villa_adapter` |
| reader-v2: 3.39 s against villa main's 51.4 s, 15.2× | `release-timing.json` `singles.inkrs-rv2`, `singles.villa-rv2` |
| v8-in against inksweep, 1.82× | `v8in-306b.json` `timing.compiled_vs_inkrs` |
| The 18 timed R343 maps within 2/255 of villa main's | `release-timing.json` `r343_gate` |
| 98.3% of pixels identical at d 0 | `r343-map-figure.json` `full_map.identical_share` |
| inkrs against UFSM and ink-sdk, and the first-run and fp32 Hecate losses | `public-ports.json` `runs` |
| inkrs against SurgeFok's MPS path on an M4 Pro | `metal-s1.json` `mean_wall_s`, `speedup` |
| The PHerc. 0306B recipe: 395 s against 650 s, 1.68× | `tta-306b-timing.json` `ab` |

## Speed

- `release-timing.json`: inkrs against inksweep, villa and villa's inkrs adapter on R343, the PHerc. 1447 segment and S1, with warm and empty caches.
- `public-ports.json`: inkrs against UFSM and ink-sdk's v8-in lane, with each runner's maps against the references.
- `tta-306b-timing.json`: the PHerc. 0306B recipe with TRACE v3B's 8-fold test-time augmentation, inkrs against the method's own code.
- `timing-matched.json`: inkrs and inksweep on S1 and R343, first and later passes, with villa eager's time per depth on R343.
- `inkrs-report-s1.json`, `inkrs-report-r343.json`: inkrs's own `inkrs.json` from those runs, with each pass's read, schedule, upload and run times.
- `inkrs-report-r343-gate.json`: the same report for the R343 run with coverage masks and the finite check.
- `inksweep-passes-s1.json`, `inksweep-passes-r343.json`: inksweep's time per pass and per stage from those runs.
- `timing-villa-fast-flags.json`: villa main with its faster flags (`--batch-size 8 --no-compile --loader-threads 8 --chunk-cache-size 64`), per depth on R343 and in total on S1.
- `software.json`: the CUDA, cuDNN and torch versions behind inkrs, villa and inksweep.

## Matching villa and PyTorch

- `p1-r343.json`, `p1-r343-multiscale.json`, `p1-s1.json`: inkrs's 18 maps against villa eager's on R343, R343 multiscale and S1.
- `coverage-mutations-s1.json`: the per-map check on four copies of one S1 map, each with one planted defect.
- `letters-r343-multiscale.json`: the recipe's one-model reading over the 18 letter boxes, inkrs against villa.
- `p0-s1.json`: every layer against PyTorch FP32, beside autocast's own error.
- `normaliser-gpu.json`: the GPU normaliser against villa's on every fixture tile.
- `repeatability.json`: maps from different passes and runs compared with `mapdiff.py --exact`.
- `inksweep-vs-villa-r343-raw.json`: inksweep's raw maps against villa eager's on R343.
- `inksweep-vs-villa-r343-calibrated.json`, `inksweep-vs-villa-s1-calibrated.json`: inksweep's timed maps against villa's, read the recipe's way.
- `r343-map-figure.png`, `r343-map-figure.json`: the README's map figure, how it was made, and the whole d 0 map against villa's.

## reader-v2 and Hecate

[MODELS.md](../docs/MODELS.md) describes each check. The input arrays and their SHA-256 are under `inputs` in `reader-v2-p1.json` and `hecate-maps-crop.json`.

- `reader-v2-p1.json`: reader-v2 against villa eager on a crop of the PHerc. 1447 segment, d −2..2, both directions.
- `reader-v2-timing.json`: reader-v2 on the whole segment, inkrs against villa main with #2428's flags.
- `reader-v2-published.json`: inkrs's and villa's whole-segment `L31s7` maps against gomyway's published reader-v2 maps.
- `reader-v2-villa-engine.json`: reader-v2 through villa's `--engine inkrs` adapter against villa's PyTorch engine.
- `hecate-p0h.json`: Hecate's intermediate outputs against PyTorch's in fp32, TF32 and bf16.
- `hecate-maps-crop.json`: inkrs's Hecate maps against `hecate.py`'s on two crops of the segment.
- `hecate-timing.json`: Hecate on the whole segment, inkrs against `hecate.py` and `hecate_fast`, with each map against theirs and gomyway's published maps.
- `hecate-inksweep-timing.json`: Hecate, inkrs against inksweep's fast mode with and without compile.
- `hecate-kernel-timing.json`: Hecate bf16 with and without its encoder-epilogue kernels.

## PTX tiers and other hardware

- `ptx-tiers-s1.json`: `hybrid_3d2d-seed42` on S1 with each PTX tier, the tiers' maps against each other, and their speed.
- `ptx-tiers-models.json`: reader-v2 and Hecate with each PTX tier forced.
- `release-checks.json`: compute-sanitizer on each PTX tier, which kernels the workloads launched, reader-v2's peak device memory and Hecate bf16's first-run cost.
- `integration-v0.1.json`: the model checks on each PTX tier, and the stem kernels and start-up changes against the engine without them.
- `integration-v0.2.json`: the model checks on each PTX tier, and inkrs's own convolutions against cuDNN.
- `t4-gates.json`, `t4-speed.json`: seed42, reader-v2, Reader V3 and Hecate on a Tesla T4 against the references, and their speed against villa eager and `hecate.py`.
- `t4-models.json`: the prize recipes' models on a Tesla T4 against the references.
- `other-gpus-v0.2.json`: the readers and Hecate on an A100, an A10, an L4 and an RTX 4090, and inkrs against villa eager on the 4090.
- `other-gpu-a10.json`, `other-gpu-tesla-t4.json`: the release wheel on an A10 and a Tesla T4 against the references.
- `metal-s1.json`: the Metal backend on an M4 Pro, one S1 map against SurgeFok's public MPS path, with and without the Neural Engine offload.
- `metal-gates.json`: the Metal backend on an M4 Pro: seed42, Reader V3 and Hecate maps against the CUDA references.
- `windows-checks.json`: the Windows build, cross-built on Linux: what builds, the DLLs the CLI and the wheel import, and the unit tests run under Wine. Nothing in it ran on Windows.

## Other models

[MODELS.md](../docs/MODELS.md) has the results, including [Reader V3](../docs/MODELS.md#reader-v3) and [the recipes](../docs/MODELS.md#running-the-recipes).

- `reader-v3-norm.json`, `reader-v3-p0.json`, `reader-v3-p1.json`: Reader V3's normaliser, its per-layer check against PyTorch, and its maps against villa eager's.
- `reader-v3-timing.json`: Reader V3 against inksweep's `readerv3` and villa.
- `v8in-306b.json`: v8-in and its three A-hn fine-tunes against their reference runners on two 306B crops, and their speed against inksweep.
- `ink3d-gates.json`: villa's 3D ink U-Net through `inkrs ink3d` against villa eager on a crop.
- `prize-models-gates.json`: the PHerc. 343 recipe's other villa models (dense_native, ink_9um seed 43, TRACE v3B, Rectina M4) and its combine against villa eager.
- `recipe343-timing.json`: the PHerc. 343 recipe through `scripts/recipe.py` against the recipe's own runners.

## Faster settings

- `accuracy-speed.json`: ink accuracy (pixel AUC and AP) of reader-v2, seed42 and Hecate at faster strides and precisions, and their speed on the PHerc. 1447 segment.
- `fast-mode-timing.json`: fast mode's timings, and villa with the inkrs engine running `--layer-starts` against one process per window.

## Rerun it

You can check the two claims that matter yourself: that inkrs writes villa's maps, and how much faster it is. Any NVIDIA GPU from the T4 up will do. Your times will only line up with mine on an RTX 5090.

### Setup

- The CLI: `cargo build --release`.
- A Python environment with villa at the commit inkrs matches. Install villa with `--no-deps`, since its own dependency list pulls in far more than you need:

  ```
  pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu129
  pip install --no-deps "vesuvius @ git+https://github.com/ScrollPrize/villa.git@0fb38c45b488e192566ea83ae36369cb6db92652#subdirectory=vesuvius"
  pip install numpy==2.5.3 zarr==3.4.0 tifffile==2026.9.20 imagecodecs==2026.8.16 safetensors==0.8.0 \
    requests==2.34.2 pynrrd==1.1.3 pillow==12.3.0 aiohttp==3.14.4 nest-asyncio==1.6.0 einops==0.8.2 timm==1.0.30
  ```

- [inksweep](https://github.com/inksweep/inksweep) at `c6964929043422ae1f07a797a4fe30d94440e562`, in its own environment. You need it to render the input, and for the timing.
- [pherc343-first-letters](https://github.com/Nieuwlaar/pherc343-first-letters) at `e8370f1`, for the mesh and umbilicus.

The commands below use `VILLA_PY` for the villa environment's `python` and `LETTERS_SRC` for the pherc343-first-letters checkout.

### Inputs

The checkpoint, villa's `hybrid_3d2d-seed42` at step 75,000 (SHA-256 `e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab`), and its export:

```
curl -LO https://huggingface.co/scrollprize/ink_9um/resolve/7109667e2607db1b90c37c8b09cb876ea7fe7bb1/hybrid_3d2d-seed42/step-075000.pth
$VILLA_PY -I scripts/export_weights.py step-075000.pth weights/
W=weights/hybrid_3d2d-seed42.safetensors
```

The PHerc. 343 render, the segment the first letters were read on:

```
inksweep render $LETTERS_SRC/outputs/concat_w047-w048_R5B2_z9500-11000.tifxyz \
  --volume https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr \
  --out r343 --layers 66 --voxel-um 8.64 --normal-sign auto --umbilicus $LETTERS_SRC/inputs/umbilicus_PHerc0343.json
```

`$VILLA_PY -I scripts/fingerprint.py r343/layers.zarr` should print `92dbbcb3df5e4a3e70fd65c96d0bf189aa4ba6a3bea8e92e6585ccaac56c5d34  [66, 2560, 6720] uint8`. If your render differs, the checks still work, but your numbers won't match mine exactly.

### Same maps as villa

Run inkrs with coverage masks, then villa, one process per depth, then compare:

```
target/release/inkrs sweep r343/layers.zarr --ckpt $W --depths=-4..4 --direction both --stride 64 --batch-size 64 \
  --out ours --name concat_w047-w048 --coverage --check-finite

mkdir -p /tmp/v ref/ink
for d in 0 -1 1 -2 2 -3 3 -4 4; do
  s=$((25 + d))
  TMPDIR=/tmp/v $VILLA_PY -m vesuvius.ink_detection.inference.infer r343/layers.zarr step-075000.pth out.tif --overlap 0.5 \
    --blend-mode hann --direction both --layer-start $s --layer-end $((s + 17)) --batch-size 4 --no-compile
  mv out.tif ref/ink/concat_w047-w048__hybrid_3d2d-seed42__L66s$s.tif
  mv out_reverse.tif ref/ink/concat_w047-w048__hybrid_3d2d-seed42__L66s${s}_reverse.tif
done

$VILLA_PY -I scripts/p1_gate.py ours ref r343/layers.zarr 64 --report ours/inkrs.json --json p1.json
```

`TMPDIR` needs a short path, because torch's DataLoader workers put sockets there. Every map should reach r ≥ 0.9994, with p99.9 and the largest difference at most 4/255, and cover exactly the pixels villa covers.

### Speed

Time each whole command. Run inksweep a few times first, until two runs in a row take about the same time, because its first run compiles. It reads the checkpoint from `checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth`.

```
time target/release/inkrs sweep r343/layers.zarr --ckpt $W --depths=-4..4 --direction both --stride 64 --batch-size 64 \
  --out time-inkrs --name concat_w047-w048

time inksweep run r343/layers.zarr --out time-inksweep --name concat_w047-w048 --voxel-um 8.64 --ckpt-dir checkpoints \
  --no-download --no-ensemble --orientations both --depths=-4..4 --readers ink9_s42 --mode fast --compile --no-raw
```

For villa with #2428's faster flags, use a villa main checkout that has that PR (mine was from 2026-11-10) and run the loop above with `--batch-size 8 --no-compile --loader-threads 8 --chunk-cache-size 64` in place of `--batch-size 4 --no-compile`. Its time is the nine processes added up.

### reader-v2 and Hecate

[MODELS.md](../docs/MODELS.md) has the results. The input is the PHerc. 1447 segment, and `make_hecate_inputs.py` also needs `opencv-python` in the villa environment:

```
B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com
P=PHerc1447/segments/20250702235910-auto_grown_20250702235910292/surface-volumes/8.64um-1.2m-116keV-volume-20250521151220.tifs
mkdir -p 1447/tifs && for i in $(seq -w 0 30); do curl -sSf -o 1447/tifs/$i.tif $B/$P/$i.tif; done
$VILLA_PY -I scripts/make_hecate_inputs.py 1447/tifs 8.64 1447
```

That writes the arrays below with their SHA-256 in `1447/inputs.json`. Mine are under `inputs` in `reader-v2-p1.json`.

**reader-v2** is `reader-v2-step040000.pth` from `huggingface.co/domenicor046/reader-v2` at `72633d8a`. Export it with `export_weights.py` as above, then run the same check as for PHerc. 343 on `1447/native-crop.zarr`, d −2..2. With 31 layers, depth d starts at layer 7 + d:

```
target/release/inkrs sweep 1447/native-crop.zarr --ckpt weights/reader-v2.safetensors --depths=-2..2 \
  --direction both --stride 64 --batch-size 64 --out rv2-inkrs --name native-crop --coverage --check-finite
# villa: the loop above over d in 0 -1 1 -2 2 with s=$((7 + d)), saving to rv2-villa/ink/native-crop__reader-v2__L31s$s.tif
$VILLA_PY -I scripts/p1_gate.py rv2-inkrs rv2-villa 1447/native-crop.zarr 64 --report rv2-inkrs/inkrs.json --json rv2.json
```

**Hecate** is `hecate.py` and `hecate_9.6um.pth` from `huggingface.co/scrollprize/hecate` at `9cb86e500e944b11a06a7020403cde5dffb5bcb2`. Compare inkrs's map with `hecate.py`'s on the crop:

```
$VILLA_PY -I scripts/export_hecate.py hecate.py hecate_9.6um.pth weights/
$VILLA_PY -I hecate.py --checkpoint hecate_9.6um.pth --spacing-um 9.6 --batch-size 8 --input 1447/r96-crop.zarr --output ref-hec/map.png
target/release/inkrs hecate --checkpoint weights/hecate-9.6um.safetensors --input 1447/r96-crop.zarr \
  --output ours-hec/map.png --spacing-um 9.6 --batch-size 8
$VILLA_PY -I scripts/hecate_mapgate.py ours-hec/map.png ref-hec/map.png --mode fp32 --direction fwd --json hec.json
```

For the bf16 maps, add `--precision bf16` to both runs and use `--mode bf16`.

To time Hecate against inksweep, on the whole segment, forward only:

```
time target/release/inkrs hecate --checkpoint weights/hecate-9.6um.safetensors --input 1447/r96.zarr \
  --output hec-time/map.png --spacing-um 9.6 --fast
time inksweep run 1447/r96.zarr --out isw-hec --readers hecate --checkpoint "hecate=hecate_9.6um.pth,hecate.py" --depths 0 \
  --orientations forward --no-ensemble --mode fast --compile --voxel-um 9.6 --no-download
```
