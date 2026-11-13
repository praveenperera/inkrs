# Models

Besides `hybrid_3d2d-seed42`, inkrs runs:

- `reader-v2` and Reader V3, through the same `infer` and `sweep` commands
- the prize recipes' other models of villa's architecture ([below](#the-prize-recipes-models)): ink_9um seed 43, dense_native, TRACE v3B and Rectina M1, M2 and M4, also through `infer` and `sweep`
- Hecate 9.6 µm, through `inkrs hecate`
- v8-in and its A-hn fine-tunes, through `infer`
- villa's 3D ink U-Net, through `inkrs ink3d` ([below](#villas-3d-ink-u-net))

This page gives each one's licence, how to export it, and how it checks out against the reference code. No weights ship with inkrs. You download each model, accept its licence, and export it once on the CPU.

For the readers and Hecate, the JSON behind each number is in [results/](../results/), and [its README](../results/README.md#rerun-it) has the commands. Everything ran on one RTX 5090 unless a section says otherwise. The README's [How fast](../README.md#how-fast) has the headline timings.

## What counts as the same map

A map counts only if it's within a fixed tolerance of its reference. For the villa readers that's villa's own bar against villa eager: r ≥ 0.9994, p99.9 and largest |difference| at most 4/255, and the same covered pixels. For Hecate it's the bounds in its section below, against `hecate.py`, where a nearly flat map may have at most max(5, 0.01% of pixels) differing, each by one step. Every timing on this page is for maps that pass.

## The test segment

reader-v2, Reader V3 and Hecate were checked on one public segment from the Vesuvius Challenge open-data bucket: PHerc. 1447 `20250702235910-auto_grown_20250702235910292`, 31 layers at 8.64 µm (2980 × 3240 px). `scripts/make_hecate_inputs.py` builds every input from its layer TIFFs and records each array's SHA-256 in `inputs.json`:

| Name | Shape | Used for |
|---|---|---|
| `native` | 31 × 2980 × 3240 | reader-v2 and Reader V3 whole segment |
| `native-crop` | 31 × 2048 × 2048, central crop | reader-v2 parity |
| `r96` | 28 × 2682 × 2916, resampled to 9.6 µm with reader-v2's `resample_96` | Hecate per layer and whole segment |
| `r96-crop` | 28 × 2048 × 2048, central crop of `r96` | Hecate crop maps |
| `r96-edge` | 27 × 48 × 2916, the first rows of `r96` | Hecate edge fixture: every tile zero-padded, empty tiles beside kept ones, and forward and reverse slabs that differ |

gomyway's published maps of this segment (`gomyway/vesuvius-ink-maps`, `ink-maps-v1`) used the same resampling, so the whole-segment maps can be compared with those.

## reader-v2

Domenico Russo, MIT. Code `DomRusso2/reader-v2@02c7b46`, weights `reader-v2-step040000.pth` from `huggingface.co/domenicor046/reader-v2@72633d8a`. It's a villa `ink_9um` checkpoint retrained on new data, with `hybrid_3d2d-seed42`'s architecture, tensors and inference settings, so inkrs runs it with no new engine code. Export it with `scripts/export_weights.py`. The reference is villa `0fb38c45` eager.

**Against villa eager** (`results/reader-v2-p1.json`). On `native-crop`, the 10 maps of d −2..2 in both directions, stride 64, match villa's within the same bar as `hybrid_3d2d-seed42`: min r 0.999997, p99.9 |diff| 1/255 and max |diff| 1/255. The coverage mask equals villa's bit for bit, with the same tiles kept.

**Against the published maps** (`results/reader-v2-published.json`). gomyway's published reader-v2 maps of this segment are its centre slab, `L31s7`. inkrs's whole-segment `L31s7` matches them at r 0.999998 in both directions, max |diff| 1/255, with 99.6% of pixels identical. villa main's own run gets r 0.999999 and the same max.

**Speed** (`results/release-timing.json`). On the whole segment, d −2..2 in both directions (10 maps), `inkrs sweep` takes 3.39 s against 51.4 s for villa main with #2428's flags (`--batch-size 8 --no-compile --loader-threads 8 --chunk-cache-size 64`, one process per depth), 15.2× faster. I didn't time inksweep's fast mode with reader-v2's weights.

## Reader V3

Domenico Russo, MIT. Weights `reader-v3.pth` from `huggingface.co/domenicor046/reader-v3@aee1d64b`. It's reader-v2 trained further, finally on 384 px tiles, with the same architecture and weights layout, and it exports the same way. It runs on 17 × 384 × 384 tiles at the model card's `--overlap 0.5 --blend-mode hann`, so stride 192 and villa's Hann window floored at 0.001. One V3 tile is 154.09 GFLOP, 9× a 128 tile, and a segment has 9× fewer of them, so a map costs the same work as a reader-v2 map. The reference is villa `0fb38c45` eager.

**Normalisation** (`results/reader-v3-norm.json`). The GPU normaliser matches villa's normalised tiles bit for bit in FP16 on all 1380 tiles of 6 maps of PHerc. 343's S1 crop (66 × 2048 × 2048) and 4 maps of the 1447 segment, including the 328 that take villa's MAD fallback.

**Layer by layer** (`results/reader-v3-p0.json`). Four S1 tiles, two on the volume's edge and one read in reverse, every layer matches villa's model in FP32 and under autocast.

**Against villa eager** (`results/reader-v3-p1.json`). The same bar as reader-v2, at stride 192:

| Input | Maps | min r | p99.9 \|diff\| | max \|diff\| | Coverage |
|---|---:|---:|---:|---:|---|
| S1 crop, d −2..2, both directions | 10 | 0.999994 | 1/255 | 1/255 | equal to villa's, same tiles kept |
| `native`, d 0, both directions | 2 | 0.999997 | 1/255 | 1/255 | equal to villa's, same tiles kept |

**Speed.** The README's [How fast](../README.md#how-fast) has Reader V3 against inksweep. inkrs's forward takes 1.80 ms of GPU time per 384 tile.

## Hecate 9.6 µm

Vesuvius Challenge, MIT. `hecate.py` and `hecate_9.6um.pth` from `huggingface.co/scrollprize/hecate@9cb86e5`. It's a 3D ResNet-152 encoder with a 3D decoder and a depth-attention head, on 64 × 64 tiles of 16 planes. inkrs ports its forward pass (`--output` only) and its blend, and writes the same PNG and JSON. Export it with `scripts/export_hecate.py` and run it with `inkrs hecate`. The reference is `hecate.py` eager at `--batch-size 8`, at fp32 (with PyTorch's default TF32 convolutions) and at bf16.

**Layer by layer** (`results/hecate-p0h.json`). On 8 real tiles of `r96`, both directions, every stage from the input to the probability is compared with PyTorch fp32 with TF32 off. Each inkrs mode must keep its relative RMS within 1.5× and its p99.9 |diff| within 2× of PyTorch's own error in the matching mode (TF32 for fp32, bf16 autocast for bf16). fp32, TF32 and bf16 all pass. The normalised input is within 1 ulp of PyTorch's in fp32 and half an ulp in bf16.

**Maps on the crop** (`results/hecate-maps-crop.json`). Against `hecate.py` at batch 8: for fp32, r ≥ 0.9994, p99.9 ≤ 4 and max ≤ 8 steps, |mean diff| ≤ 0.1; for bf16, r ≥ 0.999, p99.9 ≤ 6, max ≤ 32 and |mean diff| ≤ 0.25. No pixel may be 0 on one side and above 1 on the other, and every field of the JSON sidecar must match. Where the reference map is nearly flat (variance below 1 step²), r is replaced by a count: at most max(5, 0.01% of pixels) differ, each by 1 step, and the other bounds stay. On a nearly flat map, r can fail with every pixel within 1 step. Every map passes:

| Mode | Side | r | p99.9 | max | mean diff | identical |
|---|---|---|---|---|---|---|
| fp32 (TF32) | forward | 0.999998 | 1 | 2 | +0.0010 | 99.71% |
| fp32 (TF32) | reverse | 0.999998 | 1 | 2 | +0.0010 | 99.70% |
| bf16 | forward | 0.999985 | 1 | 3 | −0.0004 | 97.77% |
| bf16 | reverse | 0.999983 | 1 | 2 | +0.0006 | 97.40% |
| fp32, edge fixture | forward / reverse | 0.99972 / 1 | 0 / 0 | 1 / 0 | | 99.999% / 100% |

On the edge fixture inkrs keeps the same 3 tiles and skips the same 88 as `hecate.py` on each side.

**Whole segment** (`results/hecate-timing.json`, `map_diffs`). On `r96`, bf16, stride 32, inkrs's forward map matches `hecate.py --precision bf16 --batch-size 8` at r 0.999987, max |diff| 3/255, 98.41% identical, and it passes at batch 8, at batch 32 and with `--both`. Against gomyway's published Hecate maps, which used the same settings, it gets r 0.999986 forward (max 3/255) and r 0.999986 reverse (max 2/255), and the reverse passes every check. The published forward map has no JSON sidecar, so that one check can't run there; `hecate.py`'s own run gets r 0.999988 against it.

**Speed** (`results/hecate-timing.json`). The README's [How fast](../README.md#how-fast) has Hecate against inksweep's Hecate fast mode. On the whole segment, inkrs in bf16 is 2.2× faster than `hecate.py` at the same precision and batch, and 2.1× faster at both tools' defaults. It's 1.45× faster than `hecate_fast --no-compile` (`ArcheyChen/vesuvius-eligible-scouting@c73d0dd5`, MIT, the September 2026 "Faster Hecate inference" award). The gain is modest because both run the convolutions in cuDNN.

## v8-in and A-hn

v8-in (Youssef Nader, huggingface.co/YoussefMoNader/ink-8um-v8in at `d89166b4`, MIT) is a 3D ResNet-50 encoder with a 2D decoder over 24 layers at 8.64 µm, 83.4M parameters. A-hn is Prodigle's three fine-tunes of it for PHerc. 0306B (huggingface.co/Prodigle/pherc0306b-ink-ahn, weights CC BY-NC 4.0). Export each with `python -I scripts/export_v8in.py V8IN_DIR WEIGHTS NAME OUT_DIR`. inkrs folds the ×4 upsample into the stem convolution and BatchNorm into the weights, and runs the convolutions through cuDNN with fused bias and ReLU.

- **Maps:** on a 1260 × 4410 claim crop and a 1260 × 1260 edge crop of the 306B segment, every map (v8-in in both conventions and each A-hn seed in `306b`, on both crops, on the sm120 tier and with sm80 forced) matches ink8um's `predict_stack` and the 306B runner: worst r 0.999993, p99.9 0.64/255, max 1.12/255, identical coverage. The sm80 maps equal the sm120 maps bit for bit. [Turing](#turing-tesla-t4) has the same checks on a T4.
- **Stages:** on 256 real tiles every stage is within the reference's own fp16 error. On one synthetic constant tile (InstanceNorm variance 0) some A-hn decoder stages exceed it; the probability passes. The cause is unverified.
- **Speed:** 6.99 ms per tile of GPU time at batch 16. End to end on the claim crop, 84.0 s against 152.6 s for inksweep's fast mode compiled, 1.82× faster (95% CI 1.74 to 1.90×), and 2.30× faster than inksweep without compile. ink-sdk (Leonel Wainer and Paulo Sergio Camillo, MIT) also runs v8-in and the A-hn fine-tunes, through the model's own `predict.py`; `results/public-ports.json` compares inkrs with its v8-in lane.

## villa's 3D ink U-Net

villa's `ink_3d_dino_guided` v3 (`ckpt_78k_fullsup.pth`), exported with `python -I scripts/export_unet3d.py CKPT OUT_DIR` and run with `inkrs ink3d`. It needs compute capability 8.0 or above.

On a crop of a scan volume, with 256³ patches at villa's overlap 0.5, against villa eager in bf16 with its GPU blend (`results/ink3d-gates.json`): r 0.999996, p99.9 1/255, max 7/255 and identical coverage, the same with the sm80 tier forced. 0.029% of voxels differ by more than 1/255. At overlap 0.25 it's r 0.999996, p99.9 1/255, max 8/255.

## The prize recipes' models

The two 2026 First Letters prizes used these models. Each one except Hecate is a villa `vesuvius_unet_3d_stem_2d` checkpoint with the same tensors, `module_repr` and `model_config` as `hybrid_3d2d-seed42`: 264 tensors plus 244 aliases, 34,546,498 parameters, robust MAD normalisation (percentiles 1 and 99) per 17-layer tile, and FP16 autocast. So inkrs runs them through `infer` and `sweep` with no new engine code. Rectina M1 has seed 42's 17 × 128 × 128 tile; Rectina M2 and M4 have Reader V3's tensors and its 17 × 384 × 384 tile. The export also checks `model_type` and `model_config`, since a checkpoint can change an option of the network without changing the tensor shapes. ink-sdk (Leonel Wainer and Paulo Sergio Camillo, MIT) is a public runner for TRACE v3B and Rectina M4 as well.

| Model | Used in | Weights and licence | Voxel size it was trained for | Export |
|---|---|---|---|---|
| dense_native step 16000 (E. Nieuwlaar) | 343 | `Nieuwlaar/ink9um-dense-native@b894a04`, a bare safetensors state dict with its training config; MIT | about 9.362 µm stores, fine-tuned with native PHerc. 0139 segments; the 343 recipe reads 8.64 µm directly | `export_weights.py dense_native_016000.safetensors W --config train_dense_native.json --name dense_native-016000` |
| ink_9um seed 42, step 75000 | 343 | `scrollprize/ink_9um@7109667`; MIT | 9 µm class | `export_weights.py hybrid_3d2d-seed42/step-075000.pth W` |
| ink_9um seed 43, step 75000 | 343 | same repository; MIT | 9 µm class | `export_weights.py hybrid_3d2d-seed43/step-075000.pth W` |
| Hecate 9.6 µm | 343 | `scrollprize/hecate@9cb86e5`; MIT | 9.6 µm (the recipe resamples 8.64 µm renders) | `export_hecate.py` |
| TRACE v3B (J. S. Gill, N. N. Tran) | 306B (× A-hn) | `v3B_best.pth`, release v1.0 of `Jashann/vesuvius-scrolling`, SHA-256 `812ffe2b…`; CC BY-NC-SA 4.0 (non-commercial, share-alike) | 9.362 µm per its card; the 306B recipe reads 8.64 µm directly | `export_weights.py v3B_best.pth W --name trace-v3b` |
| Rectina M1 (Giovanni Pellerano) | not a prize recipe | `evilaliv3/rectina@2f9eaf0`, `m1/ckpt_130001.pth`, SHA-256 `3a84f453…`; weights CC BY-NC 4.0, recipe code MIT | 9 µm class | `export_weights.py ckpt_130001.pth W --name rectina-m1` |
| Rectina M2 (Giovanni Pellerano) | not a prize recipe | `evilaliv3/rectina@2f9eaf0`, `m2/ckpt_020000.pth`, SHA-256 `dc39d5fa…`; weights CC BY-NC 4.0, recipe code MIT | 9 µm class | `export_weights.py ckpt_020000.pth W --name rectina-m2` (needs the 384-tile engine) |
| Rectina M4 (Giovanni Pellerano) | not a prize recipe | `evilaliv3/rectina@2f9eaf0`, `m4/ckpt_020000.pth`; weights CC BY-NC 4.0, recipe code MIT | 9 µm class | `export_weights.py ckpt_020000.pth W --name rectina-m4` (needs the 384-tile engine) |

The bare dense_native export and an export of the `.pth` the recipe's `rebuild_dense_native_pth.py` writes hold the same tensors and metadata. TRACE v3B's payload records step 74999, its base model's, although the release names it step 5000. A-hn, 306B's other factor, is the v8-in fine-tunes [above](#v8-in-and-a-hn).

**Against villa `0fb38c45` eager** (`results/prize-models-gates.json`). Same bar as the readers: r ≥ 0.9994, p99.9 and max |diff| ≤ 4/255, and the kept tiles and covered pixels equal to villa's schedule. S1 is 18 maps, d −4..4 on both sides. The whole render is the plain PHerc. 343 render (66 × 2560 × 6720), d −2..2 on the reading side, 5 maps. On every model and both inputs, the maps with the sm80 tier forced are identical pixel for pixel.

| Model | S1 | Whole render |
|---|---|---|
| dense_native 16000 | min r 0.9999939, p99.9 1, max 1 | min r 0.9999972, p99.9 1, max 1 |
| ink_9um seed 43 | min r 0.9999902, p99.9 1, max 2 | min r 0.9999952, p99.9 1, max 2 |
| TRACE v3B | min r 0.9999940, p99.9 1, max 1 | min r 0.9999978, p99.9 1, max 1 |
| Rectina M4 (384 tiles, stride 192) | min r 0.9999949, p99.9 1, max 1 | min r 0.9999987, p99.9 1, max 1 |
| Rectina M1 (128 tiles, stride 64) | min r 0.9999976, p99.9 1, max 1 | min r 0.9999990, p99.9 1, max 1 |
| Rectina M2 (384 tiles, stride 192) | min r 0.9999966, p99.9 1, max 1 | min r 0.9999989, p99.9 1, max 1 |

ink_9um seed 42 is the model inkrs was built around; the README has its results. Hecate's checks are in its own section above; on the 343 render, each of its 20 recipe maps (d −2..2, plain and padded, with and without the sm80 tier forced) passes Hecate's map bounds.

**The 343 combine.** The recipe's own runners (villa eager for the three villa models, and `hecate.py`) and `recipe.py` each produce d −2..2 on both renders, and the author's `merge_tile_phase.py` and `ensemble_maps.py` combine each side unchanged. inkrs's pooled statistics and five d = +1 images (the ensemble and each model's) match theirs: worst image r 0.999983, p99.9 2 and max 4 grey levels, also with the sm80 tier forced. `ensemble_maps.py` writes 0 both for invalid pixels and for valid ones at or below the stretch's p1, so a zero on one side may be at most 4 levels on the other; every such pixel on this render is 1 or 2.

**Speed** (`results/recipe343-timing.json`). On the plain render, d 0..1, both sides, the four models and the recipe's ensemble, inkrs is 1.615 times as fast (95% CI 1.592–1.638) as the recipe's own `run_depth_sweep.sh` (villa compiled, batch 4, one villa job beside one Hecate job): 338.2 s against 544.3 s. That script beat villa main with PR #2428's flags in one run of each (550 s against 631 s). Hecate dominates both. In inkrs, the three villa models take 147 µs of GPU time per 17 × 128 × 128 tile (5545 tiles/s, 10 s for the slice), while Hecate's fp32 takes 7.8 ms per tile (129 tiles/s, 283 s of the 338 s).

### Test-time augmentation

`--tta 8` runs the PHerc. 0306B recipe's 8-fold dihedral test-time augmentation, the TRACE v3B factor of that recipe. Each map is read twice, once on the 17-layer window as stored and once turned a quarter, and in each read every tile runs as four copies (as stored, flipped in y, in x and in both) whose sigmoids are averaged before villa's Hann blend. The result is `(q0 / 255 + rot90(q1, -1) / 255) / 2` in float32, written as `.npy`. It costs about eight times the network work of a single read, runs on CUDA only, and takes neither `--coverage` nor `--check-finite`.

### Running the recipes

`scripts/recipe.py` runs a published ensemble with inkrs in place of its GPU runners. It then runs the recipe author's own combine scripts unchanged, after checking their SHA-256:

```
python -I scripts/recipe.py 343 --recipe pherc343-first-letters --render R_66.zarr --padded R_66_padded.zarr \
  --node-source R_node_source.npz --weights W --out OUT [--depths -15..15] [--write-depths 1]
```

For PHerc. 343, the three villa models run as one `inkrs sweep` per model and render. Hecate's 9.6 µm input is resampled in the same process with `hecate_prep.py`'s arithmetic, checked bit for bit against the script, and `inkrs hecate` runs it at the recipe's fp32 and batch 4. The recipe's `merge_tile_phase.py` and `ensemble_maps.py` then merge the two tile phases, pool each model's statistics over every depth and average the four z-scores. Only the reading side feeds the ensemble, so by default only that side is computed; `--sides both` computes both, as `run_ink.sh` does. Each model is found in W by the SHA-256 of the checkpoint its export was made from, against the recipe's `ink/models.tsv`.

`--hecate-inputs DIR` reads Hecate's inputs from `DIR/<plain|padded>-<zc>.npy` (written beside `hecate_prep.py`'s zarr v3 output) instead of resampling in process. `--stage gpu` stops after the maps, and `--stage combine` runs only the author's scripts on maps already in OUT.

`recipe.py 306b-fuse --trace fwd_tr.npy --members A0.npy A1.npy A2.npy` runs 306B's `fuse_v7.py` product rule on maps from any runner.

- `--ahn S0 S1 S2 --render R41.zarr` in place of `--members` runs each v8-in seed's export through `inkrs infer --convention 306b`, one after another, and stores each map as float16, as the recipe does. `fuse_v7.py` then carries the NaNs where no tile ran.
- `--trace-export TRACE_V3B.safetensors` in place of `--trace` makes the TRACE v3B factor with inkrs, the way `ink_v2.py --models tr` does: the render in row strips of 1400 rows plus 192 px of context, the columns cut to the crop plus 192 px, and in each strip one `inkrs sweep --tta 8` over depths −1, 0 and +1 (forward, layer 20 the centre), whose three maps are averaged and cut back to the strip. The crop is `ink_v2.py`'s default, the claim's context (`116,4167,694,7639`); `--trace-crop ""` reads the whole render. The result is `OUT/fwd_tr.npy`, float16, NaN outside the crop. The export must be of the checkpoint the method's `checkpoints.tsv` pins, and the strip loop's source scripts must have their pinned SHA-256.

Each stage's wall time goes to `OUT/recipe-306b-fuse.json`.

## Other GPUs

### The sm80 tier

With the sm80 PTX tier forced on the RTX 5090 (`INKRS_PTX_TIER=sm80`, `results/ptx-tiers-models.json`):

- reader-v2 matches villa with the same figures, and its maps are byte-identical to the sm120 tier's (`results/integration-v0.1.json`).
- Reader V3 passes the same normalisation, layer and map checks and writes maps byte-identical to the sm120 tier's.
- Hecate passes the same per-layer and map checks against `hecate.py` (`results/integration-v0.2.json`), and its whole-segment bf16 maps equal the sm120 tier's pixel for pixel.

### Turing (Tesla T4)

On a Tesla T4 inkrs loads the sm75 tier. Hecate runs at fp32 only, because bf16 is refused below compute capability 8.0.

| Map | Reference | r | max |
|---|---|---|---|
| reader-v2, native crop | villa eager, 5090 | ≥ 0.999996 | 1 |
| reader-v2, native crop | villa eager, same T4 | ≥ 0.999996 | 1 |
| Hecate fp32 crop, forward / reverse | `hecate.py` fp32 (TF32 convs), 5090 | 0.999986 / 0.999986 | 2 / 2 |
| Hecate fp32 crop, forward / reverse | `hecate.py` fp32 (no TF32), same T4 | 1.0000000 / 1.0000000 | 1 / 1 |
| Hecate fp32 edge fixture | `hecate.py`, 5090 and same T4 | ≥ 0.99986, same kept tiles | 1 |

On the T4, the reader-v2 crop sweep of 10 maps takes 15.5 s against 145 s for villa eager (`--no-compile`, batch 4, 5 processes), and Hecate fp32 on the crop, forward and reverse, takes 741 s against 771 s for `hecate.py` at batch 8. These are single runs, not timed comparisons. inkrs's GPU time on the T4 is 1.31 ms per tile for reader-v2 (8,130 tiles) and for seed42 (S1, 14,724 tiles), against 0.19 ms per tile for seed42 on the 5090.

The prize recipes' models also run on a T4 (`results/t4-models.json`), checked against the same references as on the 5090. The 384-tile readers run at batch 8, the 128-tile readers at 64.

All six match on both inputs, with p99.9 1 and max 1 except ink_9um seed 43 at max 2. The lowest r is ink_9um seed 43's 0.9999896 on S1.

- **The 343 combine:** `recipe.py 343 --stage gpu` on the T4, combined with the recipe's scripts, matches the recipe's own runners on the 5090 (worst image r 0.999983, p99.9 2, max 4 grey levels), and each of the 10 Hecate fp32 maps passes Hecate's map bounds.
- **v8-in and A-hn:** all 10 maps (v8-in in both conventions and each A-hn seed in `306b`, claim and edge crops) match the reference runners with identical coverage (r ≥ 0.999993, p99.9 ≤ 0.64/255, max 1.10/255). Against the 5090's sm120 maps r ≥ 0.9999998 and max 0.11/255. `inkrs.Engine` on the T4 writes the CLI's map bit for bit. GPU time is 64 ms per tile at batch 16, against 6.99 ms on the 5090 (`results/v8in-306b.json`).

## Limits

- Each model was checked on one segment (plus PHerc. 343's S1 crop for Reader V3), and most checks ran on two GPUs: the RTX 5090 and, for the sm75 tier, a Tesla T4. reader-v2, seed42, Reader V3 and Hecate also ran on four other GPUs ([results/README.md](../results/README.md)).
- Hecate 2.4 µm and `hecate.py --output-3d` aren't supported.
- inkrs runs no ensemble itself. `scripts/recipe.py` leaves the combine to the recipe authors' scripts.
- From Python: on CUDA, reader-v2 has run through `Engine` from a patched villa, and the A-hn seeds have been checked against the CLI. On Metal, reader-v2 and Reader V3 have run through villa's adapter and match the CLI's maps there. The prize models have only run through the CLI. `inkrs.hecate` runs the CLI's Hecate code.
- Every check on this page is on CUDA. The Metal backend (macOS) runs the villa readers and Hecate, and has only run on an M4 Pro.
- The published maps came from villa main and an unpinned `hecate.py`, so comparing with them is a cross-check, not the parity check.
