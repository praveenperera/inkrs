# Fast mode

`--fast` (CLI) and `fast=True` (Python) give up parity with villa's and `hecate.py`'s maps in exchange for speed. A model only gets a fast preset if it keeps its ink accuracy against public labels. Without `--fast` everything runs in parity mode: the readers stay within villa's bar against villa eager (r ≥ 0.9994, p99.9 and largest |difference| at most 4/255, the same covered pixels) and Hecate within its bounds against `hecate.py` ([MODELS.md](MODELS.md)). Fast mode's maps are judged against public ink labels instead.

## Presets

- Hecate 9.6 µm: bf16, stride 32, batch 32. `inkrs hecate --fast` is exactly `--precision bf16 --batch-size 32` (stride 32 is the default). You can still set `--batch-size`, but as in PyTorch the batch size can change which cuDNN algorithms run, and so the last bits of the map. `--stride`, `--overlap`, `--precision` and `--no-tf32` are refused alongside `--fast`.
- Readers (reader-v2, hybrid_3d2d-seed42): no reader has a fast preset. They stay in parity mode, stride 64 (overlap 0.5) in fp16. Every stride I tried above 64 (72, 80, 96 and 128) lost accuracy past the rule for both readers, so `--fast` refuses a reader before any GPU work.
- Ensemble (reader-v2 + Hecate, gomyway's recipe): reader-v2 in parity mode at stride 64, and Hecate with `--fast`. Every faster pairing failed.

The presets live in `src/fast.rs`. `inkrs fast-presets` and `inkrs.fast_presets()` print them.

## The rule

I fixed this rule before running the accuracy grid. The grid scores pixel AUC and AP against public ink labels on PHerc. 1447 (2 segments), PHerc. 0841 (3) and PHerc. 343 (1), with 95% CIs from a block bootstrap and paired differences against parity mode. The scores and verdicts are in `results/accuracy-speed.json`. A setting passes only if all three of these hold on every segment that counts for its model:

1. The paired ΔAUC CI's lower end is at or above −0.005.
2. The point ΔAUC is at or above −h, where h is the half-width of parity mode's own AUC CI there.
3. The paired ΔAP CI's lower end is at or above −0.01.

Each segment that counts is one check, and reader rows also check the other side wherever it's informative. PHerc. 343 doesn't count for reader-v2, which may have trained on that object. The preset is the passing setting with the lowest GPU time per map.

## Trade-off

The accuracy columns show the worst segment: the most negative point ΔAUC and the lowest end of its 95% CI, with each one's segment where they differ. Grid speed is GPU seconds per map on the labelled crops, relative to parity mode.

| setting | tiles / map | grid speed | worst paired ΔAUC (point; CI low; segment) | worst ΔAP CI low | verdict |
|---|---|---|---|---|---|
| reader-v2 stride 64 (parity) | 2757 | 1.00× | reference | reference | parity |
| reader-v2 stride 96 | 1235 | 2.23× | −0.0145; −0.0314; 0841-ag1742 | −0.0326 | fails 5 of 6 |
| reader-v2 stride 128 | 710 | 3.82× | −0.0473; −0.0634; 1447-w060 | −0.0792 | fails 6 of 6 |
| seed42 stride 96 | 1235 | 2.21× | −0.0352; −0.0610; 1447-w060 | −0.0807 | fails 7 of 7 |
| seed42 stride 128 | 710 | 3.81× | −0.0466 (1447-w058); −0.0705 (0841-ag1742, other side) | −0.1260 | fails 7 of 7 |
| Hecate fp32 stride 32 (parity) | 10143 | 1.00× | reference | reference | parity |
| **Hecate bf16 stride 32 (`--fast`)** | 10143 | 2.21× | −0.0006 (0841-w00); −0.0010 (1447-w060) | −0.0018 | **passes** |
| Hecate bf16 stride 48 | 4526 | 4.97× | −0.0231; −0.0368; 1447-w058 | −0.0496 | fails 6 of 6 |
| Hecate bf16 stride 64 | 2563 | 8.77× | −0.0411 (0841-w00); −0.0582 (1447-w060) | −0.0929 | fails 6 of 6 |

A second grid tried the strides between parity and the first failure, under the same rule. Every one failed:

| setting | tiles / map | grid speed | worst paired ΔAUC (point; CI low; segment) | verdict |
|---|---|---|---|---|
| reader-v2 stride 72 | 2182 | 1.25× | −0.007; −0.020; 1447-w060 | fails 6 of 6 |
| reader-v2 stride 80 | 1774 | 1.54× | −0.025; −0.045; 0841-ag1742, other side | fails 6 of 6 |
| seed42 stride 72 | 2182 | 1.25× | −0.022; −0.049; 1447-w060 | fails 7 of 7, clean segments too |
| seed42 stride 80 | 1774 | 1.54× | −0.033; −0.066; 0841-ag1445 | fails 6 of 7, clean segments too |
| Hecate bf16 stride 40 | 6503 | 3.44× | −0.017; −0.033; 0841-ag1742 | fails 6 of 6, clean 343 too |

Even where a setting's point ΔAUC is near zero, its paired CI reaches below −0.005 on almost every segment. Hecate bf16 at stride 32 is the only faster setting that passed, including on PHerc. 343, the one segment that's clean for Hecate.

Not every segment is clean for every model, which limits how far each verdict reaches. reader-v2 may have seen all of its labelled segments in its selection, so its numbers can flatter it, but its failures still stand: a stride that loses accuracy on familiar data has no reason to keep it on new data. seed42 is clean on all six segments and fails at every stride above 64 there too. Hecate is clean only on PHerc. 343, and its bf16 stride-32 pass holds there as well as on the five segments it may have seen.

## Ensemble

gomyway's recipe averages reader-v2 and Hecate. With reader-v2 at stride 64 and Hecate under `--fast`, the ensemble's mean AUC over the five segments that count for it is 0.842, against 0.814 for reader-v2 alone and 0.817 for Hecate alone in parity mode. It doesn't beat both single models on every segment, and I'm not claiming it beats villa. On the grid, its two inkrs runs (reader-v2 stride 64 and Hecate `--fast`) take 24.95 s together.

## Expert flags

These stay available, off by default. Each one leaves parity, and its cost is the grid's worst case:

| flag | model | accuracy cost on ink labels (worst paired ΔAUC) |
|---|---|---|
| `--stride 72` | readers | −0.007 (reader-v2), −0.022 (seed42) |
| `--stride 80` | readers | −0.025 (reader-v2), −0.033 (seed42) |
| `--stride 96` | readers | −0.0145 (reader-v2), −0.035 (seed42) |
| `--stride 128` | readers | −0.047 (reader-v2), −0.047 (seed42) |
| `--precision bf16` (stride 32) | Hecate | −0.0006: passes, and is `--fast` |
| `--precision bf16 --stride 40` | Hecate | −0.017 |
| `--precision bf16 --stride 48` | Hecate | −0.023 |
| `--precision bf16 --stride 64` | Hecate | −0.041 |
| `--no-tf32` (strict fp32) | Hecate | not scored; hecate.py runs TF32 convolutions |
| `--precision fp16` | readers | the only reader precision (the kernels are FP16-only) |

A reader's `--overlap` sets the same strides: 0.4375 is stride 72, 0.375 is 80, 0.25 is 96 and 0 is 128. Other strides weren't scored.

## Wall time

On the RTX 5090, over the whole PHerc. 1447 segment (`results/release-timing.json`):

| tool and setting | wall | speed-up |
|---|---|---|
| villa main with #2428's flags, reader-v2 | 51.4 s | reference |
| inkrs reader-v2 stride 64 (parity) | 3.39 s | 15.2× villa |
| `hecate.py` defaults | 81.8 s | reference |
| inkrs Hecate fp32 stride 32 (parity) | 30.8 s | 2.66× `hecate.py` |
| **inkrs Hecate `--fast` (bf16, stride 32, batch 32)** | 14.66 s | 5.58× `hecate.py`, 2.10× parity |

### Settings against each other

I also timed the settings against each other on the RTX 5090 (`results/fast-mode-timing.json`).

Hecate on all of PHerc. 1447 at 9.6 µm (28 × 2682 × 2916, 4682 tiles), forward:

| setting | wall (median) | GPU s (median) | µs per tile | speed-up vs parity (95% CI) |
|---|---|---|---|---|
| parity (fp32, TF32 convs, stride 32, batch 8) | 37.62 s | 35.11 | 7499 | reference |
| **`--fast` (bf16, stride 32, batch 32)** | 18.64 s | 15.98 | 3412 | **2.01× (1.93 to 2.09)** |
| bf16 stride 40, batch 32 (fails the rule) | 12.98 s | 10.19 | 3410 | 2.90× (2.88 to 2.92) |

bf16 halves the time per tile: 35.11 / 15.98 = 2.20× less GPU time, which matches the grid's 2.21×. The rest of each run, 2.51 s at parity and 2.66 s with `--fast`, is start-up, weight loading, planning, reading and writing, so the whole process gains 2.01×. A wider stride doesn't change the time per tile. It just runs fewer tiles.

reader-v2 `sweep` on the PHerc. 343 render (66 × 2560 × 6720), depths −4 to 4, both directions, 18 maps per process. Every stride above 64 fails the rule, so the other rows only show what each would have saved:

| stride | tiles | wall (median) | µs per tile (GPU) | speed-up vs stride 64 (95% CI) |
|---|---|---|---|---|
| 64 (parity) | 50346 | 9.59 s | 150 | reference |
| 72 | 39924 | 8.06 s | 150 | 1.19× (1.17 to 1.22) |
| 80 | 32400 | 6.95 s | 150 | 1.39× (1.34 to 1.44) |
| 96 | 22572 | 5.26 s | 150 | 1.82× (1.73 to 1.90) |
| 128 | 12924 | 3.69 s | 150 | 2.58× (2.42 to 2.74) |

### Several depths in one villa process (`--layer-starts`)

villa with `--engine inkrs` on the PHerc. 343 render, reader-v2, 9 windows (starts 21 to 29), both directions: one `--layer-starts 21:30` process against one process per window, which is how a depth sweep runs in villa today, timed twice:

| setup | round 1 | round 2 |
|---|---|---|
| one process per window (9 processes) | 92.0 s | 104.3 s |
| one `--layer-starts` process | 70.0 s | 81.8 s |
| saved | 22.0 s | 22.5 s |

That's 2.78 s saved for each window after the first, about a quarter of a 10.9 s single-window run. Both setups drifted between rounds, by 12.3 s and 11.8 s, so the ratio's interval is wide: 1.29× (1.05 to 1.54). The 18 `--layer-starts` maps are byte-identical to the per-window maps, and distinct from one another.

## How fast mode is checked

These checks ran on the RTX 5090. The map checks compare outputs byte for byte.

- Parity mode is unchanged: reader-v2's 18-map PHerc. 343 sweep and Hecate's default run are byte-identical to inkrs before fast mode, on the sm_120 tier and on the sm_80 tier forced with `INKRS_PTX_TIER`.
- `--fast` is exactly its preset: `hecate --fast` equals the explicit `--precision bf16 --batch-size 32` run byte for byte, and `inkrs.hecate(fast=True)` equals the CLI.
- The blend holds at any stride: `inkrs check-blend` runs both blend kernels against a CPU reference on both tiers, at strides from 1 to the tile edge, on uneven sides, with skipped tiles, uncovered pixels and an empty volume:
  - coverage must match exactly, including a window around villa's 1e-6 cut;
  - Hecate must match exactly, including planted half steps that round up and down to even;
  - the reader must match exactly away from quantisation boundaries, and within one step elsewhere;
  - a logit-0 field must give exactly 127 wherever a pixel is covered.
