# inkrs: villa's ink reader, GPU-resident, in Rust

inkrs runs villa's `hybrid_3d2d-seed42` ink model (ink_9um, seed 42, step 75,000) and writes the same maps villa's `infer` writes for the PHerc. 343 recipe: same tiles, normalisation, network numerics, blend and uint8 TIFFs. One process loads the volume and the model onto the GPU once, then produces all 18 maps of a depth sweep (d −4..4, both directions). Most convolutions run in cuDNN at FP16. Everything else, including the stem's two 3D convolutions, runs in my own kernels, written in Rust with cuda-oxide and committed both compiled for the common NVIDIA GPUs and as PTX, so you don't need the CUDA toolkit to build it. TIFFs get encoded on writer threads while the next maps compute.

A map only counts if it's within villa's own bar against villa eager: r ≥ 0.9994, p99.9 and largest |difference| at most 4/255, and the same covered pixels. Every speed number below is for maps that pass.

## Why

The depth sweep is the recipe's slowest stage. The recipe calls villa once per depth, and every call reloads the model. villa has got faster lately, and has a set of faster flags, but it still runs one PyTorch process per depth.

The fastest public PyTorch path I timed for this reader is inksweep's fast mode, which runs every depth in one process with `torch.compile`. UFSM (`SuperOptimizer/ufsm`) is a public C/CUDA port with its own fp16 kernels, compiled ahead of time. Warm, inkrs is 2.35× faster than UFSM on one S1 map (95% CI 2.12× to 2.57×, `results/public-ports.json`). A single run on a fresh machine, with every cache empty, takes inkrs 0.86 s (`results/first-run.json`) against UFSM's 1.87 s (`results/public-ports.json`), because inkrs's kernels come compiled for this GPU too.

inkrs replaces the whole pipeline for one reader and keeps the output contract.

## Speed

On one RTX 5090, each tool writing its usual output: villa's uint8 LZW TIFFs for inkrs, calibrated float32 maps for inksweep. The numbers are in `results/release-timing.json`.

| Input | | inkrs | inksweep fast mode, compiled | inkrs faster by |
|---|---|---|---|---|
| PHerc. 343 `concat_w047-w048_R5B2_z9500-11000`, 66 × 2560 × 6720, 18 maps (d −4..4, both directions) | warm | **8.50 s** | 64.5 s | **7.57×** (95% CI 7.34× to 7.80×) |
| | every cache empty | 9.73 s | 164.2 s | 16.9× |
| PHerc. 1447, Reader V3, 10 maps (d −2..2, both directions) | warm | 3.33 s | 52.4 s | 14.5× (11.9× to 17.0×) |

Warm means every compile and algorithm cache is already filled. With every cache empty, inksweep compiles its graphs and inkrs times cuDNN's algorithms. The empty-cache inkrs time in the table also includes 1.5 to 2 s for the driver to compile inkrs's PTX. On the RTX 5090 inkrs loads kernels compiled ahead instead and skips that step, with byte-identical maps (`results/first-run.json`).

villa main with the faster flags (`--batch-size 8 --no-compile --loader-threads 8 --chunk-cache-size 64`), one process per depth the way the recipe runs it, takes 161.7 s for the same 18 maps. Inside villa, through the inkrs adapter a villa PR adds (`--engine inkrs`), one depth in both directions on S1 takes 2.46 s against 14.0 s for villa's own PyTorch path, compiled and warm (5.68×, 95% CI 5.26× to 6.09×).

Since v0.1 I've moved the stem's two 3D convolutions into my own tensor-core kernels, with their InstanceNorm fused. That took reader-v2's whole-segment sweep pass from 3.46 s to 2.83 s.

## Faster settings

I also wanted to know whether giving up parity buys speed without losing ink. Under a pass/fail rule I fixed in advance, I scored larger strides, and bf16 for Hecate, against public ink labels on six segments of PHerc. 1447, 0841 and 343. No reader stride above 64 holds up: at stride 72, reader-v2's AUC already drops by 0.007 on one segment. Hecate in bf16 at stride 32 does, with a largest AUC difference of −0.0006 against fp32. [FAST.md](FAST.md) has the rest.

## How I checked that the maps are villa's

The reference is villa 0fb38c45 eager, one process per depth, on the same input.

- **Layer by layer.** On S1 tiles, against PyTorch FP32, inkrs's error on all 37 layers stays within 1.5× (RMS) and 2× (p99.9) of PyTorch FP16 autocast's own error.
- **The normaliser.** villa's per-tile robust normalisation, done on the GPU, gives villa's FP16 patch exactly on every tile of two test inputs, including the median and MAD paths and the fallbacks.
- **Every map.** On S1 and on the full segment, each of the 18 maps has r ≥ 0.99998 against villa's, with p99.9 |diff| of 1/255 and max |diff| of 2/255. The 4/255 bar is villa's own compiled-against-eager spread on one PHerc. 343 map, from inksweep's `docs/validation.md`.
- **Coverage.** inkrs's coverage mask is bit for bit the one villa's scheduling code gives, and its two empty-tile skips agree with villa's: 2,797 tiles per map on the full segment. The check catches a single planted defect: a zeroed 4 × 4 block, a coverage pixel dropped or added, or a nonzero pixel outside villa's coverage.
- **Repeatability.** Later passes equal the first pixel for pixel. Across processes that only holds while they pick the same cuDNN algorithms. inkrs caches its picks on disk, so later runs on the same setup reuse them.
- **The known letters.** I applied the recipe's single-model reading (`ensemble_maps.py` in pherc343-first-letters) to inkrs's maps and to villa's. Over the 18 published letter boxes the stretch points are identical, every depth and side has display r ≥ 0.99996 and a max difference of 3 of 255 levels, and at d = +1, reading side, 96.9% of letter pixels are identical.
- **The comparison.** inksweep's maps pass the same bar against villa's, so the speed comparison is between two correct outputs.

The evidence is in [results/](../results/), with the commands to rerun it.

## Limits

- **One reader checked here.** Every check in this write-up outside the faster-settings section is for `hybrid_3d2d-seed42`. [MODELS.md](MODELS.md) covers the other models. inkrs doesn't run ensembles.
- **One tile phase.** The recipe also runs a zero-padded phase and merges the two. inkrs only runs the unpadded one.
- **uint8 inputs only.** inkrs reads uint8 zarr v2 at every level. villa takes any dtype, so it runs on inputs inkrs refuses.
- **Timed on one RTX 5090.** On an A100, an A10, an L4, an RTX 4090 and a Tesla T4, where cuDNN and cuBLAS pick other kernels, seed42's S1 sweep still meets the same per-map bar against villa eager's maps ([results/README.md](../results/README.md)). The volume also has to fit in GPU memory, 1.1 GB for this segment.
- **The letter check is narrower than the recipe.** It pools the stretch over d −4..4, where the recipe pools over −15..15, and it covers one model, not the ensemble.
- **Some inputs are mine.** I rendered the segment with inksweep's renderer from the published mesh and the public CT volume, and I haven't checked it against a VC3D render voxel for voxel. For the multiscale check I built a 2 × 2 × 2 mean pyramid, and both sides read it. The render's fingerprint is in [results/](../results/README.md#inputs).
