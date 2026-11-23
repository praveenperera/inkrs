# inkrs against other runners

Each model's checks are in [MODELS.md](MODELS.md). The JSON behind the reader and Hecate numbers, and how to rerun them, is in [results/](../results/README.md); the nnU-Net numbers are in [MODELS.md](MODELS.md#villas-nnu-net-surface-model-m7). Everything ran on one RTX 5090 unless the row says otherwise.

S1 is a 2048 × 2048 crop of the PHerc. 343 render. UFSM is a native C/CUDA port with its own fp16 kernels.

## The villa readers

All 18 maps of PHerc. 343's First Letters render (66 × 2560 × 6720, depths −4..4, both directions) with villa's `hybrid_3d2d-seed42`:

| | Time | inkrs faster by |
|---|---|---|
| inkrs | **8.50 s** | |
| inksweep fast mode, with `torch.compile` | 64.5 s | **7.57×** |
| villa main with the faster flags from #2428 | 161.7 s | **19.0×** |

The other villa readers get the same kind of speedup, and UFSM, the closest native port, is still well behind:

| Against | Model and input | inkrs | Theirs | inkrs faster by |
|---|---|---|---|---|
| villa main with #2428's flags | reader-v2, PHerc. 1447 segment, 10 maps | 3.39 s | 51.4 s | 15.2× |
| inksweep fast mode, compiled | Reader V3, PHerc. 1447 segment, 10 maps | 3.33 s | 52.4 s | 14.5× |
| villa's own `infer`, compiled | `hybrid_3d2d-seed42` through villa's `infer` with the inkrs adapter (not in villa yet), S1, 2 maps | 2.46 s | 14.0 s | 5.68× |
| UFSM | `hybrid_3d2d-seed42`, S1, 1 map | 0.80 s | 1.91 s | 2.35× |
| UFSM | Reader V3, S1, 1 map | 0.78 s | 1.98 s | 2.49× |
| SurgeFok's MPS path, on an M4 Pro Mac | `hybrid_3d2d-seed42`, S1, 1 map | 3.45 s | 19.4 s | 5.62× |

## Hecate and v8-in

These gain less. They're big networks, and inkrs runs their convolutions through cuDNN, the same as PyTorch, so there's less overhead to cut. Hecate in bf16 is 1.70× faster than inksweep, v8-in 1.82× faster than inksweep and 2.38× faster than ink-sdk, and both about 1.2× faster than UFSM. The PHerc. 0306B claim's recipe, mostly v8-in, is 1.68× faster than its own scripts.

UFSM wins two Hecate cases:

- **Hecate's first run on a machine.** inkrs spends about 12 s of it timing cuDNN's convolution algorithms, then keeps its picks for later runs. UFSM's first run is no slower than its others, so it wins that one run: 17.7 s against 27.0 s on the whole PHerc. 1447 segment. The compiled kernels take about 1.3 s off inkrs's side, which still leaves UFSM ahead.
- **Hecate in fp32.** UFSM's fp16 Hecate takes 17.7 s, inkrs's fp32 takes 30.9 s. Use `--fast` (bf16), which keeps the ink: on the labelled segments I scored, its AUC moved by at most 0.0006 ([FAST.md](FAST.md)).

## The nnU-Net surface model

On one 512³ zone of PHerc. 0826, `surface_m7_nnunet` runs end to end in 130.6 s against 206.5 s for villa's compiled in-memory path, 1.58× faster. [MODELS.md](MODELS.md#villas-nnu-net-surface-model-m7) has how the runs were paired, and how inkrs's output compares with villa's.

## Features

| | inkrs | villa | inksweep | UFSM | ink-sdk |
|---|---|---|---|---|---|
| villa readers, speed | fastest | 19× slower | 7.6× to 14.5× slower | 2.4× slower | runs villa, so about villa's speed |
| Matches villa's maps | ✅ | the reference | ✅ | ✅ | ✅ |
| Hecate | ✅ | ❌ | ✅ | ⚠️ fp16 only | ✅ |
| v8-in and A-hn | ✅ | ❌ | ⚠️ v8-in only | ⚠️ its A-hn misses the 0306B recipe | ✅ |
| Apple silicon | ✅ Metal | ⚠️ PyTorch MPS, slower | ❌ | ❌ | ⚠️ PyTorch MPS, slower, not tested by its authors |
| Install | `pip` or `cargo` | `pip` | `pip` | build from source | `git clone`, then `pip` |
| Licence | MIT or Apache-2.0 | MIT | MIT | none | MIT |
| Training | ❌ | ✅ | ❌ | ✅ its own models | ❌ |
| Renders segments from a mesh | ❌ | ✅ VC3D | ✅ | ✅ its own surfaces | ✅ |
