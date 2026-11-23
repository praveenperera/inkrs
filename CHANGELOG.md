# Changelog

All notable changes to inkrs are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-11-23

### Added

- nnU-Net ResEnc-L models: `inkrs nnunet predict` and `inkrs.nnunet_predict` run
  villa's `surface_m7_nnunet` as `vesuvius.predict --blend_in_memory --mode binary`
  does and write the same array. `scripts/export_nnunet.py` (also
  `python -m inkrs.export_nnunet`) exports a model folder once, with torch
  installed. On a 512³ zone of PHerc. 0826, m7 ran end to end in 130.6 s against
  206.5 s for villa's compiled in-memory path, 1.58x faster. It missed the parity
  bar set before the run against villa compiled (largest difference 3/255 against
  a bound of 2, on 6 of 562M values; IoU 0.9996948 against a floor of 0.9996981),
  and the bar was amended after the result was seen to accept villa eager as well,
  which it meets (largest difference 1/255, IoU 0.999805). Against villa's staged
  path the IoU is 0.999682, also under the floor.
- The wheel installs the `inkrs` command: the CLI's own code, run inside Python,
  so every CLI command runs without Rust.

### Changed

- cuDNN algorithm choices now take only entries timed under the convolution's own
  math type. Taking the first supported entry could run a tensor-op kernel on a
  workspace sized for another one, which failed with PyTorch's cuDNN 9.24 loaded.
  Run reports record the loaded cuDNN version.

### Fixed

- `inkrs ink3d --overwrite` deleted whatever `--output` named, the input volume
  or a directory holding the checkpoint included. `--overwrite` now replaces only
  an earlier prediction of the same command, never through a symlink, and
  refuses anything else, as `inkrs nnunet predict` does.

## [0.3.1] - 2026-11-22

### Fixed

- A checkpoint passed as a parent-relative path such as `../step-075000.pth`
  was exported as a reader named `..`. On Python 3.14 inkrs then could not find
  that export again, and every later run of the checkpoint failed with `File
  exists` or `Directory not empty`. Such a path now names the directory it
  points to, and exports earlier releases left in the cache are found again.
  Such an export keeps the name `..`, which `Engine.sweep(out_dir=...)` cannot
  put in file names; delete its cache entry, `weights/<sha256>` under the cache
  directory, to export it again under the run directory's name.
- A cache entry that holds no usable export now raises an error naming the entry
  to delete, instead of `File exists` or `Directory not empty`.

## [0.3.0] - 2026-11-13

### Added

- An opt-in fp32 reader on the Metal backend: `inkrs infer` and `inkrs sweep` take
  `--precision fp32`, and `inkrs.Engine` takes `precision="fp32"`. fp16 stays the
  default, and its maps are unchanged byte for byte. fp16 already lands within
  2/255 of villa's float32 PyTorch run on MPS, so fp32 is only for a closer match:
  on an M4 Pro its maps came within 1/255, with 0.004% of pixels off by one against
  5% at fp16, and it ran 1.2-1.3x slower, with twice the activation and logit
  memory. On CUDA, fp32 raises `inkrs.UnsupportedError`, since the CUDA reader
  kernels are fp16 only; v8-in exports and the Neural Engine offload refuse it
  too. An fp32 run also takes a checkpoint trained without FP16 autocast, as villa
  on MPS does.

## [0.2.1] - 2026-11-13

### Added

- The `inkrs` CLI is on crates.io: `cargo install inkrs --locked` builds it with
  only Rust installed, no CUDA toolkit.
- Prebuilt CLI binaries for Linux x86-64 (glibc 2.28 or newer), Windows x86-64
  and macOS arm64, installed with `cargo binstall inkrs` or downloaded from the
  GitHub release.

### Changed

- A villa checkpoint whose keys or tensor shapes don't match the model its config
  builds now makes `inkrs.Engine` raise `inkrs.UnsupportedError`, naming the
  missing and unexpected keys and the mismatched shapes. It used to fail as a plain
  `inkrs.InkrsError`, so a caller that falls back to PyTorch on `UnsupportedError`,
  such as villa's `--engine inkrs`, stopped instead of falling back.
  `python -m inkrs.export_weights` now reports the same keys in one line instead of
  a PyTorch traceback. The export still never runs from a partial state.
- An `UnsupportedError` for a checkpoint the exporter refuses no longer suggests
  exporting it by hand, since that export would refuse it too.

## [0.2.0] - 2026-11-12

The first public release.

### Added

- The `inkrs` CLI and the `inkrs` Python package, which run villa's ink models on
  the GPU and write the same maps villa's Python code writes. The volume and the
  model stay on the GPU, every depth and both directions of a segment run in one
  pass, and the TIFFs are written on other threads while the next maps compute.
- Models: villa's `hybrid_3d2d-seed42` and the readers with its architecture
  (`reader-v2`, Reader V3, ink_9um seed 43, dense_native, TRACE v3B, and Rectina
  M1, M2 and M4), Hecate 9.6 µm, v8-in and its three A-hn fine-tunes, and villa's
  3D ink U-Net (`ink_3d_dino_guided` v3). Some of these weights are
  non-commercial; [docs/MODELS.md](docs/MODELS.md) lists each model's licence.
- NVIDIA GPUs of compute capability 7.5 and up (T4 / RTX 20 series and newer) with
  a driver of r525 or newer, plus cuDNN 9 and cuBLAS 12. The kernels ship compiled
  for the common GPU families and as PTX for any other, so neither the wheel nor a
  source build needs a CUDA toolkit.
- Apple silicon Macs on macOS 26 or newer, through Metal, for villa's readers,
  Hecate and v8-in. The 3D U-Net and test-time augmentation run on CUDA only.
- Python wheels for Linux x86-64 (glibc 2.28 or newer), Windows x86-64 and macOS
  arm64, one abi3 wheel per platform for CPython 3.9 and newer. The `cu12` extra
  installs cuDNN 9 and cuBLAS 12 from pip on Linux and Windows. The package ships
  type information. The Windows build passes its tests but hasn't yet been run on
  Windows itself.
- `--fast` presets that trade a measured amount of accuracy for speed, described
  in [docs/FAST.md](docs/FAST.md).

[Unreleased]: https://github.com/praveenperera/inkrs/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/praveenperera/inkrs/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/praveenperera/inkrs/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/praveenperera/inkrs/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/praveenperera/inkrs/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/praveenperera/inkrs/releases/tag/v0.2.0
