# Changelog

All notable changes to inkrs are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/praveenperera/inkrs/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/praveenperera/inkrs/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/praveenperera/inkrs/releases/tag/v0.2.0
