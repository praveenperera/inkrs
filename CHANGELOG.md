# Changelog

All notable changes to inkrs are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

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

[0.2.0]: https://github.com/praveenperera/inkrs/releases/tag/v0.2.0
