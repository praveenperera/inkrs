# Installing the CLI

For the Python package, see [python/README.md](../python/README.md). Its wheel also installs the `inkrs` command, the CLI's own code run inside Python, so it needs no Rust.

## Prebuilt

The `inkrs` CLI comes prebuilt for Linux x86-64, Windows x86-64 and Apple silicon Macs, through [cargo-binstall](https://github.com/cargo-bins/cargo-binstall):

```
cargo binstall inkrs
```

The prebuilt Linux binary needs glibc 2.28 or newer (RHEL 8, Debian 10, Ubuntu 20.04 and later). The archives are also attached to each [GitHub release](https://github.com/praveenperera/inkrs/releases).

## From crates.io

```
cargo install inkrs --locked
```

The GPU kernels ship compiled for the common NVIDIA GPUs, from the T4 to the RTX 50 series, and as PTX that the driver compiles for any other GPU, so all you need is Rust, not the CUDA toolkit.

## What it needs to run

### Linux

An NVIDIA driver new enough for CUDA 12 (r525 or later, r570 or later on the RTX 50 series), plus cuDNN 9 and cuBLAS 12. The CLI opens them by name, so it finds them wherever the system's dynamic loader does: installed system-wide, or in a directory on `LD_LIBRARY_PATH`. It tries the unversioned names first (`libcudnn.so`, `libcublas.so`), then versioned ones such as `libcudnn.so.9` and `libcublas.so.12`.

Unlike the Python package, it doesn't look inside a Python environment, so to use the pip copies, put their directories on the path:

```
site=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
export LD_LIBRARY_PATH="$site/nvidia/cudnn/lib:$site/nvidia/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### Windows

The same driver, plus cuDNN 9 (`cudnn64_9.dll`) and cuBLAS 12 (`cublas64_12.dll`). The CLI looks for them next to `inkrs.exe`, in System32, in each absolute `PATH` entry, then under `%CUDA_PATH%`, so a CUDA 12 toolkit works, and so do the pip copies once their `nvidia\cudnn\bin` and `nvidia\cublas\bin` folders are on `PATH`.

It never loads a DLL from the current folder or a relative `PATH` entry; if one is there, it stops and names the file.

### macOS

Apple silicon (M1 or newer) and macOS 26 or newer, for Metal 4. The binary starts on macOS 11 and later, but before 26 it stops with an error naming the version it needs. It needs no CUDA libraries and no Xcode, since the Metal shaders ship as source and compile when the engine starts.

The readers run in fp16 by default on Metal too, and their maps already land within 2/255 of villa's own float32 PyTorch run on MPS. For a closer match, `--precision fp32` (`inkrs.Engine(..., precision="fp32")`) runs them in float32: on an M4 Pro its maps came within 1/255 of PyTorch's, with 0.004% of pixels off by one against 5% at fp16, at 1.2-1.3x the run time and twice the activation and logit memory ([results/metal-fp32.json](../results/metal-fp32.json)). fp32 is Metal only; on CUDA it raises an error, and the Neural Engine offload and v8-in don't take it.
