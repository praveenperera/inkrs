# inkrs-afv-trace

A native, multi-threaded drop-in for `avf-spline-generator` 0.2.0, Qual's AFV fibre tracer
([Qualzz/AVF_SPLINE_GENERATOR](https://github.com/Qualzz/AVF_SPLINE_GENERATOR) @ `bee798f`). It takes the
same flags and writes the same `.afv`: every metadata value except the per-run uuid, and every fibre row,
point block and spatial-index row. The tracing algorithm, the AFV format and the fibre model whose
predictions it reads are Qual's; this port changes how fast the tracer runs, not what it writes.

## Install

```sh
pip install inkrs-afv-trace
```

Install it into the environment that has numpy (normally the one that runs `avf-spline-generator`). It
adds an `inkrs-afv-trace` command and a Python module, `inkrs_afv_trace`. Prebuilt CLI archives for Linux
x86-64 (glibc 2.28+) and Apple silicon (macOS 11+) are named `inkrs-afv-trace-<version>-<target>.tar.gz`.

## Use

```sh
inkrs-afv-trace --predictions P --array semantic --channel-order VH --tracing-predictions P \
  --tracing-array tracing --voxel-size-um 9.362 --storage compressed --device cpu --threads 4 \
  --output fibers.afv --work-dir work
python -m inkrs_afv_trace ARGS            # the same, from Python
```

`inkrs_afv_trace.run([...])` runs it from Python and returns the exit status.

To make an environment's `avf-spline-generator` run the port, so scripts that call it need no change:

```sh
scripts/install-dropin.sh VENV [BINARY]   # VENV/bin/avf-spline-generator -> inkrs-afv-trace
scripts/install-dropin.sh --restore VENV  # put the original back
```

BINARY defaults to the `inkrs-afv-trace` in the archive the script came from, then to the one on `PATH`.
The original entry is kept as `avf-spline-generator.reference`, and `NOTICE` and `licenses/` are copied to
`VENV/share/inkrs-afv-trace`. Callers that run `Path(sys.executable).parent / "avf-spline-generator"`
need no other change.

## Build

```sh
cargo build --release
cargo test --release
```

The crate is its own Cargo workspace and builds on Linux and macOS. The wheel is built with maturin from
`pyproject.toml` (`maturin build --release`; the published Linux wheel uses `--zig --compatibility
manylinux_2_28`, and the macOS wheel `MACOSX_DEPLOYMENT_TARGET=11.0`).

## Numerics

The direction field's eigenvectors come from `dsyevd` in numpy's own OpenBLAS, on float64 copies as
`numpy.linalg.eigh` computes them, so they match `eigh` bit for bit. The binary finds that library from its
own path (`bin/../lib/python*/site-packages/numpy.libs/libscipy_openblas64_*.so`), then from
`$VIRTUAL_ENV`, looking in each environment in numpy's copy first and then in the `scipy-openblas64`
package (`site-packages/scipy_openblas64/lib`). `INKRS_AFV_LAPACK` overrides both.

macOS numpy wheels link Accelerate and bundle no OpenBLAS, so on macOS the wheel depends on
`scipy-openblas64==0.3.31.188.0`, the OpenBLAS build that numpy 2.4.6's Linux wheel bundles. With it, a
Mac writes the same `.afv` as Linux.

## Inputs

The same arrays the original reads through numpy and zarr 2: `.npy` files and Zarr v2 arrays or groups of
uint8, float16, float32 or float64, in either byte order and either array order. Zarr chunks may use any
channel chunking, either dimension separator and the Blosc, Zstd, Zlib, GZip or LZ4 compressor; Blosc
frames are decoded by c-blosc 1.21.6, the release numcodecs 0.15 bundles. Missing chunks read as the fill
value, including `NaN` and infinities, which the value check then rejects as the original does. Not
supported: Zarr filters, other numcodecs compressors (BZ2, LZMA and so on) and extended-precision floats.
`tests/fixtures/readers` holds one small array in each supported encoding with what the original reads
from it; `cargo test` checks every case.

## Differences from the original

- Fields live in memory as sparse 32³ bricks instead of a Zarr work store, so peak memory is higher and
  grows with how much of the band holds fibres: 11.0 GiB against the original's 4.6 GiB on a 512 × 8169 ×
  8169 band of PHerc0826, and 18.4 GiB on a denser 512 × 7947 × 7947 band of PHerc0813. With
  `--field-cache-mib 4096`, 4.1 GiB of that is the chunk cache; a smaller cache lowers the peak by the
  difference but decodes chunks again (at 16 MiB the PHerc0826 band took about 45% longer). Memory each stage
  frees goes back to the system before the next stage allocates. `INKRS_AFV_DEBUG_MEM=1` logs each stage's
  field sizes and resident memory.
- The work directory holds `run.json` (with a `native` key, so neither tool resumes the other's
  directory) and one finished curve file per family. An interrupted family is traced again from its first
  seed on `--resume`, instead of from the last 16-seed batch.
- `--storage` and `--field-cache-mib` are validated, and `--storage` is recorded in `run.json` as the
  original records it, but neither selects anything. The chunk cache is bounded by `--native-cache-mib`
  plus `--field-cache-mib`.
- `--device cuda` is rejected: the original uses CUDA only for centrality, and this port computes
  centrality on the CPU.
- The command line replays argparse's pass, so exit status and selected values match, but help text and
  usage-error wording differ. Some argparse readings are not reproduced: numbers written with non-ASCII
  digits; non-UTF-8 bytes in the text options (`--array`, `--tracing-array`, `--coordinate-space`);
  negative numbers with an underscore or trailing whitespace inside a three-value option or after a
  numeric option (`-0_0`, `"-1\n"`), which this tool rejects and argparse accepts; and a short help
  cluster followed by a dash (`-h-`, `-h--`), which this tool treats as help and argparse rejects.
- The original's batch-buffer capacity checks are not reproduced. Its smoothing-snapshot limit
  (`--field-cache-mib`) is not either, because no snapshot is taken.

## Licence

Apache-2.0, like the original; the licence text is `licenses/apache-2.0.txt`. `NOTICE` lists the ported code
(AVF_SPLINE_GENERATOR, one NumPy sort routine) and the statically linked C libraries; their licence texts are in
`licenses/`.
