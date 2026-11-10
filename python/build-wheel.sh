#!/usr/bin/env bash
# Builds the inkrs wheel: manylinux_2_28 x86_64, abi3 (CPython 3.9+), the kernels embedded as PTX; on macOS, an
# arm64 wheel for the Metal engine (macOS 26 or later)
#
#   python/build-wheel.sh [OUT_DIR]      # default target/wheels
#
# Stops when scripts/check-ptx.sh finds the committed PTX stale. Needs Rust and uv; maturin and zig come from
# PyPI through uvx, and zig links against glibc 2.28, so the wheel installs on any manylinux_2_28 distribution. The wheel embeds the PTX src/gpu.rs embeds, so a change to the
# PTX set there needs no change here; the build log lists it from the built module
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
jobs="${CARGO_BUILD_JOBS:-4}"
maturin=(uvx --from 'maturin[zig]==1.16.0' maturin)

# macOS: an arm64 wheel for the Metal engine. The shaders ship as source and compile when the engine opens, so
# there is no PTX to check. Metal 4's tensor ops need macOS 26, hence the deployment target. BSD realpath has no
# -m, so the output directory is made first
if [ "$(uname -s)" = Darwin ]; then
  out="${1:-$here/../target/wheels}"
  mkdir -p "$out" && out="$(cd "$out" && pwd)"
  cd "$here"
  MACOSX_DEPLOYMENT_TARGET=26.0 "${maturin[@]}" build --release --target aarch64-apple-darwin -j "$jobs" --out "$out"
  wheel="$(ls -t "$out"/inkrs-*-cp39-abi3-macosx_*_arm64.whl | head -1)"
  unzip -l "$wheel"
  echo "$wheel"
  exit 0
fi

out="$(realpath -m "${1:-$here/../target/wheels}")"

# maturin runs no Rust tests, so check here that the embedded PTX matches its kernel sources; otherwise a
# kernel edit or a bad merge would ship stale PTX
"$here/../scripts/check-ptx.sh"

cd "$here"
"${maturin[@]}" build --release --zig --compatibility manylinux_2_28 --target x86_64-unknown-linux-gnu -j "$jobs" \
  --out "$out"
wheel="$(ls -t "$out"/inkrs-*-cp39-abi3-manylinux_2_28_x86_64.whl | head -1)"

# what got built: the platform tag's symbol check, and the PTX inside
uvx --from 'auditwheel==6.8.2' auditwheel show "$wheel"
unzip -p "$wheel" inkrs/_inkrs.abi3.so | grep -a -o -E '^\.(target|version) [^ ]+' | sort | uniq -c
echo "$wheel"
