#!/usr/bin/env bash
# Regenerates the committed PTX in ptx/ from the cuda-oxide kernel crate in inkrs-kernels, then its
# source manifest. Three tiers from the same source: sm_75 for Turing (T4, RTX 20), built with the
# kernel crate's `sm75` feature because sm_75 has no cvt.rn.f16x2.f32 and no m16n8k16 FP16 mma;
# sm_80 (PTX ISA 7.0), which the driver JITs for every GPU from RTX 30 / A100 up; and sm_120
# (ISA 8.7) for RTX 50. sm_80 and sm_120 build without the feature, so their PTX doesn't change
# when the sm_75 fallbacks do. src/gpu.rs embeds all three and loads the highest one the device runs.
# Needs the CUDA toolkit (libdevice, CUDA_OXIDE_LIBDEVICE) and the pinned toolchain; the host crate builds
# from the committed PTX without either
#
# cargo-oxide and its codegen backend come from the cuda-rust commit inkrs-kernels/Cargo.lock pins,
# installed under a CARGO_HOME of their own (the backend cache has one slot). Codegen uses the
# toolchain's own llc, opt and llvm-link and stops when the toolchain's llc is missing; their versions
# and libdevice's hash go to ptx/tools.txt, and the libdevice functions the kernels call to
# ptx/libdevice.txt, both covered by the manifest. A missing ptxas, or a tier it refuses, stops the build
#
#   scripts/build-ptx.sh
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
crate="$root/inkrs-kernels"
toolchain="$(sed -n 's/^channel = "\(.*\)"/\1/p' "$crate/rust-toolchain.toml")"
source="$(sed -n 's|^source = "git+\(https://github.com/[^/]*/cuda-rust.git\)?rev=[^#]*#\([0-9a-f]*\)"$|\1 \2|p' "$crate/Cargo.lock" | sort -u)"
repo="${source% *}"
rev="${source#* }"
if [[ ! "$rev" =~ ^[0-9a-f]{40}$ || "$repo" == "$source" ]]; then
  echo "expected one cuda-rust repository and commit in $crate/Cargo.lock, found: $source" >&2
  exit 1
fi

export CARGO_HOME="${INKRS_OXIDE_HOME:-$HOME/.cache/inkrs/cargo-oxide-${rev:0:12}}"
export PATH="$CARGO_HOME/bin:$PATH"
export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-21/lib}"
export CUDA_OXIDE_LIBDEVICE="${CUDA_OXIDE_LIBDEVICE:-${CUDA_HOME:-/usr/local/cuda}/nvvm/libdevice/libdevice.10.bc}"
unset RUSTUP_TOOLCHAIN CARGO RUSTC

# every tier goes through ptxas, so a missing one stops the build before it starts rather than recording
# PTX nothing checked against its target
# made absolute here, since the build later runs from inkrs-kernels/
ptxas="$(realpath -m -- "${PTXAS:-$(command -v ptxas || echo "${CUDA_HOME:-/usr/local/cuda}/bin/ptxas")}")"
if [[ ! -x "$ptxas" ]]; then
  echo "no executable ptxas at $ptxas; install the CUDA toolkit or set PTXAS" >&2
  exit 1
fi

if [[ -z "${CUDA_OXIDE_LLC:-}" ]]; then
  host="$(rustc "+$toolchain" -vV | sed -n 's/^host: //p')"
  llc="$(rustc "+$toolchain" --print sysroot)/lib/rustlib/$host/bin/llc"
  if [[ ! -x "$llc" ]]; then
    echo "no llc at $llc; run: rustup component add llvm-tools --toolchain $toolchain" >&2
    exit 1
  fi
  export CUDA_OXIDE_LLC="$llc"
fi
export CUDA_OXIDE_OPT="${CUDA_OXIDE_OPT:-${CUDA_OXIDE_LLC%llc*}opt${CUDA_OXIDE_LLC##*llc}}"
export CUDA_OXIDE_LLVM_LINK="${CUDA_OXIDE_LLVM_LINK:-${CUDA_OXIDE_LLC%llc*}llvm-link${CUDA_OXIDE_LLC##*llc}}"

stamp="$CARGO_HOME/cargo-oxide.rev"
if [[ "$(cat "$stamp" 2>/dev/null)" != "$repo $rev" ]]; then
  cargo "+$toolchain" install -j 8 --locked --force --git "$repo" --rev "$rev" cargo-oxide
  echo "$repo $rev" > "$stamp"
fi

llvm_version() {
  local version
  version="$("$1" --version | sed -n 's/^ *\(.*LLVM version .*\)$/\1/p')"
  if [[ -z "$version" ]]; then
    echo "no LLVM version from $1 --version" >&2
    return 1
  fi
  echo "$version"
}
llc_version="$(llvm_version "$CUDA_OXIDE_LLC")"
opt_version="$(llvm_version "$CUDA_OXIDE_OPT")"
link_version="$(llvm_version "$CUDA_OXIDE_LLVM_LINK")"
libdevice_hash="$(sha256sum < "$CUDA_OXIDE_LIBDEVICE")"
libdevice_hash="${libdevice_hash%% *}"
printf 'llc: %s\nopt: %s\nllvm-link: %s\nlibdevice: %s\n' \
  "$llc_version" "$opt_version" "$link_version" "$libdevice_hash" > "$root/ptx/tools.txt"

cd "$crate"
# the tiers src/gpu.rs embeds, each with the highest PTX ISA it may carry (the ISA LLVM picks for the
# target, so a toolchain bump that raises it, and with it the minimum driver, stops the build) and the
# kernel-crate features it builds with
for tier in 75:7.0:sm75 80:7.0: 120:8.7:; do
  IFS=: read -r arch max_isa features <<< "$tier"
  out="${INKRS_PTX_BUILD:-$root/target/ptx-build}/inkrs-kernels/sm$arch"
  mkdir -p "$out/ptx"
  rm -f "$out/ptx/inkrs_kernels.ptx"
  # cargo-oxide only writes PTX when the crate itself recompiles
  cargo "+$toolchain" clean -q --target-dir "$out/cargo" -p inkrs-kernels
  CARGO_TARGET_DIR="$out/cargo" CUDA_OXIDE_PTX_DIR="$out/ptx" cargo "+$toolchain" oxide build --arch "sm_$arch" \
    ${features:+--features "$features"}
  ptx="$out/ptx/inkrs_kernels.ptx"
  if [[ ! -s "$ptx" ]]; then
    echo "sm_$arch: cargo-oxide wrote no PTX at $ptx" >&2
    exit 1
  fi
  if ! grep -qx "\.target sm_$arch" "$ptx"; then
    echo "sm_$arch PTX has target '$(sed -n 's/^\.target //p' "$ptx")'" >&2
    exit 1
  fi
  version="$(sed -n 's/^\.version \([0-9]*\.[0-9]*\)$/\1/p' "$ptx")"
  if ! awk -v v="$version" -v m="$max_isa" 'BEGIN { exit !(v != "" && v + 0 <= m + 0) }'; then
    echo "sm_$arch PTX has ISA version '$version', above the $max_isa this tier ships at" >&2
    exit 1
  fi
  # the sm_75 fallbacks must replace every sm_80-only FP16 instruction; ptxas refuses them too, but this
  # names the cause (a kernel calling cuda_device's packed conversion or k16 mma, not common.rs's)
  if [[ "$arch" == 75 ]] && grep -q 'cvt\.rn\.f16x2\.f32\|mma\.sync\.aligned\.m16n8k16' "$ptx"; then
    echo "sm_75 PTX has sm_80-only FP16 instructions; use crate::common's pack_f16, narrow and mma_m16n8k16_f32_f16:" >&2
    grep -o 'cvt\.rn\.f16x2\.f32\|mma\.sync\.aligned\.m16n8k16[a-z0-9.]*' "$ptx" | sort | uniq -c >&2
    exit 1
  fi
  # a call left in the PTX is a body that wasn't inlined, whose pointers then arrive generic
  if grep -q 'call\.uni\|^\s*call\b' "$ptx"; then
    echo "sm_$arch PTX has calls; make the callee a plain #[inline(always)] fn:" >&2
    grep -o 'call\.uni [A-Za-z0-9_]*' "$ptx" | sort -u >&2
    exit 1
  fi
  # the libdevice functions the kernels call, from the IR before linking; every tier must link the same set,
  # since THIRD_PARTY_NOTICES names them and check-ptx.sh holds the notices to this list
  ir="$out/ptx/inkrs_kernels.ll"
  if [[ ! -s "$ir" ]]; then
    echo "sm_$arch: cargo-oxide wrote no IR at $ir" >&2
    exit 1
  fi
  grep -o 'call [^@]*@__nv_[A-Za-z0-9_]*' "$ir" | sed 's/.*@//' | LC_ALL=C sort -u > "$out/libdevice.txt"
  if [[ -z "${libdevice:-}" ]]; then
    libdevice="$out/libdevice.txt"
  elif ! diff "$libdevice" "$out/libdevice.txt" >&2; then
    echo "sm_$arch links other libdevice functions than the first tier" >&2
    exit 1
  fi

  cp "$ptx" "$root/ptx/inkrs.sm$arch.ptx"
  # registers, stack and spills per kernel, for the report; a tier ptxas refuses (an instruction its
  # target lacks) stops the build rather than shipping PTX the driver would reject at load
  if ! "$ptxas" -arch=sm_$arch -v "$ptx" -o /dev/null 2> "$root/ptx/ptxas-sm$arch.txt"; then
    cat "$root/ptx/ptxas-sm$arch.txt" >&2
    echo "ptxas refuses the sm_$arch PTX" >&2
    exit 1
  fi
done
cp "$libdevice" "$root/ptx/libdevice.txt"
"$root/scripts/check-ptx.sh" --write
