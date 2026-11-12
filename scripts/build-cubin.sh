#!/usr/bin/env bash
# Compiles the committed PTX in ptx/ to SASS (cubins) for the GPU families inkrs ships compiled, so the
# first run on a machine loads its kernels instead of waiting on the driver's JIT. Each cubin comes
# from the tier src/gpu.rs loads on that family: sm75 for Turing, sm80 for every GPU from 8.0 up to
# 12.0, and sm120 for RTX 50. A GPU with no cubin here, or a driver that refuses one, still JITs the
# tier's PTX. ptxas is deterministic, so rerunning this on the same PTX and ptxas writes the same
# files. Records the ptxas version in ptx/ptxas.txt and refreshes the manifest. Needs the CUDA
# toolkit's ptxas only, not cuda-oxide; build-ptx.sh runs it after writing new PTX
#
#   scripts/build-cubin.sh
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
ptxas="${PTXAS:-$(command -v ptxas || echo "${CUDA_HOME:-/usr/local/cuda}/bin/ptxas")}"
if [[ ! -x "$ptxas" ]]; then
  echo "no executable ptxas at $ptxas; install the CUDA toolkit or set PTXAS" >&2
  exit 1
fi

# family:tier, the same table as src/gpu.rs's CUBINS. 8.7 runs the sm_86 cubin and 10.3 the sm_100
# one (SASS runs on later minor versions of its major); 11.x has none and JITs
families=(75:sm75 80:sm80 86:sm80 89:sm80 90:sm80 100:sm80 120:sm120)

rm -f "$root"/ptx/*.cubin
for family in "${families[@]}"; do
  IFS=: read -r arch tier <<< "$family"
  "$ptxas" -arch="sm_$arch" "$root/ptx/inkrs.$tier.ptx" -o "$root/ptx/inkrs.$tier.sm_$arch.cubin"
done
"$ptxas" --version | tail -1 > "$root/ptx/ptxas.txt"
"$root/scripts/check-ptx.sh" --write
