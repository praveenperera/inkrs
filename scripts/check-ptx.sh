#!/usr/bin/env bash
# Fails when the committed PTX is stale: ptx/sources.sha256 records the hashes of the kernel crate's
# sources, manifest, lockfile and toolchain pin, the build script, the tool versions it built with
# (ptx/tools.txt), the libdevice functions the kernels link (ptx/libdevice.txt) and the PTX it produced,
# one file per tier src/gpu.rs embeds. Also fails when THIRD_PARTY_NOTICES names other libdevice
# functions than that list. Needs no CUDA toolkit or cuda-oxide
#
#   scripts/check-ptx.sh            # check
#   scripts/check-ptx.sh --write    # record the current files (build-ptx.sh does this)
set -euo pipefail

cd "$(dirname "$0")/.."
manifest=ptx/sources.sha256
crate=inkrs-kernels
mapfile -t sources < <(find "$crate/src" -name '*.rs' | LC_ALL=C sort)
files=("${sources[@]}" "$crate/Cargo.toml" "$crate/Cargo.lock" "$crate/rust-toolchain.toml"
  scripts/build-ptx.sh ptx/tools.txt ptx/libdevice.txt ptx/*.ptx)

# the libdevice functions THIRD_PARTY_NOTICES names are exactly the ones build-ptx.sh found linked, so the
# notice covers what the PTX ships
if ! diff <(sed -n '/^3\. NVIDIA libdevice/,/^=====/p' THIRD_PARTY_NOTICES | grep -o '__nv_[A-Za-z0-9_]*' | LC_ALL=C sort -u) \
  ptx/libdevice.txt; then
  echo "THIRD_PARTY_NOTICES section 3 and ptx/libdevice.txt name different libdevice functions" >&2
  exit 1
fi

# the PTX files on disk are exactly the tiers src/gpu.rs embeds, so a tier can't be built but not shipped
# or shipped from a stale file the manifest no longer covers
if ! diff <(printf '%s\n' ptx/*.ptx | LC_ALL=C sort) \
  <(grep -o 'include_str!("\.\./ptx/[^"]*")' src/gpu.rs | sed 's|include_str!("\.\./\(.*\)")|\1|' | LC_ALL=C sort); then
  echo "ptx/*.ptx and the PTX src/gpu.rs embeds differ" >&2
  exit 1
fi

if [[ "${1:-}" == "--write" ]]; then
  sha256sum "${files[@]}" > "$manifest"
  exit 0
fi

if ! diff <(sha256sum "${files[@]}") "$manifest"; then
  echo "ptx is stale: run scripts/build-ptx.sh" >&2
  exit 1
fi
echo "ptx matches its sources"
