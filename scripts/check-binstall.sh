#!/usr/bin/env bash
# Renders Cargo.toml's [package.metadata.binstall] for each target the release ships a CLI archive for, filling in
# the template as cargo-binstall does, and prints each target's download URL and binary path. Given a directory of
# the archives package.yml builds, it also checks that each target's archive is there under the rendered name and
# holds the binary at the rendered path, and that the directory holds nothing else
#
#   scripts/check-binstall.sh [ARCHIVE_DIR]
#
# Needs cargo, jq, tar and unzip
set -euo pipefail

# the targets package.yml builds; cargo-binstall falls back to a source build anywhere else
targets=(x86_64-unknown-linux-gnu aarch64-apple-darwin x86_64-pc-windows-msvc)

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="${1:-}"

package="$(cargo metadata --no-deps --format-version 1 --locked --manifest-path "$root/Cargo.toml" \
  | jq -c '.packages[] | select(.name == "inkrs")')"
name="$(jq -r .name <<< "$package")"
version="$(jq -r .version <<< "$package")"
repo="$(jq -r '.repository | sub("\\.git$"; "")' <<< "$package")"
bin="$(jq -r '[.targets[] | select(.kind[] == "bin") | .name] | if length == 1 then .[0] else error("expected one binary") end' \
  <<< "$package")"

# a target's override of a field, else the field itself
field() {
  jq -r --arg target "$1" --arg key "$2" \
    '.metadata.binstall | (.overrides[$target][$key] // .[$key]) // error("no \($key) for \($target)")' <<< "$package"
}

# cargo-binstall's template variables; leon accepts spaces inside the braces or none
render() {
  local out
  out="$(sed -E \
    -e "s|\{ *name *\}|$name|g" \
    -e "s|\{ *version *\}|$version|g" \
    -e "s|\{ *target *\}|$target|g" \
    -e "s|\{ *archive-suffix *\}|$suffix|g" \
    -e "s|\{ *binary-ext *\}|$ext|g" \
    -e "s|\{ *bin *\}|$bin|g" \
    -e "s|\{ *repo *\}|$repo|g" <<< "$1")"
  if [[ "$out" == *"{"* ]]; then
    echo "$out uses a variable this script doesn't fill in" >&2
    exit 1
  fi
  printf '%s\n' "$out"
}

expected=()
for target in "${targets[@]}"; do
  fmt="$(field "$target" pkg-fmt)"
  case "$fmt" in
    tgz) suffix=.tar.gz ;;
    zip) suffix=.zip ;;
    *) echo "$target: package.yml builds only tgz and zip archives, not $fmt" >&2; exit 1 ;;
  esac
  ext=""
  if [[ "$target" == *-windows-* ]]; then ext=.exe; fi

  url="$(render "$(field "$target" pkg-url)")"
  path="$(render "$(field "$target" bin-dir)")"
  archive="${url##*/}"
  echo "$target: $url -> $path"
  expected+=("$archive")

  if [ -z "$dir" ]; then continue; fi

  if ! [ -f "$dir/$archive" ]; then
    echo "$dir has no $archive" >&2
    exit 1
  fi

  case "$fmt" in
    tgz) entries="$(tar -tzf "$dir/$archive")" ;;
    zip) entries="$(unzip -Z1 "$dir/$archive")" ;;
  esac
  if ! grep -Fxq "$path" <<< "$entries"; then
    echo "$archive has no $path; it holds:" >&2
    echo "$entries" >&2
    exit 1
  fi
done

if [ -n "$dir" ]; then
  found="$(find "$dir" -type f -exec basename {} \; | sort)"
  want="$(printf '%s\n' "${expected[@]}" | sort)"
  if [ "$found" != "$want" ]; then
    echo "$dir holds"$'\n'"$found"$'\n'"but the binstall metadata names"$'\n'"$want" >&2
    exit 1
  fi
  echo "every archive matches the binstall metadata"
fi
