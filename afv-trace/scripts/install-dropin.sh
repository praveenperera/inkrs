#!/usr/bin/env bash
# Make VENV's avf-spline-generator the native tracer, keeping the reference script beside it
#
# usage: install-dropin.sh VENV [BINARY]
#        install-dropin.sh --restore VENV
# BINARY defaults to the inkrs-afv-trace beside this script's directory (the release archive's layout), then to
# the one on PATH. The reference entry script moves to
# VENV/bin/avf-spline-generator.reference; the native binary is copied to VENV/bin/inkrs-afv-trace and
# avf-spline-generator becomes a symlink to it; NOTICE and licenses/ go to VENV/share/inkrs-afv-trace. The
# binary finds numpy's OpenBLAS through its own path (VENV/bin/.. /lib/python*/site-packages/numpy.libs, or the
# scipy-openblas64 package's lib/ on macOS, where numpy links Accelerate instead), so
# callers that run Path(sys.executable).parent / "avf-spline-generator" need no other change
set -euo pipefail

if [[ ${1:-} == --restore ]]; then
  venv=${2:?usage: install-dropin.sh --restore VENV}
  ref=$venv/bin/avf-spline-generator.reference
  [[ -e $ref || -L $ref ]] || { echo "no $ref to restore" >&2; exit 1; }
  rm -f "$venv/bin/avf-spline-generator"
  mv "$ref" "$venv/bin/avf-spline-generator"
  rm -f "$venv/bin/inkrs-afv-trace"
  rm -rf "$venv/share/inkrs-afv-trace"
  echo "restored the reference avf-spline-generator in $venv"
  exit 0
fi

venv=${1:?usage: install-dropin.sh VENV [BINARY]}
crate=$(cd "$(dirname "$0")/.." && pwd)
bin=${2:-}
if [[ -z $bin ]]; then
  bin=$crate/inkrs-afv-trace
  [[ -x $bin ]] || bin=$(command -v inkrs-afv-trace || true)
  [[ -n $bin ]] || { echo "no inkrs-afv-trace beside $0 or on PATH; pass BINARY" >&2; exit 1; }
fi
entry=$venv/bin/avf-spline-generator
[[ -x $bin ]] || { echo "$bin is not an executable" >&2; exit 1; }
[[ -d $venv/bin ]] || { echo "$venv is not a venv" >&2; exit 1; }
# any one match will do; numpy.libs holds it on Linux, scipy_openblas64/lib on macOS
openblas=$(compgen -G "$venv/lib/python*/site-packages/numpy.libs/libscipy_openblas64_*.so" ||
  compgen -G "$venv/lib/python*/site-packages/scipy_openblas64/lib/libscipy_openblas64_*.dylib" ||
  compgen -G "$venv/lib/python*/site-packages/scipy_openblas64/lib/libscipy_openblas64_*.so" || true)
[[ -n $openblas ]] ||
  { echo "$venv has no numpy OpenBLAS (numpy.libs/libscipy_openblas64_*.so); on macOS, install it with" \
      "pip install scipy-openblas64==0.3.31.188.0" >&2; exit 1; }

# keep the reference entry once, whether a script or a symlink (some installers link it); a second install
# finds the entry already pointing at the native binary and only replaces the binary
if [[ -e $entry.reference || -L $entry.reference ]]; then
  :
elif [[ -L $entry && $(readlink "$entry") == inkrs-afv-trace ]]; then
  echo "$entry already points at inkrs-afv-trace but $entry.reference is missing" >&2
  exit 1
elif [[ -e $entry || -L $entry ]]; then
  mv "$entry" "$entry.reference"
fi
# the wheel already installs the binary there; copying a file onto itself would fail
if ! [[ -e $venv/bin/inkrs-afv-trace && $bin -ef $venv/bin/inkrs-afv-trace ]]; then
  install -m 0755 "$bin" "$venv/bin/inkrs-afv-trace"
fi
# the notices travel with the binary: the licences of the ported code and the statically linked libraries
install -d "$venv/share/inkrs-afv-trace/licenses"
install -m 0644 "$crate/NOTICE" "$venv/share/inkrs-afv-trace/"
install -m 0644 "$crate"/licenses/*.txt "$venv/share/inkrs-afv-trace/licenses/"
ln -sfn inkrs-afv-trace "$entry"
echo "$entry -> inkrs-afv-trace ($("$entry" --version)); reference kept as $entry.reference"
