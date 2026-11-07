"""Proves the P1 gate can fail: one map of a passing run, mutated four ways, must fail p1_gate.py with
the check each mutation targets, while the unmutated map passes

    python -I scripts/p1_mutate.py OURS REF ZARR STRIDE WORK [--map NAME] [--json OUT]

Mutations, each a copy of that one map (and its coverage mask) under WORK/<mutation>/:
  zeroed-4x4             the brightest covered 4x4 block of the map set to 0
                         (r and p99.9 barely move; max |diff| must catch it)
  coverage-drop-1px      one covered pixel of the engine's coverage mask set to 0
  coverage-add-1px       one pixel villa leaves uncovered set to 1 in the engine's coverage mask
  uncovered-nonzero-1px  one pixel villa leaves uncovered set to 1 in the map
Each runs `p1_gate.py --only NAME`. Exit 0 when the control passes and every mutation fails with
its expected message
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
EXPECT = {
    "zeroed-4x4": "max |diff|",
    "coverage-drop-1px": "engine coverage differs from villa's on 1 pixels",
    "coverage-add-1px": "engine coverage differs from villa's on 1 pixels",
    "uncovered-nonzero-1px": "ours nonzero on 1 pixels villa leaves uncovered",
}


def write(path, image):
    # villa's writer arguments, so only the mutated pixels differ
    tifffile.imwrite(path, image, compression="lzw", tile=(128, 128), bigtiff=True, metadata=None,
                     software="vesuvius.ink_detection.inference.infer")


def gate(ours, a, name):
    cmd = [sys.executable, "-I", str(HERE / "p1_gate.py"), str(ours), str(a.ref), str(a.zarr), str(a.stride), "--only", name]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("zarr", type=Path)
    ap.add_argument("stride", type=int)
    ap.add_argument("work", type=Path)
    ap.add_argument("--map")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    name = a.map or sorted(p.name for p in (a.ours / "ink").glob("*.tif"))[0]
    image = tifffile.imread(a.ours / "ink" / name)
    cov = tifffile.imread(a.ours / "coverage" / name)
    covered = cov.astype(bool)
    results = {}

    rc, out = gate(a.ours, a, name)
    results["control"] = dict(rc=rc, ok=rc == 0, tail=out.strip().splitlines()[-3:])

    # the brightest 4x4 block wholly inside the covered area, on a 4-pixel grid
    h, w = image.shape
    blocks = image[: h // 4 * 4, : w // 4 * 4].reshape(h // 4, 4, w // 4, 4).min(axis=(1, 3)).astype(np.int32)
    inside = covered[: h // 4 * 4, : w // 4 * 4].reshape(h // 4, 4, w // 4, 4).all(axis=(1, 3))
    blocks[~inside] = -1
    by, bx = np.unravel_index(int(np.argmax(blocks)), blocks.shape)
    cy, cx = (int(v[0]) for v in np.nonzero(covered))
    uy, ux = (int(v[0]) for v in np.nonzero(~covered)) if (~covered).any() else (None, None)

    mutations = {"zeroed-4x4": ("map", lambda m: m.__setitem__((slice(4 * by, 4 * by + 4), slice(4 * bx, 4 * bx + 4)), 0)),
                 "coverage-drop-1px": ("cov", lambda m: m.__setitem__((cy, cx), 0))}
    if uy is not None:
        mutations["coverage-add-1px"] = ("cov", lambda m: m.__setitem__((uy, ux), 1))
        mutations["uncovered-nonzero-1px"] = ("map", lambda m: m.__setitem__((uy, ux), 1))
    for mut, (which, apply) in mutations.items():
        d = a.work / mut
        shutil.rmtree(d, ignore_errors=True)
        (d / "ink").mkdir(parents=True)
        (d / "coverage").mkdir()
        m_img, m_cov = image.copy(), cov.copy()
        apply(m_img if which == "map" else m_cov)
        write(d / "ink" / name, m_img)
        write(d / "coverage" / name, m_cov)
        rc, out = gate(d, a, name)
        fails = [line for line in out.splitlines() if line.startswith("FAIL")]
        hit = any(EXPECT[mut] in line for line in fails)
        results[mut] = dict(rc=rc, ok=rc != 0 and hit, expected=EXPECT[mut], failures=fails)
    if uy is None:
        results["note"] = "no uncovered pixel in this map; the two uncovered-pixel mutations were not run"
    results["map"] = name
    results["zeroed_block"] = dict(y0=4 * int(by), x0=4 * int(bx), min_value=int(blocks[by, bx]))
    ok = all(v["ok"] for k, v in results.items() if isinstance(v, dict) and "ok" in v)
    results["ok"] = ok
    text = json.dumps(results, indent=1)
    print(text)
    if a.json:
        a.json.write_text(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
