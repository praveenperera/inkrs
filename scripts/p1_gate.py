"""P1 gate: inkrs's maps against villa's, per map

    python -I scripts/p1_gate.py OURS REF ZARR STRIDE [--report OURS/inkrs.json] [--json OUT] [--only MAP]
        [--max-abs 4] [--patch 384]

OURS and REF are run directories with maps under `ink/` (mapdiff.py's layout); OURS also holds the
engine's own coverage masks under `coverage/` (`inkrs sweep --coverage`). villa's coverage comes
from villa's own scheduling code run on the CPU (scripts/villa_coverage.py: its occupancy-level
skip, iter_blocks, raw-window skip and ChunkAccumulator with the finish rule weight > 1e-6), not
from the maps. Checks per map, every one counted:
  - the same map set on both sides, same shapes and dtypes
  - r >= 0.9994, p99.9 |diff| <= 4 and max |diff| <= MAX_ABS (uint8 steps); the max bound is
    villa's own compiled-vs-eager spread: at most 4/255 on a full PHerc. 343 map
    (inksweep docs/validation.md), 2/255 measured on S1
  - the engine's coverage mask exists, holds only 0 and 1, and equals villa's bit for bit
  - both maps are 0 on every pixel villa leaves uncovered (villa's value there)
  - with --report, inkrs's kept-tile count per map (first pass) equals villa's
--only restricts both sides to one map name (for the mutation tests). Exit 0 when every check
passes
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from villa_coverage import Input  # noqa: E402

MIN_R = 0.9994
MAX_P999 = 4.0
MAX_ABS = 4.0
NAME = re.compile(r"__L(\d+)s(\d+)(_reverse)?\.tif$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("zarr", type=Path)
    ap.add_argument("stride", type=int)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--only")
    ap.add_argument("--max-abs", type=float, default=MAX_ABS)
    ap.add_argument("--patch", type=int, default=128, choices=(128, 384), help="the reader's tile edge (384: Reader V3)")
    a = ap.parse_args()
    inp = Input(a.zarr, a.patch)
    ours = {p.name: p for p in sorted((a.ours / "ink").glob("*.tif"))}
    ref = {p.name: p for p in sorted((a.ref / "ink").glob("*.tif"))}
    if a.only:
        ours = {k: v for k, v in ours.items() if k == a.only}
        ref = {k: v for k, v in ref.items() if k == a.only}
    reported = {}
    if a.report:
        reported = {Path(p).name: n for p, n in json.loads(a.report.read_text())["passes"][0]["map_tiles"]}
    checks = passed = 0
    failures = []

    def check(ok, what):
        nonlocal checks, passed
        checks += 1
        passed += bool(ok)
        if not ok:
            failures.append(what)
        return ok

    check(bool(ref) and set(ours) == set(ref), f"map sets differ: ours-only {sorted(set(ours) - set(ref))[:4]}, ref-only {sorted(set(ref) - set(ours))[:4]}")
    rows = []
    for name in sorted(set(ours) & set(ref)):
        m = NAME.search(name)
        if not check(m and int(m[1]) == inp.depth, f"{name}: no L{inp.depth}s<S> in the name"):
            continue
        s, reverse = int(m[2]), bool(m[3])
        a8, b8 = tifffile.imread(ours[name]), tifffile.imread(ref[name])
        shape = (inp.height, inp.width)
        if not check(a8.shape == b8.shape == shape and a8.dtype == b8.dtype == np.uint8, f"{name}: shapes {a8.shape} {b8.shape} dtypes {a8.dtype} {b8.dtype}"):
            continue
        tiles, villa_cov = inp.map(s, reverse, a.stride)
        cov_path = a.ours / "coverage" / name
        eng = tifffile.imread(cov_path) if cov_path.exists() else None
        cov_ok = check(eng is not None and eng.shape == shape and eng.dtype == np.uint8 and set(np.unique(eng).tolist()) <= {0, 1},
                       f"{name}: engine coverage {cov_path} missing or not a 0/1 uint8 {shape} mask")
        mismatched = int((eng.astype(bool) != villa_cov).sum()) if cov_ok else -1
        x, y = a8.astype(np.float64).ravel(), b8.astype(np.float64).ravel()
        d = np.abs(x - y)
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float(np.array_equal(x, y))
        p999 = float(np.quantile(d, 0.999))
        un = ~villa_cov
        row = dict(map=name, r=r, p999=p999, max=float(d.max()), equal_share=float((d == 0).mean()), covered=int(villa_cov.sum()),
                   engine_covered=int(eng.astype(bool).sum()) if cov_ok else None, coverage_mismatch=mismatched, kept_tiles=len(tiles),
                   ours_nonzero_uncovered=int((a8[un] != 0).sum()), ref_nonzero_uncovered=int((b8[un] != 0).sum()))
        check(r >= MIN_R, f"{name}: r {r:.6f} < {MIN_R}")
        check(p999 <= MAX_P999, f"{name}: p99.9 |diff| {p999} > {MAX_P999}")
        check(row["max"] <= a.max_abs, f"{name}: max |diff| {row['max']:.0f} > {a.max_abs:.0f}")
        if cov_ok:
            check(mismatched == 0, f"{name}: engine coverage differs from villa's on {mismatched} pixels")
        check(row["ours_nonzero_uncovered"] == 0, f"{name}: ours nonzero on {row['ours_nonzero_uncovered']} pixels villa leaves uncovered")
        check(row["ref_nonzero_uncovered"] == 0, f"{name}: villa nonzero on {row['ref_nonzero_uncovered']} pixels its own schedule leaves uncovered")
        if reported:
            row["reported_tiles"] = reported.get(name)
            check(reported.get(name) == len(tiles), f"{name}: inkrs kept {reported.get(name)} tiles, villa {len(tiles)}")
        rows.append(row)
        print(f"{name:<52} r {r:.7f} p99.9 {p999:.0f} max {d.max():.0f} equal {row['equal_share']:.4f} tiles {len(tiles)} "
              f"coverage {row['covered']} mismatch {mismatched} uncovered-nonzero {row['ours_nonzero_uncovered']}/{row['ref_nonzero_uncovered']}", flush=True)
    for f in failures:
        print("FAIL", f)
    summary = dict(maps=len(rows), checks=checks, passed=passed, failed=checks - passed, min_r=min((r["r"] for r in rows), default=None),
                   max_p999=max((r["p999"] for r in rows), default=None), max_abs=max((r["max"] for r in rows), default=None),
                   occupancy_level=inp.level, occupancy_scale=list(inp.scale), bar=dict(min_r=MIN_R, max_p999=MAX_P999, max_abs=a.max_abs))
    print(json.dumps(summary))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(dict(summary=summary, maps=rows, failures=failures), indent=1))
    sys.exit(0 if checks == passed and rows else 1)


if __name__ == "__main__":
    main()
