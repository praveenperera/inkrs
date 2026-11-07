"""Compare the map TIFFs of two inksweep run directories, or fingerprint one

    python -I mapdiff.py A_RUN B_RUN [--exact | --min-r R [--max-abs X]] [--json OUT]
    python -I mapdiff.py --fingerprint RUN

Every `ink/**/*.tif` (contract maps and raw maps) is read as pixels, not file bytes, so TIFF metadata never counts.
The two runs must hold the same set of maps with the same shapes; a missing, extra or reshaped map fails, and so
does an empty run. `--exact` fails unless every pixel of every map is identical (NaN equal to NaN); `--min-r` fails
when any map's Pearson r is below R or, with `--max-abs`, any pixel differs by more than X. Exit 0 on pass, 1 on fail
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tifffile


def maps(run: Path) -> dict[str, Path]:
    found = {str(p.relative_to(run)): p for p in sorted((run / "ink").rglob("*.tif"))}
    if not found:
        raise SystemExit(f"{run}: no maps under ink/")
    return found


def fingerprint(run: Path) -> str:
    h = hashlib.sha256()
    for rel, p in maps(run).items():
        a = np.ascontiguousarray(tifffile.imread(p))
        h.update(rel.encode())
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.data)
    return h.hexdigest()


def compare(a_run: Path, b_run: Path) -> list[dict]:
    a_maps, b_maps = maps(a_run), maps(b_run)
    if set(a_maps) != set(b_maps):
        only_a = sorted(set(a_maps) - set(b_maps))[:5]
        only_b = sorted(set(b_maps) - set(a_maps))[:5]
        raise SystemExit(f"map sets differ: only in A {only_a}, only in B {only_b}")
    rows = []
    for rel in a_maps:
        a = tifffile.imread(a_maps[rel]).astype(np.float64)
        b = tifffile.imread(b_maps[rel]).astype(np.float64)
        if a.shape != b.shape:
            raise SystemExit(f"{rel}: shape {a.shape} vs {b.shape}")
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        d = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
        fa, fb = np.nan_to_num(a).ravel(), np.nan_to_num(b).ravel()
        if fa.std() > 0 and fb.std() > 0:
            r = float(np.corrcoef(fa, fb)[0, 1])
        else:
            r = 1.0 if np.array_equal(fa, fb) else 0.0
        rows.append(dict(map=rel, identical=bool(same.all()), equal_share=float(same.mean()), max_abs=float(d.max()),
                         r=r))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--fingerprint", action="store_true")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--min-r", type=float, default=None)
    ap.add_argument("--max-abs", type=float, default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    if a.fingerprint:
        if len(a.runs) != 1:
            raise SystemExit("--fingerprint takes one run")
        print(fingerprint(a.runs[0]))
        return
    if len(a.runs) != 2:
        raise SystemExit("give two runs")
    rows = compare(*a.runs)
    summary = dict(maps=len(rows), identical=sum(r["identical"] for r in rows),
                   min_r=min(r["r"] for r in rows), max_abs=max(r["max_abs"] for r in rows),
                   min_equal_share=min(r["equal_share"] for r in rows))
    ok = True
    if a.exact:
        ok = summary["identical"] == summary["maps"]
    elif a.min_r is not None:
        ok = summary["min_r"] >= a.min_r and (a.max_abs is None or summary["max_abs"] <= a.max_abs)
    summary["pass"] = ok
    out = dict(a=str(a.runs[0]), b=str(a.runs[1]), summary=summary, maps=rows)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=1))
    print(json.dumps(summary))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
