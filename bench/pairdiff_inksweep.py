"""Diff inksweep maps against villa maps, matched by depth and side (villa forward = inksweep blind)

    python -I pairdiff_inksweep.py VILLA_UNIT INKSWEEP_UNIT [--name crop343-2048] [--calibrated] [--json OUT]

Prints per map: depth, side, r, max |diff|, 99.9th percentile |diff|, share of equal pixels, nonzero mismatches.
Default: inksweep's raw maps (`ink/ink9_s42/raw/<side>/dNN.tif`, float probabilities) against villa's uint8 maps,
compared as floor(255p), villa's uint8 storage. --calibrated: inksweep's contract maps (`ink/ink9_s42/[blind/]dNN.tif`,
written when keep_raw is off) against the same reading applied to villa's map, clip((v - 64) / 128, 0, 1), both in
steps of 1/128; inksweep emulates villa's uint8 storage first, so its values must sit on that grid. With --json the
rows and a summary go to OUT. Exit 0 when every map meets r >= 0.9994, p99.9 <= 4 and max <= 4 steps (P1's bar)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

ap = argparse.ArgumentParser()
ap.add_argument("ref", type=Path)
ap.add_argument("isw", type=Path)
ap.add_argument("--name", default="crop343-2048")
ap.add_argument("--calibrated", action="store_true")
ap.add_argument("--json", type=Path)
a = ap.parse_args()
rows, ok = [], True
for d in range(-4, 5):
    s = 25 + d
    for side, suffix in (("blind", ""), ("reading", "_reverse")):
        v = tifffile.imread(a.ref / "ink" / f"{a.name}__hybrid_3d2d-seed42__L66s{s}{suffix}.tif")
        if a.calibrated:
            sub = "blind/" if side == "blind" else ""
            u = tifffile.imread(a.isw / "ink" / "ink9_s42" / f"{sub}d{d:+03d}.tif").astype(np.float64) * 128
            b = np.rint(u).astype(np.int16)
            off_grid = int((np.abs(u - b) > 1e-4).sum())
            a8 = np.clip(v.astype(np.int16) - 64, 0, 128)
        else:
            p = tifffile.imread(a.isw / "ink" / "ink9_s42" / "raw" / side / f"d{d:+03d}.tif")
            b = np.floor(p.astype(np.float64) * 255).astype(np.int16)
            off_grid = 0
            a8 = v.astype(np.int16)
        if a8.shape != b.shape:
            rows.append(dict(depth=d, side=side, shape=[list(a8.shape), list(b.shape)]))
            ok = False
            print(d, side, "shape", a8.shape, b.shape)
            continue
        diff = np.abs(a8 - b)
        r = float(np.corrcoef(a8.ravel(), b.ravel())[0, 1])
        row = dict(depth=d, side=side, r=r, max=int(diff.max()), p999=float(np.percentile(diff, 99.9)), equal_share=float((diff == 0).mean()),
                   nonzero_mismatch=int(((a8 > 0) != (b > 0)).sum()), off_grid=off_grid)
        ok &= r >= 0.9994 and row["p999"] <= 4 and row["max"] <= 4 and off_grid == 0
        rows.append(row)
        print(d, side, round(r, 6), row["max"], row["p999"], round(row["equal_share"], 4), row["nonzero_mismatch"], *(["off-grid", off_grid] if off_grid else []))
good = [r for r in rows if "r" in r]
summary = dict(mode="calibrated" if a.calibrated else "raw", maps=len(good), ok=bool(ok and len(good) == 18), min_r=min((r["r"] for r in good), default=None),
               max_abs=max((r["max"] for r in good), default=None), max_p999=max((r["p999"] for r in good), default=None),
               bar=dict(min_r=0.9994, max_p999=4, max_abs=4))
print(json.dumps(summary))
if a.json:
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(dict(summary=summary, maps=rows), indent=1))
sys.exit(0 if summary["ok"] else 1)
