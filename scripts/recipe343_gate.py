"""Gate for scripts/recipe.py 343: inkrs's maps through the recipe's combine against the recipe's own runners' maps
through the same combine

    python -I scripts/recipe343_gate.py prep WORK DMIN DMAX [--json OUT]
    python -I scripts/recipe343_gate.py combine OURS_IMAGES REF_IMAGES NAME --depth D [--json OUT]
    python -I scripts/recipe343_gate.py selftest

`prep`: the helper's in-process Hecate resample (recipe.py HecatePrep) equals hecate_prep.py's output in
WORK/prep/<tag>-<zc>.zarr bit for bit, for every depth and both renders, and so does the .npy copy beside it that
recipe.py --hecate-inputs reads.

`combine`: ensemble_maps.py's outputs on both sides. Checks, every one counted, at bounds fixed before the first
run:
  - stats: the same models and ensemble members; per model, |mean| and |sd| differences within STAT_TOL relative
    (the pooled statistics over every depth; a 1/255 map step moves them by far less), and the ensemble's p1 and
    p99.5 within STAT_TOL relative to their span
  - each image (the ensemble and the four per-model images at depth D, reading side): same shape, both non-constant;
    r >= MIN_R, p99.9 |diff| <= MAX_P999 and max |diff| <= MAX_ABS grey levels, and the zero (invalid) pixels the
    same on both sides up to the stretch's clip (a zero on one side is at most MAX_P999 on the other: to8 also
    writes 0 for valid pixels at or below p1; this bound replaced exact zero equality after the first run, whose
    every lone zero was a 1 or 2 on the other side). The ensemble image amplifies a map's 1/255 step:
    (v - 64) / 128 doubles it, the z-score divides by an SD near 0.16 and the stretch spans about 4.6 z over 255
    levels, so one villa step is about 0.7 grey levels in one model's term and 1/4 of that in the mean;
    MAX_P999 = 4 holds the villa contract's own bound
Exit 0 when every check passes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MIN_R = 0.9994
MAX_P999 = 4.0
MAX_ABS = 16.0
STAT_TOL = 2e-3
STACKS = ("ensemble", "dnative", "hecate", "ink9-42", "ink9-43")


class Checks:
    def __init__(self):
        self.rows = []

    def check(self, name: str, ok: bool, detail="") -> None:
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail != "" else ""))

    @property
    def ok(self) -> bool:
        return bool(self.rows) and all(r["ok"] for r in self.rows)


def compare_images(c: Checks, tag: str, ours: np.ndarray, ref: np.ndarray) -> dict:
    c.check(f"{tag}: shape", ours.shape == ref.shape, f"{ours.shape} vs {ref.shape}")
    if ours.shape != ref.shape:
        return {}
    c.check(f"{tag}: reference not constant", ref.max() > ref.min())
    c.check(f"{tag}: ours not constant", ours.max() > ours.min())
    # ensemble_maps.py's to8 writes 0 for an invalid pixel and also for a valid one at or below the stretch's p1, so
    # a one-step map difference at the clip flips a valid pixel between 0 and 1 or 2; a zero is only a defect when
    # the other side holds more than a per-pixel step there (an invalid pixel the other side filled in)
    lone = (ours == 0) != (ref == 0)
    other = np.maximum(ours, ref)[lone]
    c.check(f"{tag}: zeros agree within {MAX_P999:g} levels", bool(np.all(other <= MAX_P999)),
            f"{int(np.count_nonzero(other > MAX_P999))} of {int(lone.sum())} lone zeros")
    a, b = ours.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    diff = np.abs(a - b)
    p999, mx = float(np.percentile(diff, 99.9)), float(diff.max())
    c.check(f"{tag}: r >= {MIN_R}", r >= MIN_R, round(r, 7))
    c.check(f"{tag}: p99.9 |diff| <= {MAX_P999}", p999 <= MAX_P999, p999)
    c.check(f"{tag}: max |diff| <= {MAX_ABS}", mx <= MAX_ABS, mx)
    return {"r": r, "p999": p999, "max": mx, "identical": float(np.mean(diff == 0)), "mean_diff": float(np.mean(a - b))}


def compare_stats(c: Checks, ours: dict, ref: dict, depths: list[int] | None = None) -> None:
    om, rm = ours["models"], ref["models"]
    c.check("stats: same models", sorted(om) == sorted(rm), f"{sorted(om)} vs {sorted(rm)}")
    if depths is not None:
        # ensemble_maps.py skips a missing map silently, so each side must have pooled every requested depth and
        # the same pixels, and the ensemble must hold all four models at every depth
        for side, st in (("ours", ours), ("ref", ref)):
            for k in STACKS[1:]:
                m = st["models"].get(k, {})
                c.check(f"stats {side} {k}: every depth", m.get("depths") == depths, m.get("depths"))
            e = st["ensemble"]
            c.check(f"stats {side} ensemble: four models", sorted(e.get("models", [])) == sorted(STACKS[1:]),
                    e.get("models"))
            c.check(f"stats {side} ensemble: every depth", e.get("depths") == depths, e.get("depths"))
        for k in sorted(set(om) & set(rm)):
            c.check(f"stats {k}: same pixel count", om[k].get("n_px") == rm[k].get("n_px") and om[k].get("n_px"),
                    f"{om[k].get('n_px')} vs {rm[k].get('n_px')}")
    for k in sorted(set(om) & set(rm)):
        for f in ("mean", "sd"):
            if f not in rm[k]:
                c.check(f"stats {k}: {f} present", False)
                continue
            rel = abs(om[k][f] - rm[k][f]) / max(abs(rm[k][f]), 1e-9)
            c.check(f"stats {k}: {f} within {STAT_TOL}", rel <= STAT_TOL, f"{om[k][f]:.6f} vs {rm[k][f]:.6f}")
    oe, re_ = ours["ensemble"], ref["ensemble"]
    c.check("stats: same ensemble members", oe.get("models") == re_.get("models"), f"{oe.get('models')}")
    span = re_["p99_5"] - re_["p1"]
    for f in ("p1", "p99_5"):
        c.check(f"stats ensemble: {f} within {STAT_TOL} of span", abs(oe[f] - re_[f]) <= STAT_TOL * span,
                f"{oe[f]:.5f} vs {re_[f]:.5f}")


def cmd_combine(a) -> int:
    import cv2

    c = Checks()
    res = {}
    so = json.loads((a.ours / f"{a.name}__stats.json").read_text())
    sr = json.loads((a.ref / f"{a.name}__stats.json").read_text())
    compare_stats(c, so, sr, parse(a.depths))
    for st in STACKS:
        f = f"{a.name}__{st}__d{a.depth:+03d}.png"
        po, pr = a.ours / f, a.ref / f
        if not (po.is_file() and pr.is_file()):
            c.check(f"{st}: both images exist", False, f)
            continue
        res[st] = compare_images(c, st, cv2.imread(str(po), cv2.IMREAD_UNCHANGED), cv2.imread(str(pr), cv2.IMREAD_UNCHANGED))
    if a.json:
        Path(a.json).write_text(json.dumps({"checks": c.rows, "images": res, "ok": c.ok}, indent=1))
    print("combine gate", "PASS" if c.ok else "FAIL")
    return 0 if c.ok else 1


def parse(text: str) -> list[int]:
    lo, _, hi = text.partition("..")
    return list(range(int(lo), int(hi or lo) + 1))


def cmd_inventory(a) -> int:
    """Every map the combine reads, on every side: four models, every depth, both renders, the reading side"""
    c = Checks()
    name = "concat_w047-w048_R5B2_z9500-11000"
    for root in a.roots:
        for tag, nm in (("plain", f"{name}_66"), ("padded", f"{name}_66_padded")):
            d = root / f"maps_{tag}"
            for dd in range(a.dmin, a.dmax + 1):
                want = [f"{nm}__{k}__L66s{25 + dd}_reverse.tif" for k in ("dnative", "hybrid_3d2d-seed42", "hybrid_3d2d-seed43")]
                want.append(f"{nm}__hecate__L66zc{32.5 + dd:.2f}_reverse.png")
                for f in want:
                    p = d / f
                    c.check(f"{root.name}/{tag}: {f}", p.is_file() and p.stat().st_size > 0)
    print("inventory", "PASS" if c.ok else "FAIL")
    return 0 if c.ok else 1


def cmd_prep(a) -> int:
    import zarr

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from recipe import N_LAYERS, HecatePrep

    c = Checks()
    name = "concat_w047-w048_R5B2_z9500-11000"
    for tag, z in (("plain", f"{name}_66.zarr"), ("padded", f"{name}_66_padded.zarr")):
        prep = HecatePrep(a.work / z)
        for d in range(a.dmin, a.dmax + 1):
            zc = f"{(N_LAYERS - 1) / 2 + d:.2f}"
            want = np.asarray(zarr.open(str(a.work / "prep" / f"{tag}-{zc}.zarr"), mode="r")["0"][:])
            got = prep.plane_stack(float(zc))
            c.check(f"prep {tag} zc {zc}", want.shape == got.shape and np.array_equal(want, got), f"{got.shape}")
            npy = a.work / "prep" / f"{tag}-{zc}.npy"
            copy = np.load(npy) if npy.is_file() else None
            c.check(f"prep {tag} zc {zc}: .npy copy", copy is not None and copy.dtype == want.dtype
                    and np.array_equal(copy, want), npy.name)
        del prep
    if a.json:
        Path(a.json).write_text(json.dumps({"checks": c.rows, "ok": c.ok}, indent=1))
    print("prep gate", "PASS" if c.ok else "FAIL")
    return 0 if c.ok else 1


def cmd_selftest(_a) -> int:
    """Each image check must fire on the defect it guards"""
    rng = np.random.default_rng(0)
    ref = rng.integers(1, 255, (64, 64)).astype(np.uint8)
    ref[:8] = 0
    cases = {
        "identical": (ref.copy(), True),
        "one level": (np.clip(ref.astype(int) + (rng.random(ref.shape) < 0.1), 0, 255).astype(np.uint8) * (ref > 0), True),
        "zero moved": (np.where(np.arange(64)[:, None] == 8, 0, ref).astype(np.uint8), False),
        "clip step": (np.where(ref <= 2, 0, ref).astype(np.uint8), True),
        "shifted": (np.roll(ref, 1, axis=1), False),
        "constant": (np.full_like(ref, 7) * (ref > 0), False),
        "tail": (np.where(rng.random(ref.shape) < 0.01, np.clip(ref.astype(int) + 40, 1, 255), ref).astype(np.uint8), False),
    }
    bad = []
    for k, (ours, want) in cases.items():
        c = Checks()
        compare_images(c, k, ours, ref)
        if c.ok != want:
            bad.append(k)
    s = {"models": {"m": {"mean": 0.2, "sd": 0.16}}, "ensemble": {"models": ["m"], "p1": -0.7, "p99_5": 3.9}}
    for k, mut, want in (("stats same", lambda t: t, True),
                         ("stats sd", lambda t: {**t, "models": {"m": {"mean": 0.2, "sd": 0.161}}}, False),
                         ("stats p99", lambda t: {**t, "ensemble": {**t["ensemble"], "p99_5": 3.95}}, False)):
        c = Checks()
        compare_stats(c, mut(json.loads(json.dumps(s))), s)
        if c.ok != want:
            bad.append(k)
    print("selftest", "FAIL " + ",".join(bad) if bad else "PASS")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("work", type=Path)
    p.add_argument("dmin", type=int)
    p.add_argument("dmax", type=int)
    p.add_argument("--json")
    p.set_defaults(fn=cmd_prep)
    q = sub.add_parser("combine")
    q.add_argument("ours", type=Path)
    q.add_argument("ref", type=Path)
    q.add_argument("name")
    q.add_argument("--depth", type=int, required=True)
    q.add_argument("--depths", required=True, help="the --stat-depths both sides ran, A..B")
    q.add_argument("--json")
    q.set_defaults(fn=cmd_combine)
    i = sub.add_parser("inventory")
    i.add_argument("roots", type=Path, nargs="+")
    i.add_argument("dmin", type=int)
    i.add_argument("dmax", type=int)
    i.set_defaults(fn=cmd_inventory)
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
