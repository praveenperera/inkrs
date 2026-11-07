"""Hecate map gate: an inkrs PNG against hecate.py's PNG of the same volume, direction and precision

    python -I scripts/hecate_mapgate.py OURS.png REF.png --mode fp32|bf16 --direction fwd|rev
        [--spread A.png B.png] [--json OUT]
    python -I scripts/hecate_mapgate.py selftest

Reads pixels, not file bytes (the deflate streams differ). Checks, every one counted:
  - both 8-bit grayscale, the same shape
  - each map's JSON (written next to it by both tools) matches the other's field for field, records
    --mode's precision and --direction's side, and its shape_yx is the PNG's
  - every META_KEYS field is present in both, so two maps missing the same field do not match
  - the reference is a real map: not constant, with ink (nonzero pixels) and support; ours has nonzero
    pixels too, so two empty or two flat maps do not pass as a match
  - the similarity rule (similarity(), reported as stats["similarity"]): r >= MIN_R; or, where the
    reference's variance is below LOW_VARIANCE.below (uint8 steps squared: a near-empty map, such as the
    edge fixture's, where one pixel off by a step costs about 1.8e-4 of r), at most
    max(LOW_VARIANCE.min_off, ceil(LOW_VARIANCE.off_fraction * pixels)) pixels differ, each by at most
    LOW_VARIANCE.max_step. Added in v0.2, after the edge fixture's reverse map missed r on rented GPUs with
    every pixel within one step, and fixed before those maps were scored under it
  - p99.9 |diff| <= MAX_P999, max |diff| <= MAX_ABS and |mean(ours - ref)| <= MAX_BIAS in
    uint8 steps, at the bounds LIMITS fixes per mode before any run: fp32 runs PyTorch's TF32
    convolutions, whose 10-bit mantissa leaves a few steps of difference; bf16 autocast's 8-bit one
    leaves more. Rounding noise is unbiased, so the mean bounds catch a small uniform scaling (a blend
    or head defect) that r and the tail bounds miss
  - the zero pixels agree: a pixel is 0 in one map exactly when it is 0 in the other, except where
    the other is 1 (a rounding step), so a support or tile-skip error cannot hide in a high r
--spread A B reports hecate.py's own spread between two of its runs (for example bf16 at batch 1
and 8) beside the result. It never changes the bounds. Exit 0 when every check passes
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Literal, NamedTuple

import cv2
import numpy as np

# fixed before the first run; a port outside them fails even if hecate.py's own runs differ as much
LIMITS = {
    "fp32": dict(min_r=0.9994, max_p999=4.0, max_abs=8.0, max_bias=0.1),
    "bf16": dict(min_r=0.999, max_p999=6.0, max_abs=32.0, max_bias=0.25),
}


class LowVariance(NamedTuple):
    """The pixel-count bar that replaces r on a near-constant reference"""

    below: float  # the reference's variance, in uint8 steps squared, under which this bar applies
    min_off: int  # pixels allowed to differ on any map
    off_fraction: float  # ... or this share of the pixels, rounded up, when that is more
    max_step: int  # how far each differing pixel may be off, in uint8 steps

    def allowed(self, pixels: int) -> int:
        return max(self.min_off, math.ceil(self.off_fraction * pixels))


# both modes
LOW_VARIANCE = LowVariance(below=1.0, min_off=5, off_fraction=1e-4, max_step=1)


class Similarity(NamedTuple):
    """Which similarity bar a map pair met or missed, with the numbers either one reads"""

    rule: Literal["r", "pixels_off"]
    ref_variance: float  # uint8 steps squared
    r: float
    min_r: float
    pixels_off: int
    allowed_off: int  # under rule "pixels_off"; recorded under "r" too
    max_off_step: int


def similarity(ours, ref, min_r) -> Similarity:
    d = np.abs(ours.astype(np.int16) - ref.astype(np.int16))
    var = float(ref.astype(np.float64).var())
    return Similarity(
        rule="pixels_off" if var < LOW_VARIANCE.below else "r",
        ref_variance=var,
        r=stats(ours, ref)["r"],
        min_r=min_r,
        pixels_off=int((d > 0).sum()),
        allowed_off=LOW_VARIANCE.allowed(ref.size),
        max_off_step=int(d.max()),
    )


META_KEYS = ["checkpoint", "sampling_um", "shape_yx", "reverse_z", "central_depth", "precision", "stride",
             "probabilities"]


def read(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.dtype != np.uint8 or img.ndim != 2:
        raise SystemExit(f"{path}: not an 8-bit grayscale PNG")
    meta_path = Path(path).with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return img, meta


def stats(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).ravel()
    fa, fb = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    r = float(np.corrcoef(fa, fb)[0, 1]) if fa.std() > 0 and fb.std() > 0 else float(np.array_equal(fa, fb))
    hist = np.bincount(np.minimum(d, 9), minlength=10)
    bias = float((a.astype(np.float64) - b.astype(np.float64)).mean())
    return dict(r=r, p999=float(np.quantile(d, 0.999)), max=int(d.max()), mean=float(d.mean()), bias=bias,
                identical=float((d == 0).mean()), hist_0_to_9plus=hist.tolist())


def gate(ours, ours_meta, ref, ref_meta, mode, reverse):
    """Every check, as (checks, passed, failures, stats)"""
    checks = passed = 0
    failures = []

    def check(ok, what):
        nonlocal checks, passed
        checks += 1
        passed += bool(ok)
        if not ok:
            failures.append(what)

    check(ours_meta is not None and ref_meta is not None, "a map has no JSON next to it")
    if ours_meta is not None and ref_meta is not None:
        for k in META_KEYS:
            check(k in ours_meta and k in ref_meta, f"JSON {k} missing: ours {k in ours_meta}, ref {k in ref_meta}")
            check(ours_meta.get(k) == ref_meta.get(k), f"JSON {k}: {ours_meta.get(k)!r} vs {ref_meta.get(k)!r}")
        check(ref_meta.get("precision") == mode, f"JSON precision {ref_meta.get('precision')!r}, gate mode {mode}")
        check(ref_meta.get("reverse_z") is reverse, f"JSON reverse_z {ref_meta.get('reverse_z')!r}, expected {reverse}")
        check(ref_meta.get("shape_yx") == list(ref.shape), f"JSON shape_yx {ref_meta.get('shape_yx')} vs PNG {ref.shape}")
    check(ours.shape == ref.shape, f"shape {ours.shape} vs {ref.shape}")
    if ours.shape != ref.shape:
        return checks, passed, failures, None
    check(ref.std() > 0 and (ref > 0).any(), "reference map is constant or empty")
    check((ours > 0).any(), "our map is all zero")
    s = stats(ours, ref)
    lim = LIMITS[mode]
    sim = similarity(ours, ref, lim["min_r"])
    s["similarity"] = sim._asdict()
    if sim.rule == "r":
        check(sim.r >= sim.min_r, f"r {sim.r:.6f} < {sim.min_r}")
    else:
        check(sim.pixels_off <= sim.allowed_off,
              f"pixels off {sim.pixels_off} > {sim.allowed_off} allowed (reference variance {sim.ref_variance:.4f})")
        check(sim.max_off_step <= LOW_VARIANCE.max_step,
              f"pixel off by {sim.max_off_step} > {LOW_VARIANCE.max_step} step (reference variance {sim.ref_variance:.4f})")
    check(s["p999"] <= lim["max_p999"], f"p99.9 {s['p999']} > {lim['max_p999']}")
    check(s["max"] <= lim["max_abs"], f"max {s['max']} > {lim['max_abs']}")
    check(abs(s["bias"]) <= lim["max_bias"], f"bias {s['bias']:+.4f} steps, bound {lim['max_bias']}")
    zero_a, zero_b = ours == 0, ref == 0
    bad_zero = int(((zero_a & (ref > 1)) | (zero_b & (ours > 1))).sum())
    check(bad_zero == 0, f"zero mismatch: {bad_zero} pixels are 0 on one side and above 1 on the other")
    s["zero_mismatch"] = bad_zero
    # recorded, not gated: a 0 against a 1 is a probability rounding either side of 0.5/255, so reports
    # must not read the gated count above as "no pixel is zero on one side only"
    s["zero_one_side"] = int((zero_a != zero_b).sum())
    return checks, passed, failures, s


def low_variance_cases(meta) -> bool:
    """The near-constant bar, on maps like the edge fixture's (mostly 0, a few pixels at 1 to 5): each case is
    (ours, reference, mode, the rule the gate must pick, the failure it must report or None for a pass)"""

    def edge(h, w, seed, hot, top):
        rng = np.random.default_rng(seed)
        out = np.zeros((h, w), np.uint8)
        at = rng.choice(out.size, hot, replace=False)
        out.flat[at] = rng.integers(2, top + 1, hot)
        return out

    def off(ref, n, step=1):
        # +-step on n of the nonzero pixels, alternating, so no zero and no mean moves
        out = ref.astype(int)
        hot = np.flatnonzero(ref >= 2)[:n]
        assert hot.size == n
        out.flat[hot] += np.where(np.arange(n) % 2 == 0, step, -step)
        return out.astype(np.uint8)

    # 100 px; and 139,968 px, the edge fixture's size, 0.5% of it ink up to 5 steps as there
    small, crop = edge(10, 10, 1, 8, 3), edge(486, 288, 2, 700, 5)
    # 100 px just above the variance bar: 5 off by a step pass the pixel bar but fail r, so r must be the one used
    hivar = np.zeros((10, 10), np.uint8)
    hivar.flat[:30] = np.tile([2, 3, 4, 5, 6], 6)
    # variance exactly 1 step squared (half 0, half 2) is not below the bar, so r applies
    at_bar = np.zeros((10, 10), np.uint8)
    at_bar.flat[::2] = 2
    cases = {
        "6 of 100 px off by 1": (off(small, 6), small, "fp32", "pixels_off", "pixels off 6 > 5"),
        "5 of 100 px off by 1": (off(small, 5), small, "fp32", "pixels_off", None),
        "14 of 139,968 px off by 1": (off(crop, 14), crop, "fp32", "pixels_off", None),
        "15 of 139,968 px off by 1": (off(crop, 15), crop, "fp32", "pixels_off", "pixels off 15 > 14"),
        "one pixel off by 2": (off(crop, 1, 2), crop, "fp32", "pixels_off", "pixel off by 2"),
        "high variance still on r": (off(hivar, 5), hivar, "fp32", "r", "r "),
        "variance of exactly 1 on r": (off(at_bar, 5), at_bar, "fp32", "r", "r "),
    }
    ok = True
    for what, (img, r, mode, rule, want) in cases.items():
        m = {**meta, "precision": mode, "shape_yx": list(r.shape)}
        c, p, f, s = gate(img, m, r, m, mode, False)
        got_rule = s["similarity"]["rule"]
        hit = got_rule == rule and (p == c if want is None else len(f) == 1 and f[0].startswith(want))
        print(f"selftest {what}: {'ok' if hit else 'NOT as expected'} (rule {got_rule}, variance "
              f"{s['similarity']['ref_variance']:.3f}, off {s['similarity']['pixels_off']}, failures {f})")
        ok &= hit
    # the reference's variance picks the rule, never ours
    for what, ours, ref, rule in (("flat reference, varied ours", hivar, small, "pixels_off"),
                                  ("varied reference, flat ours", small, hivar, "r")):
        got = similarity(ours, ref, LIMITS["fp32"]["min_r"]).rule
        print(f"selftest {what}: {'ok' if got == rule else 'NOT as expected'} (rule {got})")
        ok &= got == rule
    return ok


def selftest():
    """gate() on synthetic maps: a close copy passes, and each way a port can be wrong is rejected by the
    check meant for it (and, for the numerical bounds, by that check alone), so removing any check
    fails the selftest"""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:300, 0:400]
    ref = np.clip(128 + 100 * np.sin(yy / 17.0) * np.cos(xx / 23.0), 0, 255)
    ref[:, :60] = 0  # off the support
    ref = np.rint(ref).astype(np.uint8)
    meta = {"checkpoint": "hecate_9.6um.pth", "sampling_um": 9.6, "shape_yx": [300, 400], "reverse_z": False,
            "central_depth": 16, "precision": "bf16", "stride": 32, "probabilities": "x"}
    close = np.where(ref > 1, np.clip(ref.astype(int) + rng.integers(-2, 3, ref.shape), 2, 255), ref).astype(np.uint8)

    def spikes(size, frac):
        # +-size steps on a fraction of mid-range pixels, alternating in sign so the mean does not move
        out = close.astype(int)
        idx = np.flatnonzero((ref >= 40) & (ref <= 215))
        pick = rng.choice(idx, int(frac * ref.size), replace=False)
        out.flat[pick] += np.where(np.arange(pick.size) % 2 == 0, size, -size)
        return out.astype(np.uint8)

    # a low-contrast reference, where 2-step noise is a poor correlation but inside every other bound
    flat = np.rint(128 + 3 * np.sin(yy / 17.0) * np.cos(xx / 23.0)).astype(np.uint8)
    flat_noisy = (flat.astype(int) + rng.integers(-2, 3, flat.shape)).astype(np.uint8)
    no_ckpt = {k: v for k, v in meta.items() if k != "checkpoint"}
    zeros = np.zeros_like(ref)
    # name: (ours, ours meta, reference, reference meta, direction reverse, rejecting check, only that one)
    cases = {
        "constant": (np.full_like(ref, 128), meta, ref, meta, False, "r ", False),
        "shifted 3 px": (np.roll(ref, 3, axis=1), meta, ref, meta, False, "r ", False),
        "support lost": (np.where(xx < 90, 0, ref).astype(np.uint8), meta, ref, meta, False, "zero mismatch", False),
        "support grown": (np.where(xx < 60, 40, ref).astype(np.uint8), meta, ref, meta, False, "zero mismatch", False),
        "reverse JSON": (close, {**meta, "reverse_z": True}, ref, meta, False, "JSON reverse_z", False),
        "fp32 JSON": (close, {**meta, "precision": "fp32"}, ref, meta, False, "JSON precision", False),
        "other stride": (close, {**meta, "stride": 16}, ref, meta, False, "JSON stride", False),
        "no JSON": (close, None, ref, meta, False, "a map has no JSON", False),
        "field missing on both sides": (close, no_ckpt, ref, no_ckpt, False, "JSON checkpoint missing", True),
        "cropped": (close[:, :399], meta, ref, meta, False, "shape", False),
        "wrong --direction": (close, meta, ref, meta, True, "JSON reverse_z", False),
        "both all zero": (zeros, meta, zeros, meta, False, "reference map is constant", False),
        "both constant": (np.full_like(ref, 128), meta, np.full_like(ref, 128), meta, False, "reference map is constant", True),
        "ours all zero": (zeros, meta, ref, meta, False, "our map is all zero", False),
        "both fp32, gate bf16": (close, {**meta, "precision": "fp32"}, ref, {**meta, "precision": "fp32"}, False,
                                 "JSON precision 'fp32', gate mode", True),
        "JSON shape not the PNG's": (close, {**meta, "shape_yx": [300, 399]}, ref, {**meta, "shape_yx": [300, 399]}, False,
                                     "JSON shape_yx [", True),
        "r only": (flat_noisy, meta, flat, meta, False, "r ", True),
        "p99.9 only": (spikes(7, 0.005), meta, ref, meta, False, "p99.9", True),
        "max only": (spikes(33, 0.0005), meta, ref, meta, False, "max ", True),
        "bias only": (np.where(ref > 1, ref + 1, ref).astype(np.uint8), meta, ref, meta, False, "bias", True),
    }
    c, p, f, _ = gate(close, meta, ref, meta, "bf16", False)
    ok = c > 0 and p == c
    print(f"selftest close copy: {'pass' if ok else 'FAIL ' + str(f)}")
    ok &= low_variance_cases(meta)
    for what, (img, m, r, rm, reverse, want, alone) in cases.items():
        c, p, f, _ = gate(img, m, r, rm, "bf16", reverse)
        hit = any(x.startswith(want) for x in f) and (not alone or len(f) == 1)
        print(f"selftest {what}: {'rejected by ' + repr(want) if hit else 'NOT as expected ' + repr(want) + ' ' + str(f)}")
        ok &= hit
    # 0 against 1 both ways passes and is counted exactly in zero_one_side, never in zero_mismatch
    ones_ref, ones_ours = ref.copy(), close.copy()
    on, off = np.flatnonzero(ref > 1)[:3], np.flatnonzero(ref == 0)[:2]
    ones_ref.flat[on], ones_ours.flat[on], ones_ours.flat[off] = 1, 0, 1
    c, p, f, s = gate(ones_ours, meta, ones_ref, meta, "bf16", False)
    hit = p == c and s["zero_one_side"] == 5 and s["zero_mismatch"] == 0
    print(f"selftest 0 against 1 counted: {'ok' if hit else 'NOT as expected ' + str((f, s))}")
    ok &= hit
    print("selftest", "ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def main():
    if sys.argv[1:] == ["selftest"]:
        selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("ours", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("--mode", choices=sorted(LIMITS), required=True)
    ap.add_argument("--direction", choices=["fwd", "rev"], required=True)
    ap.add_argument("--spread", nargs=2, type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    (ours, ours_meta), (ref, ref_meta) = read(a.ours), read(a.ref)
    checks, passed, failures, s = gate(ours, ours_meta, ref, ref_meta, a.mode, a.direction == "rev")
    spread = None
    if a.spread:
        (sa, _), (sb, _) = read(a.spread[0]), read(a.spread[1])
        spread = stats(sa, sb) if sa.shape == sb.shape else None
    out = dict(ours=str(a.ours), ref=str(a.ref), mode=a.mode, direction=a.direction, stats=s, limits=LIMITS[a.mode],
               rule=s["similarity"]["rule"] if s else None, low_variance=LOW_VARIANCE._asdict(),
               hecate_spread=spread, checks=checks, passed=passed, failures=failures)
    print(json.dumps(out))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
    sys.exit(0 if passed == checks and checks > 0 else 1)


if __name__ == "__main__":
    main()
