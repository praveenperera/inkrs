"""P0 gate: inkrs's per-layer activations against PyTorch FP32, next to PyTorch's own autocast error

    python -I scripts/p0_compare.py RUST_DIR TORCH_DIR [--json OUT]

RUST_DIR is `inkrs p0 --out` (channels-last float32 `.npy` and a manifest), TORCH_DIR is
scripts/p0_torch.py's OUT (NCHW / NCDHW float32 under fp32/ and amp/). Per layer it reports, for
inkrs and for autocast against FP32: relative RMS error, p99.9 and max |diff|, Pearson r.
Checks, every one counted:
  - each tap exists on both sides with the same shape (after moving channels last) and is finite
  - input: inkrs's FP16 input equals villa's normalised input rounded to FP16, bit for bit
  - every other tap: inkrs's relative RMS error <= RMS_RATIO x autocast's, and its p99.9 |diff|
    <= P999_RATIO x autocast's; each with a floor of FLOOR x the reference RMS, so a layer
    where autocast is near exact does not demand more than FP16 storage can hold
Exit 0 when every check passes
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RMS_RATIO = 1.5
P999_RATIO = 2.0
# fp16 storage alone gives a relative RMS error of about 2^-11 / sqrt(3), about 2.8e-4
FLOOR = 1e-3


def channels_last(a):
    return np.moveaxis(a, 1, -1) if a.ndim >= 4 else a


def stats(x, ref):
    d = (x.astype(np.float64) - ref).ravel()
    ad = np.abs(d)
    rr = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    xs, rs = x.ravel().astype(np.float64), ref.ravel().astype(np.float64)
    r = float(np.corrcoef(xs, rs)[0, 1]) if xs.std() > 0 and rs.std() > 0 else float(np.array_equal(xs, rs))
    return dict(rel_rms=float(np.sqrt(np.mean(d**2))) / max(rr, 1e-30), p999=float(np.quantile(ad, 0.999)), max=float(ad.max()), r=r, ref_rms=rr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rust", type=Path)
    ap.add_argument("torch", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    manifest = json.loads((a.rust / "manifest.json").read_text())
    names = [t["name"] for t in manifest["taps"]]
    torch_names = sorted(p.stem for p in (a.torch / "fp32").glob("*.npy"))
    checks = passed = 0
    failures = []
    rows = []

    def check(ok, what):
        nonlocal checks, passed
        checks += 1
        passed += bool(ok)
        if not ok:
            failures.append(what)
        return ok

    check(sorted(names) == torch_names, f"tap sets differ: rust-only {sorted(set(names) - set(torch_names))}, torch-only {sorted(set(torch_names) - set(names))}")
    for name in names:
        if name not in torch_names:
            continue
        rust = np.load(a.rust / f"{name}.npy")
        f32 = channels_last(np.load(a.torch / "fp32" / f"{name}.npy"))
        amp = channels_last(np.load(a.torch / "amp" / f"{name}.npy"))
        if not check(rust.shape == f32.shape == amp.shape, f"{name}: shapes rust {rust.shape} fp32 {f32.shape} amp {amp.shape}"):
            continue
        for side, v in (("rust", rust), ("fp32", f32), ("amp", amp)):
            check(np.isfinite(v).all(), f"{name}: {side} has {int((~np.isfinite(v)).sum())} non-finite values")
        if name == "input":
            want = f32.astype(np.float16)
            same = rust.astype(np.float16).view(np.uint16) == want.view(np.uint16)
            check(same.all(), f"input: {int((~same).sum())} values differ from villa's input in fp16")
            rows.append(dict(layer=name, exact=bool(same.all()), rust=stats(rust, f32), amp=stats(amp, f32)))
            continue
        rs, am = stats(rust, f32), stats(amp, f32)
        rms_bar = max(RMS_RATIO * am["rel_rms"], FLOOR)
        p_bar = max(P999_RATIO * am["p999"], FLOOR * am["ref_rms"])
        ok_rms = check(rs["rel_rms"] <= rms_bar, f"{name}: rel RMS {rs['rel_rms']:.3e} > {rms_bar:.3e} (autocast {am['rel_rms']:.3e})")
        ok_p = check(rs["p999"] <= p_bar, f"{name}: p99.9 |diff| {rs['p999']:.3e} > {p_bar:.3e} (autocast {am['p999']:.3e})")
        rows.append(dict(layer=name, rust=rs, amp=am, rust_vs_amp=stats(rust, amp), rms_bar=rms_bar, p999_bar=p_bar, ok=ok_rms and ok_p))

    print(f"{'layer':<14} {'shape-ok':>8} {'rust relRMS':>11} {'amp relRMS':>11} {'rust p99.9':>10} {'amp p99.9':>10} {'rust max':>9} {'amp max':>9} {'rust r':>10} {'ok':>3}")
    for r in rows:
        rs, am = r["rust"], r["amp"]
        ok = r.get("exact", r.get("ok"))
        print(f"{r['layer']:<14} {'yes':>8} {rs['rel_rms']:>11.3e} {am['rel_rms']:>11.3e} {rs['p999']:>10.3e} {am['p999']:>10.3e} {rs['max']:>9.3e} {am['max']:>9.3e} {rs['r']:>10.7f} {'ok' if ok else 'NO':>3}")
    for f in failures:
        print("FAIL", f)
    summary = dict(layers=len(rows), checks=checks, passed=passed, failed=checks - passed, rms_ratio=RMS_RATIO, p999_ratio=P999_RATIO, floor=FLOOR)
    print(json.dumps(summary))
    if a.json:
        a.json.write_text(json.dumps(dict(summary=summary, layers=rows, failures=failures, algos=manifest.get("algos")), indent=1))
    sys.exit(0 if checks == passed else 1)


if __name__ == "__main__":
    main()
