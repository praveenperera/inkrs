"""Fixture for the CPU normaliser test: villa's robust normalisation of real tiles

For every tile villa would run on the given maps, writes one JSON line with the tile, villa's
statistics as `normalize_robust` uses them (lower and upper percentile, median, the float32
divisor, whether the MAD fallback fired) and the SHA-256 of villa's normalised float32 patch
in ZYX order. The statistics are recomputed here with villa's own steps; each tile's patch is
also checked against `normalize_robust` itself, so the recomputation cannot drift from it

    python -I scripts/norm_fixture.py [--patch 384] [--arith fastink] ZARR STRIDE OUT.jsonl FIRST:STEP [FIRST:STEP ...]

`--patch` is the reader's tile edge: 128 (the default), or 384 for Reader V3. `--arith fastink` records instead
the PHerc. 0306B recipe's torch normaliser (`fastink.robust_norm_gpu`, run here on the CPU from the method's
checkout under FL_ROOT), which the 8-fold TTA reads replay: `sha256_f16` and `fallback` are then fastink's, and
each record says `"arith": "fastink"` for `inkrs check-norm --arith fastink`

CPU only; run where villa 0fb38c45 is importable (installed, or VILLA_SRC)
"""

import hashlib
import json
import os
import sys

# villa 0fb38c45: the installed package, or the vesuvius/src of a checkout named by VILLA_SRC
if os.environ.get("VILLA_SRC"):
    sys.path.insert(0, os.environ["VILLA_SRC"])

import numpy as np  # noqa: E402
import zarr  # noqa: E402
from vesuvius.image_proc.intensity.normalization import normalize_robust  # noqa: E402
from vesuvius.ink_detection.inference.infer import _sliding_positions_1d  # noqa: E402


def stats(patch_zyx):
    # villa normalize_robust, reduced to the values it uses
    arr = patch_zyx.astype(np.float32, copy=True)
    valid = arr.reshape(-1)
    std_preclip = float(np.std(valid))
    lo = float(np.percentile(valid, 1.0))
    hi = float(np.percentile(valid, 99.0))
    np.clip(arr, lo, hi, out=arr)
    valid = arr.reshape(-1)
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    scaled = 1.4826 * mad
    fallback = not np.isfinite(scaled) or scaled < 1e-6
    if fallback:
        cands = [abs(std_preclip), abs(hi - lo) / 2.0]
        scaled = next((c for c in cands if c >= 1e-6), None)
        if scaled is None or not np.isfinite(scaled):
            scaled = 1.0
    return lo, hi, median, scaled, fallback, std_preclip


def fastink_record(zyx):
    """fastink's normalised patch and whether its MAD fell back, from the method's own function"""
    import torch

    sys.path.insert(0, os.path.join(os.environ["FL_ROOT"], "x_research2", "speed"))
    import fastink

    x = torch.from_numpy(np.ascontiguousarray(zyx, dtype=np.float32))[None]
    out = fastink.robust_norm_gpu(x)[0].numpy()
    # robust_norm_gpu's own steps up to its fallback test
    s, _ = torch.sort(x.reshape(1, -1), dim=1)
    lo, hi = fastink._pct_sorted(s, 1.0), fastink._pct_sorted(s, 99.0)
    c = torch.maximum(torch.minimum(s, hi[:, None]), lo[:, None])
    med = fastink._pct_sorted(c, 50.0)
    dev, _ = torch.sort((c - med[:, None]).abs(), dim=1)
    scaled = 1.4826 * fastink._pct_sorted(dev, 50.0)
    fallback = bool((~torch.isfinite(scaled) | (scaled < 1e-6)).item())
    return out, fallback


def main():
    args = sys.argv[1:]
    patch = 128
    arith = "villa"
    while args[:1] in (["--patch"], ["--arith"]):
        if args[0] == "--patch":
            patch = int(args[1])
        else:
            arith = args[1]
        args = args[2:]
    assert arith in ("villa", "fastink"), arith
    assert patch in (128, 384), patch
    path, stride, out = args[0], int(args[1]), args[2]
    maps = [tuple(int(v) for v in m.split(":")) for m in args[3:]]
    root = zarr.open(path, mode="r")
    arr = root["0"] if hasattr(root, "keys") and "0" in root else root
    vol = np.asarray(arr[:])
    depth, height, width = vol.shape
    ys = _sliding_positions_1d(height, patch, stride)
    xs = _sliding_positions_1d(width, patch, stride)
    counts = {}
    with open(out, "w") as f:
        for first, step in maps:
            layers = [first + step * k for k in range(17)]
            n = fb = 0
            for y0 in ys:
                for x0 in xs:
                    # villa's reader: HWZ, then moveaxis to a ZYX view
                    hwz = np.transpose(vol[layers, y0 : y0 + patch, x0 : x0 + patch], (1, 2, 0)).astype(np.float32)
                    if not hwz.any():
                        continue
                    zyx = np.moveaxis(hwz, -1, 0)
                    if arith == "fastink":
                        ref, fallback = fastink_record(zyx)
                        rec = dict(y0=y0, x0=x0, first=first, step=step, arith="fastink", fallback=fallback,
                                   sha256_f16=hashlib.sha256(ref.astype(np.float16).tobytes()).hexdigest())
                        f.write(json.dumps(rec) + "\n")
                        n += 1
                        fb += fallback
                        continue
                    ref = np.ascontiguousarray(normalize_robust(zyx))
                    lo, hi, median, scaled, fallback, std = stats(zyx)
                    mine = (np.clip(zyx, lo, hi) - np.float32(median)) / np.float32(scaled)
                    assert np.array_equal(np.ascontiguousarray(mine, dtype=np.float32).view(np.uint32), ref.view(np.uint32)), (y0, x0)
                    rec = dict(
                        y0=y0, x0=x0, first=first, step=step, lo=lo, hi=hi, median=median,
                        divisor=float(np.float32(scaled)), fallback=fallback, std_preclip=std,
                        sha256=hashlib.sha256(ref.tobytes()).hexdigest(),
                        sha256_f16=hashlib.sha256(ref.astype(np.float16).tobytes()).hexdigest(),
                    )
                    f.write(json.dumps(rec) + "\n")
                    n += 1
                    fb += fallback
            counts[f"{first}:{step}"] = dict(tiles=n, fallback=fb)
            print(first, step, n, "tiles", fb, "fallback", flush=True)
    print(json.dumps(dict(volume=path, shape=[depth, height, width], patch=patch, stride=stride, arith=arith, maps=counts)))


if __name__ == "__main__":
    main()
