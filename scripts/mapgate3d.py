"""Map gate for 3D ink output: two uint8 zarr predictions (array 0 of each group) against villa's parity contract

    python -I scripts/mapgate3d.py REF.zarr NEW.zarr [--villa-plan PLAN.json] [--bbox z0,y0,x0,z1,y1,x1] [--json OUT]

Reports max |diff|, the fraction of voxels that differ, Pearson r and p99.9 |diff| over the region (default: the
whole array), read in z slabs. The contract: r >= 0.9994, p99.9 |diff| <= 4 codes (4/255) and identical coverage.

Coverage is villa's finish rule: a voxel is covered when its accumulated blend weight exceeds 1e-6 (the Gaussian
is clipped at float32 eps, so voxels only a patch's far corner reaches are not). With `--villa-plan` (the
`<tag>_plan.json` scripts/villa_map3d.py writes), villa's coverage is rebuilt from villa's own patches, target
chunks and importance map, summed in villa's patch order, and must equal the engine's mask (`NEW/coverage`, from
`inkrs ink3d --coverage`) bit for bit; both maps must also be 0 everywhere outside it. Without a plan, or for a
NEW map with no coverage group, the zero masks of the two maps must match exactly instead, which a voxel that
rounds to 0 on one side and 1 on the other fails. A constant map counts as r = 1 only when both maps are equal.
Exits 1 when the contract fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr


def open0(path: Path):
    return zarr.open_array(str(path / "0") if (path / "0").exists() else str(path), mode="r")


class VillaCoverage:
    """villa's covered voxels, weight sum > 1e-6, from its plan and its own importance map"""

    def __init__(self, plan_path: str):
        from vesuvius.ink_detection.inference import infer_full3d_tifxyz as F

        plan = json.loads(Path(plan_path).read_text())
        self.p = [int(v) for v in plan["patch_size"]]
        self.chunk = [int(v) for v in plan["chunk_shape"]]
        self.patches = sorted(tuple(int(v) for v in p) for p in plan["patches"])
        self.targets = {tuple(int(v) for v in c) for c in plan["target_chunks"]}
        self.imp = np.asarray(F.create_importance_map(tuple(self.p), mode="gaussian"), dtype=np.float32)
        if self.imp.shape != tuple(self.p):
            raise SystemExit(f"importance map {self.imp.shape} vs patch {self.p}")

    def slab(self, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        w = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.float32)
        p = self.p
        for pz, py, px in self.patches:
            lo = (max(pz, z0), max(py, y0), max(px, x0))
            hi = (min(pz + p[0], z1), min(py + p[1], y1), min(px + p[2], x1))
            if any(a >= b for a, b in zip(lo, hi)):
                continue
            w[lo[0] - z0:hi[0] - z0, lo[1] - y0:hi[1] - y0, lo[2] - x0:hi[2] - x0] += self.imp[
                lo[0] - pz:hi[0] - pz, lo[1] - py:hi[1] - py, lo[2] - px:hi[2] - px]
        covered = w > np.float32(1e-6)
        # villa accumulates and writes target chunks only; everything else stays at the fill value
        c = self.chunk
        for kz in range(z0 // c[0], (z1 - 1) // c[0] + 1):
            for ky in range(y0 // c[1], (y1 - 1) // c[1] + 1):
                for kx in range(x0 // c[2], (x1 - 1) // c[2] + 1):
                    if (kz, ky, kx) not in self.targets:
                        covered[max(kz * c[0], z0) - z0:min((kz + 1) * c[0], z1) - z0,
                                max(ky * c[1], y0) - y0:min((ky + 1) * c[1], y1) - y0,
                                max(kx * c[2], x0) - x0:min((kx + 1) * c[2], x1) - x0] = False
        return covered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--villa-plan")
    ap.add_argument("--bbox")
    ap.add_argument("--slab", type=int, default=64)
    ap.add_argument("--json")
    args = ap.parse_args()
    ref, new = Path(args.ref), Path(args.new)
    a, b = open0(ref), open0(new)
    if a.shape != b.shape:
        raise SystemExit(f"shape {a.shape} vs {b.shape}")
    box = [int(v) for v in args.bbox.split(",")] if args.bbox else [0, 0, 0, *a.shape]
    if not all(0 <= box[i] < box[i + 3] <= a.shape[i] for i in range(3)):
        raise SystemExit(f"bbox {box} empty or outside {a.shape}")
    has_cov = (new / "coverage").exists()
    villa = VillaCoverage(args.villa_plan) if args.villa_plan and has_cov else None
    nc = open0(new / "coverage") if villa else None
    if nc is not None and nc.shape != a.shape:
        raise SystemExit(f"coverage {nc.shape} vs map {a.shape}")

    hist = np.zeros(256, np.int64)
    n = 0
    s = {"a": 0.0, "b": 0.0, "aa": 0.0, "bb": 0.0, "ab": 0.0}
    nz_a = nz_b = zero_mismatch = 0
    cov = {"villa_covered": 0, "new_covered": 0, "coverage_mismatch": 0, "ref_nonzero_uncovered": 0,
           "new_nonzero_uncovered": 0, "new_mask_not_01": 0}
    for z in range(box[0], box[3], args.slab):
        z1 = min(z + args.slab, box[3])
        x = np.asarray(a[z:z1, box[1]:box[4], box[2]:box[5]]).astype(np.int16)
        y = np.asarray(b[z:z1, box[1]:box[4], box[2]:box[5]]).astype(np.int16)
        hist += np.bincount(np.abs(x - y).ravel(), minlength=256)
        n += x.size
        xf, yf = x.astype(np.float64), y.astype(np.float64)
        s["a"] += xf.sum()
        s["b"] += yf.sum()
        s["aa"] += (xf * xf).sum()
        s["bb"] += (yf * yf).sum()
        s["ab"] += (xf * yf).sum()
        ma, mb = x > 0, y > 0
        nz_a += int(ma.sum())
        nz_b += int(mb.sum())
        zero_mismatch += int((ma != mb).sum())
        if villa:
            vc = villa.slab(z, z1, box[1], box[4], box[2], box[5])
            m = np.asarray(nc[z:z1, box[1]:box[4], box[2]:box[5]])
            cov["new_mask_not_01"] += int((m > 1).sum())
            m = m == 1
            cov["villa_covered"] += int(vc.sum())
            cov["new_covered"] += int(m.sum())
            cov["coverage_mismatch"] += int((vc != m).sum())
            cov["ref_nonzero_uncovered"] += int((ma & ~vc).sum())
            cov["new_nonzero_uncovered"] += int((mb & ~vc).sum())
    mean_a, mean_b = s["a"] / n, s["b"] / n
    va, vb = s["aa"] / n - mean_a * mean_a, s["bb"] / n - mean_b * mean_b
    if va > 0 and vb > 0:
        r = (s["ab"] / n - mean_a * mean_b) / np.sqrt(va * vb)
    else:
        # Pearson r is undefined for a constant map: only identical maps pass
        r = 1.0 if hist[0] == n else 0.0
    p999 = int(np.searchsorted(np.cumsum(hist) / n, 0.999))
    res = {
        "voxels": n,
        "max_abs": int(np.nonzero(hist)[0].max()),
        "frac_differ": float(1 - hist[0] / n),
        "frac_differ_gt1": float(hist[2:].sum() / n),
        "r": float(r),
        "p999_abs": p999,
        "mean_ref": mean_a,
        "mean_new": mean_b,
        "nonzero_ref": nz_a / n,
        "nonzero_new": nz_b / n,
        "zero_mismatch_voxels": zero_mismatch,
        "diff_hist_0_8": hist[:9].tolist(),
    }
    if villa:
        res["coverage_mode"] = "villa plan vs engine mask"
        res.update(cov)
        cov_ok = (cov["coverage_mismatch"] == 0 and cov["ref_nonzero_uncovered"] == 0
                  and cov["new_nonzero_uncovered"] == 0 and cov["new_mask_not_01"] == 0 and cov["villa_covered"] > 0)
    else:
        res["coverage_mode"] = "zero masks equal"
        cov_ok = zero_mismatch == 0
    res["coverage_ok"] = bool(cov_ok)
    res["contract_pass"] = bool(res["r"] >= 0.9994 and p999 <= 4 and cov_ok)
    print(json.dumps(res, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))
    sys.exit(0 if res["contract_pass"] else 1)


if __name__ == "__main__":
    main()
