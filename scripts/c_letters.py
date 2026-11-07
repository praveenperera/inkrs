"""Known-letter check, downstream part: the recipe's reading of one model, from inkrs's maps and from
villa's, compared over the published letter boxes

    python -I -B scripts/c_letters.py OURS REF ZARR BOXES.json --name NAME --depths=-4..4 [--json OUT] [--png DIR]

Transcribes `ensemble_maps.py` (pherc343-first-letters, imported read-only for `read_map`, `pct_hist`
and `to8`) for hybrid_3d2d-seed42 alone, separately for each side:
  - valid pixels: CT > 0 in all 66 layers of level 0
  - u = clip((v - 64) / 128, 0, 1) per uint8 map
  - stats pooled over the reading-side (reverse) maps of every depth, valid pixels: p1 and p99.5 exact
    from the 129-bin histogram, mean and SD
  - per depth and side, the per-model image to8(u, p1, p99.5) and the z-score (u - mean) / sd
Bounds, fixed before the check ran: |dp1|, |dp99.5| <= 1/128; |dmean|, |dsd| <= 1e-3; over the letter mask, at
every depth and side, display r >= 0.9994, max |d| <= B = ceil(255 * 6/128 / (p99.5 - p1)_ref) levels
and mean |d| <= 0.5 levels. Exit 0 when every check passes
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# a checkout of github.com/Nieuwlaar/pherc343-first-letters (MIT; see THIRD_PARTY_NOTICES), named by LETTERS_SRC
sys.path.insert(0, os.path.join(os.environ["LETTERS_SRC"], "ink", "scripts"))
import ensemble_maps as em  # noqa: E402

READER = "hybrid_3d2d-seed42"


def parse_depths(s):
    a, b = s.split("..")
    return list(range(int(a), int(b) + 1))


def valid_mask(zarr_path):
    import zarr

    st = zarr.open_group(str(zarr_path), mode="r")["0"]
    assert st.shape[0] == em.N66, st.shape
    h, w = int(st.shape[1]), int(st.shape[2])
    valid = np.zeros((h, w), bool)
    for r0 in range(0, h, 128):
        valid[r0:r0 + 128] = (np.asarray(st[:, r0:r0 + 128, :]) > 0).all(0)
    return valid


def letter_mask(boxes_path, shape):
    boxes = json.loads(Path(boxes_path).read_text())
    assert boxes["frame"] == "raw", boxes["frame"]
    m = np.zeros(shape, bool)
    for b in boxes["letters"]:
        x0, y0, x1, y1 = b["box"]
        m[max(0, math.floor(y0)):math.ceil(y1), max(0, math.floor(x0)):math.ceil(x1)] = True
    return m, len(boxes["letters"])


def path(run, name, d, reverse):
    return run / "ink" / f"{name}__{READER}__L66s{25 + d}{'_reverse' if reverse else ''}.tif"


def stats(run, name, depths, valid, shape):
    hist = np.zeros(129, np.int64)
    s1 = s2 = 0.0
    n = 0
    for d in depths:
        v = em.read_map(str(path(run, name, d, True)), shape)[valid]
        hist += np.bincount(np.rint(v * 128).astype(np.int64), minlength=129)[:129]
        s1 += float(v.sum(dtype=np.float64))
        s2 += float(np.square(v, dtype=np.float64).sum())
        n += v.size
    lo, hi = em.pct_hist(hist, np.arange(129) / 128.0, [em.P_LO, em.P_HI])
    mu = s1 / n
    return dict(p1=lo, p99_5=hi, mean=mu, sd=math.sqrt(max(s2 / n - mu * mu, 0.0)) + 1e-6, n_px=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ours", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("zarr", type=Path)
    ap.add_argument("boxes", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--depths", default="-4..4")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--png", type=Path)
    a = ap.parse_args()
    depths = parse_depths(a.depths)
    valid = valid_mask(a.zarr)
    shape = valid.shape
    letters, n_boxes = letter_mask(a.boxes, shape)
    so, sr = stats(a.ours, a.name, depths, valid, shape), stats(a.ref, a.name, depths, valid, shape)
    bound = math.ceil(255 * (6 / 128) / (sr["p99_5"] - sr["p1"]))
    checks = passed = 0
    failures = []

    def check(ok, what):
        nonlocal checks, passed
        checks += 1
        passed += bool(ok)
        if not ok:
            failures.append(what)

    for k, lim in (("p1", 1 / 128), ("p99_5", 1 / 128), ("mean", 1e-3), ("sd", 1e-3)):
        check(abs(so[k] - sr[k]) <= lim, f"stat {k}: ours {so[k]:.6f} ref {sr[k]:.6f}, |d| > {lim:.6f}")
    rows = []
    for d in depths:
        for reverse in (True, False):
            mo = em.read_map(str(path(a.ours, a.name, d, reverse)), shape)
            mr = em.read_map(str(path(a.ref, a.name, d, reverse)), shape)
            do = em.to8(mo, so["p1"], so["p99_5"], valid, "raw").astype(np.float64)
            dr = em.to8(mr, sr["p1"], sr["p99_5"], valid, "raw").astype(np.float64)
            zo, zr = (mo - so["mean"]) / so["sd"], (mr - sr["mean"]) / sr["sd"]
            x, y = do[letters], dr[letters]
            diff = np.abs(x - y)
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float(np.array_equal(x, y))
            dz = np.abs(zo - zr)[letters & valid]
            tag = f"d{d:+d} {'reading' if reverse else 'blind'}"
            row = dict(depth=d, side="reading" if reverse else "blind", r=r, max=float(diff.max()), mean=float(diff.mean()),
                       p999=float(np.quantile(diff, 0.999)), equal_share=float((diff == 0).mean()), max_dz=float(dz.max()),
                       mean_dz=float(dz.mean()), letter_px=int(letters.sum()))
            check(r >= 0.9994, f"{tag}: display r {r:.6f} < 0.9994 over the letters")
            check(row["max"] <= bound, f"{tag}: display max |d| {row['max']:.0f} > {bound}")
            check(row["mean"] <= 0.5, f"{tag}: display mean |d| {row['mean']:.3f} > 0.5")
            rows.append(row)
            print(f"{tag:<14} r {r:.7f} max {row['max']:.0f} mean {row['mean']:.4f} p99.9 {row['p999']:.0f} equal {row['equal_share']:.4f} max|dz| {row['max_dz']:.4f}", flush=True)
            if a.png and d == 1 and reverse:
                import cv2

                a.png.mkdir(parents=True, exist_ok=True)
                ys, xs = np.nonzero(letters)
                crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
                # reading orientation, as the published image shows it
                for label, img in (("inkrs", do), ("villa", dr), ("absdiff-x16", np.minimum(np.abs(do - dr) * 16, 255))):
                    cv2.imwrite(str(a.png / f"{a.name}__d+01__{label}.png"), np.ascontiguousarray(img[crop][::-1, ::-1]).astype(np.uint8))
    for f in failures:
        print("FAIL", f)
    head = next(r for r in rows if r["depth"] == 1 and r["side"] == "reading") if 1 in depths else None
    summary = dict(checks=checks, passed=passed, failed=checks - passed, boxes=n_boxes, letter_px=int(letters.sum()),
                   valid_px=int(valid.sum()), stats=dict(ours=so, ref=sr), display_bound=bound, headline_d1_reading=head)
    print(json.dumps(summary, indent=1))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(dict(summary=summary, rows=rows, failures=failures), indent=1))
    sys.exit(0 if checks == passed else 1)


if __name__ == "__main__":
    main()
