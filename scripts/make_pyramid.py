"""Builds a multiscale zarr v2 group from a level-0-only render, for the known-letter check

    python -I scripts/make_pyramid.py SRC_GROUP DEST_GROUP [--levels 3]

Level 0 is a symlink to SRC_GROUP/0, so its voxels are the render's own. Each level k >= 1 is the
2x2x2 block mean of level k-1 (all three axes, partial edge blocks averaged over the voxels they
hold, rounded half to even), uint8, uncompressed, in [depth, 128, 128] chunks. This is our own
construction: the recipe's PHerc. 343 renders have no pyramid (its run_ink.sh strips it), and
whether VC3D's pyramid uses the same reduction is unconfirmed. villa and inkrs read the same
group, so the parity check does not depend on that choice. A mean pyramid drops sparse nonzero
voxels at the papyrus edge, which is where villa's level-3 skip and a level-0 rule differ.
Writes OME-Zarr 0.4 `multiscales` metadata and copies the source's other attributes
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def read_level0(src):
    meta = json.loads((src / "0" / ".zarray").read_text())
    import zarr

    return np.asarray(zarr.open_array(str(src / "0"), mode="r")[:]), meta


def down2(a):
    """2x2x2 block mean, ceil sizes, rounded half to even"""
    d, h, w = a.shape
    nd, nh, nw = -(-d // 2), -(-h // 2), -(-w // 2)
    total = np.zeros((nd, nh, nw), np.float64)
    count = np.zeros((nd, nh, nw), np.float64)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                s = a[dz::2, dy::2, dx::2]
                total[: s.shape[0], : s.shape[1], : s.shape[2]] += s
                count[: s.shape[0], : s.shape[1], : s.shape[2]] += 1
    return np.rint(total / count).astype(np.uint8)


def write_array(path, a, chunk=128):
    path.mkdir(parents=True)
    d, h, w = a.shape
    meta = dict(zarr_format=2, shape=[d, h, w], chunks=[d, chunk, chunk], dtype="|u1", fill_value=0, order="C", filters=None,
                compressor=None, dimension_separator="/")
    (path / ".zarray").write_text(json.dumps(meta, indent=1))
    for iy in range(-(-h // chunk)):
        for ix in range(-(-w // chunk)):
            block = np.zeros((d, chunk, chunk), np.uint8)
            part = a[:, iy * chunk:(iy + 1) * chunk, ix * chunk:(ix + 1) * chunk]
            block[:, : part.shape[1], : part.shape[2]] = part
            f = path / "0" / str(iy) / str(ix)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(block.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--levels", type=int, default=3)
    a = ap.parse_args()
    if a.dest.exists():
        raise SystemExit(f"{a.dest} exists")
    tmp = a.dest.with_name(a.dest.name + ".partial")
    if tmp.exists():
        raise SystemExit(f"{tmp} exists; remove it first")
    tmp.mkdir(parents=True)
    (tmp / ".zgroup").write_text(json.dumps(dict(zarr_format=2)))
    os.symlink((a.src / "0").resolve(), tmp / "0")
    level, meta = read_level0(a.src)
    shapes = [list(level.shape)]
    for k in range(1, a.levels + 1):
        level = down2(level)
        write_array(tmp / str(k), level)
        shapes.append(list(level.shape))
        print(f"level {k}: {level.shape}, nonzero {float((level != 0).mean()):.4f}", flush=True)
    attrs = json.loads((a.src / ".zattrs").read_text()) if (a.src / ".zattrs").exists() else {}
    attrs["multiscales"] = [dict(version="0.4", name=a.src.name, axes=[dict(name=n, type="space") for n in "zyx"],
                                 datasets=[dict(path=str(k), coordinateTransformations=[dict(type="scale", scale=[2.0 ** k] * 3)])
                                           for k in range(a.levels + 1)],
                                 metadata=dict(method="2x2x2 block mean, rounded half to even (inkrs scripts/make_pyramid.py)",
                                               level0=str((a.src / "0").resolve())))]
    (tmp / ".zattrs").write_text(json.dumps(attrs, indent=1))
    tmp.rename(a.dest)
    print(json.dumps(dict(dest=str(a.dest), shapes=shapes, level0_chunks=meta["chunks"])))


if __name__ == "__main__":
    main()
