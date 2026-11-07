"""Inputs of the reader-v2 and Hecate gates: one public segment at native resolution and resampled to 9.6 um

    python -I scripts/make_hecate_inputs.py TIFS_DIR UM OUT_DIR

TIFS_DIR holds a segment's surface-volume layers `00.tif`, `01.tif`, ... (uint8), UM their voxel size. Writes
zarr v2 groups with array `0` (zstd level 0) and no pyramid, so neither villa nor inkrs runs a low-resolution
occupancy scan:
  native.zarr        the whole surface volume
  native-crop.zarr   its central 2048 x 2048 crop, S1's size (reader-v2 parity)
  r96.zarr, r96.npy  the whole volume resampled to 9.6 um with reader-v2's `resample_96`, as gomyway's published maps were
  r96-crop.zarr      the central 2048 x 2048 crop of r96 (Hecate parity)
and `inputs.json` with the shapes, crop origins and each array's SHA-256.

    python -I scripts/make_hecate_inputs.py edge OUT_DIR
    python -I scripts/make_hecate_inputs.py verify OUT_DIR NAME...

`edge` cuts r96-edge.zarr from r96.npy: the first 27 planes (an odd depth, so the forward and reverse
central slabs differ by a plane) of the first 48-row band (shorter than a tile, so every tile is
zero-padded) where, in both directions, at least one tile is entirely empty (skipped, its window still
in the denominator) and at least one is kept. It records those counts in inputs.json and fails if no
band qualifies. `verify` re-hashes the named arrays and fails unless each matches inputs.json, so a gate
job cannot read an input that changed after it was made.

`resample_96` is copied from DomRusso2/reader-v2 02c7b46 `scripts/infer_with_hecate.py:15-22` (MIT,
"Copyright (c) 2026 Domenico Russo"), unchanged
"""

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile
import zarr
from numcodecs import Zstd

CROP = 2048


def resample_96(vol, um):
    f = um / 9.6
    nz, ny, nx = vol.shape
    oy, ox, oz = int(round(ny * f)), int(round(nx * f)), int(round(nz * f))
    xy = np.stack([cv2.resize(np.asarray(vol[k]), (ox, oy), interpolation=cv2.INTER_AREA) for k in range(nz)])
    zs = (np.arange(oz) + 0.5) / f - 0.5
    z0 = np.clip(np.floor(zs).astype(int), 0, nz - 1); z1 = np.clip(z0 + 1, 0, nz - 1); w = np.clip(zs - z0, 0, 1)
    return np.stack([np.round((1 - w[k]) * xy[z0[k]] + w[k] * xy[z1[k]]).astype(np.uint8) for k in range(oz)])


def centre_crop(vol):
    _, h, w = vol.shape
    y0, x0 = (h - CROP) // 2, (w - CROP) // 2
    return vol[:, y0:y0 + CROP, x0:x0 + CROP], [y0, x0]


def write(path: Path, vol):
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    group.create_array("0", data=vol, chunks=(vol.shape[0], 256, 256), compressors=Zstd(level=0), filters=None)
    return {"shape": list(vol.shape), "sha256": hashlib.sha256(np.ascontiguousarray(vol).tobytes()).hexdigest()}


EDGE_DEPTH, EDGE_ROWS, P, D, STRIDE = 27, 48, 64, 16, 32


def tile_counts(vol, reverse):
    """(kept, empty) tiles of hecate.py's predict on `vol` at the default stride"""
    depth, h, w = vol.shape
    z0 = depth // 2 - D // 2
    slab = vol[depth - z0 - D:depth - z0] if reverse else vol[z0:z0 + D]
    occ = slab.any(0)
    starts = lambda n: list(range(0, max(0, n - P), STRIDE)) + [max(0, n - P)]
    kept = sum(bool(occ[y:y + P, x:x + P].any()) for y in starts(h) for x in starts(w))
    return kept, len(starts(h)) * len(starts(w)) - kept


def edge(out):
    r96 = np.load(out / "r96.npy")[:EDGE_DEPTH]
    for y0 in range(0, r96.shape[1] - EDGE_ROWS + 1, 16):
        band = r96[:, y0:y0 + EDGE_ROWS]
        fwd, rev = tile_counts(band, False), tile_counts(band, True)
        if min(fwd + rev) > 0:
            break
    else:
        raise SystemExit("no 48-row band of r96 has both kept and empty tiles in both directions")
    info = json.loads((out / "inputs.json").read_text())
    info["arrays"]["r96-edge"] = {**write(out / "r96-edge.zarr", band), "origin_yx": [y0, 0],
                                  "tiles_kept_empty": {"forward": fwd, "reverse": rev}}
    (out / "inputs.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info["arrays"]["r96-edge"]))


def verify(out, names):
    info = json.loads((out / "inputs.json").read_text())["arrays"]
    bad = []
    for name in names:
        arr = np.load(out / name) if name.endswith(".npy") else zarr.open_group(str(out / f"{name}.zarr"), mode="r")["0"][:]
        key = name.removesuffix(".npy")
        sha = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        ok = key in info and sha == info[key]["sha256"] and list(arr.shape) == info[key]["shape"]
        print(f"{name}: {'ok' if ok else 'CHANGED'} {sha[:12]}")
        if not ok:
            bad.append(name)
    if bad:
        raise SystemExit(f"inputs differ from inputs.json: {bad}")


def main():
    if sys.argv[1] == "edge":
        return edge(Path(sys.argv[2]))
    if sys.argv[1] == "verify":
        return verify(Path(sys.argv[2]), sys.argv[3:])
    tifs, um, out = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
    layers = sorted(tifs.glob("*.tif"), key=lambda p: int(p.stem))
    assert [int(p.stem) for p in layers] == list(range(len(layers))), "layers are not numbered 0..n-1"
    vol = np.stack([tifffile.imread(p) for p in layers])
    assert vol.dtype == np.uint8 and vol.ndim == 3, (vol.dtype, vol.shape)
    out.mkdir(parents=True, exist_ok=True)
    info = {"source": str(tifs), "um": um, "arrays": {}}
    info["arrays"]["native"] = write(out / "native.zarr", vol)
    crop, origin = centre_crop(vol)
    info["arrays"]["native-crop"] = {**write(out / "native-crop.zarr", crop), "origin_yx": origin}
    r96 = resample_96(vol, um)
    info["arrays"]["r96"] = write(out / "r96.zarr", r96)
    np.save(out / "r96.npy", r96)
    crop, origin = centre_crop(r96)
    info["arrays"]["r96-crop"] = {**write(out / "r96-crop.zarr", crop), "origin_yx": origin}
    (out / "inputs.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
