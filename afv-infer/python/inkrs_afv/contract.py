"""The output contract: how a CT is tiled, read, blended and recorded, as villa ``lasagna/qualzz_inference.py``
@ 7e6b82164 does it

Tiles of 128 on a lattice of stride 96 over the input padded by a border of 16, reflect padding at its edges, and a
blend ramp that is 0 on the 16-voxel border and 1 on the 96-voxel core. These functions are copies of the band
reference this package was checked against (which reproduces villa for a z band), kept verbatim so band mode cannot
drift from it in tiling, air skipping or provenance; tests/afv checks them against that file when it is present
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

TILE, OVERLAP, BORDER = 128, 32, 16
STRIDE = TILE - OVERLAP
# equal to the stride, so the cores tile each axis with only the last, shifted one overlapping
CORE = TILE - 2 * BORDER
MODEL_FILES = ("model.py", "plans.json", "checkpoint_final.pth")
REFERENCE = "villa lasagna/qualzz_inference.py @ 7e6b82164"
DROP_IN_FOR = "AFV v2 reference predict.py @ 498ca259 (sha256 12784d48d98e…ce916)"


def blend_ramp(length: int = TILE, border: int = BORDER, overlap: int = OVERLAP) -> np.ndarray:
    """lasagna ``tiled_predict3d.blend_ramp`` (villa 7e6b82164), verbatim"""
    pad = max(0, int(border))
    ramp = np.zeros(length, dtype=np.float32)
    core_start, core_end = min(pad, length), max(min(pad, length), length - pad)
    core_len = core_end - core_start
    if core_len <= 0:
        return ramp
    core = np.ones(core_len, dtype=np.float32)
    ov = min(max(0, int(overlap) - 2 * pad), core_len // 2)
    if ov:
        edges = np.linspace(0.0, 1.0, ov + 1, dtype=np.float32)[1:]
        core[:ov], core[-ov:] = edges, edges[::-1]
    ramp[core_start:core_end] = core
    return ramp


def blend_weights() -> np.ndarray:
    """The 128³ float32 tile weights, formed in the band reference's product order"""
    w1 = blend_ramp()
    return w1[:, None, None] * w1[None, :, None] * w1[None, None, :]


def positions(size: int) -> list[int]:
    """Tile origins in padded coordinates (the reference ``_build_tile_positions`` over size + 2 * border)"""
    padded = size + 2 * BORDER
    if padded <= TILE:
        return [0]
    out = list(range(0, padded - TILE + 1, STRIDE))
    if out[-1] != padded - TILE:
        out.append(padded - TILE)
    return out


def core_range(p: int) -> tuple[int, int]:
    """Volume coordinates covered by the nonzero blend weights of the tile at padded origin p"""
    return p, p + CORE


def read_tile(ct, z0: int, z1: int, pz: int, py: int, px: int) -> np.ndarray:
    """The 128³ uint8 input of a tile; the band [z0, z1) is the volume, outside it is reflected"""
    lo = np.array([pz - BORDER, py - BORDER, px - BORDER])
    hi = lo + TILE
    shape = np.array([z1 - z0, ct.shape[1], ct.shape[2]])
    a, b = np.maximum(lo, 0), np.minimum(hi, shape)
    raw = np.asarray(ct[z0 + a[0]:z0 + b[0], a[1]:b[1], a[2]:b[2]])
    pad = [(int(x), int(y)) for x, y in zip(a - lo, hi - b)]
    if any(x or y for x, y in pad):
        raw = np.pad(raw, pad, mode="reflect")
    return raw


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ct_identity(ct: Path, z0: int, z1: int) -> str:
    """sha256 of the CT array's .zarray and the names and sizes of its chunk files in the band's chunk rows"""
    meta = json.loads((ct / ".zarray").read_text())
    h = hashlib.sha256((ct / ".zarray").read_bytes())
    cz = meta["chunks"][0]
    rows = range(z0 // cz, (z1 - 1) // cz + 1)
    if meta.get("dimension_separator") == "/":
        # nested: one directory per chunk row, so only the band's rows are listed
        files = [f for r in rows if (ct / str(r)).exists() for f in (ct / str(r)).rglob("*") if f.is_file()]
    else:
        files = [f for f in ct.iterdir() if f.is_file() and not f.name.startswith(".")
                 and int(f.name.split(".")[0]) in rows]
    for f in sorted(files):
        h.update(f"{f.relative_to(ct)} {f.stat().st_size}\n".encode())
    return h.hexdigest()


def prediction_identity(out: Path) -> str:
    """sha256 of a prediction's inference attrs and its progress file: changes whenever the prediction does"""
    attrs = json.loads((out / ".zattrs").read_text())
    h = hashlib.sha256(json.dumps(attrs, sort_keys=True).encode())
    prog = out / "progress.jsonl"
    h.update(prog.read_bytes() if prog.exists() else b"")
    return h.hexdigest()


def load_snapshot(model_dir: Path, device):
    """The snapshot's model.py ``load_model`` and ``normalize_ct``, as the reference adapter executes them"""
    spec = importlib.util.spec_from_file_location("qualzz_afv_model", model_dir / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_model(str(model_dir), device=device), module.normalize_ct


class Columns:
    """xy columns of the band, each a [y0, y1) x [x0, x1) output region with the tiles whose core meets it"""

    def __init__(self, ny: int, nx: int, size: int):
        # a multiple of the core puts every lattice tile in exactly one column (only the shifted last one
        # straddles), so no tile runs twice; even keeps the tracing pooling aligned
        if size % CORE or size % 2:
            raise ValueError(f"column size must be an even multiple of {CORE}")
        self.py, self.px = positions(ny), positions(nx)
        self.cols = [(y0, min(y0 + size, ny), x0, min(x0 + size, nx))
                     for y0 in range(0, ny, size) for x0 in range(0, nx, size)]

    def tiles(self, col, pz_all):
        y0, y1, x0, x1 = col
        ys = [p for p in self.py if core_range(p)[0] < y1 and core_range(p)[1] > y0]
        xs = [p for p in self.px if core_range(p)[0] < x1 and core_range(p)[1] > x0]
        return [(pz, py, px) for pz in pz_all for py in ys for px in xs]
