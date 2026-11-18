"""``--mode villa``: which tiles run and which output chunks are written, as villa's ``lasagna.qualzz_inference`` does

villa main (97fb9336c) runs AFV V2 over a whole CT array with ``lasagna.tiled_predict3d.run_tiled_inference_3d``.
It decides what to compute from what the input stores, not from its values:

- an output chunk (128³ on the volume's own lattice, ``chunk_size``) is *supported* when any input chunk inside its
  footprint, clipped to the volume, is stored (``_chunk_supported``, tiled_predict3d.py:3317-3341, through
  ``_input_has_chunks`` :841-853)
- a tile runs when its input footprint (the 128³ tile at padded origin p covers volume [p - 16, p + 112), clipped)
  has a stored input chunk and some output chunk meeting that footprint is supported (``_tile_work`` :3402-3431,
  ``_needed_chunks`` :3343-3369); an all-zero tile over stored chunks still runs
- tiles add only into supported chunks and unsupported chunks are never written, so they read as the fill value 0
  (``_accumulate_group`` :3433-3466, qualzz_inference.py ``ProbabilityZarrOutputAdapter``)

The network sees one tile per forward there (the serial and worker paths default ``max_batch_size`` to 1,
tiled_predict3d.py:2880, and qualzz_inference never sets it), so this mode runs batches of one: CPU convolutions round
differently at other batch sizes. Tiling, padding, normalisation, softmax, blend and quantisation are already the
same in both modes (contract.py, finalize.py)
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from .contract import BORDER, TILE

CHUNK = 128
REFERENCE = "villa lasagna.qualzz_inference @ 97fb9336c (batch 1, villa's storage-based tile skip)"
SKIP_RULE = ("villa: a tile runs when its input footprint holds a stored input chunk and meets a supported output "
             "chunk; unsupported output chunks are not written")


def stored_chunks(ct: Path) -> tuple[tuple[int, int, int], set[tuple[int, int, int]]]:
    """The input array's chunk shape and the indices of its stored chunks, from the chunk files on disk"""
    meta = json.loads((ct / ".zarray").read_text())
    chunks = tuple(int(c) for c in meta["chunks"])
    if meta.get("dimension_separator") == "/":
        names = ("/".join(f.relative_to(ct).parts) for f in ct.rglob("*") if f.is_file())
        sep = "/"
    else:
        names = (f.name for f in ct.iterdir() if f.is_file())
        sep = "."
    stored = set()
    for name in names:
        parts = name.split(sep)
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            stored.add(tuple(int(p) for p in parts))
    return chunks, stored


class VillaSelection:
    """villa's supported output chunks and running tiles for one whole-volume input"""

    def __init__(self, ct: Path, shape: tuple[int, int, int]):
        self.shape = tuple(int(n) for n in shape)
        self.in_chunks, self.stored = stored_chunks(ct)
        grid = [-(-n // CHUNK) for n in self.shape]
        self.supported = np.zeros(grid, dtype=bool)
        for c in itertools.product(*(range(g) for g in grid)):
            lo = [i * CHUNK for i in c]
            hi = [min(n, l + CHUNK) for n, l in zip(self.shape, lo)]
            self.supported[c] = self._has_input(lo, hi)

    def _has_input(self, lo, hi) -> bool:
        """Whether any stored input chunk meets the half-open region [lo, hi) (``_input_has_chunks``)"""
        ranges = [range(max(0, a // c), (b + c - 1) // c) for a, b, c in zip(lo, hi, self.in_chunks)]
        return any(i in self.stored for i in itertools.product(*ranges))

    def tile_runs(self, pz: int, py: int, px: int) -> bool:
        """Whether villa runs the tile at padded origin (pz, py, px)"""
        lo = [max(0, p - BORDER) for p in (pz, py, px)]
        hi = [min(n, p - BORDER + TILE) for n, p in zip(self.shape, (pz, py, px))]
        if any(b <= a for a, b in zip(lo, hi)) or not self._has_input(lo, hi):
            return False
        sel = tuple(slice(a // CHUNK, (b - 1) // CHUNK + 1) for a, b in zip(lo, hi))
        return bool(self.supported[sel].any())

    def mask_column(self, col, sem: np.ndarray, tra: np.ndarray) -> None:
        """Zero, in place, a column's semantic (2, Z, y1-y0, x1-x0) and tracing values in unsupported chunks"""
        y0, y1, x0, x1 = col
        for cz, cy, cx in np.argwhere(~self.supported):
            ys, ye = max(y0, cy * CHUNK), min(y1, (cy + 1) * CHUNK)
            xs, xe = max(x0, cx * CHUNK), min(x1, (cx + 1) * CHUNK)
            if ye <= ys or xe <= xs:
                continue
            sem[:, cz * CHUNK:(cz + 1) * CHUNK, ys - y0:ye - y0, xs - x0:xe - x0] = 0
            # the tracing chunk of a native chunk is the same index at half size; columns start on even rows
            h = CHUNK // 2
            tys, tye = max(y0 // 2, cy * h), min(y0 // 2 + tra.shape[2], (cy + 1) * h)
            txs, txe = max(x0 // 2, cx * h), min(x0 // 2 + tra.shape[3], (cx + 1) * h)
            if tye > tys and txe > txs:
                tra[:, cz * h:(cz + 1) * h, tys - y0 // 2:tye - y0 // 2, txs - x0 // 2:txe - x0 // 2] = 0
