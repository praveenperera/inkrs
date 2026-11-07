"""villa's own CPU scheduling for one map: the tiles villa 0fb38c45 runs and the pixels it covers

    python -I scripts/villa_coverage.py ZARR STRIDE S [--reverse] [--json OUT] [--patch 384]

Everything here is villa's code, called the way `infer.py` calls it, with no model:
  - the input opened as villa opens it (`open_volume_root`, `select_volume_level` at "0",
    `_volume_axes`)
  - the first skip: `build_lowres_block_mask` (pyramid level 3 of a group, or the nearest level;
    none for a bare array) and `iter_blocks`
  - the layers: `select_layer_indices` for --layer-start S --layer-end S+17 and the direction
  - the raw-window skip: `FlatPatchReader.read(...).any()`, as `FlatBlockDataset` measures it
  - coverage: villa's `ChunkAccumulator` fed each kept tile with the floored Hann window
    (`compute_importance_map_2d`) and zero probabilities, then villa's finish rule
    `weight_sum > 1e-6`
The reader's array is the level-0 volume read once into memory; `_read_raw` indexes it exactly as
it indexes the zarr array. Prints JSON: the kept tiles in villa's order and the covered count.
`--patch` is the reader's tile edge: 128, or 384 for Reader V3. Importable: `Input(path, patch)` then
`.map(s, reverse, stride)`
"""

import argparse
import json
import os
import sys
from pathlib import Path

# villa 0fb38c45: the installed package, or the vesuvius/src of a checkout named by VILLA_SRC
if os.environ.get("VILLA_SRC"):
    sys.path.insert(0, os.environ["VILLA_SRC"])

import numpy as np  # noqa: E402
from vesuvius.ink_detection.inference.infer import (  # noqa: E402
    ChunkAccumulator,
    FlatPatchReader,
    _volume_axes,
    build_lowres_block_mask,
    compute_chunk_contribution_counts,
    compute_importance_map_2d,
    iter_blocks,
    select_layer_indices,
)
from vesuvius.ink_detection.inference.infer import open_volume_root, select_volume_level  # noqa: E402

DEPTH = 17
# hybrid_3d2d-seed42's preprocessing (the export sidecar); the any() test reads raw voxels either way
PREPROCESSING = "tifxyz_robust"


class Input:
    """One input zarr, opened and read once, with villa's tile schedule per stride"""

    def __init__(self, path, patch=128):
        self.path = str(path)
        self.patch = int(patch)
        self.root = open_volume_root(self.path)
        self.volume = select_volume_level(self.root, "0", source=self.path)
        self.depth_first, self.depth, self.height, self.width, _, _ = _volume_axes(self.volume)
        self.data = np.asarray(self.volume[:])
        self.occupancy, self.scale, self.level = build_lowres_block_mask(self.root, height=self.height, width=self.width, user_mask=None)
        self._blocks = {}

    def blocks(self, stride):
        if stride not in self._blocks:
            self._blocks[stride] = iter_blocks((self.height, self.width), self.patch, stride, self.occupancy, self.scale)
        return self._blocks[stride]

    def layers(self, s, reverse):
        return select_layer_indices(self.depth, layer_start=s, layer_end=s + DEPTH, output_depth=DEPTH, direction="reverse" if reverse else "forward")

    def map(self, s, reverse, stride):
        """villa's kept tiles `[(y0, x0)]` for the map at layer start `s`, and its coverage mask"""
        blocks = self.blocks(stride)
        reader = FlatPatchReader(
            input_path=self.path, resolution="0", depth_axis_first=self.depth_first, height=self.height, width=self.width,
            layer_indices=self.layers(s, reverse), output_depth=DEPTH, preprocessing=PREPROCESSING,
        )
        reader._array = self.data
        p = self.patch
        kept = [b for b in blocks if reader.read(b.y0, b.x0, p, p).any()]
        h, w = self.height, self.width
        chunks = (min(p, h), min(p, w))
        prob = np.zeros((h, w), np.float32)
        weight = np.zeros((h, w), np.float32)
        acc = ChunkAccumulator(shape=(h, w), chunk_shape=chunks, prob_sum_store=prob, weight_sum_store=weight,
                               contribution_counts=compute_chunk_contribution_counts(blocks, chunk_shape=chunks))
        window = compute_importance_map_2d(patch_size=(p, p), mode="hann").numpy()
        for b in kept:
            acc.add_tile(y0=b.y0, x0=b.x0, tile=np.zeros((b.valid_h, b.valid_w), np.float32), tile_weights=window[: b.valid_h, : b.valid_w])
        acc.flush_remaining()
        return [(b.y0, b.x0) for b in kept], weight > 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr", type=Path)
    ap.add_argument("stride", type=int)
    ap.add_argument("s", type=int)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--patch", type=int, default=128, choices=(128, 384))
    a = ap.parse_args()
    inp = Input(a.zarr, a.patch)
    tiles, covered = inp.map(a.s, a.reverse, a.stride)
    out = dict(level=inp.level, scale=list(inp.scale), scheduled=len(inp.blocks(a.stride)), tiles=tiles, covered=int(covered.sum()),
               layers=[int(v) for v in inp.layers(a.s, a.reverse)])
    text = json.dumps(out)
    if a.json:
        a.json.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
