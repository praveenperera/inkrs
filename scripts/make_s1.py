"""Writes S1, the fixed small input of the timing and parity checks: the 2048 x 2048 crop of a PHerc. 343
render's level 0 at rows 256:2304 and columns 2048:4096, all 66 layers, as a zarr v2 group with array `0`

    python -I scripts/make_s1.py RENDER.zarr S1.zarr

RENDER.zarr is the concat_w047-w048_R5B2_z9500-11000 render (a group with array `0`, or the array itself).
scripts/fingerprint.py should print, for S1,
a5181baf5bd4c5f0537e9bc9c3909b91fd72bbffd8c3d5d808f02ca3debf3b9d  [66, 2048, 2048] uint8
"""

import sys

import zarr
from numcodecs import Zstd

src, dest = sys.argv[1], sys.argv[2]
node = zarr.open(src, mode="r")
arr = node["0"] if hasattr(node, "keys") and "0" in node else node
crop = arr[:, 256:2304, 2048:4096]
group = zarr.open_group(dest, mode="w", zarr_format=2)
group.create_array("0", data=crop, chunks=(crop.shape[0], 256, 256), compressors=Zstd(level=0), filters=None)
print(dest, crop.shape, crop.dtype)
