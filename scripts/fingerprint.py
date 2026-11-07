"""SHA-256 of a zarr volume's level-0 voxels (C order, uint8), independent of its chunking and compression

    python -I scripts/fingerprint.py VOLUME.zarr
"""

import hashlib
import sys

import numpy as np
import zarr

node = zarr.open(sys.argv[1], mode="r")
arr = node["0"] if hasattr(node, "keys") and "0" in node else node
h = hashlib.sha256()
for z in range(arr.shape[0]):
    h.update(np.ascontiguousarray(arr[z]).tobytes())
print(f"{h.hexdigest()}  {list(arr.shape)} {arr.dtype}")
