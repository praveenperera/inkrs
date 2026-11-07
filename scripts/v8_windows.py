"""Write the v8-in stitching windows inkrs embeds: `gkern(64, nsig) / max` as float32, for nsig 1 (the
release) and 3 (the 306B runner), with the model repository's own `gkern` (ink8um/inference.py)

    python -I scripts/v8_windows.py V8IN_DIR OUT_DIR

Writes OUT_DIR/v8-gauss1.f32 and OUT_DIR/v8-gauss3.f32 (4096 little-endian float32 each) and prints their
sha256, which `src/v8in.rs`'s test pins
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])
from ink8um.inference import gkern  # noqa: E402

out = Path(sys.argv[2])
for nsig in (1, 3):
    # both runners: (g / g.max()).astype(np.float32)
    g = gkern(64, nsig)
    w = (g / g.max()).astype(np.float32)
    assert w.shape == (64, 64) and w.dtype == np.float32
    data = w.astype("<f4").tobytes()
    (out / f"v8-gauss{nsig}.f32").write_bytes(data)
    print(nsig, hashlib.sha256(data).hexdigest(), w.min(), w.max())
