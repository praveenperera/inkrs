"""inkrs-afv: AFV v2 fibre probabilities (Qualzz20/afv_fiber_9um_v2) for a CT, with the host work moved off the
critical path

    inkrs-afv --mode villa --ct CT_L0_ARRAY --model-dir SNAPSHOT --out OUT.zarr [--device cuda]
    inkrs-afv --ct CT_L0_ARRAY --z0 Z0 --z1 Z1 --model-dir SNAPSHOT --out OUT.zarr [--batch 4] [--max-minutes M]
              [--expect-sha256 SHA] [--readers 4] [--precision fp32|bf16|fp16] [--only-columns I,J,...]
    PYTHONPATH=<this package's parent> python -m inkrs_afv ...

Both modes write the probability zarr of villa ``lasagna/qualzz_inference.py``: ``semantic`` (2, Z, Y, X) and
``tracing`` (2, Z/2, Y/2, X/2) uint8, plus ``attempts.jsonl``, ``progress.jsonl`` and ``summary-*.json`` records, so
a capped run resumes. They tile, normalise, blend and quantise as villa does and differ in what they run:

- ``--mode villa`` runs the whole array as villa main does: one tile per forward, and villa's rule for which tiles
  run and which output chunks are written, from the input's stored chunks (``villa.py``)
- band mode (the default) runs a z band [z0, z1), reflected at the band's own edges, four tiles per forward, and
  skips every tile whose input is all zero as air

At ``--precision fp32`` (the default) the network, the blend and the finalise use the same float32 arithmetic as
the reference in each mode (``finalize.py``). ``--precision bf16|fp16`` autocasts the network only and changes the
output.

The module is ``inkrs_afv``, not ``inkrs.afv``: it runs in the model's own Python environment, without inkrs's
native extension
"""

from __future__ import annotations

from .run import main

__all__ = ["main"]
