"""One column's blend, normalisation, quantisation and pooling on the inference device, in the reference float32 order

A host-side predictor accumulates in numpy and finalises a column on the CPU: 17–19 s of ``quantize`` and 1.1–1.4 s
of ``pool2`` per non-air column, against 6.3 s of forward for a full one (CPU profile of the band reference). Here
the same arithmetic runs on the device. Every step is an elementwise IEEE float32 operation applied in the
reference's order, so the result is bit-identical:
- accumulate: per tile ``acc += p * w`` then ``wsum += w``, as two separate rounded operations (torch eager never
  fuses them into an FMA), over the tiles in the column's order
- normalise: ``acc / wsum`` where ``wsum > 0``; elsewhere ``acc`` is divided by 1, which leaves it unchanged
- quantise: reject non-finite values, then ``rint(clip(p, 0, 1) * 255)``; ``torch.round`` rounds half to even like
  ``np.rint``
- tracing: the reference ``_pool2``, eight ordered additions from zero then ``/ 8``, before quantisation

``tests/afv/test_finalize.py`` checks each step against the reference functions on adversarial values (exact
rounding ties, out-of-range and tiny probabilities, odd shapes)
"""

from __future__ import annotations

import itertools

import numpy as np

from .contract import BORDER, TILE


def quantize(torch, p):
    """The reference ``quantize`` (villa qualzz_inference.py:83-86) on a device tensor, returning uint8 there"""
    if not bool(torch.isfinite(p).all()):
        raise ValueError("non-finite probabilities")
    q = p.clamp(0, 1)
    q.mul_(255)
    q.round_()
    return q.to(torch.uint8)


def pool2(torch, p):
    """The reference ``_pool2`` on a device tensor: (2, Z, Y, X) -> (2, Z//2, Y//2, X//2)"""
    shape = tuple(n // 2 for n in p.shape[1:])
    out = torch.zeros((2, *shape), dtype=torch.float32, device=p.device)
    for z, y, x in itertools.product((0, 1), repeat=3):
        out += p[:, z:shape[0] * 2:2, y:shape[1] * 2:2, x:shape[2] * 2:2]
    return out / 8


class ColumnAccumulator:
    """Weighted tile probabilities over one column [y0, y1) x [x0, x1) of a band nz voxels tall, on ``device``"""

    def __init__(self, torch, device, nz: int, col: tuple[int, int, int, int], w3):
        y0, y1, x0, x1 = col
        self.torch = torch
        self.y0, self.x0 = y0, x0
        self.shape = (nz, y1 - y0, x1 - x0)
        self.acc = torch.zeros((2, *self.shape), dtype=torch.float32, device=device)
        self.wsum = torch.zeros(self.shape, dtype=torch.float32, device=device)
        self.w3 = w3

    def add(self, pz: int, py: int, px: int, p) -> None:
        """Blend one tile's (2, 128, 128, 128) probabilities in; the index arithmetic is the reference ``add``"""
        # tile voxel i sits at volume coordinate p_axis - BORDER + i; clip to the column
        lo = np.array([pz - BORDER, py - BORDER - self.y0, px - BORDER - self.x0])
        cl = np.maximum(lo, 0)
        ch = np.minimum(lo + TILE, self.shape)
        if (ch <= cl).any():
            return
        s = tuple(slice(int(a - b), int(c - b)) for a, b, c in zip(cl, lo, ch))
        d = tuple(slice(int(a), int(c)) for a, c in zip(cl, ch))
        w = self.w3[s]
        self.acc[(slice(None),) + d] += p[(slice(None),) + s] * w
        self.wsum[d] += w

    def finish(self, quantized: bool):
        """Normalise the column; returns (semantic uint8, tracing uint8, uncovered voxel count) on the host, with
        the two arrays None when ``quantized`` is false (a column whose tiles were all air writes nothing)"""
        torch = self.torch
        covered = self.wsum > 0
        uncovered = int(covered.numel() - int(covered.sum()))
        if not quantized:
            return None, None, uncovered
        self.acc.div_(torch.where(covered, self.wsum, torch.ones((), dtype=torch.float32, device=covered.device)))
        del covered, self.wsum
        sem = quantize(torch, self.acc).cpu().numpy()
        tra = quantize(torch, pool2(torch, self.acc)).cpu().numpy()
        del self.acc
        return sem, tra, uncovered
