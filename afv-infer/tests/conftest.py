"""Shared fixtures for the inkrs-afv tests: a stub AFV snapshot and a way to run the command

The tests run the installed ``inkrs_afv`` (or, with ``PYTHONPATH=python``, the source tree) in an environment with
torch, zarr 2 and numcodecs:

    pip install dist/inkrs_afv-*.whl pytest && python -m pytest tests

Everything runs on the CPU in seconds. ``test_villa.py`` also compares villa mode with villa's own inference when
``INKRS_AFV_VILLA_PYTHON`` and ``INKRS_AFV_VILLA_SRC`` name a villa environment, and skips that test otherwise
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# a snapshot whose load_model is a small fixed-seed 3D conv net with AFV's output heads, so a CPU run takes seconds,
# and whose normalize_ct has the shape of the real snapshot's (per-tile mean and std, std floored at 10)
STUB_MODEL = '''
import json
from pathlib import Path

import torch
from torch import nn


class Stub(nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.a = nn.Conv3d(1, 4, 3, padding=1)
        self.b = nn.Conv3d(4, 4, 3, padding=1)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)

    def forward(self, x):
        h = self.b(torch.nn.functional.leaky_relu(self.a(x), 0.01))
        return {"seg": h * 3, "centre": h[:, :1]}


def load_model(folder, device="cpu"):
    json.loads((Path(folder) / "plans.json").read_text())
    return Stub().to(device).eval()


def normalize_ct(ct, mean=None, std=None):
    ct = ct.float()
    mean = ct.mean(dim=(2, 3, 4), keepdim=True)
    std = ct.std(dim=(2, 3, 4), keepdim=True)
    return (ct - mean) / std.clamp_min(10.0)
'''


@pytest.fixture(scope="session")
def stub_snapshot(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("stub-snapshot")
    (d / "model.py").write_text("import torch\n" + STUB_MODEL)
    (d / "plans.json").write_text("{}")
    (d / "checkpoint_final.pth").write_bytes(b"stub")
    return d


def python_cmd() -> list[str]:
    return [sys.executable, "-B"]


def run(cmd: list[str], extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a command with small thread pools and fail the test with its output if it fails"""
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               NUMEXPR_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    env.update(extra_env or {})
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900, check=False)
    assert p.returncode == 0, f"{cmd[:4]} failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
    return p
