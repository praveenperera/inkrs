"""``inkrs-afv --mode villa`` against villa main's ``python -m lasagna.qualzz_inference``, end to end on the CPU

Both run the stub snapshot (conftest.py, with the real ``normalize_ct``) on a synthetic whole-volume CT whose shape
is odd on every axis and whose 64³ input chunks are not the 128³ output chunks. The CT has deleted chunk files
(including whole 128³ output chunks with no stored input, and output chunks with only part of their input stored),
a stored all-zero region (air that villa still runs) and structure everywhere else. The semantic and tracing
arrays must match bit for bit, reading absent chunks as 0.

villa runs under its own environment: INKRS_AFV_VILLA_PYTHON (a Python with torch and villa's requirements) and
INKRS_AFV_VILLA_SRC (a checkout of https://github.com/ScrollPrize/villa, with ``lasagna/`` at its root). The test
skips when either is unset or missing
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import zarr
from conftest import python_cmd, run

VILLA_PYTHON = os.environ.get("INKRS_AFV_VILLA_PYTHON")
VILLA_SRC = os.environ.get("INKRS_AFV_VILLA_SRC")
SHAPE = (157, 301, 299)


@pytest.fixture(scope="module")
def sparse_ct(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("villa-ct") / "ct.zarr"
    arr = zarr.open_array(str(d), mode="w", shape=SHAPE, chunks=(64, 64, 64), dtype="u1", compressor=None,
                          fill_value=0, dimension_separator="/", write_empty_chunks=True)
    rng = np.random.default_rng(11)
    z, y, x = np.meshgrid(*(np.arange(n) for n in SHAPE), indexing="ij")
    v = 120 + 60 * np.sin(x / 6.0 + z / 9.0) * np.cos(y / 5.0) + rng.normal(0, 25, x.shape)
    v = np.clip(v, 1, 255).astype(np.uint8)
    # stored zeros: the tiles at padded y 192 and 205, x 0 read only zeros, which villa still runs (band mode would
    # skip them as air)
    v[:, 160:, :200] = 0
    arr[:] = v
    # absent input: the whole 128³ output chunk (1, 0, 2) has nothing stored under it (input chunks 2-3, 0-1, 4)
    # and output chunk (0, 1, 1) keeps only input chunk (1, 3, 2) of its eight
    gone = [(iz, iy, 4) for iz in (2,) for iy in (0, 1)]
    gone += [(iz, iy, ix) for iz in (0, 1) for iy in (2, 3) for ix in (2, 3) if (iz, iy, ix) != (1, 3, 2)]
    gone += [(0, 4, 0), (2, 4, 4)]
    for c in gone:
        (d / "/".join(map(str, c))).unlink()
    return d


def test_villa_mode_matches_villa_main(sparse_ct, stub_snapshot, tmp_path):
    if not VILLA_PYTHON or not VILLA_SRC or not (Path(VILLA_SRC) / "lasagna" / "qualzz_inference.py").exists():
        pytest.skip("set INKRS_AFV_VILLA_PYTHON and INKRS_AFV_VILLA_SRC to compare with villa")
    villa_out, afv_out = tmp_path / "villa" / "pred.zarr", tmp_path / "afv" / "pred.zarr"
    villa_out.parent.mkdir()
    env = {"PYTHONPATH": VILLA_SRC}
    run([VILLA_PYTHON, "-B", "-m", "lasagna.qualzz_inference", "--model-dir", str(stub_snapshot),
         "--input", str(sparse_ct), "--output", str(villa_out), "--voxel-size-um", "9.362", "--device", "cpu",
         "--flush-workers", "1", "--accumulator-workers", "1"], extra_env=env)
    run(python_cmd() + ["-m", "inkrs_afv", "--mode", "villa", "--ct", str(sparse_ct), "--model-dir",
                        str(stub_snapshot), "--out", str(afv_out), "--column", "192", "--device", "cpu"])

    villa, afv = zarr.open_group(str(villa_out), mode="r"), zarr.open_group(str(afv_out), mode="r")
    for name in ("semantic", "tracing"):
        a, b = afv[name][:], villa[name][:]
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert np.array_equal(a, b), f"{name}: {int((a != b).sum())} of {a.size} values differ"
    sem = afv["semantic"][:]
    # the CT exercises what it claims: an unsupported output chunk is zero, stored air produced values
    assert not sem[:, 128:, :128, 256:].any()
    assert sem[:, :, 192:, :96].any()
    assert afv.attrs["inference"]["mode"] == "villa"
    assert "drop_in_for" not in afv.attrs["inference"]


def test_villa_mode_refuses_other_batches_and_bands(sparse_ct, stub_snapshot, tmp_path):
    import subprocess

    base = python_cmd() + ["-m", "inkrs_afv", "--mode", "villa", "--ct", str(sparse_ct), "--model-dir",
                           str(stub_snapshot), "--device", "cpu"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for extra in (["--batch", "4"], ["--z0", "0", "--z1", "64"]):
        out = tmp_path / ("o" + str(len(extra)) + extra[0])
        p = subprocess.run(base + ["--out", str(out), *extra], env=env, capture_output=True, text=True, timeout=300,
                           check=False)
        assert p.returncode != 0, extra
        shutil.rmtree(out, ignore_errors=True)
