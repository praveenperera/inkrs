"""Band mode end to end on the CPU: a CT with partial batches, air tiles, a column whose tiles are all air, voxels
no tile covers and missing chunks writes arrays of the right shape that neither the reader count nor a resumed run
changes, and a run with other settings refuses an existing output
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import zarr
from conftest import python_cmd, run

Z_TOTAL, NY, NX = 128, 200, 400
Z0, Z1 = 32, 96


@pytest.fixture(scope="module")
def ct(tmp_path_factory) -> Path:
    """A uint8 CT with structure, an air slab (x >= 270) and missing chunks, chunked like a scroll (nested, raw)"""
    d = tmp_path_factory.mktemp("ct") / "ct.zarr"
    arr = zarr.open_array(str(d), mode="w", shape=(Z_TOTAL, NY, NX), chunks=(64, 64, 64), dtype="u1",
                          compressor=None, fill_value=0, dimension_separator="/")
    rng = np.random.default_rng(3)
    z, y, x = np.meshgrid(np.arange(Z_TOTAL), np.arange(NY), np.arange(NX), indexing="ij")
    v = 120 + 60 * np.sin(x / 7.0 + z / 11.0) * np.cos(y / 5.0) + rng.normal(0, 25, x.shape)
    v = np.clip(v, 1, 255).astype(np.uint8)
    v[:, :, 270:] = 0
    # a dim corner, so normalize_ct's std floor of 10 is reached in some tiles
    v[:, :60, :80] = 3 + (x[:, :60, :80] % 2)
    arr[:] = v
    # whole chunks of the band left unwritten read as fill 0
    arr[64:128, 128:192, 192:256] = 0
    return d


def common(ct: Path, snap: Path, out: Path, extra=()) -> list[str]:
    return ["--ct", str(ct), "--z0", str(Z0), "--z1", str(Z1), "--model-dir", str(snap), "--out", str(out),
            "--column", "192", "--batch", "3", "--device", "cpu", *extra]


def afv_run(ct, snap, out, extra=()):
    return run(python_cmd() + ["-m", "inkrs_afv"] + common(ct, snap, out, extra))


def records(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines() if line]


def chunk_keys(out: Path, name: str) -> list[str]:
    root = out / name
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))


def assert_same_output(ref: Path, got: Path) -> None:
    for name in ("semantic", "tracing"):
        a, b = zarr.open_array(str(ref / name), mode="r"), zarr.open_array(str(got / name), mode="r")
        assert a.shape == b.shape and a.chunks == b.chunks and a.dtype == b.dtype
        assert a.compressor == b.compressor
        av, bv = a[:], b[:]
        assert np.array_equal(av, bv), f"{name}: {int((av != bv).sum())} voxels differ"
        assert chunk_keys(ref, name) == chunk_keys(got, name)
    assert json.loads((ref / ".zattrs").read_text()) == json.loads((got / ".zattrs").read_text())
    keys = ("col", "tiles", "tiles_run", "tiles_air", "voxels_uncovered")
    rr = sorted(tuple(json.dumps(r[k]) for k in keys) for r in records(ref))
    gr = sorted(tuple(json.dumps(r[k]) for k in keys) for r in records(got))
    assert rr == gr


@pytest.fixture(scope="module")
def full(ct, stub_snapshot, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("full") / "pred.zarr"
    afv_run(ct, stub_snapshot, out, ["--readers", "3"])
    return out


def test_output_covers_the_edge_cases(full):
    """The fixture covers what the comparisons must cover; a vacuous output would make every test pass"""
    recs = records(full)
    assert len(recs) == 6
    assert any(r["tiles_run"] == 0 and r["tiles_air"] > 0 for r in recs), "no all-air column"
    assert any(r["tiles_run"] > 0 and r["tiles_air"] > 0 for r in recs), "no mixed column"
    assert any(r["tiles_run"] % 3 for r in recs), "no partial batch"
    assert any(r["tiles_run"] > 0 and r["voxels_uncovered"] > 0 for r in recs), "no uncovered voxels in a run column"
    sem = zarr.open_array(str(full / "semantic"), mode="r")
    tra = zarr.open_array(str(full / "tracing"), mode="r")
    assert sem.shape == (2, Z1 - Z0, NY, NX) and sem.dtype == np.uint8
    assert tra.shape == (2, (Z1 - Z0) // 2, NY // 2, NX // 2) and tra.dtype == np.uint8
    s = sem[:]
    assert 0 < np.count_nonzero(s) < s.size
    assert len(np.unique(s)) > 50
    summary = json.loads(next(full.glob("summary-*.json")).read_text())
    assert summary["complete"] and summary["columns_done"] == 6
    assert summary["precision"] == "fp32"


def test_reader_count_leaves_the_output_unchanged(ct, stub_snapshot, full, tmp_path):
    out = tmp_path / "afv.zarr"
    afv_run(ct, stub_snapshot, out, ["--readers", "1"])
    assert_same_output(full, out)


def test_resume_from_a_partial_output(ct, stub_snapshot, full, tmp_path):
    out = tmp_path / "afv.zarr"
    afv_run(ct, stub_snapshot, out, ["--only-columns", "1,3"])
    first = records(out)
    assert sorted(tuple(r["col"]) for r in first) == [(0, 192, 192, 384), (192, 200, 0, 192)]
    afv_run(ct, stub_snapshot, out, ["--readers", "2"])
    assert len(records(out)) == 6
    assert_same_output(full, out)
    assert len((out / "attempts.jsonl").read_text().splitlines()) == 2


def test_changed_provenance_is_refused(ct, stub_snapshot, tmp_path):
    out = tmp_path / "afv.zarr"
    afv_run(ct, stub_snapshot, out, ["--only-columns", "0"])
    p = subprocess.run(python_cmd() + ["-m", "inkrs_afv"] + common(ct, stub_snapshot, out, ["--precision", "bf16"]),
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True, timeout=600,
                       check=False)
    assert p.returncode != 0 and "different provenance" in p.stderr
