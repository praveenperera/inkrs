"""The ``inkrs-afv`` command: arguments, outputs and resume behaviour, on the overlapped pipeline"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np

from .contract import (
    BORDER,
    DROP_IN_FOR,
    MODEL_FILES,
    OVERLAP,
    REFERENCE,
    TILE,
    Columns,
    blend_weights,
    ct_identity,
    load_snapshot,
    positions,
    prediction_identity,
    sha256,
)
from .finalize import ColumnAccumulator
from .pipeline import ColumnEnd, ColumnWriter, TileFeed
from .villa import REFERENCE as VILLA_REFERENCE
from .villa import SKIP_RULE as VILLA_SKIP_RULE
from .villa import VillaSelection

# what each --precision writes to the provenance and the summary; fp32 is the band reference's own wording, so a
# band-mode output carries the reference's precision record
# torch 2.14's defaults run fp32 convolutions in TF32 (cudnn.conv.fp32_precision "tf32"), so the reference contract
# is TF32 convs; fp32-ieee turns TF32 off everywhere, for the reference's own precision noise band
PRECISION = {
    "fp32": ("fp32 (torch defaults)", "fp32", None),
    "fp32-ieee": ("fp32, TF32 off (torch.backends.fp32_precision ieee)", "fp32-ieee", None),
    "bf16": ("bf16 autocast (network only; normalize_ct, softmax and blend fp32)", "bf16", "bfloat16"),
    "fp16": ("fp16 autocast (network only; normalize_ct, softmax and blend fp32)", "fp16", "float16"),
}


# the band reference's forward_s is host time around each batch (upload, forward, softmax, download, with syncs) and its
# columns run one after another, so its wall_s sum is the run's column time. Here neither holds, so the summary
# says what the same fields measure
TIMING = dict(
    forward_s="device time of normalize_ct, network and softmax per batch (CUDA events); excludes uploads, blend "
              "and finalise, which the band reference's forward_s partly includes",
    wall_s="from the column's first batch to its write finishing; columns overlap, so column_wall_s (their sum) "
           "exceeds the run's time and is not comparable with the band reference's",
    this_run_wall_s="end to end, the comparable figure",
)


def code_hashes() -> dict[str, str]:
    """sha256 of every module of this package, for the output's provenance"""
    here = Path(__file__).resolve().parent
    return {f"inkrs_afv/{p.name}": sha256(p) for p in sorted(here.glob("*.py"))}


def parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="inkrs-afv", description=__doc__)
    ap.add_argument("--ct", type=Path, required=True, help="level-0 uint8 zarr v2 array (the /0 directory)")
    ap.add_argument("--mode", choices=("band", "villa"), default="band",
                    help="band: a z band [z0, z1) in batches of --batch, all-zero tiles skipped (the default); "
                         "villa: the whole array as villa's lasagna.qualzz_inference computes it (batch 1, "
                         "villa's storage-based tile skip)")
    ap.add_argument("--z0", type=int, help="first plane of the band (band mode; villa mode takes the whole array)")
    ap.add_argument("--z1", type=int, help="end plane of the band, exclusive")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--column", type=int, default=480)
    ap.add_argument("--batch", type=int, default=None, help="tiles per forward (band mode, default 4; villa mode: 1)")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop between columns after this long; 0 = no cap")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--expect-sha256", help="checkpoint sha256 the run must load")
    # inkrs-afv only; none of these changes the output at fp32
    ap.add_argument("--readers", type=int, default=4, help="tile reader threads (output unchanged)")
    ap.add_argument("--precision", choices=sorted(PRECISION), default="fp32",
                    help="fp32 is the reference contract (torch defaults); fp32-ieee, bf16 and fp16 change the output")
    ap.add_argument("--only-columns", default=None,
                    help="comma list of column indices (in column order) to run; for timing subsets")
    a = ap.parse_args(argv)
    if a.mode == "band":
        if a.z0 is None or a.z1 is None:
            ap.error("band mode needs --z0 and --z1")
        a.batch = 4 if a.batch is None else a.batch
    elif a.batch not in (None, 1):
        # villa forwards one tile at a time, and CPU convolutions round differently at other batch sizes
        ap.error("villa mode runs batches of 1; leave --batch out or pass 1")
    else:
        a.batch = 1
    if a.batch < 1 or a.readers < 1:
        raise SystemExit("inkrs-afv: --batch and --readers must be at least 1")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse(argv)

    import numcodecs
    import torch
    import zarr

    t_start = time.time()
    # every attempt is recorded at its start, before the model loads, so a capped or killed run still shows it
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "attempts.jsonl").open("a") as f:
        f.write(json.dumps(dict(attempt=t_start, start=t_start)) + "\n")
    ct = zarr.open_array(str(a.ct), mode="r")
    if ct.dtype != np.uint8 or ct.ndim != 3:
        raise ValueError(f"expected a 3D uint8 CT array, got {ct.dtype} {ct.shape}")
    villa = a.mode == "villa"
    if villa:
        # the whole array is the volume, so tiles reflect at its own edges, as villa's do
        if (a.z0, a.z1) not in ((None, None), (0, ct.shape[0])):
            raise ValueError(f"villa mode runs the whole array: --z0/--z1 must be left out or be 0 {ct.shape[0]}")
        a.z0, a.z1 = 0, int(ct.shape[0])
    elif (a.z1 - a.z0) % 2:
        raise ValueError("band height must be even")
    hashes = {n: sha256(a.model_dir / n) for n in MODEL_FILES}
    if a.expect_sha256 and hashes["checkpoint_final.pth"] != a.expect_sha256:
        raise ValueError(f"checkpoint sha256 {hashes['checkpoint_final.pth']} != {a.expect_sha256}")
    if a.precision == "fp32-ieee":
        # the global setting, so every backend agrees (mixing per-backend and legacy flags raises in torch 2.14)
        torch.backends.fp32_precision = "ieee"
        if torch.backends.cudnn.conv.fp32_precision != "ieee":
            raise RuntimeError("fp32-ieee: cuDNN conv precision did not follow torch.backends.fp32_precision")
    device = torch.device(a.device)
    model, normalize = load_snapshot(a.model_dir, device)
    label, short, amp_dtype = PRECISION[a.precision]

    nz, ny, nx = a.z1 - a.z0, ct.shape[1], ct.shape[2]
    provenance = dict(model_sha256=hashes, code_sha256=code_hashes(), ct=str(a.ct.resolve()),
                      ct_identity=ct_identity(a.ct, a.z0, a.z1), z0=a.z0, z1=a.z1, tile=TILE, overlap=OVERLAP,
                      border=BORDER, column=a.column, batch=a.batch, precision=label,
                      normalization="model.py:normalize_ct per tile", air_tiles="skipped (all-zero input)",
                      torch=torch.__version__, cuda=torch.version.cuda, channel_order="VH", reference=REFERENCE,
                      engine="inkrs-afv", drop_in_for=DROP_IN_FOR)
    selection = None
    if villa:
        # band mode's record stays exactly as it was; villa mode names itself and its reference instead
        del provenance["drop_in_for"]
        provenance.update(mode="villa", air_tiles=VILLA_SKIP_RULE, reference=VILLA_REFERENCE)
        selection = VillaSelection(a.ct, ct.shape)
    comp = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
    root = zarr.open_group(str(a.out), mode="a")
    if "semantic" not in root:
        root.create("semantic", shape=(2, nz, ny, nx), chunks=(2, 128, 128, 128), dtype="u1", compressor=comp,
                    fill_value=0, write_empty_chunks=False)
        root.create("tracing", shape=(2, nz // 2, ny // 2, nx // 2), chunks=(2, 64, 64, 64), dtype="u1",
                    compressor=comp, fill_value=0, write_empty_chunks=False)
        root.attrs.update(channel_order="VH", axes=["c", "z", "y", "x"], voxel_size_um=9.362,
                          origin_zyx=[a.z0, 0, 0], inference=provenance)
    elif root.attrs["inference"] != json.loads(json.dumps(provenance)):
        raise ValueError("output exists with different provenance; use a new output")
    sem, tra = root["semantic"], root["tracing"]

    progress_f = a.out / "progress.jsonl"
    done = set()
    if progress_f.exists():
        done = {tuple(json.loads(line)["col"]) for line in progress_f.read_text().splitlines() if line}
    cols = Columns(ny, nx, a.column)
    only = None
    if a.only_columns:
        only = {int(v) for v in a.only_columns.split(",") if v.strip()}
        if not only <= set(range(len(cols.cols))):
            raise ValueError(f"--only-columns outside 0..{len(cols.cols) - 1}")
    pz_all = positions(nz)
    todo = [(i, col, cols.tiles(col, pz_all)) for i, col in enumerate(cols.cols)
            if col not in done and (only is None or i in only)]
    cuda = device.type == "cuda"
    w3 = torch.from_numpy(blend_weights()).to(device)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    if amp_dtype:
        def amp():
            return torch.autocast(device.type, dtype=getattr(torch, amp_dtype))
    else:
        amp = contextlib.nullcontext

    feed = TileFeed(torch, str(a.ct), a.z0, a.z1, [(i, tiles) for i, _, tiles in todo], a.batch, a.readers, pin=cuda,
                    runs=selection.tile_runs if villa else None)
    writer = ColumnWriter(sem, tra, progress_f)
    stopped = False
    try:
        with torch.inference_mode():
            for i, col, tiles in todo:
                if a.max_minutes and time.time() - t_start > a.max_minutes * 60:
                    stopped = True
                    break
                t0 = time.time()
                stats = dict(forward_s=0.0, tiles_run=0, tiles_air=0, voxels_uncovered=0)
                acc = ColumnAccumulator(torch, device, nz, col, w3)
                spans = []
                while True:
                    item = feed.get()
                    if item.col != i:
                        raise RuntimeError(f"tile feed out of order: column {item.col} while running {i}")
                    if isinstance(item, ColumnEnd):
                        stats["tiles_air"] = item.air
                        break
                    # pinned staging: the copy is queued without a host sync, so the host runs ahead of the device
                    x = item.x.to(device, non_blocking=cuda)
                    span = _Span(torch, cuda)
                    with amp():
                        out = model(normalize(x))["seg"]
                    if out.ndim != 5 or out.shape[1] != 4:
                        raise ValueError(f"expected B,4,Z,Y,X seg logits, got {tuple(out.shape)}")
                    prob = out.float().softmax(1)[:, 1:3]
                    span.end()
                    spans.append(span)
                    stats["tiles_run"] += len(item.tiles)
                    for (pz, py, px), p in zip(item.tiles, prob):
                        acc.add(pz, py, px, p)
                    del x, out, prob
                s, t, stats["voxels_uncovered"] = acc.finish(stats["tiles_run"] > 0)
                del acc
                if villa and s is not None:
                    selection.mask_column(col, s, t)
                stats["forward_s"] = sum(sp.seconds() for sp in spans)
                rec = dict(col=list(col), tiles=len(tiles), attempt=t_start, **stats)
                writer.submit(col, s, t, rec, t0)
    finally:
        feed.close()
        writer.drain()

    recs = [json.loads(line) for line in progress_f.read_text().splitlines() if line]
    # the band reference's timing fields, with the meanings they take in an overlapped run
    summary = dict(model="AFV v2", precision=short, attempt=t_start, end=time.time(),
                   complete=len(recs) == len(cols.cols), identity=prediction_identity(a.out),
                   columns_total=len(cols.cols), columns_done=len(recs), stopped_early=stopped,
                   tiles_run=sum(r["tiles_run"] for r in recs), tiles_air=sum(r["tiles_air"] for r in recs),
                   forward_s=sum(r["forward_s"] for r in recs), column_wall_s=sum(r["wall_s"] for r in recs),
                   this_run_wall_s=time.time() - t_start,
                   peak_vram_gib=(torch.cuda.max_memory_allocated() / 2**30) if cuda else None,
                   gpu=torch.cuda.get_device_name() if cuda else None, engine="inkrs-afv", readers=a.readers,
                   timing=TIMING)
    if villa:
        summary["mode"] = "villa"
    (a.out / f"summary-{int(t_start)}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    return 0


class _Span:
    """Device time of one batch's normalise, forward and softmax, without a host sync (CUDA events); host time
    on the CPU, where every op is synchronous"""

    def __init__(self, torch, cuda: bool):
        self._torch, self._cuda = torch, cuda
        if cuda:
            self._a = torch.cuda.Event(enable_timing=True)
            self._b = torch.cuda.Event(enable_timing=True)
            self._a.record()
        else:
            self._t0 = time.perf_counter()

    def end(self) -> None:
        if self._cuda:
            self._b.record()
        else:
            self._t1 = time.perf_counter()

    def seconds(self) -> float:
        if self._cuda:
            self._b.synchronize()
            return self._a.elapsed_time(self._b) / 1000
        return self._t1 - self._t0
