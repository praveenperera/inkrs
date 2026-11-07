"""villa's native 3D inference on a crop, as the reference map (and the timed baseline) for `inkrs ink3d`

    python -I scripts/villa_map3d.py --ckpt CKPT.pth --crop CROP.zarr --workdir DIR --result OUT.json \
        [--eager] [--blend-device cuda|cpu] [--normalize-device cuda|cpu] [--villa-src SRC] [--runs 2] [--tag v3]  (GPU)

Plans every 128^3 chunk of the crop as occupied with no halo, partial edge chunks included (the 245-patch plan
on the 768x1024x1024 Paris 4 crop), runs villa's `run_native_inference` `--runs` times on one loaded model, and
writes each run's level-0 zarr as `<tag>_<i>.zarr`, plus villa's plan as `<tag>_plan.json` (patch corners,
target chunks, shapes) for `mapgate3d.py --villa-plan`, which rebuilds villa's coverage from it. `--eager` passes villa's `--no-compile`, the parity reference; without it villa compiles, its
default and fastest setting. Two runs give the run-to-run band of the reference itself
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import sys

# --villa-src runs villa from a checkout (its vesuvius/src) instead of the installed package; it must be on
# the path before villa is imported, so it is read here, ahead of argparse
if "--villa-src" in sys.argv:
    sys.path.insert(0, sys.argv[sys.argv.index("--villa-src") + 1])

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vesuvius.ink_detection.inference import infer_full3d_tifxyz as F  # noqa: E402
from vesuvius.ink_detection.models.checkpoint import config_from_checkpoint, load_checkpoint  # noqa: E402


def plan_for(crop, shape, ch, ps, overlap=0.5):
    # one point inside every chunk of villa's ceil-divided grid, partial edge chunks included
    grid = F.chunk_grid_shape(shape, ch)
    centres = [np.arange(g) * c + np.minimum(c, s - np.arange(g) * c) // 2 for g, c, s in zip(grid, ch, shape)]
    zz, yy, xx = np.meshgrid(*centres, indexing="ij")
    plan = F.build_native_plan(volume_path=str(crop), array_shape_zyx=shape, chunk_shape_zyx=ch,
                               x=xx.ravel().astype(np.float32), y=yy.ravel().astype(np.float32),
                               z=zz.ravel().astype(np.float32), valid=np.ones(zz.size, bool),
                               patch_size_zyx=ps, overlap=overlap, chunk_halo=0, write_region="occupied",
                               max_target_chunks=None)
    want = int(np.prod(grid))
    if len(plan.target_chunks) != want:
        raise SystemExit(f"plan targets {len(plan.target_chunks)} chunks, the crop has {want}")
    return plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--crop", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--blend-device", default="cuda")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--overlap", type=float, default=0.5, help="villa's --overlap, for the plan and the args")
    ap.add_argument("--villa-src", help="a villa checkout's vesuvius/src to import instead of the installed villa")
    ap.add_argument("--normalize-device", help="villa's --normalize-device (the loader thread's patch 0003)")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    import zarr

    shape = tuple(zarr.open_array(str(Path(args.crop) / "0"), mode="r").shape)
    ch = (128, 128, 128)
    payload = load_checkpoint(args.ckpt)
    config = config_from_checkpoint(payload, source=args.ckpt)
    ps = tuple(config.model.crop_size)
    a = F.parse_args(["unused_tifxyz", str(args.ckpt), "unused_out", "--overlap", str(args.overlap)]
                     + (["--no-compile"] if args.eager else []))
    a.blend_device = args.blend_device
    if args.normalize_device:
        if not hasattr(a, "normalize_device"):
            raise SystemExit("this villa has no --normalize-device")
        a.normalize_device = args.normalize_device
    bundle = F._load_native_model(payload, config, a)
    plan = plan_for(args.crop, shape, ch, ps, args.overlap)
    Path(args.workdir, f"{args.tag}_plan.json").write_text(json.dumps({
        "array_shape": list(shape), "chunk_shape": list(ch), "patch_size": list(ps),
        "patches": [[p.z0, p.y0, p.x0] for p in plan.patches],
        "target_chunks": [[c.z, c.y, c.x] for c in sorted(plan.target_chunks)]}))
    res = {"torch": torch.__version__, "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
           "villa": F.__file__, "eager": args.eager, "overlap": args.overlap, "blend_device": args.blend_device,
           "normalize_device": getattr(a, "normalize_device", None), "amp_dtype": str(bundle.amp_dtype),
           "batch_size": a.batch_size, "num_workers": getattr(a, "num_workers", None),
           "patches": len(plan.patches), "runs": []}
    with torch.inference_mode(), torch.autocast("cuda", dtype=bundle.amp_dtype):
        t = time.perf_counter()
        for _ in range(2 if args.eager else 3):
            bundle.model(torch.rand((a.batch_size, 1, *ps), device="cuda"))
        torch.cuda.synchronize()
        res["warmup_s"] = round(time.perf_counter() - t, 1)
    for i in range(args.runs):
        outp = Path(args.workdir) / f"{args.tag}_{i}.zarr"
        arrays = F.create_output_zarr(outp, shape_zyx=shape, chunks_zyx=ch, overwrite=True)
        # the loader thread's GPU normalize is a dataset mode; villa without it takes no such argument
        dev_norm = {"device_normalize": True} if args.normalize_device == "cuda" else {}
        ds = F.NativePatchDataset(tifxyz_dir=Path("unused"), volume_path=str(args.crop), resolution="0",
                                  patches=plan.patches, patch_size_zyx=plan.patch_size_zyx, config=config, **dev_norm)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        F.run_native_inference(args=a, bundle=bundle, dataset=ds, plan=plan, output=arrays[0],
                               importance_map=F.create_importance_map(plan.patch_size_zyx, mode="gaussian"))
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        row = {"zarr": str(outp), "wall_s": round(wall, 2), "s_per_patch": round(wall / len(plan.patches), 4)}
        res["runs"].append(row)
        print(json.dumps(row), flush=True)
    Path(args.result).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
