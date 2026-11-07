"""P0 reference: villa's model on a fixed batch of real tiles, every layer, in FP32 and under
villa's FP16 autocast

Builds the model with villa's `infer.configure_model`, reads each tile with
villa's `FlatPatchReader` and `normalize_flat_patch`, and saves each tapped activation as NCHW
float32 `.npy` under OUT/fp32 and OUT/amp, plus the input. The FP32 pass has TF32 off, so it
is the accurate reference; the autocast pass matches villa's inference settings

    python -I scripts/p0_torch.py ZARR CKPT TILES.json OUT

Needs a CUDA GPU
"""

import json
import os
import sys
from pathlib import Path

# villa 0fb38c45: the installed package, or the vesuvius/src of a checkout named by VILLA_SRC
if os.environ.get("VILLA_SRC"):
    sys.path.insert(0, os.environ["VILLA_SRC"])

import numpy as np  # noqa: E402
import torch  # noqa: E402
from vesuvius.ink_detection.inference.infer import FlatPatchReader, configure_model, normalize_flat_patch  # noqa: E402


def build(ckpt):
    # villa's own model construction: TargetModel(base) with the checkpoint's AMP dtype
    import argparse
    cm = configure_model(argparse.Namespace(checkpoint=Path(ckpt), amp_dtype="auto"))
    assert cm.amp_dtype == torch.float16, cm.amp_dtype
    # 128 for hybrid_3d2d-seed42 and reader-v2, 384 for Reader V3
    assert cm.patch_size in (128, 384) and cm.input_depth == 17, (cm.patch_size, cm.input_depth)
    return cm.model.cuda().eval(), cm.preprocessing, cm.patch_size


def taps(target):
    model = target.model
    named = {
        "stem.block1": model.depth_fusion.features[2],
        "stem.block2": model.depth_fusion.features[5],
        "stem.out": model.depth_fusion,
        "enc.stem": model.network.shared_encoder.stem,
    }
    for s, stage in enumerate(model.network.shared_encoder.stages):
        for j, blk in enumerate(stage.blocks):
            named[f"enc.{s}.{j}"] = blk
    for i, st in enumerate(model.network.shared_decoder.stages):
        named[f"dec.{i}"] = st
    return named


def run(model, x, named, autocast):
    got = {}
    hooks = [m.register_forward_hook(lambda _m, _i, o, n=n: got.__setitem__(n, o.detach().float().clone())) for n, m in named.items()]
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            out = model(x)
    finally:
        for h in hooks:
            h.remove()
    got["logits"] = out.detach().float().clone()
    return got


def main():
    zarr_path, ckpt, tiles_path, out = sys.argv[1:5]
    out = Path(out)
    tiles = json.loads(Path(tiles_path).read_text())
    model, preprocessing, patch = build(ckpt)
    assert preprocessing == "tifxyz_robust", preprocessing

    import zarr
    root = zarr.open(zarr_path, mode="r")
    arr = root["0"] if "0" in root else root
    depth, height, width = arr.shape
    patches = []
    for t in tiles:
        layers = np.array([t["first"] + t["step"] * k for k in range(17)], dtype=np.int64)
        reader = FlatPatchReader(
            input_path=zarr_path, resolution="0", depth_axis_first=True, height=height, width=width,
            layer_indices=layers, output_depth=17, preprocessing=preprocessing,
        )
        hwz = reader.read(t["y0"], t["x0"], patch, patch)
        assert hwz.any()
        patches.append(normalize_flat_patch(np.moveaxis(hwz, -1, 0), preprocessing))
    x = torch.from_numpy(np.stack(patches)[:, None]).cuda()

    named = taps(model)
    for mode, autocast in (("fp32", False), ("amp", True)):
        if autocast:
            # villa's inference settings
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("high")
        else:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        got = run(model, x, named, autocast)
        d = out / mode
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "input.npy", x.float().cpu().numpy())
        for name, v in got.items():
            np.save(d / f"{name}.npy", v.cpu().numpy())
        print(mode, len(got) + 1, "taps", flush=True)
    (out / "meta.json").write_text(json.dumps(dict(tiles=tiles, patch=patch, torch=torch.__version__, cudnn=torch.backends.cudnn.version())))


if __name__ == "__main__":
    main()
