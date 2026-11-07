"""Export a v8-in family checkpoint to safetensors plus the contract inkrs checks (CPU only)

Runs once per checkpoint. It loads the weights into Youssef Nader's own `InkDetector` (`ink8um/modeling.py`
from huggingface.co/YoussefMoNader/ink-8um-v8in at d89166b4, MIT) with strict=True, and refuses any model
code but the pinned file and any weights but the pinned releases below. BatchNorm's `num_batches_tracked` is
left out and listed in the sidecar.

    python -I scripts/export_v8in.py V8IN_DIR WEIGHTS NAME OUT_DIR

V8IN_DIR holds `ink8um/` and `config.json` of the base release; WEIGHTS is a `model.safetensors`; NAME is
one of the pinned names. Writes OUT_DIR/NAME.safetensors and OUT_DIR/NAME.json.

The sidecar's `map_convention` says how the export's maps are stitched (layer order, tile admission, window,
normalisation, sigmoid precision); inkrs checks it field by field against the conventions it implements.
The default is the convention of the runner that made the checkpoint's published maps: `release`
(ink8um's `predict_stack`) for v8-in, `306b` (`aclick/bench/infer.py --tta 1`) for A-hn. A run can override
it (`--convention`, `convention=`).

The checkpoint's own `config.json` flag `reverse_layers` is recorded as `upstream_reverse_layers` and never
used: A-hn's sets it to true, yet the 306B runner reads the render in stored order unless its own
`--reverse` flag is given, and the A-hn member maps were made without it. Exporting an A-hn checkpoint checks
that the pinned runner still never reads the flag. inkrs takes the depth direction from the caller only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

MODELING_SHA256 = "0656550dc10556ad09f5b6c21b36c97aa97a5696f71ef4b384eada124095c109"
# name -> (sha256 of model.safetensors, source, licence of the weights)
PINNED = {
    "v8in": (
        "3b94548d7f9ba9f99bc4b0ad8cafcec6d42585c06304e1bd11e47b5d0f88c4cd",
        "huggingface.co/YoussefMoNader/ink-8um-v8in d89166b4",
        "MIT",
    ),
    "v8in-ahn-s0": (
        "9a2812843924c50cafaa2b598f62c92ceca9f0d1f5d05b7bbf25764814ec7009",
        "huggingface.co/Prodigle/pherc0306b-ink-ahn checkpoints/hn4_s0_final",
        "CC BY-NC 4.0",
    ),
    "v8in-ahn-s1": (
        "5c2740c1f33ee5db73ca97425669607a2ffe4ae9af8b9d09d61f1c43165e60ad",
        "huggingface.co/Prodigle/pherc0306b-ink-ahn checkpoints/hn4_s1_final",
        "CC BY-NC 4.0",
    ),
    "v8in-ahn-s2": (
        "a294716b083769e8116d5f8d3cfa6a09378ca5c7e51c32225d04703266ad8ce7",
        "huggingface.co/Prodigle/pherc0306b-ink-ahn checkpoints/hn4_s2_final",
        "CC BY-NC 4.0",
    ),
}
# the map conventions inkrs implements, as `src/v8in.rs` `Convention::spec` spells them
CONVENTIONS = {
    "release": {
        "name": "release", "runner": "ink8um.inference.predict_stack", "layers": "central 24, stored order",
        "reverse": False, "admit": "every pixel inside the 7x7-ellipse-closed coverage of the 24 layers",
        "admit_layer": None, "lattice": "stride 21 from the origin, tiles entirely inside", "window_nsig": 1,
        "divide": "tile count", "empty": "0", "clip": [0, 1], "sigmoid": "fp16 logit",
    },
    "306b": {
        "name": "306b", "runner": "aclick/bench/infer.py --tta 1", "layers": "central 24, stored order",
        "reverse": False, "admit": "at least a quarter of the tile nonzero on one layer", "admit_layer": 12,
        "lattice": "stride 21 from the origin, tiles entirely inside", "window_nsig": 3, "divide": "weight sum",
        "empty": "NaN", "clip": None, "sigmoid": "fp32 logit",
    },
}
DEFAULT_CONVENTION = {"v8in": "release", "v8in-ahn-s0": "306b", "v8in-ahn-s1": "306b", "v8in-ahn-s2": "306b"}
# the 306B runner the A-hn maps came from, in a checkout of the pherc0306b-first-letters repository
# (PHERC0306B_REPO, default ./pherc0306b-first-letters); its depth order comes from its own --reverse flag only
RUNNER_306B_IN_REPO = "submission/code/training/x_push3/aclick/bench/infer.py"  # in pherc0306b-first-letters
RUNNER_306B = Path(os.environ.get("PHERC0306B_REPO", "pherc0306b-first-letters")) / RUNNER_306B_IN_REPO
RUNNER_306B_SHA256 = "90800e75f5246555f01ab2efc2b37a4b671511fbc56b92324451c16e3b559d48"

# the architecture the engine implements; anything else in a config is refused
ARCH = {"in_depth": 24, "tile_size": 64, "input_upsample_depth": 96, "input_upsample_hw": 256, "encoder_depth": 50,
        "clip_max": 200, "stride": 21}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pinned_modeling(path: Path) -> types.ModuleType:
    """Runs exactly the bytes whose hash matched, as a standalone module: importing the `ink8um` package would
    also run its unhashed `__init__.py`, and a path entry for the release directory would let any file there
    shadow a module. `modeling.py` imports only torch and huggingface_hub, so it needs neither"""
    src = path.read_bytes()
    if (s := hashlib.sha256(src).hexdigest()) != MODELING_SHA256:
        raise SystemExit(f"{path}: sha256 {s}, expected the pinned {MODELING_SHA256}")
    mod = types.ModuleType("_v8in_pinned_modeling")
    mod.__file__ = str(path)
    # dataclass and hub-mixin machinery look the defining module up by name
    sys.modules[mod.__name__] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("v8in_dir", type=Path)
    ap.add_argument("weights", type=Path)
    ap.add_argument("name", choices=sorted(PINNED))
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    want, source, licence = PINNED[args.name]
    if (s := sha256(args.weights)) != want:
        raise SystemExit(f"{args.weights}: sha256 {s}, expected {want} for {args.name}")
    InkDetector = load_pinned_modeling(args.v8in_dir / "ink8um" / "modeling.py").InkDetector

    cfg_path = args.weights.parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else json.loads((args.v8in_dir / "config.json").read_text())
    if any(cfg.get(k) != v for k, v in ARCH.items()):
        raise SystemExit(f"config {cfg} is not the v8-in architecture {ARCH}")
    convention = CONVENTIONS[DEFAULT_CONVENTION[args.name]]
    if convention["name"] == "306b":
        # pin the quirk: the runner ignores the checkpoint's reverse_layers, so the export must too
        if (s := sha256(RUNNER_306B)) != RUNNER_306B_SHA256:
            raise SystemExit(f"{RUNNER_306B}: sha256 {s}, expected the pinned {RUNNER_306B_SHA256}")
        if "reverse_layers" in RUNNER_306B.read_text():
            raise SystemExit(f"{RUNNER_306B} reads reverse_layers; the 306b convention's stored order is stale")
    model = InkDetector(**cfg)
    model.load_state_dict(load_file(str(args.weights)), strict=True)
    model.eval()

    norms = [m for m in model.modules() if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.InstanceNorm3d))]
    eps = {m.eps for m in norms}
    inorm = model.normalization
    if eps != {1e-5} or not (inorm.affine and not inorm.track_running_stats and inorm.num_features == 1):
        raise SystemExit(f"unexpected norm settings: eps {eps}, input norm {inorm}")

    unique: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for key, tensor in model.state_dict().items():
        if key.endswith(".num_batches_tracked"):
            dropped.append(key)
            continue
        if tensor.dtype != torch.float32:
            raise SystemExit(f"{key}: dtype {tensor.dtype}, expected float32")
        if not torch.isfinite(tensor).all():
            raise SystemExit(f"{key}: non-finite values")
        unique[key] = tensor.detach().contiguous().clone()

    contract = {
        "reader": args.name,
        "architecture_family": "v8in",
        "source": source,
        "weights_licence": licence,
        "checkpoint": str(args.weights),
        "checkpoint_sha256": want,
        "modeling_py_sha256": MODELING_SHA256,
        "config": cfg,
        "upstream_reverse_layers": bool(cfg.get("reverse_layers", False)),
        "map_convention": convention,
        "eps": 1e-5,
        "keys": {k: list(v.shape) for k, v in unique.items()},
        "aliases": {},
        "dropped": dropped,
        "module_repr": repr(model),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    st = args.out_dir / f"{args.name}.safetensors"
    save_file(unique, str(st), metadata={"reader": args.name, "aliases": "{}"})
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(contract, indent=1))
    print(f"wrote {st}: {len(unique)} tensors, {sum(v.numel() for v in unique.values())} params, {len(dropped)} dropped")


if __name__ == "__main__":
    main()
