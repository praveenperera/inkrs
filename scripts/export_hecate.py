"""Export the released Hecate 9.6 um checkpoint to safetensors plus the contract inkrs checks (CPU only)

Runs once per checkpoint. It loads the checkpoint with Hecate's own `load_model` from `hecate.py`, which
checks the release's architecture family, EMA weights and configuration and loads the state dict with
strict=True. The script refuses any `hecate.py` but the pinned one. Training-only tensors are left out and
listed in the sidecar: the classifier `fc`, the deep-supervision heads and BatchNorm's
`num_batches_tracked`.

    python -I scripts/export_hecate.py HECATE_PY CKPT OUT_DIR

HECATE_PY and CKPT come from huggingface.co/scrollprize/hecate at 9cb86e500e944b11a06a7020403cde5dffb5bcb2
(MIT, "Copyright (c) 2026 Vesuvius Challenge"). Writes OUT_DIR/hecate-9.6um.safetensors and
OUT_DIR/hecate-9.6um.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

# hecate.py at scrollprize/hecate 9cb86e5, the revision whose numerics inkrs reproduces
HECATE_PY_SHA256 = "c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59"
NAME = "hecate-9.6um"
# not used by 2D inference: Hecate.forward never reaches the classifier or the aux heads
DROPPED_PREFIXES = ("canonical.backbone.fc.", "canonical.decoder.aux_head_s1.", "canonical.decoder.aux_head_s2.")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hecate_py", type=Path)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    code_sha = sha256(args.hecate_py)
    if code_sha != HECATE_PY_SHA256:
        raise SystemExit(f"{args.hecate_py}: sha256 {code_sha}, expected the pinned {HECATE_PY_SHA256}")
    spec = importlib.util.spec_from_file_location("hecate", args.hecate_py)
    hecate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hecate)

    model = hecate.load_model(str(args.checkpoint), "cpu")
    if (model.sampling_um, model.patch_size, model.margin, model.xy_stride, model.divisor) != (9.6, (16, 64, 64), 0, 1, 255.0):
        raise SystemExit(f"not the 9.6 um release: {model.sampling_um} {model.patch_size}; inkrs ports only that one")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)

    unique: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for key, tensor in model.state_dict().items():
        if key.startswith(DROPPED_PREFIXES) or key.endswith(".num_batches_tracked"):
            dropped.append(key)
            continue
        if tensor.dtype != torch.float32:
            raise SystemExit(f"{key}: dtype {tensor.dtype}, expected float32")
        unique[key] = tensor.detach().contiguous().clone()
    # BatchNorm's eps is not in the state dict; every BN in the release is the default
    eps = {m.eps for m in model.modules() if isinstance(m, torch.nn.BatchNorm3d)}
    gn_eps = {m.eps for m in model.modules() if isinstance(m, torch.nn.GroupNorm)}
    groups = sorted({(m.num_groups, m.num_channels) for m in model.modules() if isinstance(m, torch.nn.GroupNorm)})
    if eps != {1e-5} or gn_eps != {1e-5} or any(g != min(32, c) for g, c in groups):
        raise SystemExit(f"unexpected norm settings: bn eps {eps}, gn eps {gn_eps}, groups {groups}")

    contract = {
        "reader": NAME,
        "architecture_family": payload["architecture_family"],
        "weight_source": payload["weight_source"],
        "sampling_um": payload["sampling_um"],
        "patch_size": list(model.patch_size),
        "image_normalization": payload["config"]["image_normalization"],
        "model_config": payload["config"]["model_config"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "hecate_py_sha256": code_sha,
        "bn_eps": 1e-5,
        "gn_eps": 1e-5,
        "keys": {k: list(v.shape) for k, v in unique.items()},
        "aliases": {},
        "dropped": dropped,
        "module_repr": repr(model),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    st = args.out_dir / f"{NAME}.safetensors"
    save_file(unique, str(st), metadata={"reader": NAME, "aliases": "{}"})
    # the sidecar names the tensor file's own bytes, so a sidecar left beside other tensors (an interrupted
    # re-export) is caught by a reader that checks it (scripts/recipe.py)
    contract["safetensors_sha256"] = sha256(st)
    (args.out_dir / f"{NAME}.json").write_text(json.dumps(contract, indent=1))
    print(f"wrote {st}: {len(unique)} tensors, {sum(v.numel() for v in unique.values())} params, "
          f"{len(dropped)} dropped")


if __name__ == "__main__":
    main()
