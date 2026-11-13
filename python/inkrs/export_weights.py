"""Export one villa flat ink checkpoint to safetensors plus the config inkrs needs (CPU only)

Runs once per checkpoint, on the CPU, where villa 0fb38c45 is installed. It rebuilds the model exactly as villa's
`infer.configure_model` does, then loads the state with strict=True, which villa does not (it uses strict=False and
so hides mismatches). Aliased parameters (`conv.weight` and `all_modules.0.weight` are one tensor) are written once
under the first name, and the aliases go into the metadata so the Rust side can check the full key set.

    python -m inkrs.export_weights CKPT OUT_DIR      # from the installed package
    python -I scripts/export_weights.py CKPT OUT_DIR  # from the source tree
    python -m inkrs.export_weights STATE.safetensors OUT_DIR --config TRAIN.json --name NAME

CKPT is a villa `.pth` payload (`config` plus a state dict). A release that ships a bare state dict as
safetensors plus its training config JSON instead (dense_native) is exported from the pair with `--config`: the
two are rebuilt into the same payload villa's loader would see, so the model is built from the same config.

or `inkrs.export_weights.export(CKPT, OUT_DIR)` from Python. Writes OUT_DIR/<reader>.safetensors and
OUT_DIR/<reader>.json; inkrs looks a `.pth` up there through INKRS_WEIGHTS_DIR (default `weights/`), or the
`weights_dir` of `inkrs.Engine`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

# a checkpoint file stem that only names a step of a run (`step-075000`, `ckpt_020000`, `best`), whose run is the
# directory holding it, and a step suffix on a run's name (`reader-v2-step040000`)
_STEP_STEM = re.compile(r"(?i)(?:(?:step|ckpt|checkpoint|epoch|iter|model)[-_]?\d*|\d+|best|last|final|latest)")
_STEP_SUFFIX = re.compile(r"(?i)[-_](?:step|ckpt|checkpoint|epoch|iter)[-_]?\d+$")


def default_name(checkpoint: str | Path) -> str:
    """The reader name for an export of `checkpoint` when none is given: the file's stem without a step suffix
    (`reader-v2-step040000.pth` gives `reader-v2`), or the run directory's name when the stem only names a step
    (`ink_9um/hybrid_3d2d-seed43/step-075000.pth` gives `hybrid_3d2d-seed43`)

    The name keys output file names, so two checkpoints must not share one by default
    """
    path = Path(checkpoint)
    if _STEP_STEM.fullmatch(path.stem) and path.parent.name not in ("", "."):
        return path.parent.name
    return _STEP_SUFFIX.sub("", path.stem) or path.stem



class ExportUnsupported(ValueError):
    """The checkpoint is valid, but not one inkrs implements: not a flat model, or weights that are not float32"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _bare_payload(checkpoint: Path, config: Path) -> dict:
    """A villa payload from a bare safetensors state dict and its training config JSON

    The config must be a whole villa training config (it names `model_type`, `model_config` and `targets`), not a
    model card's summary: the model is rebuilt from it, so a partial config could build another network that the
    strict load below would then reject
    """
    from safetensors.torch import load_file

    cfg = json.loads(config.read_text())
    if not isinstance(cfg, dict):
        raise ExportUnsupported(f"{config}: not a JSON object")
    missing = [k for k in ("mode", "model_type", "model_config", "targets") if k not in cfg]
    if missing:
        raise ExportUnsupported(f"{config}: no {missing}; pass the release's training config")
    return {"config": cfg, "model": load_file(str(checkpoint))}


def export(
    checkpoint: str | Path,
    out_dir: str | Path,
    name: str | None = None,
    config: str | Path | None = None,
) -> Path:
    """Exports `checkpoint` to OUT_DIR/<name>.safetensors and its .json sidecar; returns the .safetensors path

    `checkpoint` is a villa `.pth`, or with `config` a bare safetensors state dict and its training config JSON.
    `name` defaults to `default_name(checkpoint)`
    """
    if name is None:
        name = default_name(checkpoint)
    # villa and torch are needed only here, so `import inkrs` works without them
    import torch
    from safetensors.torch import save_file
    from vesuvius.ink_detection.config import InkConfig
    from vesuvius.ink_detection.inference.infer import flat_preprocessing_from_config
    from vesuvius.ink_detection.inference.inference_runtime import resolve_amp_dtype
    from vesuvius.ink_detection.models.checkpoint import load_checkpoint, select_inference_weights
    from vesuvius.ink_detection.models.model import make_model

    args = argparse.Namespace(checkpoint=Path(checkpoint), out_dir=Path(out_dir), name=name,
                              config=None if config is None else Path(config))
    bare = args.checkpoint.suffix == ".safetensors"
    if bare != (args.config is not None):
        raise ExportUnsupported("a .safetensors state dict needs --config, and --config needs a .safetensors state dict")
    payload = _bare_payload(args.checkpoint, args.config) if bare else load_checkpoint(args.checkpoint)
    config = InkConfig.from_mapping(payload["config"])
    if config.data.mode != "flat":
        raise ExportUnsupported(f"mode {config.data.mode!r} is not flat")
    selected, state = select_inference_weights(payload, source=args.checkpoint)
    if state and all(str(k).startswith("module.") for k in state):
        state = {str(k)[len("module."):]: v for k, v in state.items()}
    model = make_model(config)
    # strict: every checkpoint key is used and every model key is present
    model.load_state_dict(state, strict=True)
    model.eval()

    sd = model.state_dict(keep_vars=False)
    unique: dict[str, torch.Tensor] = {}
    aliases: dict[str, str] = {}
    by_ptr: dict[tuple[int, int, tuple], str] = {}
    for key, tensor in sd.items():
        ident = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape))
        if ident in by_ptr:
            aliases[key] = by_ptr[ident]
            continue
        by_ptr[ident] = key
        if tensor.dtype != torch.float32:
            raise ExportUnsupported(f"{key}: dtype {tensor.dtype}, expected float32")
        unique[key] = tensor.detach().contiguous().clone()

    amp = resolve_amp_dtype("auto", payload, args.checkpoint)
    crop = list(config.model.crop_size)
    contract = {
        "reader": args.name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "config_json": None if args.config is None else str(args.config),
        "config_json_sha256": None if args.config is None else _sha256(args.config),
        "step": payload.get("step"),
        "selected_state": selected,
        "model_type": config.model.model_type,
        "crop_size": crop,
        "input_pad_depth_to": config.model.input_pad_depth_to,
        "preprocessing": flat_preprocessing_from_config(config.data.normalization),
        "amp_dtype": None if amp is None else str(amp).removeprefix("torch."),
        "model_config": payload["config"].get("model_config"),
        "targets": payload["config"].get("targets"),
        "keys": {k: list(v.shape) for k, v in unique.items()},
        "aliases": aliases,
        "module_repr": repr(model),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    st = args.out_dir / f"{args.name}.safetensors"
    save_file(unique, str(st), metadata={"reader": args.name, "aliases": json.dumps(aliases)})
    # the sidecar names the tensor file's own bytes, so a sidecar left beside other tensors (an interrupted
    # re-export) is caught by a reader that checks it (scripts/recipe.py)
    contract["safetensors_sha256"] = _sha256(st)
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(contract, indent=1))
    print(f"wrote {st}: {len(unique)} tensors, {len(aliases)} aliases, "
          f"{sum(v.numel() for v in unique.values())} params; amp={contract['amp_dtype']} "
          f"preprocessing={contract['preprocessing']} crop={crop} pad_depth_to={contract['input_pad_depth_to']}")
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--name", help="reader name for the output files; default: the checkpoint's stem, or its run "
                    "directory's name when the stem only names a step")
    ap.add_argument("--config", type=Path, help="training config JSON, for a bare safetensors state dict")
    args = ap.parse_args()
    try:
        export(args.checkpoint, args.out_dir, args.name, args.config)
    except ValueError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
