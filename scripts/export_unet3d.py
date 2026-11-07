"""Export a villa 3D ink checkpoint (NetworkFromConfig ResEnc U-Net) to safetensors plus the layer program inkrs runs (CPU only)

Runs once per checkpoint, in a venv with villa's `vesuvius` package. It builds the network with villa's own
`make_model` and loads the EMA weights (else the model weights) the way villa's native 3D inference does
(`infer_full3d_tifxyz.select_native_inference_weights`, `load_model_state`). It then walks the module tree and
writes the layer program: every conv, norm, activation, skip, transposed conv and head with its tensor names,
shapes, strides and padding. inkrs builds the network from that program, so other checkpoints of the same family
load without code changes. Anything outside the op set inkrs implements stops the export:
- 3D convs with "same" padding at stride 1 or 2;
- affine InstanceNorm3d;
- LeakyReLU, or no activation;
- BasicBlockD encoder blocks (avg-pool + 1x1 conv + norm skip, or identity);
- ConvTranspose3d k2 s2 upsampling;
- ConvBlock decoder stages;
- one 1x1 head with no activation.

    python -I scripts/export_unet3d.py CKPT OUT_DIR [--name NAME]

Writes OUT_DIR/NAME.safetensors and OUT_DIR/NAME.json. NAME defaults to the checkpoint's stem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

FORMAT = "inkrs-unet3d-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fail(msg: str) -> None:
    raise SystemExit(f"unsupported for inkrs: {msg}")


class Walker:
    def __init__(self, model: nn.Module):
        self.names = {id(m): n for n, m in model.named_modules()}
        self.eps: set[float] = set()
        self.slopes: set[float] = set()

    def name(self, m: nn.Module) -> str:
        return self.names[id(m)]

    def conv(self, c: nn.Module, *, transposed: bool = False) -> dict:
        want = nn.ConvTranspose3d if transposed else nn.Conv3d
        if type(c) is not want:
            fail(f"{self.name(c)} is {type(c).__name__}, expected {want.__name__}")
        k, s, p = list(c.kernel_size), list(c.stride), list(c.padding)
        if c.groups != 1 or list(c.dilation) != [1, 1, 1] or c.padding_mode != "zeros":
            fail(f"{self.name(c)}: groups {c.groups}, dilation {c.dilation}, padding mode {c.padding_mode}")
        if transposed:
            if k != [2, 2, 2] or s != [2, 2, 2] or p != [0, 0, 0] or list(c.output_padding) != [0, 0, 0]:
                fail(f"{self.name(c)}: transposed conv k{k} s{s} p{p}, only k2 s2 p0")
        elif any(kk % 2 == 0 or pp != (kk - 1) // 2 for kk, pp in zip(k, p)) or any(ss not in (1, 2) for ss in s):
            fail(f"{self.name(c)}: conv k{k} s{s} p{p}, only odd 'same' kernels at stride 1 or 2")
        n = self.name(c)
        return {
            "w": f"{n}.weight",
            "b": f"{n}.bias" if c.bias is not None else None,
            "cin": c.in_channels,
            "cout": c.out_channels,
            "kernel": k,
            "stride": s,
        }

    def norm(self, m: nn.Module) -> dict:
        if type(m) is not nn.InstanceNorm3d or not m.affine or m.track_running_stats:
            fail(f"{self.name(m)} is {type(m).__name__} (affine {getattr(m, 'affine', None)}), expected affine InstanceNorm3d")
        self.eps.add(float(m.eps))
        n = self.name(m)
        return {"w": f"{n}.weight", "b": f"{n}.bias", "eps": float(m.eps)}

    def act(self, m) -> float | None:
        if m is None or isinstance(m, nn.Identity):
            return None
        if type(m) is not nn.LeakyReLU:
            fail(f"activation {type(m).__name__}, only LeakyReLU")
        self.slopes.add(float(m.negative_slope))
        return float(m.negative_slope)

    def unit(self, u: nn.Module) -> dict:
        """One ConvDropoutNormReLU: conv, optional norm, optional activation, no dropout"""
        if type(u).__name__ != "ConvDropoutNormReLU":
            fail(f"{self.name(u)} is {type(u).__name__}, expected ConvDropoutNormReLU")
        if getattr(u, "dropout", None) is not None:
            fail(f"{self.name(u)} has dropout")
        out = self.conv(u.conv)
        norm = getattr(u, "norm", None)
        if norm is None:
            fail(f"{self.name(u)} has no norm")
        out["norm"] = self.norm(norm)
        out["leaky"] = self.act(getattr(u, "nonlin", None))
        # the module's own order must be conv, norm, nonlin
        seq = [type(m).__name__ for m in u.all_modules]
        want = ["Conv3d", "InstanceNorm3d"] + (["LeakyReLU"] if out["leaky"] is not None else [])
        if seq != want:
            fail(f"{self.name(u)}: module order {seq}, expected {want}")
        return out

    def block(self, b: nn.Module) -> dict:
        if type(b).__name__ != "BasicBlockD":
            fail(f"{self.name(b)} is {type(b).__name__}, only BasicBlockD")
        if getattr(b, "apply_se", False) or getattr(b, "apply_stochastic_depth", False):
            fail(f"{self.name(b)}: squeeze-excitation or stochastic depth")
        conv1, conv2 = self.unit(b.conv1), self.unit(b.conv2)
        if conv1["leaky"] is None or conv2["leaky"] is not None:
            fail(f"{self.name(b)}: conv1 must end in LeakyReLU and conv2 in none")
        nonlin2 = self.act(b.nonlin2)
        if nonlin2 is None:
            fail(f"{self.name(b)}: no output activation")
        if isinstance(b.skip, nn.Sequential):
            pool, proj = 1, None
            for m in b.skip:
                if type(m) is nn.AvgPool3d:
                    ks = m.kernel_size if isinstance(m.kernel_size, (list, tuple)) else [m.kernel_size] * 3
                    st = m.stride if isinstance(m.stride, (list, tuple)) else [m.stride] * 3
                    if list(ks) != [2, 2, 2] or list(st) != [2, 2, 2] or m.padding != 0 or m.ceil_mode or not m.count_include_pad:
                        fail(f"{self.name(m)}: avg pool k{ks} s{st}")
                    if m.divisor_override is not None:
                        fail(f"{self.name(m)}: divisor_override")
                    pool = 2
                else:
                    proj = self.unit(m)
                    if proj["leaky"] is not None or proj["kernel"] != [1, 1, 1] or proj["b"] is not None:
                        fail(f"{self.name(m)}: skip projection must be a bias-free 1x1 conv + norm")
            skip = {"pool": pool, "proj": proj}
        else:
            skip = {"pool": 1, "proj": None}
        stride = conv1["stride"]
        if (stride == [2, 2, 2]) != (skip["pool"] == 2):
            fail(f"{self.name(b)}: stride {stride} with skip pool {skip['pool']}")
        if skip["proj"] is None and conv1["cin"] != conv2["cout"]:
            fail(f"{self.name(b)}: identity skip across {conv1['cin']} -> {conv2['cout']} channels")
        return {"conv1": conv1, "conv2": conv2, "skip": skip, "leaky": nonlin2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from vesuvius.ink_detection.inference import infer_full3d_tifxyz as F
    from vesuvius.ink_detection.models.checkpoint import config_from_checkpoint, load_checkpoint

    payload = load_checkpoint(args.checkpoint)
    config = config_from_checkpoint(payload, source=args.checkpoint)
    weights_name, state = F.select_native_inference_weights(payload, source=args.checkpoint)
    model = F.make_model(config)
    F.load_model_state(model, state)
    model.eval()

    if config.data.mode != "full_3d":
        fail(f"data mode {config.data.mode!r}; only full_3d (one CT channel)")
    if config.model.in_channels != 1 or config.model.input_pad_depth_to is not None:
        fail(f"in_channels {config.model.in_channels}, input_pad_depth_to {config.model.input_pad_depth_to}")
    norm_cfg = config.data.normalization
    if norm_cfg.mode != "percentile_minmax":
        fail(f"normalization {norm_cfg.mode!r}; only percentile_minmax")
    if getattr(model, "guide_backbone", None) is not None:
        fail("guide backbone")

    enc, dec = model.shared_encoder, model.shared_decoder
    heads = dict(model.task_heads.items())
    if list(heads) != ["ink"] or getattr(dec, "deep_supervision", False) or not getattr(dec, "return_features_only", True):
        fail(f"heads {list(heads)}, deep supervision {getattr(dec, 'deep_supervision', None)}")
    w = Walker(model)

    stem = []
    if enc.stem is not None:
        stem = [w.unit(u) for u in enc.stem.convs]
    stages = []
    for s in enc.stages:
        blocks = s.blocks if hasattr(s, "blocks") else s
        stages.append([w.block(b) for b in blocks])
    decoder = []
    for t, st in zip(dec.transpconvs, dec.stages):
        decoder.append({"transp": w.conv(t, transposed=True), "convs": [w.unit(u) for u in st.convs]})
    if len(decoder) != len(stages) - 1:
        fail(f"{len(decoder)} decoder stages for {len(stages)} encoder stages")
    head_mod = heads["ink"]
    head = w.conv(head_mod)
    if head["kernel"] != [1, 1, 1] or head["cout"] != 1 or head["b"] is None:
        fail(f"head {head}")
    target = config.targets["ink"]
    activation = target.get("activation", "none") if isinstance(target, dict) else getattr(target, "activation", "none")
    if activation not in ("none", None):
        fail(f"ink target activation {activation!r}")

    # aliases: the all_modules.* copies of conv and norm tensors
    sd = model.state_dict()
    by_ptr: dict[int, str] = {}
    unique: dict[str, torch.Tensor] = {}
    aliases: dict[str, str] = {}
    for key, t in sd.items():
        if t.dtype != torch.float32:
            fail(f"{key}: dtype {t.dtype}")
        ptr = t.untyped_storage().data_ptr() + t.storage_offset() * t.element_size()
        if ptr in by_ptr and sd[by_ptr[ptr]].shape == t.shape:
            aliases[key] = by_ptr[ptr]
            continue
        by_ptr[ptr] = key
        unique[key] = t.detach().contiguous().clone()

    villa_dir = Path(F.__file__).resolve().parents[4]
    try:
        villa_rev = subprocess.run(["git", "-C", str(villa_dir), "rev-parse", "HEAD"], capture_output=True, text=True,
                                   check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        villa_rev = None
    # villa's Gaussian importance map (create_importance_map, sigma 0.125 x patch), as its float32
    # factors; inkrs rebuilds it as (a0 * a1) * a2 / max, clipped at float32 eps, which is checked
    # here to equal villa's map bit for bit
    import numpy as np

    ps = tuple(config.model.crop_size)
    want = F.create_importance_map(ps, mode="gaussian")
    axes = []
    for size in ps:
        sigma = max(float(size) * 0.125, 1e-6)
        coordinate = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
        axes.append(np.exp(-0.5 * (coordinate / sigma) ** 2).astype(np.float32))
    prod = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    peak = np.float32(prod.max())
    rebuilt = np.maximum(prod / peak, np.finfo(np.float32).eps).astype(np.float32)
    if not np.array_equal(rebuilt.view(np.uint32), want.view(np.uint32)):
        fail("could not rebuild villa's importance map bit for bit")
    blend = {"mode": "gaussian", "sigma_scale": 0.125,
             "axes_f32_bits": [a.view(np.uint32).tolist() for a in axes]}
    program = {"stem": stem, "stages": stages, "decoder": decoder, "head": head}
    contract = {
        "format": FORMAT,
        "patch_size": list(config.model.crop_size),
        "in_channels": 1,
        "amp_dtype": "bf16" if str(payload.get("config", {}).get("mixed_precision")) == "bf16" else None,
        "normalization": {"mode": norm_cfg.mode, "lower": float(norm_cfg.percentile_lower),
                          "upper": float(norm_cfg.percentile_upper)},
        "weights": weights_name,
        "program": program,
        "blend": blend,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "villa_rev": villa_rev,
        "keys": {k: list(v.shape) for k, v in unique.items()},
        "aliases": aliases,
        "module_repr": repr(model),
    }
    if contract["amp_dtype"] is None:
        fail(f"mixed_precision {payload.get('config', {}).get('mixed_precision')!r}; only bf16 autocast")
    name = args.name or args.checkpoint.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    st = args.out_dir / f"{name}.safetensors"
    save_file(unique, str(st), metadata={"reader": FORMAT, "aliases": json.dumps(aliases)})
    (args.out_dir / f"{name}.json").write_text(json.dumps(contract, indent=1))
    n_conv = len(stem) + sum(2 + (b["skip"]["proj"] is not None) for s in stages for b in s) + \
        sum(1 + len(d["convs"]) for d in decoder) + 1
    print(f"wrote {st}: {len(unique)} tensors, {len(aliases)} aliases, "
          f"{sum(v.numel() for v in unique.values())} params, {n_conv} convs, eps {sorted(w.eps)}, slopes {sorted(w.slopes)}")


if __name__ == "__main__":
    main()
