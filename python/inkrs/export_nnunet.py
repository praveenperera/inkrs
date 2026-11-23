"""Export an nnU-Net ResEnc-L checkpoint (m7, Socratic F0, HercUNet v0) to safetensors plus the contract inkrs
runs (CPU only)

Runs once per checkpoint, on the CPU, in an environment with torch and `dynamic_network_architectures`. It builds
the network the way its reference pipeline does (`ResidualEncoderUNet` from the plans' `3d_fullres`
architecture, deep supervision off), loads the weights with `strict=True`, and walks the module tree into the
layer program `inkrs nnunet` builds its forward from: the stem, the BasicBlockD encoder stages, the transposed
convs and conv stacks of the decoder, and the one segmentation layer inference runs. Anything outside the op set
the program can express stops the export:
- 3D convs with odd "same" kernels at per-axis stride 1 or 2, no groups or dilation;
- affine InstanceNorm3d without running statistics;
- LeakyReLU or no activation, no dropout;
- BasicBlockD blocks: average-pool (kernel = stride) + bias-free 1x1 conv + norm skip, or a 1x1 projection, or
  identity;
- ConvTranspose3d with kernel = stride (2 or 1 per axis), no padding;
- one 1x1 head with bias.

The program records kernels, strides and pools per axis, but `inkrs nnunet` runs only isotropic programs (the
same kernel and stride on every axis, 2 x 2 x 2 pools and k2 s2 transposed convs) and refuses the rest as
unsupported.

The recipe (`--recipe`) is the pipeline the weights are run in, which fixes what the weights do not: the
precision autocast runs, the input normalisation, test-time augmentation, the patch blend and the output
finalisation. Each recipe records the reference code it reproduces. inkrs refuses recipes it does not implement.

    python -m inkrs.export_nnunet MODEL OUT_DIR --recipe vesuvius-predict [--name m7]
    python -I scripts/export_nnunet.py MODEL OUT_DIR --recipe vesuvius-predict

MODEL is an nnU-Net model folder (plans.json, dataset.json and a fold_*/checkpoint_*.pth or a flat checkpoint)
or Socratic F0's checkpoint `.pt` next to its `huggingface/f0/config.json` (with `--recipe program-only`). Writes
OUT_DIR/NAME.safetensors, OUT_DIR/NAME.json (the sidecar) and OUT_DIR/NAME.modules.txt (the module tree).

Tensors the inference graph never reads (the deep-supervision segmentation layers and HercUNet's affinity head)
are left out and listed under `dropped` with the reason; any other unused tensor stops the export, so the loader's
check that every exported tensor is used exactly once covers the whole checkpoint. Called as a library, `export`
raises `ExportUnsupported` for these; the command line exits with its message.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

try:
    import torch
    from safetensors.torch import save_file
    from torch import nn
except ImportError:
    # inkrs reads a model folder's inputs and this file's hash from here to find a cached export, which then runs
    # without torch; only `export` needs it, and `missing_packages` names what is absent
    torch = nn = save_file = None

FORMAT = "inkrs-nnunet-v1"
#: the class the plans must name; inkrs implements this network only
ARCH = "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"
# The ops a ResEnc-L plans file may name, by the dotted path nnU-Net writes, mapped to the class's name in torch.nn
OPS = {
    "conv_op": {"torch.nn.modules.conv.Conv3d": "Conv3d", "torch.nn.Conv3d": "Conv3d"},
    "norm_op": {"torch.nn.modules.instancenorm.InstanceNorm3d": "InstanceNorm3d",
                "torch.nn.InstanceNorm3d": "InstanceNorm3d"},
    "nonlin": {"torch.nn.LeakyReLU": "LeakyReLU", "torch.nn.modules.activation.LeakyReLU": "LeakyReLU"},
}
#: the oldest torch `safe_load` runs on
TORCH_MIN = (2, 6)
#: the tensor holding the patch blend weights, stored with the network's
GAUSSIAN_KEY = "inkrs.blend.gaussian"


def missing_packages() -> list[str]:
    """The packages an export needs that do not import here"""
    needed = ["torch", "safetensors", "dynamic_network_architectures"]
    # the blend weights come from villa when it is installed, else from the inline copy, which needs scipy
    if importlib.util.find_spec("vesuvius") is None:
        needed.append("scipy")
    missing = [name for name in needed if importlib.util.find_spec(name) is None]
    # `safe_load`'s allowlist names a global by module and name, which torch takes from 2.6 on
    if torch is not None and _release(torch.__version__) < TORCH_MIN:
        missing.append(f"torch>={'.'.join(map(str, TORCH_MIN))} (torch {torch.__version__} is installed)")
    return missing


def _release(version: str) -> tuple[int, int]:
    """The major and minor release of a version string such as `2.6.0+cpu`"""
    major, minor = version.split("+")[0].split(".")[:2]
    return int(major), int("".join(ch for ch in minor if ch.isdigit()) or 0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ExportUnsupported(ValueError):
    """The checkpoint uses something outside the op set or recipes inkrs implements"""


def fail(msg: str) -> None:
    raise ExportUnsupported(f"unsupported for inkrs: {msg}")


def triple(v) -> list[int]:
    return [int(x) for x in v] if isinstance(v, (list, tuple)) else [int(v)] * 3


def safe_load(path: Path) -> dict:
    """A checkpoint downloaded from a hub is untrusted, so it is unpickled with `weights_only`: tensors, plain
    containers and the numpy scalars and dtypes nnU-Net's training log holds, nothing that runs code"""
    try:
        from numpy._core import multiarray
    except ImportError:  # numpy 1
        from numpy.core import multiarray

    # a checkpoint names the scalar reconstructor by the module of the numpy that saved it, numpy.core before 2.0
    scalar = [(multiarray.scalar, f"{m}.multiarray.scalar") for m in ("numpy._core", "numpy.core")]
    safe = [*scalar, np.dtype, *(type(np.dtype(t)) for t in ["float32", "float64", "int64", "bool"])]
    # the context removes every entry it was given on exit, so pass only those the caller has not allowed already
    allowed = torch.serialization.get_safe_globals()
    safe = [g for g in safe if g not in allowed]
    with torch.serialization.safe_globals(safe):
        return torch.load(path, map_location="cpu", weights_only=True)


class Walker:
    """The layer program of one ResidualEncoderUNet, with its tensor names"""

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
        k, s, p = triple(c.kernel_size), triple(c.stride), triple(c.padding)
        if c.groups != 1 or triple(c.dilation) != [1, 1, 1] or c.padding_mode != "zeros":
            fail(f"{self.name(c)}: groups {c.groups}, dilation {c.dilation}, padding mode {c.padding_mode}")
        if transposed:
            if k != s or any(v not in (1, 2) for v in s) or p != [0, 0, 0] or triple(c.output_padding) != [0, 0, 0]:
                fail(f"{self.name(c)}: transposed conv k{k} s{s} p{p}, only kernel = stride in (1, 2), no padding")
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
        """One ConvDropoutNormReLU: conv, norm, optional activation, no dropout, in that order"""
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
        pool, proj = [1, 1, 1], None
        if isinstance(b.skip, nn.Sequential):
            skip_mods = list(b.skip)
        elif isinstance(b.skip, nn.Identity):
            skip_mods = []
        elif not isinstance(b.skip, nn.Module):
            # BasicBlockD's identity skip is `lambda x: x`; confirm it returns its input unchanged
            probe = torch.zeros(1)
            if b.skip(probe) is not probe:
                fail(f"{self.name(b)}: skip {b.skip!r} is neither a module nor the identity")
            skip_mods = []
        else:
            skip_mods = [b.skip]
        for m in skip_mods:
            if type(m) is nn.AvgPool3d:
                ks, st = triple(m.kernel_size), triple(m.stride)
                if ks != st or any(v not in (1, 2) for v in st) or triple(m.padding) != [0, 0, 0] or m.ceil_mode \
                        or not m.count_include_pad or m.divisor_override is not None:
                    fail(f"{self.name(m)}: avg pool k{ks} s{st} p{m.padding}")
                pool = st
            else:
                proj = self.unit(m)
                if proj["leaky"] is not None or proj["kernel"] != [1, 1, 1] or proj["stride"] != [1, 1, 1] \
                        or proj["b"] is not None:
                    fail(f"{self.name(m)}: skip projection must be a bias-free 1x1 conv + norm")
        if conv1["stride"] != pool:
            fail(f"{self.name(b)}: stride {conv1['stride']} with skip pool {pool}")
        if proj is None and conv1["cin"] != conv2["cout"]:
            fail(f"{self.name(b)}: identity skip across {conv1['cin']} -> {conv2['cout']} channels")
        return {"conv1": conv1, "conv2": conv2, "skip": {"pool": pool, "proj": proj}, "leaky": nonlin2}


def build_network(plans: dict, configuration: str, in_channels: int, num_classes: int) -> nn.Module:
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

    arch = plans["configurations"][configuration]["architecture"]
    if arch["network_class_name"] != ARCH:
        fail(f"network {arch['network_class_name']}, only {ARCH}")
    kw = dict(arch["arch_kwargs"])
    for key in arch["_kw_requires_import"]:
        name = kw.get(key)
        if name is None:
            continue
        # plans files come with downloaded checkpoints: a class name is looked up in a fixed table, never
        # imported, since locating an arbitrary dotted path imports (and runs) the module it names
        cls = OPS.get(key, {}).get(name)
        if cls is None:
            fail(f"{key} {name!r}: not one of the ResEnc ops inkrs runs")
        kw[key] = getattr(nn, cls)
    return ResidualEncoderUNet(input_channels=in_channels, num_classes=num_classes, deep_supervision=False, **kw)


def ct_table(props: dict) -> dict:
    """nnU-Net CTNormalization of every uint8 code, as villa computes it for `vesuvius.predict`
    (vc_reader normalization.hpp `normalize_fixed`: clamp, then (x - center) / scale in float64, stored as
    float32); the engine rounds the table to the model's input dtype as `VCDataset` does"""
    center, scale = float(props["mean"]), max(float(props["std"]), 1e-8)
    low, high = float(props["percentile_00_5"]), float(props["percentile_99_5"])
    codes = np.arange(256, dtype=np.float64)
    table = ((np.clip(codes, low, high) - center) / scale).astype(np.float32)
    return {"mode": "ct", "center": center, "scale": scale, "low": low, "high": high,
            "table_f32_bits": table.view(np.uint32).tolist()}


def villa_gaussian(patch: tuple[int, int, int], sigma_scale: float) -> np.ndarray:
    """villa's `blending.generate_gaussian_map` (vesuvius.predict's blend weights), called from villa when it is
    installed; the inline copy is that function verbatim, for environments without it"""
    try:
        from vesuvius.models.run.blending import generate_gaussian_map

        return np.asarray(generate_gaussian_map(patch, sigma_scale=sigma_scale)[0], dtype=np.float32)
    except ImportError:
        from scipy.ndimage import gaussian_filter

        tmp = np.zeros(patch, dtype=np.float32)
        tmp[tuple(i // 2 for i in patch)] = 1
        g = gaussian_filter(tmp, [i / sigma_scale for i in patch], 0, mode="constant", cval=0)
        g /= max(g.max(), 1e-12)
        return np.clip(g, 0, None).astype(np.float32)


def recipe_for(
    name: str, plans: dict, configuration: str, fingerprint: dict | None, patch: list[int]
) -> tuple[dict, dict]:
    """The pipeline contract and the extra tensors it needs"""
    cfg = plans["configurations"][configuration]
    schemes = cfg.get("normalization_schemes", [])
    extra: dict[str, torch.Tensor] = {}
    if name == "vesuvius-predict":
        # villa vesuvius.predict: fp16 autocast (models/run/inference.py:1186-1190, input fp16 :881-884), 8-flip
        # mirroring in one batch, mean of the fp16 logits (models/run/tta.py:30-69), patch logits blended in
        # float32 with generate_gaussian_map (blending.py:324-340, 437-507), finalize_outputs (softmax[1] * 255
        # truncated, or a logit cutoff; finalize_outputs.py:51-140)
        if schemes != ["CTNormalization"]:
            fail(f"vesuvius-predict recipe implemented for CTNormalization only, plans give {schemes}")
        props = plans["foreground_intensity_properties_per_channel"]
        props = props.get("0", props.get(0))
        extra[GAUSSIAN_KEY] = torch.from_numpy(villa_gaussian(tuple(patch), 8.0))
        recipe = {
            "pipeline": "vesuvius-predict",
            "precision": "fp16",
            "normalization": ct_table(props),
            "tta": {"mode": "mirror8", "combine": "mean_logits"},
            "blend": {"mode": "gaussian_map", "tensor": GAUSSIAN_KEY, "sigma_scale": 8.0},
            "finalize": {"mode": "softmax_u8"},
            "overlap": 0.5,
            "skip_constant_patches": True,
        }
        return recipe, extra
    if name == "hercunet":
        # hercUNet jacobi_refine: fp16 autocast at batch 4, CT from dataset_fingerprint.json clipped to [0, 255],
        # 8 input channels [CT, prev, structure tensor x6], villa Gaussian blend of the logit margin, 4 passes
        if fingerprint is None:
            fail("hercunet recipe needs dataset_fingerprint.json")
        props = fingerprint["foreground_intensity_properties_per_channel"]["0"]
        ct = ct_table({"mean": props["mean"], "std": props["std"], "percentile_00_5": 0.0, "percentile_99_5": 255.0})
        extra[GAUSSIAN_KEY] = torch.from_numpy(villa_gaussian(tuple(patch), 8.0))
        return {
            "pipeline": "hercunet",
            "precision": "fp16",
            "normalization": ct,
            "inputs": ["ct", "prev", "st_xx", "st_xy", "st_xz", "st_yy", "st_yz", "st_zz"],
            "tta": {"mode": "none", "combine": "mean_logits"},
            "blend": {"mode": "gaussian_map", "tensor": GAUSSIAN_KEY, "sigma_scale": 8.0},
            "finalize": {"mode": "margin_sigmoid_u8", "clamp": 30.0},
            "overlap": 0.25,
            "passes": 4,
            "skip_constant_patches": False,
        }, extra
    if name == "program-only":
        # the network alone, for a pipeline inkrs does not implement yet (Socratic F0's crossres_pred
        # inference); the engine refuses it, the CPU checks still cover its program and tensors
        return {"pipeline": "program-only"}, extra
    fail(f"recipe {name!r}")


def find_checkpoint(model_dir: Path) -> Path:
    """The checkpoint villa's `load_model` picks (`utils/models/load_nnunet_model.py:211-238` at 7e6b82164):
    `fold_0/checkpoint_final.pth`, then `fold_0/checkpoint_best.pth`, then the same names flat beside plans.json;
    a `fold_*` directory is searched itself, its parent holding plans.json. No other fold is tried, as villa
    tries none"""
    fold = model_dir if model_dir.name.startswith("fold_") else model_dir / "fold_0"
    root = model_root(model_dir)
    for ck in [fold / "checkpoint_final.pth", fold / "checkpoint_best.pth",
               root / "checkpoint_final.pth", root / "checkpoint_best.pth"]:
        if ck.exists():
            return ck
    fail(f"no checkpoint_final.pth or checkpoint_best.pth in {fold} or {root}")


def model_root(model_dir: Path) -> Path:
    """The directory holding plans.json and dataset.json: a `fold_*` directory's parent, as villa reads it"""
    return model_dir.parent if model_dir.name.startswith("fold_") else model_dir


def inputs(model_dir: Path) -> list[Path]:
    """Every file an export of the nnU-Net folder `model_dir` reads: the checkpoint villa's loader picks, and the
    plans, dataset and fingerprint JSON beside it"""
    root = model_root(model_dir)
    files = [find_checkpoint(model_dir), root / "plans.json", root / "dataset.json"]
    if (root / "dataset_fingerprint.json").exists():
        files.append(root / "dataset_fingerprint.json")
    return files


def load_nnunet(model_dir: Path) -> dict:
    root = model_root(model_dir)
    plans = json.loads((root / "plans.json").read_text())
    dataset = json.loads((root / "dataset.json").read_text())
    fp_path = root / "dataset_fingerprint.json"
    fingerprint = json.loads(fp_path.read_text()) if fp_path.exists() else None
    ck_path = find_checkpoint(model_dir)
    payload = safe_load(ck_path)
    state = payload["network_weights"]
    configuration = payload.get("init_args", {}).get("configuration", "3d_fullres")
    # checked before any tensor is read: another network (PlainConvUNet, say) lacks the ResEnc tensor names, and
    # a missing one must report the checkpoint as unsupported, not as a failed export
    arch = plans.get("configurations", {}).get(configuration, {}).get("architecture", {})
    if arch.get("network_class_name") != ARCH:
        fail(f"network {arch.get('network_class_name')} in configuration {configuration}, only {ARCH}")
    labels = dataset["labels"]
    # nnU-Net's LabelManager: one head per label, minus the ignore label
    num_classes = len(labels) - (1 if "ignore" in labels else 0)
    if any(isinstance(v, (list, tuple)) for v in labels.values()):
        fail("region-based labels")
    prefix, dropped = "", {}
    if any(k.startswith("base.") for k in state):
        # HercUNet (nnUNetTrainer_AffinityMalis_*): the U-Net is `base`, and the affinity head is training only
        # (hercunet/infer/nnunet_infer.py:83-86)
        prefix = "base."
        for k in state:
            if not k.startswith(prefix):
                if not k.startswith("aff_head."):
                    fail(f"tensor {k} outside base. and aff_head.")
                dropped[k] = "affinity head, unused at inference"
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    stem = state["encoder.stem.convs.0.conv.weight"]
    in_channels = int(stem.shape[1])
    declared = len(dataset["channel_names"])
    return {
        "plans": plans, "dataset": dataset, "fingerprint": fingerprint, "checkpoint": ck_path, "state": state,
        "prefix": prefix, "dropped": dropped, "configuration": configuration, "in_channels": in_channels,
        "declared_channels": declared, "num_classes": num_classes, "trainer": payload.get("trainer_name"),
        "mirror_axes": list(payload.get("inference_allowed_mirroring_axes") or []),
    }


#: Socratic's `m7-resenc-l` preset (crossres_pred/voxel/model.py `build_network`), written out here rather
#: than imported, so the export runs none of the downloaded code
SOCRATIC_M7_RESENC_L = {
    "n_stages": 6,
    "features_per_stage": [32, 64, 128, 256, 320, 320],
    "conv_op": "torch.nn.modules.conv.Conv3d",
    "kernel_sizes": [[3, 3, 3]] * 6,
    "strides": [[1, 1, 1]] + [[2, 2, 2]] * 5,
    "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
    "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
    "conv_bias": True,
    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
    "norm_op_kwargs": {"eps": 1e-05, "affine": True},
    "dropout_op": None,
    "dropout_op_kwargs": None,
    "nonlin": "torch.nn.LeakyReLU",
    "nonlin_kwargs": {"inplace": True},
}


def load_socratic(pt: Path) -> dict:
    """Socratic F0: `network.*` of a `.pt` payload, built as the `m7-resenc-l` ResEnc-L its HF config names"""
    root = pt.parents[2] if pt.parent.parent.name == "releases" else pt.parent
    config = json.loads((root / "huggingface" / "f0" / "config.json").read_text())
    mc = config["model_config"]
    if mc.get("preset") != "m7-resenc-l" or mc.get("input_channels") != 1 or mc.get("num_classes") != 2:
        fail(f"Socratic model_config {mc}, only the 1-channel 2-class m7-resenc-l preset")
    payload = safe_load(pt)
    state = payload.get("model", payload)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    prefixes = {k.split(".", 1)[0] for k in state}
    if "network" not in prefixes:
        fail(f"Socratic payload has top-level keys {sorted(prefixes)[:8]}, expected network.*")
    dropped = {k: "training-only tensor outside the network" for k in state if not k.startswith("network.")}
    state = {k[len("network."):]: v for k, v in state.items() if k.startswith("network.")}
    plans = {"configurations": {"socratic": {
        "patch_size": [192, 192, 192],
        "normalization_schemes": [],
        "architecture": {"network_class_name": ARCH, "arch_kwargs": SOCRATIC_M7_RESENC_L,
                         "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]},
    }}}
    return {"config": config, "plans": plans, "configuration": "socratic", "checkpoint": pt, "state": state,
            "prefix": "", "dropped": dropped, "in_channels": 1, "declared_channels": 1, "num_classes": 2,
            "trainer": config.get("model_type"), "mirror_axes": [0, 1, 2]}


def export(model: Path, out_dir: Path, recipe_name: str, name: str | None = None) -> Path:
    missing = missing_packages()
    if missing:
        raise ImportError(f"exporting {model} needs {', '.join(missing)}")
    src = load_socratic(model) if model.suffix == ".pt" else load_nnunet(model)
    plans, configuration = src["plans"], src["configuration"]
    patch = [int(v) for v in plans["configurations"][configuration]["patch_size"]]
    net = build_network(plans, configuration, src["in_channels"], src["num_classes"])
    net.load_state_dict(src["state"], strict=True)
    net.eval()

    w = Walker(net)
    enc, dec = net.encoder, net.decoder
    stem = [w.unit(u) for u in enc.stem.convs] if enc.stem is not None else []
    stages = [[w.block(b) for b in s.blocks] for s in enc.stages]
    decoder = [{"transp": w.conv(t, transposed=True), "convs": [w.unit(u) for u in st.convs]}
               for t, st in zip(dec.transpconvs, dec.stages)]
    if len(decoder) != len(stages) - 1:
        fail(f"{len(decoder)} decoder stages for {len(stages)} encoder stages")
    if getattr(dec, "deep_supervision", False):
        fail("deep supervision on")
    head = w.conv(dec.seg_layers[-1])
    if head["kernel"] != [1, 1, 1] or head["b"] is None:
        fail(f"head {head}")
    for t in dec.transpconvs:
        if t.bias is None:
            fail(f"{w.name(t)}: transposed conv without bias")

    # one tensor per storage; the all_modules.* and decoder.encoder.* names alias them
    sd = net.state_dict()
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

    # tensors the program reads; the deep-supervision seg layers are the only other network tensors allowed
    used: set[str] = set()

    def note(spec: dict | None) -> None:
        if spec is None:
            return
        used.add(spec["w"])
        if spec.get("b"):
            used.add(spec["b"])
        if spec.get("norm"):
            used.update([spec["norm"]["w"], spec["norm"]["b"]])

    for s in stem:
        note(s)
    for stage in stages:
        for b in stage:
            note(b["conv1"]), note(b["conv2"]), note(b["skip"]["proj"])
    for d in decoder:
        note(d["transp"])
        for c in d["convs"]:
            note(c)
    note(head)
    dropped = {src["prefix"] + k: v for k, v in src["dropped"].items()} if src["prefix"] else dict(src["dropped"])
    n_seg = len(dec.seg_layers)
    for key in list(unique):
        if key in used:
            continue
        if any(key.startswith(f"decoder.seg_layers.{i}.") for i in range(n_seg - 1)):
            dropped[src["prefix"] + key] = "deep-supervision segmentation layer, unused at inference"
            del unique[key]
            continue
        fail(f"tensor {key} is neither read by the program nor a known inference-unused tensor")
    missing = used - set(unique)
    if missing:
        fail(f"program reads tensors the network does not hold: {sorted(missing)[:5]}")
    for key in list(aliases):
        if aliases[key] not in unique:
            # an alias of a dropped tensor (all_modules copies of a dropped layer) goes with it
            del aliases[key]

    recipe, extra = recipe_for(recipe_name, plans, configuration, src.get("fingerprint"), patch)
    for k, v in extra.items():
        unique[k] = v.contiguous()
    program = {"stem": stem, "stages": stages, "decoder": decoder, "head": head}
    # resolved, so `.` or `..` names the folder it points to
    name = name or model.resolve().name
    sidecar = {
        "format": FORMAT,
        "name": name,
        "patch_size": patch,
        "in_channels": src["in_channels"],
        "declared_channels": src.get("declared_channels"),
        "num_classes": int(head["cout"]),
        "configuration": configuration,
        "trainer": src.get("trainer"),
        "mirror_axes": src.get("mirror_axes"),
        "recipe": recipe,
        "program": program,
        "checkpoint": str(src["checkpoint"]),
        "checkpoint_sha256": sha256(Path(src["checkpoint"])),
        "keys": {k: list(v.shape) for k, v in unique.items()},
        "aliases": aliases,
        "dropped": dropped,
        "flops_per_patch": 2 * program_macs(program, patch),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    st = out_dir / f"{name}.safetensors"
    save_file(unique, str(st), metadata={"reader": FORMAT, "aliases": json.dumps(aliases)})
    (out_dir / f"{name}.json").write_text(json.dumps(sidecar, indent=1))
    (out_dir / f"{name}.modules.txt").write_text(repr(net))
    n_params = sum(v.numel() for k, v in unique.items() if k not in extra)
    print(f"wrote {st}: {len(unique)} tensors ({len(extra)} recipe), {len(aliases)} aliases, {len(dropped)} dropped, "
          f"{n_params} network params, eps {sorted(w.eps)}, slopes {sorted(w.slopes)}, "
          f"{sidecar['flops_per_patch'] / 1e12:.3f} TFLOP per {'x'.join(map(str, patch))} patch")
    return st


def program_macs(program: dict, patch: list[int]) -> int:
    """Multiply-accumulates of one forward: every conv (output voxels x Cout x Cin x taps), every transposed conv
    (input voxels x Cin x Cout x taps) and the head. The Rust contract computes the same number"""
    shape = list(patch)
    macs = 0

    def conv(spec: dict, shape: list[int]) -> list[int]:
        nonlocal macs
        out = [math.ceil(s / st) for s, st in zip(shape, spec["stride"])]
        macs += math.prod(out) * spec["cout"] * spec["cin"] * math.prod(spec["kernel"])
        return out

    for s in program["stem"]:
        shape = conv(s, shape)
    shapes = []
    for stage in program["stages"]:
        for b in stage:
            if b["skip"]["proj"] is not None:
                pooled = [s // p for s, p in zip(shape, b["skip"]["pool"])]
                conv(b["skip"]["proj"], pooled)
            shape = conv(b["conv1"], shape)
            shape = conv(b["conv2"], shape)
        shapes.append(shape)
    for d, dec in enumerate(program["decoder"]):
        t = dec["transp"]
        macs += math.prod(shape) * t["cin"] * t["cout"] * math.prod(t["kernel"])
        shape = shapes[len(shapes) - 2 - d]
        for c in dec["convs"]:
            shape = conv(c, shape)
    conv(program["head"], shape)
    return macs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--recipe", required=True, choices=["vesuvius-predict", "hercunet", "program-only"])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    missing = missing_packages()
    if missing:
        raise SystemExit(f"the export needs {', '.join(missing)}, which do not import here")
    torch.set_num_threads(2)
    try:
        export(args.model, args.out_dir, args.recipe, args.name)
    except ExportUnsupported as e:
        raise SystemExit(str(e)) from None


if __name__ == "__main__":
    main()
