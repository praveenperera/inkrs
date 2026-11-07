"""P0-3D: villa's eager bf16 forward of one 3D U-Net patch, tapped where `inkrs p0-ink3d` taps, and the comparison

    python -I scripts/p0_ink3d.py dump --ckpt CKPT.pth --volume VOL.zarr --corner z,y,x --out DIR [--crop 64]   (GPU)
    python -I scripts/p0_ink3d.py compare REF_DIR INKRS_DIR [--json OUT]                                     (CPU)

`dump` reads the patch with villa's own reader and normaliser (NativePatchDataset), runs villa's eager model
(`--no-compile`, bf16 autocast) once, and saves the centre cube of: the bf16 model input, the stem output, each
encoder level's last block output, and each decoder stage's output, channels-last float32, plus the logits and
`prob`, villa's `logits.float().sigmoid()`. It runs the forward twice and records whether the taps repeat bit
for bit. `compare` reports per tap: max and mean |diff|, max |diff| over the reference's std, Pearson r and the
fraction of values that differ.

`compare` fails (exit 1) when a manifest is missing, when any reference tap other than `logits` is missing from
the new side or the new side has a tap the reference lacks, on a shape mismatch, an empty or non-finite tap, an
input that is not bit-exact, any tap with r < 0.99 (a gross error, not the parity bar), or `prob` with
r < 0.999 or p99.9 |diff| > 4/255. The parity bar itself is the map gate (`mapgate3d.py`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def dump(args) -> None:
    import torch
    from vesuvius.ink_detection.inference import infer_full3d_tifxyz as F
    from vesuvius.ink_detection.models.checkpoint import config_from_checkpoint, load_checkpoint

    torch.backends.cudnn.benchmark = True
    payload = load_checkpoint(args.ckpt)
    config = config_from_checkpoint(payload, source=args.ckpt)
    a = F.parse_args(["unused_tifxyz", str(args.ckpt), "unused_out", "--no-compile"])
    bundle = F._load_native_model(payload, config, a)
    model = bundle.model
    ps = tuple(config.model.crop_size)
    corner = [int(v) for v in args.corner.split(",")]
    patch = F.PatchSpec(*corner)
    ds = F.NativePatchDataset(tifxyz_dir=Path("unused"), volume_path=str(args.volume), resolution="0",
                              patches=[patch], patch_size_zyx=ps, config=config)
    image, _ = ds[0]
    x = image[None].to("cuda")

    mods = dict(model.named_modules())

    def find(suffix):
        hits = [n for n in mods if n.endswith(suffix)]
        if len(hits) != 1:
            raise SystemExit(f"{suffix}: {hits}")
        return mods[hits[0]]

    enc = find("shared_encoder")
    dec = find("shared_decoder")
    head = find("task_heads.ink")
    taps: dict[str, torch.Tensor] = {}

    def hook(name):
        def f(_m, _i, o):
            taps[name] = o.detach().clone()
        return f

    hooks = []
    if enc.stem is not None:
        hooks.append(enc.stem.register_forward_hook(hook("stem")))
    # encoder levels: a stage's output is its last block's; consecutive stages at one level keep the last
    level, names = 0, []
    for i, st in enumerate(enc.stages):
        blocks = st.blocks if hasattr(st, "blocks") else st
        stride = list(blocks[0].conv1.conv.stride)
        if i > 0 and stride == [2, 2, 2]:
            level += 1
        names.append(f"enc{level}")
    for i, st in enumerate(enc.stages):
        last_at_level = i + 1 == len(names) or names[i + 1] != names[i]
        if last_at_level:
            hooks.append(st.register_forward_hook(hook(names[i])))
    for i, st in enumerate(dec.stages):
        hooks.append(st.register_forward_hook(hook(f"dec{i}")))
    hooks.append(head.register_forward_hook(hook("logits")))

    runs = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=bundle.amp_dtype):
        for _ in range(2):
            taps.clear()
            model(x)
            torch.cuda.synchronize()
            runs.append({k: v.clone() for k, v in taps.items()})
    for h in hooks:
        h.remove()
    repeat = {k: bool(torch.equal(runs[0][k], runs[1][k])) for k in runs[0]}
    taps = runs[0]
    taps = {"input": x.to(bundle.amp_dtype), **taps, "prob": taps["logits"].float().sigmoid()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, t in taps.items():
        t = t[0].float()  # [C, D, H, W]
        e = t.shape[-1]
        k = min(args.crop, e)
        o = (e - k) // 2
        cube = t[:, o:o + k, o:o + k, o:o + k].permute(1, 2, 3, 0).contiguous().cpu().numpy()
        np.save(out / f"{name}.npy", cube)
        manifest.append({"name": name, "edge": e, "crop": k, "offset": o, "channels": int(t.shape[0]),
                         "dtype": str(taps[name].dtype)})
    (out / "manifest.json").write_text(json.dumps({"corner": corner, "patch": ps, "taps": manifest,
                                                   "repeat_bitwise": repeat, "torch": torch.__version__,
                                                   "cudnn": torch.backends.cudnn.version()}, indent=1))
    print(json.dumps({"repeat_bitwise": repeat}))


def compare(args) -> None:
    ref, new = Path(args.ref), Path(args.new)
    fails = []
    for d in (ref, new):
        if not (d / "manifest.json").is_file():
            raise SystemExit(f"{d}: no manifest.json")
    want = {t["name"] for t in json.loads((ref / "manifest.json").read_text())["taps"]} - {"logits"}
    have = {t["name"] for t in json.loads((new / "manifest.json").read_text())["taps"]}
    if want - have:
        fails.append(f"missing taps {sorted(want - have)}")
    if have - want:
        fails.append(f"taps the reference lacks {sorted(have - want)}")
    rows = {}
    for name in sorted(want & have):
        a, b = np.load(new / f"{name}.npy").astype(np.float64), np.load(ref / f"{name}.npy").astype(np.float64)
        if a.shape != b.shape or a.size == 0:
            fails.append(f"{name}: shape {a.shape} vs reference {b.shape}")
            continue
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            fails.append(f"{name}: non-finite values")
            continue
        d = np.abs(a - b)
        sd = float(b.std()) or 1.0
        if a.std() > 0 and b.std() > 0:
            r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        else:
            r = 1.0 if np.array_equal(a, b) else 0.0
        rows[name] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "max_over_std": float(d.max() / sd),
                      "p999_abs": float(np.quantile(d, 0.999)), "r": r, "frac_diff": float((d > 0).mean()),
                      "n": int(a.size)}
        if name == "input" and rows[name]["max_abs"] != 0:
            fails.append(f"input differs (max {rows[name]['max_abs']})")
        if not r >= 0.99:
            fails.append(f"{name}: r {r:.6f} < 0.99")
        if name == "prob" and not (r >= 0.999 and rows[name]["p999_abs"] <= 4 / 255):
            fails.append(f"prob: r {r:.6f}, p99.9 {rows[name]['p999_abs']:.4g}")
    order = ["input", "stem"] + sorted(k for k in rows if k.startswith("enc")) + \
        sorted((k for k in rows if k.startswith("dec")), key=lambda s: int(s[3:])) + ["prob"]
    for k in order:
        if k in rows:
            v = rows[k]
            print(f"{k:8s} max {v['max_abs']:.4g}  mean {v['mean_abs']:.3g}  max/std {v['max_over_std']:.3g}  "
                  f"r {v['r']:.6f}  differ {v['frac_diff']:.3%}")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "fails": fails}, indent=1))
    for f in fails:
        print("FAIL", f)
    raise SystemExit(1 if fails else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--volume", required=True)
    d.add_argument("--corner", required=True)
    d.add_argument("--crop", type=int, default=64)
    d.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("new")
    c.add_argument("--json")
    args = ap.parse_args()
    dump(args) if args.cmd == "dump" else compare(args)


if __name__ == "__main__":
    main()
