"""P0-H gate: inkrs's Hecate activations, layer by layer, against hecate.py's own model

    python -I scripts/p0_hecate.py ref VOLUME HECATE_PY CKPT TILES.json OUT
    python -I scripts/p0_hecate.py compare RUST_DIR TORCH_DIR --mode fp32|tf32|bf16 [--json OUT]
    python -I scripts/p0_hecate.py selftest

`ref` (needs CUDA) builds the model with the pinned hecate.py's `load_model`, gathers each tile of
TILES.json (`{y0, x0, reverse}` on the uint8 ZYX VOLUME, `.npy` or zarr) exactly as `predict` does,
and saves every tap as NCDHW float32 `.npy` in three passes: OUT/fp32 (TF32 off, the accurate
reference), OUT/tf32 (PyTorch's default cuDNN TF32, which hecate.py runs at `--precision fp32`) and
OUT/bf16 (`--precision bf16` autocast). Taps: the normalised input, the stem, layer1-4, each
channel_reduce, each decoder block and `prob`, the tile probability times its column support.

`compare` checks `inkrs p0-hecate --out RUST_DIR` run at the same mode against OUT/fp32, next to
PyTorch's own error at that mode (OUT/<mode> against OUT/fp32), every check counted:
  - each tap exists on both sides with the same shape (channels last) and is finite
  - inkrs's relative RMS error <= RMS_RATIO x PyTorch's, and its p99.9 |diff| <= P999_RATIO x
    PyTorch's, each floored at FLOOR[mode] x the reference RMS (at fp32, PyTorch is its own
    reference, so only the floor applies)
  - the taps are exactly TAPS, once each, non-empty and not all zero; the manifest's precision is
    --mode's and its tile list is the reference's; at bf16 the bf16-stored taps hold bf16 values
  - the input tap, which no conv has touched, is within one unit in the last place of its storage
    format (bf16 or float32) of torch's float32 normalisation
`selftest` runs those checks on synthetic dumps, a faithful one and one per way a dump can be wrong.
Exit 0 when every check passes
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HECATE_PY_SHA256 = "c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59"
RMS_RATIO = 1.5
P999_RATIO = 2.0
# relative error a correct port may show where PyTorch's own error at that mode is near zero:
# float32 summation order (fp32), TF32's 10-bit mantissa (tf32), bf16's 8-bit (bf16)
FLOOR = {"fp32": 1e-4, "tf32": 2e-3, "bf16": 1e-2}
D, P = 16, 64
# every tap both sides must dump, by hecate.py module: the BatchNorm of the input, the stem's ReLU,
# the four encoder layers, the decoder's channel reductions and blocks, and the masked probability.
# The head's logits and attention are not tapped: inkrs computes them inside one kernel, and `prob`,
# which they alone determine from block.2, is checked at every mode
TAPS = ["input", "stem", "layer1", "layer2", "layer3", "layer4", "reduce.0", "reduce.1", "reduce.2", "reduce.3",
        "block.0", "block.1", "block.2", "prob"]
# taps inkrs stores in bf16 under autocast (conv and BatchNorm outputs), whose low 16 bits must be zero
BF16_STORED = ["stem", "layer1", "layer2", "layer3", "layer4"]
# `inkrs p0-hecate`'s manifest records its Precision as serde writes it
MANIFEST_PRECISION = {"fp32": {"Fp32": {"tf32": False}}, "tf32": {"Fp32": {"tf32": True}}, "bf16": "Bf16"}


def load_hecate(path):
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha != HECATE_PY_SHA256:
        raise SystemExit(f"{path}: sha256 {sha}, expected the pinned {HECATE_PY_SHA256}")
    spec = importlib.util.spec_from_file_location("hecate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ref(a):
    import torch

    hecate = load_hecate(a.hecate_py)
    model = hecate.load_model(str(a.ckpt), "cuda")
    volume = hecate.open_volume(str(a.volume))
    tiles = json.loads(Path(a.tiles).read_text())
    depth = volume.shape[0]
    z0 = depth // 2 - D // 2
    patches = []
    for t in tiles:
        zs = slice(depth - z0 - D, depth - z0) if t["reverse"] else slice(z0, z0 + D)
        raw = np.asarray(volume[zs, t["y0"]:t["y0"] + P, t["x0"]:t["x0"] + P])
        if t["reverse"]:
            raw = raw[::-1]
        assert raw.shape == (D, P, P), raw.shape
        patches.append(raw)
    # predict's own preparation of a batch
    raw = torch.from_numpy(np.stack(patches)).cuda()
    support = raw.any(1)
    normalized = np.stack(patches).astype(np.float32)
    normalized /= model.divisor
    image = torch.from_numpy(normalized[:, None]).cuda()
    valid = support[:, None, None].expand(-1, 1, D, -1, -1)

    c = model.canonical
    named = {"input": c.normalization, "stem": c.backbone.relu}
    named.update({f"layer{i}": getattr(c.backbone, f"layer{i}") for i in range(1, 5)})
    named.update({f"reduce.{i}": m for i, m in enumerate(c.decoder.channel_reduce)})
    named.update({f"block.{i}": m for i, m in enumerate(c.decoder.decoder_blocks)})
    out = Path(a.out)
    for mode in ("fp32", "tf32", "bf16"):
        torch.backends.cudnn.allow_tf32 = mode == "tf32"
        torch.backends.cuda.matmul.allow_tf32 = False
        got = {}
        hooks = []
        for name, m in named.items():
            def hook(_m, _i, o, name=name):
                # the stem ReLU module is called once, in ResNet.forward
                assert name not in got, name
                got[name] = o.detach().float().cpu().numpy().copy()
            hooks.append(m.register_forward_hook(hook))
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=mode == "bf16"):
            logits = model(image, valid)
            probability = logits.float().sigmoid()[:, 0]
        got["prob"] = (probability * support).float().cpu().numpy()
        for h in hooks:
            h.remove()
        (out / mode).mkdir(parents=True, exist_ok=True)
        for name, arr in got.items():
            np.save(out / mode / f"{name}.npy", arr)
        print(mode, {k: v.shape for k, v in got.items()})
    (out / "ref.json").write_text(json.dumps({
        "hecate_py_sha256": HECATE_PY_SHA256, "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(), "tiles": tiles,
    }, indent=1))


def channels_last(a):
    return np.moveaxis(a, 1, -1) if a.ndim >= 4 else a


def stats(x, r):
    d = (x.astype(np.float64) - r).ravel()
    rr = float(np.sqrt(np.mean(r.astype(np.float64) ** 2)))
    return dict(rel_rms=float(np.sqrt(np.mean(d ** 2))) / max(rr, 1e-30), p999=float(np.quantile(np.abs(d), 0.999)),
                max=float(np.abs(d).max()), ref_rms=rr, rms=float(np.sqrt(np.mean(x.astype(np.float64) ** 2))))


def bf16_spacing(r):
    """The gap between bf16 neighbours at |r|: float32's, times the 16 mantissa bits bf16 drops"""
    return np.spacing(np.abs(r).astype(np.float32)).astype(np.float64) * 65536.0


def compare_dirs(rust, torch_dir, mode):
    """Every check of `compare`, as (checks, passed, failures, rows); selftest reuses it"""
    checks = passed = 0
    failures, rows = [], []

    def check(ok, what):
        nonlocal checks, passed
        checks += 1
        passed += bool(ok)
        if not ok:
            failures.append(what)

    try:
        manifest = json.loads((rust / "manifest.json").read_text())
        ref_info = json.loads((torch_dir / "ref.json").read_text())
    except (OSError, ValueError) as e:
        check(False, f"manifest or ref.json unreadable: {e}")
        return checks, passed, failures, rows
    names = [t["name"] for t in manifest.get("taps", [])]
    # the run must be the precision under test: an fp32 dump compared as bf16 would pass trivially
    check(manifest.get("precision") == MANIFEST_PRECISION[mode],
          f"manifest precision {manifest.get('precision')!r}, expected {MANIFEST_PRECISION[mode]!r} for --mode {mode}")
    tiles = manifest.get("tile_list")
    # absent on both sides would compare equal, so a tile list is required, not only matched
    check(isinstance(tiles, list) and len(tiles) > 0, f"tile list missing or empty: {tiles!r}")
    check(tiles == ref_info.get("tiles"), "inkrs and torch ran different tile lists")
    n_tiles = len(tiles) if isinstance(tiles, list) else -1
    check(ref_info.get("hecate_py_sha256") == HECATE_PY_SHA256, "torch reference from another hecate.py")
    check(len(names) == len(set(names)), f"duplicate tap names: {names}")
    check(sorted(names) == sorted(TAPS), f"inkrs taps {sorted(names)} are not the required {sorted(TAPS)}")
    for sub in ("fp32", mode):
        have = sorted(p.stem for p in (torch_dir / sub).glob("*.npy"))
        check(have == sorted(TAPS), f"torch {sub} taps {have} are not the required {sorted(TAPS)}")
    for name in TAPS:
        paths = [rust / f"{name}.npy", torch_dir / "fp32" / f"{name}.npy", torch_dir / mode / f"{name}.npy"]
        if not all(p.exists() for p in paths):
            check(False, f"{name}: missing {[str(p) for p in paths if not p.exists()]}")
            continue
        rust_a = np.load(paths[0])
        r32 = channels_last(np.load(paths[1]))
        own = channels_last(np.load(paths[2]))
        check(rust_a.shape == r32.shape == own.shape, f"{name}: shapes {rust_a.shape} {r32.shape} {own.shape}")
        if not rust_a.shape == r32.shape == own.shape:
            continue
        check(rust_a.size > 0, f"{name}: empty")
        if rust_a.size == 0:
            continue
        check(rust_a.shape[0] == n_tiles, f"{name}: batch {rust_a.shape[0]}, but the tile list has {n_tiles} tiles")
        check(np.isfinite(rust_a).all() and np.isfinite(r32).all(), f"{name}: non-finite values")
        s, t = stats(rust_a, r32), stats(own, r32)
        # the fixed CT tiles give non-zero activations everywhere tapped; zeros mean nothing ran
        check(s["ref_rms"] > 0 and s["rms"] > 0, f"{name}: all-zero tap (inkrs rms {s['rms']}, torch {s['ref_rms']})")
        if mode == "bf16" and name in BF16_STORED:
            bits = rust_a.astype(np.float32).view(np.uint32)
            check(not (bits & 0xFFFF).any(), f"{name}: values are not bf16, so the bf16 path did not run")
        if name == "input":
            # the input is the BatchNorm of uint8 / divisor, before any conv; inkrs stores it in the run's
            # precision, the hook sees normalization's float32, so the check is one unit in the last place,
            # not an error budget. bf16 storage: per value, at bf16's spacing. float32: at the spacing of the
            # tap's largest magnitude, since inkrs and cuDNN round the affine's terms in a different order,
            # and one ulp of a term is many ulps of a result that cancels to near zero (the 1447 tiles have
            # a level at 5.4e-4, 14 of its ulps off but 8e-10 absolute, under one ulp of the 7.2 maximum)
            if mode == "bf16":
                spacing = bf16_spacing(r32)
            else:
                spacing = np.full(r32.shape, float(np.spacing(np.float32(np.abs(r32).max()))))
            ulps = np.abs(rust_a.astype(np.float64) - r32) / np.maximum(spacing, 1e-45)
            check(ulps.max() <= 1.0, f"input: {ulps.max():.2f} ulps ({mode} storage) from torch's float32 normalisation")
            rows.append(dict(name=name, inkrs=s, max_ulps=float(ulps.max())))
            print(f"{name:10s} inkrs max {ulps.max():.2f} ulp ({mode} storage)")
            continue
        floor = FLOOR[mode] * s["ref_rms"]
        rms_lim = max(RMS_RATIO * t["rel_rms"], FLOOR[mode])
        p999_lim = max(P999_RATIO * t["p999"], floor)
        ok_rms, ok_p = s["rel_rms"] <= rms_lim, s["p999"] <= p999_lim
        check(ok_rms, f"{name}: rel RMS {s['rel_rms']:.3g} > {rms_lim:.3g}")
        check(ok_p, f"{name}: p99.9 {s['p999']:.3g} > {p999_lim:.3g}")
        rows.append(dict(name=name, inkrs=s, torch=t, rms_limit=rms_lim, p999_limit=p999_lim))
        print(f"{name:10s} inkrs rel {s['rel_rms']:.3g} p999 {s['p999']:.3g} max {s['max']:.3g} | "
              f"torch-{mode} rel {t['rel_rms']:.3g} p999 {t['p999']:.3g} | {'ok' if ok_rms and ok_p else 'FAIL'}")
    return checks, passed, failures, rows


def compare(a):
    checks, passed, failures, rows = compare_dirs(a.rust, a.torch, a.mode)
    print(f"P0-H {a.mode}: {passed}/{checks} checks pass")
    for f in failures:
        print("  FAIL", f)
    if a.json:
        a.json.write_text(json.dumps(dict(mode=a.mode, checks=checks, passed=passed, failures=failures, rows=rows), indent=1))
    sys.exit(0 if passed == checks and checks > 0 else 1)


def selftest(_a):
    """compare_dirs on synthetic dumps: a faithful one passes, and each way a dump can be wrong fails"""
    import shutil
    import tempfile

    rng = np.random.default_rng(0)
    tiles = [{"y0": 0, "x0": 0, "reverse": False}]

    def make(root, mode, mutate=None):
        rust, ref = root / "rust", root / "ref"
        shapes = {n: (1, 4, 2, 2, 3) for n in TAPS}
        shapes["input"] = (1, 4, 2, 2, 1)
        shapes["prob"] = (1, 2, 2)
        taps = []
        for sub in ("fp32", mode):
            (ref / sub).mkdir(parents=True, exist_ok=True)
        rust.mkdir(parents=True, exist_ok=True)
        for n, shp in shapes.items():
            v = rng.standard_normal(shp).astype(np.float32) + 0.5
            # a faithful bf16 run stores bf16 where autocast does, as torch's own bf16 pass does
            r = v
            if mode == "bf16" and (n in BF16_STORED or n == "input"):
                r = (v.view(np.uint32) & 0xFFFF0000).view(np.float32)
            nc = lambda a: np.moveaxis(a, -1, 1) if a.ndim >= 4 else a
            np.save(ref / "fp32" / f"{n}.npy", nc(v))
            if mode != "fp32":
                np.save(ref / mode / f"{n}.npy", nc(r))
            np.save(rust / f"{n}.npy", r)
            taps.append({"name": n, "shape": list(shp)})
        m = {"precision": MANIFEST_PRECISION[mode], "taps": taps, "tile_list": tiles}
        info = {"hecate_py_sha256": HECATE_PY_SHA256, "tiles": tiles}
        after = mutate(rust, ref, m, info) if mutate else None
        (rust / "manifest.json").write_text(json.dumps(m))
        (ref / "ref.json").write_text(json.dumps(info))
        if callable(after):  # a mutation of the written files
            after()
        return rust, ref

    def zero(rust, ref, m, info):
        np.save(rust / "layer2.npy", np.zeros_like(np.load(rust / "layer2.npy")))
        np.save(ref / "fp32" / "layer2.npy", np.zeros_like(np.load(ref / "fp32" / "layer2.npy")))

    def nan(rust, ref, m, info):
        a = np.load(rust / "stem.npy"); a.flat[0] = np.nan; np.save(rust / "stem.npy", a)

    def off_by_much(rust, ref, m, info):
        a = np.load(rust / "block.1.npy"); np.save(rust / "block.1.npy", a * 1.01)

    def input_ulps(k):
        def mutate(rust, ref, m, info):
            a = np.load(rust / "input.npy")
            np.save(rust / "input.npy", (a.view(np.uint32) + np.uint32(k << (16 if m["precision"] == "Bf16" else 0))).view(np.float32))
        return mutate

    def low_bits(rust, ref, m, info):
        # one float32 ulp on a bf16 tap: far inside every error bound, so only the storage check sees it
        a = np.load(rust / "stem.npy"); np.save(rust / "stem.npy", (a.view(np.uint32) | 1).view(np.float32))

    def big_tap(spoil):
        # a 3000-value block.1, so the RMS and the 99.9th percentile can be crossed one at a time
        def mutate(rust, ref, m, info):
            shp = (1, 10, 10, 10, 3)
            v = rng.standard_normal(shp).astype(np.float32) + 0.5
            bad = spoil(v.copy(), FLOOR["bf16" if m["precision"] == "Bf16" else "fp32"] * float(np.sqrt((v.astype(np.float64) ** 2).mean())))
            np.save(ref / "fp32" / "block.1.npy", np.moveaxis(v, -1, 1))
            if (ref / "bf16").exists():
                np.save(ref / "bf16" / "block.1.npy", np.moveaxis(v, -1, 1))
            np.save(rust / "block.1.npy", bad)
            next(t for t in m["taps"] if t["name"] == "block.1")["shape"] = list(shp)
        return mutate

    def rare_outliers(v, floor):
        # 2 values (under 0.1 %) far off: p99.9 stays 0, only the RMS sees it
        v.flat[:2] += 100
        return v

    def wide_offset(v, floor):
        # 1 % of values 3 floors off: p99.9 crosses its floor, the RMS (a tenth of that) stays inside
        v.flat[::100] += 3 * floor
        return v

    def other_shape(rust, ref, m, info):
        a = np.load(rust / "layer1.npy"); np.save(rust / "layer1.npy", a[:, :2])

    def empty_tap(rust, ref, m, info):
        np.save(rust / "layer1.npy", np.zeros((1, 0, 2, 2, 3), np.float32))
        for sub in ("fp32", "bf16"):
            if (ref / sub).exists():
                np.save(ref / sub / "layer1.npy", np.zeros((1, 3, 0, 2, 2), np.float32))

    def no_tiles(rust, ref, m, info):
        del m["tile_list"], info["tiles"]

    def two_tiles(rust, ref, m, info):
        both = [*m["tile_list"], {"y0": 64, "x0": 0, "reverse": False}]
        m["tile_list"], info["tiles"] = both, list(both)

    # each case names the check that must reject it, so a check whose removal leaves every case still
    # rejected by some other check fails the selftest
    cases = {
        "unreadable ref.json": (lambda r, f, m, i: lambda: (f / "ref.json").write_text("{"), "manifest or ref.json unreadable"),
        "tap shape": (other_shape, "layer1: shapes"),
        "empty tap": (empty_tap, "layer1: empty"),
        "empty manifest": (lambda r, f, m, i: m.update(taps=[]), "inkrs taps"),
        "duplicate tap": (lambda r, f, m, i: m["taps"].append(m["taps"][0]), "duplicate tap"),
        "missing torch file": (lambda r, f, m, i: (f / "fp32" / "prob.npy").unlink(), "missing"),
        "extra torch tap": (lambda r, f, m, i: np.save(f / "fp32" / "head.npy", np.ones(3, np.float32)), "torch fp32 taps"),
        "missing inkrs file": (lambda r, f, m, i: (r / "layer3.npy").unlink(), "missing"),
        "zero-filled tap": (zero, "all-zero tap"),
        "non-finite tap": (nan, "non-finite"),
        "precision mismatch": (lambda r, f, m, i: m.update(
            precision=MANIFEST_PRECISION["tf32" if m["precision"] == MANIFEST_PRECISION["fp32"] else "fp32"]),
            "manifest precision"),
        "other tiles": (lambda r, f, m, i: m.update(tile_list=[{"y0": 64, "x0": 0, "reverse": True}]), "different tile lists"),
        "no tile lists": (no_tiles, "tile list missing"),
        "tile count": (two_tiles, "but the tile list has"),
        "other hecate.py": (lambda r, f, m, i: i.update(hecate_py_sha256="0" * 64), "another hecate.py"),
        "1% off": (off_by_much, "block.1:"),
        "rare outliers": (big_tap(rare_outliers), "block.1: rel RMS"),
        "wide offset": (big_tap(wide_offset), "block.1: p99.9"),
        "input 2 ulps": (input_ulps(2), "input:"),
    }
    ok = True
    for mode in ("fp32", "bf16"):
        with tempfile.TemporaryDirectory() as d:
            c, p, fails, _ = compare_dirs(*make(Path(d), mode), mode)
            good = c > 0 and p == c
            print(f"selftest {mode} faithful: {'pass' if good else 'FAIL ' + str(fails)}")
            ok &= good
        # one ulp of input is inside the bound
        with tempfile.TemporaryDirectory() as d:
            c, p, fails, _ = compare_dirs(*make(Path(d), mode, input_ulps(1)), mode)
            print(f"selftest {mode} input 1 ulp: {'pass' if p == c else 'FAIL ' + str(fails)}")
            ok &= p == c
        mode_cases = dict(cases, **({"bf16 tap with float32 bits": (low_bits, "values are not bf16")} if mode == "bf16" else {}))
        for what, (mutate, want) in mode_cases.items():
            with tempfile.TemporaryDirectory() as d:
                c, p, fails, _ = compare_dirs(*make(Path(d), mode, mutate), mode)
                hit = any(want in f for f in fails)
                print(f"selftest {mode} {what}: {'rejected by ' + repr(want) if hit else 'NOT rejected by ' + repr(want) + ' ' + str(fails)}")
                ok &= hit
    # an fp32 dump offered as bf16 fails even though it is closer to the reference
    with tempfile.TemporaryDirectory() as d:
        rust, ref = make(Path(d), "fp32")
        shutil.copytree(ref / "fp32", ref / "bf16")
        c, p, fails, _ = compare_dirs(rust, ref, "bf16")
        print(f"selftest fp32 dump as bf16: {'rejected' if p < c else 'ACCEPTED'}")
        ok &= p < c
    print("selftest", "ok" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ref")
    for k in ("volume", "hecate_py", "ckpt", "tiles", "out"):
        r.add_argument(k)
    c = sub.add_parser("compare")
    c.add_argument("rust", type=Path)
    c.add_argument("torch", type=Path)
    c.add_argument("--mode", choices=sorted(FLOOR), required=True)
    c.add_argument("--json", type=Path)
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"ref": ref, "compare": compare, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
