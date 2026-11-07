"""Run a published ink ensemble recipe with inkrs in place of its GPU runners, then combine the maps with the
recipe author's own scripts, unchanged

    python -I scripts/recipe.py 343 --recipe RECIPE_CHECKOUT --render R_66.zarr --weights WEIGHTS_DIR --out OUT \
        [--padded R_66_padded.zarr --node-source R_node_source.npz] [--depths -15..15] [--write-depths 1] \
        [--sides reverse|both] [--inkrs target/release/inkrs]
    python -I scripts/recipe.py 306b-fuse --recipe RECIPE_CHECKOUT --out OUT [--render R_41.zarr] \
        (--trace-export TRACE_V3B.safetensors [--trace-crop y0,y1,x0,x1] | --trace fwd_tr.npy) \
        (--ahn S0.safetensors S1.safetensors S2.safetensors | --members A0.npy A1.npy A2.npy)

`343` is Erwin Nieuwlaar's PHerc. 343 First Letters recipe (github.com/Nieuwlaar/pherc343-first-letters, e8370f1,
MIT): four 9 um models (dense_native step 16000, ink_9um seeds 42 and 43 at step 75000, Hecate 9.6 um) on a 66-layer
render at 8.64 um, depths -15..+15, the reading side (villa's `reverse`), a z-score per model pooled over every depth,
and the mean of the four z-scores. Its `run_ink.sh` runs villa 0fb38c45 `infer` (overlap 0.5, hann, batch 4) for the
three villa models and `hecate_prep.py` + `hecate.py` (fp32, batch 4) for Hecate. Here:

- the three villa models run as one `inkrs sweep` per model and render (every depth in one process), which writes
  the maps villa does to within the villa contract (MODELS.md), under the recipe's file names;
- Hecate's 9.6 um input is resampled in this process with `hecate_prep.py`'s arithmetic, from one read of the
  render, and checked bit for bit against the script itself on the first depth; `inkrs hecate` then writes
  `hecate.py`'s PNG at the recipe's precision and batch;
- the tile-phase merge (`merge_tile_phase.py`) and the statistics and ensemble (`ensemble_maps.py`) are the recipe's
  own scripts, run unchanged from RECIPE_CHECKOUT after their SHA-256 is checked, so the combine step is the
  author's by construction.

Only the reading side feeds the ensemble, so `--sides reverse` (the default) skips the blind side; `--sides both`
also writes it, as `run_ink.sh` does. A render with pyramid levels is presented to inkrs as level 0 only, as
`run_ink.sh` presents it to villa, so both skip the same tiles.

`306b-fuse` is the PHerc. 306B product rule (github.com/SirProdigle/pherc0306b-first-letters, `fuse_v7.py`):
float16(float32(TRACE) * float32(A)), A the float16 mean of the A-hn members, NaN where any input is NaN. It runs
the author's `fuse_v7.py` unchanged, so its NaN handling is the method's. The A-hn members come from `--ahn` (one
`inkrs infer RENDER EXPORT OUT.npy --convention 306b` per seed, stored as float16 as the method stores them) or
from `--members` maps of any runner. TRACE's map comes from `--trace-export` (inkrs) or `--trace` (any runner).
With `--trace-export`, this runs `ink_v2.py`'s TRACE member: its row strips (1400 rows plus 192 px of context on
each side, the columns cut to the crop plus 192 px, a strip skipped when layer 20 is all zero), and per strip one
`inkrs sweep --tta 8` over depths -1..+1, forward, whose three maps are averaged as `ink_v2.py` averages them and
cut back to the strip. The result is `fwd_tr.npy`, float16, NaN outside the crop (zero in skipped strips of a
whole-render run, `--trace-crop ""`). `--tta 8` is `villa_map(..., tta8=True)`: two reads, upright and turned a
quarter, each with mirror4 per tile (README.md). The step always runs it: inkrs without TTA runs villa's occupancy
scan, which `villa_map(..., tta8=False)` does not. The export must be of the checkpoint the method's
`checkpoints.tsv` pins.

The weights directory holds the inkrs exports; each model is found by the SHA-256 of the checkpoint its export was
made from, against the recipe's own manifest (`ink/models.tsv`), never by file name. inkrs never bundles weights.
Every stage's wall time goes to OUT/recipe-<stage>.json; `--stage gpu` stops after the maps and `--stage combine`
runs only the merge and ensemble on them, so a queued GPU job holds the GPU only while it computes.
"""
from __future__ import annotations

import argparse
import atexit
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# the recipe scripts this helper runs or ports, at pherc343-first-letters e8370f1
SCRIPTS_343 = {
    "hecate_prep.py": "8249994e6da8d29a1c4005c518450456f570642da03531f10a874ee97f2e4a5a",
    "merge_tile_phase.py": "6eda2605df10dd9e066350f4c9bac8083671f153aa54f2b7bee8f759b043dc20",
    "ensemble_maps.py": "eb5b8b48f5b51605b8896842c8b45018a29772f83ce06482b74e7586b9e24930",
    "pad_render.py": "962c9f7b01b8f5fb99d3aefc75fe56a78bba31f7632747544f0816f5e37d6271",
}
# and the 306B fusion, at pherc0306b-first-letters 3f60827
SCRIPTS_306B = {"fuse_v7.py": "e9cad93bdb81db997796c056e8d88d52b76580d231487183134462a1e7c72f6e"}
# and the TRACE member's runner and its TTA, which the TRACE step ports (strips, crop, depth mean; README.md's spec)
PORTED_306B = {
    "ink_v2.py": "309902dc9577ae6578b240b6717744f1d20cdac454c0432b50eeb161cc2fcc13",
    "x_zoom/zoomlib.py": "c1f574b4699ece27c7c9de05e41a880c0595506d986071985d0c9e72a084a8a2",
    "x_research2/speed/fastink.py": "9e4c2e44a2d7e2f2ed3e460ce26e972efb654fcb171f591b267c72ea14022704",
}
# ink_v2.py's constants: the read layer, the row strip and its context, and the default crop (the claim's context)
TRACE_C = 20
TRACE_DEPTHS = (-1, 0, 1)
TRACE_STRIP = 1400
TRACE_CTX = 192
TRACE_CROP = "116,4167,694,7639"
TRACE_CKPT = "models/community/trace/v3B_best.pth"
N_LAYERS = 66
NATIVE_UM = 8.64
HECATE_UM = 9.6
HECATE_NZ = 16
VILLA_KEYS = ("dnative", "hybrid_3d2d-seed42", "hybrid_3d2d-seed43")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def parse_depths(text: str) -> list[int]:
    a, _, b = text.partition("..")
    lo, hi = int(a), int(b or a)
    if lo > hi:
        raise SystemExit(f"depths {text}: empty range")
    return list(range(lo, hi + 1))


def remove_private(d: Path) -> None:
    """check_scripts' directory, made writable again so its read-only copies can go"""
    if d.exists():
        for q in [d, *(q for q in d.rglob("*") if q.is_dir())]:
            q.chmod(0o700)
        shutil.rmtree(d)


def selftest() -> None:
    """check_scripts runs the bytes it checked: a script changed after the check must not reach the copy, a pinned
    name under a subdirectory (306B's x_zoom/zoomlib.py) is copied to the same path, and a script whose hash is not
    the pin must be refused"""
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "code"
        (src / "a" / "b").mkdir(parents=True)
        good, nested = b"print('checked')\n", b"print('nested')\n"
        (src / "s.py").write_bytes(good)
        (src / "a" / "b" / "n.py").write_bytes(nested)
        pin = {"s.py": hashlib.sha256(good).hexdigest(), "a/b/n.py": hashlib.sha256(nested).hexdigest()}
        copies = check_scripts(Path(t), "code", pin)
        got, got_nested = copies["s.py"], copies["a/b/n.py"]
        assert got_nested.read_bytes() == nested and got_nested.relative_to(got.parent) == Path("a/b/n.py"), got_nested
        remove_private(got.parent)
        assert not got.parent.exists(), "remove_private must remove read-only nested copies"
        copies = check_scripts(Path(t), "code", pin)
        got = copies["s.py"]
        (src / "s.py").write_bytes(b"print('swapped')\n")
        assert got.read_bytes() == good and got.parent != src, "the checked copy must not follow the checkout"
        out = subprocess.run([sys.executable, "-I", got], capture_output=True, text=True, check=True).stdout
        assert out == "checked\n", out
        try:
            check_scripts(Path(t), "code", pin)
        except SystemExit:
            pass
        else:
            raise AssertionError("a script that no longer matches its pin must be refused")
    algos_selftest()
    print("recipe selftest ok")


def algos_selftest() -> None:
    """run_algos keeps a CUDA report with algorithms and a Metal one without, and stops on the rest"""
    cases = [  # backend, report (None: no file), kept
        ("cuda", {"algos": ["conv1 | fwd 1 | 0.2 ms"], "algo_cache_hits": 3}, True),
        ("metal", {"algos": [], "algo_cache_hits": 0}, True),
        ("cuda", {"algos": [], "algo_cache_hits": 0}, False),
        ("cuda", {"algo_cache_hits": 3}, False),
        ("cuda", {"algos": ["a"], "algo_cache_hits": True}, False),
        ("metal", {"algos": ["a"], "algo_cache_hits": 0}, False),
        ("metal", {"algos": [], "algo_cache_hits": 2}, False),
        ("cuda", None, False),
    ]
    with tempfile.TemporaryDirectory() as t:
        for i, (backend, rep, kept) in enumerate(cases):
            f = Path(t) / f"r{i}.json"
            if rep is not None:
                f.write_text(json.dumps(rep))
            try:
                got = run_algos(f, "hecate", ["m.png"], backend)
            except SystemExit:
                assert not kept, f"case {i}: {backend} {rep} refused"
                continue
            assert kept, f"case {i}: {backend} {rep} kept"
            assert got["algos"] == rep["algos"] and got["backend"] == backend, got


def check_scripts(recipe: Path, sub: str, pinned: dict[str, str]) -> dict[str, Path]:
    """The recipe scripts by name, each refused unless its SHA-256 is the pinned one

    Each script is read once and its checked bytes are written to a private directory, whose copy is what runs:
    the recipe's own files are opened again only after a long GPU stage, and a checkout changed in between must not
    run unchecked code. The scripts import nothing from their own directory, so running a copy changes nothing
    """
    out = {}
    private = Path(tempfile.mkdtemp(prefix="recipe-scripts-"))
    atexit.register(remove_private, private)
    for name, want in pinned.items():
        p = recipe / sub / name
        if not p.is_file():
            raise SystemExit(f"{p}: missing; --recipe must be the recipe's checkout")
        data = p.read_bytes()
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise SystemExit(f"{p}: sha256 {got}, this helper is pinned to {want}")
        # a pinned name may be a path under sub (x_zoom/zoomlib.py); its copy keeps that path
        copy = private / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(data)
        copy.chmod(0o400)
        out[name] = copy
    for d in [*(q for q in private.rglob("*") if q.is_dir()), private]:
        d.chmod(0o500)
    return out


def manifest_343(recipe: Path) -> dict[str, set[str]]:
    """Recipe key -> the checkpoint SHA-256s its exports may come from, from ink/models.tsv

    dense_native counts as its original checkpoint, or as the public safetensors (exported with its config)
    """
    rows = {}
    with open(recipe / "ink" / "models.tsv") as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            r = dict(zip(head, line.rstrip("\n").split("\t")))
            rows[r["key"]] = r["sha256"]
    return {
        "dnative": {rows["dnative"], rows["dnative_public_weights"]},
        "hybrid_3d2d-seed42": {rows["hybrid_3d2d-seed42"]},
        "hybrid_3d2d-seed43": {rows["hybrid_3d2d-seed43"]},
        "hecate": {rows["hecate"]},
        "_dnative_config": {rows["dnative_public_config"]},
    }


def find_exports(weights: Path, manifest: dict[str, set[str]]) -> dict[str, Path]:
    """Recipe key -> its export in WEIGHTS, matched by the checkpoint SHA-256 its sidecar records

    An export whose sidecar names the recipe's checkpoint must also be of those bytes still: the sidecar's hash is
    what the export was made from, so a stale export of a replaced file is never taken. Exactly one export per key
    """
    found: dict[str, list[Path]] = {k: [] for k in manifest if not k.startswith("_")}
    for side in sorted(weights.glob("*.json")):
        try:
            meta = json.loads(side.read_text())
        except (OSError, ValueError):
            continue
        sha = meta.get("checkpoint_sha256") if isinstance(meta, dict) else None
        st = side.with_suffix(".safetensors")
        if not sha or not st.is_file():
            continue
        for key in found:
            if sha in manifest[key]:
                # the sidecar must describe these tensor bytes, not an earlier export's
                want = meta.get("safetensors_sha256")
                if want is None:
                    raise SystemExit(f"{side}: no safetensors_sha256; re-export it with this version's exporter")
                if sha256(st) != want:
                    raise SystemExit(f"{st}: its bytes are not the ones {side} was written for; re-export")
                # the public dense_native weights only count with the public config they were trained under
                if key == "dnative" and meta.get("config_json") is not None \
                        and meta.get("config_json_sha256") not in manifest["_dnative_config"]:
                    raise SystemExit(f"{side}: dense_native exported with a config other than the recipe's")
                found[key].append(st)
    missing = [k for k, v in found.items() if not v]
    if missing:
        raise SystemExit(f"no export in {weights} of the recipe's {missing}; export them first (MODELS.md)")
    dup = {k: [str(p) for p in v] for k, v in found.items() if len(v) > 1}
    if dup:
        raise SystemExit(f"more than one export of the same checkpoint in {weights}: {dup}")
    return {k: v[0] for k, v in found.items()}


def level0(render: Path, tmp: Path) -> Path:
    """The render as villa saw it in the recipe: a group holding array `0` only"""
    if not (render / "0" / ".zarray").is_file():
        raise SystemExit(f"{render}: not a zarr v2 group with array 0")
    if not (render / "1").exists():
        return render
    # made afresh each run, so a link left by another render of the same name is never read
    link = tmp / f"{render.name}-level0.zarr"
    shutil.rmtree(link, ignore_errors=True)
    link.mkdir(parents=True)
    (link / ".zgroup").write_text('{"zarr_format": 2}')
    (link / "0").symlink_to((render / "0").resolve())
    return link


class Timer:
    def __init__(self):
        self.stages: dict[str, float] = {}

    def run(self, name: str, fn, *a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t0


def run(cmd: list, log: Path, env: dict | None = None) -> None:
    with open(log, "a") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed ({r.returncode}); see {log}")


class HecatePrep:
    """`hecate_prep.py`'s arithmetic on one resident read of the render

    The script reads the whole render for every depth; this reads it once. The maths below is the script's, line
    for line (`hecate_prep.py:56-71`): planes at z_center + (i - 7.5) * 9.6 / 8.64, linear in depth from the two
    neighbouring layers in float32, then cv2.resize to round(0.9 H) x round(0.9 W), INTER_AREA, clip, uint8
    """

    def __init__(self, render: Path):
        import zarr

        g = zarr.open(str(render), mode="r")
        a = g if hasattr(g, "shape") else g["0"]
        self.vol = np.asarray(a[:])
        if self.vol.ndim != 3 or self.vol.dtype != np.uint8:
            raise SystemExit(f"{render}: shape {self.vol.shape} {self.vol.dtype}, need uint8 [Z, Y, X]")

    def plane_stack(self, z_center: float) -> np.ndarray:
        import cv2

        vol = self.vol
        d, h, w = vol.shape
        step = HECATE_UM / NATIVE_UM
        pos = z_center + (np.arange(HECATE_NZ) - (HECATE_NZ - 1) / 2.0) * step
        if pos.min() < 0 or pos.max() > d - 1:
            raise SystemExit(f"z window {pos.min():.2f}..{pos.max():.2f} lies outside 0..{d - 1}")
        lo = np.floor(pos).astype(int)
        frac = (pos - lo).astype(np.float32)
        hi = np.minimum(lo + 1, d - 1)
        zs = vol[lo].astype(np.float32) * (1 - frac)[:, None, None] + vol[hi].astype(np.float32) * frac[:, None, None]
        s = NATIVE_UM / HECATE_UM
        nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        interp = cv2.INTER_AREA if s < 0.95 else cv2.INTER_LINEAR
        out = np.empty((HECATE_NZ, nh, nw), np.uint8)
        for i in range(HECATE_NZ):
            out[i] = np.clip(cv2.resize(zs[i], (nw, nh), interpolation=interp), 0, 255).astype(np.uint8)
        return out


def zc_of(d: int) -> float:
    return (N_LAYERS - 1) / 2 + d


def recipe_343(a) -> None:
    out = a.out.resolve()
    tmp = out / "tmp"
    logs = out / "logs"
    for p in (tmp, logs):
        p.mkdir(parents=True, exist_ok=True)
    scripts = check_scripts(a.recipe, "ink/scripts", SCRIPTS_343)
    exports = find_exports(a.weights, manifest_343(a.recipe))
    depths = parse_depths(a.depths)
    for d in depths:
        s = N_LAYERS // 2 + d - 8
        if s < 0 or s + 17 > N_LAYERS:
            raise SystemExit(f"d={d} gives layers {s}..{s + 16}, outside 0..{N_LAYERS - 1}")
    if (a.padded is None) != (a.node_source is None):
        raise SystemExit("--padded and --node-source go together (the tile-phase merge needs both)")
    timer = Timer()
    py = sys.executable
    # absolute, not resolved: the render's file name is the maps' name prefix, and may be a symlink
    renders = {"plain": a.render.absolute()}
    if a.padded is not None:
        renders["padded"] = a.padded.absolute()
    rec = {
        "recipe": "pherc343-first-letters e8370f1",
        "render": str(renders["plain"]),
        "padded": str(renders.get("padded")),
        "depths": [depths[0], depths[-1]],
        "sides": a.sides,
        "exports": {k: str(v) for k, v in exports.items()},
        "inkrs": str(a.inkrs),
        "hecate": {"precision": "fp32", "batch_size": a.hecate_batch},
        "sweep": {"batch_size": a.batch_size, "stride": 64},
        "stage": a.stage,
        "ptx_tier_env": os.environ.get("INKRS_PTX_TIER"),
        # each inkrs process's cuDNN algorithms and cache hits, keyed by the maps it wrote: with --algo find on an
        # empty cache a process times them afresh and may pick others than another build did, so a byte comparison
        # between builds needs them to tell a find-timed difference from a code change
        "runs": [],
    }
    # Hecate's JSON names the checkpoint file it was given, as hecate.py's does, so inkrs gets the `.pth` the
    # export was made from when it is present (inkrs finds the export through INKRS_WEIGHTS_DIR and checks its
    # SHA-256), else the export itself
    side_ck = Path(json.loads(exports["hecate"].with_suffix(".json").read_text()).get("checkpoint") or "")
    hecate_ck = side_ck if side_ck.suffix == ".pth" and side_ck.is_file() else exports["hecate"]
    if hecate_ck != side_ck:
        print(f"note: {side_ck} is not present, so Hecate's JSON names {exports['hecate'].name}, where hecate.py's "
              f"would name {side_ck.name}; the maps are the same", file=sys.stderr)
    rec["hecate"]["checkpoint_name"] = Path(hecate_ck).name
    env = dict(os.environ, INKRS_WEIGHTS_DIR=str(a.weights.resolve()))
    t_all = time.perf_counter()
    direction = "reverse" if a.sides == "reverse" else "both"
    maps_dirs = {tag: (out / f"maps_{tag}", render.name.removesuffix(".zarr")) for tag, render in renders.items()}
    for tag, render in renders.items():
        if a.stage == "combine":
            break
        mdir, name = maps_dirs[tag]
        mdir.mkdir(exist_ok=True)
        rin = level0(render, tmp)
        # Hecate's inputs are resampled on the CPU while the GPU runs the villa models
        # with --hecate-inputs the resampled inputs are hecate_prep.py's own, made before (a GPU job then holds the
        # GPU only for inference); otherwise they are resampled here, on CPU threads beside the GPU
        prep = None if a.hecate_inputs else HecatePrep(render)
        with cf.ThreadPoolExecutor(max_workers=a.prep_workers) as pool:
            futures = {} if prep is None else {
                d: pool.submit(timer.run, f"hecate-prep-{tag}", prep.plane_stack, zc_of(d)) for d in depths}
            for key in VILLA_KEYS:
                sweep_out = tmp / f"sweep-{tag}-{key}"
                shutil.rmtree(sweep_out, ignore_errors=True)
                timer.run(f"sweep-{tag}", run, [a.inkrs, "sweep", rin, "--ckpt", exports[key],
                                                f"--depths={depths[0]}..{depths[-1]}", "--direction", direction,
                                                "--stride", "64", "--batch-size", str(a.batch_size), "--out", sweep_out,
                                                "--name", name] + (["--check-finite"] if a.check_finite else []),
                          logs / f"sweep-{tag}-{key}.log")
                reader = json.loads(exports[key].with_suffix(".json").read_text())["reader"]
                moved = []
                for d in depths:
                    s = N_LAYERS // 2 + d - 8
                    for suf in ("_reverse",) + (("",) if a.sides == "both" else ()):
                        src = sweep_out / "ink" / f"{name}__{reader}__L{N_LAYERS}s{s}{suf}.tif"
                        if not src.is_file():
                            raise SystemExit(f"inkrs sweep wrote no {src}")
                        os.replace(src, mdir / f"{name}__{key}__L{N_LAYERS}s{s}{suf}.tif")
                        moved.append(f"{mdir.name}/{name}__{key}__L{N_LAYERS}s{s}{suf}.tif")
                rec["runs"].append(run_algos(sweep_out / "inkrs.json", "sweep", moved))
                shutil.rmtree(sweep_out)
            first = True
            for d in depths:
                zc = f"{zc_of(d):.2f}"
                if prep is None:
                    npy = a.hecate_inputs / f"{tag}-{zc}.npy"
                    if not npy.is_file():
                        raise SystemExit(f"--hecate-inputs: no {npy}")
                    head = np.load(npy, mmap_mode="r")
                    if head.ndim != 3 or head.shape[0] != HECATE_NZ or head.dtype != np.uint8:
                        raise SystemExit(f"--hecate-inputs: {npy} is {head.dtype} {head.shape}, not uint8 "
                                         f"({HECATE_NZ}, H, W)")
                    del head
                    stack = None
                else:
                    stack = futures[d].result()
                    npy = tmp / f"hecate-{tag}-{zc}.npy"
                    np.save(npy, stack)
                if first and a.check_prep and stack is not None:
                    timer.run("hecate-prep-check", check_prep, py, scripts["hecate_prep.py"], render, zc, stack, tmp,
                              logs / "prep-check.log")
                    first = False
                base = mdir / f"{name}__hecate__L{N_LAYERS}zc{zc}"
                target = base.with_name(base.name + ("_reverse.png" if a.sides == "reverse" else ".png"))
                for stale in (target, target.with_suffix(".json")):
                    stale.unlink(missing_ok=True)
                if a.sides == "both":
                    for stale in (base.with_name(base.name + "_reverse.png"), base.with_name(base.name + "_reverse.json")):
                        stale.unlink(missing_ok=True)
                side = ["--reverse"] if a.sides == "reverse" else ["--both"]
                report = tmp / f"hecate-{tag}-{zc}.report.json"
                report.unlink(missing_ok=True)
                timer.run(f"hecate-{tag}", run, [a.inkrs, "hecate", "--checkpoint", hecate_ck, "--input", npy,
                                                 "--output", target, "--spacing-um", str(HECATE_UM), "--precision",
                                                 "fp32", "--batch-size", str(a.hecate_batch), "--report",
                                                 report] + side,
                          logs / f"hecate-{tag}.log", env=env)
                pngs = [target] + ([base.with_name(base.name + "_reverse.png")] if a.sides == "both" else [])
                rec["runs"].append(run_algos(report, "hecate", [f"{mdir.name}/{p.name}" for p in pngs]))
                report.unlink()
                if prep is not None:
                    npy.unlink()
        del prep
    if a.stage == "gpu":
        return finish(out, rec, timer, t_all)
    # the combine: the recipe's own scripts
    if a.padded is not None:
        (plain_dir, plain_name), (pad_dir, pad_name) = maps_dirs["plain"], maps_dirs["padded"]
        merged = out / "maps"
        merged.mkdir(exist_ok=True)
        timer.run("merge", run, [py, "-I", scripts["merge_tile_phase.py"], "--plain", plain_dir, "--plain-name",
                                 plain_name, "--padded", pad_dir, "--padded-name", pad_name, "--node-source",
                                 a.node_source, "--out", merged, "--out-name", a.name or plain_name],
                  logs / "merge.log")
        ens_name = a.name or plain_name
        maps = merged
    else:
        maps, ens_name = maps_dirs["plain"]
    images = out / "images"
    images.mkdir(exist_ok=True)
    timer.run("ensemble", run, [py, "-I", scripts["ensemble_maps.py"], "--surface", renders["plain"], "--name", ens_name,
                                # one token: before Python 3.14 argparse reads a separate "-2..2" as an option
                                "--maps", maps, "--out", images, f"--stat-depths={depths[0]}..{depths[-1]}",
                                "--write-depths", *[str(d) for d in a.write_depths]], logs / "ensemble.log")
    finish(out, rec, timer, t_all)


def run_algos(report: Path, kind: str, maps: list[str], backend: str | None = None) -> dict:
    """One inkrs process's cuDNN algorithm choices from its report, for the maps it wrote; a report without them
    stops the run, so the record never silently lacks a process. A CUDA process plans at least one algorithm; the
    Metal backend (every macOS build) plans none and reports an empty list and no cache hits"""
    backend = backend or ("metal" if sys.platform == "darwin" else "cuda")
    if not report.is_file():
        raise SystemExit(f"inkrs {kind} wrote no report {report}")
    r = json.loads(report.read_text())
    algos, hits = r.get("algos"), r.get("algo_cache_hits")
    if not isinstance(algos, list) or not isinstance(hits, int) or isinstance(hits, bool):
        raise SystemExit(f"{report}: no algos list or algo_cache_hits count")
    if backend == "cuda" and not algos:
        raise SystemExit(f"{report}: a CUDA run with no cuDNN algorithms")
    if backend == "metal" and (algos or hits):
        raise SystemExit(f"{report}: a Metal run reporting cuDNN algorithms {algos[:3]} or {hits} cache hits")
    return {"kind": kind, "backend": backend, "maps": maps, "algos": algos, "algo_cache_hits": hits}


def finish(out: Path, rec: dict, timer: Timer, t_all: float) -> None:
    rec["seconds"] = {k: round(v, 3) for k, v in timer.stages.items()}
    rec["seconds"]["total"] = round(time.perf_counter() - t_all, 3)
    (out / f"recipe-{rec['stage']}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec["seconds"]))


def check_prep(py: str, script: Path, render: Path, zc: str, stack: np.ndarray, tmp: Path, log: Path) -> None:
    """The in-process resample must equal `hecate_prep.py`'s output bit for bit"""
    import zarr

    ref = tmp / f"prep-check-{zc}.zarr"
    shutil.rmtree(ref, ignore_errors=True)
    run([py, "-I", script, "--src", render, "--out", ref, "--native-um", str(NATIVE_UM), "--native-um-z",
         str(NATIVE_UM), "--target-um", str(HECATE_UM), "--z-center", zc, "--nz", str(HECATE_NZ)], log)
    want = np.asarray(zarr.open(str(ref), mode="r")["0"][:])
    if want.shape != stack.shape or not np.array_equal(want, stack):
        raise SystemExit(f"the in-process Hecate resample differs from hecate_prep.py at zc {zc}")
    shutil.rmtree(ref)


def trace_checkpoint_sha(recipe: Path) -> str:
    """The SHA-256 the method's `checkpoints.tsv` pins for TRACE v3B"""
    rows = [r.split() for r in (recipe / "submission/docker/checkpoints.tsv").read_text().splitlines()]
    shas = [r[-1] for r in rows if r and r[0] == TRACE_CKPT]
    if len(shas) != 1:
        raise SystemExit(f"checkpoints.tsv pins {len(shas)} rows for {TRACE_CKPT}, want 1")
    return shas[0]


def check_export(export: Path, want: str) -> None:
    """The export must be made from the checkpoint `want` hashes, and be the bytes its sidecar describes"""
    side = export.with_suffix(".json")
    meta = json.loads(side.read_text())
    if meta.get("checkpoint_sha256") != want:
        raise SystemExit(f"{side}: made from {meta.get('checkpoint_sha256')}, the method pins {want}")
    if meta.get("safetensors_sha256") is None or sha256(export) != meta["safetensors_sha256"]:
        raise SystemExit(f"{export}: its bytes are not the ones {side} was written for; re-export")


def write_strip(path: Path, vol: np.ndarray) -> None:
    """A zarr v2 group with the uncompressed array 0, which `inkrs sweep` reads"""
    import zarr

    shutil.rmtree(path, ignore_errors=True)
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    g.create_array("0", data=vol, chunks=(vol.shape[0], 256, 256), compressors=None, filters=None)


def trace_306b(a, out: Path) -> None:
    """`ink_v2.py --models tr --crop CROP` with each strip's maps from `inkrs sweep`; writes `out` (fwd_tr.npy)

    The strip loop is `ink_v2.py:52-64`, line for line; each strip's map is `run_tile`'s
    `np.mean([villa_map(..., d, 'forward', tta8=True) for d in (-1, 0, 1)], 0)` with `villa_map`'s float32
    map from inkrs, so the float32 mean and its float16 cast are the method's own operations
    """
    import zarr

    arr = zarr.open(str(a.render), mode="r")["0"]
    depth, h, w = arr.shape
    # sweep reads depth d at layers Z // 2 + d - 8 forward, ink_v2.py at c + d - 8
    if depth // 2 != TRACE_C:
        raise SystemExit(f"{a.render}: {depth} layers; the TRACE step reads layer {TRACE_C} as the centre, a 41-layer render")
    crop = a.trace_crop
    cy0, cy1, cx0, cx1 = (int(x) for x in crop.split(",")) if crop else (0, h, 0, w)
    if not (0 <= cy0 < cy1 <= h and 0 <= cx0 < cx1 <= w):
        raise SystemExit(f"--trace-crop {crop}: outside the render's {h} x {w}")
    mean = np.zeros((h, w), np.float32) if not crop else np.full((h, w), np.nan, np.float32)
    q0, q1 = max(0, cx0 - TRACE_CTX), min(w, cx1 + TRACE_CTX)
    tmp = a.out / "trace-strips"
    tmp.mkdir(parents=True, exist_ok=True)
    reader = json.loads(a.trace_export.with_suffix(".json").read_text())["reader"]
    for y0 in range(cy0, cy1, TRACE_STRIP):
        y1 = min(cy1, y0 + TRACE_STRIP)
        r0, r1 = max(0, y0 - TRACE_CTX), min(h, y1 + TRACE_CTX)
        vol = np.asarray(arr[:, r0:r1, q0:q1])
        if not (vol[TRACE_C] > 0).any():
            continue
        strip, maps = tmp / f"rows{y0}.zarr", tmp / f"rows{y0}"
        write_strip(strip, vol)
        shutil.rmtree(maps, ignore_errors=True)
        run([a.inkrs, "sweep", strip, "--ckpt", a.trace_export, f"--depths={TRACE_DEPTHS[0]}..{TRACE_DEPTHS[-1]}",
             "--direction", "forward", "--tta", "8", "--out", maps, "--name", "strip"], a.out / "trace.log")
        ps = []
        for d in TRACE_DEPTHS:
            f = maps / "ink" / f"strip__{reader}__L{depth}s{TRACE_C + d - 8}.npy"
            ps.append(np.load(f))
            want = (r1 - r0, q1 - q0)
            if ps[-1].dtype != np.float32 or ps[-1].shape != want:
                raise SystemExit(f"{f}: {ps[-1].dtype} {ps[-1].shape}, want float32 {want}")
        p = np.mean(ps, 0)
        mean[y0:y1, cx0:cx1] = p[y0 - r0:y1 - r0, cx0 - q0:cx1 - q0]
        shutil.rmtree(strip)
        shutil.rmtree(maps)
    tmp.rmdir()
    np.save(out, mean.astype(np.float16))


def fuse_306b(a) -> None:
    scripts = check_scripts(a.recipe, "submission/code", SCRIPTS_306B)
    a.out.mkdir(parents=True, exist_ok=True)
    timer, t_all = Timer(), time.perf_counter()
    if (a.trace_export or a.ahn) and a.render is None:
        raise SystemExit("--trace-export and --ahn run on the render: pass --render")
    if a.trace_export:
        # the strip loop and depth mean are ported, so the scripts they come from are pinned like the ones run
        check_scripts(a.recipe, "submission/code", PORTED_306B)
        check_export(a.trace_export, trace_checkpoint_sha(a.recipe))
        a.trace = a.out / "fwd_tr.npy"
        timer.run("trace", trace_306b, a, a.trace)
    if a.ahn:
        # the A-hn members from inkrs: one `inkrs infer` per seed, one after another (the engine holds GPU 0), each
        # export's sidecar fixing the v8-in architecture and the 306b convention (layers 8..31 in stored order).
        # Each map is stored as float16, the members' dtype in the method, NaN where no tile ran; fuse_v7.py
        # then does all the NaN handling and arithmetic itself
        members = []
        for i, export in enumerate(a.ahn):
            raw, m16 = a.out / f"ahn{i}.f32.npy", a.out / f"fwd_ahn{i}.npy"
            timer.run("ahn", run, [a.inkrs, "infer", a.render, export, raw, "--convention", "306b"], a.out / f"ahn{i}.log")
            p = np.load(raw)
            if p.dtype != np.float32 or p.ndim != 2:
                raise SystemExit(f"{raw}: {p.dtype} {p.shape}, want a float32 [H, W] map")
            np.save(m16, p.astype(np.float16))
            raw.unlink()
            members.append(m16)
        a.members = members + list(a.members or [])
    if not a.members:
        raise SystemExit("no A-hn members: pass --ahn exports or --members maps")
    # fuse_v7.py takes OUT as its first positional argument
    timer.run("fuse", run, [sys.executable, "-I", scripts["fuse_v7.py"], a.out, "--trace", a.trace, "--members",
                            *a.members], a.out / "fuse.log")
    seconds = {k: round(v, 3) for k, v in timer.stages.items()}
    seconds["total"] = round(time.perf_counter() - t_all, 3)
    (a.out / "recipe-306b-fuse.json").write_text(json.dumps({"seconds": seconds}, indent=1))
    print(json.dumps(seconds))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("343")
    p.add_argument("--recipe", type=Path, required=True, help="pherc343-first-letters checkout (e8370f1)")
    p.add_argument("--render", type=Path, required=True, help="the 66-layer render, zarr v2 group with array 0")
    p.add_argument("--padded", type=Path, help="the same render through pad_render.py, for the tile-phase merge")
    p.add_argument("--node-source", type=Path, help="mesh/build_concat.py's node_source.npz, for the merge")
    p.add_argument("--weights", type=Path, required=True, help="directory of inkrs exports")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", help="output name prefix; default the plain render's stem")
    p.add_argument("--depths", default="-15..15")
    p.add_argument("--write-depths", nargs="+", type=int, default=[1])
    p.add_argument("--sides", choices=("reverse", "both"), default="reverse")
    p.add_argument("--inkrs", type=Path, default=Path("target/release/inkrs"))
    p.add_argument("--batch-size", type=int, default=64, help="inkrs sweep batch")
    p.add_argument("--hecate-batch", type=int, default=4, help="the recipe's HEC_BATCH")
    p.add_argument("--prep-workers", type=int, default=2)
    p.add_argument("--no-check-prep", dest="check_prep", action="store_false",
                   help="skip the bit-for-bit check of the first Hecate resample against hecate_prep.py")
    p.add_argument("--check-finite", action="store_true")
    p.add_argument("--hecate-inputs", type=Path,
                   help="hecate_prep.py's outputs as <plain|padded>-<zc>.npy (written beside hecate_prep.py's zarr v3 output), used "
                        "instead of resampling here")
    p.add_argument("--stage", choices=("all", "gpu", "combine"), default="all",
                   help="gpu: the maps only; combine: the merge and ensemble on maps an earlier gpu stage wrote")
    p.set_defaults(fn=recipe_343)
    q = sub.add_parser("306b-fuse")
    q.add_argument("--recipe", type=Path, required=True, help="pherc0306b-first-letters checkout")
    t = q.add_mutually_exclusive_group(required=True)
    t.add_argument("--trace", type=Path, help="TRACE v3B's map from another runner, .npy on the render's raster")
    t.add_argument("--trace-export", type=Path, help="TRACE v3B's inkrs export, run here as ink_v2.py runs it")
    q.add_argument("--trace-crop", default=TRACE_CROP,
                   help='ink_v2.py --crop: y0,y1,x0,x1 render px, default the claim\'s context ("" = whole render)')
    q.add_argument("--members", type=Path, nargs="+", default=[], help="A-hn member maps from another runner")
    q.add_argument("--ahn", type=Path, nargs="+", default=[], help="A-hn seed exports for `inkrs infer` (v8-in)")
    q.add_argument("--render", type=Path, help="the 41-layer render (zarr v2 group, array 0), for --trace-export and --ahn")
    q.add_argument("--inkrs", type=Path, default=Path("target/release/inkrs"))
    q.add_argument("--out", type=Path, required=True)
    q.set_defaults(fn=fuse_306b)
    t = sub.add_parser("selftest", help="check that the recipe scripts run are the bytes checked")
    t.set_defaults(fn=lambda _a: selftest())
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
