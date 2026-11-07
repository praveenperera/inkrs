"""inksweep's matched cold and steady passes on one input, in one process, driving an inksweep checkout

    TORCHINDUCTOR_CACHE_DIR=WARM INKSWEEP_SRC=<inksweep c696492>/src python -I \
        scripts/inksweep_matched.py INPUT NAME OUT [--ckpt-dir checkpoints] [--steady 3] [--profile] [--raw]

Settings: inksweep's fast mode (ink9_s42, d -4..4, both orientations, fast mode,
--compile, no ensemble) with --no-raw, so every timed pass writes only the calibrated contract map per
(depth, side), as inkrs writes only its map. Passes, each into its own OUT/pass<k>:
  1        `inksweep.cli.main` in this fresh process: the cold pass (model load, compile wrapper, first calls)
  2..1+N   steady: `Sweep.run_reader(module=...)` with one module loaded and prepared once before pass 2 and
           held across all of them; each pass opens the surface and builds a new Sweep, so it reads the volume
           to the GPU again, as inkrs's later passes do
  profile  with --profile, one more reuse pass with inksweep's StageTimer (CUDA syncs at every stage
           boundary), for the stage split only; its wall is not a speed figure
  raw      with --raw, one more reuse pass with keep_raw on, so bench/pairdiff_inksweep.py can compare the
           blended maps with villa's; not timed for the comparison
Writes OUT/matched.json: the import time (script start to pass 1), the module prepare time, and per pass the
wall time around the call, the volume load, per-map seconds and (profile pass) per-stage seconds
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

T_START = time.perf_counter()
# -I drops PYTHONPATH, so the inksweep tree comes from INKSWEEP_SRC
sys.path.insert(0, os.environ["INKSWEEP_SRC"])

import inksweep.runner as runner  # noqa: E402
from inksweep import cli  # noqa: E402
from inksweep import io as iox  # noqa: E402
from inksweep.sweep import Sweep  # noqa: E402


def per_map(records):
    return [dict(side=m.get("side"), depth=m.get("depth"), tiles=m.get("tiles"), seconds=m.get("seconds"), profile=m.get("profile"))
            for m in records if not m.get("skipped")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("name")
    ap.add_argument("out", type=Path)
    ap.add_argument("--ckpt-dir", default="checkpoints", help="inksweep's checkpoint directory (ink_9um/hybrid_3d2d-seed42/step-075000.pth inside)")
    ap.add_argument("--steady", type=int, default=3)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    args = ["run", a.input, "--name", a.name, "--voxel-um", "8.64", "--ckpt-dir", a.ckpt_dir, "--no-download",
            "--no-ensemble", "--orientations", "both", "--depths=-4..4", "--readers", "ink9_s42", "--mode", "fast", "--compile", "--no-raw"]
    a.out.mkdir(parents=True, exist_ok=True)
    import_s = time.perf_counter() - T_START
    real = runner.run
    captured = {}

    def recording_run(*rargs, **kwargs):
        captured.update(args=rargs, kwargs=kwargs)
        return real(*rargs, **kwargs)

    # cmd_run looks run up on the module at call time, so the wrapper sees the CLI's own arguments
    runner.run = recording_run
    t0 = time.perf_counter()
    rc = cli.main(args + ["--out", str(a.out / "pass1")])
    wall = time.perf_counter() - t0
    runner.run = real
    assert rc == 0, f"pass 1: rc {rc}"
    ink = json.loads((a.out / "pass1" / "ink.json").read_text())
    maps = per_map(ink["maps"])
    assert len(maps) == 18, f"pass 1: {len(maps)} maps"
    passes = [dict(n=1, kind="cold", route="cli.main", wall_s=wall, volume_load_s=ink["timings"].get("volume_load_s"),
                   sweep_wall_s=ink["timings"].get("sweep_wall_s"), total_s=ink["timings"].get("total_s"), maps=maps)]
    print(f"pass 1 (cold, cli.main): wall {wall:.3f}s first map {maps[0]['seconds']:.2f}s", flush=True)

    surface_path, _, reader_names, cfg = captured["args"][:4]
    kw = captured["kwargs"]
    assert cfg.keep_raw is False and cfg.mode == "fast" and cfg.compile, cfg
    specs = runner.make_specs(reader_names, kw.get("plugins", ()), kw.get("overrides"))
    paths, _ = runner.resolve_checkpoints(specs, kw["ckpt_dir"], kw["download"], kw.get("explicit_checkpoints"))
    spec = specs[0]
    device = kw["devices"][0]

    def reuse_pass(k, kind, module, pass_cfg, profile=False):
        t0 = time.perf_counter()
        surface = iox.open_surface(surface_path, voxel_size_um=kw.get("voxel_um"), name=kw.get("name"))
        sweep = Sweep(surface, a.out / f"pass{k}", pass_cfg, device=device, profile=profile)
        load = sweep.load_seconds
        recs = sweep.run_reader(spec, paths[spec.name], module=module)
        del sweep
        wall = time.perf_counter() - t0
        maps = per_map([asdict(r) for r in recs])
        assert len(maps) == 18, f"pass {k}: {len(maps)} maps"
        rec = dict(n=k, kind=kind, route="Sweep.run_reader(module=...)", wall_s=wall, volume_load_s=load, maps=maps)
        if profile:
            stages = {}
            for m in maps:
                for s, v in (m["profile"] or {}).get("seconds", {}).items():
                    stages[s] = stages.get(s, 0.0) + v
            total = sum(stages.values())
            rec["stage_s"] = stages
            rec["stage_share"] = {s: v / total for s, v in stages.items()} if total else None
        print(f"pass {k} ({kind}): wall {wall:.3f}s volume {load:.2f}s first map {maps[0]['seconds']:.2f}s", flush=True)
        return rec

    # one module for every later pass: loaded and prepared once, outside the timed passes, as inkrs keeps its model
    t0 = time.perf_counter()
    # prepare_module reads only self.cfg, so an uninitialised Sweep (no volume read) prepares it
    probe = Sweep.__new__(Sweep)
    probe.cfg = cfg
    module = probe.prepare_module(spec, spec.load(paths[spec.name], device))
    prepare_s = time.perf_counter() - t0
    print(f"module load + prepare {prepare_s:.3f}s", flush=True)
    k = 1
    for _ in range(a.steady):
        k += 1
        passes.append(reuse_pass(k, "steady", module, cfg))
    if a.profile:
        k += 1
        passes.append(reuse_pass(k, "profile", module, cfg, profile=True))
    if a.raw:
        k += 1
        passes.append(reuse_pass(k, "raw", module, replace(cfg, keep_raw=True)))
    steady = sorted(p["wall_s"] for p in passes if p["kind"] == "steady")
    summary = dict(import_s=import_s, cold_s=passes[0]["wall_s"], prepare_s=prepare_s, steady_s=steady,
                   steady_median_s=steady[len(steady) // 2] if steady else None)
    (a.out / "matched.json").write_text(json.dumps(dict(args=args, summary=summary, passes=passes), indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
