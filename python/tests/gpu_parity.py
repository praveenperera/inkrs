"""GPU check: the Python API gives the CLI's maps, byte for byte

    python gpu_parity.py ZARR_GROUP WEIGHTS CLI_INK_DIR OUT_DIR [--name NAME] [--algo heuristic] [--informational]

CLI_INK_DIR holds `inkrs sweep ZARR_GROUP --ckpt WEIGHTS --depths=-4..4 --direction both --stride 64 --name NAME
--algo ALGO`'s ink/ directory. Use `heuristic` on both sides: `find` times cuDNN's algorithms in each process and
can pick a different one in another process, which moves a few pixels by 1/255 between any two processes, CLI or
not. This script runs, at batch 64:

0. checks that the engine loaded the cuDNN version the CLI did (CLI_INK_DIR/../inkrs.json) and made the same cuDNN
   algorithm choice for every layer, so a difference below is the API's and not the libraries';
1. `Engine.sweep(path, out_dir=...)`, whose TIFFs must equal the CLI's byte for byte;
2. `Engine.sweep(path)` in memory, whose maps must equal the CLI's decoded TIFFs;
3. `Engine.infer` for d = 0, both directions, on the level-0 array in memory, with the group's pyramid passed as
   `occupancy` (a path, then the array of the level villa scans), which must equal the CLI's d = 0 maps;
4. `Engine.infer` on the array with `occupancy=None`, whose covered pixels may only shrink or stay: it reports the
   count of pixels that differ, since villa's low-resolution skip is then off;
5. a second engine, made and released on another thread three times while the first stays open, must give back
   its device memory (the release binds the engine's context on that thread).

Every check compares the full set of maps, so a missing or extra map fails. Writes OUT_DIR/parity.json and exits 1
on any failure in 0-3 or 5. --informational is for a run on other libraries (INKRS_CUDA_LIBS=cudnn, the pip cuDNN):
it runs 2. only, still fails on a missing map, and records pixel differences and the libraries without failing,
since another cuDNN may choose other algorithms. It is not part of the byte-identity acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import tifffile
import zarr

import inkrs


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr", type=Path)
    ap.add_argument("weights", type=Path)
    ap.add_argument("cli_ink", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--name", default="s1")
    ap.add_argument("--algo", default="heuristic", choices=["heuristic", "find"])
    ap.add_argument("--informational", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "inkrs": inkrs.__version__,
        "build": inkrs.build_info(),
        "acceptance": not args.informational,
        "checks": {},
    }
    ok = True

    t = time.perf_counter()
    eng = inkrs.Engine(args.weights, batch_size=64, algo=args.algo)
    result["engine_s"] = time.perf_counter() - t
    result["load_report"] = eng.load_report

    # 0. the same libraries and cuDNN choices as the CLI run being compared against
    cli_report = json.loads((args.cli_ink.parent / "inkrs.json").read_text())
    want_algos = [list(a) for a in cli_report["algos"]]
    got_algos = [list(a) for a in eng.load_report["algos"]]
    same_libs = cli_report["cudnn"] == eng.load_report["cudnn"] and want_algos == got_algos
    result["checks"]["libraries"] = {
        "identical": same_libs,
        "cudnn": [cli_report["cudnn"], eng.load_report["cudnn"]],
        "algos_differ": [w[0] for w, g in zip(want_algos, got_algos) if w != g] + (["count"] if len(want_algos) != len(got_algos) else []),
        "cuda_libraries": eng.cuda_libraries,
    }
    print(f"libraries: cuDNN {cli_report['cudnn']} vs {eng.load_report['cudnn']}, algos {'same' if want_algos == got_algos else 'differ'}")
    if not args.informational:
        ok &= same_libs
    depth = int(json.loads((args.zarr / "0" / ".zarray").read_text())["shape"][0])
    cli = {p.name: p for p in sorted(args.cli_ink.glob("*.tif"))}
    if len(cli) != 18:
        print(f"expected 18 CLI maps in {args.cli_ink}, found {len(cli)}", file=sys.stderr)
        return 1

    def fname(d: int, direction: str) -> str:
        s = depth // 2 + d - 8
        return f"{args.name}__{eng.reader}__L{depth}s{s}{'_reverse' if direction == 'reverse' else ''}.tif"

    cli_maps = {name: tifffile.imread(path) for name, path in cli.items()}

    def compare(maps: dict, label: str, want_keys: set) -> bool:
        diffs = {}
        missing, extra = sorted(want_keys - set(maps)), sorted(set(maps) - want_keys)
        if missing or extra:
            diffs["keys"] = {"missing": [list(k) for k in missing], "extra": [list(k) for k in extra]}
        for (d, direction), image in maps.items():
            if (d, direction) not in want_keys:
                continue
            want = cli_maps[fname(d, direction)]
            if image.shape != want.shape or image.dtype != want.dtype:
                diffs[f"{d}/{direction}"] = f"shape {image.shape} {image.dtype} vs {want.shape} {want.dtype}"
                continue
            n = int(np.count_nonzero(image != want))
            if n:
                delta = np.abs(image.astype(np.int16) - want.astype(np.int16))
                diffs[f"{d}/{direction}"] = {
                    "pixels": n,
                    "max_abs": int(delta.max()),
                    "covered_xor": int(np.count_nonzero((image > 0) != (want > 0))),
                }
        result["checks"][label] = {"maps": len(maps), "identical": not diffs, "diffs": diffs}
        print(f"{label}: {len(maps)} maps, {'identical' if not diffs else f'{len(diffs)} differ'}", flush=True)
        return not diffs

    sweep_keys = {(d, direction) for d in range(-4, 5) for direction in ("forward", "reverse")}
    d0_keys = {(0, "forward"), (0, "reverse")}

    # 2. in memory
    t = time.perf_counter()
    mem = eng.sweep(args.zarr, range(-4, 5), direction="both", stride=64)
    result["sweep_memory_s"] = time.perf_counter() - t
    result["sweep_memory_report"] = {k: v for k, v in eng.last_report.items() if k != "map_tiles"}
    same = compare(mem, "sweep_memory", sweep_keys)
    if args.informational:
        complete = set(mem) == sweep_keys
        result["ok"] = complete
        result["mapped_libraries"] = inkrs._cuda.mapped_cuda_libraries()
        (args.out / "parity.json").write_text(json.dumps(result, indent=1))
        print(f"informational: {'identical' if same else 'differs'}; {'all' if complete else 'NOT all'} maps returned")
        return 0 if complete else 1
    ok &= same

    # 1. TIFF files
    ink = args.out / "py-ink"
    t = time.perf_counter()
    eng.sweep(args.zarr, range(-4, 5), direction="both", stride=64, out_dir=ink, name=args.name)
    result["sweep_files_s"] = time.perf_counter() - t
    files = {p.name: p for p in sorted(ink.glob("*.tif"))}
    bad = sorted(n for n in cli if n not in files or sha(files[n]) != sha(cli[n]))
    extra = sorted(set(files) - set(cli))
    result["checks"]["sweep_files"] = {"files": len(files), "identical": not bad and not extra, "differ": bad, "extra": extra}
    print(f"sweep_files: {len(files)} TIFFs, {'byte-identical' if not bad and not extra else f'{bad} {extra} differ'}")
    ok &= not bad and not extra

    # 3. the level-0 array in memory, with the group's pyramid given, as a villa backend would pass it
    group = zarr.open_group(str(args.zarr), mode="r")
    level0 = np.asarray(group["0"][:])
    # villa choose_pyramid_array: level "3", else the nearest numeric level (the lower on a tie)
    levels = sorted((int(k), k) for k in group.array_keys() if k.isdigit())
    scan_key = min(levels, key=lambda lk: (abs(lk[0] - 3), lk[0]))[1]
    result["occupancy_level"] = scan_key
    level3 = np.asarray(group[scan_key][:])
    s0 = depth // 2 - 8
    for label, occupancy in (("infer_array_occupancy_path", args.zarr), ("infer_array_occupancy_level", level3)):
        got = eng.infer(level0, layer_start=s0, layer_end=s0 + inkrs.DEPTH, direction="both", occupancy=occupancy)
        ok &= set(got) == {"forward", "reverse"}
        ok &= compare({(0, k): v for k, v in got.items()}, label, d0_keys)

    # 4. no low-resolution skip: villa's bare-array behaviour, reported only
    got = eng.infer(level0, layer_start=s0, layer_end=s0 + inkrs.DEPTH, direction="both", occupancy=None)
    compare({(0, k): v for k, v in got.items()}, "infer_array_no_scan_info", d0_keys)

    # 5. an engine released on a thread that never ran it gives its device memory back; this process's own use is
    # read from nvidia-smi, since other processes on the GPU move the device's free memory
    del mem, got, level0

    def used() -> int:
        if shutil.which("nvidia-smi"):
            rows = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            ).stdout.splitlines()
            mine = [int(r.split(",")[1]) for r in rows if r.split(",")[0].strip() == str(os.getpid())]
            if mine:
                return mine[0] * 2**20
        free, total = eng._memory_info()
        return total - free

    before = used()
    for _ in range(3):
        holder = [inkrs.Engine(args.weights, batch_size=64, algo=args.algo)]
        t = threading.Thread(target=holder.clear)
        t.start()
        t.join()
    lost = used() - before
    result["checks"]["drop_on_other_thread"] = {"used_before": before, "lost": lost, "identical": lost < 64 * 2**20}
    print(f"drop_on_other_thread: {lost / 2**20:.1f} MiB not returned after 3 engines")
    ok &= lost < 64 * 2**20

    result["mapped_libraries"] = inkrs._cuda.mapped_cuda_libraries()
    result["ok"] = bool(ok)
    (args.out / "parity.json").write_text(json.dumps(result, indent=1))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
