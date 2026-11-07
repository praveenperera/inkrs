"""Mutation test for the gate scripts' selftests: each check must be one a selftest case needs

    python -I scripts/gate_mutations.py WORKDIR SCRIPT...

For each SCRIPT (scripts/p0_hecate.py, scripts/hecate_mapgate.py), finds every call to `check` with
Python's parser (so `check (`, a call split over lines, or one nested in an expression all count), writes
a copy to a fresh directory under WORKDIR with that one call's condition made `True or (...)`, and runs the copy's selftest. The selftest must
fail for every such copy, and pass for the unchanged script. A check no case depends on could be
deleted or broken without any test noticing, so it is reported and the exit status is 1. A new check
is covered without being listed here. Before mutating, the finder runs on a fixed sample of call forms
and stops if it misses or invents one
"""

import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SAMPLE = """def check(ok, msg):
    pass
check(a, "m")
check (b, "m")
check(
    c, "m")
x = [check(d, "m") for d in e]
obj.check(f, "m")
"""


def calls(src):
    """Source offsets of the first argument of every call to the bare name `check`"""
    starts = [0]
    for line in src.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    found = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "check":
            if not node.args:
                raise SystemExit(f"line {node.lineno}: check() with no condition")
            a = node.args[0]
            # col_offset counts UTF-8 bytes
            line = src[starts[a.lineno - 1]:starts[a.lineno]]
            found.append((starts[a.lineno - 1] + len(line.encode()[:a.col_offset].decode()),
                          starts[a.end_lineno - 1] + len(src[starts[a.end_lineno - 1]:].encode()[:a.end_col_offset].decode()),
                          node.lineno))
    return sorted(found)


def mutate(src, at):
    start, end, _ = at
    return src[:start] + "True or (" + src[start:end] + ")" + src[end:]


def selftest(path):
    r = subprocess.run([sys.executable, "-I", str(path), "selftest"], capture_output=True, text=True, timeout=300)
    return r.returncode


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    sample = calls(SAMPLE)
    want = ["a", "b", "c", "d"]
    got = [SAMPLE[s:e] for s, e, _ in sample]
    if got != want or any("True or (a)" not in mutate(SAMPLE, sample[0]) for _ in [0]):
        raise SystemExit(f"call finder: found {got} in the sample, want {want}")
    # a private directory per run: concurrent stages share TMPDIR, and fixed mutant names collided across them
    Path(sys.argv[1]).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run.", dir=sys.argv[1]))
    bad = 0
    for script in map(Path, sys.argv[2:]):
        src = script.read_text()
        if selftest(script) != 0:
            print(f"{script}: selftest fails unchanged")
            bad += 1
            continue
        found = calls(src)
        if not found:
            print(f"{script}: no check calls found")
            bad += 1
            continue
        for at in found:
            mutant = mutate(src, at)
            line = at[2]
            path = work / f"{script.stem}_L{line}_{at[0]}.py"
            path.write_text(mutant)
            caught = selftest(path) != 0
            text = src.splitlines()[line - 1].strip()
            print(f"{'caught' if caught else 'MISSED'} {script.name}:{line} {text[:90]}")
            bad += not caught
            path.unlink()
        print(f"{script}: {len(found)} checks mutated")
    shutil.rmtree(work)
    print("gate mutations", "ok" if bad == 0 else f"FAILED ({bad})")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
