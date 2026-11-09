"""Time edalint's stream_tarball against edalint_n14io's readers on the first N maps of a CircuitNet-N14 tarball.

usage: python bench_n14.py <tarball> <out.json> [--members 3000] [--repeats 2]
Every reader stops after the same N maps (the handle raises), so each one decompresses the same prefix of the archive.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import edalint_n14io as n14
from edalint.datasets import circuitnet


class _Enough(Exception):
    pass


def run(impl, path, design, dirs, n):
    got = {"n": 0, "bytes": 0}

    def handle(f, name, a):
        got["n"] += 1
        got["bytes"] += a.nbytes
        if got["n"] >= n:
            raise _Enough
    t = time.perf_counter()
    try:
        impl(path, design, dirs, None, handle)
    except _Enough:
        pass
    return time.perf_counter() - t, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tarball")
    ap.add_argument("out")
    ap.add_argument("--members", type=int, default=3000)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    p = Path(a.tarball)
    design = p.name.split(".tar.gz")[0]            # also accepts a prefix copy such as RISCY.tar.gz.head
    task = "ir_drop" if p.parent.name == circuitnet.IRD else "congestion"
    feats = circuitnet.TASKS[task]["inputs"] + circuitnet.TASKS[task]["label"]
    dirs = {circuitnet.FEATURE_DIRS[f]: f for f in feats}
    impls = {"edalint": circuitnet.stream_tarball, "python": n14.stream_tarball_python}
    if n14.HAVE_NATIVE:
        impls["native"] = n14.stream_tarball_native
    if n14.HAVE_RUST:
        impls["rust"] = n14.stream_tarball_rust
    rows = []
    for name, impl in impls.items():
        best, got = None, None
        for _ in range(a.repeats):
            dt, got = run(impl, p, design, dirs, a.members)
            best = dt if best is None else min(best, dt)
        rows.append({"impl": name, "seconds": best, "maps": got["n"], "array_MB": got["bytes"] / 2**20})
        print(f"{name:<8} {best:7.2f} s for {got['n']} maps ({got['bytes'] / 2**20 / best:6.0f} MB/s of arrays)", flush=True)
    base = rows[0]["seconds"]
    for r in rows[1:]:
        print(f"{r['impl']}: {base / r['seconds']:.1f}x faster than edalint's reader")
    Path(a.out).write_text(json.dumps({"tarball": str(p), "platform": platform.platform(), "python": platform.python_version(),
                                       "numpy": np.__version__, "rows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
