"""Parse + DC solve timings, edalint.spice (Python/SciPy) against edalint_pdn (C++/Eigen), on ICCAD'23 train netlists.

Reads only the train split (fake-circuit-data and the visible real-circuit-data); the hidden real cases are the guarded
test split and are never opened. Writes one JSON row per (netlist, implementation, method) and prints a summary.

usage: python bench_iccad23.py <iccad23 repo root> <out.json> [--limit N] [--repeats R] [--methods direct cg_ic]
With --repeats > 1 every measurement is the best of R runs, so both parsers read warm files.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import scipy

import edalint_pdn as pdn
from edalint import spice

SETS = ("fake-circuit-data", "real-circuit-data")  # train split only; never "hidden-real-circuit-data"


def netlists(root: Path, limit):
    out = []
    for s in SETS:
        files = sorted((root / "benchmarks" / s).rglob("*.sp"))
        out += [(s, f) for f in (files[:limit] if limit else files)]
    return out


def timed(fn, repeats):
    best, value = None, None
    for _ in range(repeats):
        t = time.perf_counter()
        value = fn()
        dt = time.perf_counter() - t
        best = dt if best is None else min(best, dt)
    return best, value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0, help="netlists per set (0 = all)")
    ap.add_argument("--repeats", type=int, default=1, help="best of R runs per measurement")
    ap.add_argument("--methods", nargs="+", default=["direct", "cg_ic"], choices=["direct", "cg_ic", "cg"])
    a = ap.parse_args()
    rows = []
    for group, path in netlists(Path(a.root), a.limit):
        t_py_parse, py_net = timed(lambda: spice.parse_netlist(str(path)), a.repeats)
        t_nat_parse, nat_net = timed(lambda: pdn.parse_netlist(str(path)), a.repeats)
        assert nat_net.counts() == py_net.counts(), path
        t_py_solve, py_res = timed(lambda: spice.solve_dc(py_net, "direct"), a.repeats)
        ref = dict(zip(py_net.names.tolist(), py_res.voltage))
        names = nat_net.names.tolist()
        idx = np.array([i for i, k in enumerate(names) if np.isfinite(ref[k])])
        ref_v = np.array([ref[names[i]] for i in idx])
        rows.append({"set": group, "netlist": str(path.relative_to(path.parents[1])), "impl": "python", "method": "direct",
                     "parse_s": t_py_parse, "solve_s": t_py_solve, "max_abs_diff_V": 0.0, **py_net.counts(),
                     "n_free": py_res.info["n_free"]})
        for m in a.methods:
            t_solve, res = timed(lambda: pdn.solve_dc(nat_net, m, tol=1e-10), a.repeats)
            diff = float(np.max(np.abs(res.voltage[idx] - ref_v)))
            rows.append({"set": group, "netlist": str(path.relative_to(path.parents[1])), "impl": "native", "method": m, "parse_s": t_nat_parse,
                         "solve_s": t_solve, "max_abs_diff_V": diff, **nat_net.counts(), "n_free": res.info["n_free"],
                         "iterations": res.info["iterations"]})
        last = rows[-(1 + len(a.methods)):]
        solves = " ".join(f"{r['method']} {r['solve_s']:6.2f}s" for r in last[1:])
        print(f"{group[:4]} {str(path.relative_to(path.parents[1])):<28} nodes {last[0]['nodes']:>8}  parse py "
              f"{last[0]['parse_s']:6.2f}s nat {last[1]['parse_s']:6.2f}s | solve py {last[0]['solve_s']:6.2f}s {solves} "
              f"| max diff {max(r['max_abs_diff_V'] for r in last):.1e} V", flush=True)
    env = {"python": platform.python_version(), "scipy": scipy.__version__, "numpy": np.__version__,
           "machine": platform.machine(), "processor": platform.processor(), "repeats": a.repeats}
    Path(a.out).write_text(json.dumps({"env": env, "rows": rows}, indent=1), encoding="utf-8")

    def med(impl, method, key):
        return statistics.median(r[key] for r in rows if r["impl"] == impl and r["method"] == method)

    py_total = [r["parse_s"] + r["solve_s"] for r in rows if r["impl"] == "python"]
    for m in a.methods:
        nat_total = [r["parse_s"] + r["solve_s"] for r in rows if r["impl"] == "native" and r["method"] == m]
        speed = [p / n for p, n in zip(py_total, nat_total)]
        print(f"native {m}: median parse {med('native', m, 'parse_s'):.2f}s vs {med('python', 'direct', 'parse_s'):.2f}s, "
              f"solve {med('native', m, 'solve_s'):.2f}s vs {med('python', 'direct', 'solve_s'):.2f}s; "
              f"parse+solve speed-up median {statistics.median(speed):.1f}x (min {min(speed):.1f}x, max {max(speed):.1f}x); "
              f"max diff {max(r['max_abs_diff_V'] for r in rows if r['impl'] == 'native' and r['method'] == m):.1e} V")


if __name__ == "__main__":
    main()
