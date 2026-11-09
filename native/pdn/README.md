# edalint_pdn: native DC solve of ICCAD'23 power-delivery netlists

A C++17 / Eigen implementation of `edalint.spice.parse_netlist` + `solve_dc` (static IR drop of an ICCAD'23 Problem C
SPICE netlist), with pybind11 bindings. It returns `edalint.spice.Netlist` / `SolveResult` objects, so either parser can
feed either solver, and `edalint_pdn.exact_ir_drop` rasterises with `edalint.spice.rasterize_ir_drop` (identical
interpolation and label convention).

It lives outside `src/edalint` on purpose: `edalint.protocol.code_hash()` covers `src/edalint/**/*.py`, and the
registered freezes and the code-hash lineage must not move for an optional accelerator.

## What it computes

Nodal analysis `G v = i` with
- zero-ohm resistors merged (union-find), parallel resistors summed;
- ground and ideal supplies as Dirichlet nodes (conflicting supplies on one merged node are a `ValueError`);
- nodes with no conductive path to a fixed node marked floating (`NaN`), as in edalint;
- the reduced SPD system `G_ff v_f = i_f - G_fx v_x` solved by sparse LDL^T with AMD ordering (`direct`), or by
  conjugate gradient with incomplete Cholesky (`cg_ic`) or Jacobi (`cg`) preconditioning.

The parser streams the file once (no regex, no pandas), keys nodes by `string_view` into the file buffer, and parses
numbers with `std::from_chars`; lines are split exactly as edalint does (first four whitespace fields; `.` is not a
comment character). Only node order differs from edalint (first appearance per line), so compare by node name.

## Build and use

Needs a C++17 compiler and Eigen 3 headers (`EIGEN3_INCLUDE_DIR`, default `/usr/include/eigen3`); Linux and macOS.

```
pip install -e /path/to/edalint            # the parent package
pip install --no-build-isolation -e native/pdn
python -m pytest native/pdn/tests
```

```python
import edalint_pdn as pdn
net = pdn.parse_netlist("netlist.sp")          # an edalint.spice.Netlist
res = pdn.solve_dc(net, "direct")              # an edalint.spice.SolveResult (voltage per node, info with timings)
ir_map, info = pdn.exact_ir_drop("netlist.sp", shape)   # same signature as edalint.spice.exact_ir_drop
```

## Correctness

`tests/test_pdn.py` checks the analytic circuits of edalint's own tests (series chain, decimals, ground written first,
zero-ohm merge with a floating island, conflicting supplies), random two-layer meshes with vias, zero-ohm segments and
islands where every (parser, solver) combination is compared node by node with `edalint.spice` (1e-9 V), the IR-drop
map, and, when the benchmark is on disk, a real ICCAD'23 netlist.

## Speed (ICCAD'23 train split)

`bench/bench_iccad23.py` times edalint.spice (pandas parse, SciPy `spsolve`) against the native path on the train split
only (fake-circuit-data and the visible real-circuit-data; the hidden real cases are the guarded test split and are
never opened). WSL2 on an i7-1355U, Python 3.14, SciPy 1.18, best of 3 runs per measurement, all 110 train netlists
(100 fake + 10 visible real, 10k-206k nodes; `results/bench_iccad23_train.json` and `.log`):

| | median speed-up | range |
|---|---|---|
| parse + solve, `direct` | **4.7x** (1.30 s -> 0.31 s per netlist) | 2.0x - 7.4x |
| parse only | 5.4x | |
| solve only, `direct` | 3.5x | |

Fake and visible real netlists give the same picture (4.7x and 4.9x). Largest netlist (206k nodes): parse 1.70 -> 0.33 s,
solve 1.05 -> 0.32 s. Native `direct` voltages agree with SciPy's within 1.4e-13 V on every node of every netlist.
`cg_ic` was slower than SciPy's direct solve (median 0.6x, from an earlier run over 64 netlists), so it is not timed here.

## Limits

- No `cg_amg` (edalint's pyamg-preconditioned CG); `cg_ic` is the native iterative option and is slower than the
  direct solve on these netlists (badly conditioned: via and wire resistances span orders of magnitude).
- No Windows build here (the extension is built in WSL / Linux).
- Rasterisation stays in Python (SciPy Clough-Tocher), so `exact_ir_drop` speed-ups are smaller than parse + solve.
