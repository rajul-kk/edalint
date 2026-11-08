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
never opened). WSL2 on an i7-1355U, Python 3.14, SciPy 1.18, best of 3 runs per measurement
(`results/bench_iccad23_train_partial.log`: 64 of the 100 fake netlists, 10k-199k nodes, before the run was stopped):

| | median speed-up | range |
|---|---|---|
| parse + solve, `direct` | **4.4x** (1.38 s -> 0.35 s per netlist) | 1.8x - 8.0x |
| parse only | 5.0x | 1.4x - 9.3x |
| solve only, `direct` | 3.4x | 2.2x - 7.0x |
| parse + solve, `cg_ic` | 0.6x (slower) | 0.3x - 2.1x |

Largest netlist (199k nodes): parse 2.16 -> 1.03 s, solve 1.22 -> 0.30 s. An earlier single-pass run over all 110 train
netlists (100 fake + 10 visible real) gave a median of 4.2x (1.6x - 8.2x; 4.2x on both subsets), with the native
`direct` voltages within 1.4e-13 V of SciPy's on every node; its raw output was not kept.

## Limits

- No `cg_amg` (edalint's pyamg-preconditioned CG); `cg_ic` is the native iterative option and is slower than the
  direct solve on these netlists (badly conditioned: via and wire resistances span orders of magnitude).
- No Windows build here (the extension is built in WSL / Linux).
- Rasterisation stays in Python (SciPy Clough-Tocher), so `exact_ir_drop` speed-ups are smaller than parse + solve.
