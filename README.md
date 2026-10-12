# edalint

Linter, metrics and baseline zoo for ML-for-EDA map benchmarks: ICCAD'23 Problem C (static IR drop) and CircuitNet-N14
(IR drop, congestion, DRC). Part of an open audit; see `PLAN.md`, `data_cards.md`, `docs/`.

```
pip install -e .            # numpy, scipy, pandas; extras: [sklearn] [amg] [test]
edalint lint iccad23 --split train
edalint lint n14-ir_drop --split train --max-per-design 20 --cache-dir E:/edalint_cache
edalint baselines iccad23 --fit fake --eval visible        # development: train data only
edalint freeze-status
```

* `edalint.datasets`: ICCAD'23 case loader (native size, optional resize, twin pairs recorded), CircuitNet-N14 streaming loader
  (tarballs, never extracted; `authors` and `raw` modes; official split CSVs; skip-and-log of incomplete archives and
  feature/label mismatches), N28 stub, synthetic data with fault injectors.
* `edalint.checks`: E001 integrity, E002 degenerate, E003 duplicates, E004 split leakage, E005 orientation, E006 metric sanity,
  E007 baseline ladder (informational). Each returns status, severity, findings, metrics and a skip reason.
* `edalint.metrics`: ICCAD official MAE/F1 (named variants) and CircuitNet NRMS/SSIM/NMAE/correlations/AUCs.
* `edalint.baselines`: zero, mean map, scalar, ridge, kNN, effective-distance regression, Green's-function superposition,
  gradient boosting, exact sparse solve of the SPICE PDN (runtime reference, ICCAD only).
* `edalint.spice`: netlist parser, DC solver, node-voltage to pixel-grid mapping.
* Test splits are guarded: nothing reads or scores a test split unless `protocol/FROZEN.json` exists and its `code_hash`
  equals `edalint code-hash`. Result files written with `--out` are run-once (never overwritten) and record the git HEAD,
  code hash and library versions.

Licence: MIT. The packaged `train_N14.csv` / `test_N14.csv` are copied unmodified from the CircuitNet repository (BSD-3-Clause).
