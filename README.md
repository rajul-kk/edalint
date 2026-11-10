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
* Test splits are guarded: nothing reads or scores a test split unless that dataset's freeze file (`protocol/FROZEN_<key>.json`; the ICCAD'23 freeze is `protocol/FROZEN.json` with `"dataset": "iccad23"`) exists and its `code_hash`
  equals `edalint code-hash`. Result files written with `--out` are run-once (never overwritten) and record the git HEAD,
  code hash and library versions.

## Lint your own dataset in CI

Point `edalint lint` at a folder of `.npz` files, one per sample: the label map (key `label`) and any input maps. Sub-folders are
splits; an optional `samples.csv` (`id,design[,twin_of]`) gives design ids, which the leakage check (E004) needs. Details in
`src/edalint/datasets/folder.py`.

```
mydata/train/case001.npz   mydata/val/case101.npz   mydata/test/case201.npz   mydata/samples.csv
edalint lint mydata                                   # text report; exit code below
edalint lint mydata --ignore E006 --fail-on error --sarif edalint.sarif --junit edalint.xml
edalint lint mydata --write-baseline edalint-baseline.json   # accept today's findings ...
edalint lint mydata --baseline edalint-baseline.json         # ... and fail only on new ones
edalint checks                                         # what each check looks for
```

| exit code | meaning |
|---|---|
| 0 | no finding at or above `--fail-on` (default `warning`) |
| 1 | worst finding is a warning (or an info with `--fail-on info`) |
| 2 | worst finding is an error |
| 3 | refused: reading a guarded test split needs a valid freeze record |
| 4 | a run-once `--out` result file already exists |
| 5 | bad configuration, unknown check or split, unreadable dataset |

| output | flag | use |
|---|---|---|
| text, JSON, Markdown | `--format`, `--json FILE`, `--markdown FILE` | logs, scripts, job summaries |
| SARIF 2.1.0 | `--sarif FILE` | GitHub code scanning (results point at the affected `.npz` files) |
| JUnit XML | `--junit FILE` | one test case per check, for CI test reports |
| GitHub annotations | `--format github` | warnings and errors on the run and the pull request |

Settings live in `edalint.toml` or `[tool.edalint]` in `pyproject.toml` (searched in the current folder, then in the dataset
folder); flags override them. Unknown keys are an error, so a typo does not silently switch a check off.

```toml
dataset = "data/irdrop"
ignore = ["E006"]
fail-on = "error"
baseline = "edalint-baseline.json"
[options]                 # check thresholds (edalint.checks.Options)
dup_corr = 0.999
[dataset-format]          # folder datasets
label_key = "ir_drop"
hotspot = { kind = "frac_of_max", value = 0.9 }
```

GitHub Action (`action.yml`) and pre-commit hook (`.pre-commit-hooks.yaml`):

```yaml
- uses: rajul-kk/edalint@v0.2.0
  with:
    dataset: data/irdrop
    fail-on: error
    upload-sarif: 'true'        # needs permissions: security-events: write
```

```yaml
- repo: https://github.com/rajul-kk/edalint
  rev: v0.2.0
  hooks: [{id: edalint}]       # lints the dataset named in edalint.toml when .npz / .csv / .toml files change
```

For your own folder no split is guarded. The frozen-protocol guard is for the registered benchmark studies in this repository;
set `test_splits` under `[dataset-format]` to use it on your data. The wheel holds the `edalint` package only; the study code
(`edalint_p7`, `_p8`, `_p9`) stays in the repository.

Licence: MIT. The packaged `train_N14.csv` / `test_N14.csv` are copied unmodified from the CircuitNet repository (BSD-3-Clause).

## What edalint does and does not detect (P8 fault-injection validation)

Measured by injecting known faults into copies of clean real data (20 draws per fault and severity, six bases: ICCAD'23, CircuitNet-N14 IR drop, IC-ThermBench levels 2-5).
Protocol `protocol/protocol_faults.md` (amendments A1-A6), results `results/faults/` (`summary_p8.md`, reruns `summary_p8r.md`). "Clean" means no known fault, not verified fault-free.

* **Reliably found, even at the weakest injected strength:** NaN/Inf pixels (E001, 1 pixel), an all-zero label or a negative label pixel (E002, about 0.2% of samples),
  byte-identical copies across splits (E003, 1 copy), a design present on both sides of a split (E004, 1 sample), a twin pair split across splits (E004, ICCAD'23).
* **Not found:** a single all-zero input channel (E002 reports it only as `info`, by design). Metadata or label faults that E006/E007 would have to catch are out of scope.
* **Found only when large:** a transposed or flipped label (E005) is found when essentially every sample is affected (100%: found in 100% of draws, except a transposed
  label on IC-ThermBench level 5: 80%). At 75% of samples affected the detection rate is 5-70% depending on the base, because E005 needs the alternative orientation to win in
  at least 75% of the sampled maps. A partially transposed dataset is likely to be missed. With every sample transposed, label noise (SNR 1-20) did not hide it, except on IC-ThermBench level 5, where it was already only 80% detected without noise.
* **Near-duplicates (E003) needed a correction, now merged.** The registered E003 divided the input correlation by the total channel count, so any input with
  constant channels (N14, IC-ThermBench levels 4-5, ICCAD'23 partly) could never reach the 0.999 threshold: near-duplicate detection was 0/20 at every strength on N14 and IC-ThermBench 4-5.
  The corrected variant (`edalint_p8.e003_fixed`, amendment A3) detects them once the pair's descriptor correlation is 0.999-0.9995 or higher (the thresholds' neighbourhood; 0.90+ detection from the third or fourth strength level; ICCAD'23 and N14 meet the validation criterion).
  Amendment A7 merged it into `edalint.checks.e003_duplicates` (a test asserts the two agree), which moved `code_hash()` from `d3f6d2f6...` to `a9900e6f...`. The freeze records `FROZEN_n14.json`
  and `FROZEN_p7.json` are not rewritten: they certify the registered N14/P7/P8 results, which reproduce at commit `c252323` (`protocol/CODE_HASH_LINEAGE.json`). A new test read needs a new freeze.
* **False positives:** none (0/20 draws) for E001, E002, E004, E005 on any base once the known findings of protocol section 7 are excluded. E003 flags 50-90% of clean IC-ThermBench draws
  (with the corrected variant on levels 4-5, and with the registered check on levels 2-3): the benchmark contains byte-identical inputs and labels on level 2 and repeated power maps on all levels
  (see `notes/` in the thermal repository); whether all flagged pairs are true duplicates on levels 3-5 is not established.
* **Caveats:** the registered ICCAD'23 `design_overlap` row is invalid (single-sample designs, amendment A4(b)); N14 used a reduced configuration (A5: 100 samples per draw); E004 on IC-ThermBench
  uses cases as designs, with the case assignment of levels 2-4 taken from the README; a detection rate describes injected faults of one kind, not every fault of that name.
