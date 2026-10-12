# edalint ICCAD'23 Problem C: pre-registered audit protocol

**STATUS: FROZEN at the freeze commit (section 11).** Written 2026-10-12 under the lead's decisions (P4 brief). Nothing in this
document, and no code in `src/edalint/`, has been used to read or score any hidden case. CircuitNet-N14 has its own protocol,
written later; it is out of scope here.

## 1. Question

How much of the published deep-learning skill on ICCAD'23 Problem C (static IR drop) is reproduced by trivial, linear and physics
baselines that see only the contest's input maps, and does the exact solve of the provided netlist dominate the learned models in
both accuracy and runtime? And does the hidden set have a data structure that changes how its numbers should be read?

The research claim is about **task design** (the contest input contains the full linear system of the PDN), not about ML skill.

## 2. What was looked at before the freeze (disclosure)

* Hidden cases (tc 7, 8, 9, 10, 13, 14, 15, 16, 19, 20): in P1/P3 only file layout was read: names, CSV shapes, line counts, netlist
  element counts and the `V0` line of testcase7 (`data_cards.md`; `results/dev/iccad23_all_p3dev_metadata_only.*`;
  `results/dev/smoke_test_shapes_p3dev.json`, which checks that the loader runs on the hidden cases without computing any value
  statistic). No hidden map or label value was read by this project before the freeze.
* Public information that the authors know and that is not independent of the hidden labels: CFIRSTNET Table 2 prints the hidden
  labels' mean IR drop (0.44 to 4.65 mV, average 1.98 mV), maximum (average 5.87 mV) and hotspot fraction (average 0.168 %); the
  published per-case errors of ten models are known (section 8). Neither was used to choose any baseline setting.
* Train-only development (labelled "development (train only)"): P3 runs in `results/dev/` (baseline ladder fake to visible,
  leave-one-visible-design-out) and the two P4 developments of section 5 (`results/dev/iccad23_spice_raster_p4dev.json`,
  `results/dev/iccad23_hparam_cv_p4dev.json`). Only the 100 fake and 10 visible real cases were read.
* **Dev expectation, stated openly and set after seeing train-only numbers:** in P3 development (fit on 100 fake, scored on the 10
  visible real) mean MAE was zero 1.745, scalar 0.842, mean_map 0.778, knn 0.780, effdist 0.747, ridge 0.673, gbm 0.661, green 0.573 mV, F1
  (official strict rule) 0 to 0.02, and the exact solve 0.0031 mV. So the expectation is **H1 FALSE** (the best non-solver baseline
  reaches roughly a quarter of the mean-map MAE reduction, WACA-UNet reaches about 93 %) and **H2 TRUE**. These expectations are not
  criteria; section 7 decides.

## 3. Data, split, inference unit

* Source: `github.com/ASU-VDA-Lab/ML-for-IR-drop` commit `bda90cbe53f973f28d7c96951af665f678a84180`, local copy
  `E:\edalint_data\iccad23\repo` (sha256 in `manifests/MANIFEST.json`). Read-only.
* **Train** = 100 fake (BeGAN, `current_map00..99`) + 10 visible real (testcase 1, 2, 3, 4, 5, 6, 11, 12, 17, 18) = 110 cases.
  **Test** = the 10 hidden real cases: testcase 7, 8, 9, 10, 13, 14, 15, 16, 19, 20. There is no validation split: hyperparameters are
  chosen on train by cross-validation (section 5) and the final models are refit on all of train.
* **Inference unit.** The hidden cases are 5 designs, each a twin pair (same grid and number of current sources, different PDN;
  loader key `design_id`: 257/489/601/835/870 pixels per side). The effective sample size is **5**, not 10. We report (a) per case,
  (b) per design (the mean of a twin pair) and (c) the overall mean over the 10 cases, which is the published convention and is
  the headline number.
* **No p-values.** With n = 5 designs there is no inferential statistic in this protocol. We give descriptive statistics, the exact
  range over the 5 design means, and a sign count (number of the 5 designs on which a baseline's MAE is below the mean-map MAE),
  labelled descriptive.
* **Native resolution, no resize.** Every baseline predicts on the case's own grid and is scored on it, as the official contest
  scoring does. The published models resized internally (CFIRSTNET and WACA to 384 x 384, LMM-IR to 512 x 512); that does not
  affect the comparability of their reported numbers, which are at native size per the contest, but it is a difference in how they were
  obtained.

## 4. Metrics (primary = official)

All maps are in volts in the files; MAE is reported in millivolts.

* **MAE** (primary): per case, the mean over all pixels of |pred - golden|, in mV. Mean over the 10 cases for the headline.
* **F1** (primary): hotspot = golden pixel **strictly greater** than 0.9 x max(golden map); the same absolute threshold is applied to the
  prediction; F1 = 2TP / (2TP + FP + FN). A prediction with no hotspot pixel against a non-empty golden map scores 0 (as in the official
  xlsx). The golden map always has at least one hotspot pixel (its maximum), so F1 is always defined.
* **Also reported, labelled secondary:** the `>=` variant of the same rule; the top-10 % variant (golden hotspot = pixels at or above the
  90th percentile of golden, prediction cut at the same value; this is the reading AttUNet used); the number of cases with F1 = 0 for
  each of the three variants. Definitions: `src/edalint/metrics.py`.
* **Skill_MAE** = 1 - MAE / MAE(mean_map) with all MAEs over the same hidden cases (mean over the 10). Computed for our baselines. For
  published models the reference is **our** mean_map MAE on the hidden labels (that mean_map is ours, but the labels are the same).
  For the 3rd-place entry (valid on 8 cases only) the reference is our mean_map MAE on those 8 cases.
* Runtime is reported for the exact solver only (section 6.2); runtimes of other baselines are recorded for information.

## 5. Train-only development that fixed the choices (done before the freeze)

### 5.1 Rasterisation of the exact solver

Script `scripts/p4_dev_raster.py`, result `results/dev/iccad23_spice_raster_p4dev.json`, on the 10 visible real + 5 fake train cases
(fake ids = 5 evenly spaced: current_map00, 25, 50, 74, 99). Interpolation from the m1 node voltages to the 1 um pixel grid:

| method | mean MAE (mV) | max case MAE (mV) | cases with MAE > 0.01 mV | median end-to-end (s) | median rasterise (s) | median solve-only (s) |
|---|---|---|---|---|---|---|
| cubic | 0.0025 | 0.0072 | 0 | 7.58 | 6.32 | 0.169 |
| linear | 0.0088 | 0.0230 | 5 | 6.57 | 5.36 | 0.145 |
| nearest | 0.0206 | 0.0506 | 14 | 1.73 | 0.48 | 0.160 |

Rule (lead): freeze the fastest method whose train MAE stays <= 0.01 mV on every one of the 15 train cases. Only cubic qualifies.
**Frozen: cubic** (Clough-Tocher inside the convex hull of the m1 nodes, nearest node value outside), sparse direct LU solve. Note for
readers: the solve itself takes about 0.17 s; the interpolation to the dense pixel grid takes about 80 % of the end-to-end time. Both
are reported separately (section 6.2).

### 5.2 Baseline hyperparameters

Script `scripts/p4_dev_hparams.py`, result `results/dev/iccad23_hparam_cv_p4dev.json`. Leave-one-**twin-pair**-out cross-validation on
the 10 visible real cases (5 folds, one design per fold); the fit set of each fold is the 100 fake + the other 4 visible designs (8 real
cases); the held-out pair is scored at native resolution. **Selection criterion, set before the run: lowest mean MAE (mV) over the 10
held-out cases; ties go to the first listed option.** F1 is reported but not used (about 0 for all non-solver baselines on the visible
cases). Grids (fixed): ridge penalty {1e-6, 1e-4, 1e-2, 1} (in units of training rows); Green ring-bin edge sets coarse / medium /
fine (`edalint.baselines.zoo.GREEN_EDGES`); kNN k {1, 3, 5}; GBM {100 iterations, max_depth 3; 300 iterations, max_depth 6}.

| family | option | mean MAE (mV), 10 held-out cases | mean F1 strict | chosen |
|---|---|---|---|---|
| ridge | alpha=1e-06 | 0.5878 | 0.031 |  |
| ridge | alpha=0.0001 | 0.5878 | 0.031 |  |
| ridge | alpha=0.01 | 0.5875 | 0.032 | **yes** |
| ridge | alpha=1 | 0.6513 | 0.016 |  |
| green | edges=coarse | 0.5583 | 0.000 |  |
| green | edges=medium | 0.5569 | 0.000 | **yes** |
| green | edges=fine | 0.5570 | 0.000 |  |
| knn | k=1 | 0.7711 | 0.001 |  |
| knn | k=3 | 0.6796 | 0.003 | **yes** |
| knn | k=5 | 0.6828 | 0.000 |  |
| gbm | iter100_depth3 | 0.6439 | 0.000 |  |
| gbm | iter300_depth6 | 0.6323 | 0.000 | **yes** |

**Frozen** (`FROZEN_HPARAMS` in `src/edalint/p4_iccad23.py`): ridge alpha = 0.01; Green edges = medium (16 ring bins to 192 px, the library default); kNN k = 3; GBM 300 iterations, max_depth 6, no leaf limit, learning rate 0.1.
Unvaried settings stay at the defaults documented in the code (box means 3 and 9 for ridge and GBM; kNN on 16 x 16 downsampled
inputs; effdist window 9, degree 1; effdist_deg2 degree 2; fixed random seeds). The final baselines are fitted on **all 110 train
cases** (100 fake + 10 visible real), as the published models were trained on fake plus real cases.

## 6. Registered baselines and runs

### 6.1 Baselines (all registered; none may be added or dropped after the freeze)

`zero`, `mean_map`, `scalar`, `ridge`, `knn`, `effdist`, `effdist_deg2`, `green`, `gbm`, with the frozen hyperparameters of section 5.2,
fitted on all train and scored once on the 10 hidden cases at native resolution. They see only the three contest input maps (`current`,
`eff_dist`, `pdn_density`) and the training labels; none sees a netlist or a hidden label. Definitions: `src/edalint/baselines/zoo.py`.

### 6.2 Exact solve

Parse the case's SPICE netlist, assemble the nodal system (zero-ohm resistors merged, ideal sources as Dirichlet nodes), solve with
sparse direct LU (`scipy.sparse.linalg.spsolve`), interpolate VDD - V on the m1 nodes to the pixel grid with the frozen
rasterisation (cubic). No fitting, no label used. Timings, per case, on one process:

* **end-to-end** = `time.perf_counter()` around `exact_ir_drop()`: netlist parse + assembly + solve + rasterisation to the native grid.
  Reading the CSV maps is not included (the solver does not use them). One untimed warm-up solve on the train case `real/testcase11`
  precedes the timed cases (lazy library imports).
* **solve-only** = the sparse solve call alone; the assembly and parse times are recorded separately.
* Machine: recorded in every result file (provenance `cpu_model`, `logical_cores`, Python / numpy / scipy / sklearn versions). At the time
  of writing: Intel Core i7-1355U (10 cores, 12 threads), Windows 11, Python 3.10.0, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.6.1. The solve
  is single-threaded for the scipy sparse LU. Other jobs (background downloads) may run on the same machine; this is disclosed, not controlled.

### 6.3 Order and runs

Four stages, each one result file in `results/iccad23/` (section 9): `models.json`, `solver.json`, `h3.json`, then `verdicts.json` and
`tables.md` computed mechanically from the first three.

## 7. Hypotheses and criteria (decided now; the outcome does not change them)

**H1, "trivial/linear baselines reproduce most published skill".** Let S_b be the largest Skill_MAE among the nine registered non-solver
baselines (mean_map has Skill 0). Let S_pub be the Skill_MAE of the best published model by MAE, WACA-UNet (0.0524 mV), computed against
our mean_map MAE. **H1 is TRUE if S_b >= 0.8 x S_pub, otherwise FALSE.** Our dev expectation is FALSE (section 2; it was set after
seeing train-only numbers).

**H2, "the provided-netlist exact solve dominates".** **TRUE** if (i) the exact solver's mean MAE over the 10 hidden cases is below the
best published MAE (0.0524 mV, WACA-UNet) **and** (ii) its **median end-to-end runtime per case** on our machine is <= **14.77 s**, the
mean runtime of the official winner (xlsx mean 14.772 s). Solve-only runtime is reported as well but does not enter the criterion.
Caveats that must appear in the report: (a) the labels were produced by an equivalent solver of the same netlists, so accuracy is expected
by construction, and this is a statement about the task, not about ML skill; (b) runtimes measured on different hardware (the contest
organisers' machine, GPUs, other CPUs) are order-of-magnitude comparisons only; (c) the published ML runtimes are inference only, and do
not include whatever feature extraction the model needs, except where the paper says so.

**H3, "data structure that affects interpretation".** Reported, not tested:

* the twin structure of the hidden set (5 designs of 2 cases, effective n = 5; which channels the twins share);
* the cross-split duplicate and leakage checks E003 and E004 between train and hidden (`edalint` checks, run on split `all` after the
  freeze), plus an explicit exact-identity check: for every hidden case, every input channel (`current`, `eff_dist`, `pdn_density`) and the
  label are hashed (SHA-1 of the bytes and shape) and compared with every train case and with the other hidden cases;
* the hotspot base rate and the F1 fragility (E006) on the hidden labels: hotspot pixel counts per case, F1 of the label scaled by 0.95 / 1.05 and
  of a Gaussian-smoothed label at thresholds 0.85 / 0.9 / 0.95 x max, F1 of the all-zero prediction.

**Severity** (classes from the fieldlint protocol): **MATERIAL** if any of: (a) a **non-constant** input channel of a hidden case is byte-identical (same
shape) to the same channel of a train case; (b) a hidden label is byte-identical to a train label; (c) E003 reports an exact or near-duplicate pair
(input and label correlation >= 0.999 on the 32 x 32 descriptors) across train and test. **BENIGN** otherwise. Twin pairs sharing `current` and
`eff_dist` maps within the hidden set are expected by construction and are benign (within one split). The verdict word in `verdicts.json` is computed
by `compute_verdicts()` from these three conditions only.

**Adjudication of any other finding** on the hidden data (from E001, E002, E005, E006): carried over from fieldlint protocol v4 section 7. A finding
counts as a TRUE data fault only if (a) the contest documentation or the maintainers' repository states the property, or (b) a physics or identity
check stated in this protocol confirms it (identity check above; non-finite or shape-inconsistent array; negative IR drop, which contradicts the
documented non-negative drop for a positive current load), or (c) the maintainers state it. Everything else is reported as an unadjudicated
observation, not a fault. TRUE faults are classed benign (a convention that does not change what a model learns) or material (changes what it learns,
or contaminates the split). A benign finding is never presented as a headline discovery.

## 8. Comparison table (copied from `docs/published_numbers.md` at the freeze)

All on the 10 hidden real cases. MAE in mV, F1 as printed. Per-case values are in `PUBLISHED` in `src/edalint/p4_iccad23.py`.

| model | mean MAE | mean F1 | runtime | caveats |
|---|---|---|---|---|
| contest 1st, cadc0009 (xlsx) | 0.1347 | 0.4549 | 14.772 s | official; all 10 valid; this is the H2 runtime reference |
| contest 2nd, cadc1023 (xlsx) | 0.1498 | 0.4550 | 3.042 s | official |
| contest 3rd, cadc1020 (xlsx) | 0.1069 (8 valid cases) | 0.5186 (8 cases; 0.4143 over 10 with zeros) | none recorded | **failed 2 cases (19, 20)**: not comparable to 10-case means |
| CFIRSTNET (2502.12168, ICCAD 2024) | 0.0533 | 0.723 | 0.409 s | 384 x 384 resize; Xeon E3-1230 v6 + Quadro P4000, GPU use unstated; uses a netlist-derived physics feature |
| **WACA-UNet (2507.19197, preprint)** | **0.0524** | **0.778** | 1.390 s (CPU, model not stated) | 384 x 384 Lanczos resize; checkpoint selected by validation F1; best published MAE |
| LMM-IR (2511.12581, preprint) | 0.1356 | 0.583 | 3.05 s | 512 x 512; H100 |
| GL-Inception (2406.06541, preprint) | 0.101 | 0.62 | 3.37 s | includes feature extraction; Intel Core, 2.42 GHz |
| AttUNet (2408.03292, preprint) | 0.110 printed (per-case mean 0.1172) | 0.520 | 5.37 s | **F1 rule differs** (top 10 % of pixels, not the contest rule); hardware not stated |

Re-implementations of IREDGe by third parties (0.233 to 0.628 mV, F1 0.08 to 0.24) are mentioned in the report for context only and have no
registered role. No published paper reports a mean-map, ridge, kNN, GBM, Green or exact-solve baseline on this task
(`docs/published_numbers.md` section 3), so no comparison for those exists beyond our own runs.

## 9. Run-once, crash and post-hoc rules

* `PYTHONPATH=src python -m edalint.p4_iccad23 {models,solver,h3,verdicts}` (the package is not pip-installed) run in this order from the repository root. A stage requires `protocol/FROZEN.json`
  with a matching code hash and with `src`, `protocol`, `scripts` and `tests` matching HEAD, writes its file with exclusive creation (`results/iccad23/<stage>.json`) and **refuses to
  overwrite**. The dataset layer also refuses to read a hidden case without the freeze.
* A result is never re-run because of its content, a bug found afterwards, or a changed mind. If a stage crashes **before** its file is written, it may be
  re-launched; the crash and the re-launch are logged in section 11. If an error is found after a result exists, the result stands, the error is reported, and
  a corrected run is a separate, labelled post-hoc analysis.
* Anything not specified here is **POST HOC**: it is labelled in every table and sentence, written to `results/posthoc/`, must not change any
  section 7 number or verdict, and is reported separately.
* No criterion, threshold, baseline or hyperparameter changes after the freeze commit. At most 2 CPU jobs at a time.

## 10. Environment

Recorded automatically in each result file (provenance): git HEAD and uncommitted tracked changes, edalint code hash, the frozen-protocol record, Python /
numpy / scipy / pandas / scikit-learn / pyamg versions, CPU model, logical core count, platform, command line, start time. See also section 6.2.

## 11. Freeze and deviation log

* Freeze: the commit that adds `protocol/FROZEN.json` (hash in the git log; message begins "ICCAD'23: FREEZE protocol"). `FROZEN.json` holds the
  `code_hash` (`python -m edalint.cli code-hash`), the sha256 of this file, the date and time, and the commit of the code and protocol it freezes. `python -m edalint.cli freeze-status`
  must report a matching hash.
* Deviations after the freeze: none yet. (Entries below.)
