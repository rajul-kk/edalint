# edalint P9: split-leakage / test-set-tuning audit and UQ / OOD calibration baselines (pre-registration)

**STATUS: DRAFT, NOT FROZEN.** Binding from the commit that adds `protocol/FROZEN_p9.json` (section 7). Until then no P9 code opens an N14 test
design or an ICCAD'23 hidden case. Design and alternatives: `docs/leakage_uq_design.md`; scoop check: `docs/scoop_check_leakage_uq.md`.

## 1. Questions
* **Q1 (leakage).** Do the test rows of CircuitNet-N14 (IR drop, congestion, DRC) and the ICCAD'23 hidden cases have near-duplicate layouts in the
  training data; are family-related test designs (same RTL family, other configuration) as close to training designs as placement variants of one
  design are to each other; do published split files put one design on two sides; and do published recipes select on the test set?
* **Q2 (UQ / OOD).** Do split-conformal intervals and bootstrap ensembles on top of the zoo's ridge and GBM keep nominal coverage on unseen designs
  when calibrated in-distribution, and does calibrating on held-out *designs* fix it? Metrics: coverage vs nominal, width, interval calibration
  error, high-value-pixel coverage, OOD degradation.

## 2. Disclosure (known before this protocol)
* The registered N14 and ICCAD'23 results (`REPORT_n14.md`, `REPORT_iccad23.md`), including exact cross-split duplicates of `macro_region`
  (congestion, DRC), within-test exact IR maps (zero-riscy mp_1 vs mp_3), ICCAD'23: 0 E003 near pairs over 120 cases, C1 random-vs-design gap not
  material. Registered point errors of ridge / GBM on the test designs are known (e.g. N14 IR ridge ROC 0.851); no interval, coverage or NN
  similarity on any test row is known.
* Recipes read before this protocol (L004 evidence; the classes of section 4.4 were written after reading them, so L004 is descriptive, not a test):
  CircuitNet `train.py` / `utils/configs.py` (local copy, commit 41ade1d); CFIRSTNET `train.py`, `src/run.py`, `src/dataset.py` (GitHub, main);
  CLDRoute README. P2 PDF notes (`docs/published_numbers.md`) for WACA, LMM-IR, GL-Inception, AttUNet, GIF, CLDRoute.
* CLDRoute split CSVs were downloaded to `E:\edalint_data\p9_external\cldroute\` (sha256 recorded by the L003 run). While checking the download I
  saw the first three rows of `n14_test.csv` (three nvdla-small file names). No overlap was computed.
* Development (train only, section 8): timings and pipeline checks on RISCY / RISCY-FPU / Vortex-small / nvdla-small and ICCAD'23 fake/visible.

## 3. Data (no download)
* N14: the registered staging cache `E:\edalint_data\n14_registered_cache\<task>\authors\` (authors-space 256 x 256 members; train = registered fit
  sample, 979 rows per task; test = every official row with all member files). Every file read is sha256-hashed and folded into a per-design digest
  recorded in the result file. Rows = cached names that are listed in the packaged official CSVs and have every member file.
* ICCAD'23: `E:\edalint_data\iccad23\repo` (manifest sha256), train = 100 fake + 10 visible real, test = 10 hidden real.
* Split files (L003, metadata only): official `train_N14/test_N14/train_N28/test_N28.csv` (local copy of the CircuitNet repo files), CLDRoute
  `n14_{train,val,test}.csv`, `n28_{train,val,test}.csv`.

## 4. Leakage (L001-L004)
### 4.1 L001 nearest-neighbour near duplicates
Descriptor = E003 descriptor (32 x 32 area resize, per-channel z-score, mean Pearson over channels varying in either sample). For every test row, its
input-NN among all cached train rows (N14: 979; ICCAD'23: 110), the input and label similarity of that pair. Thresholds: primary 0.999 on input AND
label; sensitivity 0.99, 0.95. Flagged pairs (primary) are confirmed iff max |difference| <= 0.01 on every input channel and the label at full
resolution (authors space; pairs on different grids are not confirmed).
**LK1** per N14 task and ICCAD'23: *material* if confirmed pairs involve >= 1 % of the test rows; *present* if >= 1; else *not found* (a lower
bound: the reference is 979 of 6,790 N14 train rows).
### 4.2 L002 design-family proximity (N14 only; ICCAD'23 has no family metadata)
Reference: within-design NN similarity of each train design (leave-one-out). For each test design: S_same / S_other = median NN similarity of its
rows to same-family / other-family train rows (families: `edalint.checks.N14_FAMILIES`). **LK2**: proximity material iff S_same >= 25th percentile
of the pooled within-design reference AND S_same - S_other >= 0.02. Families are documented (data card 2), so a material proximity is a confirmed
design-level proximity; severity *material* if the design holds >= 10 % of the task's test rows, else *benign*.
### 4.3 L003 published split files
Design name per row (N14 `<design>_freq_...`, N28 `<index>-<design>-<variant>-...`). **LK3** per split set: an exact design name in both test and
train/val = confirmed fault (identity on names); *material* if the overlapping test rows are >= 1 % of the test file. Family overlap = info.
### 4.4 L004 test-set tuning in published recipes (descriptive)
Classes: T0 fixed budget, last checkpoint, no validation; T1 selection on a validation set disjoint from test; T2 test metrics computed or logged
during training (exposure), selection on validation; T3 selection or hyperparameters chosen on test (stated or visible in code); U unknown.
Only T3 is a fault. Every entry tagged code / README / PDF / abstract. Output `docs/p9_recipe_audit.md`.

## 5. UQ
### 5.1 Splits (seed 20261105, `edalint_p9.data.seeded_split`)
N14 (IR drop, congestion): F = 300 RISCY + 300 RISCY-FPU; C_id = 50 + 50; T_id = the remaining 50 + 50; C_xd = all cached Vortex-small (95) and
nvdla-small (84). Test designs: zero-riscy, Vortex-large, openc910-1. ICCAD'23: F = 70 fake, C_id = 15 fake, T_id = 15 fake, C_xd = 10 visible
real, test = 10 hidden. DRC is excluded from UQ (design D10).
### 5.2 Models and intervals
Bootstrap ensembles, M = 5, of `ridge` and `gbm` with the registered hyperparameters (`p4_n14.FROZEN_HPARAMS`, `p4_iccad23.FROZEN_HPARAMS`);
member k: bootstrap of the F maps with seed 20261105 + k, pixel-subsample seed k (GBM random_state k). Intervals: `ens_gauss` mu +/- z sigma;
`crc_abs` mu +/- lambda; `crc_norm` mu +/- lambda (sigma + eps), eps = 0.05 x mean calibration sigma; lambda by conformal risk control with loss =
miscovered pixel fraction, separately per alpha in {0.5, 0.4, 0.3, 0.2, 0.1, 0.05} and per regime (C_id, C_xd). Primary: `crc_norm`, alpha 0.1.
### 5.3 Evaluation
Every T_id map, every Vortex-large and openc910-1 test row, and a seeded subsample of 600 of the 3,441 zero-riscy rows (`seeded_split(names, [600], 20261105)`,
fixed from dev timings before the freeze: ~0.5 s per IR row read on E: plus 5 member predictions; 600 maps give a coverage standard error < 0.01).
ICCAD'23: all 10 hidden cases. Per map: coverage, mean
width, coverage on pixels >= the map's 90th label percentile, MAE of mu. Per design: means; interval calibration error = mean over the six nominal
levels of |coverage - nominal|; OOD degradation = coverage(design) - coverage(T_id), same regime.
### 5.4 Hypotheses (per target x task x model; nominal 0.9)
* **U1** (implementation validity) 0.9 - 0.03 <= coverage(T_id, C_id) <= 0.9 + max(0.03, 1/(n_C_id + 1)) for crc_abs and crc_norm (the upper
  band is the CRC upper bound; it only differs for ICCAD'23, n = 15 -> 0.9625; set in development, section 8). Expected TRUE; FALSE is reported
  as an anomaly.
* **U2** (naive calibration under-covers OOD) crc_norm calibrated on C_id has coverage < 0.85 on >= 2 of the 3 N14 test designs (ICCAD'23: on the
  hidden set).
* **U3** (cross-design calibration repairs it) mean over test designs of |coverage - 0.9| with C_xd <= the same with C_id - 0.02, AND design-macro
  coverage with C_xd >= 0.85.
* **U4** (raw ensembles under-cover) ens_gauss coverage at 0.9 on T_id < 0.85.
A task-level statement ("U2 holds for N14 IR drop") needs the hypothesis TRUE for both models. These are statements about these baselines and the
calibration machinery under design shift, not about deep models.

## 6. Verdicts
Computed mechanically by `edalint_p9.verdicts` from the run-once result files (`scripts/p9_run_verdicts.py` -> `results/p9/verdicts_p9.json`,
`results/p9/tables_p9.md`). Anything computed afterwards is labelled POST HOC.

## 7. Freeze and run-once
* Freeze record `protocol/FROZEN_p9.json`: dataset `p9`, `code_hash` = `edalint.protocol.code_hash()` at the freeze, `p9_code_hash` (guard recipe),
  protocol sha256 (CRLF -> LF), frozen commit. The guard (`edalint_p9.guard`) is the only path to test data and checks both hashes.
* Registered runs: `p9_run_leakage.py --target splits`, `--target n14 --task {ir_drop,congestion,drc}`, `--target iccad23`;
  `p9_run_uq.py --target n14 --task {ir_drop,congestion} --model {ridge,gbm}`, `--target iccad23 --model {ridge,gbm}`; then the verdicts.
  Each writes once (`write_once`); a crashed run is reported, re-run only with a dated deviation note, never after reading a partial test number.
* At most two CPU jobs at once (PLAN.md).

## 8. Development (train only) and deviations
All development used train data only (N14: RISCY, RISCY-FPU, Vortex-small, nvdla-small from the cache; ICCAD'23: fake and visible real).
Outputs in `results/dev/*_p9dev.json`; nothing in them is a test number.
* **Leakage, N14 congestion (dev)**: reference RISCY + RISCY-FPU (800 rows), queries Vortex-small + nvdla-small (179). 0 NN pairs >= 0.95.
  Within-design leave-one-out NN similarity: RISCY median 0.891 (q25 0.826), RISCY-FPU median 0.928 (q25 0.888); cross-design NN: Vortex-small
  median 0.466, nvdla-small 0.388. So placement variants of one design sit near 0.9 and unrelated designs near 0.4-0.5 on this descriptor; the
  L002 rule (q25 of the pooled reference, margin 0.02) was kept unchanged. Describing 979 rows took ~10 min.
* **UQ, ICCAD'23 ridge (dev)**: T_id = 15 fake maps; C_xd split into 5 / 5 visible real for a dev cross-check. Fit of 5 members 145 s.
  crc_norm / C_id: T_id coverage 0.957 at nominal 0.9 (n_cal = 15); visible-half coverage 0.756. ens_gauss coverage 0.106 on T_id (bootstrap
  ridge members barely disagree). With n = 5 calibration maps lambda@0.1 is infinite (alpha (n+1) - 1 < 0), as designed. Change made from this
  run: the U1 upper band now allows the CRC upper bound 1/(n+1) (0.9625 for n = 15), because 0.03 is below the finite-sample slack for 15
  maps. The lower band (the guarantee side) is unchanged.
* **UQ, N14 congestion ridge (dev)**: F 600 / C_id 100 / T_id 100 / C_xd 179; ensemble fit 98 s, whole run 4 min. crc_norm / C_id T_id coverage
  0.888 (U1 band holds); cross-design dev check (C_id calibration, evaluated on Vortex-small / nvdla-small): 0.948 / 0.841. lambda_norm ~100
  at alpha 0.1: the bootstrap spread is ~100x smaller than the error, so crc_norm mainly rescales a weak shape signal. No change.
* Two earlier ICCAD'23 dev attempts died with numpy MemoryError while the shared machine was out of commit memory (other processes); no code
  change, the third attempt completed. Registered runs are therefore run one at a time.
* Dev timings set the zero-riscy subsample (section 5.3).
