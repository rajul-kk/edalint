# CircuitNet-N14: registered audit report

Protocol `protocol/protocol_n14.md` (frozen by `protocol/FROZEN_n14.json`, freeze commit `eba0ab4`; frozen commit `dee1851`; code hash `d3f6d2f67218...`, identical in all eight JSON result files, each of which is labelled `REGISTERED` and records an empty `git_dirty_tracked`; `tables.md` is plain text and carries neither field). Registered results: `results/n14/{faults,models_ir_drop,models_congestion,models_drc,gap_ir_drop,gap_congestion,gap_drc,verdicts}.json` and `tables.md`, committed in `786c59f`. Every number below comes from those files; the file is named where it is not `verdicts.json`.

All three N14 comparisons use a **weak reference** (other split, other preprocessing or an unstated hotspot rule; `verdicts.json` `H1.tasks.*.weak_reference` is true for all three).

## 1. Verdicts

| hypothesis | verdict | numbers |
|---|---|---|
| H1 IR drop (ROC area, higher better) | **TRUE** (weak reference) | best baseline `ridge` 0.8510 design-macro against mean_map 0.6301; published MAVIREC 0.8671. Gain 0.2209 against published gain 0.2370: ratio 0.932 (line 0.8). Ridge beats mean_map on 3 of 3 designs |
| H1 congestion (NRMS, lower better) | **FALSE** (weak reference) | best baseline `ridge` 0.0520 against mean_map 0.0521; published Circuit GNN 0.020. Gain 7.6e-5 against 0.0321: ratio 0.0024. Ridge beats mean_map on 2 of 3 designs |
| H1 DRC (SSIM, higher better) | **not evaluable (metric saturated)** | mean_map design-macro SSIM 0.9996 >= 0.99 (rule ii) |
| **H1 overall** (>= 2 of 3 tasks TRUE) | **FALSE** | 1 of 3 tasks TRUE; DRC cannot count, so both other tasks were needed |
| H2 data faults of material size | **TRUE** | DRC: all-zero authors-space label in 277 of 3,543 scanned test rows (7.8 % pooled; zero-riscy 275 of 3,441 = 8.0 %). Congestion and DRC: 3 cross-split exact duplicate groups each (`macro_region` only). IR drop: nothing material |
| H3 metric sanity | **TRUE** | DRC: 99.9 % of test rows have no positive pixel at the official threshold 0.1, 7.8 % a constant label (NRMS undefined), 9.1 % mean_map SSIM exactly 1. IR drop: 17.0 % have no positive pixel at 0.9885 |
| R1 raw IR label above the clip (>= 50 %) | **MATERIAL** | 92.8 % of the 3,544 test samples have a raw maximum above 50 |
| C1 random split materially optimistic (gap >= 0.10 on >= 2 of 3 tasks) | **FALSE** | C1b gaps: IR drop 0.0018, congestion -0.0004, DRC 0.0 |

Dev expectations in the protocol (section 2): H3 TRUE for DRC was borne out; "H1 undecided on IR drop" resolved to TRUE; the DRC saturation expectation was borne out (0.9996 against the dev 0.9998).

## 2. H1 per task (design-macro basis; `verdicts.json` `H1`, `models_*.json`)

**IR drop, official ROC area (higher better)** (`H1.tasks.ir_drop.design_values`, `design_macro`, `sample_pooled`)

| baseline | Vortex-large | openc910-1 | zero-riscy | design-macro | sample-pooled |
|---|---|---|---|---|---|
| zero | 0 | 0 | 0 | 0.0000 | 0.0000 |
| scalar | 0 | 0 | 0 | 0.0000 | 0.0000 |
| mean_map | 0.6172 | 0.6277 | 0.6454 | 0.6301 | 0.6446 |
| knn (k=1) | 0.5994 | 0.6384 | 0.7337 | 0.6571 | 0.7300 |
| green | 0.8302 | 0.7020 | 0.8590 | 0.7971 | 0.8527 |
| gbm | 0.8731 | 0.7738 | 0.8730 | 0.8400 | 0.8564 |
| ridge (alpha=100) | 0.8847 | 0.7967 | 0.8715 | 0.8510 | 0.8685 |

Sign count (designs on which a baseline beats mean_map): ridge 3, green 3, gbm 3, knn 2, zero 0, scalar 0.

**Congestion, official NRMS (lower better)** (`H1.tasks.congestion`)

| baseline | Vortex-large | openc910-1 | zero-riscy | design-macro | sample-pooled |
|---|---|---|---|---|---|
| zero | 0.1745 | 0.1126 | 0.1251 | 0.1374 | 0.1255 |
| mean_map | 0.0818 | 0.0460 | 0.0284 | 0.0521 | 0.0294 |
| scalar | 0.0816 | 0.0464 | 0.0282 | 0.0521 | 0.0292 |
| ridge (alpha=10) | 0.0816 | 0.0461 | 0.0284 | 0.0520 | 0.0294 |
| knn (k=5) | 0.0846 | 0.0539 | 0.0308 | 0.0565 | 0.0319 |
| gbm | 0.0820 | 0.0475 | 0.0287 | 0.0527 | 0.0297 |

Sign count: scalar 2, ridge 2, knn 0, gbm 0. Ridge has the largest macro gain; the verdict is the same whichever baseline is named best (ratio 0.0024).

**DRC, official SSIM (higher better)** (`H1.tasks.drc`). Not evaluable. Design-macro: zero, mean_map, scalar, ridge, gbm 0.9996; knn 0.9992. Per design the zero, mean_map, scalar and ridge values coincide (Vortex-large 0.99997, openc910-1 0.99976, zero-riscy 0.99921): on this metric ridge reproduces the constant prediction. Secondary, not decisive: the authors' ROC area at threshold 0.1 (design-macro, `secondary_roc_official_design_macro`) is 0 for zero, mean_map, scalar and gbm, 0.132 for ridge and 0.4996 for kNN; with 99.9 % of rows lacking a positive pixel (section 4) it is almost entirely undefined.

## 3. Sensitivity analysis and the E006 ROC-anchor finding

- **Anchored-ROC sensitivity** (`H1.tasks.ir_drop.sensitivity_anchored`). With the (0, 0)-anchored area the IR-drop H1 verdict is TRUE, ratio 0.9320 (official 0.9320), `agrees_with_official` = true. Anchored design-macro: zero 0.5, scalar 0.5, mean_map 0.6301, ridge 0.8510, knn 0.6572, green 0.8412, gbm 0.8400. The official area decides; there is no disagreement to flag.
- **C1 sensitivity** (`C1.ir_drop.sensitivity_anchored`). Anchored C1b gap 0.0036 (official 0.0018): not material, agrees.
- **E006 finding (registered): the authors' ROC area has no (0, 0) anchor** (`E006_roc_anchor`). On the test split `zero` and `scalar` score 0.0 with the official procedure and 0.5 with the anchored one; mean_map scores 0.6301 either way. A constant predictor therefore scores 0, not 0.5. The effect is visible in the table above (zero and scalar at 0.0000) and also moves `green` (official 0.7971, anchored 0.8412; `secondary_roc_anchored_design_macro`). It changes no verdict.

## 4. Data faults (H2), metric sanity (H3), raw versus clipped units (R1)

Rates are over every scanned test row (`faults.json`, uncapped): IR drop 3,544, congestion 3,544, DRC 3,543. Vortex-large has 46 of 47 DRC rows complete: `Vortex-large_freq_200_mp_1_fpu_60_fpa_1.0_p_4_fi_ar` lacks `DRC_all` (`models_drc.json` `skipped_by_loader`); it is the only dropped listed row.

**H2** (`verdicts.json` `H2`)

| task | all-zero label | const non-zero label | all-zero bbox input | non-finite | cross-split exact duplicates | material |
|---|---|---|---|---|---|---|
| IR drop | 0 | 0 | n/a | 0 | 0 | no |
| congestion | 0 | 0 | 0 | 0 | 3 groups | **yes (duplicates)** |
| DRC | 277 of 3,543 (7.8 %; zero-riscy 275 / 3,441, openc910-1 2 / 56, Vortex-large 0 / 46) | 0 | 0 | 0 | 3 groups | **yes (zero label, duplicates)** |

- The three duplicate groups are the same on congestion and DRC and are all the `macro_region` input channel (non-constant), of sizes 64, 64 and 48 arrays (`faults.json` `duplicates.*.cross_split_exact`). Each group's recorded member list is truncated to 20: for the first two groups the 20 listed members are all train RISCY-FPU rows (their test members, if any, are in the unlisted remainder); the third lists 16 train RISCY-FPU rows and 4 test zero-riscy rows (`zero-riscy_freq_200_mp_4_fpu_50_fpa_1.5_...`). No label array and no other input channel is duplicated. By the registered rule (any cross-split exact duplicate of a non-constant input array) it is material. The number of test rows involved was not recorded and was not re-derived after the freeze.
- The all-bbox-zero proxy for GIF's claim is 0 on every task and design: "not reproduced on map features", not a refutation (registered limitation).
- Cross-split family overlap (informational, not an H2 item): `riscy` (RISCY and RISCY-FPU in train, zero-riscy in test) and `vortex` (Vortex-small in train, Vortex-large in test).

**H3** (`verdicts.json` `H3`)

| task | no positive pixel at official threshold | const label (NRMS undefined) | mean_map SSIM exactly 1 | hit (>= 5 %) |
|---|---|---|---|---|
| IR drop (0.9885) | 17.0 % (zero-riscy 602 / 3,441; Vortex-large 0 / 47; openc910-1 0 / 56) | 0 % | 0 % | **yes** |
| congestion | n/a | 0 % | 0 % | no |
| DRC (0.1) | 99.9 % (zero-riscy 3,441 / 3,441; openc910-1 53 / 56; Vortex-large 46 / 46) | 7.8 % | 9.1 % | **yes** |

Rows without a positive pixel are silently dropped by the authors' ROC average: the official IR-drop ROC area is computed on the remaining 83 % of rows, and the DRC one on almost none.

**R1** (`verdicts.json` `R1`, `R1_drc_informational`). IR-drop raw label maximum above the clip 50 on 92.8 % of the 3,544 test samples (threshold 50 %): **MATERIAL**. Mean fraction of pixels clipped 1.4 %; 54.4 % of the raw pixels at or above the hotspot-equivalent raw value 40.78 are saturated at the clip. Unit status: undocumented (the data card says "inferred mV"). DRC clip 200 (informational only): 0.11 % of the 3,543 test samples exceed it; mean fraction of pixels clipped 1e-9.

## 5. Comparison C1: random versus design-level split (`verdicts.json` `C1`; `gap_*.json`)

| task | Skill_official (ridge over scalar, design-macro) | Skill_random (C1b, 5 reps) | gap C1b | C1a gap (train only) | material (>= 0.10) |
|---|---|---|---|---|---|
| IR drop | 0.8510 | 0.8528 | 0.0018 | -0.0054 | no |
| congestion | 0.0020 | 0.0015 | -0.0004 | -0.0008 | no |
| DRC | 0.0 | 0.0 | 0.0 | 0.0 | no |

The random split is **not** materially optimistic on this benchmark (0 of 3 tasks). Two cautions: for DRC the ridge and scalar SSIM coincide, so skill and gap are zero by construction and say nothing about split difficulty; and RISCY and RISCY-FPU are siblings, so the leave-one-design-out folds of C1a are themselves partly leaky (protocol section 8).

## 6. Exclusions and accounting (`faults.json` `accounting`; `models_*.json` `unlisted_in_csv`)

- **Archive members in neither CSV (excluded from every verdict):**

  | design | IR-drop archive | routability archive |
  |---|---|---|
  | RISCY | 208 | 294 |
  | RISCY-FPU | 7 | 7 |
  | Vortex-small | 1 | 1 |
  | nvdla-small | 4 | 12 |
  | zero-riscy | 15 | 15 |
  | openc910-1 | 40 | 40 |
  | Vortex-large | 14 | 27 |

  Sums: 289 (IR drop), 396 (routability). These are accounting facts, not faults.
- **nvdla-large** (in neither split file; excluded): 32 labelled samples in the IR-drop archive, 60 in the routability archive, none in either CSV (`manifests/n14_inventory.json`).
- **Listed rows lacking files:** 0 on every task and design except DRC Vortex-large (1 of 47, section 4).

## 7. Fit sample, runtime and provenance

- **Fit sample:** 979 rows per task (RISCY and RISCY-FPU capped at K = 400, Vortex-small 95 and nvdla-small 84 whole), seed 20261013; indices in `models_*.json` `fit_sample`. Test scoring used every listed row (3,544 / 3,544 / 3,543 `eval_ids`).
- **Hyperparameters:** `FROZEN_HPARAMS`, identical to protocol section 5.2.
- **Stage timings** (seconds, `seconds_total` / `seconds` in the result files): `faults` 8,454; `models` IR drop 7,671, congestion 7,699, DRC 6,743; `gap` IR drop 3,221, congestion 2,168, DRC 7,169. Wall clock with two concurrent jobs (job 1: `faults` then the three `gap` stages; job 2: the three `models` stages), from the job-log timestamps in `E:\edalint_data\n14_registered_logs`: job 1 21,035 s (5.84 h), job 2 22,129 s (6.15 h).
- **IR-drop fit times** (`models_ir_drop.json`): mean_map 169, scalar 46, ridge 135, kNN 72, Green 247, GBM 125; test scoring 3,791.

## 8. Limitations

- The published references are weak (other splits and normalisations, a graph input for the congestion reference, an unstated hotspot rule for the IR-drop reference); H1 ratios compare against a number, not a replication.
- The IR-drop H1 verdict uses the official unanchored ROC area; the anchored sensitivity agrees.
- The test split has 3 designs and zero-riscy supplies 97 % of the rows; no p-values are given, and the design-macro mean weights Vortex-large (47 rows) and openc910-1 (56 rows) equally with zero-riscy.
- DRC could not be adjudicated on SSIM (saturated), so the DRC part of H1 stays open; the metric and data faults (H2, H3) are the finding there.
- The duplicate `macro_region` finding is material by the registered rule but its extent in test rows was not recorded.
- Within-test exact duplicates (POST HOC observation, section 9) are recorded in `faults.json` but are not part of any registered criterion; they bear on how many effectively independent test samples there are, which no verdict here accounts for.
- One dataset family, baselines with a small fixed tuning grid, one run.

## 9. Post-run notes (written after the registered run)

- **No POST HOC analysis, no re-run, no crashed stage.** Every stage ran exactly once to completion from the freeze state; the only commands after the stages were `verdicts` and the commits.
- **Pre-freeze review nits (non-blocking), recorded for the record:**
  1. The protocol banner still reads DRAFT / NOT FROZEN and section 12 says "Freeze (to happen)"; the text was left unedited and the discrepancy is recorded in `protocol/FROZEN_n14.json` `protocol_status_note`.
  2. `results/n14/tables.md` does not print the per-row weak-reference flag or the sensitivity-agreement flag; both are in `verdicts.json` (all three pairs weak; the single sensitivity agrees).
  3. `stage_models` does not write the loader's `errors` dict, so a sample that failed to load would be dropped silently; here the one dropped listed row is in `skipped_by_loader` and every other listed test row was scored.
  4. The round-2 hyperparameter CV file (`results/dev/n14_hparam_cv_round2.json`) was produced from a dirty working tree.
  5. `write_once` would leave a partial file if `json.dump` failed mid-write, blocking a rerun (not triggered).
  6. The dry-run outputs in `E:\edalint_data\n14_dry_run` predate the saturation rule; the current verdict code was replayed on them (train-only data) to check the path.
- **POST HOC (observation, not a registered test): within-test exact duplicates.** `faults.json` `duplicates.<task>.within_test_exact` lists, per task, the groups of test arrays that are byte-identical (SHA-1 of bytes and shape of the stored native-resolution array, non-constant arrays only) and whose members are all in the test split; each group is one (feature, hash) pair with `n` >= 2 members, and the member list is truncated to 20. The counts are therefore **groups, not rows**: IR drop 1,632 groups, congestion 5,471, DRC 11,073. They cover every input and label array of the task, not one channel: IR drop `IR_drop` 807 groups (the label), `power_all`, `power_i`, `power_s`, `power_sca`, `power_t` 165 each; congestion `GR_horizontal_overflow` 1,316, `GR_vertical_overflow` 1,308 (labels), `macro_region` 173, `RUDY` 1,337, `RUDY_pin` 1,337; DRC `DRC_all` 1,169 (label), `eGR_horizontal_overflow` 454, `eGR_vertical_overflow` 1,305, `GR_horizontal_overflow` 1,316, `GR_vertical_overflow` 1,308, `cell_density` 1,337, `macro_region` 173, `RUDY_long`, `RUDY_pin_long`, `RUDY_short` 1,337 each. Almost all lie inside zero-riscy (IR drop 1,632 of 1,632 groups; congestion 5,337 of 5,471; DRC 10,751 of 11,073), the rest inside Vortex-large or openc910-1. Group sizes are 2 (IR drop) up to 64 (congestion, DRC). This bears on **no registered verdict**: H2 counts cross-split duplicates only (and within-design duplicates were never an H2 item), so the H2 verdict above is unchanged. It was not analysed further after the freeze (for example how many distinct test samples remain after collapsing identical arrays), and it is reported only as context for the effective size of the test split.
- **Number verification pending.** Every number in sections 1 to 7 was taken from the result files named above, but a separate verification pass over this report against those files has not yet been run.

## Appendix: tables (verbatim from `results/n14/tables.md`)
## H1 per task (design-macro primary metric; every reference is a WEAK reference: other split or preprocessing)

| task | metric | mean_map | best baseline | its value | published (ref) | gain ratio | verdict |
|---|---|---|---|---|---|---|---|
| ir_drop | roc_auc | 0.6301 | ridge | 0.8510 | 0.8671 (MAVIREC) | 0.93 | True |
| congestion | nrms | 0.0521 | ridge | 0.0520 | 0.02 (Circuit GNN) | 0.00 | False |
| drc | ssim | n/a | | | 0.7136 | | not evaluable |

H1 overall: False (1 of 3 tasks)

## Primary metric per baseline (design-macro / sample-pooled)

### ir_drop (roc_auc)
| baseline | design-macro | pooled |
|---|---|---|
| zero | 0.0000 | 0.0000 |
| mean_map | 0.6301 | 0.6446 |
| scalar | 0.0000 | 0.0000 |
| ridge | 0.8510 | 0.8685 |
| knn | 0.6571 | 0.7300 |
| green | 0.7971 | 0.8527 |
| gbm | 0.8400 | 0.8564 |

### congestion (nrms)
| baseline | design-macro | pooled |
|---|---|---|
| zero | 0.1374 | 0.1255 |
| mean_map | 0.0521 | 0.0294 |
| scalar | 0.0521 | 0.0292 |
| ridge | 0.0520 | 0.0294 |
| knn | 0.0565 | 0.0319 |
| gbm | 0.0527 | 0.0297 |

### drc (ssim)
| baseline | design-macro | pooled |
|---|---|---|
| zero | 0.9996 | 0.9992 |
| mean_map | 0.9996 | 0.9992 |
| scalar | 0.9996 | 0.9992 |
| ridge | 0.9996 | 0.9992 |
| knn | 0.9992 | 0.9984 |
| gbm | 0.9996 | 0.9992 |

H2 (material data fault on some task): True
H3 (metric undefined or restricted for >= 5 % of samples): True

R1 (IR raw label above the clip on >= 50 % of test samples): True

C1 (random split materially optimistic on >= 2 tasks): False
