# ICCAD'23 Problem C: registered audit report

Protocol `protocol/protocol_iccad23.md` (frozen at commit 7707b20; code hash `750d252f863b...`, identical in all four result files). Registered results: `results/iccad23/{models,solver,h3,verdicts}.json` and `tables.md`. Every number is from the single registered run: nothing was re-run, no stage crashed, there are no deviations. Machine: Intel Core i7-1355U (10 cores, 12 threads), Windows 11, Python 3.10.0, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.6.1. Six unrelated background download processes ran on the same machine during all stages. Native resolution throughout, no resize. The hidden set is 5 designs (twin pairs), so the effective n is 5; no p-values are given.

## 1. Verdicts

| hypothesis | verdict | numbers |
|---|---|---|
| H1 trivial/linear baselines reproduce most published skill | **FALSE** | best non-solver baseline is `green`, Skill_MAE 0.480. WACA Skill_MAE is 0.955 (0.0524 mV against our mean_map 1.1605 mV). Threshold 0.8 × 0.955 = 0.764; green reaches 0.503 of WACA's skill. |
| H2 exact solve dominates | **TRUE** | solver mean MAE 0.0030 mV < 0.0524 mV; median end-to-end 6.37 s ≤ 14.77 s |
| H3 data structure | **BENIGN** (reported, not tested) | 5 designs of 2 cases; 0 hidden channels or labels duplicated in train; 0 E003 cross-split findings; 0 cross-split twin links |

Both dev expectations stated in the protocol (H1 FALSE, H2 TRUE) were borne out. They were set after seeing train-only numbers, so they carry no weight beyond consistency.

## 2. Exact solve (H2)

- **Configuration.** Sparse direct LU with cubic rasterisation, chosen on 15 train cases.
- **Accuracy.** Mean MAE 0.0030 mV, worst case 0.0065 mV (testcase 16). Mean strict F1 0.974, minimum 0.948 (testcase 19), no case with F1 = 0. The largest single-pixel difference to a label is 0.46 mV. F1 is below 1 only because a few hotspot-boundary pixels fall on the other side of the 0.9 × max threshold. Relative residual is 5.6e-15 to 7.8e-15 on every case.
- **End-to-end runtime** (netlist parse + assembly + solve + rasterisation, excluding CSV reads): median 6.37 s, mean 7.50 s, range 1.08 s (tc14) to 13.68 s (tc20). About 80 to 83% of it is the dense-grid interpolation (0.9 to 11.4 s). Netlist parsing takes 0.14 to 1.9 s.
- **Solve-only runtime** (the sparse LU call): median 0.197 s, mean 0.247 s, max 0.717 s.
- **Caveats.**
  - The labels come from an equivalent solver, so accuracy is expected by construction. The point is task design: the input contains the full linear system. It is not a statement about ML skill.
  - Cross-hardware runtimes are order-of-magnitude comparisons only.
  - H2 as registered compares against the winner's 14.77 s, which the solver meets by about 2.3× on the median. In raw end-to-end time it is slower than contest 2nd place (3.04 s) and all five published learned models (0.41 to 5.37 s). Only its solve-only time (0.2 s) is below all of them.
  - Nearest-neighbour rasterisation (about 1.7 s median end-to-end in development) failed the 0.01 mV train rule.

## 3. Baselines against published models (H1)

- **Training data.** Baselines were fitted on 110 train cases. The published models used 2,000 to 6,100 additional synthetic circuits and, in several cases, resized inputs.
- **Baseline MAE and Skill_MAE.** green 0.603 (0.480), knn 0.797 (0.313), ridge 0.877 (0.244), gbm 0.940 (0.190), effdist 1.024 (0.118), effdist_deg2 1.026 (0.116), mean_map 1.160 (0), scalar 1.193 (−0.028), zero 1.979 (−0.705).
- **Published Skill_MAE against our mean_map:**

  | model | Skill_MAE |
  |---|---|
  | contest 2nd | 0.871 |
  | LMM-IR | 0.883 |
  | contest 1st | 0.884 |
  | AttUNet | 0.905 |
  | GL-Inception | 0.913 |
  | contest 3rd (8 valid cases, referenced to our mean_map on those 8) | 0.917 |
  | CFIRSTNET | 0.954 |
  | WACA | 0.955 |

  The best baseline reaches about 53 to 55% of the weakest models' skill and 50% of WACA's.
- **Design level** (descriptive, n = 5):
  - green beats mean_map on 5 of 5 designs and has the lowest baseline MAE on 3 of 5 (kNN is lowest on the 257 and 835 designs). Its design means range 0.177 to 1.072 mV.
  - ridge and gbm also beat mean_map on 5 of 5, and knn on 4 of 5.
- **F1.** Mean strict F1 is 0.000 for six of the nine baselines and at most 0.040 (ridge). Cases with F1 = 0: 10 for zero, mean_map, scalar, knn, effdist_deg2 and green; 8 for effdist and gbm; 4 for ridge. The `>=` variant is identical. Under the secondary top-10% rule the best baselines score green 0.266, ridge 0.252, knn 0.137 and gbm 0.123, against 0.996 for the exact solve. AttUNet's 0.520 uses the top-10% rule and is not comparable to the strict-rule numbers.

## 4. Data structure (H3)

- **Twin pairs.** The pairs are 257 (tc 13/14), 489 (15/16), 601 (7/8), 835 (9/10) and 870 (19/20). In all five the `current` and `eff_dist` maps are byte-identical between the twins; only the PDN differs.
- **Leakage and duplicates.**
  - E003 over 120 cases: 0 exact input groups, 0 exact label groups, 0 near-duplicate pairs.
  - E004: 105 designs in train, 5 in test, 0 shared, 0 cross-split twin links.
  - The explicit identity check found no non-constant input channel and no label of a hidden case equal to any train one.
  - Severity: benign.
- **Other checks.** E001 and E005 are clean. E002 warns that `pdn_density` is constant in 5 cases (the second member of each pair), consistent with the contest slides' "4 versus 1 PDN templates". Reported as an unadjudicated observation.
- **Hotspot base rate and F1 fragility (E006).** Hotspot pixels per case range from 53 to 1,933 (median 336); base rate 0.010% to 0.535% (median 0.083%). F1 of pseudo-predictions at thresholds 0.85 / 0.90 / 0.95 × max:
  - label × 0.95: 0.58 / 0.48 / 0.00
  - label × 1.05: 0.66 / 0.59 / 0.49
  - smoothed label (sigma 1.5 px): 0.74 / 0.49 / 0.19

  F1 therefore collapses when a prediction under-shoots the peak by a few per cent. That is why every baseline scores about 0 on F1 despite real MAE skill, and why contest 1st and 3rd score exactly 0 on cases 13 and 14. It is a property of the metric, not a data fault.

## 5. Limitations

- Training-set size and resize differences between our baselines and the published models.
- Baselines are simple, with a small fixed tuning grid. The claim covers only the nine listed baselines.
- Published numbers are copied from papers (several are preprints, none reproduced here). AttUNet's MAE is printed as 0.110, but its per-case mean is 0.1172.
- Skill_MAE for published models uses our mean_map as the reference; it is a normalisation, not a replication of their baselines.
- Timing is one run on a shared laptop CPU.
- One dataset, 5 independent designs; nothing here speaks to CircuitNet-N14.

## 6. Post-run notes (written after the registered run)

- **Uncommitted file at run time.** The result files record `git_dirty_tracked` = `manifests/n14_inventory.json`, the CircuitNet-N14 inventory being rewritten by the concurrent download. It is not read by any ICCAD'23 stage and is outside the code hash; it was committed separately afterwards.
- **Misleading key.** In `results/iccad23/h3.json`, `checks_train_vs_hidden` holds a train-only development check (120 cases, info-level within-train channel duplicates). The cross-split E003 result is in `verdicts.json` and is empty. The H3 verdict is unaffected.
- **Tests.** After the freeze, 4 of 96 tests in `tests/test_checks.py` fail because they assume no `protocol/FROZEN.json` exists. The results are unaffected; the tests will be fixed in a separate commit labelled post hoc.
- **Post hoc code change (after the freeze at 7707b20).** The tests were fixed afterwards, and the freeze guard was made per dataset (`protocol/FROZEN_<key>.json`; `protocol/FROZEN.json` with `"dataset": "iccad23"` is still recognised) so that this freeze cannot unlock CircuitNet-N14. This edits `src/` and `tests/`, so the code hash changes; the ICCAD'23 results stay tied to the hash recorded in `protocol/FROZEN.json` (750d252f..., code at 7707b20), and no registered stage was re-run.
- **Independent check.** Every number in sections 1 to 4 was checked against the result files by a separate verification pass; all matched.

## Appendix: tables (verbatim from `results/iccad23/tables.md`)

## Per case (hidden set; MAE in mV, F1 official strict rule)

| baseline | 7 | 8 | 9 | 10 | 13 | 14 | 15 | 16 | 19 | 20 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero MAE | 1.0439 | 1.5540 | 1.1811 | 1.9483 | 2.2638 | 3.2298 | 2.7686 | 4.6521 | 0.4442 | 0.6994 | 1.9785 |
| mean_map MAE | 0.4711 | 0.5426 | 0.3538 | 0.7716 | 1.0678 | 2.0338 | 1.5730 | 3.4551 | 0.7658 | 0.5708 | 1.1605 |
| scalar MAE | 0.5267 | 0.6322 | 0.4049 | 0.8521 | 1.0968 | 2.0620 | 1.6078 | 3.4819 | 0.7323 | 0.5350 | 1.1932 |
| ridge MAE | 0.4105 | 0.5152 | 0.3644 | 0.6058 | 0.7214 | 1.4524 | 1.0846 | 2.5287 | 0.4972 | 0.5898 | 0.8770 |
| knn MAE | 0.3593 | 0.5140 | 0.3210 | 0.5479 | 0.5400 | 1.0717 | 0.8877 | 2.2535 | 0.4351 | 1.0383 | 0.7968 |
| effdist MAE | 0.4417 | 0.5985 | 0.3612 | 0.8828 | 0.7610 | 1.6576 | 1.3351 | 3.2043 | 0.5835 | 0.4139 | 1.0240 |
| effdist_deg2 MAE | 0.4233 | 0.5996 | 0.3581 | 0.8938 | 0.8421 | 1.7581 | 1.2941 | 3.1644 | 0.5425 | 0.3830 | 1.0259 |
| green MAE | 0.2397 | 0.6036 | 0.2853 | 0.6728 | 0.4224 | 1.3096 | 0.6270 | 1.5165 | 0.1076 | 0.2468 | 0.6031 |
| gbm MAE | 0.4357 | 0.5438 | 0.3171 | 0.6086 | 0.7508 | 1.4940 | 1.2745 | 2.7651 | 0.6234 | 0.5819 | 0.9395 |
| spice MAE | 0.0015 | 0.0023 | 0.0018 | 0.0032 | 0.0038 | 0.0053 | 0.0037 | 0.0065 | 0.0007 | 0.0011 | 0.0030 |
| zero F1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| mean_map F1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| scalar F1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ridge F1 | 0.000 | 0.000 | 0.034 | 0.000 | 0.099 | 0.025 | 0.213 | 0.000 | 0.017 | 0.015 | 0.040 |
| knn F1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| effdist F1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.202 | 0.000 | 0.111 | 0.000 | 0.000 | 0.000 | 0.031 |
| effdist_deg2 F1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| green F1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| gbm F1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.015 | 0.039 | 0.005 |
| spice F1 | 0.980 | 0.980 | 0.963 | 0.977 | 0.981 | 0.966 | 0.984 | 0.970 | 0.948 | 0.987 | 0.974 |

## Per design (mean of the twin pair) and overall

| baseline | real_257x257_11864src | real_489x489_44117src | real_601x601_24988src | real_835x835_64064src | real_870x870_63347src | overall MAE | range over designs | Skill_MAE | designs beating mean_map (of 5) |
|---|---|---|---|---|---|---|---|---|---|
| zero | 2.7468 / 0.000 | 3.7104 / 0.000 | 1.2989 / 0.000 | 1.5647 / 0.000 | 0.5718 / 0.000 | 1.9785 | 0.5718 to 3.7104 | -0.705 | 1 |
| mean_map | 1.5508 / 0.000 | 2.5141 / 0.000 | 0.5069 / 0.000 | 0.5627 / 0.000 | 0.6683 / 0.000 | 1.1605 | 0.5069 to 2.5141 | 0.000 | 0 |
| scalar | 1.5794 / 0.000 | 2.5449 / 0.000 | 0.5795 / 0.000 | 0.6285 / 0.000 | 0.6337 / 0.000 | 1.1932 | 0.5795 to 2.5449 | -0.028 | 1 |
| ridge | 1.0869 / 0.062 | 1.8067 / 0.107 | 0.4628 / 0.000 | 0.4851 / 0.017 | 0.5435 / 0.016 | 0.8770 | 0.4628 to 1.8067 | 0.244 | 5 |
| knn | 0.8058 / 0.000 | 1.5706 / 0.000 | 0.4367 / 0.000 | 0.4344 / 0.000 | 0.7367 / 0.000 | 0.7968 | 0.4344 to 1.5706 | 0.313 | 4 |
| effdist | 1.2093 / 0.101 | 2.2697 / 0.056 | 0.5201 / 0.000 | 0.6220 / 0.000 | 0.4987 / 0.000 | 1.0240 | 0.4987 to 2.2697 | 0.118 | 3 |
| effdist_deg2 | 1.3001 / 0.000 | 2.2292 / 0.000 | 0.5114 / 0.000 | 0.6260 / 0.000 | 0.4627 / 0.000 | 1.0259 | 0.4627 to 2.2292 | 0.116 | 3 |
| green | 0.8660 / 0.000 | 1.0717 / 0.000 | 0.4217 / 0.000 | 0.4790 / 0.000 | 0.1772 / 0.000 | 0.6031 | 0.1772 to 1.0717 | 0.480 | 5 |
| gbm | 1.1224 / 0.000 | 2.0198 / 0.000 | 0.4897 / 0.000 | 0.4628 / 0.000 | 0.6026 / 0.027 | 0.9395 | 0.4628 to 2.0198 | 0.190 | 5 |
| spice | 0.0046 / 0.973 | 0.0051 / 0.977 | 0.0019 / 0.980 | 0.0025 / 0.970 | 0.0009 / 0.968 | 0.0030 | 0.0009 to 0.0051 | 0.997 | 5 |

(per-design cells: MAE mV / F1)

## F1 variants (mean over 10 cases) and number of cases with F1 = 0

| baseline | F1 strict (primary) | F1 >= | F1 top-10% | #F1=0 strict | #F1=0 >= | #F1=0 top-10% |
|---|---|---|---|---|---|---|
| zero | 0.000 | 0.000 | 0.000 | 10 | 10 | 10 |
| mean_map | 0.000 | 0.000 | 0.036 | 10 | 10 | 8 |
| scalar | 0.000 | 0.000 | 0.018 | 10 | 10 | 9 |
| ridge | 0.040 | 0.040 | 0.252 | 4 | 4 | 0 |
| knn | 0.000 | 0.000 | 0.137 | 10 | 10 | 2 |
| effdist | 0.031 | 0.031 | 0.095 | 8 | 8 | 2 |
| effdist_deg2 | 0.000 | 0.000 | 0.037 | 10 | 10 | 6 |
| green | 0.000 | 0.000 | 0.266 | 10 | 10 | 2 |
| gbm | 0.005 | 0.005 | 0.123 | 8 | 8 | 5 |
| spice | 0.974 | 0.974 | 0.996 | 0 | 0 | 0 |

## Published comparison (hidden 10; Skill_MAE reference = our mean_map MAE 1.1605 mV)

| model | MAE mV | F1 | runtime s | Skill_MAE | note |
|---|---|---|---|---|---|
| contest_1st (cadc0009, xlsx) | 0.1347 | 0.455 | 14.772 | 0.884 | official; all 10 valid |
| contest_2nd (cadc1023, xlsx) | 0.1498 | 0.455 | 3.042 | 0.871 | official; all 10 valid |
| contest_3rd (cadc1020, xlsx) | 0.1069 | 0.519 | n/a | 0.917 | FAILED cases 19 and 20; MAE and F1 are means over the 8 valid cases (7-16); xlsx F1 over 10 cases with 0 for the failures is 0.4143 |
| CFIRSTNET (2502.12168) | 0.0533 | 0.723 | 0.409 | 0.954 | inputs resized to 384x384; Xeon E3-1230 v6 + Quadro P4000 |
| WACA-UNet (2507.19197) | 0.0524 | 0.778 | 1.390 | 0.955 | 384x384 Lanczos resize; checkpoint by best validation F1; CPU model not stated |
| LMM-IR (2511.12581) | 0.1356 | 0.583 | 3.050 | 0.883 | 512x512; H100 |
| GL-Inception (2406.06541) | 0.1010 | 0.620 | 3.370 | 0.913 | includes feature extraction |
| AttUNet (2408.03292) | 0.1100 | 0.520 | 5.370 | 0.905 | MAE as printed (per-case mean 0.1172); F1 rule DIFFERS (top 10% of pixels), not the contest rule |
| ours: ridge | 0.8770 | 0.040 | | 0.244 | |
| ours: green | 0.6031 | 0.000 | | 0.480 | |
| ours: gbm | 0.9395 | 0.005 | | 0.190 | |
| ours: spice | 0.0030 | 0.974 | | 0.997 | |
