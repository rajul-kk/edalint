# Scoop check: split-leakage / test-set-tuning audits and UQ for IR-drop / congestion predictors (P9)

Date: 2026-11-05. Scope: is there published work that (a) audits ICCAD'23 Problem C or CircuitNet (IR drop, congestion, DRC) for
cross-split exact or near-duplicate layouts, design-level leakage or test-set tuning, or (b) gives calibrated uncertainty (conformal
intervals, ensembles with coverage, calibration error, OOD degradation) for IR-drop or congestion map predictors?

Source tags: **[abstract]** = abstract page read; **[PDF]** = full text read (here: in the P2 literature pass, `docs/published_numbers.md`);
**[README]** = repository README read; **[code]** = repository source file read; **[snippet]** = search-engine result text only (not opened).
Seven web searches (standard and extended mode, 2026-11-05) plus the P2 sources. A search that returns nothing is not proof of absence.

## Searches run
1. "uncertainty quantification IR drop prediction conformal OR Bayesian OR ensemble": only generic conformal/ensemble papers (neural operators, PINNs). Nothing on IR drop. [snippet]
2. "routing congestion prediction uncertainty quantification out-of-distribution design calibration": traffic forecasting only. [snippet]
3. "CircuitNet data leakage duplicate train test split audit": generic leakage tools (leakfence, leakaudit pHash). Nothing on CircuitNet. [snippet]
4. "IR drop prediction neural network uncertainty estimation power grid arXiv" (extended): PowerNet 2011.13494, PDNNet 2403.18569, AttUNet 2408.03292, Jin et al. 2502.05345, CFIRSTNET 2502.12168, 2601.22707 (CNN surrogate). None reports intervals, coverage or calibration. [snippet]
5. "conformal prediction EDA physical design congestion OR DRC OR IR drop" (extended): Holtz et al. 2403.00103, EDALearn 2312.01674, CircuitNet 2022 letter. No conformal/calibrated intervals. [snippet]
6. "CircuitNet OR ICCAD 2023 split leakage design-level generalization" (extended): CircuitNet 3.0 (source-level split, augmented variants of test sources excluded, "verified with netlist-level graph comparison"), CircuitNet 2.0 cross-design-group congestion experiment, DE-HNN cross-design evaluation. [snippet]
7. "uncertainty-aware routability / congestion / hotspot deep ensemble MC dropout" (extended): RouteNet, CLDRoute 2607.16674, FPGA routability ensembles (accuracy ensembles, not UQ). [snippet]

## Closest prior work

| work | what it does | overlap with P9 | tag |
|---|---|---|---|
| Holtz, Wang, Cheng, Lin, arXiv 2403.00103 (2024) | adversarial "imperceptible" perturbations of placements make CNN/GNN congestion predictors report false congestion drops; robust training | robustness, not leakage, not calibrated UQ, not OOD-across-designs coverage | [abstract] |
| CLDRoute, arXiv 2607.16674 | latent diffusion for congestion/DRC maps on CircuitNet N28/N14; reports an "uncertainty-error correlation" for congestion from 8 diffusion draws; design-wise split with a validation design | the only UQ-adjacent number found; no coverage, no conformal calibration, no interval width, no OOD degradation analysis | [PDF] (P2), [README] |
| CircuitNet 2.0 (ICLR 2024) | train on one design group, test on others (congestion) | cross-design generalisation of point accuracy; no UQ, no duplicate audit | [PDF] (P2) |
| CircuitNet 3.0 (ICLR 2026) | source-level split, augmented variants of test sources excluded, checked by netlist graph comparison | leakage prevention for timing/power tasks only (no map tasks; data card 4) | [README], [snippet] |
| GIF, arXiv 2604.09999 | notes all-zero bounding boxes in N14 as an aside | degenerate samples, not leakage | [PDF] (P2) |
| ICCAD'23 contest v4 note | "removal of duplicate files in fake circuit benchmarks" | organisers removed exact duplicates; no near-duplicate or cross-split analysis published | [PDF] (P1) |
| leakaudit (CRAN), leakfence (PyPI) | generic pHash / row-hash leakage tools | method only; not applied to EDA maps | [snippet] |
| generic conformal / ensemble UQ for surrogates (2606.09923, 2509.13717) | ensembles under-cover until conformalised (physics simulation) | method background only | [snippet] |
| this repo: E003/E004, N14 C1, ICCAD'23 H3 | exact and near duplicates (E003), design overlap / twins / families (E004), random-vs-design split gap (C1) | the existing base that P9 extends (see `docs/leakage_uq_design.md` section 1) | repo |

## Verdict
* **(a) Leakage / tuning audit: open.** No published audit of cross-split near-duplicates, design-family proximity or test-set tuning
  for ICCAD'23 Problem C or CircuitNet IR drop / congestion / DRC was found. This repo's own registered N14 and ICCAD'23 runs already
  cover exact duplicates and the random-vs-design gap, so the new part is the near-duplicate nearest-neighbour analysis on test rows,
  the family-proximity analysis, published-split checks and the recipe (test-set exposure) review.
* **(b) UQ / OOD calibration for IR-drop and congestion map predictors: open**, with one adjacent item: CLDRoute reports an
  uncertainty-error correlation for congestion diffusion samples, without coverage or calibration. No conformal intervals, coverage
  versus nominal or cross-design OOD calibration study was found for either task.
* Confidence: moderate. Searches were English-language web search plus the P2 PDF reading; Chinese-language venues, DAC/ICCAD
  proceedings behind paywalls and 2026 workshop papers may hold unindexed work. Re-check before any write-up.
