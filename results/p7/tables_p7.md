# P7 (IR drop only): retrained MAVI on the N14 split

Authors' architecture and recipe, reduced budget (30,000 iterations, fp16 training), our split. Reference is a same-split retrain, not the published model.

Seeds defined: 20261101, 20261102, 20261103 of 20261101, 20261102, 20261103

## Verdict

| item | value |
|---|---|
| M_t (mean_map, design-macro ROC area) | 0.6301 |
| R_t (mean over seeds, design-macro) | 0.7928 |
| H-A gate threshold (0.5 x G_ref) | 0.7486 |
| H-A gate | PASS |
| best baseline / G_b | ridge / 0.2209 |
| G_R | 0.1627 |
| H-B | TRUE |
| H-C D = R_t - 0.8671 | -0.0743 (split/preprocessing effect, signed D) |

Registered N14 H1 (IR drop, unchanged): True, ratio 0.93; P7 agrees: True

## Sensitivities (cannot change the verdict)

| sensitivity | result |
|---|---|
| best seed (20261101) | 0.7995: TRUE |
| worst seed (20261103) | 0.7854: TRUE |
| seed-sensitive | False |
| anchored ROC area | R 0.8061, G_b 0.2209: TRUE (agrees True) |
| sample-pooled | R 0.8664, G_b 0.2238: TRUE (agrees True) |
| G_R = max(G_R, G_ref) | TRUE; budget-sensitive: False |
| designs beating mean_map | 3 of 3 |

## Per design (official ROC area)

| design | mean_map | model (seed mean) |
|---|---|---|
| Vortex-large | 0.6172 | 0.8501 |
| openc910-1 | 0.6277 | 0.6570 |
| zero-riscy | 0.6454 | 0.8713 |

## Rows with no positive pixel (ROC rule) and undefined NRMS, per design and seed

| seed | design | n_samples | n_no_positive | n_undefined_nrms | NRMS finite mean | NRMS (inf=0.05) |
|---|---|---|---|---|---|---|
| 20261101 | Vortex-large | 47 | 0 | 0 | 0.4799 | 0.4799 |
| 20261101 | openc910-1 | 56 | 0 | 0 | 0.2507 | 0.2507 |
| 20261101 | zero-riscy | 3441 | 602 | 0 | 0.2344 | 0.2344 |
| 20261102 | Vortex-large | 47 | 0 | 0 | 0.4791 | 0.4791 |
| 20261102 | openc910-1 | 56 | 0 | 0 | 0.2524 | 0.2524 |
| 20261102 | zero-riscy | 3441 | 602 | 0 | 0.2354 | 0.2354 |
| 20261103 | Vortex-large | 47 | 0 | 0 | 0.4819 | 0.4819 |
| 20261103 | openc910-1 | 56 | 0 | 0 | 0.2535 | 0.2535 |
| 20261103 | zero-riscy | 3441 | 602 | 0 | 0.2359 | 0.2359 |
