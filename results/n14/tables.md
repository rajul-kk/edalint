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
