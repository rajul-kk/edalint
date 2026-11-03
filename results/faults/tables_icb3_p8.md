# P8 fault validation: base `icb3` (20 draws x 200 samples; pool 600 of 600, removed known defects {})

## False positives (clean draws, error or warning)

| check | flagged / draws | rate (Wilson 95%) | findings |
|---|---|---|---|
| E001 | 0/20 | 0.00 (0.00-0.16) | - |
| E002 | 0/20 | 0.00 (0.00-0.16) | - |
| E003 | 16/20 | 0.80 (0.58-0.92) | warning: 4 near-duplicate pair(s) within a split (input and label corr >= 0.999) x2; warning: 1 near-duplicate pair(s) within a split (input and label corr >= 0.999) x7; warning: 2 near-duplicate pair(s) within a split (input and label corr >= 0.999) x2; warning: 3 near-duplicate pair(s) within a split (input and label corr >= 0.999) x4; error: 2 near-duplicate pair(s) ACROSS splits (input and label corr >= 0.999) x1; error: 1 near-duplicate pair(s) ACROSS splits (input and label corr >= 0.999) x1 |
| E004 | 0/20 | 0.00 (0.00-0.16) | - |
| E005 | 0/20 | 0.00 (0.00-0.16) | - |

## Detection by severity level (0 weakest .. 4 strongest)

| check | fault | L0 | L1 | L2 | L3 | L4 | min level >= 0.90 | validated |
|---|---|---|---|---|---|---|---|---|
| E001 | nan_pixels | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | yes |
| E001 | inf_pixels | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | yes |
| E001 | grid_mismatch | - | - | - | - | 1.00 (0.84-1.00) | 4 | yes |
| E002 | zero_label | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | yes |
| E002 | negative_pixel | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | yes |
| E002 | zero_channel | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | none | NO |
| E003 | exact_copy | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | NO |
| E003 | near_copy | 0.10 (0.03-0.30) | 0.10 (0.03-0.30) | 0.85 (0.64-0.95) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 3 | NO |
| E004 | design_overlap | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | yes |
| E005 | transpose_label | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.40 (0.22-0.61) | 1.00 (0.84-1.00) | 4 | NO |
| E005 | flip_label | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.50 (0.30-0.70) | 1.00 (0.84-1.00) | 4 | NO |
| E005 | transpose_noisy | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | yes |

Not applicable on this base: twin_cross (base has no twin metadata)
