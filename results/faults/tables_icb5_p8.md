# P8 fault validation: base `icb5` (20 draws x 200 samples; pool 600 of 600, removed known defects {})

## False positives (clean draws, error or warning)

| check | flagged / draws | rate (Wilson 95%) | findings |
|---|---|---|---|
| E001 | 0/20 | 0.00 (0.00-0.16) | - |
| E002 | 20/20 | 1.00 (0.84-1.00) | warning: input channel 'ambient_K' is a constant non-zero map in 200 sample(s) x20; warning: input channel 'h_w_m2k' is a constant non-zero map in 200 sample(s) x20; warning: input channel 'r_convec_k_per_w' is a constant non-zero map in 200 sample(s) x20 |
| E003 | 0/20 | 0.00 (0.00-0.16) | - |
| E004 | 0/20 | 0.00 (0.00-0.16) | - |
| E005 | 0/20 | 0.00 (0.00-0.16) | - |

## Detection by severity level (0 weakest .. 4 strongest)

| check | fault | L0 | L1 | L2 | L3 | L4 | min level >= 0.90 | validated |
|---|---|---|---|---|---|---|---|---|
| E001 | nan_pixels | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | yes |
| E001 | inf_pixels | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | yes |
| E001 | grid_mismatch | - | - | - | - | 1.00 (0.84-1.00) | 4 | yes |
| E002 | zero_label | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | NO |
| E002 | negative_pixel | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | NO |
| E002 | zero_channel | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | none | NO |
| E003 | exact_copy | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | yes |
| E003 | near_copy | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | none | NO |
| E004 | design_overlap | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | yes |
| E005 | transpose_label | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.05 (0.01-0.24) | 0.80 (0.58-0.92) | none | NO |
| E005 | flip_label | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.15 (0.05-0.36) | 1.00 (0.84-1.00) | 4 | NO |
| E005 | transpose_noisy | 0.80 (0.58-0.92) | 0.80 (0.58-0.92) | 0.80 (0.58-0.92) | 0.80 (0.58-0.92) | 0.80 (0.58-0.92) | none | NO |

Not applicable on this base: twin_cross (base has no twin metadata)
