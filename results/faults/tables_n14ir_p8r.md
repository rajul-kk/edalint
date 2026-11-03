# P8 fault validation: base `n14ir` (20 draws x 100 samples; pool 300 of 300, removed known defects {})

## False positives (clean draws, error or warning)

| check | flagged / draws | rate (Wilson 95%) | findings |
|---|---|---|---|
| E003 | 0/20 | 0.00 (0.00-0.16) | - |
| E004 | 0/20 | 0.00 (0.00-0.16) | - |

## Detection by severity level (0 weakest .. 4 strongest)

| check | fault | L0 | L1 | L2 | L3 | L4 | min level >= 0.90 | validated |
|---|---|---|---|---|---|---|---|---|
| E003 | exact_copy | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | yes |
| E003 | near_copy | 0.00 (0.00-0.16) | 0.00 (0.00-0.16) | 0.95 (0.76-0.99) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 2 | yes |
| E004 | design_overlap | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | yes |

Not applicable on this base: twin_cross (base has no twin metadata)
