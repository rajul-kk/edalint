# P8 fault validation: base `icb3` (20 draws x 200 samples; pool 600 of 600, removed known defects {})

## False positives (clean draws, error or warning)

| check | flagged / draws | rate (Wilson 95%) | findings |
|---|---|---|---|
| E003 | 12/20 | 0.60 (0.39-0.78) | error: 1 near-duplicate pair(s) ACROSS splits (input and label corr >= 0.999) x1; warning: 1 near-duplicate pair(s) within a split (input and label corr >= 0.999) x5; warning: 3 near-duplicate pair(s) within a split (input and label corr >= 0.999) x3; warning: 2 near-duplicate pair(s) within a split (input and label corr >= 0.999) x3 |

## Detection by severity level (0 weakest .. 4 strongest)

| check | fault | L0 | L1 | L2 | L3 | L4 | min level >= 0.90 | validated |
|---|---|---|---|---|---|---|---|---|
| E003 | exact_copy | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | NO |
| E003 | near_copy | 0.05 (0.01-0.24) | 0.05 (0.01-0.24) | 0.90 (0.70-0.97) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 2 | NO |
