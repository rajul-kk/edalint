# P8 fault validation: base `icb4` (20 draws x 200 samples; pool 600 of 600, removed known defects {})

## False positives (clean draws, error or warning)

| check | flagged / draws | rate (Wilson 95%) | findings |
|---|---|---|---|
| E003 | 10/20 | 0.50 (0.30-0.70) | error: 1 near-duplicate pair(s) ACROSS splits (input and label corr >= 0.999) x5; warning: 1 near-duplicate pair(s) within a split (input and label corr >= 0.999) x6 |

## Detection by severity level (0 weakest .. 4 strongest)

| check | fault | L0 | L1 | L2 | L3 | L4 | min level >= 0.90 | validated |
|---|---|---|---|---|---|---|---|---|
| E003 | exact_copy | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | NO |
| E003 | near_copy | 0.25 (0.11-0.47) | 0.25 (0.11-0.47) | 0.75 (0.53-0.89) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 3 | NO |
