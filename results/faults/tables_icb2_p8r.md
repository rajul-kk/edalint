# P8 fault validation: base `icb2` (20 draws x 200 samples; pool 600 of 600, removed known defects {})

## False positives (clean draws, error or warning)

| check | flagged / draws | rate (Wilson 95%) | findings |
|---|---|---|---|
| E003 | 18/20 | 0.90 (0.70-0.97) | error: 2 samples have byte-identical inputs (ACROSS splits ['train', 'val']) x5; warning: 2 samples have byte-identical inputs (within one split) x19; error: 2 samples have byte-identical labels (ACROSS splits ['train', 'val']) x5; error: 1 near-duplicate pair(s) ACROSS splits (input and label corr >= 0.999) x5; warning: 3 near-duplicate pair(s) within a split (input and label corr >= 0.999) x4; warning: 1 near-duplicate pair(s) within a split (input and label corr >= 0.999) x9; warning: 2 samples have byte-identical labels (within one split) x16; warning: 2 near-duplicate pair(s) within a split (input and label corr >= 0.999) x3; error: 3 samples have byte-identical inputs (ACROSS splits ['train', 'val']) x2; error: 3 samples have byte-identical labels (ACROSS splits ['train', 'val']) x2; error: 3 near-duplicate pair(s) ACROSS splits (input and label corr >= 0.999) x1; error: 2 near-duplicate pair(s) ACROSS splits (input and label corr >= 0.999) x2; warning: 4 near-duplicate pair(s) within a split (input and label corr >= 0.999) x1; warning: 3 samples have byte-identical inputs (within one split) x1; warning: 3 samples have byte-identical labels (within one split) x1; warning: 5 near-duplicate pair(s) within a split (input and label corr >= 0.999) x1 |

## Detection by severity level (0 weakest .. 4 strongest)

| check | fault | L0 | L1 | L2 | L3 | L4 | min level >= 0.90 | validated |
|---|---|---|---|---|---|---|---|---|
| E003 | exact_copy | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 0 | NO |
| E003 | near_copy | 0.40 (0.22-0.61) | 0.40 (0.22-0.61) | 0.85 (0.64-0.95) | 1.00 (0.84-1.00) | 1.00 (0.84-1.00) | 3 | NO |
