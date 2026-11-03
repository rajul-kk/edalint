# P8 summary (iccad23, n14ir, icb2, icb3, icb4, icb5)

## False-positive rate excluding known findings (flagged draws / 20)

| check | iccad23 | n14ir | icb2 | icb3 | icb4 | icb5 |
|---|---|---|---|---|---|---|
| E001 | - | - | - | - | - | - |
| E002 | - | - | - | - | - | - |
| E003 | 0/20 | 0/20 | 18/20 | 12/20 | 10/20 | 15/20 |
| E004 | 0/20 | 0/20 | - | - | - | - |
| E005 | - | - | - | - | - | - |

## Minimum severity level with detection >= 0.90 (levels 0..4; "-" = never; n/a = not applicable); validated per protocol section 5

| check | fault | iccad23 | n14ir | icb2 | icb3 | icb4 | icb5 |
|---|---|---|---|---|---|---|---|
| E001 | nan_pixels | n/a | n/a | n/a | n/a | n/a | n/a |
| E001 | inf_pixels | n/a | n/a | n/a | n/a | n/a | n/a |
| E001 | grid_mismatch | n/a | n/a | n/a | n/a | n/a | n/a |
| E002 | zero_label | n/a | n/a | n/a | n/a | n/a | n/a |
| E002 | negative_pixel | n/a | n/a | n/a | n/a | n/a | n/a |
| E002 | zero_channel | n/a | n/a | n/a | n/a | n/a | n/a |
| E003 | exact_copy | L0 (validated) | L0 (validated) | L0 | L0 | L0 | L0 |
| E003 | near_copy | L3 (validated) | L2 (validated) | L3 | L2 | L3 | L3 |
| E004 | design_overlap | L0 (validated) | L0 (validated) | n/a | n/a | n/a | n/a |
| E004 | twin_cross | L0 (validated) | n/a | n/a | n/a | n/a | n/a |
| E005 | transpose_label | n/a | n/a | n/a | n/a | n/a | n/a |
| E005 | flip_label | n/a | n/a | n/a | n/a | n/a | n/a |
| E005 | transpose_noisy | n/a | n/a | n/a | n/a | n/a | n/a |
