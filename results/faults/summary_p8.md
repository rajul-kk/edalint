# P8 summary (iccad23, n14ir, icb2, icb3, icb4, icb5)

## False-positive rate excluding known findings (flagged draws / 20)

| check | iccad23 | n14ir | icb2 | icb3 | icb4 | icb5 |
|---|---|---|---|---|---|---|
| E001 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| E002 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| E003 | 0/20 | 0/20 | 18/20 | 16/20 | 0/20 | 0/20 |
| E004 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| E005 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |

## Minimum severity level with detection >= 0.90 (levels 0..4; "-" = never; n/a = not applicable); validated per protocol section 5

| check | fault | iccad23 | n14ir | icb2 | icb3 | icb4 | icb5 |
|---|---|---|---|---|---|---|---|
| E001 | nan_pixels | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) |
| E001 | inf_pixels | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) |
| E001 | grid_mismatch | L4 (validated) | L4 (validated) | L4 (validated) | L4 (validated) | L4 (validated) | L4 (validated) |
| E002 | zero_label | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) |
| E002 | negative_pixel | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) |
| E002 | zero_channel | - | - | - | - | - | - |
| E003 | exact_copy | L0 (validated) | L0 (validated) | L0 | L0 | L0 (validated) | L0 (validated) |
| E003 | near_copy | L3 (validated) | - | L2 | L3 | - | - |
| E004 | design_overlap | - | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) |
| E004 | twin_cross | L0 (validated) | n/a | n/a | n/a | n/a | n/a |
| E005 | transpose_label | L4 | L4 | L4 | L4 | L4 | - |
| E005 | flip_label | L4 | L4 | L4 | L4 | L4 | L4 |
| E005 | transpose_noisy | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) | L0 (validated) | - |
