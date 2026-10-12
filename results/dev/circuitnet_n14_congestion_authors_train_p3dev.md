# edalint report: circuitnet_n14_congestion_authors / train

*development (train only).* 80 samples, 4.3 s, worst severity **info**.

| check | status | summary |
|---|---|---|
| E001 integrity | ok | 80 samples checked, 0 failed to load, 0 with non-finite values |
| E002 degenerate | ok | 80 samples: 0 zero label, 0 negative, 0 out of range, 0 all-zero bbox, 0 count mismatches |
| E003 duplicates | ok | 80 samples: 0 exact input-duplicate groups, 0 exact label-duplicate groups, 0 near-duplicate pairs |
| E004 leakage | skipped | skipped: needs at least two splits, have ['train'] |
| E005 orientation | ok | 40 samples; RUDY: stored 0.33 |
| E006 metric-sanity | ok | 80 cases; hotspot base rate median 5.00e-02; 0 without hotspot |

## E001 integrity

Status: **ok**. 80 samples checked, 0 failed to load, 0 with non-finite values

```json
{
 "n_checked": 80
}
```

## E002 degenerate

Status: **ok**. 80 samples: 0 zero label, 0 negative, 0 out of range, 0 all-zero bbox, 0 count mismatches
- **info**: input channel 'macro_region' is all-zero in 40/80 samples
- **info**: archive members not listed in the split CSVs (per design): {'Vortex-small': 1, 'nvdla-small': 12}

```json
{
 "n_checked": 80,
 "n_zero_label": 0,
 "n_const_label": 0,
 "n_negative_label": 0,
 "n_out_of_range_label": 0,
 "n_all_zero_inputs": 0,
 "n_all_zero_bbox": 0,
 "channel_all_zero_counts": {
  "macro_region": 40
 }
}
```

## E003 duplicates

Status: **ok**. 80 samples: 0 exact input-duplicate groups, 0 exact label-duplicate groups, 0 near-duplicate pairs
- **info**: 14 channel-level exact duplicates across samples (channels ['macro_region']); expected when samples share a configuration (ICCAD twin cases share a current map, CircuitNet runs with the same macro placement share macro_region) (e.g. `nvdla-small_freq_200_mp_1_fpu_55_fpa_1.0_p_1_fi_ar`, `nvdla-small_freq_200_mp_1_fpu_55_fpa_1.0_p_4_fi_ap`)

```json
{
 "n_checked": 80,
 "exact_input_groups": 0,
 "exact_label_groups": 0,
 "n_near_dup_pairs": 0,
 "n_same_input_different_label_pairs": 0,
 "max_input_corr_offdiag": 0.9808400675873372,
 "max_label_corr_offdiag": 0.582771366602664
}
```

## E004 leakage

Status: **skipped** (needs at least two splits, have ['train']). skipped: needs at least two splits, have ['train']

## E005 orientation

Status: **ok**. 40 samples; RUDY: stored 0.33

```json
{
 "n_checked": 40,
 "median_abs_corr": {
  "RUDY": {
   "stored": 0.3284889997975674,
   "flip_ud": 0.2217029553193135,
   "flip_lr": 0.20313621467707021,
   "rot180": 0.1820048807115115,
   "transpose": 0.2433555644428762,
   "rot90": 0.22997837628762746,
   "rot270": 0.19445725528690727,
   "antitranspose": 0.24863719978853077
  },
  "RUDY_pin": {
   "stored": 0.8395850281739683,
   "flip_ud": 0.33472938915763956,
   "flip_lr": 0.34204226769007184,
   "rot180": 0.1849922944560533,
   "transpose": 0.34609315663928264,
   "rot90": 0.17975602693414583,
   "rot270": 0.19796754138148834,
   "antitranspose": 0.23949828389052197
  },
  "macro_region": {
   "stored": 0.5940445682764426,
   "flip_ud": 0.2448640339591917,
   "flip_lr": 0.3710912876339877,
   "rot180": 0.2015188555907771,
   "transpose": 0.27270953800580655,
   "rot90": 0.08200800066703338,
   "rot270": 0.10277818661642574,
   "antitranspose": 0.1211625196082961
  }
 },
 "primary": {
  "channel": "RUDY",
  "stored": 0.3284889997975674,
  "best_alternative": 0.24863719978853077
 }
}
```

## E006 metric-sanity

Status: **ok**. 80 cases; hotspot base rate median 5.00e-02; 0 without hotspot
- **info**: hotspot base rate (rule {'kind': 'top_frac', 'value': 0.05}): min 5.00e-02, median 5.00e-02, max 5.01e-02; hotspot pixels per case: median 3277

```json
{
 "n_checked": 80,
 "rule": {
  "kind": "top_frac",
  "value": 0.05
 },
 "hotspot_base_rate": {
  "min": 0.0500030517578125,
  "median": 0.0500030517578125,
  "max": 0.05010986328125
 },
 "hotspot_pixels": {
  "min": 3277,
  "median": 3277.0,
  "max": 3284
 }
}
```
