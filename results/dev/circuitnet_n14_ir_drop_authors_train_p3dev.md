# edalint report: circuitnet_n14_ir_drop_authors / train

*development (train only).* 80 samples, 20.5 s, worst severity **error**.

| check | status | summary |
|---|---|---|
| E001 integrity | ok | 80 samples checked, 0 failed to load, 0 with non-finite values |
| E002 degenerate | flagged | 80 samples: 1 zero label, 0 negative, 0 out of range, 0 all-zero bbox, 0 count mismatches |
| E003 duplicates | ok | 80 samples: 0 exact input-duplicate groups, 0 exact label-duplicate groups, 0 near-duplicate pairs |
| E004 leakage | skipped | skipped: needs at least two splits, have ['train'] |
| E005 orientation | ok | 39 samples; power_all: stored 0.85 |
| E006 metric-sanity | flagged | 80 cases; hotspot base rate median 3.23e-03; 2 without hotspot |

## E001 integrity

Status: **ok**. 80 samples checked, 0 failed to load, 0 with non-finite values

```json
{
 "n_checked": 80
}
```

## E002 degenerate

Status: **flagged**. 80 samples: 1 zero label, 0 negative, 0 out of range, 0 all-zero bbox, 0 count mismatches
- **error**: 1/80 samples have an all-zero label (e.g. `nvdla-small_freq_200_mp_1_fpu_55_fpa_1.0_p_1_fi_ar`)
- **info**: archive members not listed in the split CSVs (per design): {'Vortex-small': 1, 'nvdla-small': 4}

```json
{
 "n_checked": 80,
 "n_zero_label": 1,
 "n_const_label": 0,
 "n_negative_label": 0,
 "n_out_of_range_label": 0,
 "n_all_zero_inputs": 0,
 "n_all_zero_bbox": 0,
 "channel_all_zero_counts": {}
}
```

## E003 duplicates

Status: **ok**. 80 samples: 0 exact input-duplicate groups, 0 exact label-duplicate groups, 0 near-duplicate pairs

```json
{
 "n_checked": 80,
 "exact_input_groups": 0,
 "exact_label_groups": 0,
 "n_near_dup_pairs": 0,
 "n_same_input_different_label_pairs": 0,
 "max_input_corr_offdiag": 0.7296330518847748,
 "max_label_corr_offdiag": 0.9663380157635594
}
```

## E004 leakage

Status: **skipped** (needs at least two splits, have ['train']). skipped: needs at least two splits, have ['train']

## E005 orientation

Status: **ok**. 39 samples; power_all: stored 0.85

```json
{
 "n_checked": 39,
 "median_abs_corr": {
  "power_i": {
   "stored": 0.788053058678496,
   "flip_ud": 0.34241075760595496,
   "flip_lr": 0.37974844058522705,
   "rot180": 0.25978407384624275,
   "transpose": 0.41634409869420275,
   "rot90": 0.2624084204083881,
   "rot270": 0.24196819410591178,
   "antitranspose": 0.3566751785611091
  },
  "power_s": {
   "stored": 0.888561475313872,
   "flip_ud": 0.3489056323824635,
   "flip_lr": 0.37818763111739656,
   "rot180": 0.3229577876992443,
   "transpose": 0.4443958575920187,
   "rot90": 0.24977130307810905,
   "rot270": 0.2551313470172555,
   "antitranspose": 0.39239369118615897
  },
  "power_sca": {
   "stored": 0.8793134760782724,
   "flip_ud": 0.34408652508322674,
   "flip_lr": 0.37436443275119513,
   "rot180": 0.312068791698547,
   "transpose": 0.4463645834504883,
   "rot90": 0.2427138590827267,
   "rot270": 0.24339284061729669,
   "antitranspose": 0.38312111500850043
  },
  "power_all": {
   "stored": 0.8455346552237956,
   "flip_ud": 0.3445142270140548,
   "flip_lr": 0.3906893821553369,
   "rot180": 0.2873128359965273,
   "transpose": 0.4432288481630479,
   "rot90": 0.25197596244844267,
   "rot270": 0.2422356967272646,
   "antitranspose": 0.37598339239032874
  },
  "power_t": {
   "stored": 0.8402652770895219,
   "flip_ud": 0.32438773589857123,
   "flip_lr": 0.3553623753647488,
   "rot180": 0.26047994100542876,
   "transpose": 0.41493137901573396,
   "rot90": 0.1958977497635825,
   "rot270": 0.22468890363594884,
   "antitranspose": 0.37982091008154717
  }
 },
 "primary": {
  "channel": "power_all",
  "stored": 0.8455346552237956,
  "best_alternative": 0.4432288481630479
 }
}
```

## E006 metric-sanity

Status: **flagged**. 80 cases; hotspot base rate median 3.23e-03; 2 without hotspot
- **error**: 2/80 cases have NO hotspot under the rule {'kind': 'abs', 'value': 0.9885}: F1 is undefined (0/0) for them (e.g. `nvdla-small_freq_200_mp_1_fpu_55_fpa_1.0_p_1_fi_ar`, `nvdla-small_freq_500_mp_1_fpu_55_fpa_1.0_p_1_fi_ar`)
- **warning**: 1/80 cases have fewer than 5 hotspot pixels: their F1 is decided by a few pixels
- **info**: hotspot base rate (rule {'kind': 'abs', 'value': 0.9885}): min 0.00e+00, median 3.23e-03, max 4.35e-02; hotspot pixels per case: median 212
- **warning**: 2/80 cases have a single class at threshold 0.9885: ROC/PR AUC undefined for them (e.g. `nvdla-small_freq_200_mp_1_fpu_55_fpa_1.0_p_1_fi_ar`, `nvdla-small_freq_500_mp_1_fpu_55_fpa_1.0_p_1_fi_ar`)

```json
{
 "n_checked": 80,
 "rule": {
  "kind": "abs",
  "value": 0.9885
 },
 "hotspot_base_rate": {
  "min": 0.0,
  "median": 0.00322723388671875,
  "max": 0.0435028076171875
 },
 "hotspot_pixels": {
  "min": 0,
  "median": 211.5,
  "max": 2851
 }
}
```
