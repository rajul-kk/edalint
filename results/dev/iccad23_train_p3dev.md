# edalint report: iccad23 / train

*development (train only).* 110 samples, 211.6 s, worst severity **warning**.

| check | status | summary |
|---|---|---|
| E001 integrity | ok | 110 samples checked, 0 failed to load, 0 with non-finite values |
| E002 degenerate | flagged | 110 samples: 0 zero label, 0 negative, 0 out of range, 0 all-zero bbox, 0 count mismatches |
| E003 duplicates | ok | 110 samples: 0 exact input-duplicate groups, 0 exact label-duplicate groups, 0 near-duplicate pairs |
| E004 leakage | ok | splits ['train', 'test']; designs per split {'train': 105, 'test': 5}; 0 cross-split twin links |
| E005 orientation | ok | 40 samples; current: stored 0.68 |
| E006 metric-sanity | flagged | 110 cases; hotspot base rate median 1.39e-03; 0 without hotspot |

## E001 integrity

Status: **ok**. 110 samples checked, 0 failed to load, 0 with non-finite values
- **info**: 97 distinct grid shapes across 110 samples (min (204, 204), max (930, 930))

```json
{
 "n_checked": 110
}
```

## E002 degenerate

Status: **flagged**. 110 samples: 0 zero label, 0 negative, 0 out of range, 0 all-zero bbox, 0 count mismatches
- **warning**: input channel 'pdn_density' is a constant non-zero map in 5 sample(s) (e.g. `real/testcase2`, `real/testcase4`, `real/testcase6`)

```json
{
 "n_checked": 110,
 "n_zero_label": 0,
 "n_const_label": 0,
 "n_negative_label": 0,
 "n_out_of_range_label": 0,
 "n_all_zero_inputs": 0,
 "n_all_zero_bbox": 0,
 "channel_all_zero_counts": {}
}
```

## E003 duplicates

Status: **ok**. 110 samples: 0 exact input-duplicate groups, 0 exact label-duplicate groups, 0 near-duplicate pairs
- **info**: 10 channel-level exact duplicates across samples (channels ['current', 'eff_dist']); expected when samples share a configuration (ICCAD twin cases share a current map, CircuitNet runs with the same macro placement share macro_region) (e.g. `real/testcase1`, `real/testcase2`)

```json
{
 "n_checked": 110,
 "exact_input_groups": 0,
 "exact_label_groups": 0,
 "n_near_dup_pairs": 0,
 "n_same_input_different_label_pairs": 0,
 "max_input_corr_offdiag": 0.6962431658190762,
 "max_label_corr_offdiag": 0.9638560493891637
}
```

## E004 leakage

Status: **ok**. splits ['train', 'test']; designs per split {'train': 105, 'test': 5}; 0 cross-split twin links

```json
{
 "designs_per_split": {
  "train": 105,
  "test": 5
 },
 "n_samples_with_twin": 20,
 "twin_links_across_splits": 0,
 "independent_designs_with_twins": {
  "train": 5,
  "test": 5
 },
 "random_vs_design_gap": "not computed (run_gap is off: the gap is computed only when the protocol runs it)"
}
```

## E005 orientation

Status: **ok**. 40 samples; current: stored 0.68
- **info**: channel 'eff_dist': |corr| with the label is 0.18 as stored but 0.31 when 'rot270' (better in 70% of samples) [non-primary channel: informational]

```json
{
 "n_checked": 40,
 "median_abs_corr": {
  "current": {
   "stored": 0.68279063731484,
   "flip_ud": 0.40654134672281816,
   "flip_lr": 0.2977498966714225,
   "rot180": 0.2625300389391133,
   "transpose": 0.29077086028192156,
   "rot90": 0.2918665456434531,
   "rot270": 0.28030437982256884,
   "antitranspose": 0.3471143708586953
  },
  "eff_dist": {
   "stored": 0.17691199214236752,
   "flip_ud": 0.2819857088019569,
   "flip_lr": 0.29054867120371697,
   "rot180": 0.2856029906241079,
   "transpose": 0.2393012974652997,
   "rot90": 0.2721494073357045,
   "rot270": 0.31087986786133603,
   "antitranspose": 0.26330981235869627
  },
  "pdn_density": {
   "stored": 0.6305256792385362,
   "flip_ud": 0.14199277009929762,
   "flip_lr": 0.12379968166043527,
   "rot180": 0.13943753119209779,
   "transpose": 0.15218023454955867,
   "rot90": 0.1587805867949703,
   "rot270": 0.09920526448874709,
   "antitranspose": 0.16172395083081123
  }
 },
 "primary": {
  "channel": "current",
  "stored": 0.68279063731484,
  "best_alternative": 0.40654134672281816
 }
}
```

## E006 metric-sanity

Status: **flagged**. 110 cases; hotspot base rate median 1.39e-03; 0 without hotspot
- **info**: hotspot base rate (rule {'kind': 'frac_of_max', 'value': 0.9}): min 1.49e-04, median 1.39e-03, max 3.81e-02; hotspot pixels per case: median 472
- **warning**: F1 of the pseudo-prediction 'smooth1.5' moves by 0.57 when the threshold changes among [0.85, 0.9, 0.95] x max (values ['0.84', '0.72', '0.27'])
- **warning**: F1 of the pseudo-prediction 'scale0.95' moves by 0.56 when the threshold changes among [0.85, 0.9, 0.95] x max (values ['0.56', '0.41', '0.00'])
- **warning**: F1 of the pseudo-prediction 'scale1.05' moves by 0.27 when the threshold changes among [0.85, 0.9, 0.95] x max (values ['0.68', '0.57', '0.41'])
- **info**: F1 of an all-zero prediction is 0.00 (the contest scores a prediction with no hotspot pixel as F1 = 0); F1 of the label itself is 1.00

```json
{
 "n_checked": 110,
 "rule": {
  "kind": "frac_of_max",
  "value": 0.9
 },
 "hotspot_base_rate": {
  "min": 0.00014880498153292023,
  "median": 0.001394274635792484,
  "max": 0.038081410837018485
 },
 "hotspot_pixels": {
  "min": 49,
  "median": 471.5,
  "max": 2859
 },
 "mean_f1_pseudo_predictions": {
  "scale0.95@0.85": 0.5640262310547309,
  "scale0.95@0.9": 0.4084811065291178,
  "scale0.95@0.95": 0.0,
  "scale1.05@0.85": 0.6818313925754736,
  "scale1.05@0.9": 0.5677393752146159,
  "scale1.05@0.95": 0.4136143343129032,
  "smooth1.5@0.85": 0.8357300987542415,
  "smooth1.5@0.9": 0.723887680293791,
  "smooth1.5@0.95": 0.26678174310017,
  "zero@0.9": 0.0,
  "golden@0.9": 1.0
 },
 "f1_spread_over_threshold": {
  "smooth1.5": 0.5689483556540715,
  "scale0.95": 0.5640262310547309,
  "scale1.05": 0.2682170582625705
 }
}
```
