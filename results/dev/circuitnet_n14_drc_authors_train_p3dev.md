# edalint report: circuitnet_n14_drc_authors / train

*development (train only).* 80 samples, 10.4 s, worst severity **error**.

| check | status | summary |
|---|---|---|
| E001 integrity | ok | 80 samples checked, 0 failed to load, 0 with non-finite values |
| E002 degenerate | flagged | 80 samples: 1 zero label, 0 negative, 0 out of range, 0 all-zero bbox, 0 count mismatches |
| E003 duplicates | ok | 80 samples: 0 exact input-duplicate groups, 0 exact label-duplicate groups, 0 near-duplicate pairs |
| E004 leakage | skipped | skipped: needs at least two splits, have ['train'] |
| E005 orientation | ok | 39 samples; RUDY_long: stored 0.18 |
| E006 metric-sanity | flagged | 80 cases; hotspot base rate median 0.00e+00; 80 without hotspot |

## E001 integrity

Status: **ok**. 80 samples checked, 0 failed to load, 0 with non-finite values

```json
{
 "n_checked": 80
}
```

## E002 degenerate

Status: **flagged**. 80 samples: 1 zero label, 0 negative, 0 out of range, 0 all-zero bbox, 0 count mismatches
- **error**: 1/80 samples have an all-zero label (e.g. `Vortex-small_freq_500_mp_1_fpu_60_fpa_1.0_p_2_fi_ar`)
- **info**: input channel 'macro_region' is all-zero in 40/80 samples
- **info**: archive members not listed in the split CSVs (per design): {'Vortex-small': 1, 'nvdla-small': 12}

```json
{
 "n_checked": 80,
 "n_zero_label": 1,
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
 "max_input_corr_offdiag": 0.7406864371786668,
 "max_label_corr_offdiag": 0.9608051944726811
}
```

## E004 leakage

Status: **skipped** (needs at least two splits, have ['train']). skipped: needs at least two splits, have ['train']

## E005 orientation

Status: **ok**. 39 samples; RUDY_long: stored 0.18

```json
{
 "n_checked": 39,
 "median_abs_corr": {
  "cell_density": {
   "stored": 0.35374200748001977,
   "flip_ud": 0.09529120474305783,
   "flip_lr": 0.11135698202996533,
   "rot180": 0.11049797346004284,
   "transpose": 0.10271171994675048,
   "rot90": 0.11022313989387544,
   "rot270": 0.09566981017915571,
   "antitranspose": 0.10643620019729966
  },
  "RUDY_long": {
   "stored": 0.17854297053292775,
   "flip_ud": 0.13867373797875202,
   "flip_lr": 0.09997825435546626,
   "rot180": 0.10653454835554867,
   "transpose": 0.09060312392093466,
   "rot90": 0.09817663543062631,
   "rot270": 0.10785342496911422,
   "antitranspose": 0.10421883423500718
  },
  "RUDY_short": {
   "stored": 0.49490453250900784,
   "flip_ud": 0.07274335010262538,
   "flip_lr": 0.09778353899947734,
   "rot180": 0.07562855756962761,
   "transpose": 0.08330697347178147,
   "rot90": 0.0901664733804036,
   "rot270": 0.07325591161373701,
   "antitranspose": 0.08965150988806057
  },
  "RUDY_pin_long": {
   "stored": 0.575847823639979,
   "flip_ud": 0.07752608027087618,
   "flip_lr": 0.09942546562705183,
   "rot180": 0.09207050698926447,
   "transpose": 0.08745116923545967,
   "rot90": 0.10072429143382214,
   "rot270": 0.0968582617381125,
   "antitranspose": 0.09106623853482776
  },
  "eGR_horizontal_overflow": {
   "stored": 0.09975977533522855,
   "flip_ud": 0.05513901401852574,
   "flip_lr": 0.06162826989233142,
   "rot180": 0.04859838472335952,
   "transpose": 0.037097989602924285,
   "rot90": 0.04628534895716223,
   "rot270": 0.0382694103597448,
   "antitranspose": 0.04198469914953961
  },
  "eGR_vertical_overflow": {
   "stored": 0.0860055128666619,
   "flip_ud": 0.05110741603840746,
   "flip_lr": 0.056559823100763164,
   "rot180": 0.04514864102976588,
   "transpose": 0.038074745674237166,
   "rot90": 0.03630274654529036,
   "rot270": 0.03709700967675978,
   "antitranspose": 0.03899855317811455
  },
  "GR_horizontal_overflow": {
   "stored": 0.19924823514020723,
   "flip_ud": 0.04792265230415762,
   "flip_lr": 0.03980515088539045,
   "rot180": 0.03836300417424784,
   "transpose": 0.056898647429428344,
   "rot90": 0.04029335253177972,
   "rot270": 0.045968640836941875,
   "antitranspose": 0.040078247356522184
  },
  "GR_vertical_overflow": {
   "stored": 0.25819965954182367,
   "flip_ud": 0.07187983743786716,
   "flip_lr": 0.08993333243970689,
   "rot180": 0.06741650706345172,
   "transpose": 0.092695397603793,
   "rot90": 0.08579511143342348,
   "rot270": 0.07920239362402902,
   "antitranspose": 0.08662638162445105
  },
  "macro_region": {
   "stored": 0.5016241401898512,
   "flip_ud": 0.11322555940970855,
   "flip_lr": 0.2278956828724506,
   "rot180": 0.14449708530070882,
   "transpose": 0.26461360037557313,
   "rot90": 0.1659974655397003,
   "rot270": 0.12272969948953075,
   "antitranspose": 0.08719414316621886
  }
 },
 "primary": {
  "channel": "RUDY_long",
  "stored": 0.17854297053292775,
  "best_alternative": 0.13867373797875202
 }
}
```

## E006 metric-sanity

Status: **flagged**. 80 cases; hotspot base rate median 0.00e+00; 80 without hotspot
- **error**: 80/80 cases have NO hotspot under the rule {'kind': 'abs', 'value': 0.1}: F1 is undefined (0/0) for them (e.g. `Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap`, `Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_2_fi_ap`, `Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_4_fi_ar`)
- **info**: hotspot base rate (rule {'kind': 'abs', 'value': 0.1}): min 0.00e+00, median 0.00e+00, max 0.00e+00; hotspot pixels per case: median 0
- **warning**: 80/80 cases have a single class at threshold 0.1: ROC/PR AUC undefined for them (e.g. `Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap`, `Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_2_fi_ap`, `Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_4_fi_ar`)

```json
{
 "n_checked": 80,
 "rule": {
  "kind": "abs",
  "value": 0.1
 },
 "hotspot_base_rate": {
  "min": 0.0,
  "median": 0.0,
  "max": 0.0
 },
 "hotspot_pixels": {
  "min": 0,
  "median": 0.0,
  "max": 0
 }
}
```
