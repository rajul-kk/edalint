# edalint report: iccad23 / all

*development (train only).* 120 samples, 0.0 s, worst severity **none**.

| check | status | summary |
|---|---|---|
| E004 leakage | ok | splits ['train', 'test']; designs per split {'train': 105, 'test': 5}; 0 cross-split twin links |

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
