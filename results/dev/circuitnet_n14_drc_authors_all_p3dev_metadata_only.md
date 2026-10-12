# edalint report: circuitnet_n14_drc_authors / all

*development (train only).* 200 samples, 0.0 s, worst severity **info**.

| check | status | summary |
|---|---|---|
| E004 leakage | ok | splits ['train', 'test']; designs per split {'train': 2, 'test': 3}; 0 cross-split twin links |

## E004 leakage

Status: **ok**. splits ['train', 'test']; designs per split {'train': 2, 'test': 3}; 0 cross-split twin links
- **info**: related designs (same RTL family) on both sides of train/test: {'vortex': (['Vortex-small'], ['Vortex-large'])}

```json
{
 "designs_per_split": {
  "train": 2,
  "test": 3
 },
 "n_samples_with_twin": 0,
 "twin_links_across_splits": 0,
 "random_vs_design_gap": "not computed (run_gap is off: the gap is computed only when the protocol runs it)"
}
```
