# Development results (train only)

Everything in this directory was produced in P3 to check that the tool works. **No test split was read or scored**: ICCAD'23
hidden cases and the CircuitNet-N14 test designs were not touched (the dataset layer refuses them without `protocol/FROZEN.json`).
Label every number as "development (train only)". They are not registered results and are not part of the audit.

Each JSON carries its provenance (git HEAD, code hash, versions, command line). Files are run-once; they were produced on the
P3 working tree, so `git_dirty_tracked` may be non-empty in them.

| file | what | data |
|---|---|---|
| `iccad23_train_p3dev.*` | checks E001-E006 | 100 fake + 10 visible real (train) |
| `iccad23_all_p3dev_metadata_only.*` | E004 on metadata of all splits (no map or label read) | ids and file layout only |
| `iccad23_baselines_fake_to_visible_p3dev_fit100.json` | zoo fitted on 100 fake, scored on the 10 visible real, incl. exact solve | train only |
| `iccad23_loo_visible_p3dev.json` | leave-one-visible-design-out (fit = 60 fake + 8 real, score = held-out twin pair) | train only |
| `circuitnet_n14_{ir_drop,drc,congestion}_authors_train_p3dev.*` | checks on a 40-per-design subset of Vortex-small and nvdla-small (authors mode) | N14 train designs available at the time |
| `circuitnet_n14_drc_authors_all_p3dev_metadata_only.*` | E004 on metadata, train vs test designs | ids only |
| `circuitnet_n14_ir_drop_authors_baselines_Vortex-small_to_nvdla-small_p3dev.json` | zoo fitted on Vortex-small, scored on nvdla-small | N14 train designs |
| `n14_ir_label_clip_probe_p3dev.json` | raw IR_drop label statistics against the authors' clip at 50 | N14 train designs |
