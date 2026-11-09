# P9 L004: test-set tuning in published recipes (descriptive)

Protocol `protocol/protocol_p9.md` section 4.4. Classes: **T0** fixed budget, last checkpoint, no validation; **T1** selection on a validation set
disjoint from test; **T2** test metrics computed or logged during training (exposure), selection on validation; **T3** selection or hyperparameters
chosen on test (stated or visible in code); **U** unknown. Only T3 is a fault. Tags: [code] source read, [README], [PDF] full text read
(P2, `docs/published_numbers.md`), [abstract]. The classes were written after the CircuitNet and CFIRSTNET code had been read (protocol section 2),
so this table is descriptive and carries no hypothesis test. Read 2026-11-05.

| recipe | benchmark | what was read | evidence | class |
|---|---|---|---|---|
| CircuitNet official (MAVI / GPDL / RouteNet), commit 41ade1d | CircuitNet N28 / N14 | [code] `routability_ir_drop_prediction/train.py`, `utils/configs.py`, `test.py`; [README] tutorial | no validation split or loop; fixed `max_iters` 200,000, cosine schedule, checkpoint every 10,000 iterations; `test.py --pretrained <path>` evaluates whichever checkpoint the user passes; the only held-out file is `ann_file_test` | **T0** for the code path. Which checkpoint and hyperparameters produced the published CircuitNet numbers is not stated: **U** for the published numbers. A user who tunes anything with this code can only tune on the test CSV (no validation file is provided) |
| CFIRSTNET (Liu et al., ICCAD 2024), GitHub main | ICCAD'23 | [code] `train.py`, `src/run.py`, `src/dataset.py`; [PDF] | train = fake + BeGAN_01 + BeGAN_02, valid = `real-circuit-data_20230615` (visible 10), test = `hidden-real-circuit-data`; `train(...)` receives the test loader and every epoch computes test loss / MSE / MAE / RMSE / max error / F1 / recall / precision, prints them and logs them to wandb; best checkpoint saved when **validation MAE** improves (`if valid_result['mae'] < best_MAE`); the function returns the last-epoch model (not reloaded). PDF: "model with lowest validation loss is saved" (code: validation MAE) | **T2** (test exposure every epoch; selection in code on validation). No evidence of T3. Hyperparameter choice (lr 1e-3, wd 0.05, 100 epochs, 384 px, seed 1224) not documented: U |
| WACA-UNet (2507.19197) | ICCAD'23 | [PDF] | checkpoint chosen by best validation F1 on the 10 visible real cases; no code released | **T1** |
| GL-Inception (2406.06541) | ICCAD'23 | [PDF] | 5 visible real cases held out as validation, best-validation checkpoint | **T1** |
| LMM-IR (2511.12581) | ICCAD'23 | [PDF] | all 10 visible real cases used in training (x20 oversampled); no validation set or selection rule stated | **U**. No non-test real case is left for selection, so any selection beyond training loss would have to use the hidden set; not evidence of T3 |
| AttUNet (2408.03292) | ICCAD'23 | [PDF] | pretrain on fake, fine-tune on the 10 visible real; no validation or selection rule stated | **U** (same remark as LMM-IR) |
| GIF (2604.09999) | N14 / N28 IR drop | [PDF] | N14 Table 4: validation Vortex-small, test zero-riscy (design-wise); selection rule not stated | **T1** (split stated), selection rule U |
| CLDRoute (2607.16674), GitHub master | N14 / N28 congestion, DRC | [PDF]; [README]; split CSVs (L003, registered run) | design-wise train / val / test stated in the PDF; README refers to `best_ldm.pt` / `best_gen.pt` without stating the criterion; training code not read | **T1** (val split stated), selection criterion U; whether the split is design-disjoint is checked by L003 |

## Reading
* No recipe read shows **T3** (test-set selection). One shows **T2**: CFIRSTNET computes and logs every test metric on the hidden ICCAD'23 cases at
  every epoch. Its checkpoint rule uses the validation set, so this is exposure (anyone running the script sees hidden-set scores while choosing
  settings), not demonstrated tuning. It is reported as a risk, not a fault, and is not adjudicated further.
* Two ICCAD'23 recipes (LMM-IR, AttUNet) train on all non-test real cases and state no selection rule (**U**). With the hidden cases public in the
  contest repository, the benchmark cannot distinguish their reported numbers from test-selected ones. This is a property of the benchmark
  (public hidden labels, no designated validation split), recorded as an observation.
* CircuitNet ships no validation split for N14 or N28. The official code has no selection step (T0), but any tuning by downstream users of the
  official files has no non-test data to use unless they carve a validation design out of train (GIF and CLDRoute do).
