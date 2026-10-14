# edalint P7: retrain the authors' published CircuitNet models on our N14 split (pre-registration)

**STATUS: DRAFT, NOT FROZEN.** Written 2026-10-14. Nothing here is binding until a freeze commit adds `protocol/FROZEN_p7.json` (section 9).
No P7 model has been trained on anything but the 3-step timing smoke test of section 8; no P7 number on any test row exists.

## 1. Purpose

The registered N14 audit (`protocol/protocol_n14.md`, `REPORT_n14.md`, `results/n14/`) compares baselines with **published** numbers that were produced on a
different or unstated split, so every H1 pair is flagged weak-reference (IR drop: ridge 0.851 against MAVIREC 0.867, mean_map 0.630, registered H1 TRUE;
congestion: ridge NRMS 0.0520 against Circuit GNN 0.020, mean_map 0.0521, registered FALSE; DRC not evaluable, SSIM saturated).
P7 retrains the CircuitNet authors' own networks with the authors' recipe on **our** split, preprocessing, metrics and exclusions, so that H1 can be re-read
against a same-split reference. It is an additional analysis. **It never overwrites or edits a registered N14 verdict or result file.**
It does not make the reference "the published model": it is "the authors' architecture and recipe, reduced training budget (section 3.3), our split".

## 2. Disclosure (what is known before this draft)

* The registered N14 baseline results on the test split (mean_map, ridge, ... per design) are known to the author of this draft. The thresholds below were
  set knowing them (e.g. IR ridge gain 0.2209, congestion best-baseline gain 7.6e-5). No retrained-model value on any test row is known.
* Train-only development: `results/dev/p7_smoke_p7dev.json` (log `.log`): RISCY and RISCY-FPU maps from the registered fit-sample cache (50 IR, 50 congestion rows),
  3 MAVI steps and 3 GPDL steps, no metric computed, no test design opened. The model ports are checked against the authors' code (section 3.2).
* No Kaggle kernel or dataset has been created. Kaggle quota is not readable through the CLI (the web UI only) and was not read.

## 3. Models

### 3.1 Chosen (both from github.com/circuitnet/CircuitNet, BSD-3, commit `41ade1d7e913a418b4495b6b13630b26b0ff7279`, directory `routability_ir_drop_prediction/`)

| task | model | why |
|---|---|---|
| IR drop | **MAVI** (`models/mavi.py`; the CircuitNet 2.0 "MAVIREC" row, ROC-AUC 0.8671, is the registered reference) | the authors' IR-drop model; the only registered H1 pair that came out TRUE, so the one where a same-split reference matters most |
| congestion | **GPDL** (`models/gpdl.py`; CircuitNet 2.0 calls the congestion CNN "FCN", mapping to GPDL assumed from the code layout, not verified) | the authors' congestion model; the registered pair is FALSE with an unreachable (GNN, graph-input) reference, so a model-matched same-split reference is the informative test |

Not retrained: **RouteNet / DRC** (the DRC pair is not evaluable: authors' SSIM saturated at 0.9996 for mean_map; a retrained model cannot change that, and no other DRC primary is registered).
**ICCAD'23**: no trained model or training code is in the contest repo (`E:\edalint_data\iccad23\repo` has map generators and the solver only). The candidate is CFIRSTNET
(github.com/jason122490/CFIRSTNET, Apache-2.0, HEAD 7fd5b4e on 2026-10-14, has `train.py`), but its published number uses ~6,100 synthetic BeGAN training circuits we do not have (we hold 100 fake + 10 real),
384x384 inputs and netlist-derived PDN features beyond the contest maps. A retrain on our 110 cases would not be the published recipe, so it cannot remove the ICCAD'23 weak-reference flag. **ICCAD'23 is out of scope for P7** (open decision D6).

### 3.2 Architecture, recipe and citations (file:line in the pinned commit)

| item | authors' value | where |
|---|---|---|
| MAVI network | 3-D U-Net: `DoubleConv3d` 1->64, 3 `Down` (64->128->256->512, pool (1,2,2)), depth axis averaged before three `Up` (ConvTranspose2d, bilinear=False), `OutConv` 64->4, output = sum over 4 channels of (first 4 depth slices x predicted maps) | `mavi.py` 63-179 (forward 165-179) |
| MAVI init | kaiming_normal (fan_out, relu) on Conv3d, BN weight 1 / bias 0; other layers PyTorch default | `mavi.py` 181-199 |
| GPDL network | encoder (two `conv` blocks 3->32->64 with InstanceNorm + LeakyReLU, 2 max-pools, conv->32 + BN + tanh), decoder (conv, two `upconv`, skip from stage 1, conv->1 + sigmoid) | `gpdl.py` 67-158 |
| GPDL init | N(0, 0.02) on Conv/Linear weights, bias 0 | `gpdl.py` 8-19, 168 |
| IR config | batch 2, lr 2e-4, weight decay 1e-2, L1 loss, in 1 / out 4, hotspot 0.9885, max_iters 200000 | `utils/configs.py` 60-74, 19 |
| congestion config | batch 16, lr 2e-4, weight decay 0, MSE loss, in 3 / out 1, aug `['Flip']`, 200000 iterations | `utils/configs.py` 26-40, 19 |
| loss | `loss_weight` 100 x mean L1 (IR) or mean MSE (congestion) | `utils/losses.py` 62-69, 71-101 |
| optimiser, schedule | AdamW betas (0.9, 0.999); one cosine period from lr to 1e-7 over `max_iters`, set per iteration; checkpoint every 10,000 iterations; **no validation**, the last checkpoint is the one tested | `train.py` 122-125, 26-83, 131, 133-163 |
| data loader | shuffle, drop_last, 16 workers | `datasets/build_dataset.py` 31-38 |
| augmentation | congestion: horizontal flip of feature and label, p = 0.5; IR drop: none | `datasets/augmentation.py` 6-28, `congestion_dataset.py` 32-40, `irdrop_dataset.py` 29-35 |
| eval | `model.eval()` (BatchNorm running statistics), batch 1 | `models/build_model.py` 10-11, `test.py` 52-91 |
| preprocessing | `generate_training_set.py` as in N14 protocol section 4 (authors mode of `edalint.datasets.circuitnet.assemble`) | N14 protocol section 4 |

**Port.** mmcv does not install on current torch, so the networks are ported to plain torch in `src/edalint_p7/models.py` (outside `src/edalint`, whose hash the N14 freeze records).
Verified by `scripts/p7_dev_smoke.py`: with the authors' `mavi.py` / `gpdl.py` imported under mmcv stubs, identical state_dict keys and shapes, identical parameter counts (MAVI 17,063,812 trainable, GPDL 119,169), and
a maximum absolute forward difference of 0.0 after loading the original weights into the port (64x64 inputs). The training script, loss and schedule are re-implemented from `train.py`, not imported.

### 3.3 What is kept and what deviates

Kept: architecture, initialisation, optimiser and betas, learning rate, weight decay, loss and its weight 100, batch sizes, flip augmentation (congestion only), the cosine shape, last-checkpoint rule, no validation, eval mode.
Deviations (each reported in the result files):

1. **Iterations (compute).** MAVI 30,000 (15 % of 200,000; 60,000 samples, ~8.8 epochs of the full train), GPDL 50,000 (25 %; 800,000 samples, ~118 epochs). The cosine horizon is the reduced count (the schedule is compressed, not truncated). Open decision D2.
2. **Mixed precision** for MAVI on the T4 (torch autocast fp16, GradScaler), if the T4 benchmark shows fp32 is too slow (section 7). GPDL fp32. Open decision D3.
3. **Training set.** Arm A (primary): every official train row with files for the task (~6,790 rows listed, less loader-skipped rows, counted). The authors train on all of it. The registered baselines were fitted on the 979-row seeded fit sample, i.e. on less data; this favours the model, so a ridge success under H-B is conservative.
   Arm B (optional, secondary): the 979 registered `fit_ids`, for a matched-data comparison. Open decision D4.
4. `num_workers` and data-loading details differ (inputs are preloaded shards).
5. Seeds: torch/numpy/Python seeded; cuDNN non-determinism is not removed, so a rerun on the same seed is not bit-identical.

## 4. Data, split, metrics (identical to the N14 protocol)

* Official split files, exclusions, authors-space 256 x 256 maps, `assemble(..., 'authors')`: N14 protocol sections 3 and 4. nvdla-large and rows absent from both CSVs are excluded.
* Test scoring uses **all** official test rows (3,544 for both tasks), uncapped, design-macro primary as in N14 section 3. The set of scored ids must equal `eval_ids` in `results/n14/models_<task>.json`; the scorer aborts on any difference.
* Metrics: `edalint.metrics.ssim_official / nrms_official` and the official ROC machinery (`PrimaryAcc`, `RocAcc` in `edalint.p4_n14`, imported, not copied), IR primary = official ROC area at 0.9885, congestion primary = official NRMS.
  Predictions are float32 model outputs, passed through `np.nan_to_num` exactly as for the baselines (`_score_rows`).
* **Hyperparameter selection on train only: none.** The published hyperparameters are kept; the only free choices (iteration budget, seeds, precision) are fixed at the freeze from the train-only benchmark of section 7, never from a test score.
  Optional train-only development D1: hold out Vortex-small and nvdla-small, train at the registered budget, check that the held-out ROC/NRMS is finite and the loss fell. It informs nothing but "the run is not broken".
* Seeds: **3** per task and arm, `20261101, 20261102, 20261103` (fixed now). Verdict value = mean over seeds of the design-macro metric; per-seed values reported.
* Compact export for Kaggle (sizes measured on 10 train rows, section 7): per-row `.npz` with `x` (24 or 3 channels x 256 x 256) and `y` (256 x 256) float32, compressed, the exact `assemble` output, sha256 per shard recorded in a manifest. Test shards are created only after the P7 freeze.

## 5. Hypotheses and verdict rules (mechanical; all per task t in {ir_drop, congestion})

Notation (design-macro, primary metric, test split): `M_t` = registered mean_map value (`results/n14/models_<t>.json`; IR 0.6301, congestion 0.05207), `R_t` = mean over the 3 seeds of the retrained model,
`Gain(v)` = v - M_t (IR, higher is better) or M_t - v (congestion), `G_R = Gain(R_t)`, `G_b` = largest Gain among the six registered baselines (IR ridge 0.2209; congestion ridge 7.6e-5; read from `results/n14/`, never recomputed with other settings).

**H-A, reproduction gate ("the retrained model is skilled").** Skill share over mean_map `S = G_R / headroom` (IR headroom 1 - M_t = 0.3699; congestion `S = 1 - R_t / M_t`).
Gate passes if `G_R >= 0.5 x G_ref`, where `G_ref` is the model-matched published gain: IR MAVIREC 0.8671 (G_ref 0.2370, so R_t >= 0.7486); congestion FCN/GPDL N14 NRMS 0.036 (G_ref 0.01607, so R_t <= 0.0440).
A failed gate means "retraining not reproduced at this budget"; the weak-reference flag of that task **stays**, H-B and H-C for it are reported as descriptive only, and no verdict is drawn. (This blocks a "baselines match" TRUE obtained from an under-trained network.) The 0.5 factor is open decision D5.

**H-B, the re-read of H1 ("ridge-class baselines reach >= 0.8 of the same-split model's gain").** Only if the gate passes: TRUE if `G_b >= 0.8 x G_R` (0.8 is the registered N14 line), else FALSE.
Overall: TRUE if both tasks are TRUE, FALSE if neither, MIXED otherwise (DRC is not evaluable, so the registered "2 of 3" rule cannot be applied). The output table puts next to each task the registered verdict (unchanged), the registered ratio, and whether the P7 verdict agrees.
Sensitivities (reported, cannot change the verdict): best seed and worst seed in place of the mean (flag "seed-sensitive" if the verdict flips); IR anchored ROC area; sample-pooled instead of design-macro mean; per-design values and the sign count of designs on which the retrained model beats mean_map.

**H-C, the published number on our split (descriptive).** `D = R_t - published` (IR vs 0.8671, congestion vs 0.036). Reading fixed now: |D| <= 0.03 AUC / 0.01 NRMS means the published number is "consistent with a same-split model"; otherwise the signed D is reported as the split/preprocessing effect.
If the gate passes and |D| is within the band, the report may describe the registered weak-reference flag for that task as "resolved by a same-split retrain at reduced budget (15 % / 25 % of authors' iterations)". It may not drop the qualifier.

## 6. Run-once rules, outputs, freeze guard

* Outputs (all via `edalint.runner.write_once(..., dataset='p7')`, exclusive creation, never overwritten): `results/p7/train_<task>_<arm>_s<seed>.json` (train-only: budget, loss every 100 iterations, hardware, wall time, checkpoint sha256),
  `results/p7/models_<task>_<arm>_s<seed>.json` (same row format as the N14 `models_*.json`, plus `official_roc` per design), `results/p7/verdicts_p7.json`, `results/p7/tables_p7.md`. Kaggle writes the `train_*` files and checkpoints; scoring is local.
* Training is part of the registered run (after the freeze), so the training code is covered by the code hash. Test inputs are exported and scored only after the freeze. A crash before a file is written may be re-launched once and is logged; a non-finite training loss is a result (seed reported with undefined metric; a task with fewer than 2 finite seeds has no verdict).
* Anything not specified here is POST HOC, labelled, written to `results/posthoc/`, and cannot change a section 5 verdict. No criterion, threshold, budget, seed or hyperparameter changes after the freeze. At most 2 local CPU jobs. Stop processes by PID only.
* **Freeze file** `protocol/FROZEN_p7.json`: `{"dataset": "p7", "protocol_file", "protocol_sha256" (CRLF normalised), "code_hash" (the field name `require_frozen` reads; must equal today's `edalint.protocol.code_hash()`, d3f6d2f6..., and be checked at run time), "p7_code_hash" (sha256 over `src/edalint_p7/**/*.py` and `scripts/p7_run_*.py`, dev scripts excluded as in section 11 D7, same recipe as `code_hash()`; implemented in `src/edalint_p7/guard.py`), "frozen_at", "frozen_commit", "budget": {...}, "seeds": [...]}`.
* **Does `protocol_key` need a new key? No, and it should not be edited.** `protocol_key('p7')` already returns `'p7'`, so `frozen_info(None, 'p7')` / `write_once(dataset='p7')` read `FROZEN_p7.json` with no change. Two gaps, handled outside `src/edalint`:
  (i) `code_hash()` covers `src/edalint` only, so a P7 guard in `src/edalint_p7` must check `p7_code_hash` itself; (ii) test rows are read through `N14Dataset`, whose guard keys on `circuitnet_n14_*` -> `FROZEN_n14.json`, which is **already satisfied**, so the N14 guard does not stop a pre-P7-freeze test read.
  The P7 runner must therefore call `require_frozen(dataset='p7')` plus its own hash check before touching any test shard. Editing `protocol_key` or anything else under `src/edalint` would change the hash recorded in `FROZEN_n14.json` and in every `results/n14` file and make the N14 stages refuse to run (as e541efe did for ICCAD'23); do not do that.
  The stale test `test_real_repo_iccad23_freeze_is_recognised_and_n14_is_not` (asserted N14 unfrozen) was replaced by the reviewer with `test_real_repo_freezes_are_recognised_per_dataset` (`tests/` is outside `code_hash()`, which hashes `src/edalint/**/*.py` only).

## 7. Compute and feasibility (estimates; the T4 figures are unmeasured)

Measured here (i7-1355U, 6 threads, torch 2.10 CPU, `results/dev/p7_smoke_p7dev.json`): MAVI train 63-77 s per batch-2 step (~35 s/sample), eval 9.4-12.6 s/sample; GPDL train 3.0-3.7 s per batch-16 step (0.195 s/sample), eval 0.069 s/sample.
MAVI forward is ~5.3e11 MACs (1.06 TFLOP) per sample by layer count, ~3.2 TFLOP per training sample; GPDL ~3.8 GFLOP forward, ~11 GFLOP training.

**Local CPU.** MAVI: 30,000 iterations = 60,000 samples x 35 s = ~24 days per seed; even 1,000 iterations is ~19 h. Test inference alone is 3,544 x ~10 s = ~10 h per model-seed. **Not viable.**
GPDL: 10,000 iterations (5 % budget) = ~8.6 h per seed, two seeds in parallel feasible, test inference ~4 min. Viable as a fallback at a budget 5x below the T4 plan (would be a further deviation and weaker evidence).

**Kaggle T4** (~30 h/week, quota unknown). Estimated, not measured: MAVI fp32 ~0.8-1.3 s/sample (effective 2.4-4 TFLOPS of 8.1 peak), fp16 autocast ~0.2-0.5 s/sample; GPDL fp32 ~5-15 ms/sample (memory bound).

| run | samples | per seed | 3 seeds |
|---|---|---|---|
| MAVI 30k it, fp16 | 60,000 | 3.3-8.3 h (mid 5 h) | 10-25 h (mid 15 h) |
| MAVI 30k it, fp32 | 60,000 | 13-22 h | 40-65 h (more than a week of quota) |
| MAVI full 200k it (authors), fp16 | 400,000 | 22-56 h | not affordable |
| GPDL 50k it, fp32 | 800,000 | 1.1-3.3 h | 3-10 h |
| MAVI test inference, 3,544 rows, fp16 | | ~5-10 min | ~15-30 min |

Plan total about 18-25 h of the weekly quota for arm A. Arm B (979 rows, same budget) would add the same again and does not fit one week. A 5-minute T4 benchmark (train-only) is needed before the freeze to fix the budget; rule: if measured MAVI fp16 exceeds 0.6 s/sample, halve the MAVI iterations (15,000); if fp32-only fits, report fp32. Kaggle sessions cap at ~12 h per run, each seed is its own run.

**Compact data.** Measured on 10 train IR rows (`assemble` output): float32 6.55 MB per row raw (IR 24 channels + label), float16 3.28 MB, compressed `.npz` **1.9 MB (float32)** or **0.93 MB (float16)**; float16 rounding error on the [0,1] inputs is at most 2.4e-4.
Sizes (compressed float32 / float16): IR full train 6,790 rows ~12.9 / 6.3 GB; IR test 3,544 rows ~6.7 / 3.3 GB; IR fit sample (979) ~1.9 / 0.9 GB. Congestion (3 channels + label, ~0.14 MB per row compressed float16, ~0.27 MB float32): train ~1.8 GB (float32), test ~1.0 GB. Uncompressed float32 would be 44 GB (IR train) so the raw 51 GB feature size is avoided.
Recommend compressed float32 (no rounding deviation). Kaggle dataset size limits (100 GB total is my recollection, a smaller per-dataset cap is possible) are unverified; shard by design if needed.
**Local staging of the full train** (not yet cached: 5,811 more IR rows and 6,0xx more routability rows beyond the 979-row fit cache): estimated from the N14 dry-run scan times (IR RISCY 828 s, RISCY-FPU 975 s; routability 1,930 s and 2,058 s) at ~0.5-1 h (IR) and ~1-2 h (congestion) of one CPU job, adding ~40 GB (IR) and ~10 GB (congestion) of float32 member cache. **E: has 85 GB free now**, so staging both tasks is tight; stage one task at a time and clear the cache, or write only the compact export.

## 8. Dev smoke test (train-only, done)

`scripts/p7_dev_smoke.py` -> `results/dev/p7_smoke_p7dev.json`: port equivalence (section 3.2), 3 MAVI steps (losses 45.7, 36.7, 39.9, not a convergence signal), 3 GPDL steps (15.5, 15.1, 13.6), timings above, export sizes, zero non-finite values in the 50 IR rows read. Only RISCY and RISCY-FPU rows were opened.

## 9. Open decisions for the reviewer (before any freeze)

* D1 run the optional train-only held-out-design sanity run (Vortex-small + nvdla-small)?
* D2 iteration budgets (MAVI 30k, GPDL 50k) and the rule tying them to the T4 benchmark; or accept a single seed of a larger budget instead of 3 seeds.
* D3 mixed precision for MAVI (deviation from the authors' fp32) versus fewer iterations.
* D4 arm B (matched 979 rows) in or out; arm A trains on ~6,790 rows while the registered baselines saw 979.
* D5 gate factor 0.5 (G_R >= 0.5 x G_ref) and the model-matched congestion reference (FCN 0.036, mapping to GPDL unverified) versus the registered GNN 0.020.
* D6 ICCAD'23 (CFIRSTNET) in or out; DRC / RouteNet in or out.
* D7 whether to accept that the N14 freeze guard cannot enforce the P7 freeze (section 6) or to pay the price of a `src/edalint` edit.
* D8 Kaggle logistics once approved: private datasets and kernels named `edalint-p7-*`; train-only dataset first, test shards only after the freeze; quota to be checked in the web UI.

## 10. Deviation log

None (draft).

## 11. Reviewer decisions (independent review, 2026-10-14; this protocol stays NOT FROZEN)

Reviewer did not write the draft. Checked: the draft against the authors' pinned `train.py`, `utils/configs.py`, `utils/losses.py`, `datasets/irdrop_dataset.py` (fetched at commit 41ade1d), the local copies of `models/mavi.py` and `models/gpdl.py` (`E:\edalint_data\cn_models_src`), `src/edalint/protocol.py`, `results/dev/p7_smoke_p7dev.json`. No test row, no Kaggle, no training was touched.

**Prerequisites that do not yet exist (block the freeze, not the review).** `src/edalint_p7` holds only `models.py`. The training loop, sampler, schedule, exporter, scorer, P7 guard and `p7_code_hash` are not written, and section 6 says the training code is covered by the hash. Nothing can be frozen until they exist and the tests in D7 pass.

**Blocking fix to the draft (applied as a reading rule here).** `require_frozen` demands a field named `code_hash`; the draft's freeze record used `edalint_code_hash` only, so `require_frozen(dataset='p7')` would refuse with "no code_hash field". The freeze record must carry `"code_hash"` (= `edalint.protocol.code_hash()`, d3f6d2f6...) **and** `"p7_code_hash"`. Section 6 is read with that change.

* **D1 yes, GPDL only, on CPU, train-only.** Reason: the draft's check shows only that 3 steps run; a wrong label scale, a sigmoid-range mismatch or a broken flip would stay invisible until the Kaggle quota is spent. Spec fixed now: hold out two *train* designs (Vortex-small and nvdla-small if the cached train rows contain them, else any two train designs), GPDL at 2,000 iterations (cosine compressed to 2,000), fp32, one seed, one CPU job (about 2 h). Pass = loss finite and falling by at least 30 % from the first 100-iteration block to the last, and held-out official NRMS below the train-only mean_map on the same held-out rows. Failing it blocks the freeze; passing it changes nothing in section 5. MAVI: not on CPU; run the same check inside the T4 benchmark (D8 step 3) with 500 iterations on train-only shards.
* **D2 keep 3 seeds; fix the ladder.** A gate at 0.5 tolerates seed noise, whereas a failed gate yields no verdict, so iterations are the scarcer resource. Replace the draft's "halve the MAVI iterations" rule by this ladder, decided from the train-only T4 benchmark only: (1) MAVI 30k x 3 seeds; (2) if the projected MAVI total exceeds 22 h, 30k x 2 seeds; (3) never below 30k iterations and never below 2 seeds (section 6 already gives no verdict with fewer than 2 finite seeds). One larger-budget seed is rejected: no seed variance, and no second try. GPDL stays 50k x 3. The trainer must replicate the authors' sampling: they restart the data iterator every 100 iterations (`break` at `print_freq` in `train.py`), so each 100-iteration block is a fresh shuffle without replacement; draw blocks that way, not from one continuous epoch stream.
* **D3 fp16 autocast for MAVI is preferred to fewer iterations.** Halving iterations changes the optimisation budget; autocast changes only arithmetic. Conditions: autocast plus GradScaler, BatchNorm and loss in fp32 (autocast default), skipped-step count logged, `deviation: fp16` recorded in every `train_*.json`. The T4 benchmark also compares fp16 and fp32 forward on 20 train rows (max absolute difference reported). Test inference runs in fp32 unless fp32 projects above 4 h in total; if fp16 inference is used it is recorded as a deviation. T4 has no bf16.
* **D4 MAVI arm B out. GPDL arm B in as a secondary, one seed (20261101), run last and only if quota remains.** Arm A (about 6,790 rows) against baselines fitted on 979 favours the model, so an H-B TRUE is conservative; an H-B FALSE is where the data-volume confound bites. GPDL arm B costs one GPDL seed (about 1-3 h on a T4, estimated) and cannot change a verdict. MAVI arm B is unaffordable. Refitting the baselines on the full train would change a registered baseline: POST HOC only.
* **D5 keep the 0.5 factor; strengthen the caveat; reference = FCN 0.036 with the mapping stated as unverified.** (a) FCN = GPDL: supported by the CircuitNet experiment tutorial ("The network of ... GPDL ... uses an FCN based encoder-decoder architecture", `E:\edalint_data\docs_cache\experiment_tutorial.txt` line 35), by the 2022 letter's "congestion FCN" and by CLDRoute calling GPDL its SOTA baseline; **not verified**: CircuitNet 2.0 Appendix B (the source of 0.036) says "FCN" and never names GPDL. The same method on N28 is 0.040 (letter), 0.047 (Appendix B) and 0.0337 (CLDRoute): about +-20 % unexplained spread, and "NRMSE" as printed is not shown to equal the official NRMS. Consequence: G_ref = 0.01607 is soft; the gate (R_t <= 0.0440) is a coarse skill check, not a reproduction of 0.036. (b) GNN 0.020 is a graph model and not model-matched; do not use it for the gate; report the GNN-based gate (R_t <= 0.0361) as a sensitivity only. (c) Threshold bias: the thresholds were set knowing the registered baseline results, and the outcome is largely predictable: IR H-B is TRUE for any R_t up to about 0.906 AUC (ridge 0.851, published 0.867); congestion H-B is FALSE whenever the gate passes (ridge gain 7.6e-5 against a model gain of at least 0.008). P7 is therefore a confirmation of an expected outcome, not a blind test; what it adds is whether the gate passes at the reduced budget and where R_t falls relative to 0.906 and the published numbers. The report must say this in its first P7 sentence. (d) Add one registered sensitivity: H-B recomputed with G_R replaced by max(G_R, G_ref) (published credit); a flip against the mean-seed verdict is flagged "budget-sensitive" (cannot change the verdict). (e) "Resolved" wording needs the gate **and** |D| within the band; a gate pass with |D| above the band is reported as "gate passed, retrain below published".
* **D6 out for both.** ICCAD'23/CFIRSTNET: its published number rests on about 6,100 BeGAN circuits plus netlist-derived features we do not have, so a retrain on 110 cases is a different recipe and cannot remove the flag. DRC/RouteNet: the DRC pair is not evaluable (SSIM saturated for mean_map); a retrain adds compute and no verdict.
* **D7 sound only with the following; enforcement is by convention, as for N14.** `require_frozen(dataset='p7')` checks the right thing and cannot fall through to another freeze (`frozen_file` falls back to legacy `FROZEN.json`, whose `dataset` is `iccad23`, so `frozen_info` returns None for p7). What it cannot do is stop code that reads test rows without calling it, because the N14 guard is already satisfied. So: (i) all test-row access in P7 goes through one function in `src/edalint_p7` (`open_test_rows`), which calls `require_frozen(dataset='p7')`, recomputes `p7_code_hash` over `src/edalint_p7/**/*.py` plus `scripts/p7_run_*.py` (dev scripts excluded so a later edit of `scripts/p7_dev_smoke.py` cannot void the freeze; CRLF normalised, sorted posix paths, same recipe as `code_hash()`) and compares it to `FROZEN_p7.json`; (ii) the Kaggle kernel carries `FROZEN_p7.json` and the code, recomputes both hashes at start and aborts on a mismatch; (iii) the test-shard exporter uses the same function. Tests required before the freeze (`tests/test_p7_*.py`, none touching real test rows): hash recipe (changes on a one-byte edit, equal under CRLF, ignores `__pycache__`); guard refusal for no freeze file, wrong `dataset`, missing `code_hash`, edalint hash mismatch, p7 hash mismatch, and acceptance when all match (tmp protocol dir); `open_test_rows` refuses before the freeze; a static test that nothing under `src/edalint_p7` or `scripts/p7_run_*.py` builds `N14Dataset` on the test split or reads the test CSV outside `open_test_rows`; train id list disjoint from `eval_ids` by id and by design, its count equal to the loader-skip-corrected official train count; cosine schedule equal to the authors' `CosineRestartLr` at iterations 0, 1, N/2, N-1; losses equal 100 x mean L1/MSE; block sampler yields a fresh shuffle per 100 iterations; flip applies to feature and label together and for congestion only; `assemble('authors')` IR puts power_i, power_s, power_sca, power_all in depth slices 0-3 (MAVI multiplies those four by its predicted maps) with layout (1, 24, 256, 256) as in `irdrop_dataset.py`; the scorer aborts when the scored id set differs from `eval_ids`; the scorer, fed a registered baseline's predictions on a train-only subset, reproduces `_score_rows` to 1e-9 (ties P7 metrics to the frozen N14 metrics without touching test); export round-trip bit-exact against `assemble` with sha256 manifest; an fp16 train step on a fixed batch has no NaN; `write_once` exclusivity for `dataset='p7'`; port equivalence against the authors' modules as a skippable test (currently only in the dev script).
* **D8 Kaggle plan (nothing done; user go-ahead required).** Private datasets and kernels only, named `edalint-p7-*`; no existing kernel or dataset touched. Steps: (1) user checks the weekly GPU quota in the web UI (not readable by CLI) and confirms the other planned Kaggle runs leave at least 25 h; (2) after the pre-freeze tests pass, upload **train-only** shards (IR about 12.9 GB float32, congestion about 1.8 GB, plus manifest; labels float32), no test shards and no `FROZEN_p7.json`; (3) a train-only T4 benchmark kernel (about 5 minutes plus the 500-iteration MAVI sanity run) fixes the D2/D3 ladder; (4) freeze commit; (5) upload code bundle plus `FROZEN_p7.json` as a private dataset, one kernel per seed (12 h cap), each writing `train_*.json` and a checkpoint with sha256; (6) only now export and upload test shards (IR about 6.7 GB, congestion about 1.0 GB) as a separate private dataset and run scoring inference; (7) download results and `write_once` into `results/p7/`. Kaggle per-dataset size limits are unverified: shard by design. Total 18-25 h is an unmeasured estimate.

**Port fidelity (reviewer check).** Compared line by line with the local copy of the authors' `mavi.py` and `gpdl.py`: layer order, widths, padding, bias flags, normalisation, activations, skip connection, the `x_in` multiply-and-sum in MAVI and the init rules are the same; the only difference (no checkpoint-loading helper) is harmless. The smoke JSON supports "identical state_dict keys and shapes, identical counts, forward difference 0.0 at 64 x 64 with the original weights". Wording fix: the JSON counts (17,069,444 and 119,233) include BatchNorm buffers; trainable counts are 17,063,812 and 119,169 (MAVI difference 5,632 = 2 x 2,816 BN channels). Not covered by the check: initialisation statistics (same code path, no statistical test), training-mode behaviour, and the sampler, schedule and optimiser loop, which are re-implemented; the tests above cover them. Recipe constants in section 3.2 (batch, lr, weight decay, loss weight 100, 200k iterations, AdamW betas, cosine to 1e-7, checkpoint every 10k, no validation) match the fetched pinned files.
