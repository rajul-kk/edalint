# edalint P7: retrain the authors' published CircuitNet models on our N14 split (pre-registration)

**STATUS: DRAFT, NOT FROZEN.** Written 2026-10-14. Nothing here is binding until a freeze commit adds `protocol/FROZEN_p7.json` (section 9).
No P7 model has been trained on anything but the 3-step timing smoke test of section 8, except the train-only dev runs of section 12; no P7 number on any test row exists. **Amended 2026-10-15 by section 12 (draft amendment): the congestion arm is dropped, P7 is IR-only.**

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
* No Kaggle kernel or dataset had been created when this section was written (2026-10-14); since then the train-only dev kernels `edalint-p7-*-bench` exist (section 12). Kaggle quota is not readable through the CLI (the web UI only) and was not read.

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
* **Freeze file** `protocol/FROZEN_p7.json`: `{"dataset": "p7", "protocol_file", "protocol_sha256" (CRLF normalised), "code_hash" (the field name `require_frozen` reads; must equal today's `edalint.protocol.code_hash()`, d3f6d2f6..., and be checked at run time), "p7_code_hash" (sha256 over `src/edalint_p7/**/*.py` and `scripts/p7_run_*.py`, dev scripts excluded as in section 11 D7, same recipe as `code_hash()`; implemented in `src/edalint_p7/guard.py`), "frozen_at", "frozen_commit", "budget": {"model": "MAVI", "task": "ir_drop", "arm": "A", "iterations": 30000, "precision": "fp16", "sampler_mode": "epoch", "batch_size": 2}, "seeds": [20261101, 20261102, 20261103]}`.
* **Does `protocol_key` need a new key? No, and it should not be edited.** `protocol_key('p7')` already returns `'p7'`, so `frozen_info(None, 'p7')` / `write_once(dataset='p7')` read `FROZEN_p7.json` with no change. Two gaps, handled outside `src/edalint`:
  (i) `code_hash()` covers `src/edalint` only, so a P7 guard in `src/edalint_p7` must check `p7_code_hash` itself; (ii) test rows are read through `N14Dataset`, whose guard keys on `circuitnet_n14_*` -> `FROZEN_n14.json`, which is **already satisfied**, so the N14 guard does not stop a pre-P7-freeze test read.
  The P7 runner must therefore call `require_frozen(dataset='p7')` plus its own hash check before touching any test shard. Editing `protocol_key` or anything else under `src/edalint` would change the hash recorded in `FROZEN_n14.json` and in every `results/n14` file and make the N14 stages refuse to run (as e541efe did for ICCAD'23); do not do that.
  The stale test `test_real_repo_iccad23_freeze_is_recognised_and_n14_is_not` (asserted N14 unfrozen) was replaced by the reviewer with `test_real_repo_freezes_are_recognised_per_dataset` (`tests/` is outside `code_hash()`, which hashes `src/edalint/**/*.py` only).

## 7. Compute and feasibility (estimates; the T4 figures were measured later and are superseded by sections 12.4 and 12.5)

Measured here (i7-1355U, 6 threads, torch 2.10 CPU, `results/dev/p7_smoke_p7dev.json`): MAVI train 63-77 s per batch-2 step (~35 s/sample), eval 9.4-12.6 s/sample; GPDL train 3.0-3.7 s per batch-16 step (0.195 s/sample), eval 0.069 s/sample.
MAVI forward is ~5.3e11 MACs (1.06 TFLOP) per sample by layer count, ~3.2 TFLOP per training sample; GPDL ~3.8 GFLOP forward, ~11 GFLOP training.

**Local CPU.** MAVI: 30,000 iterations = 60,000 samples x 35 s = ~24 days per seed; even 1,000 iterations is ~19 h. Test inference alone is 3,544 x ~10 s = ~10 h per model-seed. **Not viable.**
GPDL: 10,000 iterations (5 % budget) = ~8.6 h per seed, two seeds in parallel feasible, test inference ~4 min. Viable as a fallback at a budget 5x below the T4 plan (would be a further deviation and weaker evidence).

**Kaggle T4** (~30 h/week, quota unknown). Estimated before measurement (SUPERSEDED by 12.4: measured MAVI fp32 2.343 s per batch-2 step, fp16 0.587 s per step, 0.613 s per iteration end to end): MAVI fp32 ~0.8-1.3 s/sample (effective 2.4-4 TFLOPS of 8.1 peak), fp16 autocast ~0.2-0.5 s/sample; GPDL fp32 ~5-15 ms/sample (memory bound).

| run | samples | per seed | 3 seeds |
|---|---|---|---|
| MAVI 30k it, fp16 | 60,000 | 3.3-8.3 h (mid 5 h) | 10-25 h (mid 15 h) |
| MAVI 30k it, fp32 | 60,000 | 13-22 h | 40-65 h (more than a week of quota) |
| MAVI full 200k it (authors), fp16 | 400,000 | 22-56 h | not affordable |
| GPDL 50k it, fp32 | 800,000 | 1.1-3.3 h | 3-10 h |
| MAVI test inference, 3,544 rows, fp16 | | ~5-10 min | ~15-30 min |

Plan total about 18-25 h of the weekly quota for arm A. **SUPERSEDED by section 12.5**: the measured projection is 5.11 h per seed and 15.3 h for 3 seeds (MAVI 30,000 iterations, fp16), ladder step (1) holds, the 0.6 s/sample rule below is not used (the benchmark is per iteration, section 12.5), and the budget is 30,000 iterations, not 15,000. Arm B is out (D4 / 12.2). The original text, kept for the record: Arm B (979 rows, same budget) would add the same again and does not fit one week. A 5-minute T4 benchmark (train-only) is needed before the freeze to fix the budget; rule: if measured MAVI fp16 exceeds 0.6 s/sample, halve the MAVI iterations (15,000); if fp32-only fits, report fp32. Kaggle sessions cap at ~12 h per run, each seed is its own run.

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

Filled before the freeze (all recorded in section 12; none uses a test value):

1. Congestion (GPDL) arm dropped, P7 is IR-only: **post hoc**, after the D1 / D1' failures (12.1, 12.2).
2. Epoch sampler instead of the D2 block100 sampler: post-D2, decided before the freeze (12.9).
3. MAVI trained with fp16 autocast + GradScaler (deviation from the authors' fp32); scoring in fp32 (12.4, 12.5).
4. Section 12 amendment itself (A1) and the pre-freeze review fixes to it.
5. Aggregator clarifications (12.10): exact gate constants, `G_b` definition, undefined / diverged seeds, sensitivities, H-C edge, NRMS rule, file checks.
6. Reduced training budget (30,000 of 200,000 iterations), unchanged from section 3.3 item 1.

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

## 12. Amendment A1 (draft, 2026-10-15; this protocol stays NOT FROZEN)

Appended after the train-only development runs of sections 8 and 11. Sections 1-11 are left as written; where this section conflicts with them, this section governs once the freeze commit exists. No test row, no test shard and no `FROZEN_p7.json` exist or were touched. The dev runs below read only train rows.

### 12.1 Development record: GPDL (congestion) held-out check, D1 / D1' (section 11, D1)

Pass rule (section 11, D1, fixed before the runs): loss finite and falling by at least 30 %, **and** held-out official NRMS below the train-only mean_map on the same held-out rows (the same 179 held-out rows of Vortex-small + nvdla-small in every run; lower NRMS is better). D1' (10,000 iterations) was named in the subject of f2f462e; that commit touched no protocol file. Sections 1-11 say only that failing D1 "blocks the freeze". The consequence "if D1' fails, the congestion arm is dropped and P7 becomes IR-only" was set in the task prompt that ordered the D1' run, **not** in sections 1-11. The drop in 12.2 is therefore a **post-hoc decision**, disclosed as such (as 12.3 does for the MAVI criteria).

| run | where | train rows | iterations | sampler | held-out NRMS GPDL | held-out NRMS mean_map | loss drop | pass | commit |
|---|---|---|---|---|---|---|---|---|---|
| D1 | local CPU | 800 (400 RISCY + 400 RISCY-FPU) | 2,000 | block100 | 0.03583 | 0.03408 | 98.5 % | NO | 45bd4e1 |
| Kaggle T4 | Kaggle | 6,611 (RISCY 3,162 + RISCY-FPU 3,449) | 2,000 | epoch | 0.03564 | 0.03384 | 98.6 % | NO | 773ecba |
| Kaggle T4 | Kaggle | 6,611 (RISCY 3,162 + RISCY-FPU 3,449) | 10,000 | epoch | 0.03706 | 0.03384 | 98.8 % | NO | 773ecba |
| D1' | local CPU | 800 (400 RISCY + 400 RISCY-FPU) | 10,000 | epoch | 0.03722 | 0.03408 | 99.0 % | NO | 14f23d6 |

The loss criterion is met in every run; the held-out criterion fails in every run, and training 5x longer makes held-out NRMS worse (local: 0.03583 -> 0.03722), mostly on nvdla-small (D1' 0.04408 for GPDL; Kaggle 10k 0.04590 against 0.03961 for mean_map). The failure therefore holds at both training-set sizes (800 and 6,611 rows). The local and Kaggle mean_map values differ (0.03408 / 0.03384) because the mean_map is fit on different rows (`scripts/p7_dev_d1.py:60`: the 800 fit rows; `scripts/p7_dev_kaggle_common.py:39-52,183`: all 6,611 train rows); the 179 held-out rows are the same. Confound: local D1 used the block100 sampler, the other three runs the epoch sampler. Train-only references: `results/dev/p7_d1b_heldout_refs_p7dev.json`, `results/dev/p7_d1c_train_loss_refs_p7dev.json`. This is a statement about the development setting (training on RISCY and RISCY-FPU only, two unseen designs), not a verdict on GPDL.

### 12.2 Decision: the congestion arm is dropped; P7 is IR-only

Following the task-prompt rule described in 12.1 (post hoc with respect to the protocol text), the congestion (GPDL) arm is **removed from P7**. Consequences:

* Void: section 3.1 and 3.2 GPDL rows, section 3.3 item 1 (GPDL 50,000 iterations), the congestion parts of section 5 (H-A, H-B, H-C, overall rule), the GPDL rows of section 7, the GPDL parts of D2, D4, D5(b-e), and in particular the D5(c) sentence that congestion H-B is FALSE whenever the gate passes (void, not a prediction). Section 4 "congestion primary = official NRMS" and the "per task and arm" output naming of section 6 are void; outputs are IR only. Section 3.3 item 2 (conditional mixed precision) is replaced by 12.4. Also void, by name: the section 5 congestion gate text ("congestion FCN/GPDL N14 NRMS 0.036 (G_ref 0.01607, so R_t <= 0.0440)") and the section 5 "Overall" rule (TRUE if both tasks are TRUE, FALSE if neither, MIXED otherwise); the IR gate (R_t >= 0.7486) and the IR H-B rule stand.
* All remaining "both tasks", "per task" and "task t in {ir_drop, congestion}" wording in sections 1-11 is read as IR drop only. No GPDL model is trained on the full train and no GPDL number on any test row is produced.
* H-B overall rule: the P7 verdict is the IR drop verdict alone (TRUE, FALSE, or no verdict if the H-A gate fails). The registered N14 congestion verdict (ridge NRMS 0.0520 vs Circuit GNN 0.020, registered FALSE) and its weak-reference flag are **unchanged**. The P7 report must say that congestion was attempted at development scale only (GPDL, 12.1) and failed the held-out check, and makes no other statement about congestion.
* The GPDL port in `src/edalint_p7/models.py` and its dev scripts stay in the repository as development material.
* Unchanged: IR arm A only, seeds 20261101, 20261102, 20261103.

### 12.3 MAVI (IR drop) recorded dev check: 500 iterations, fp16

Run: Kaggle T4, train-only, MAVI, fp16 AMP, 500 iterations (cosine compressed to 500), seed 20261101, train fold 6,611 rows (RISCY, RISCY-FPU), held-out 179 rows (Vortex-small, nvdla-small). Files: `results/dev/p7dev_kaggle_mavi/`. Run commit: **a50017e**.

**Provenance of the criteria.** The four criteria were set in the instruction given to the implementing agent before the run. They are **not** in sections 1-11 (section 11 D1 requires a MAVI check inside the T4 benchmark but gives no criteria; there is no section 11b). Criteria (a), (c) and (d) follow the D1 pattern (finite loss, loss below reference, held-out better than train-only mean_map); (b) has no counterpart in D1 (D1 has no GradScaler criterion) and comes from the fp16 deviation of D3. All four were fixed before the outcome was known. They are recorded here, after the run, as the dev check; the check only says "the run is not broken" and informs nothing else. **Decision: the four criteria stay a development record only; they are not a standing rule for the registered runs.** The training loss had not plateaued at 500 iterations (100-iteration blocks 28.65, 23.76, 23.04, 22.55, 22.57; the flat tail is the end of a schedule compressed to 500 iterations), so the numbers say nothing about the 30,000-iteration level.

| # | criterion | measured | met |
|---|---|---|---|
| a | all training losses finite | 0 non-finite losses | yes |
| b | skipped GradScaler steps <= 5 % | 4 of 500 (0.8 %) | yes |
| c | last 100-iteration block train loss < loss of the constant train mean_map | 22.57 < 28.43 (first block 28.65, drop 21.2 %) | yes |
| d | held-out official ROC (0.9885) > train-only mean_map on the same rows | 0.7678 > 0.5077 (anchored 0.8184 vs 0.5077) | yes |

Per design held-out ROC: Vortex-small 0.7134 (mean_map 0.5296), nvdla-small 0.8095 (mean_map 0.4816). All four met. This is not evidence about the 30,000-iteration result. The 0.7678 is not compared with the H-A gate and the gate is not tuned toward it (12.8).

### 12.4 Precision: fp16 (AMP) for MAVI training, fp32 for scoring (replaces the conditional in 3.3 item 2 and D3)

Measured on the T4 (`results/dev/p7dev_kaggle_mavi/p7dev_kaggle_mavi.json`, a50017e): training step 2.343 s (fp32) against 0.587 s (fp16) per batch-2 iteration; the 500-iteration fp16 run took 0.613 s per iteration end to end. Projected for 30,000 iterations: fp16 **5.11 h per seed**, fp32 about **19.5 h per seed** (2.343 s x 30,000; a projection, not a run). Three fp32 seeds would be about 58 h, more than a weekly quota, so fp16 is required.

Fixed: MAVI training uses `torch.autocast(float16)` with `GradScaler`, BatchNorm and loss in fp32 (autocast default), skipped steps counted, `"deviation": "fp16"` written to every `train_*.json` (as in D3). **Scoring (test inference) runs in fp32.** fp16 against fp32 forward on 20 train rows: max absolute difference 1.49e-3 (0.10 % of the maximum fp32 magnitude). The precision is fixed from train-only timing, not from any test score.

### 12.5 Budget ladder result (section 11, D2)

Train-only benchmark: MAVI 30,000 iterations x 3 seeds in fp16 projects to 5.11 h per seed, **15.3 h** in total (0.613 s end-to-end per iteration x 30,000 x 3), below the 22 h line, so ladder step (1) holds: **MAVI 30,000 iterations x 3 seeds**. The 15.3 h is a projection from the 500-iteration run: data loading and checkpoint I/O may differ at 30,000 iterations, and there is no allowance for reruns (section 6 allows one relaunch after a crash). One seed per Kaggle run (about 12 h cap) is respected. Skipped GradScaler steps are to be reported per seed in each `train_*.json` summary.

Test inference time: the 179 held-out rows took 74.6 s (0.417 s per row) in `heldout_eval`, which calls `edalint_p7.train.predict`; that function runs in eval mode, batch 1, without autocast (fp32, per its docstring and code), and the 74.6 s also includes the mean_map prediction and metric accumulation. So this was **fp32 inference**, and the figure is an upper bound for inference alone. Projection for 3,544 test rows: about 1,478 s = **24.6 min per seed, 1.23 h for 3 seeds**, well under the 4 h line of D3. D3 is settled: **test inference runs in fp32**.

### 12.6 Rule for undefined NRMS rows (IR drop, secondary metric)

Finding (dev, train-only, from `E:\edalint_data\p7_export\ir_drop_train`): the held-out nvdla-small NRMS is `inf` for both the model and mean_map because **2 of the 84 nvdla-small rows have a constant label** (`y_2` and `y_14` in `nvdla-small/shard_000.npz`, every pixel -1.4242e-10, i.e. zero IR drop in effect). The other exported designs have none (Vortex-small 0 of 95, RISCY 0 of 3,162, RISCY-FPU 0 of 3,449; 2 of 6,790 in total). For these rows `max(gold) - min(gold) = 0`, so `edalint.metrics.nrms` returns `inf` for any prediction with RMSE > 0 (and `nan` for a prediction equal to the constant gold, 0/0, `metrics.py:99`; both are dropped by `np.isfinite`); the label also stays constant (0) after the clamp-and-quantise step of `nrms_official`. This is the same behaviour as skimage min-max NRMS and as the authors' wrapper, except that the authors replace `inf` by 0.05 (utils/metrics.py 122-123) while P7 scoring passes `official_inf=None`. The `inf` reached the dev summary only because `heldout_eval` in `scripts/p7_dev_kaggle_common.py` aggregates with `np.nanmean`, which keeps `inf`; `edalint.p4_n14.PrimaryAcc.value`, the aggregator the P7 scorer uses (section 4), already drops non-finite values. No metric code is changed.

Rule (the N14 convention, protocol_n14.md section 6 NRMS paragraph, applied to P7): per-row NRMS with `official_inf=None`; aggregates use finite rows only; the count and ids of undefined rows are reported per design and per model-seed; the 0.05-substituted mean is reported as a secondary. The IR primary (official ROC area at 0.9885) does not use NRMS, but it is not blind to such rows: the official ROC rule skips rows with no positive pixel at 0.9885 (`src/edalint/p4_n14.py:184-189`, `n_no_positive`), as in N14 H3, and the 2 constant rows have none. P7 must report, per design and per model-seed, the count of rows with no positive pixel next to the undefined-NRMS count. The overall and nvdla-small NRMS values in the 500-iteration dev file are Infinity and must not be quoted. On the NRMS of Vortex-small alone the model (0.3997) is worse than mean_map (0.3752); this is not a criterion and not evidence of skill either way. `scripts/p7_dev_kaggle_common.py` now aggregates with finite-only means and records the undefined count (a dev script outside the p7 code hash; edited after the run, no run repeated, so the stored dev JSON still holds the old values).

### 12.7 Not changed, and still open before a freeze

Unchanged: IR split, exclusions, metrics, seeds, H-A/H-B/H-C for IR drop, the run-once and freeze rules (section 6), the pre-freeze tests of section 11 D7, the Kaggle plan D8 minus the congestion kernel, the code-hash recipe. MAVI test inference (scoring) runs on a Kaggle T4 in fp32 (12.5, D8 step 6); the resulting scored files are copied to `results/p7/` and read by the aggregator of 12.10. Open: the freeze commit and `FROZEN_p7.json`; the D7 tests; the weekly GPU quota check and the user's go-ahead for any further Kaggle use; whether the pre-freeze tests of section 11 D7 are complete (12.8).

### 12.8 D5 for IR, and pre-freeze tests

D5 for IR drop is **unchanged**: gate factor 0.5, `G_ref` = 0.8671 - 0.6301 = 0.2370, so the gate is `R_t >= 0.6301 + 0.5 x 0.2370 = 0.7486`. The factor was fixed in section 5 before any dev run and is explicitly not tuned toward the 0.7678 dev value of 12.3 (a 500-iteration, two-design, one-seed number). Pre-freeze tests: `tests/test_p7_data_score.py`, `test_p7_guard.py`, `test_p7_models.py`, `test_p7_recipe.py` and `test_p7_static.py` exist and passed when this amendment was written (71 tests); whether they cover every item listed in section 11 D7 has not been audited here. **Added later (pre-freeze):** D7 coverage was audited in the independent review of d1e730d, which mapped every section 11 D7 item to a test; the only partial item, the exporter's refusal before the freeze, is now covered by `tests/test_p7_verdicts.py::test_export_script_refuses_the_evaluation_export_before_the_freeze`, so the D7 test list is complete.

### 12.9 Sampler deviation: the registered MAVI run uses the epoch sampler (post-D2, decided before the freeze)

Section 11 D2 says the trainer must draw 100-iteration blocks, each a fresh shuffle without replacement (the `block100` sampler). `src/edalint_p7/train.py` defaults to `sampler_mode='epoch'` (one continuous stream of permutations, restarted only when exhausted), changed in f2f462e ("sampler corrected to epoch") and 4d3e1df ("epoch sampler is the default (block100 kept as sensitivity mode)"), and pinned by `tests/test_p7_recipe.py::test_default_sampler_is_epoch_and_block100_remains_available`. The reason recorded in the code (`src/edalint_p7/recipe.py`, `BlockSampler` docstring): in the pinned authors' `build_dataset.IterLoader`, `__iter__` returns self, so the `break` at `print_freq` in `train.py` does not restart the loader; the authors' stream is continuous, which is `epoch`, not a reshuffle every 100 iterations as D2 assumed. **Decision (before the freeze): the registered MAVI run uses `sampler_mode='epoch'`; the code default is not changed.** Every dev run used epoch (the MAVI check a50017e and all GPDL runs except the local D1 of 12.1, which used block100). `block100` remains in the code as a sensitivity / alternative mode and is not part of the registered run. This is a deviation from the text of section 11 D2 made after D2 and before the freeze; every `train_*.json` records `sampler_mode`.

### 12.10 Pre-freeze clarifications made while implementing the aggregator (`src/edalint_p7/verdicts.py`, `scripts/p7_run_verdicts.py`)

The aggregator reads only `results/p7/models_ir_drop_A_s<seed>.json` and `results/p7/train_ir_drop_A_s<seed>.json` (seeds 20261101-3) and the registered N14 files `results/n14/models_ir_drop.json` and `results/n14/verdicts.json`; it refuses before the freeze and is run-once. Points the text left open, with the rule chosen (made before the freeze; none uses a test value):

1. **Exact versus rounded gate constants.** `M_t` is the exact registered mean_map design-macro ROC area (0.6301175...), `G_ref = 0.8671 - M_t` (0.23698...), and the gate is evaluated as `(R_t - M_t) >= 0.5 x (0.8671 - M_t)`, i.e. `R_t >= 0.74861`. The text's 0.2370 and 0.7486 are the rounded values; an R_t between 0.74860 and 0.74861 is therefore below the gate.
2. **`G_b`** is the largest gain over `mean_map` among the six registered non-`mean_map` baselines (zero, scalar, ridge, knn, green, gbm), design-macro, read from `results/n14/models_ir_drop.json`; it is ridge (0.22086). H-B has no `G_b > 0` condition (the gate already forces `G_R > 0`).
3. **`R_t`** is the mean over the defined seeds; with fewer than 2 defined seeds the verdict is "no verdict". A seed is **diverged, hence undefined, iff** its train result (`results/p7/train_ir_drop_A_s<seed>.json`, read next to the scored file) is missing, or its `last_block_loss` (mean loss of the last 100-iteration block; `nan` if training stopped on a non-finite loss) is missing or non-finite, or `final_params_finite` is not `true` (any non-finite floating-point entry of the final state dict; a record without the field counts as not finite). It is also undefined if its scored file is missing or its design-macro ROC area is not finite. The rule is applied before the scores are used, so a diverged seed's scored file (where `nan_to_num` could make a finite constant-like predictor) is excluded and the reason is written in `verdicts_p7.json`. The skipped-step fraction is reported per seed but is not a divergence rule. `train.py` writes `last_block_loss` and `final_params_finite` into the train record (added pre-freeze; `diverged_at`, `skipped_steps` and `loss_per_100_iters` already existed).
4. **Gate failure** gives verdict "no verdict" (JSON `null`); H-B and H-C are still printed, marked descriptive.
5. **Sensitivities recompute H-B only.** For best seed / worst seed (by design-macro ROC area) the whole verdict (gate and H-B) is recomputed; "seed-sensitive" if either differs from the mean-seed verdict. For the anchored ROC area and the sample-pooled value, `M_t`, `G_b` and `R_t` are all recomputed in that metric (pooled `R_t` is the mean over seeds of the pooled value) and the H-B rule applied; the **gate is taken from the primary** (the published 0.8671 is an unanchored, per-design-averaged figure), and when the primary gate fails these sensitivities report "no verdict". The D5(d) sensitivity uses `G_R = max(G_R, G_ref)`, flagged "budget-sensitive" if the H-B verdict flips (only when the gate passes). D5(b) (GNN-based gate) is congestion only and is void. Per-design values and the count of designs on which the seed-mean beats mean_map are reported.
6. **H-C.** `|D| <= 0.03` is evaluated on `D` rounded to 12 decimals (floating-point equality at the band edge). The wording rules of D5(e) are emitted as flags: `resolved_wording_allowed` (gate passed and |D| within the band), otherwise "gate passed, retrain below published" (D < 0) or "gate passed, retrain above published band".
7. **NRMS (12.6).** Per design and model-seed: the official `nrms` field, mean over finite rows, count and ids of non-finite rows; the design-macro is the mean over designs with a finite mean. Secondary: `inf` rows replaced by 0.05, `nan` rows still excluded. `n_no_positive` and `n_samples` come from each design's `official_roc` block, per model-seed.
8. **Registered N14 comparison.** The registered IR verdict and ratio are copied from `results/n14/verdicts.json` unchanged, with "P7 agrees" (None when P7 has no verdict).
9. **File checks.** Each scored file must carry task `ir_drop`, arm `A`, the seed of its file name, and `eval_ids` / scored row ids equal to the registered `eval_ids`; otherwise the aggregator aborts. The id check is applied to every seed that has a scored file, diverged seeds included, so a bad id set aborts the whole run. A train record whose task / arm / seed does not match its seed aborts in the same way.
10. **Train-record consistency (undefined, not an abort).** In this order a seed with a scored file is undefined if: the divergence rule of item 3 fires; its train `iters_done` is not 30,000; its scored file's `p7_code_hash`, or its train record's, differs from the current (= frozen) `p7_code_hash`. The reason is written in `verdicts_p7.json`.
11. **Best / worst seed** of the seed sensitivity are taken among the defined seeds only (by design-macro official ROC area).
12. **Undefined sensitivities.** If the anchored or pooled `R_t` (or `M_t`) is not finite, that sensitivity is reported as "undefined" (verdict null), never as FALSE.
13. **`ratio_G_b_over_G_R`** in the JSON is `G_b / G_R` with the primary (design-macro, mean-seed) values; it is reported for information, the rule being `G_b >= 0.8 x G_R`.
14. **The `rule` string** in `verdicts_p7.json` (`H-A: R_t - M_t >= 0.5 x (0.8671 - M_t); H-B (only if H-A passes): TRUE iff G_b >= 0.8 x G_R; H-C descriptive |R_t - 0.8671| <= 0.03`) is a label for readers; the code in `src/edalint_p7/verdicts.py` is the rule.
15. **Run-once.** Before writing, the aggregator checks that neither `verdicts_p7.json` nor `tables_p7.md` exists, and writes nothing if either does.
