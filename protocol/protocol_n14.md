# edalint CircuitNet-N14: pre-registered audit protocol

**STATUS: DRAFT, lead decisions of 2026-10-13 incorporated (section 13). NOT FROZEN; the lead reviews before the freeze.** Written 2026-10-13, modelled on
`protocol/protocol_iccad23.md`. Nothing here is binding until the freeze commit (section 12). Section 13 lists what is still open.
No N14 test sample has been read, scored or fitted on by this project beyond the metadata and shape reads disclosed in section 2. The
ICCAD'23 freeze (`protocol/FROZEN.json`, `"dataset": "iccad23"`) does not unlock any N14 test split: since the post hoc guard change
(commit e541efe) the N14 freeze is its own file, `protocol/FROZEN_n14.json`.

## 1. Question

Same question as ICCAD'23, for the three CircuitNet-N14 map tasks that are on disk (IR drop, congestion, DRC): how much of the published
deep-learning skill is reproduced by trivial, linear and (for IR drop) physics-flavoured baselines trained on the authors' own
preprocessed inputs, and do the official split and labels carry data faults or metric artefacts that change how the published numbers
should be read? In addition, one comparison specific to this benchmark: how much easier is a random sample-level split than the
official design-level split (section 8, C1).

The claim is about **benchmark design**, not about ML skill. N14 has no exact solver in our hands, so there is no "dominates the learned
model" hypothesis; the ICCAD'23 H2 (exact solve) has no counterpart.

## 2. What was looked at before the freeze (disclosure)

* **Test designs (zero-riscy, openc910-1, Vortex-large; nvdla-large is in neither split file).** Before this draft, the project read:
  (a) the names and member counts of the test tarballs (`manifests/n14_inventory.json`, `results/dev/smoke_test_shapes_p3dev.json`);
  (b) the **array shapes** of the first file of each routability feature of each test design (same smoke test; no values);
  (c) design and sample **names only** for E004 (`results/dev/circuitnet_n14_drc_authors_all_p3dev_metadata_only.*`, 200 ids);
  (d) the test CSV rows. No test map or label **value** was read, no test statistic computed, and no baseline fitted or scored on a test
  sample. No committed record shows that the IR-drop tarballs of the test designs were opened by this project (the inventory and the smoke test cover routability tarballs only).
  (e) **Pre-freeze housekeeping on all 16 tarballs, test designs included (2026-10-13):** sha256 of the whole files (all 16 match the Hugging Face LFS oids) and the member-name inventory (`manifests/n14_inventory.json`, names and counts per directory only). No array was decoded.
* **Published information not independent of the test labels:** CircuitNet 2.0 and GIF report results on N14 whose test designs may
  overlap the official test designs (section 9 states which ones and why they are only weak references). None of those numbers was used
  to choose a baseline setting.
* **Train-only development (labelled dev):** `results/dev/circuitnet_n14_*_p3dev.*` and `results/dev/n14_ir_label_clip_probe_p3dev.json`. They
  cover **only a 40-per-design evenly spaced subset of Vortex-small and nvdla-small** (the two small train designs; 80 samples per task; the clip probe read all 96 and 85 IR labels of those two designs) in the authors' preprocessing. RISCY and RISCY-FPU (93 % of the train rows) have not been read by this project.
  Everything cited from dev below is tagged **(dev)**; these numbers motivate choices, they are not criteria and not results.
* **Dev expectations, set after seeing the dev numbers and stated now:** IR drop: ridge and GBM beat mean_map clearly on NRMS, SSIM and
  AUC (dev: GBM NRMS skill 0.45, ridge 0.23; section 6), and H1 is **undecided** on IR drop (the dev SSIM ratio is 0.78, close to the 0.8 line).
  DRC: the official hotspot threshold is empty for essentially every sample (dev: 80 of 80), so H3 is expected **TRUE** for DRC.
  Congestion: no dev baseline numbers exist; no expectation. Sections 7 and 8 decide, not these expectations.

## 3. Data, split, inference unit

* Source: Hugging Face `CircuitNet/CircuitNet`, revision `df74f33e302f927bf0548dc0d7936ca43b634f2a`, `CircuitNet-N14/IR_drop_features/<design>.tar.gz`
  and `CircuitNet-N14/routability_features/<design>.tar.gz`, 8 designs each, local `E:\edalint_data\circuitnet_n14` (sizes and HF LFS sha256 in
  `manifests/MANIFEST.json`). Read-only. Tasks and channels follow the authors' `generate_training_set.py` (section 4).
* **Split: the authors' official files**, `routability_ir_drop_prediction/files/train_N14.csv` (6,790 rows) and `test_N14.csv` (3,544 rows) in
  github.com/circuitnet/CircuitNet commit `41ade1d7e913a418b4495b6b13630b26b0ff7279`. The packaged copies `src/edalint/data/*_N14.csv` are
  byte-identical to the cached originals (sha256 `c4fca2e4...` train, `bec6e322...` test). One pair of files serves all three tasks.
  **Train** = RISCY 3,162 + RISCY-FPU 3,449 + nvdla-small 84 + Vortex-small 95 (4 designs). **Test** = zero-riscy 3,441 + openc910-1 56 + Vortex-large 47
  (3 designs). The split is by design. nvdla-large is in neither file and is not used. Rows whose files are missing for a task (label or any
  input absent; documented file-count mismatches in the authors' issues #31, #36, #42) are skipped and counted per task and design; the counts
  are reported, not repaired.
* **Validation:** the official files have none. Hyperparameters are chosen by leave-one-design-out cross-validation on train (section 5.2); the final
  models are refit on the seeded fit sample of section 5.3.
* **Design families (pre-declared, lead decision):** {RISCY, RISCY-FPU, zero-riscy} = `riscy`; {Vortex-small, Vortex-large} = `vortex`; nvdla-small and nvdla-large = `nvdla`
  (alone); openc910-1 = `openc910` (`N14_FAMILIES` in `src/edalint/checks.py`). E004 reports cross-split family overlap (here `riscy`: RISCY and RISCY-FPU in train, zero-riscy in test; `vortex`:
  Vortex-small in train, Vortex-large in test) as an **informational observation with the family names**. It is **not** in the H2 material list: the authors' split is design-level by intent.
* **Exclusions (lead decision):** nvdla-large and every sample absent from both CSVs are excluded from all verdicts. Counts per design and task (listed rows lacking files; archive members
  not in either CSV) are reported by the `faults` stage.
* **Inference unit = the design.** The test split has 3 designs and zero-riscy supplies 97 % of the rows. The **verdict basis is the
  design-macro mean** (lead decision) (mean over samples within a design, then mean over the 3 designs); the sample-pooled mean (the published convention)
  is reported next to it; the macro one decides. No p-values:
  n = 3 designs. Descriptive statistics only: per-design values and a sign count (designs on which a baseline beats mean_map).
* **Resolution:** the authors' preprocessing resizes every map to 256 x 256, and all published N14 and N28 metrics in the authors' code are
  computed on those maps. The primary metrics are therefore computed on the 256 x 256 authors-space maps. A native-resolution check is not
  registered (grids range from about 277 x 276 to 2363 x 2362).

## 4. Preprocessing: exactly the authors' (primary mode `authors`)

Source: `routability_ir_drop_prediction/preprocess_scripts/generate_training_set.py` (commit above). Reproduced in `edalint.datasets.circuitnet`
(mode `authors`). Line numbers are those of that file.

| step | authors' code | lines |
|---|---|---|
| cubic resize of congestion and DRC features | `ndimage.zoom(x, (256/H, 256/W), order=3)` | 22-25, 60, 63 |
| area resize of IR features and DRC label | `cv2.resize(x, (256,256), INTER_AREA)` | 27-29, 69, 71, 89, 96 |
| per-map, per-sample min-max scaling of every input channel | `std(x)`: `(x-min)/(max-min)`, unchanged if `max == 0` | 31-36 |
| congestion label | GR horizontal + vertical overflow, each cubic-resized, summed, then `std` | 79-85, 101-102 |
| **DRC label** | `clip(label, 0, 200)`, area resize, `/ 200` | 88-89 |
| **IR-drop label** | `squeeze`, `clip(label, 1e-6, 50)`, area resize, `(log10(.) + 6) / (log10(50) + 6)` | 94-96 |
| IR input: `power_t` 20 slices separately, other maps `squeeze` | 24 channels in total | 66-72 |
| channel lists | congestion `macro_region, RUDY, RUDY_pin`; DRC the 9 channels of section 5; IR `power_i, power_s, power_sca, power_all, power_t` | 119-138 |

Differences from the authors' pipeline that cannot be avoided: the HF release stores `.npz` (key `data`), the script calls `np.load(...)` on `.npy` (script
targets the Google Drive layout, `docs/format_notes.md` 2.2); the early-global-routing maps live on a finer grid than the other maps of the same
sample (ratio about 1.8) and are cubic-resized to 256 x 256 as any other feature. No data augmentation (the authors' `Flip` applies to training of neural
models only).

**Raw-versus-clipped units, an E002/E006 item (registered, section 7.2 R1).** The IR label's physical unit is not documented. The authors clip raw values to
[1e-6, 50] before the log scaling, and every published N14 metric on IR drop is computed on the clipped, log-scaled, min-max-style map. Dev: of the
181 train samples probed, 179 (98.9 %) have a raw maximum above 50 (Vortex-small 96 of 96, raw max median 291, range 120 to 1,386; nvdla-small 83 of 85, median
309, one all-zero label); the median share of pixels above 50 is 0.9 % (Vortex-small) and 0.4 % (nvdla-small), up to 4.3 %; about 52 to 59 % of pixels are exactly
zero. The hotspot threshold 0.9885 corresponds to a raw value of 40.78 (`10^(0.9885 (log10 50 + 6) - 6)`), below the clip, so the positive class is
the band 40.78 to 50 plus every saturated pixel. *Own reasoning, unverified, not a literature claim:* 40.78 is close to 5 % of 0.8 V = 40 mV, which would
make the raw unit mV and "5 % after log" a 5 % supply-drop rule; the data card calls the unit "inferred mV".

## 5. Tasks, inputs, labels, hyperparameter protocol

### 5.1 Tasks (authors' definitions)

| task | loader key | inputs (channels in authors' space) | label | official metrics (`utils/configs.py` defaults) |
|---|---|---|---|---|
| IR drop | `ir_drop` | 24: `power_i, power_s, power_sca, power_all` + 20 `power_t` slices | `IR_drop` (clip, log, scaled) | NRMS, SSIM; ROC/PRC at label threshold 0.9885 (configs.py 73-74) |
| congestion | `congestion` | 3: `macro_region, RUDY, RUDY_pin` | GR overflow (H + V) | NRMS, SSIM, EMD |
| DRC | `drc` | 9: `macro_region, cell_density, RUDY_long, RUDY_short, RUDY_pin_long, eGR H/V overflow, GR H/V overflow` | `DRC_all` clip 200 / 200 | NRMS, SSIM; ROC/PRC at label threshold 0.1 (configs.py 57) |

The repository config defaults point to the N28 split files; the N14 files are used as the authors' documentation instructs
(`--ann_file_train ./files/train_N14.csv --ann_file_test ./files/test_N14.csv`).

### 5.2 Hyperparameters (train-only development, done before the freeze)

Script `scripts/p4_dev_n14_hparams.py`, result `results/dev/n14_hparam_cv.json`. Leave-one-design-out over the 4 train designs (RISCY, RISCY-FPU, Vortex-small, nvdla-small); the
fit set of a fold is the other three designs, each limited to K = 100 rows (the first 100 of the seeded order of section 5.3, a prefix of the registered fit sample); the split CSV given to the
dataset has an empty test file, so no test row can be listed or read. RISCY and RISCY-FPU are siblings, so a fold that holds out one of them still fits on the other: the dev estimate is optimistic and is used only to rank options.
**Selection criterion, set before the run:** best design-macro mean over the 4 held-out designs of the task's primary metric (congestion NRMS lower, DRC SSIM higher, IR drop official ROC area higher); ties go to the first listed option.
Reduced grid (fixed): ridge penalty {1e-4, 1e-2, 1}; kNN k {1, 3, 5}; GBM {100 iterations depth 3; 300 iterations depth 6}; Green ring-bin edges {coarse, medium} (IR drop only).

**Round 1** (`results/dev/n14_hparam_cv.json`) used ridge alpha {1e-4, 1e-2, 1} and our float NRMS / SSIM; it picked alpha = 1, the edge of the grid, and it was found afterwards that the authors compute NRMS and SSIM on uint8-quantised maps (section 6).
**Round 2** (`results/dev/n14_hparam_cv_round2.json`, lead decisions of 2026-10-13) re-ran congestion and DRC in full with the authors' metric definitions and the ridge grid extended to {10, 100, 1000}; for IR drop (ROC metric unchanged) only the three new ridge penalties were run and merged with round 1. Same folds, same K = 100, same criterion.
Frozen values (design-macro primary metric over the 4 held-out designs):

| task (metric) | ridge alpha | kNN k | GBM | Green edges |
|---|---|---|---|---|
| IR drop (official ROC area, higher better) | **100** (0.8641; 10: 0.8573, 1000: 0.8564, 1: 0.8444): an interior optimum | 1 (0.612; k=3 0.609, k=5 0.611) | 100 iterations, depth 3 (0.856; 300/6: 0.844) | coarse (0.8132; medium 0.8129) |
| congestion (official NRMS, lower better) | **10** (0.0345; 100 and 1000 equal to 4 digits: a plateau, so 10 is the edge of the plateau, not of an improving trend; 1: 0.0346) | 5 (0.038; k=1 0.0437, k=3 0.0391) | 100 iterations, depth 3 (0.0361; 300/6: 0.0376) | n/a |
| DRC (official SSIM, higher better) | 1e-4 (all six penalties equal to 4 digits, 0.9993) | 5 (0.9992) | 100 iterations, depth 3 (0.9993; 300/6: 0.9986) | n/a |

Frozen in `FROZEN_HPARAMS` of `src/edalint/p4_n14.py`. Cautions for the report: the DRC selection is a tie (the metric is saturated, section 7.2) and does not enter any verdict; the IR-drop kNN choice (k = 1) is a near tie with k = 3 and 5; the best IR-drop ridge (0.864) now beats the best GBM (0.856) in dev.

### 5.3 Fit sample and test scoring (lead decision)

* **Fitting:** at most K = 400 rows per train design, a **seeded random** choice: `numpy.random.default_rng([20261013, crc32(design)])` permutes the name-sorted CSV rows of the design and the first 400 are used
  (Vortex-small 95 and nvdla-small 84 are used whole), so about 979 rows per task. The seed and the chosen indices are written into every `models_<task>.json` (`fit_sample`). The selection is by CSV name only, the same for all tasks, and never looks at a value.
* **Test scoring and every fault rate use every official test row, uncapped** (zero-riscy 3,441, openc910-1 56, Vortex-large 47 before the per-task missing-file drops). Test designs are streamed one at a time; only the 256 x 256 authors-space maps are cached (float32).
  Cache budget: the cache must stay under 60 GB (estimate: IR drop 6.3 MB per row, DRC 2.6 MB, congestion 1.3 MB; capped train plus uncapped test about 13 GB + 22 GB + 9 GB + 5 GB + 5 GB, about 46 GB, to be measured in the dry run).
* The C1 comparison uses nested subsets of this selection (section 8).

## 6. Metrics, skill, published references

All primary metrics are computed per sample on the 256 x 256 authors-space maps with `src/edalint/metrics.py`, which reproduces the authors' `utils/metrics.py`:

* **NRMS** = RMSE / (max(gold) - min(gold)) (skimage min-max; `metrics.py` 111-123). A constant gold gives inf, which the authors' wrapper replaces by 0.05 (lines 122-123), or 0/0 = NaN.
  Registered primary: defined values only, with the counts of undefined and of substituted samples reported (and the 0.05-substituted variant as secondary).
* **SSIM**: the authors' implementation (`metrics.py` 72-107: 11 x 11 Gaussian window, sigma 1.5, C1 = (0.01 x 255)^2, C2 = (0.03 x 255)^2, the 5-pixel border cropped, mean over the map, no pooling over samples).
  **The authors' SSIM and NRMS do not see the float maps:** `input_converter` (`metrics.py` 35-55) passes both arguments through `tensor2img` (339-368), which clamps to [0, 1], multiplies by 255, rounds and casts to uint8; `test.py` 53-54 calls `metric(target, prediction)` and drops values exactly equal to 1
  while still dividing by the dataset length. Our primary NRMS and SSIM therefore use `metrics.ssim_official` / `nrms_official` (same quantisation; checked against a cv2 transcription in `tests/test_metrics.py`); the float versions are secondaries (`ssim_float`, `nrms_float`).
  Train-only evidence (`results/dev/n14_metric_checks.json`, fit on RISCY and RISCY-FPU, scored on 179 Vortex-small and nvdla-small rows): quantisation turns 16 of 179 DRC labels into all-zero images (NRMS undefined for them) and leaves SSIM at 0.9998 for zero, scalar and mean_map predictions alike, so **the authors' own SSIM is saturated on DRC**: the gap to CLDRoute's 0.7136 is not a definition mismatch of ours.
* **ROC-AUC / PR-AUC (IR and DRC), official procedure** (`metrics.py` 175-250, 273): 200 prediction thresholds in `linspace(0, 1, 200, endpoint=False)`, label cut with `>= threshold`; per-threshold TPR, FPR and precision are **averaged over the samples that are not skipped**, where a sample is skipped if
  `fp = tn = 0` or `tp = fn = 0` or `tp = fp = 0` (`metrics.py` 175-221), i.e. every sample **without a positive pixel is silently dropped**. Implemented as `RocAcc` in `src/edalint/p4_n14.py` and tested against a line-by-line transcription of the authors' code. The area is the authors' `roc_prc` (duplicate FPR values dropped in threshold order, sorted, the point (1, 1) appended, trapezoid).
  **Artefact found while implementing it (a metric-sanity observation, not yet adjudicated):** the authors' curve has no (0, 0) anchor and drops thresholds at which nothing is predicted positive, so a **constant predictor scores 0, not 0.5**; the official area of weak predictors therefore depends on
  how high their predictions reach. The primary IR-drop metric stays the official procedure (lead decision, for comparability with published numbers); the (0, 0)-anchored area and the mean per-sample rank AUC (undefined for single-class samples) are secondaries.
  **E006 metric finding, registered:** "the authors' ROC area has no (0, 0) anchor; a constant predictor scores 0". Train-only evidence (`results/dev/n14_metric_checks.json`, IR drop, 179 rows): zero, scalar and constant 0.5 predictors have official area 0.000, anchored area 0.500, per-sample rank AUC 0.500; mean_map 0.505 / 0.505 / 0.505;
  4 of 179 rows have no positive pixel and are skipped by the authors' rule. The `verdicts` stage repeats this on the test split from the constant baselines (`zero`, `scalar`).
  **Declared sensitivity analysis:** the IR-drop H1 verdict and the IR-drop C1 gap are recomputed with the anchored area under the same mechanical rule and reported alongside (`sensitivity_anchored` in `verdicts.json`); any disagreement with the official-area verdict is flagged in the verdict table. The official area decides.
* Mean over samples is over defined values (not the `test.py` 53-54 running mean that skips values exactly 1 but divides by the full length; that variant, `official_running_mean`, is reported as secondary).
* Secondary, all tasks: NMAE, Pearson. Spearman, Kendall, PR-AUC and congestion EMD are not computed (cost; not comparison quantities here).

**Primary metric per task (fixed now), and best published reference** (`docs/published_numbers.md` sections 2.1 to 2.4; all reference numbers are on splits or normalisations that differ from ours or are unstated, so H1 is a **weak comparison** and is labelled so everywhere):

| task | primary | best published N14 value used as reference | source | comparability caveat |
|---|---|---|---|---|
| congestion | NRMS (lower better) | 0.020, Circuit GNN | CircuitNet 2.0 App. B (ICLR 2024), N14; FCN 0.036, cGAN 0.044 | split unstated ("unseen designs"); "NRMSE" as printed; GNN uses a graph input, not the map channels |
| DRC | SSIM (higher better) | 0.7136, CLDRoute LDM | CLDRoute (2607.16674) Table 5 N14: split 10,368 / 169 / 250, test designs unnamed | different split; only N14 DRC numbers published; N14 MAE 0.00627 not comparable (no NMAE definition match) |
| IR drop | ROC-AUC (higher better) | 0.8671 +- 0.0032, MAVIREC | CircuitNet 2.0 App. B.2 | hotspot rule "FPR <= 5 %", split and threshold not stated |

Secondary references: congestion SSIM 0.872 (FCN) / 0.870 (GNN); IR SSIM 0.558 (GIF Table 7, zero-riscy test, VDD-normalised, different split); DRC none. N28 numbers (letter, PDNNet, CLDRoute N28) are
context only and never enter a criterion. **No published paper reports a mean-map, ridge, kNN, GBM or Green baseline on N14** (`docs/published_numbers.md` section 3).

**Skill (the analogue of Skill_MAE).** For a metric m, Gain_m(model) = m(mean_map) - m(model) if lower is better, m(model) - m(mean_map) if higher is better, with mean_map evaluated by us on our test split
(design-macro); H1 compares gains (a ratio, scale free). For C1 and the tables, Skill = Gain / headroom (lower better: 1 - m/m_ref; higher better: (m - m_ref)/(1 - m_ref)). Published models are referenced to **our** mean_map, as in ICCAD'23, because their own mean-map numbers do not exist.

Dev (train only, 40 Vortex-small samples fit, 40 nvdla-small samples scored, IR drop; the only dev baseline numbers that exist): mean NRMS zero 0.682, scalar 0.378, mean_map 0.388, kNN(k=3) 0.428, ridge 0.299, GBM 0.214
(skill vs mean_map 0.03, -0.10, 0.23, 0.45); mean SSIM mean_map 0.149, scalar 0.198, kNN 0.092, ridge 0.241, GBM 0.469; mean AUC mean_map 0.521, ridge 0.891, GBM 0.882, kNN 0.546 (2 of 40 samples undefined).
One consequence for how to read it: with these SSIM numbers GBM's gain over mean_map would be 0.78 of the gain of GIF's 0.558, so the H1 line of 0.8 is not foregone.

## 7. Baselines, runs, hypotheses and criteria

### 7.1 Registered baselines (none may be added or dropped after the freeze)

Per task, fitted on train, scored once on test, authors' 256 x 256 space, definitions in `src/edalint/baselines/zoo.py`:

* all tasks: `zero`, `mean_map`, `scalar`, `ridge`, `knn`, `gbm` (hyperparameters from section 5.2);
* IR drop only: `green` (Green's-function superposition of `power_all`, the physics-flavoured baseline; it needs no netlist and ignores the layout of the PDN);
* not applicable and not registered: `effdist`, `effdist_deg2` (ICCAD-only input map), `spice` (no netlist in the release).

They see only the authors' input channels and the train labels. No baseline sees a test label.

### 7.2 Hypotheses and criteria (decided now; the outcome does not change them)

**H1, "trivial/linear baselines reproduce most published skill", per task on the primary metric (design-macro).** Let G_b be the largest Gain_m over the registered baselines and G_pub the Gain of the reference in section 6 (both against our mean_map).
The pair (task, primary metric) is **TRUE if G_b > 0 and G_b >= 0.8 x G_pub**; if G_pub <= 0 (the reference is not better than our mean_map) the pair is TRUE if any baseline beats mean_map, and is labelled "degenerate".
**Not evaluable** (mechanical rules): (i) the primary metric is undefined on the test split by the official procedure (for ROC: every test sample skipped by the authors' rule, or no finite value for mean_map); (ii) **metric saturated: the primary metric is SSIM and the design-macro SSIM of mean_map is >= 0.99** (`SATURATION` in `p4_n14.py`).
Rule (ii) is expected to apply to DRC (train-only evidence in section 6: 0.9998). The DRC pair then has no verdict; H1 overall needs both other tasks to be TRUE. The authors report no N14 ROC-type DRC number (the 2022 letter's ROC-AUC 0.95 / PR-AUC 0.63 is N28 and its split is not stated), so no ROC-type DRC primary replaces SSIM; the authors' DRC ROC area (threshold 0.1) is reported as a secondary
(`secondary_roc_official_design_macro`; dev: none of the 179 train rows scored has a pixel at 0.1, so it may be undefined). **Per-task verdicts are the primary output; H1 overall is TRUE if at least 2 of the 3 tasks are TRUE.**
Every pair is flagged **weak reference** in the table (other split, other designs, other preprocessing or an unstated hotspot rule). 0.8 is the ICCAD'23 line.

**H2, "data faults of material size exist", reported per task and per test design, from a full (uncapped) scan of every listed test row.** Items:
(a) all-zero label (the authors-space 256 x 256 label is zero); (b) **all-zero bounding-box input**: every bbox-derived input channel of the task (`RUDY, RUDY_pin` for congestion; `RUDY_long, RUDY_short, RUDY_pin_long, cell_density` for DRC;
not applicable to IR drop) is all zero in the sample, the closest map-level proxy of GIF's claim (arXiv 2604.09999, Appendix E: "instance bounding boxes are all zero"). **Limitation (lead decision, graph features are not downloaded): this proxy can only confirm GIF's claim, never refute it;** a null result is reported as "not reproduced on map features";
(c) **cross-split exact duplicates:** SHA-1 of bytes and shape of every stored (native resolution) input array and label array of the task, every listed train row against every listed test row (the scan streams all listed rows of all used designs); constant arrays (all-zero or constant) are excluded because every design has them;
(d) non-finite values; (e) constant non-zero labels. **An item is MATERIAL if** its rate is **>= Y = 1 %** of the task's scanned test rows (pooled), **or >= 5 % in any one test design**
(openc910-1 and Vortex-large have 56 and 47 rows, so one row is already about 2 %), **or if any cross-split exact duplicate exists** (a non-constant input array or a label array). **H2 is TRUE if any item is material on any task.**
Y = 1 % equals the `edalint` E002 default error rate. Dev rates for orientation (train subset, 80 rows per task, so 1 row = 1.25 %): zero label 1 of 80 (IR) and 1 of 80 (DRC), 0 of 80 (congestion); all-zero bbox 0 of 80; duplicate groups 0;
channel-level duplicates only for `macro_region` inside one design (info). **The empty-hotspot rate is not an H2 item** (lead decision: a metric issue, not a data fault); it is part of H3. Cross-split family overlap is informational and not an H2 item.

**H3, "metric sanity: an official metric is undefined, substituted or silently restricted for a material share of test rows".** Per task, shares of scanned test rows with: a constant label (NRMS is inf, replaced by 0.05 by the authors' wrapper, or NaN); for IR drop and DRC, **no positive pixel at the official threshold**
(per-sample AUC undefined; the sample is silently skipped by the authors' ROC average) and no negative pixel; and the share of rows on which the mean_map prediction has SSIM exactly 1 (dropped by `test.py`). **H3 is TRUE if any share is >= 5 % on some task.**
Dev: DRC 80 of 80 rows have no positive pixel (threshold 0.1), IR drop 2 of 80, so DRC is expected TRUE. Also reported, not in the criterion: the constant-predictor ROC artefact of section 6, the number of rows the official ROC procedure retains, and the effect of the 0.05 substitution.

**E002/E006 item R1 (raw versus clipped IR label units), reported with its own pre-stated reading.** On the test split: share of samples whose raw label maximum exceeds the clip 50; share of pixels clipped; share of raw pixels at or above 40.78
that are saturated; unit status. The clip is **confirmed by the authors' code** (line 95) and the unit is **undocumented** (data card), which satisfy the adjudication rule (a) of the ICCAD'23 protocol. Class: **MATERIAL if >= 50 % of test samples have raw max > 50**
(the scored target is then a truncated quantity for most samples; dev 98.9 %); otherwise BENIGN. The DRC clip at 200 (line 88) gets the same probe, **informational only** (no materiality class). The probe is part of the `faults` scan (raw stored label arrays, no model).

**Adjudication of any other finding** (E001, E003 near-duplicates, E005 orientation, E006): carried over from ICCAD'23 section 7. A finding counts as a TRUE fault only if the documentation or the authors' code states the property, or a physics/identity check stated here
confirms it, or the maintainers confirm; everything else is an unadjudicated observation. The archive-versus-CSV accounting (members not listed in the CSVs; dev: Vortex-small 1, nvdla-small 4 to 12 per task) is reported as an accounting fact, not a fault.

### 7.3 Order and runs

`python -m edalint.p4_n14 {faults, models --task T, gap --task T, verdicts}` (the package is not pip-installed: run with `PYTHONPATH=src`), one result file per call in `results/n14/` (`faults.json`, `models_<task>.json`, `gap_<task>.json`, `verdicts.json`, `tables.md`), all
written with exclusive creation. Each stage requires `protocol/FROZEN_n14.json` with a matching code hash and `src`, `protocol`, `scripts`, `tests` matching HEAD. Suggested order: `faults`, then `models` and `gap` for the three tasks (two jobs at most), then `verdicts`.
The dataset layer also refuses to read or stage a test row without the freeze (`stage_all(include_test=True)` and `get`). A **dry run** (`--dry-run DIR --hold-out nvdla-small,Vortex-small`) swaps the test split for held-out train designs, uses a throw-away freeze record under DIR, writes only under DIR, and refuses if any
official test row appears in the substituted test CSV. All 16 tarballs have a recorded sha256 equal to the HF LFS oid; the loader accepts any tarball with a recorded hash that does not contradict the HF oid.

## 8. Pre-registered comparison C1: random-split versus design-split gap (E004)

Not a hypothesis; a measurement with a stated reading. For each task, using the ridge baseline (registered hyperparameters) and the primary metric:

* **C1a (train only, no test data).** The first K = 100 rows per train design of the seeded order (a prefix of the fit sample): 5-fold random sample-level cross-validation versus leave-one-design-out (4 folds). Skill of ridge over `scalar` on the task's primary metric (headroom-normalised, section 6); gap = Skill_random - Skill_design (mean over folds).
* **C1b (reads the test split as part of the single registered run, lead decision).** Pool the capped train rows (the K = 400 fit sample) and, for each test design, the first 400 rows of its seeded order; draw a random sample-level split with the **same train fraction as the official split** (6,790 of 10,334, about 66 %), 5 repetitions (seeds 0 to 4); gap = Skill_random (mean over repetitions)
  - Skill_official (ridge fitted on the official train sample, scored on the full official test split, design-macro).
* **Reading, fixed now:** the random split is **materially optimistic** if the C1b gap is >= 0.10 on at least 2 of 3 tasks. C1a is the train-only counterpart and cannot change that reading. Because RISCY and RISCY-FPU are siblings, the leave-one-design-out folds of C1a are themselves partly leaky; the report says so.

## 9. Comparison table (copied from `docs/published_numbers.md` at the freeze)

To be completed at the freeze from sections 2.1 to 2.4 of `docs/published_numbers.md`; the rows in section 6 above are the ones that enter H1. Three facts to carry in the report (`published_numbers.md` sections 4 and 5.2):
no N14 split is published with agreed counts (official files 6,790 / 3,544; GIF 10,444 with test zero-riscy 3,456; CLDRoute 10,368 / 169 / 250); GIF's Table 4 and Table E.1 disagree with each other; and published N14 numbers are much weaker than N28 ones (SSIM 0.56 to 0.77 versus 0.78 to 0.95).

## 10. Runtime budget (measured in the dry run; replaced by the provenance of the registered files)

Constraints: at most 2 CPU jobs, tarballs streamed one design at a time (never extracted wholesale), data read-only.

* **Dry run** (`python -m edalint.p4_n14 ... --dry-run E:/edalint_data/n14_dry_run --hold-out nvdla-small,Vortex-small`; train = RISCY + RISCY-FPU, K = 400 each, 800 rows; "test" = the held-out train designs Vortex-small and nvdla-small, 179 rows; cold caches; two or
  three jobs at once on a laptop that was also running unrelated work, so the times are upper bounds). Wall-clock, seconds: `faults` 6,173 in total (IR drop: RISCY 828, RISCY-FPU 975, Vortex-small 47, nvdla-small 114; routability: RISCY 1,930, RISCY-FPU 2,058, Vortex-small 70, nvdla-small 150);
  `models`: IR drop 2,053 (fits: ridge 99, kNN 59, Green 192, GBM 105, mean_map 130; scoring 179 rows 143), congestion 3,176 (scoring 40), DRC 2,505 (scoring 60); `gap`: IR drop 1,044, congestion 105, DRC 231. Cache: 6.1 GB (IR drop), 1.3 GB (congestion), 0.4 GB (DRC) for 979 rows, i.e. 6.2 MB per IR-drop row.
* **Extrapolation to the registered run** (linear in rows and bytes): test scoring is 3,544 / 179 = 19.8 times the dry scoring (IR drop about 47 min, congestion about 13 min, DRC about 20 min); test staging streams 17.8 GB (IR drop) and 8.2 GB (routability) more; the full `faults` scan covers
  46 GB (IR drop) and 22 GB (routability) of tarballs, about 3 h. Per-task `models` about 2 h (IR drop), 2.5 h (congestion), 2.5 h (DRC) because congestion and DRC stage the routability tarballs separately; `gap` about 20 min in total. Total about 10 h of CPU, **about 5 to 6 h of wall-clock with 2 jobs**
  (job 1: `faults` then `gap`; job 2: the three `models` stages in turn). Possible saving, not implemented (it is hashed code): sharing one routability cache between congestion and DRC removes about 1.5 h.
* **Cache size** for the registered run, extrapolated from the dry run: IR drop (979 fit + 3,544 test rows) about 28 GB, DRC about 12 GB, congestion about 6 GB: **about 46 GB, under the 60 GB limit**; E: has 145 GB free before this run.
* Pre-freeze development (done): the IR-drop CV took 3,104 s of fitting (about 1.2 h with staging), congestion 1,084 s, DRC 343 s.

## 11. Run-once, crash and post-hoc rules

Identical to ICCAD'23 section 9, with `protocol/FROZEN_n14.json` in place of `protocol/FROZEN.json`: each stage requires a matching freeze record and `src`, `protocol`, `scripts`, `tests` matching HEAD, writes with exclusive creation, refuses to overwrite, and is never re-run because of its content; a crash before the file is written may be re-launched once and is logged in section 12;
anything not specified here is POST HOC, labelled, written to `results/posthoc/`, and cannot change a section 7 or 8 verdict. No criterion, threshold, baseline, cap or hyperparameter changes after the freeze commit. At most 2 CPU jobs at a time. Stop processes by PID only.

## 12. Freeze and deviation log

* Freeze (to happen): a commit adding `protocol/FROZEN_n14.json` with `"dataset": "n14"`, the code hash, the sha256 of this file (CRLF normalised), the date and the frozen commit. **The code hash is not the ICCAD'23 one**: the ICCAD'23 results stay tied to their recorded hash (750d252f..., code at 7707b20);
  the post hoc guard change (e541efe) already moved the live code hash away from it. Any further `src/` or `tests/` change before the N14 freeze (the `p4_n14.py` runner, possibly E004 family rules) is made before it and recorded in the freeze commit message.
* Deviations after the freeze: none yet.

## 13. Decisions (lead, 2026-10-13) and what is still open

Decisions, with the section where each is implemented:

1. **Sample cap:** K = 400 per design for fitting, a seeded random choice with the seed and indices recorded; test scoring and all fault rates use every official test row, uncapped; test designs streamed one at a time; cache under 60 GB (5.3).
2. **Verdict basis:** design-macro mean primary; sample-pooled mean also reported (3).
3. **H1:** per-task verdicts at the 0.8 line are the primary output; overall TRUE if at least 2 of 3 tasks; every pair flagged "weak reference" (7.2).
4. **Primary metrics:** as drafted (6). "Not evaluable" only if the metric is undefined on the test split by the official procedure (mechanical rule in 7.2).
5. **Thresholds:** H2 at >= 1 % pooled, >= 5 % in any test design, or any cross-split exact duplicate; empty-hotspot rate moved out of H2 into H3 only; H3 at 5 % (7.2).
6. **R1:** threshold 50 %; the DRC clip-200 probe is informational only (7.2).
7. **C1:** gap threshold 0.10; C1b reads the test split inside the single registered run (8).
8. **Hyperparameters:** N14 train-only leave-one-design-out CV, reduced grid, K = 100 per design, values frozen (5.2).
9. **Family rule:** RISCY, RISCY-FPU and zero-riscy one family; Vortex-small and Vortex-large another; nvdla alone; informational, not in H2 (3).
10. **graph_features:** not downloaded; the all-zero-bbox proxy can only confirm GIF's claim, never refute it (7.2).
11. **Exclusions:** samples absent from both CSVs and nvdla-large are excluded from all verdicts; counts reported per design (3).

Items raised in the first pre-freeze report and their resolution; nothing else is open except the freeze itself:

a. *(decided 2026-10-13)* The IR-drop primary stays the official unanchored ROC area; the anchored sensitivity analysis and the E006 finding are registered (section 6).
b. *(decided 2026-10-13)* SSIM definition checked against the authors' code (section 6): ours differed only by the uint8 quantisation, now adopted; the metric is nevertheless saturated on DRC, so the pair is **not evaluable (rule ii)**, not "degenerate TRUE".
c. *(decided 2026-10-13)* Ridge grid extended to {10, 100, 1000}; optima in 5.2 (IR drop interior at 100; congestion plateau from 10; DRC tied).
d. **Freeze procedure (the only remaining step):** `protocol/FROZEN_n14.json` is created in the freeze commit only; the runner refuses everything until then.
