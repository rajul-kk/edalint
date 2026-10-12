# edalint CircuitNet-N14: pre-registered audit protocol

**STATUS: DRAFT for the lead's review. NOT FROZEN.** Written 2026-10-13, modelled on `protocol/protocol_iccad23.md`. Nothing here is
binding until the freeze commit (section 12). Items marked **[DECIDE]** are choices the lead must make; the full list is in section 13.
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
* **Published information not independent of the test labels:** CircuitNet 2.0 and GIF report results on N14 whose test designs may
  overlap the official test designs (section 9 states which ones and why they are only weak references). None of those numbers was used
  to choose a baseline setting.
* **Train-only development (labelled dev):** `results/dev/circuitnet_n14_*_p3dev.*` and `results/dev/n14_ir_label_clip_probe_p3dev.json`. They
  cover **only a 40-per-design evenly spaced subset of Vortex-small and nvdla-small** (the two small train designs; 80 samples per task; the clip probe read all 96 and 85 IR labels of those two designs) in the authors' preprocessing. RISCY and RISCY-FPU (93 % of the train rows) have not been read by this project.
  Everything cited from dev below is tagged **(dev)**; these numbers motivate choices, they are not criteria and not results.
* **Dev expectations, set after seeing the dev numbers and stated now:** IR drop: ridge and GBM beat mean_map clearly on NRMS, SSIM and
  AUC (dev: GBM NRMS skill 0.45, ridge 0.23; section 6), and H1 is **undecided** on IR drop (the dev SSIM ratio is 0.78, close to the 0.8 line).
  DRC: the official hotspot threshold is empty for essentially every sample (dev: 80 of 80), so H2(c) and H3 are expected **TRUE** for DRC.
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
* **Validation:** the official files have none. Hyperparameters are chosen by group-wise cross-validation on train (section 5.2); the final
  models are refit on all of train (up to the cap K of section 5.3).
* **Related designs across the split (disclosure, not a fault by itself):** Vortex-small (train) and Vortex-large (test) are variants of one design; RISCY and
  RISCY-FPU (train) and zero-riscy (test) are cores of one ISA family. E004 reports the Vortex pair (it flags `family` overlap); the
  RISCY/zero-riscy relation is stated here by the lead's reading of the designs, not detected by code **[DECIDE: add a `riscy` family to E004 before the freeze?]**.
* **Inference unit = the design.** The test split has 3 designs and zero-riscy supplies 97 % of the rows. The **verdict basis is the
  design-macro mean** (mean over samples within a design, then mean over the 3 designs); the sample-pooled mean (the published convention)
  is reported next to it and, if the two lead to different verdicts, both are stated and the macro one decides **[DECIDE]**. No p-values:
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

### 5.2 Hyperparameters (pre-freeze development on train only) **[DECIDE]**

Not done yet. Proposal: the grids of ICCAD'23 section 5.2 (ridge penalty {1e-6, 1e-4, 1e-2, 1}; kNN k {1, 3, 5}; GBM {100 iter depth 3; 300 iter depth 6};
Green ring-bin edges {coarse, medium, fine}, IR only), selected per task by group-wise cross-validation on train. Folds = the 4 train designs (leave one design out; RISCY and RISCY-FPU are held out
one at a time although they are siblings, so this dev estimate is optimistic and is used only to rank options). Selection criterion, fixed now: lowest mean of the task's primary metric
(section 6) over the 4 held-out designs (design-macro), ties to the first listed option. Fit sets are capped by K_cv = 100 samples per design. Result file
`results/dev/n14_hparam_cv_p4dev.json`, produced before the freeze, with the chosen values copied into the runner code (as for ICCAD'23). Alternative: reuse the ICCAD'23 frozen
values (ridge 0.01, kNN 3, GBM 300/6, Green medium) without a dev run, which saves about 2 hours and changes no registered criterion.

### 5.3 Sample caps **[DECIDE]**

Dev cost (section 10) makes a full run expensive in wall-clock and in cache size (about 104 GB of float32 maps for all samples of all three tasks; E: has 145 GB free). Proposal (**K = 400**): for fitting and for
scoring baselines use at most K samples per design (evenly spaced over the name-sorted CSV rows of that design and task, which is what the loader's `max_per_design` does), i.e. train about 979 and test about 503 samples per task
(RISCY 400, RISCY-FPU 400, Vortex-small 95, nvdla-small 84; zero-riscy 400, openc910-1 56, Vortex-large 47). The **fault scans of H2 and H3 are not capped**: they stream every listed sample. Option A: no cap (about 4 times the staging and 10 times the model time, section 10).
The cap changes no verdict logic (the unit is the design), reduces the zero-riscy weight in the pooled secondary numbers only, and is fixed by name order, not by label.

## 6. Metrics, skill, published references

All primary metrics are computed per sample on the 256 x 256 authors-space maps with `src/edalint/metrics.py`, which reproduces the authors' `utils/metrics.py`:

* **NRMS** = RMSE / (max(gold) - min(gold)) (skimage min-max; `metrics.py` 111-123). A constant gold gives inf, which the authors' wrapper replaces by 0.05 (lines 122-123), or 0/0 = NaN.
  Registered primary: defined values only, with the counts of undefined and of substituted samples reported (and the 0.05-substituted variant as secondary).
* **SSIM**: the authors' implementation (`metrics.py` 72-109), data range 1.
* **ROC-AUC / PR-AUC (IR and DRC), official procedure** (`metrics.py` 175-250, 273): 200 prediction thresholds in `linspace(0, 1, 200, endpoint=False)`, label cut with `>= threshold`; per-threshold TPR, FPR and precision are **averaged over the samples that are not skipped**, where a sample is skipped if
  `fp = tn = 0` or `tp = fn = 0` or `tp = fp = 0` (`metrics.py` 175-221), i.e. every sample **without a positive pixel is silently dropped**. Implemented as `official_threshold_metrics`. Secondary: per-sample AUC (undefined for single-class samples).
* Mean over samples is over defined values (not the `test.py` 53-54 running mean that skips values exactly 1 but divides by the full length; that variant, `official_running_mean`, is reported as secondary).
* Secondary, all tasks: NMAE, Pearson, Spearman, Kendall (as in `circuitnet_case_metrics`). Congestion EMD is not computed (not a published-comparison quantity here).

**Primary metric per task (fixed now), and best published reference** (`docs/published_numbers.md` sections 2.1 to 2.4; all reference numbers are on splits or normalisations that differ from ours or are unstated, so H1 is a **weak comparison** and is labelled so everywhere):

| task | primary | best published N14 value used as reference | source | comparability caveat |
|---|---|---|---|---|
| congestion | NRMS (lower better) | 0.020, Circuit GNN | CircuitNet 2.0 App. B (ICLR 2024), N14; FCN 0.036, cGAN 0.044 | split unstated ("unseen designs"); "NRMSE" as printed; GNN uses a graph input, not the map channels |
| DRC | SSIM (higher better) | 0.7136, CLDRoute LDM | CLDRoute (2607.16674) Table 5 N14: split 10,368 / 169 / 250, test designs unnamed | different split; only N14 DRC numbers published; N14 MAE 0.00627 not comparable (no NMAE definition match) |
| IR drop | ROC-AUC (higher better) | 0.8671 +- 0.0032, MAVIREC | CircuitNet 2.0 App. B.2 | hotspot rule "FPR <= 5 %", split and threshold not stated |

Secondary references: congestion SSIM 0.872 (FCN) / 0.870 (GNN); IR SSIM 0.558 (GIF Table 7, zero-riscy test, VDD-normalised, different split); DRC none. N28 numbers (letter, PDNNet, CLDRoute N28) are
context only and never enter a criterion. **No published paper reports a mean-map, ridge, kNN, GBM or Green baseline on N14** (`docs/published_numbers.md` section 3).

**Skill (the analogue of Skill_MAE).** For a metric m, Gain_m(model) = m(mean_map) - m(model) if lower is better, m(model) - m(mean_map) if higher is better, with mean_map evaluated by us on our test split
(design-macro). Skill_m(model) = Gain_m(model) / |m(mean_map)| for NRMS (this equals 1 - NRMS/NRMS_mean_map). Published models are referenced to **our** mean_map, as in ICCAD'23, because their own mean-map numbers do not exist.

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

**H1, "trivial/linear baselines reproduce most published skill", per task on the primary metric.** Let G_b be the largest Gain_m over the registered baselines (mean_map has Gain 0, so G_b is
at least 0 unless every baseline is worse than mean_map) and G_pub the Gain of the reference in section 6. The pair (task, primary metric) is **TRUE if G_b >= 0.8 x G_pub** and G_b > 0;
if G_pub <= 0 (the published reference is not better than our mean_map) the pair is TRUE if any baseline beats mean_map, and labelled "degenerate". **H1 overall is TRUE if the pair is TRUE
on at least 2 of the 3 tasks** (design-macro basis); a per-task table is always given. 0.8 is the ICCAD'23 line, kept so the two audits are comparable **[DECIDE: keep 0.8, and "at least 2 of 3", or demand all 3?]**.
Weak-comparison caveat required in the report: the references are from other splits, other designs and, for IR drop, an AUC whose hotspot rule is unstated.

**H2, "data faults of material size exist", reported per task and per test design, from a full (uncapped) scan of every listed test sample.** Items:
(a) all-zero label (the whole 256 x 256 label is zero); (b) **all-zero bounding-box input**: every bbox-derived input channel of the task (`RUDY, RUDY_pin` for congestion; `RUDY_long, RUDY_short, RUDY_pin_long, cell_density` for DRC;
not applicable to IR drop) is all zero in the sample, the closest map-level proxy of GIF's claim (arXiv 2604.09999, Appendix E: "instance bounding boxes are all zero") whose exact object
(graph features) we did not download, so a null result here **does not refute** the claim and is reported as "not reproduced on map features";
(c) **empty-hotspot rate**: the label has no pixel at the authors' hotspot threshold (IR 0.9885, DRC 0.1 in authors-space; congestion has no official threshold and is not scored here);
(d) **cross-split exact duplicates** (as in ICCAD'23 section 7): SHA-1 of bytes and shape of every input channel and of the label, authors-space arrays, each test sample against every train sample and against other test samples; constant (all-zero) arrays are excluded because every design has them;
(e) constant non-zero labels. **An item is MATERIAL if** its rate is **>= Y = 1 %** of the task's scanned test samples (pooled), **or >= 5 % in any one test design**
(openc910-1 and Vortex-large have 56 and 47 samples, so one sample is already about 2 %), **or if there is any cross-split exact duplicate** of a non-constant input channel or of a label. **H2 is TRUE if any item is material on any task.**
Y = 1 % equals the `edalint` E002 default error rate. Dev rates for orientation (train subset, 80 samples per task, so 1 sample = 1.25 %): zero label 1 of 80 (IR) and 1 of 80 (DRC), 0 of 80 (congestion); all-zero bbox 0 of 80; empty hotspot 2 of 80 (IR), 80 of 80 (DRC);
exact input/label duplicate groups 0; channel-level duplicates only for `macro_region` inside one design (info). Expectation (dev, not criterion): (a) and (c) will be material on DRC, (b) not reproduced.

**H3, "metric sanity: an official metric is undefined, substituted or silently restricted for a material share of samples".** For each task and each official per-sample quantity (NRMS undefined or inf-substituted; per-sample AUC undefined because the label has no positive pixel or no negative pixel;
the authors' skip rule dropping the sample from the ROC/PRC average; SSIM exactly 1 dropped by `test.py`), compute the share of scanned test samples affected. **H3 is TRUE if any share is >= 5 % on some task.**
Also reported (not in the criterion): AUC and hotspot-share fragility under threshold x 0.95 and x 1.05, the number of test samples the official ROC procedure uses at its thresholds, and the effect of the 0.05 substitution on mean NRMS.
H2(c) and H3 are the same measurement seen as a data property and as a metric property; they have different thresholds (1 % and 5 %) and different questions, and the report states the overlap.

**E002/E006 item R1 (raw versus clipped IR label units), reported with its own pre-stated reading.** On the test split: share of samples whose raw label maximum exceeds the clip 50; share of pixels clipped; share of raw pixels at or above 40.78
that are saturated; unit status. The clip is **confirmed by the authors' code** (line 95) and the unit is **undocumented** (data card), which satisfy the adjudication rule (a) of the ICCAD'23 protocol. Class: **MATERIAL if >= 50 % of test samples have raw max > 50**
(the scored target is then a truncated quantity for most samples; dev 98.9 %); otherwise BENIGN. Same probe for the DRC clip at 200 (line 88; not yet probed even in dev). Raw labels are read in mode `raw`, which needs no model fitting. **[DECIDE: threshold 50 %]**.

**Adjudication of any other finding** (E001, E003 near-duplicates, E005 orientation, E006): carried over from ICCAD'23 section 7. A finding counts as a TRUE fault only if the documentation or the authors' code states the property, or a physics/identity check stated here
confirms it, or the maintainers confirm; everything else is an unadjudicated observation. The archive-versus-CSV accounting (members not listed in the CSVs; dev: Vortex-small 1, nvdla-small 4 to 12 per task) is reported as an accounting fact, not a fault.

### 7.3 Order and runs

Stage files (one result file each, run-once), proposed: `stage` (stream and cache the capped sets, write the faults scan counts), `faults` (H2, H3, R1, duplicates), `models` (per task, section 7.1), `gap` (C1), `verdicts` and `tables.md`
computed mechanically. **A runner `p4_n14.py` with preflight (`require_frozen(None, dataset='n14')`) does not exist yet and must be written and tested on train and synthetic data before the freeze** (it is hashed code). Hash-matched tarballs only: 9 of the 16 tarballs have no recorded sha256 match
in `MANIFEST.json` (the loader skips them), so the manifest and the inventory must be refreshed first (`scripts/p1_make_manifest.py`, `scripts/p1_n14_inventory.py`).

## 8. Pre-registered comparison C1: random-split versus design-split gap (E004)

Not a hypothesis; a measurement with a stated reading. For each task, using the ridge baseline (registered hyperparameters) and the primary metric:

* **C1a (train only, no test data).** Within the capped train set (4 designs): 5-fold random sample-level cross-validation versus leave-one-design-out (4 folds). Skill = Gain of ridge over `scalar` (the E004 definition, but on the task's primary metric); gap = Skill_random - Skill_design. E004 code: `_split_gap`, with `gap_max_samples` = 979 and `gap_folds` = 5 (the code uses `min(folds, designs)`).
* **C1b (reads test; run in the registered stage).** Pool train and test (capped), draw a random sample-level split with the **same train fraction as the official split** (about 66 % / 34 % of the pooled rows, 6,790 of 10,334), 5 repetitions with seeds 0 to 4; gap = Skill_random (mean) - Skill_official (ridge fitted on official train, scored on official test), design-macro on the official side, sample-mean on the random side (a random split has no design unit).
* **Reading, fixed now:** the random split is called **materially optimistic** if the C1b gap is >= 0.10 in absolute Skill on at least 2 of 3 tasks (0.10 is a pragmatic size, about half the dev ridge-versus-GBM NRMS-skill spread on IR drop, 0.22) **[DECIDE: 0.10]**. C1a is reported as the train-only counterpart and cannot change that reading.
  Because RISCY and RISCY-FPU are siblings, the leave-one-design-out folds of C1a are themselves partly leaky; the report says so.

## 9. Comparison table (copied from `docs/published_numbers.md` at the freeze)

To be completed at the freeze from sections 2.1 to 2.4 of `docs/published_numbers.md`; the rows in section 6 above are the ones that enter H1. Three facts to carry in the report (`published_numbers.md` sections 4 and 5.2):
no N14 split is published with agreed counts (official files 6,790 / 3,544; GIF 10,444 with test zero-riscy 3,456; CLDRoute 10,368 / 169 / 250); GIF's Table 4 and Table E.1 disagree with each other; and published N14 numbers are much weaker than N28 ones (SSIM 0.56 to 0.77 versus 0.78 to 0.95).

## 10. Runtime budget (estimate from dev; to be replaced by measured values in the result files)

Constraints: at most 2 CPU jobs, tarballs streamed one design at a time (never extracted wholesale), data read-only.

* **Dev measurements (train subset, 40 per design, 2 designs):** lint wall-clock 20.5 s (IR, 24 channels), 10.4 s (DRC, 9 ch), 4.3 s (congestion, 3 ch), including tarball streaming (about 130 MB/s with the large-buffer gzip reader, `circuitnet.py` line 240) and resizing;
  dev baselines on IR (40 train samples, 256 x 256): fit seconds mean_map 2.1, scalar 1.4, kNN 2.2, ridge 13.3, GBM 16.1; predict seconds per sample ridge 0.13, GBM 0.51, kNN 0.027, mean_map 0.003.
* **Extrapolation, uncapped (Option A):** staging and the full fault scan about 30 to 80 min for the three tasks (76.6 GB of tarballs, at least 10 min of pure read), cache 104 GB; models on IR about 2 h (ridge and GBM feature passes about 0.33 s per train sample, i.e. 37 min each at 6,790 samples, GBM predict 30 min), DRC about 50 min, congestion about 20 min; total about 3 to 4 h of CPU, 2 to 3 h wall with 2 jobs.
* **With K = 400 (proposal):** staging for model fitting about 10 to 20 min, models about 40 min in total, C1 about 20 min, hyperparameter CV (section 5.2, K_cv = 100) about 1 to 1.5 h on IR and 30 min on the others (pre-freeze), cache about 15 GB. Registered stages together about 1.5 to 2.5 h wall with 2 jobs; the uncapped fault scan (reading and checking all 10,334 rows per task) dominates and can run on the second job.
* These are linear extrapolations from 40-sample runs; the true cost may differ by a factor of 2. Machine and timings are recorded in each result file (provenance).

## 11. Run-once, crash and post-hoc rules

Identical to ICCAD'23 section 9, with `protocol/FROZEN_n14.json` in place of `protocol/FROZEN.json`: each stage requires a matching freeze record and `src`, `protocol`, `scripts`, `tests` matching HEAD, writes with exclusive creation, refuses to overwrite, and is never re-run because of its content; a crash before the file is written may be re-launched once and is logged in section 12;
anything not specified here is POST HOC, labelled, written to `results/posthoc/`, and cannot change a section 7 or 8 verdict. No criterion, threshold, baseline, cap or hyperparameter changes after the freeze commit. At most 2 CPU jobs at a time. Stop processes by PID only.

## 12. Freeze and deviation log

* Freeze (to happen): a commit adding `protocol/FROZEN_n14.json` with `"dataset": "n14"`, the code hash, the sha256 of this file (CRLF normalised), the date and the frozen commit. **The code hash is not the ICCAD'23 one**: the ICCAD'23 results stay tied to their recorded hash (750d252f..., code at 7707b20);
  the post hoc guard change (e541efe) already moved the live code hash away from it. Any further `src/` or `tests/` change before the N14 freeze (the `p4_n14.py` runner, possibly E004 family rules) is made before it and recorded in the freeze commit message.
* Deviations after the freeze: none yet.

## 13. Open questions the lead must decide before the freeze

1. **Sample cap K** (section 5.3): 400 per design (proposed) or no cap; and whether the cap applies to scoring on test as well as to fitting.
2. **Verdict basis** (section 3): design-macro (proposed, n = 3) or sample-pooled as the deciding mean.
3. **H1 aggregation and line** (7.2): "at least 2 of 3 tasks" and 0.8, or all 3, or a different line given that the references are weak.
4. **Primary metrics and references** (section 6): NRMS/congestion (reference 0.020 from a GNN), SSIM/DRC (CLDRoute, other split), ROC-AUC/IR (MAVIREC, unknown threshold); or restrict H1 to the tasks whose reference is on the same metric family and declare DRC and IR "not evaluable".
5. **Y for H2** (1 % pooled, 5 % in any single test design) and the separate 5 % for H3; and whether empty-hotspot (c) belongs in H2 (as specified) or only in H3.
6. **R1 threshold** (clip: material if >= 50 % of test samples have raw max above 50) and the added DRC clip-200 probe.
7. **C1 reading** (gap >= 0.10 on 2 of 3 tasks) and whether C1b, which reads the test split for a random split, is wanted at all.
8. **Hyperparameters** (5.2): run an N14 train-only CV (about 2 h) or reuse the ICCAD'23 values.
9. **E004 family rule** for RISCY / RISCY-FPU / zero-riscy: add it to code before the freeze or keep it as prose.
10. **Prerequisites that block the freeze (work, not decisions):** refresh `manifests/MANIFEST.json` sha256 matches for the 9 unverified tarballs (about 10 minutes of hashing; includes the test tarballs' bytes but no analysis); refresh `n14_inventory.json` for all 16; write and test `p4_n14.py`; confirm the cache budget on E:. 
11. **GIF claim:** accept that the map-level proxy can only confirm, never refute, the all-zero-bbox claim (7.2 b), or download `graph_features` for the three test designs (size not yet checked) to look at the object GIF named.
12. **nvdla-large and the unlisted samples** (inventory, routability: 27 of Vortex-large's 73 DRC labels and 40 of openc910-1's 96 are in neither CSV): excluded as in the authors' pipeline; say so in the report, or add an exploratory, labelled post hoc pass on them.
