# P9 design: split-leakage / test-set-tuning audit and UQ / OOD calibration baselines

Status: design, written 2026-11-05 before any P9 computation on any split. The binding pre-registration is
`protocol/protocol_p9.md`; where the two differ, the protocol wins. Scoop check: `docs/scoop_check_leakage_uq.md`.

## 1. Inventory: what exists, what is new

| need | already in the repo (not redone) | new in P9 |
|---|---|---|
| exact duplicates across splits | E003 exact (byte hash of inputs / labels, per channel); N14 registered `faults` stage hashed **every** member of every listed row in the raw archives (`results/n14/faults.json` `duplicates`: 3 cross-split `macro_region` groups on congestion/DRC, within-test exact IR maps); ICCAD'23 H3: 0 exact | nothing; P9 cites these. |
| near duplicates | E003 near (Pearson >= 0.999 of 32x32 z-scored descriptors, inputs AND labels, all pairs, <= 800 samples per split; A3/A7 fix merged); ICCAD'23 registered: 0 near pairs over 120 cases; N14 registered run did **not** run near-duplicate scans on test rows | **L001**: nearest-neighbour (NN) similarity of every test row to every cached train row (all three N14 tasks, ICCAD'23), the full NN distribution (not only a 0.999 count), a full-resolution identity confirmation of flagged pairs, sensitivity thresholds 0.99 / 0.95 |
| design-level leakage | E004 design overlap, twins, related families as `info`; N14 C1 random-vs-design gap (not material, 0 of 3 tasks) | **L002**: family proximity: are test rows of a family-related test design (zero-riscy ~ RISCY/RISCY-FPU, Vortex-large ~ Vortex-small) closer to train rows than unrelated test designs, measured against the within-design "placement variant" similarity of train designs; **L003**: design-overlap check of *published* split files (official N14/N28; CLDRoute N14/N28 CSVs) |
| test-set tuning in recipes | P2 `docs/published_numbers.md` records each paper's validation/selection text; P7 protocol records the CircuitNet `train.py` (no validation, last checkpoint) | **L004**: a tagged recipe review (code read where it exists) with pre-defined evidence classes T0-T3/U, `docs/p9_recipe_audit.md` |
| uncertainty / OOD | nothing (point baselines only) | **U-module**: bootstrap ensembles of the zoo's ridge and GBM, ensemble-Gaussian intervals, split-conformal intervals by conformal risk control (absolute and ensemble-normalised scores), in-distribution vs cross-design calibration, metrics: coverage vs nominal, width, interval calibration error, high-value-pixel coverage, OOD degradation |

**Data size needed: no download.** Everything runs on data already on E:.
* N14: `E:\edalint_data\n14_registered_cache\{ir_drop,congestion,drc}\authors\...` holds the authors-space (256 x 256, float32) members of
  the registered fit sample (979 train rows per task: RISCY 400, RISCY-FPU 400, Vortex-small 95, nvdla-small 84) and every official
  test row (3,544 / 3,544 / 3,543). Produced by the registered N14 run (code hash `d3f6d2f6...`) from tarballs whose sha256 is in
  `manifests/MANIFEST.json`. P9 hashes every cache file it reads (sha256, inline while loading) and records a per-design digest.
* ICCAD'23: `E:\edalint_data\iccad23\repo` (sha256 per file in the manifest).
* Not used, documented instead: a near-duplicate scan against **all** 6,790 N14 train rows (not just the 979 cached) needs a
  descriptor-only streaming pass over the train tarballs (IR RISCY + RISCY-FPU 25 GB, routability 11 GB; on disk already, no download),
  about 2-3 h CPU. Staging them as maps would need ~35 GB on E: (43 GB free): rejected. With the 979-row subset, L001 counts are
  **lower bounds** on test rows with a train near-duplicate, and NN similarities are lower bounds on the full-train NN similarity.
* N28 (not obtained, data card 3) is out of scope; L003 checks the N28 split *files* only (metadata).

## 2. Module layout and how it plugs in

```
src/edalint_p9/            outside src/edalint, so edalint.protocol.code_hash() (bound by CODE_HASH_LINEAGE.json) does not move
  guard.py                 p9_code_hash (src/edalint_p9/**/*.py + scripts/p9_run_*.py), check_p9_freeze, the ONLY test openers
  data.py                  N14CacheView: authors-space samples from the registered DirStore cache (edalint.datasets.circuitnet
                           DirStore + assemble, imported, not copied), sha256 of every file read; ICCAD'23 view via ICCAD23Dataset
  leakage.py               descriptors (= E003 descriptor), NN similarity, identity confirmation, family proximity, split-file checks
  uq.py                    BootstrapEnsemble (wraps any zoo Baseline factory), interval methods, CRC calibration, metrics
  checks.py                L001-L003 as linter checks `check(ds, opt) -> CheckResult`; P9_CHECKS registry and all_checks()
scripts/p9_run_leakage.py  registered runner (run-once, refuses to overwrite, requires FROZEN_p9.json)
scripts/p9_run_uq.py       registered runner (same rules)
scripts/p9_dev_*.py        train-only development (outside the hash, like p7_dev_*)
tests/test_p9_*.py         synthetic tests for every new check and UQ component
results/p9/                registered outputs; results/dev/*_p9dev.json for train-only development
```

* **Check registry.** `edalint.checks.CHECKS` is a dict in the hashed package; adding to it would move `code_hash()` (A6/A7 history).
  P9 keeps `P9_CHECKS = {'L001', 'L002', 'L003'}` in `edalint_p9.checks` with the same signature and `CheckResult` type, and
  `all_checks()` returns `{**CHECKS, **P9_CHECKS}`. Merging into `edalint.checks` is an owner decision (like A7) and is not done.
* **Baseline zoo.** The UQ layer wraps zoo classes, it does not subclass them: `BootstrapEnsemble(factory, M)` calls
  `factory()` M times (factory = `p4_n14.make_baseline(name, task)` or the ICCAD'23 equivalent, i.e. the **registered**
  hyperparameters), fits member m on a bootstrap resample of training maps and sets the member's pixel-subsample seed to m.
  `predict_members(sample) -> (M, H, W)`. Any `Baseline` with `fit/predict` works, so a future learned model drops in.
* **Guard.** `open_n14_test(task)` / `open_iccad_hidden()` in `guard.py` are the only functions that read test rows. They call
  `check_p9_freeze`: `protocol/FROZEN_p9.json` exists, `dataset == 'p9'`, `code_hash` = current `edalint` hash (`a9900e6f...`),
  `p9_code_hash` = hash now. The N14 and ICCAD'23 dataset guards cannot be used for this: FROZEN_n14.json / FROZEN.json bind older
  hashes (A7) and must not be rewritten. A static test asserts that no other P9 file opens test data.

Data flow (leakage): cache files -> sha256 + `assemble(task, 'authors')` -> per-channel 32 x 32 z-scored descriptor (the E003
descriptor) -> NN search test-vs-train (and train-vs-train within design, leave-one-out) -> flagged pairs -> full-resolution identity
check -> L001/L002 records. Data flow (UQ): fit rows -> M bootstrap members -> predictions on calibration maps -> nonconformity
scores -> lambda by CRC -> intervals on ID-test and test designs -> per-map coverage/width -> per-design and design-macro summaries.

## 3. Decisions, alternatives, reasons

**D1 Near-duplicate metric: mean-over-varying-channels Pearson correlation of 32 x 32 area-resized maps (the E003 descriptor).**
* Chosen because (i) it is the check P8 validated by fault injection (near copies detected once the pair correlation is >= 0.999-0.9995,
  0 false positives on ICCAD'23/N14), so its operating characteristics are known; (ii) it is amplitude- and offset-invariant per channel,
  which matches the authors' per-sample min-max scaling (two layouts that differ only by a global power scale are the same "layout");
  (iii) it is cheap: 4,523 x 979 x 24 channels x 1,024 is one matrix product per task.
* Rejected: **perceptual hash** (64-bit DCT pHash): built for natural photographs, binarises at the median, so sparse maps (macro_region,
  DRC-like, mostly-zero RUDY) collapse to near-identical hashes; its Hamming threshold has no validation on these maps.
  **Embedding distance** (ImageNet CNN or a self-trained autoencoder): no pretrained model for EDA maps exists, an ImageNet embedding
  is not justified for 24-channel power stacks, a trained one adds a model whose own generalisation would be in question, and no GPU.
  **Full-resolution NCC as the screen**: 64x more compute for 4.4M pairs per task; used only for confirmation (D3).
* Channels: all input channels incl. the 20 `power_t` slices (as E003). Labels are compared separately.

**D2 Thresholds.** Primary 0.999 on inputs AND label (= E003 `dup_corr`, unchanged, so P9 and the linter agree). Sensitivity: 0.99 and
0.95 (reported, not decisive). Input-only near duplicates (input >= 0.999, label < 0.999) are reported separately: same layout,
different outcome is itself informative (e.g. the IR within-test exact pairs of `faults.json`).
A fixed threshold, not a percentile of the NN distribution: a percentile always flags something and would turn L001 into a ranking.

**D3 Confirmation (adjudication rule).** A flagged pair is a *confirmed* near duplicate only if, at full 256 x 256 authors-space
resolution, every input channel and the label differ by at most 0.01 (max abs, maps are min-max scaled to [0, 1]). This is the
pre-stated identity check required by PLAN.md; a flag that fails it stays "flagged, unconfirmed" and is not a fault.

**D4 Design-level leakage (L002).** Reference = within-design NN similarity: for each train row, its NN among *other rows of the same
design* (placement / utilisation / power-mesh variants of one netlist). This is the similarity scale of "same design, different
placement". For each test design: S_same = median NN similarity to rows of same-family train designs, S_other = to other families.
Pre-stated reading: design-level proximity is *material* for a test design if S_same >= the 25th percentile of the within-design
reference AND S_same - S_other >= 0.02. Family relations are documented (data card 2, E004), so a material proximity on a
documented family pair is adjudicated by docs + the pre-stated rule; on an undocumented pair it stays provisional.
Rejected: comparing netlists (raw_data not downloaded, 300+ GB); a classifier "train vs test" (needs a model, harder to read).

**D5 Published splits (L003).** Metadata only: design name parsed from each row; exact design-name overlap between train / val /
test = confirmed (identity check on names); family overlap = info.

**D6 Test-set tuning (L004).** Evidence classes fixed in advance: T0 fixed budget, last checkpoint, no validation; T1 selection on a
validation set disjoint from test; T2 test metrics computed or logged during training (exposure), selection on validation; T3 selection
or hyperparameters chosen on test (stated in text or visible in code); U unknown. Only T3 counts as test-set tuning (a fault, confirmed
by the code/text); T2 is reported as exposure risk, never as a fault. Each entry is tagged code / README / PDF / abstract.

**D7 UQ base models: bootstrap ensembles (M = 5) of the zoo's ridge and GBM with the registered hyperparameters.**
Rejected: MC dropout / deep ensembles of the authors' CNNs (no GPU; P7 MAVI needed Kaggle T4 for 3 seeds); quantile GBM (a different
model per alpha, not a wrapper on the existing zoo); kNN (its spread across bootstrap members is degenerate when k = 1, IR drop).
M = 5: smallest common deep-ensemble size; members differ by map bootstrap and pixel-subsample seed.

**D8 Conformal method: conformal risk control (CRC, Angelopoulos et al. 2022) with loss = fraction of miscovered pixels in a map.**
Pixels within a map are not exchangeable, maps are. CRC on per-map miscoverage gives E[miscovered fraction of a new exchangeable map]
<= alpha with a finite-sample guarantee: lambda = inf{l : n/(n+1) R_n(l) + 1/(n+1) <= alpha}, R_n = mean over calibration maps.
Scores: (a) absolute |y - mu| (constant width), (b) normalised |y - mu| / (sigma_ens + eps), eps = 0.05 x mean sigma_ens over the
calibration pixels. Also reported: the uncalibrated ensemble-Gaussian interval mu +/- z sigma_ens.
Rejected: pooled-pixel split conformal (treats ~65k dependent pixels per map as exchangeable: the nominal guarantee is wrong); per-map
max score (simultaneous band, far too wide to be useful for maps); CQR (needs quantile regressors, see D7).

**D9 Calibration regimes and splits (cross-design).** N14 (cached rows only): fit F = 300 RISCY + 300 RISCY-FPU (seeded); the other
100 + 100 split into ID calibration C_id (50 + 50) and ID test T_id (50 + 50); cross-design calibration C_xd = Vortex-small + nvdla-small
(train designs never fitted). Test designs: zero-riscy (same family as F), Vortex-large (family of C_xd's Vortex-small), openc910-1
(unrelated). ICCAD'23: fit 70 fake, C_id 15 fake, T_id 15 fake, C_xd = the 10 visible real (5 designs), test = 10 hidden real.
Two regimes: "naive" (C_id) and "cross-design" (C_xd). OOD degradation = coverage(test design) - coverage(T_id), same calibration.

**D10 Tasks.** Leakage: IR drop, congestion, DRC. UQ: IR drop and congestion (N14) and ICCAD'23 IR drop. DRC is excluded from UQ in
advance: 99.9 % of test rows have no positive pixel at 0.1 and 7.8 % an all-zero label (REPORT_n14 H3), so pixel coverage is dominated
by zeros and says nothing about DRC hotspots.

**D11 Metrics.** Per map: pixel coverage, mean width, coverage on high-value pixels (label >= the map's 90th percentile).
Per design: mean over maps; design-macro over test designs (as in N14). Interval calibration error = mean over nominal levels
{0.5, 0.6, 0.7, 0.8, 0.9, 0.95} of |coverage - nominal| (each level calibrated separately). Width also divided by the design's label std.

## 4. Risks
* **Subset lower bound.** Near-duplicate counts use 979 of 6,790 train rows: an absent finding is weak evidence; a present one is real.
* **Cache provenance.** Cache files were written by the registered N14 code (`d3f6d2f6`), not re-derived; P9 hashes them but cannot
  re-verify them against the tarballs without restaging. Mitigation: digest recorded; a spot re-stage of a few rows from the tarballs in dev.
* **Test-set exposure.** The author of P9 knows the registered N14/ICCAD'23 point results (disclosed in the protocol). No UQ quantity on
  any test row is known before the freeze.
* **Weak models.** Ridge/GBM intervals describe these baselines, not deep models; conclusions about "UQ for IR-drop predictors" are
  conclusions about the coverage machinery under design shift, and must be framed that way.
* **ICCAD n.** C_xd has 10 maps: CRC needs alpha >= 1/11; at alpha = 0.1 the empirical risk must be <= 0.01, so intervals are wide.
* **Runtime.** GBM ensembles on 3,441 zero-riscy maps: timed in dev; if over budget, a seeded subsample is fixed in the protocol
  before the freeze (never after a test number).
* **Freeze drift.** Any edit of src/edalint_p9 or scripts/p9_run_*.py after the freeze invalidates it (guard refuses).
