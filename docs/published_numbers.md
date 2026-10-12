# Published numbers (Phase P2)

Compiled 2026-10-12. Every number below was read from the primary PDF, the official xlsx, or the official
repository files, unless the row is tagged UNVERIFIED. "VERIFIED (PDF)" means the value was read from the paper's own
table or text (extracted with `pdftotext -layout` and, for garbled tables, `pdfplumber` word positions). "VERIFIED
(xlsx)" means read from `doc/ICCAD-Final-Scores-Release.xlsx` in the ASU contest repository (local copy
`E:\edalint_data\iccad23\repo`). "VERIFIED (repo)" means read from a file in a public code repository.

This file records what was published. It does not decide the audit protocol.

## 0. Sources

| Short name | arXiv / source | Venue | Status |
|---|---|---|---|
| Contest description | `doc/contest-description.pdf` (ASU-VDA-Lab/ML-for-IR-drop), 4th version 2023-06-22 | ICCAD 2023 CAD Contest, Problem C | read in full |
| Contest slides | `doc/ICCAD23-Contest-ProblemC.pdf` | ICCAD 2023 | read (results, ranking, scoring) |
| Contest scores | `doc/ICCAD-Final-Scores-Release.xlsx` | ICCAD 2023 | parsed, all sheets |
| Invited paper | `doc/invited-paper.pdf` (Kadagala and Chhabria) | ICCAD 2023 | read (metrics, scoring) |
| CFIRSTNET | 2502.12168 | ICCAD 2024 | read |
| WACA-UNet | 2507.19197 | arXiv preprint (v1, 2025-07-25) | read |
| LMM-IR | 2511.12581 (v2, 2026-06-03) | arXiv preprint | read |
| Global/Local Inception U-Net ("GL-Inception") | 2406.06541 (v2) | arXiv preprint | read |
| AttUNet | 2408.03292 (v1) | arXiv preprint | read |
| PDNNet | 2403.18569 (v2) | arXiv (IEEE journal format) | read |
| GIF | 2604.09999 (v1, 2026-04-11) | arXiv preprint | read |
| CLDRoute | 2607.16674 (v1, 2026-07-18) | arXiv preprint | read |
| CircuitNet (2022 letter) | 2208.01040 | Sci China Inf Sci, News and Views | read in full |
| CircuitNet 2.0 | ICLR 2024, `proceedings.iclr.cc/.../464917b6...-Paper-Conference.pdf` (no arXiv version found) | ICLR 2024 | read (statistics, tasks, Appendix B) |
| CircuitNet code | github.com/circuitnet/CircuitNet, `routability_ir_drop_prediction/` (`files/*.csv`, `utils/configs.py`, `utils/metrics.py`) | code | read |

The venue of the arXiv-only entries was taken from the arXiv header and the paper body. None of the entries beyond
CFIRSTNET, CircuitNet 2.0 and the contest documents states a peer-reviewed venue in the PDF.

---

## 1. ICCAD 2023 Problem C (static IR drop, NanGate45)

### 1.1 Task, data and metric definitions (quoted from the contest documents)

- Hidden ("test") set: 10 real-circuit testcases, numbered **7, 8, 9, 10, 13, 14, 15, 16, 19, 20**.
  VERIFIED (xlsx column headers; repo `benchmarks/hidden-real-circuit-data/` contains exactly testcase7, 8, 9, 10,
  13, 14, 15, 16, 19, 20 plus `list_testcase.txt`).
- Visible real training cases: the other 10 of the 20 real circuits (repo `benchmarks/real-circuit-data/`, 10
  entries). Fake (BeGAN) training cases: the repo has 500 entries in `benchmarks/fake-circuit-data/`. The contest
  documents and four papers (GL-Inception, AttUNet, LMM-IR, WACA) say "100 fake cases". The 500 is a directory entry count
  and is not reconciled here (P1 should confirm the number of fake cases and files per case).
- Resolution: 1 um x 1 um per pixel. Grid size varies per case (80x80 to 1000x1000 per AttUNet). Label is the IR-drop
  map (volts) of the lowest-metal-layer nodes.
- MAE (contest description, section 4a): "The average of the absolute difference between a prediction and the actual
  value, calculated for each example in a dataset." Worked example in the document: |diff| matrix with entries
  0.2e-3, 0.6e-3, 0, 1e-3 gives "MAE = 0.45mV". So **MAE is the per-pixel mean of |pred - golden| in mV (xlsx stores
  volts; e.g. winner mean 1.3473e-4 V = 0.1347 mV)**. It is not normalised.
- F1 (contest description, section 4b): "A binary classification metric that uses 10% of the maximum IR drop of each
  benchmark as the threshold ... The positive class is nodes with the top 10% of IR drop. ... An element in the
  ir_drop_map.csv is considered as an IR drop hotspot if its value is greater than 90% of the maximum ir drop of that
  testcase." Worked example: threshold = 0.9 x 5.93 mV = 5.337 mV, applied to the golden map. `F1 = 2 P R / (P + R)`,
  `P = TP/(TP+FP)`, `R = TP/(TP+FN)`. The same sentence says "top 10% of IR drop" and "> 90% of the maximum"; these
  are different rules (a pixel-fraction rule versus a value-fraction rule). The worked example and the slides
  ("10% of the worst-case IR drop in the golden data") support the value-fraction rule. The threshold is applied to the
  golden map; the contest text does not say explicitly whether the same absolute threshold is applied to the
  prediction. Edge case: if the prediction has no pixel above the threshold, P is undefined and the xlsx records F1 = 0
  (see the winner on testcases 13 and 14).
- Runtime (description, 4c; slides): "Inference time of the ML model" / "for a single datapoint". Hardware of the
  organisers' evaluation machine is not stated in the documents read.
- Score: 60% MAE, 30% F1, 10% runtime per hidden testcase; best team gets 60/30/10 points, others scaled
  ("relative scoring", "scaled based on a normal distribution" in the description; "relative to the maximum metric of
  the testcase" on the slides). Sum over 10 testcases, maximum 1000.
- Contest scale: 72 registered, 24 alpha, 25 beta, 27 final submissions (slides).

### 1.2 Official results (VERIFIED (xlsx) and VERIFIED (slides: final scores and rank))

Winners per slides: 1st = cadc0009 (National Yang Ming Chiao Tung University), 2nd = cadc1023 (Fuzhou and Fudan
University team), 3rd = cadc1020 (IDPM RAS and NRU MIET). Final scores (slides, reproduced from the xlsx score
columns by summing per-case points): cadc0009 688.84, cadc1023 634.19, cadc1020 596.17, cadc1032 443.65, cadc0007 351.88.

Per-testcase MAE (mV), F1, and runtime (s). Testcases in the order 7, 8, 9, 10, 13, 14, 15, 16, 19, 20.

| Entry | MAE (mV) | F1 | Runtime (s) |
|---|---|---|---|
| 1st (cadc0009) | 0.0656, 0.0815, 0.0406, 0.0659, 0.2068, 0.4215, 0.0968, 0.1601, 0.0905, 0.1180 | 0.7828, 0.8164, 0.5889, 0.5320, 0, 0, 0.0877, 0.5286, 0.5014, 0.7111 | 14.61, 12.64, 18.84, 19.05, 9.60, 10.07, 12.99, 12.12, 19.05, 18.75 |
| 2nd (cadc1023) | 0.0776, 0.1132, 0.0727, 0.1142, 0.1255, 0.2321, 0.1920, 0.3440, 0.1202, 0.1069 | 0.5604, 0.8022, 0.5505, 0.1490, 0.6735, 0.0984, 0, 0.4808, 0.4942, 0.7413 | 3.22, 2.70, 4.25, 4.13, 1.25, 1.40, 2.15, 2.19, 4.55, 4.58 |
| 3rd (cadc1020) | 0.0749, 0.1141, 0.0512, 0.1003, 0.0901, 0.2115, 0.0521, 0.1609, FAIL, FAIL | 0.8657, 0.8742, 0.4720, 0.6468, 0, 0, 0.7363, 0.5478, 0 (failed), 0 (failed) | no runtime recorded (sentinel 1e+99 on every case; "PY" label in the sheet) |

Averages as they appear in the xlsx and as recomputed:

| Entry | Mean MAE (mV) | Mean F1 | Mean runtime (s) | Note |
|---|---|---|---|---|
| 1st | 0.1347 | 0.4549 | 14.772 | all 10 cases valid |
| 2nd | 0.1498 | 0.4550 | 3.042 | all 10 cases valid |
| 3rd | xlsx "inf" (failed on 19, 20); 0.1069 over the 8 valid cases | xlsx 0.4143 over 10 cases (zeros for 19, 20); 0.5186 over the 8 valid cases | none | the 0.107 / 0.52 quoted in GL-Inception are the 8-case means |

The xlsx also lists 22 further teams. Next best by rank, mean over the 10 cases where valid (MAE mV / F1 / runtime s):
4th cadc1032 0.2380 / 0.3576 / 7.10; 5th cadc0007 0.3161 / 0.0895 / 2.05; 6th cadc0025 0.2944 / 0.2589 / 46.17;
7th cadc0022 0.3869 / 0.2239 / 16.71; 8th cadc1067 (failed on 3 cases) / 0.1634; 9th cadc0001 0.3520 / 0.1280 / 21.77;
10th cadc0008 0.3649 / 0.0989 / no runtime. Several teams have F1 = 0 on most cases.

Scale of the hidden labels (CFIRSTNET Table 2, HSPICE columns, VERIFIED (PDF)): mean IR drop per case 0.44 to 4.65 mV
(average 1.98 mV), maximum 1.72 to 13.15 mV (average 5.87 mV), hotspot fraction 0.010% to 0.535% of pixels
(average 0.168%). A constant-zero prediction therefore has MAE equal to the mean IR drop, 1.98 mV on average. This is
quoted for context only; no baseline was computed.

### 1.3 Published learned-model results on the hidden 10 (per paper)

Common to all rows: test set = the 10 hidden real cases. Technology NanGate45. "Val" = the 10 visible real cases unless
stated. Per-case values are in section 1.4.

| Paper (arXiv, venue) | Split / training data | Metric as stated | Mean MAE (mV) | Mean F1 | Runtime and hardware | Code | Tag |
|---|---|---|---|---|---|---|---|
| CFIRSTNET (2502.12168, ICCAD 2024) | train: 6,100 synthetic BeGAN; val: 10 real (visible); test: 10 real hidden. "synthetic circuits are assigned to the training set, and the real circuits are distributed to the validation set and testing set equally". No real circuit is in the training set. Model with lowest validation loss is saved. Inputs resized to 384x384 | MAE: "sum of absolute errors (Manhattan distance) divided by the sample size", eq. (11). F1: hotspot "IR drop larger than 90% of the testcase [max]", eq. (12) `2TP/(2TP+FP+FN)` | **0.0533** (per-case mean recomputed 0.05334) | **0.723** | **0.409 s** per case (per-case mean 0.4086). Xeon E3-1230 v6 CPU server with NVIDIA Quadro P4000; the text does not say whether inference ran on the GPU. HSPICE 2.81 to 23.33 s per case, average 12.2 s | github.com/jason122490/CFIRSTNET (link as printed in the paper; not tested) | VERIFIED (PDF) |
| WACA-UNet (2507.19197, preprint) | train: 2,100 = 2,000 BeGAN (nangate45_set1, set2) + 100 contest synthetic; val: 10 visible real; test: 10 hidden. Checkpoint chosen by best **validation F1**. 384x384 Lanczos resize | MAE eq. (10) mean over pixels; F1 eq. (11), hotspot `y_i >= 0.9 y_max` "following the same method in [1]" | **0.0524** (recomputed 0.05238) | **0.778** (recomputed 0.7782) | **1.390 s** mean, "measured on CPU" (CPU model not stated). Training on one RTX A6000 | none stated in the text | VERIFIED (PDF) |
| LMM-IR (2511.12581, preprint) | train: 100 fake x10 oversample + 10 real x20 + 2,000 BeGAN, "total of 3310 cases" (components give 3,200, see inconsistency 5); test: 10 hidden. 512x512 inputs | F1 "Definition 1", MAE "Definition 2" (same as contest); MAE reported in units of 1e-4 V | **0.1356 mV** (table prints 1.35 x 1e-4 V; per-case mean 1.356) | **0.583** (printed 0.58) | TAT 3.05 s (per-case mean 3.055). NVIDIA H100 80GB platform | none found in the text | VERIFIED (PDF) |
| GL-Inception (2406.06541, preprint) | 2,000 BeGAN + 100 fake x10 + real x20; 5 visible real cases held out as validation; "the training set comprises a total of 3100 cases" (2,000 + 1,000 + 100 = 3,100, consistent). Best-validation checkpoint | MAE eq. (1) mean over points, F1 eq. (2)-(4): "IR drop values exceeding 90% of the maximum ground truth are classified as positive" | **0.101** (recomputed 0.1012) | **0.62** (recomputed 0.615) | 3.37 s total (feature extraction + inference). Training: one NVIDIA A40; test: "Ubuntu 20.04 with a 16GB 2.42GHz Intel Core" | none found in the text | VERIFIED (PDF) |
| AttUNet (2408.03292, preprint) | pretrain on 100 fake (augmented), finetune on the 10 visible real (augmented), test on the 10 hidden. "800 pretrain data points and 80 fine-tune data points" after "6 fold" augmentation (see inconsistency 5) | MAE: mean |V^ - V| over N pixels. F1: "the highest 10% IR drops are labeled as positive and the rest as negative" (a top-10%-of-pixels rule, different from the contest rule) | printed **0.110**; per-case mean 0.1172 (see inconsistency 6) | **0.520** (recomputed 0.5198) | 5.37 s average inference; hardware for inference not stated; training 2x RTX 2080 Ti | none found in the text | VERIFIED (PDF) |

Sub-points on CFIRSTNET's ablation (Table 4, VERIFIED (PDF)): the same network with only the contest image maps
("Image-based") gives MAE 0.1864 mV, F1 0.217; adding the physics-derived "Hypothetical IR Drop Distillation" (HIRD)
feature gives 0.0795 mV, F1 0.677; all features 0.0559; final "comprehensive" set 0.0533, F1 0.723. HIRD is a
superposition-based feature computed from the SPICE netlist, i.e. a physics input to the network, not a standalone
baseline.

### 1.4 Per-case table (hidden 10; MAE in mV, F1)

| Testcase | 1st MAE / F1 (official) | CFIRSTNET MAE / F1 / s | WACA MAE / F1 / s | GL-Inception MAE / F1 | LMM-IR MAE / F1 | AttUNet MAE / F1 |
|---|---|---|---|---|---|---|
| 7 | 0.0656 / 0.783 | 0.0177 / 0.923 / 0.366 | 0.0147 / 0.953 / 1.384 | 0.081 / 0.69 | 0.063 / 0.72 | 0.065 / 0.469 |
| 8 | 0.0815 / 0.816 | 0.0257 / 0.916 / 0.367 | 0.0215 / 0.964 / 1.447 | 0.104 / 0.63 | 0.084 / 0.84 | 0.081 / 0.428 |
| 9 | 0.0406 / 0.589 | 0.0278 / 0.526 / 0.572 | 0.0241 / 0.721 / 1.408 | 0.057 / 0.72 | 0.042 / 0.47 | 0.067 / 0.169 |
| 10 | 0.0659 / 0.532 | 0.0459 / 0.464 / 0.551 | 0.0402 / 0.575 / 1.327 | 0.088 / 0.34 | 0.071 / 0.60 | 0.090 / 0.536 |
| 13 | 0.2068 / 0 | 0.0774 / 0.680 / 0.204 | 0.0861 / 0.561 / 1.346 | 0.121 / 0.68 | 0.152 / 0.52 | 0.188 / 0.895 |
| 14 | 0.4215 / 0 | 0.1895 / 0.678 / 0.203 | 0.1518 / 0.667 / 1.488 | 0.220 / 0.43 | 0.324 / 0.44 | 0.078 / 0.881 |
| 15 | 0.0968 / 0.088 | 0.0353 / 0.733 / 0.327 | 0.0549 / 0.749 / 1.359 | 0.066 / 0.76 | 0.149 / 0.54 | 0.115 / 0.700 |
| 16 | 0.1601 / 0.529 | 0.0763 / 0.785 / 0.324 | 0.0734 / 0.865 / 1.308 | 0.163 / 0.40 | 0.333 / 0.55 | 0.338 / 0.634 |
| 19 | 0.0905 / 0.501 | 0.0187 / 0.752 / 0.596 | 0.0366 / 0.817 / 1.432 | 0.047 / 0.70 | 0.074 / 0.61 | 0.082 / 0.330 |
| 20 | 0.1180 / 0.711 | 0.0191 / 0.773 / 0.576 | 0.0205 / 0.910 / 1.405 | 0.065 / 0.80 | 0.064 / 0.54 | 0.068 / 0.156 |
| Mean | 0.1347 / 0.455 | 0.0533 / 0.723 | 0.0524 / 0.778 | 0.101 / 0.62 | 0.136 / 0.58 | 0.117 (printed 0.110) / 0.520 |

The "1st" column equals the official xlsx values to the printed precision in CFIRSTNET Table 2, WACA Table 1, AttUNet
Table 2 and GL-Inception Table I (and LMM-IR Table III for MAE and F1). CFIRSTNET's and WACA's 1st-place runtime
columns (7.996, 8.396, ...) are not the official runtimes (14.61, 12.64, ...); see inconsistency 3.

### 1.5 Re-implementations of IREDGe (a U-Net baseline) on the hidden 10

| Paper | MAE (mV) | F1 | Runtime (s) | Note |
|---|---|---|---|---|
| CFIRSTNET Table 2 (also copied into WACA Table 1) | 0.4566 | 0.082 | 0.166 | "reproduced with its source code from GitHub despite the discrepancy between the code and the paper" |
| GL-Inception Table I | 0.588 | 0.12 | 1.68 | "a Pytorch reimplementation by us since its data input is unsuitable for this dataset" |
| LMM-IR Table III | 0.628 | 0.13 | 2.02 | "re-implemented in PyTorch for this paper"; also IRPnet re-implementation 0.398 mV, F1 0.03 |
| AttUNet Table 2 | 0.233 | 0.244 | n/a | trained with the same pretrain/finetune stages as AttUNet |

The ranges are therefore 0.233 to 0.628 mV and F1 0.08 to 0.24 (draft confirmed). These are re-implementations, not
the original authors' numbers, and training data differs.

### 1.6 Other contest-related points

- Contest 3rd place's code was a Python script; GL-Inception states "they don't have the runtime results. Moreover,
  their codes crashed in the last two cases". The xlsx agrees (MAE and runtime sentinels on cases 19 and 20; runtime
  sentinel on all cases).
- CFIRSTNET footnote 2: "We have obtained the 1st Place's executable but that of 2nd Place is not available."

---

## 2. CircuitNet (28 nm and 14 nm)

### 2.1 Official dataset facts (VERIFIED (PDF) / VERIFIED (repo))

- CircuitNet (2022 letter, 2208.01040): 6 RISC-V designs (RISCY-a, RISCY-FPU-a, zero-riscy-a, RISCY-b, RISCY-FPU-b,
  zero-riscy-b), 28 nm, 12,960 back-end runs, **10,242 layouts** after excluding failed runs. Tasks: congestion, DRC,
  IR drop. The letter does not describe a train/test split. Its reference results (the letter, "same evaluation
  metrics as in the original studies"): congestion with a FCN (GPDL-like) NRMS 0.040, SSIM 0.80; DRC with a FCN
  (RouteNet-like) ROC-AUC 0.95, PR-AUC 0.63; IR drop with U-Net (MAVIREC) ROC-AUC 0.94, PR-AUC 0.83. Download and
  user guide: circuitnet.github.io.
- **Official split files** (github.com/circuitnet/CircuitNet, `routability_ir_drop_prediction/files/`; counts of
  lines and of design-name prefixes in the CSV, no dataset file opened):
  - `train_N28.csv`: 7,078 samples = RISCY-a 2,003 + RISCY-b 1,858 + RISCY-FPU-a 1,969 + RISCY-FPU-b 1,248.
  - `test_N28.csv`: 3,164 samples = zero-riscy-a 2,042 + zero-riscy-b 1,122.
  - This is a design-level split (6 designs; 4 train, 2 test), and `utils/configs.py` uses the same two files for the
    congestion (GPDL), DRC (RouteNet) and IR-drop (MAVIREC) tasks. There is no validation file.
  - `train_N14.csv`: 6,790 = RISCY 3,162 + RISCY-FPU 3,449 + nvdla-small 84 + Vortex-small 95.
  - `test_N14.csv`: 3,544 = zero-riscy 3,441 + openc910-1 56 + Vortex-large 47. Design-level.
  - The repository config defaults point to the N28 files only; the N14 files are present but not wired into the
    default task configs.
- Official evaluation code (`utils/metrics.py`, `utils/configs.py`): NRMS = scikit-image
  `normalized_root_mse(img1.flatten(), img2.flatten(), normalization='min-max')` per sample (the pixel-wise RMSE
  divided by the range of the first argument); SSIM own implementation; PSNR with peak 255; EMD. Default
  `--eval-metric`: congestion `['NRMS','SSIM','EMD']`, DRC `['NRMS','SSIM']` with `threshold 0.1`, IR drop
  `['NRMS','SSIM']` with `threshold 0.9885` ("5% after log"). ROC/PR curves are computed by thresholding at 200
  values in [0, 1).
- IR drop in CircuitNet N28 is **dynamic, vectorless** (PDNNet: "CircuitNet-N28 leverages Innovus and Voltus flow to
  obtain vectorless dynamic IR drop solutions"; CircuitNet 2.0 appendix: IR-Drop is "a 2D array that encodes the
  distribution of the maximum IR-drop value on the grids"). ICCAD'23 IR drop is static, from a SPICE-netlist solve.
  The two are not the same quantity.
- CircuitNet 2.0 (ICLR 2024, N14 FinFET; Table 1, VERIFIED (PDF)): samples per design RISCY 3,456; RISCY-FPU 3,456;
  zero-riscy 3,456; OpenC910-1 96; Vortex-large 74; Vortex-small 96; NVDLA-large 68; NVDLA-small 89; total stated as
  10,791. The paper gives its split for the net-delay task only (80 train samples from RISCY, RISCY-FPU, Vortex; 20 test
  samples from zero-riscy, NVDLA, OpenC910-1). For congestion and IR drop it says only that the test set contains
  unseen designs; Appendix B results (VERIFIED (PDF)): congestion FCN NRMSE 0.036 +- 0.001 / SSIM 0.872 +- 0.005 (N14)
  versus 0.047 +- 0.002 / 0.773 +- 0.006 (N28); cGAN 0.044 / 0.796 (N14), 0.055 / 0.739 (N28); Circuit GNN 0.020 / 0.870
  (N14), 0.046 / 0.761 (N28). IR drop (MAVIREC, hotspot detection with FPR <= 5%): N14 TPR 39.6 +- 1.9%, accuracy
  95.2%, ROC-AUC 0.8671; N28 TPR 29.7%, accuracy 95.4%, ROC-AUC 0.8537. No train/test counts are given for these
  experiments.

### 2.2 CircuitNet-N28 IR drop (dynamic)

| Paper | Split | Metric as stated | Numbers | Hardware / runtime | Code | Tag |
|---|---|---|---|---|---|---|
| PDNNet (2403.18569) Table III | "We follow the same protocol as CircuitNet [34]": train = RISCY-a, RISCY-b, RISCY-FPU-a, RISCY-FPU-b (7,078 samples); test = zero-riscy-a, zero-riscy-b (3,164). Table II sample counts: 2,003 / 1,858 / 1,969 / 1,248 and 2,042 / 1,122. Design-level. This matches the official `train_N28.csv` / `test_N28.csv` | NMAE = `(1/N) sum_i |y_i - y~_i| / (max(y_i) - min(y_i))` (index set of i, pixel or sample, not defined; as printed); also R2, PSNR, SSIM, Pearson, Spearman, Kendall, AUC (AUC threshold and positive class not defined) | PowerNet NMAE 0.149, R2 0.32, PSNR 11.60, SSIM 0.56, Pear 0.77, Spea 0.83, Kend 0.70, AUC 0.503. MAVIREC 0.039, 0.81, 18.27, 0.68, 0.91, 0.85, 0.77, 0.942. MAVIREC + image PDN features 0.037, 0.82, 18.55, 0.70, 0.91, 0.86, 0.78, 0.946. **PDNNet 0.028, 0.84, 19.35, 0.72, 0.92, 0.87, 0.81, 0.953** | "545x speedup compared to the commercial tool" (hardware not located in the text read). Adam, lr 0.0008, no augmentation | none found in the text | VERIFIED (PDF) |
| GIF (2604.09999) Table 6 | text (section 4.3): "under the design split defined by PDNNet ... Four designs ... training, and two unseen designs (zero-riscy-a, zero-riscy-b) ... testing". **Table 3 of the same paper shows a different split**: train RISCY-a 2,003 + RISCY-b 1,858 + RISCY-FPU-a 1,969 + zero-riscy-a 2,042; val RISCY-FPU-b 1,248; test zero-riscy-b 1,122 (total 10,242). Maps resized to 256x256 | NMAE not defined in the paper. Its own MAE: mean per-pixel `|y^ - y|` with maps "normalized to [0, 1] relative to the supply voltage". PSNR peak not defined | Table 6: PowerNet / MAVIREC / PDNNet rows are copied from PDNNet ("taken from their published performance"); **GIF NMAE 0.0266 (abstract: 0.026), PSNR 21.77, SSIM 0.786 (abstract 0.78), Pearson 0.9536**. Table 5 (own ablation): best PSNR 19.583, best SSIM 0.785, best Pearson 0.9215, MAE 0.0345 to 0.0405; the Table 6 values do not appear in Table 5 | 1,000 denoising steps per sample (no wall-clock figure located) | none found in the text | VERIFIED (PDF) |
| CLDRoute (2607.16674) | N28 routability only (see 2.4) | | | | | VERIFIED (PDF) |

Note on GIF Table 6: the baselines were evaluated on a 3,164-sample, two-design test set (PDNNet); the GIF text says it
used that split, while its own Table 3 describes a 1,122-sample, one-design test set with a different training set.
The paper does not say which set produced 0.0266.

### 2.3 CircuitNet-N14 IR drop

| Paper | Split as printed | Numbers | Tag |
|---|---|---|---|
| GIF Table 4 (design-wise) | Train: Nvidia-small 85, RISCY 3,162, RISCY-FPU 3,456, Vortex-large 61, openc910-1 96, Nvidia-large 32 (as printed, row by row); Val: Vortex-small 96; **Test: zero-riscy 3,456**; total 10,444 (the rows sum to 10,444; verified by word positions). Appendix Table E.1 ("official metadata") lists different per-design totals (RISCY 3,456; RISCY-FPU 3,456; Vortex-small 74; nvdia-large 96; nvdia-small 68; openc910-1 93; zero-riscy 96; an unlabelled row of 3,456; total 10,795) and valid-graph counts. Table 4 and Table E.1 label the small designs differently, and CircuitNet 2.0 Table 1 gives yet different per-design counts (OpenC910-1 96, Vortex-large 74, Vortex-small 96, NVDLA-large 68, NVDLA-small 89) | Table 7, best row: PSNR 14.987, SSIM 0.558, MAE 0.0667, RMSE 0.1797, Pearson 0.9106, Spearman 0.8284 (MAE and RMSE on maps normalised to [0, 1] by the supply voltage). Claims "first work to report IR-drop results on CircuitNet-N14". Appendix E: "CircuitNet-N14 contains a large number of placement snapshots whose instance bounding boxes are all zero" | VERIFIED (PDF) |
| Official repo `train_N14.csv` / `test_N14.csv` | train 6,790 (RISCY 3,162, RISCY-FPU 3,449, nvdla-small 84, Vortex-small 95); test 3,544 (zero-riscy 3,441, openc910-1 56, Vortex-large 47) | no published IR-drop numbers for this split found in the papers read | VERIFIED (repo) |
| CircuitNet 2.0 Appendix B.2 | split not stated | MAVIREC hotspot detection: TPR 39.6 +- 1.9% at FPR 4.7%, accuracy 95.2%, ROC-AUC 0.8671 +- 0.0032 | VERIFIED (PDF) |

### 2.4 CircuitNet congestion and DRC

| Paper | Split | Task and metric as stated | Numbers | Hardware | Code | Tag |
|---|---|---|---|---|---|---|
| CLDRoute (2607.16674) Table 5 | N28: train 7,872 / val 1,248 / test 1,122. N14: 10,368 / 169 / 250 (total 10,787). "We use a design-wise split for both congestion and DRC, so that training, validation, and test sets contain disjoint designs." Test design names are not given. The N28 counts equal GIF's Table 3 counts (7,872 = 2,003 + 1,858 + 1,969 + 2,042; 1,248 = RISCY-FPU-b; 1,122 = zero-riscy-b) | NRMS, MAE, SSIM, Pearson ("MAE and NRMS measured pixel-wise reconstruction error"; NRMS formula not given); DRC also TopK@1%, TopK@0.5%, Hotspot-MAE, NZ-Pearson (correlation over non-zero ground-truth pixels), F1@0.1; congestion also Spatial Bias, uncertainty-error correlation | **N28 DRC (Table 7, 8)**: RouteNet (labelled SOTA) MAE 0.00313, NRMS 0.03300, SSIM 0.95481, Pearson 0.38156, TopK@1% 0.24164, NZ-Pearson 0.39655, F1@0.1 0.41343. CLDRoute (LDM+ControlNet, 15 ch) MAE 0.00280, NRMS 0.02893, SSIM 0.96780, Pearson 0.52483, TopK@1% 0.34940, F1@0.1 0.44203. **N28 congestion (Table 9, 10)**: GPDL (labelled SOTA) MAE 0.02958, NRMS 0.03367, SSIM 0.90960, Pearson 0.25433, NZ-Pearson 0.25487; Pixel Diffusion (3 ch) 0.02730 / 0.03167 / 0.91990 / 0.30770; LDM+ControlNet (10 ch) 0.02859 / 0.03430 / 0.90310 / 0.36870, NZ-Pearson 0.36923. **N14**: DRC LDM MAE 0.00627, SSIM 0.7136, TopK@1% 0.0125; congestion LDM MAE 0.03416, SSIM 0.7678, Pearson 0.0369 | AMD EPYC 7763, 4x RTX A6000; DDIM, 8 draws averaged per design | github.com/kiranthorat3/CLDRoute (as printed; not tested) | VERIFIED (PDF) |
| CircuitNet 2022 letter | no split described | congestion FCN NRMS 0.040, SSIM 0.80; DRC FCN ROC-AUC 0.95, PR-AUC 0.63 | | | circuitnet.github.io | VERIFIED (PDF) |
| CircuitNet 2.0 Appendix B | not stated per experiment | congestion FCN / cGAN / GNN N14 and N28 (section 2.1) | | | github.com/circuitnet/CircuitNet | VERIFIED (PDF) |
| Official repo configs | the N28 files above for all three tasks | congestion `NRMS, SSIM, EMD`; DRC `NRMS, SSIM`, ROC/PR with `threshold 0.1`; IR drop `NRMS, SSIM`, ROC/PR with `threshold 0.9885` | | | | VERIFIED (repo) |

Whether CLDRoute re-trained RouteNet and GPDL on its split, or took published numbers, is not stated in the paper. The
task-specific RouteNet metrics (TopK, Hotspot-MAE, F1@0.1) cannot come from the original RouteNet paper, which suggests a
re-run, but this is an inference.

The PhysEDA paper in the first-pass draft was not located (no arXiv id was supplied) and is not covered here. UNVERIFIED.

---

## 3. Trivial, linear and physics baselines reported: none found

Checked (full-text search for ridge, linear regression/model, XGBoost, mean/constant prediction, persistence, nearest
neighbour, trivial, superposition, analytic): 2502.12168, 2507.19197, 2511.12581, 2406.06541, 2408.03292, 2403.18569,
2604.09999, 2607.16674, 2208.01040, CircuitNet 2.0 (ICLR 2024), and the three contest documents.

Findings:
- No paper reports a mean-map, constant, linear/ridge, kNN or gradient-boosting baseline, or an exact-solve skill
  comparison, on ICCAD'23 or on CircuitNet IR drop / congestion / DRC. The baselines are learned networks: IREDGe,
  IRPnet, ConvNeXtV2 contest winner (ICCAD'23); PowerNet, MAVIREC (CircuitNet IR drop); RouteNet, GPDL / FCN, cGAN, GNN
  (CircuitNet routability).
- Mentions of non-deep methods are in related-work text only (XGBIR, IncPIRD) and are not evaluated.
- CFIRSTNET's ablation is the closest thing: its image-only network (0.1864 mV, F1 0.217) versus a network with the
  physics-derived HIRD feature (0.0795 mV, F1 0.677); both are still a CNN.
- The contest zero-information reference is derivable from CFIRSTNET Table 2 (mean golden IR drop 1.98 mV), but no paper
  reports it as a baseline.

---

## 4. Inconsistencies across papers

Factual notes for audit context. They describe differences in what was published and do not imply intent.

1. **F1 rule differs.** Contest text says both "top 10% of IR drop" and "greater than 90% of the maximum" (a
   pixel-fraction and a value-fraction rule). CFIRSTNET, WACA, GL-Inception state the value-fraction rule. AttUNet
   states "the highest 10% IR drops are labeled as positive". Its per-case F1 for testcases 13 and 14 is 0.895 / 0.881
   while the official winner has 0 / 0 on those cases, so AttUNet's F1 is probably not on the contest's scale; what its
   code does was not checked (no code link). The winner's F1 is exactly 0 on cases 13 and 14 in the official xlsx
   (draft confirmed); CFIRSTNET's IREDGe also scores 0 on most cases.
2. **IREDGe re-implementation spread.** MAE 0.233, 0.4566, 0.588, 0.628 mV and F1 0.244, 0.082, 0.12, 0.13 across four
   papers (section 1.5). Training data and recipes differ.
3. **Contest-winner numbers are shared, runtimes differ.** All papers reuse the official 1st-place per-case MAE and F1
   (identical to the xlsx). The runtime differs: official 14.77 s (xlsx, organisers' machine); CFIRSTNET 8.921 s (the
   authors ran the 1st-place executable on their own Xeon/P4000 server, per-case values 7.996 ... 11.905); WACA Table 1
   reproduces CFIRSTNET's 8.921 s per-case runtimes under the heading "measured on CPU" while its own entry is 1.390 s;
   GL-Inception and LMM-IR use the official 14.77 s. The draft's "per CFIRSTNET 8.921 versus 14.77 elsewhere" is
   confirmed and explained by a re-run. The 2nd place runtime (3.04 s) exists only from the official data.
4. **Averages of the 3rd-place entry.** The 3rd-place team failed on cases 19 and 20. GL-Inception averages over 8
   cases (MAE 0.107, F1 0.52); the xlsx averages are "inf" (MAE) and 0.414 (F1, zeros for failed cases). The 3rd place
   score of 596.17 includes 0 points on those two cases.
5. **Training-data counts do not reconcile.**
   CFIRSTNET: 6,100 synthetic BeGAN, no real training cases (10 visible real = validation).
   WACA: 2,100 (2,000 BeGAN + 100 contest fake), 10 visible real = validation.
   GL-Inception: 3,100 = 2,000 + 100x10 + 5 real x20, consistent; 5 real held out for validation.
   LMM-IR: states 3,310 from "2000 BeGAN, 100 fake x10, 10 real x20" which sums to 3,200 (difference 110).
   AttUNet: "6 fold" augmentation but "800 pretrain data points and 80 fine-tune data points" from 100 fake and 10 real,
   which is 8x, not 6x.
   Also the repo's fake-circuit directory has 500 entries versus "100 fake cases" in the papers (to be checked by P1).
6. **Averages and claims that differ from the paper's own tables.**
   - AttUNet: printed average MAE 0.110 mV; the mean of its ten printed per-case values is 0.1172 mV (F1 mean 0.5198
     matches). The quoted "18% better MAE than the winner" uses 0.110; with 0.1172 it is 13%.
   - LMM-IR: abstract says "lowest MAE". Table III shows 1.35 for both 1st place and LMM-IR (1e-4 V); unrounded per-case
     means are 1.347 (1st) and 1.356 (LMM-IR), so it is tied at two digits and not lower. Text: "F1 score improves by 20%
     compared to the first-place team"; table values 0.58 versus 0.46 are +26% (ratio row 0.80 = 1st / ours).
   - GL-Inception: text "34% reduction in MAE and 26% improvement in F1" versus the first place equal the "Norm." row
     (1.337 and 0.74) taken relative to the paper's own values. The conventional relative changes of its Table I are
     25% lower MAE (0.135 to 0.101) and 35% higher F1 (0.46 to 0.62). Likewise "6% / 16%" versus 3rd place is 5.6% / 19%.
   - WACA and CFIRSTNET percentages (61.1%, 71.0%, 60%, 59%) recompute correctly.
7. **Text versus table.**
   - WACA ablation: text gives CBAM MAE 0.0671, F1 0.704; Table 2 gives 0.0665, 0.731. Also, in WACA's ablation
     the "+ CBAM" MAE (0.0665) is worse than the baseline (0.0588) while the text calls it a "moderate improvement".
   - GIF headline: abstract NMAE 0.026 / SSIM 0.78 / PSNR 21.77 / Pearson 0.95 versus body 0.0266 / 0.786 / 21.77 /
     0.9536; none of the PSNR, Pearson, NMAE values appear in the paper's own Table 5 (best 19.583 / 0.9215, MAE 0.0345).
   - PDNNet: Table III caption says "consistent performance gain on five evaluation metrics in both large and small
     datasets"; the table has eight metrics and one dataset. Tables IV and V are described as trained "on the whole
     CircuitNet", but the PDNNet row (NMAE 0.0282, R2 0.8369, PSNR 19.3483) rounds to the Table III values
     (0.028, 0.84, 19.35). The draft's "PDNNet R2 text versus table" item was not confirmed (text 0.81 vs 0.84 etc.
     match Table III) and is dropped.
   - CLDRoute: its abstract numbers for congestion (SSIM 0.9031, MAE 0.0286) are for LDM+ControlNet; Pixel Diffusion has
     better MAE (0.0273) and SSIM (0.9199) in its Table 9 (the text acknowledges this).
8. **GIF versus PDNNet N28 split.** PDNNet and the official `train_N28.csv`/`test_N28.csv`: train 7,078 (4 designs), test
   3,164 (zero-riscy-a and -b). GIF Table 3 and CLDRoute Table 5: train 7,872 (includes zero-riscy-a), val 1,248
   (RISCY-FPU-b), test 1,122 (zero-riscy-b). GIF's text says it uses PDNNet's split and copies PDNNet's baselines, so
   Table 6 compares across two test sets. Confirmed.
9. **Test-set sizes.** N28: 3,164 (PDNNet, official) versus 1,122 (GIF Table 3, CLDRoute). N14: official repo file test
   3,544 (zero-riscy 3,441 + openc910-1 56 + Vortex-large 47) / train 6,790; GIF Table 4 test 3,456 (zero-riscy) with
   total 10,444; CLDRoute test 250 of 10,787 (routability tasks); CircuitNet 2.0 Table 1 total 10,791; GIF Table E.1
   total 10,795 with per-design counts that differ from Table 4. Draft correction: the draft compared CLDRoute's 250 (a
   routability split) with GIF's 3,456 (an IR-drop split); the tasks differ and both are as printed, but no common N14
   split exists, and GIF's own Tables 4 and E.1 do not agree with each other.
10. **Static versus dynamic IR drop.** ICCAD'23 labels are static IR drop from a netlist solve, MAE in mV, unnormalised.
    CircuitNet N28 labels are dynamic vectorless (Innovus/Voltus) maximum IR drop, evaluated with min-max-normalised
    metrics (NMAE, NRMS). The two sets of numbers cannot be compared (confirmed).
11. **Congestion baselines differ across sources for the same method on the same dataset.** CircuitNet N28 with a
    GPDL/FCN: NRMS 0.040 / SSIM 0.80 (2022 letter); NRMSE 0.047 +- 0.002 / SSIM 0.773 (CircuitNet 2.0 Appendix, "CircuitNet"
    rows); NRMS 0.03367 / SSIM 0.9096 (CLDRoute). Splits and normalisation are not fully stated in each source.
12. **Undefined or loosely defined metrics.** GIF's NMAE and PSNR peak are undefined (and its MAE is VDD-normalised,
    PDNNet's NMAE is range-normalised). PDNNet's AUC lacks a threshold and positive class (the official code thresholds
    at 0.9885 after a log transform for the IR-drop ROC; whether PDNNet used it is not stated). The `N` and `i` in
    PDNNet's NMAE formula are ambiguous between pixels and samples. CLDRoute does not define NRMS; the official code
    uses min-max-normalised RMSE per sample.
13. **Attribution of the augmented-data team.** LMM-IR attributes "around 5,400 cases" of augmented training data to the
    second-place team; GL-Inception attributes it to the third-place team. The contest slides list the Fuzhou/Fudan
    team as 2nd place; GL-Inception's author list includes members of that team (names per the slides). The mapping
    was not resolved.
14. **CFIRSTNET naming of the reference solver.** CFIRSTNET calls HSPICE "golden" for the hidden cases. The contest
    labels were generated with an open-source solver (the contest description cites the BeGAN solver [8] and PDNSim [9]).
    Not a numerical conflict; noted because the audit uses the contest labels.

Dropped from the draft: "PDNNet R2" (not confirmed), "PhysEDA dataset unclear" (paper not located), and the draft's
statement of runtime for WACA "1.390 s CPU" is confirmed as printed.

---

## 5. Comparison targets for the audit (options only; the protocol is for the lead to decide)

### 5.1 ICCAD'23

Fixed facts:
- Held-out set = hidden testcases 7, 8, 9, 10, 13, 14, 15, 16, 19, 20 (public in the repo, so held-out only by
  pre-registration).
- Official MAE = per-pixel mean |pred - golden|, reported in mV (volts in the xlsx), unnormalised.
- Official F1 = hotspot if golden value > 0.9 x max of that testcase's golden map; F1 = 2TP/(2TP+FP+FN). The rule for
  the prediction side is not spelled out; the usual reading, and the one in WACA/CFIRSTNET text, applies the golden
  max-based threshold to both maps. This choice should be recorded in the protocol.
- Reference rows (official, all 10 cases): winner 0.1347 mV / 0.455 / 14.77 s; 2nd 0.1498 / 0.455 / 3.04 s; 3rd 0.1069
  (8 valid cases) / 0.4143 (10 cases, failures as 0) or 0.5186 (8 cases) / no runtime. Published best: CFIRSTNET 0.0533 /
  0.723, WACA 0.0524 / 0.778 (no code for WACA; CFIRSTNET has a repo link).

Options and trade-offs:
- A. Same hidden 10, official MAE and F1 definitions, mean over 10. Directly comparable to every published row. The
  pre-registration freeze is what makes the cases held out. Trade-off: published learned models also used the 10 visible
  real cases for validation (CFIRSTNET, WACA, GL-Inception) or for fine-tuning (AttUNet, LMM-IR), so the training
  sets differ.
- B. Report per-case values and the mean, plus the number of cases with F1 = 0. The mean F1 depends heavily on whether
  zero-hotspot predictions count as 0 (the winner has two such cases). Trade-off: longer tables, but makes the
  F1 sensitivity visible.
- C. Also report F1 under the "top 10% of pixels" rule used by AttUNet, as a labelled secondary metric. Trade-off:
  lets AttUNet's F1 be compared, but adds a second definition.
- D. Compare the skill relative to a zero or mean-map reference using the label scale (mean IR drop 1.98 mV, CFIRSTNET
  Table 2). Trade-off: needs the lead to decide the reference; the published tables do not contain one.
- E. Runtime: only the official 14.77 s (winner) and 3.04 s (2nd) were measured on the organisers' machine. The
  papers' runtimes use different hardware (section 1.3). Any runtime comparison for the exact solve is a different-machine
  comparison.

### 5.2 CircuitNet

Which splits are defensible, given the conflicts:
- **N28 official split (7,078 train / 3,164 test, 4 vs 2 designs, no validation).** It is the split in the official
  repo files, the original task configs for congestion, DRC and IR drop, and PDNNet's Table III (the only IR-drop
  baseline table with PowerNet / MAVIREC numbers). Design-level (no design appears on both sides). Trade-offs:
  no validation set (a validation design would have to be carved from the training designs, which departs from the
  official files); the baselines' numbers use their own normalisation (NMAE per-sample range).
- **GIF / CLDRoute N28 split (7,872 / 1,248 / 1,122, zero-riscy-b as the only test design).** Also design-level and
  has a validation design. It is the only split with published 2026 numbers for the diffusion models (CLDRoute for
  congestion and DRC). Trade-offs: a single test design (1,122 samples), its training set contains zero-riscy-a, a
  sibling of the test design (the same RTL family, different synthesis/placement runs), and GIF's text and table disagree
  about which split it used. Test design names are not given in CLDRoute.
- **N14 official files (6,790 / 3,544).** Design-level, but the test set is 97% zero-riscy and the train set
  contains no zero-riscy; the papers read give no IR-drop numbers for this split. GIF's N14 split (10,444 total, test
  zero-riscy 3,456) and CLDRoute's N14 split (250 test) differ from it and from each other. No N14 split is
  published with agreed counts, so N14 comparisons to published numbers are weak; N14 IR drop has only GIF's single
  baseline-less table. Both GIF and CLDRoute report N14 numbers much weaker than N28 (SSIM 0.56 to 0.77).
- A random or sample-level split is not used by any of the papers read, so it has no published comparison.
- Cross-benchmark comparison with ICCAD'23 is not valid (dynamic versus static IR drop; normalised versus unnormalised
  metrics).
- Metric choice: for IR drop, PDNNet's NMAE formula (ambiguous pixel versus sample index) and the official-code NRMS
  (scikit-image min-max) differ; for congestion and DRC, CLDRoute's NRMS is undefined in the text and the official
  code's NRMS is defined. Trade-off: using the official code's definitions is reproducible, but the published
  PDNNet/GIF tables use NMAE.
