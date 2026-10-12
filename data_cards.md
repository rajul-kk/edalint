# Data cards (P1: acquisition and format inspection)

Status date: 2026-10-12. Phase P1 of `PLAN.md`: files, shapes, dtypes, key names and documentation only. **No baseline, no
correlation, no error and no statistic of labels against inputs has been computed.** Every value I looked at is in the
disclosure section of each card. Data is outside the repo in `E:\edalint_data\<dataset>\` (read-only by convention); sha256
and sizes are in `manifests/MANIFEST.json` (rebuild with `python -I scripts/p1_make_manifest.py`). Loader specifications are in
`docs/format_notes.md`.

| Dataset | Obtained | On disk | Task(s) with maps |
|---|---|---|---|
| ICCAD 2023 Problem C | yes, complete | 3.5 GB (617 files) | static IR drop |
| CircuitNet-N14 (IR drop + routability subsets) | **in progress** (background download, see card 2) | 76.6 GB when complete | IR drop, congestion, DRC |
| CircuitNet-N28 | **no** (Google Drive quota exceeded; Baidu needs an account) | 0 | (IR drop, congestion, DRC) |
| CircuitNet 3.0 (N45) | not downloaded, by design | README only | **no map task** (timing/power) |

---------------------------------------------------------------------------------------------------------------

## 1. ICCAD 2023 CAD Contest, Problem C: static IR drop estimation using machine learning

- **Source**: `https://github.com/ASU-VDA-Lab/ML-for-IR-drop`, one commit `bda90cbe53f973f28d7c96951af665f678a84180`
  ("Add benchmark generation scripts"), cloned `--depth 1` to `E:\edalint_data\iccad23\repo\`. No Git LFS in use.
  Maintainers: Gana Surya Prakash Kadagala, Vidya A. Chhabria (Arizona State University); supported by Steel Perlot and
  the OpenROAD Project.
- **Licence**: BSD 3-Clause (`LICENSE`, "Copyright (c) 2023, ASU-VDA-Lab").
- **Citation** (the repo names no preferred one): G. S. P. Kadagala and V. A. Chhabria, "Invited Paper: 2023 ICCAD CAD Contest
  Problem C: Static IR Drop Estimation Using Machine Learning", ICCAD 2023 (`doc/invited-paper.pdf`); the synthetic data
  generator is V. A. Chhabria, K. Kunal, M. Zabihi, S. S. Sapatnekar, "BeGAN: Power Grid Benchmark Generation Using a
  Process-portable GAN-based Methodology", ICCAD 2021 (cited as [8] in the contest description).
- **Size**: 617 files, 3,542,856,338 bytes (fake 500 files, ~2.9 GB; hidden real 51 files, ~274 MB; visible real 50 files, ~227 MB;
  `src/` 8, `doc/` 5, plus README, LICENSE, .gitignore). `.git` adds 645 MB and is not in the manifest (the commit id is).
- **Files and sha256**: every file is listed with size and sha256 in `manifests/MANIFEST.json` under
  `datasets.iccad23_problem_c.files`. Key documents: `doc/contest-description.pdf`
  `139b23fd2b6264fe6b166e213a4249c2ca8685f5d6a9af44a7df20cff4b13bd6`; `doc/invited-paper.pdf`
  `2c8faca396eaff8061c34be9b12f302447817cda4b972f44e456c935bd41c227`; `doc/ICCAD23-Contest-ProblemC.pdf`
  `06f570cc13002c5a8722c836cdfb2a57532f1d2459acfe80634b1b53f0d5ac7c`; `doc/ICCAD-Final-Scores-Release.xlsx`
  `ace86c213aeeeef81b868dd475d191f67c4ae3a304968f3179f27685a1c1bec4` (the final contest scores: **not opened in P1**, it is a P2
  source of published numbers).
- **Contents**: 100 "fake" cases (BeGAN GAN current maps, synthesized Nangate45 PDN, solver IR drop), 10 visible real
  cases (`testcase1,2,3,4,5,6,11,12,17,18`), 10 "hidden" real cases (`7,8,9,10,13,14,15,16,19,20`). The hidden cases are
  public in the repo, including their `ir_drop_map.csv`; `hidden-real-circuit-data/list_testcase.txt` lists them. Real
  circuits come from OpenROAD-flow-scripts placement, PDN extracted with PDNSim (contest slides).
- **Task definition**: predict the static IR-drop map. Inputs: current map, PDN density map, effective distance to voltage
  source map (all 2-D, 1 um/pixel); optionally the SPICE netlist (R, I, V elements, node names with coordinates).
  Label: IR-drop map in volts, `VDD - V` (VDD = 1.1 V) at the lowest metal layer nodes, cubic-interpolated onto the
  1 um grid. Metrics: MAE (V), F1 of hotspots (> 90% of that case's maximum label), runtime; score weights 60/30/10 over 10
  hidden cases.
- **Input channels / label**: 3 channels (`current`, `eff_dist`, `pdn_density`) -> 1 label channel, plus netlist. Units: A,
  (unstated), (unstated), V.
- **Grid size**: native, square, 205 to 930 pixels per side (fake 205-913, real 204-930, hidden 257-870); all four maps of a
  case share one shape (checked on all 120 cases). **No 512 x 512 resize exists in the repo** (see `docs/format_notes.md`
  1.4); the contest requires outputs at native size.
- **Netlist structure** (element counts; V = voltage sources, R = resistors, I = current sources):
  visible real: tc1 4/24088/11599, tc2 4/22328/11599, tc3 49/228440/101684, tc4 49/214890/101684, tc5 25/110466/38934,
  tc6 25/102526/38934, tc11 4/10860/7718, tc12 4/10408/7718, tc17 16/84045/59078, tc18 16/80039/59078.
  hidden: tc7 16/94590/24988, tc8 16/89562/24988, tc9 36/186291/64064, tc10 36/172877/64064, tc13 4/17183/11864,
  tc14 4/16535/11864, tc15 16/64120/44117, tc16 16/59560/44117, tc19 36/201902/63347, tc20 36/188475/63347; see
  `E:\edalint_data\_tools\iccad_structure.json` for all 120 cases (fake V in {4, 9, 16, 25, 36}). Voltage sources are at `m9`, current sources and labels at `m1`.
- **Official split**: train = 100 fake + 10 visible real; test = 10 hidden real (contest description section 4 and the invited
  paper: "ten seen during training and ten as hidden"). Suggested use: pre-train on fake, fine-tune on the 10 visible real.
  Evidence: `list_testcase.txt` and directory names; contest description section 4. There is no validation split.
- **Structure relevant to the audit (from shapes and element counts only)**: real cases are twins. For each of the 5 hidden
  designs there are two cases with identical grid size, identical current-source count and different resistor counts
  (7/8, 9/10, 13/14, 15/16, 19/20), and the same holds for the visible set (1/2, 3/4, 5/6, 11/12, 17/18). The contest slides
  say the twins are the same design with 4 versus 1 PDN templates. So the hidden test has 5 independent designs. Visible and
  hidden sets share no grid size (visible 204/298/566/641/930; hidden 257/489/601/835/870).
- **Known issues and data versions** (contest description version history, `doc/contest-description.pdf`):
  - v3 (2023-06-12): benchmarks "updated to eliminate those with high values of voltage drop and negative voltage drop";
    testcase11 and testcase12 re-issued; fake cases 12, 18, 21, 29, 31, 26, 42, 44, 62, 63, 67, 69, 96 re-issued.
  - v4 (2023-06-22): "Removal of duplicate files in fake circuit benchmarks"; clarified metrics.
  - The repo does not say which release the files correspond to, nor how many duplicates were removed. The repo's fake
    indices run 00..99 with no gap, so the case indices in the v3 note may not map one-to-one onto the file names (unverified).
  - The contest's F1 worked example is internally inconsistent (marks 5.23 mV as a hotspot against a 5.337 mV threshold).
  - README: "The scripts used to generate the benchmarks will be uploaded shortly." The scripts in `src/` are the BeGAN-style
    generators, not the real-circuit flow.
  - E005 candidate: the generator writes the label as `[x, y]` from a meshgrid; CSV orientation of the real cases is
    undocumented.
- **Disclosure (what I looked at)**: file lists and names; row/column counts of all CSVs (120 cases x 4 maps); the first
  characters of one fake `current` CSV (`current_map00_current.csv`, first row, 3 values) and of the visible real
  `testcase1/current_map.csv` (first 8 values, all `0.00000e+00`) and `testcase1/ir_drop_map.csv` (first 4 values, around 4.3e-04 V);
  the first 15 lines and element-type counts of netlists (`testcase1`, fake `current_map00`, hidden `testcase7`: its `V0` line
  and counts); counts of R/V/I lines for all 120 cases; the PDF text of the contest description, invited paper and slides.
  **No min, max, mean or any other statistic of any map was computed. Hidden cases: shapes, names and netlist element counts
  only** (plus the `V0` line of hidden testcase7 and its R/V/I counts; no CSV values).

---------------------------------------------------------------------------------------------------------------

## 2. CircuitNet-N14

- **Source**: Hugging Face dataset `CircuitNet/CircuitNet`, revision `df74f33e302f927bf0548dc0d7936ca43b634f2a`, directory
  `CircuitNet-N14/`. Code and docs: `https://github.com/circuitnet/CircuitNet` (commit
  `41ade1d7e913a418b4495b6b13630b26b0ff7279`, 2026-05-17) and `https://circuitnet.github.io`. Maintained by Peking University
  (School of Integrated Circuits) and collaborators.
- **Licence**: BSD 3-Clause (HF card `license: bsd-3-clause`; GitHub `LICENSE`, "Copyright (c) 2022 - present, the School of
  Integrated Circuits, Peking University").
- **Citation** (README): Z. Chai, Y. Zhao, W. Liu, Y. Lin, R. Wang, R. Huang, "CircuitNet: An Open-Source Dataset for Machine
  Learning in VLSI CAD Applications with Improved Domain-Specific Evaluation Metric and Learning Strategies", IEEE TCAD 2023,
  doi 10.1109/TCAD.2023.3287970; and J. Xun, Z. Chai, Y. Zhao, Y. Lin, R. Wang, R. Huang, "CircuitNet 2.0: An Advanced Dataset for
  Promoting Machine Learning Innovations in Realistic Chip Design Environment", ICLR 2024 (the N14 release).
- **Subsets obtained** (only these; 16 tarballs, 76,575,302,253 bytes): `CircuitNet-N14/IR_drop_features/<design>.tar.gz`
  (8 designs, 50.9 GB) and `CircuitNet-N14/routability_features/<design>.tar.gz` (8 designs, 25.6 GB). Not downloaded: `graph_features`,
  `timing_features`, `raw_data` (DEF, netlist, RTL, LEF), 300+ GB. File list, sizes and the HF-reported LFS sha256 are in
  `manifests/MANIFEST.json` (`datasets.circuitnet_n14.files`, field `sha256_expected_hf_lfs`; our own `sha256` is filled for each
  completed tarball and compared to it).

  | design | IR_drop_features | routability_features |
  |---|---|---|
  | RISCY-FPU | 14,536,296,658 | 6,295,286,066 |
  | RISCY | 10,423,471,258 | 4,925,519,091 |
  | zero-riscy | 8,892,579,388 | 3,914,477,627 |
  | nvdla-large | 4,966,123,393 | 4,462,461,331 |
  | openc910-1 | 4,913,858,709 | 1,866,677,672 |
  | Vortex-large | 4,040,647,821 | 2,435,656,980 |
  | nvdla-small | 2,236,565,370 | 1,237,003,539 |
  | Vortex-small | 1,001,717,964 | 426,959,386 |

- **Download status at the time of this commit**: 7 of 16 tarballs complete, each with our sha256 equal to the HF LFS sha256:
  IR_drop `Vortex-small`, `nvdla-small`; routability `Vortex-small`, `nvdla-small`, `openc910-1`, `Vortex-large`, `zero-riscy`.
  The other 9 are downloading in the background (6 worker processes, resumable; see the report accompanying this commit).
  After it finishes run `python -I scripts/p1_make_manifest.py` (fills hashes) and `python -I scripts/p1_n14_inventory.py`
  (per-design file counts and split coverage -> `manifests/n14_inventory.json`; the committed copy covers only the completed tarballs).
- **Samples**: 10,345 samples from 8 RTL designs with synthesis and physical-design variations (docs, "Basic Properties"):
  frequency 50/200/500 MHz (2 values for the second design group), macro placement (4 or 2 variants), utilization 50-75 % in 6 steps
  (first design group) or 50-65 % in 4 steps (second group), aspect ratio 1.0/1.5/2.0 (1.0 only for the second group), power-mesh
  setting 1-8 (or 1-3); which designs belong to which group is not machine-readable in the page text I saved, filler insertion stage `ap` (after placement) / `ar` (after routing). File naming:
  `{design}_freq_{f}_mp_{m}_fpu_{u}_fpa_{ar}_p_{p}_fi_{ap|ar}`; `fpu` = utilization %, `fpa` = aspect ratio, `p` = power-mesh
  setting. Per-design counts seen so far (label-directory files): Vortex-small 96, openc910-1 96, Vortex-large 73 (DRC_all) / 74 (any feature), zero-riscy pending; 96 / 94 (DRC_all) for
  nvdla-small routability; 88 power maps / 85 IR_drop maps for nvdla-small IR. **Feature directories of one design do
  not hold the same number of files** (nvdla-small: DRC_all 94 of 96, IR_drop 85 of 88 power maps), which is the
  documented file-mismatch issue (see known issues).
- **Array format (inspected on Vortex-small)**: each map is one `.npz` (NumPy zip) with one key `'data'`, dtype float64.
  2-D `(H, W)` per map, per-sample sizes vary (Vortex-small: 15 distinct grid-size classes, 403..464 per side, correlated with
  utilization/aspect ratio); `power_t` is `(20, H, W)` (20 time slices, time axis first). **Not `.npy`**: the authors' scripts
  call `np.load(...)` and treat the result as an array, which holds for N28 `.npy` files but not for these N14 `.npz` (the
  loader needs `['data']`). Within a sample the early-global-routing congestion maps (`congestion_eGR_*`) live on a different,
  finer grid (e.g. 832 x 830 against 459 x 456 for every other map of the same sample, ratio about 1.8); the global-routing
  congestion maps and everything else share the sample grid. Extraction expands about 10x (427 MB tarball -> 4.1 GB), so full
  extraction of the 25.6 GB routability set is about 250 GB: not feasible on this disk. P3 should stream from the tarballs or
  cache downsampled arrays.
- **Tasks, input channels and labels** (the authors' `generate_training_set.py`, mapped to 256 x 256; details in
  `docs/format_notes.md` section 2):
  - *Congestion* (GPDL): 3 input channels `macro_region`, `RUDY`, `RUDY_pin`; label = `congestion_GR_horizontal_overflow` +
    `congestion_GR_vertical_overflow` (global-routing overflow), 1 channel. Metrics NRMS, SSIM, EMD.
  - *DRC* (RouteNet): 9 channels `macro_region`, `cell_density`, `RUDY_long`, `RUDY_short`, `RUDY_pin_long`, `congestion_eGR`
    horizontal and vertical overflow, `congestion_GR` horizontal and vertical overflow; label = `DRC_all` (DRC violations per
    tile), clipped at 200 and divided by 200. Metrics NRMS, SSIM and ROC AUC at threshold 0.1.
  - *IR drop* (MAVI): 24 channels = `power_i`, `power_s`, `power_sca`, `power_all` and the 20 slices of `power_t`; label =
    `IR_drop` (maximum IR drop per tile from a vectorless power-rail analysis; **unit not documented**; the preprocessing clips
    it to [1e-6, 50] before `log10`, which suggests mV, inferred). Metrics NRMS, SSIM and ROC AUC at hotspot threshold 0.9885
    (documented as "5% after log" in `utils/configs.py`; defined for N28 there).
  - Grid: native per sample, then resized to 256 x 256 by the authors' preprocessing (congestion/DRC features: cubic
    `scipy.ndimage.zoom`; DRC label and all IR maps: `cv2.INTER_AREA`), each channel min-max scaled per sample.
- **Official splits** (GitHub `routability_ir_drop_prediction/files/train_N14.csv`, 6,790 rows, and `test_N14.csv`, 3,544 rows,
  no overlap; one pair of CSV files for all three tasks, `feature/<name>.npy,label/<name>.npy` rows):
  - **The split is by design.** train: RISCY-FPU 3,449; RISCY 3,162; Vortex-small 95; nvdla-small 84. test: zero-riscy 3,441;
    openc910-1 56; Vortex-large 47. nvdla-large (in the HF repo) appears in neither file; docs list 10,345 samples, the CSVs
    10,334.
  - The split is **dominated by one test design** (zero-riscy 97% of the test rows) and the train designs RISCY and
    RISCY-FPU share an RTL family; Vortex-small/-large and nvdla-small/-large are small/large variants of one design.
  - Not every sample of a tarball is in the CSVs: e.g. openc910-1 has 96 files and 56 listed; Vortex-small 96 and 95;
    nvdla-small routability 96 and 84. The reason for the unlisted samples is undocumented (possibly the file-mismatch
    clean-up; unverified).
  - The repo's `utils/configs.py` still points at `train_N28.csv`/`test_N28.csv` by default; N14 needs
    `--ann_file_train ./files/train_N14.csv --ann_file_test ./files/test_N14.csv`.
  - Evidence: the CSV files themselves (design prefix of every row, counted by file name).
- **Known issues** (README, FAQ, GitHub issues titles; not independently verified except where stated):
  - README: "Known issue (2024/12/16): some instance names in the DEF are wrong. The DEF will be re-upload later. A fixing script is
    uploaded (`feature_extraction/fix_module_name_241216.py`)". Affects the DEF (raw_data), which we did not download; the
    map features are unaffected as far as documented.
  - README 2024/11/09: LEF/DEF, netlist and graph information re-uploaded (issue #38).
  - GitHub issues #31, #36, #42 (closed, 2024): feature and label file counts do not match in N14 ("there are 11 files just in
    feature directory, but not in label directory"; "10 files in the label that do not match"); maintainer reply: "The issue
    originates from a mismatch in the number of files for each feature in N14. I have updated generate_training_set.py to avoid
    this issue for now, and we will update the features to match their numbers soon." **Verified in our download** (nvdla-small:
    IR_drop 85 files vs 88 power maps; DRC_all 94 vs 96 for the other routability features). The current
    `generate_training_set.py` takes names from the label directory and reads each feature by that name, so a label file without a
    feature file would raise; the CSVs were provided as the workaround.
  - GitHub issue #40 (open): the N14 release lists 7-8 designs, the paper names others; OpenC910-1, NVDLA-large, Vortex-large
    were asked about. In the HF repo all 8 are present.
  - GitHub issue #19 (N28, closed): congestion labels hold small non-zero values in tiles with no cells (unconfirmed here).
  - Docs: "the decompress instructions for N14 on Hugging Face are coming soon"; the decompress scripts target the Google Drive
    layout (one tarball per feature), not the HF layout (one tarball per design, split parts `-part-aa..` for DEF and
    timing only).
  - arXiv 2604.09999 (GIF) reports all-zero bounding boxes in CircuitNet-N14 (from the PLAN; not checked in P1).
- **Disclosure (what I looked at)**: the HF file list and sizes; README/FAQ/changelog/feature pages and the tutorial text; the
  GitHub repo tree and the Python source and CSV split lists; the tar member listings (names, counts, sizes) of the five
  tarballs downloaded so far; for the extracted `Vortex-small` (IR and routability) the array **shapes, dtypes and `.npz` key
  names** of the first file in each feature directory and the shapes of 13 feature maps of all 96 routability samples (to see
  the grid structure). Loading a map reads its values but I printed none. No min/max or other statistic was computed, and no
  label was compared with an input.

---------------------------------------------------------------------------------------------------------------

## 3. CircuitNet-N28

- **Obtained**: **no.** Time-boxed check (well under the 1 hour box), 2026-10-12.
- **Links** (README and download page): Google Drive folder
  `https://drive.google.com/drive/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB` and Baidu Netdisk
  `https://pan.baidu.com/s/1udXVZnfjqniH9paKfyc2eQ?pwd=ijdh` (the download page gives a second Baidu link,
  `https://pan.baidu.com/s/1evSTtuvphyl1_aSedsEQLA?pwd=wihf`).
- **What works**: the Drive folder is public and `gdown` lists it without an account (the file ids and names, for
  `CircuitNet-N28/IR_drop_features` (IR_drop, power_i/s/sca/all, power_t split in 11 parts `.00-.10`) and
  `routability_features` (congestion eGR/GR overflow and utilization, DRC_all, DRC_seperated, RUDY x5, cell_density, macro_region),
  plus graph, timing, raw LEF/DEF and netlist). Drive does not expose file sizes to an anonymous visitor.
- **What does not**: every download attempt (one request per file for 9 files, plus one re-fetch of one to read the message) returned the Drive page "Quota exceeded: too
  many users have viewed or downloaded this file recently". I did not retry in a loop, use cookies or mirrors; Baidu needs an
  account and I did not sign up. Size against the 60 GB cap is therefore unverified. N28 has two designs in the official split
  (train RISCY, RISCY-FPU; test zero-riscy, see below).
- **Official split (from the GitHub files, which I did read)**: `train_N28.csv` 7,078 rows (RISCY 3,861; RISCY-FPU 3,217) and
  `test_N28.csv` 3,164 rows (zero-riscy 3,164): design-level, no overlap. N28 file names are
  `{index}-{design}-{variant}-{macros}-c{clock}-u{util}-m{macro placement}-p{power mesh}-f{filler}`.
- **Task definitions**: same three tasks as N14, with the same loaders (the repo's default config is N28). Per-sample
  `.npy` (not `.npz`).
- **Recommendation**: retry the Drive download later (quota resets within 24 h) or ask the user for approval to contact the
  maintainers; the N28 results are not needed for the first audit if N14 and ICCAD'23 suffice.
- **Disclosure**: only the Drive folder listing and the quota page; no N28 data.

---------------------------------------------------------------------------------------------------------------

## 4. CircuitNet 3.0 (N45)

- **Source**: Hugging Face `SKLP-EDA-LAB/CircuitNet3.0` (revision `5683d95097bb27bb380f135101d7da6991025557`; code in
  `sklp-eda-lab/iclr-circuitnet_3.0`); paper "CircuitNet 3.0: A Multi-Modal Dataset with Task-Oriented Augmentation for AI-Driven
  Circuit Design" (ICLR, openreview `lEDb4gQ4dB`). Licence: Apache-2.0 (HF card; note: different from N14/N28).
- **Files**: `README.md` (2,103 bytes, downloaded to `E:\edalint_data\circuitnet3\`, sha256 in the manifest), `.gitattributes`,
  `circuitNetv3.zip` (1,032,704,519 bytes, **not downloaded**).
- **Does it have map tasks? No.** The README states the archive exposes RTL, synthesized netlists, per-design `feature.json`,
  `final_netlist.v` and `power_summary.txt`; labels are timing (AT, WNS, TNS) and total power; "layout artifacts are large and
  are being uploaded incrementally". I confirmed by reading the zip central directory over HTTP range requests (8.7 MB, no
  data extracted): 57,975 entries, `dataset/` and `dataset_augment/`, extensions `.v`, `.json`, `.txt` and extension-less
  files; no `.npy`/`.npz`/IR-drop/congestion/DRC entries. The member list is saved at
  `E:\edalint_data\_tools\cn3_zip_names.txt`.
- **Split**: source-level train/validation/test, with augmented variants of test sources excluded from train/val (README).
- **Decision**: not a target of the audit; not downloaded. The README says the layout subset (the only one that could carry maps)
  is not yet public.
- **Disclosure**: the README and the zip member names.
