# Format notes (P1) - loader specifications

Scope: exact file formats and the transforms used by the original papers' loaders, enough to build the P3 loaders.
Nothing here is a result. Data lives on `E:\edalint_data\<dataset>\` (read-only after download); hashes are in
`manifests/MANIFEST.json`. Sources are cited per section; "inferred" marks anything not stated by the authors.

---------------------------------------------------------------------------------------------------------------

## 1. ICCAD 2023 Problem C (static IR drop)

Local copy: `E:\edalint_data\iccad23\repo\` (git clone, `--depth 1`, commit `bda90cbe53f973f28d7c96951af665f678a84180`,
"Add benchmark generation scripts"). No Git LFS pointers: all 617 files are real content (checked: `git lfs ls-files`
is empty and the CSVs parse as numbers).

### 1.1 Layout

```
benchmarks/
  fake-circuit-data/            500 files = 100 cases x 5 files, flat directory
      current_map{NN}.sp                NN = 00..99
      current_map{NN}_current.csv       input: current map
      current_map{NN}_eff_dist.csv      input: effective-distance-to-voltage-source map
      current_map{NN}_pdn_density.csv   input: PDN density map
      current_map{NN}_ir_drop.csv       label: IR-drop map
  real-circuit-data/testcase{1,2,3,4,5,6,11,12,17,18}/    10 visible real cases (training)
  hidden-real-circuit-data/testcase{7,8,9,10,13,14,15,16,19,20}/   10 hidden real cases (test) + list_testcase.txt
      each case directory: netlist.sp, current_map.csv, eff_dist_map.csv, pdn_density.csv, ir_drop_map.csv
doc/  contest-description.pdf (v4, 2023-06-22), invited-paper.pdf, ICCAD23-Contest-ProblemC.pdf/.pptx, ICCAD-Final-Scores-Release.xlsx
src/  BeGAN-based generation scripts (see 1.6)
```

The "hidden" cases are public in the repo. Their IR-drop labels (`ir_drop_map.csv`) are present in the clone. For P1 only
file names and shapes of the hidden cases were looked at.

### 1.2 CSV maps

- Plain text, comma-separated, no header, no index. Scientific notation (`4.21944e-08`, 5-6 significant digits).
- Square arrays in every case seen (rows == columns); all four maps of a case have identical shape (checked on all 120
  cases: row and column counts of the four CSVs agree).
- Pixel = 1 um x 1 um of die ("every value ... represents ... a 1um x 1um area", contest description, section 3a).
  Grid side therefore equals die side in um (e.g. real testcase1: 298 x 298, area 0.089 mm^2 on the contest slides).
- Units (documentation, section 3 and the BeGAN script): current in A, IR drop in V (`vdd - V` at the lowest metal
  layer, VDD = 1.1 V), PDN density and effective distance have no stated unit (inferred: dimensionless density; distance
  in um). Effective distance is described only as "the distance from each PDN node to the PDN node with the voltage source".
- Grid sizes (rows = cols). Real/visible: 204 (tc11, tc12), 298 (tc1, tc2), 566 (tc17, tc18), 641 (tc5, tc6), 930 (tc3, tc4).
  Hidden: 257 (tc13, tc14), 489 (tc15, tc16), 601 (tc7, tc8), 835 (tc9, tc10), 870 (tc19, tc20).
  Fake: 205 to 913, 100 cases, near-unique sizes (a few sizes occur twice: 330, 530, 570, 574, 679, 895, 899).
- **Cases come in pairs**: every real case has a twin with the same grid size, the same number of current sources and
  the same die area but a different resistor count (contest slides: same design, PDN with 4 templates vs 1 template). The
  visible and hidden sets share no grid size, so the pairs are not split across train/test by size; the 10 hidden cases
  are 5 independent designs x 2 PDNs (effective n = 5 for any per-design analysis).
- Orientation: not documented. The docs print the CSV as a matrix with the first row on top; the generation script
  writes `zi_vol` (indexed `[x, y]` from `meshgrid(yi, xi)`) with `np.savetxt`, so **axis 0 of the CSV is x** (inferred
  from `src/generate_benchmark_maps.py`, function `save_heat_map`). Whether the real-circuit CSVs follow the same
  convention is not documented. This is an E005 candidate (axis fault) and must be tested after the freeze, not now.

### 1.3 SPICE netlist (`.sp`)

Line formats (generic `<component> <node1> <node2> <value>`; contest description, section 3b):

```
R<idx> <nodeA> <nodeB> <ohm>      resistor (wire segment, or via when x,y equal and layer differs)
I<idx> <node> 0 <ampere>          current source (a cell), at a lowest-layer node
V<idx> <node> 0 1.100000          voltage source (a C4 bump), at a top-layer node
.op
.end                              (no space/newline after .end in the files seen)
```

- Node name: `n1_<layer>_<x>_<y>`, layer in `{m1, m4, m7, m8, m9}` (Nangate45 PDN; mixed case `M3` appears only in the
  description's example), `x`,`y` integers in **database units, 2000 dbu = 1 um** (`113400 -> 56.7 um`).
  Ground is node `0`.
- Resistor values are plain ohms (e.g. `5.356235`), written with `%f` (6 decimals) by the BeGAN script.
- The lowest layer `m1` carries the current sources and defines the label nodes; the label is `VDD - V(m1 node)` interpolated
  onto the 1 um grid.
- Element counts per case are in `data_cards.md` (voltage sources: 4, 9, 16, 25, 36, 49 across cases).
- Parsing: line-oriented, first character selects the type; ~120k lines for hidden testcase7, ~35k for testcase1; the fake
  netlists are larger (testcase-size dependent; line counts per case are in `E:\edalint_data\_tools\iccad_structure.json`). No continuation lines, no `.subckt`, no comments in the files inspected (checked the
  first characters of testcase1; full parse deferred to P3).

### 1.4 What the original pipeline does about size (the "512 x 512 resize")

**The repository contains no resizing step to 512 x 512.** `grep -rn 512 src/` finds only a commented layer-width list;
the only `resize` calls are in the GAN-map generator (`skimage.transform.resize` of GAN images to a design area). The contest
description requires the opposite: "all contestants must generate outputs as voltage drop matrices (.csv) of the same size
as the input matrices". Any 512 x 512 resampling therefore belongs to individual contestants' code, not to the benchmark.
Consequences for P3/P4: loaders must keep native resolution; any resize is a protocol decision to be frozen in P4. Where a
published number uses a resize, record it in `published_numbers.md` (P2).

### 1.5 Metrics as specified by the contest (section 4 of the description)

- MAE: mean absolute difference between prediction and label CSV (volts); the example computes the plain mean
  (`MAE = 0.45 mV`).
- F1: hotspot = label pixel strictly greater than 0.9 * max(label of that case) ("larger than 90% of 5.93 mV = 5.337 mV");
  the prediction is thresholded with **the same threshold derived from the label's maximum** (the example uses 5.337 mV
  for both matrices) - stated by the worked example only, not as a sentence. Positive class = hotspots.
  Worked example: TP=1, TN=2, FP=0, FN=1, F1 = 0.67. **Doc inconsistency**: the example's label matrix is
  `[[4.63, 5.23], [5.93, 0.04]] mV` and its stated threshold is 5.337 mV, yet its "actual hotspot matrix" is `[[0,1],[1,0]]`,
  i.e. 5.23 mV is marked a hotspot although it is below the threshold. With the stated rule the label would have one hotspot
  and the example would give TP=1, FP=0, FN=0, F1=1. The official scorer is not in the repo, so the intended rule
  (strict `>` vs `>=`, a 90th-percentile rule, or something else) is undetermined; the P4 protocol must pick and state one.
- Runtime, plus relative scoring across teams (60% MAE, 30% F1, 10% runtime); 10 hidden cases, 1000 points maximum.
- Evaluation is per hidden test case, 10 cases (5 independent designs, see 1.2).

### 1.6 Data versions and removals (contest description version history)

- v1 2023-03-23; v2 2023-05-31 (benchmarks released); v3 2023-06-12: "Fake and real circuit benchmarks ... updated to
  eliminate those with high values of voltage drop and negative voltage drop. Testcase11 and Testcase12 updated in real
  circuits and 12, 18, 21, 29, 31, 26, 42, 44, 62, 63, 67, 69, 96 updated in fake circuits"; v4 2023-06-22:
  "Removal of duplicate files in fake circuit benchmarks" (clarified evaluation metrics, Python 3.6, executable interface).
- The repo's fake set has indices 00..99 with no gaps (100 cases). Whether the 00..99 numbering is the pre- or post-dedup
  numbering is not stated. The contest document numbers cases from the early releases ("12, 18, 21 ..."), the repo files start
  at 00, so the "updated" indices may be off by one (inferred, unverified).
- README: "The scripts used to generate the benchmarks will be uploaded shortly." The repo's single commit adds scripts
  (`src/`) that are BeGAN/asap7/nangate45 generators; the real-circuit data came from OpenROAD-flow-scripts + PDNSim
  (slides), which is not in the repo. Fake data: Nangate45 PDN synthesized by the BeGAN Python scripts on GAN current maps,
  IR drop from a sparse solve (`G V = J`, `src/ir_solver.py`).

### 1.7 Label generation, as in `src/generate_benchmark_maps.py` (fake-style pipeline)

- `solve_ir`: sparse LU on `G + 1e-13 I`; for each stripe node on the bottom layer, `ir_drop = vdd - V`.
- `save_heat_map`: `scipy.interpolate.griddata(..., method='cubic', fill_value=0)` onto an integer 1 um grid, then
  `zi_vol[10:, 10:]` (drops the first 10 rows and columns). So labels are cubic interpolations of node values, with zeros
  where the grid lies outside the convex hull of the nodes. Not verified against the shipped fake CSVs in P1.
- `load_params`: vdd 1.1 V, layers `m1,m4,m7,m8,m9`, pitches `[4800, 28000, 80000, 22400, 22400]` dbu, via resistances
  `[15, 9, 1, 1, 1]`, 4 PDN templates (`l1*28000` m4 pitch), regions 100 um x 100 um, bump pitch 100 um.
  (Shipped netlists show `m1` segments at 4800 dbu spacing, consistent with the first pitch.)
- `load_current_maps`: sums `scale x scale` blocks (`scale = 1/tech_area_res = 5`) then Gaussian filter sigma = 2.5 px
  - the current maps of the generated set are smoothed reference-design maps. Real-circuit maps are not smoothed this way
  (inferred; not documented).

### 1.8 Loader spec for P3 (ICCAD'23)

```
case = {name, group in {fake, real, hidden}, netlist_path, maps: {current, eff_dist, pdn_density, ir_drop} each float64 (H, W)}
read: np.loadtxt(path, delimiter=',')  # or a faster C-parser; ~(H*W) numbers, files up to 930x930
fake name  current_map{NN}  -> files current_map{NN}.sp, current_map{NN}_{current,eff_dist,pdn_density,ir_drop}.csv
real name  testcase{K}      -> netlist.sp, current_map.csv, eff_dist_map.csv, pdn_density.csv, ir_drop_map.csv
hidden test cases are the 10 in hidden-real-circuit-data/list_testcase.txt (7,8,9,10,13,14,15,16,19,20).
```

---------------------------------------------------------------------------------------------------------------

## 2. CircuitNet-N14 (HF `CircuitNet/CircuitNet`, directory `CircuitNet-N14/`)

See `data_cards.md` for the file list and hashes. Revision pinned to HF commit
`df74f33e302f927bf0548dc0d7936ca43b634f2a`. Original loader code: GitHub `circuitnet/CircuitNet` at commit
`41ade1d7e913a418b4495b6b13630b26b0ff7279` (2026-05-17), directory `routability_ir_drop_prediction/` and scripts copied under
`E:\edalint_data\docs_cache\cn_gh\`.

### 2.1 What is on disk (HF layout)

```
CircuitNet-N14/IR_drop_features/<design>.tar.gz        8 designs; members ./<design>/{IR_drop,power_i,power_s,power_sca,power_all,power_t}/<sample>.npz
CircuitNet-N14/routability_features/<design>.tar.gz    8 designs; members ./<design>/ with, per sample (.npz):
    macro_region/ cell_density/
    RUDY/{RUDY,RUDY_long,RUDY_short,RUDY_pin,RUDY_pin_long}/
    congestion/congestion_early_global_routing/{overflow_based/congestion_eGR_{horizontal,vertical}_overflow,
                                                utilization_based/congestion_eGR_{horizontal,vertical}_util}/
    congestion/congestion_global_routing/{overflow_based/congestion_GR_{horizontal,vertical}_overflow,
                                          utilization_based/congestion_GR_{horizontal,vertical}_util}/
    DRC/DRC_all/            DRC/DRC_seperated/<violation type>/   (only samples with that violation have a file)
```

- Design names: `RISCY`, `RISCY-FPU`, `zero-riscy`, `nvdla-small`, `nvdla-large`, `openc910-1`, `Vortex-small`, `Vortex-large`.
  In the two subsets used here every design is a single `.tar.gz` (the `-part-aa..` splitting in the HF README concerns other subsets).
- Sample name: `{design}_freq_{f}_mp_{m}_fpu_{u}_fpa_{ar}_p_{p}_fi_{ap|ar}.npz`, identical across feature directories of one design
  (the sets differ by a few missing files, see data card).
- Each `.npz` has one key, `data`, float64. Shapes: `(H, W)`; `power_t` `(20, H, W)` (axis 0 = time slice).
  `H` and `W` differ per sample (e.g. Vortex-small: 403 to 464) and need not be equal. All maps of one sample share `(H, W)`
  **except the `congestion_eGR_*` maps**, which are on a finer grid (about 1.8x: 832 x 830 vs 459 x 456 observed).
  Axis meaning is not documented; the FAQ says that to visualise one must rotate the image 90 degrees clockwise (`np.rot90`)
  because numpy and image coordinates differ, i.e. axis 0 is probably x (inferred, E005 test needed).
- Expanded size is about 10x the tarball (npz stored uncompressed): do not extract everything. Read members with `tarfile`
  streaming (`tar.extractfile(member)` then `np.load(io.BytesIO(...))`), or extract one design at a time to a scratch directory.
  Extracted copies made in P1 (deletable): `E:\edalint_data\circuitnet_n14_x\{routability_features,IR_drop_features}\Vortex-small`.

### 2.2 Original loader (GitHub `routability_ir_drop_prediction/`, commit 41ade1d)

Pre-processing, `preprocess_scripts/generate_training_set.py --task {congestion,DRC,IR_drop} --data_path <decompressed> --save_path <out>`:

- `resize(x)` = `scipy.ndimage.zoom(x, (256/H, 256/W), order=3)` (cubic); `resize_cv2(x)` = `cv2.resize(x, (256,256), cv2.INTER_AREA)`;
  `std(x)` = `(x - x.min()) / (x.max() - x.min())` per map per sample, returned unchanged if `x.max() == 0`.
- *congestion*: features `[macro_region, RUDY/RUDY, RUDY/RUDY_pin]`, each `std(resize(.))`; label = `std( resize(GR_horizontal_overflow) +
  resize(GR_vertical_overflow) )` (a single summed channel).
- *DRC*: features `[macro_region, cell_density, RUDY_long, RUDY_short, RUDY_pin_long, eGR_horizontal_overflow, eGR_vertical_overflow,
  GR_horizontal_overflow, GR_vertical_overflow]`, each `std(resize(.))`; label `DRC_all`: `clip(label, 0, 200)`, `resize_cv2`, `/ 200`.
- *IR_drop*: features `[power_i, power_s, power_sca, power_all, power_t]`; for `power_t` each of the 20 slices separately
  `std(resize_cv2(slice))`, the others `std(resize_cv2(x.squeeze()))` -> 24 channels; label `IR_drop`: `squeeze`, `clip(label, 1e-6, 50)`,
  `(log10(resize_cv2(label)) + 6) / (log10(50) + 6)`.
- Saved as `feature/<name>.npy` of shape `(256, 256, C)` and `label/<name>.npy` `(256, 256, C_label)` (channel last; `np.transpose(out, (1,2,0))`).
  Note the script calls `np.load(path)` and then array methods: with the HF `.npz` the result is an `NpzFile` and would fail without `['data']`
  (not tested; the script targets the Google Drive `.npy` layout).
- Datasets (`datasets/*_dataset.py`): rows of the split CSV are `feature/<name>.npy,label/<name>.npy`; `np.load(...).transpose(2, 0, 1).astype(float32)`
  gives `(C, 256, 256)`; the IR-drop dataset adds a leading axis: feature `(1, 24, 256, 256)`, label `(256, 256)`; congestion
  and DRC give label `(1, 256, 256)`. Augmentation for congestion/DRC: horizontal `Flip` (config `aug_pipeline=['Flip']`).
- Models and config (`utils/configs.py`, defaults are the N28 values): GPDL (3 in, 1 out, MSE, lr 2e-4, batch 16), RouteNet (9 in, 1 out, MSE,
  lr 2e-4, wd 1e-4, batch 8, DRC threshold 0.1), MAVI (`in_channels` 1 with the 24 channels as depth, `out_channels` 4, L1, lr 2e-4, wd 1e-2, batch 2, hotspot threshold
  0.9885 "5% after log"), 200k iterations.
- Metrics (`utils/metrics.py`): NRMS, SSIM, EMD for congestion; NRMS, SSIM for DRC/IR; ROC AUC and PRC on 200 thresholds in `linspace(0, 1, 200)`
  against the label thresholded at the task threshold. Computed on the 256 x 256 resized/normalised maps, not on native resolution.
  `test.py` adds a sample's metric to the running sum only if it is not exactly 1 (`if not metric(...) == 1`) but divides by the
  full dataset length, so a perfect SSIM would count as 0 (possible E006 metric artefact; read from the source, not run).

### 2.3 Split files

`train_N14.csv` (6,790 rows) and `test_N14.csv` (3,544 rows): see the data card. The rows list the same sample names for all three
tasks. Mapping a row to files: strip `feature/`, replace `.npy` by `.npz`, look the stem up in the per-design directory of the
tarball for the task. Counting rows whose stem is present in the label directory and in all feature directories of each task is
done by `scripts/p1_n14_inventory.py` (file names only).

### 2.4 Loader spec for P3 (N14)

```
sample = {design, name, task_inputs: dict[name -> (H, W) float64 or (20, H, W)], label}
IR drop     inputs: power_i power_s power_sca power_all (H,W) + power_t (20,H,W)   label: IR_drop (H,W)
congestion  inputs: macro_region RUDY RUDY_pin                                      label: congestion_GR_{horizontal,vertical}_overflow
DRC         inputs: the 9 maps above                                                 label: DRC_all
native resolution is kept; resize to 256x256 only inside a baseline wrapper that reproduces the authors' preprocessing
```

---------------------------------------------------------------------------------------------------------------

## 3. CircuitNet-N28 (not obtained)

Per the authors' scripts the N28 files are `.npy` per sample with the same feature names and the same 256 x 256 preprocessing
(`generate_training_set.py` was written for this layout). Split CSVs `train_N28.csv` / `test_N28.csv` (design-level: train RISCY and
RISCY-FPU, test zero-riscy). Download not possible without bypassing the Google Drive quota (see data card).

## 4. CircuitNet 3.0 (not obtained; no map task)
