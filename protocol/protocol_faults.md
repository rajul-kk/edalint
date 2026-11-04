# edalint P8: fault-injection validation of checks E001-E005 (pre-registration)

**STATUS: written 2026-11-04, binding from the commit that adds it.** No injection run, false-positive run or IC-ThermBench check run exists at this date.
Thresholds are those of `edalint.checks.Options` at the parent commit (recorded in the result file as `code_hash`). They are NOT changed by this protocol.

## 1. Question
How strong must each fault be before E001-E005 flag it, and how often do the checks flag clean real data? E006 (diagnostic) and E007 (informational) are not fault detectors and are excluded.

## 2. Disclosure
- The IC-ThermBench transpose (power and k stored transposed against temperature) was found on 2026-10-05 with fieldlint, on scopes S3 and S4. It is known before this run, so
  the E005 result on those sets is a **recall check**, not a discovery.
- No injected-fault detection rate has been computed for any check before this protocol. The existing unit tests assert one fault at one severity.
- The train/val boundary of the IC-ThermBench index split falls inside case 8 (1,500 samples per case, 72% / 8% / 20%) by arithmetic from the README. This has not been checked
  against the data and is reported only after it is.

## 3. Base data (clean = no known fault, not verified fault-free)
| key | data | notes |
|---|---|---|
| iccad23 | ICCAD'23 train (fake + real, existing loader) | |
| n14ir | CircuitNet-N14 IR-drop train, the registered fit-sample subset, cache `E:/edalint_cache` | |
| icb | IC-ThermBench `Thermo/data/ic-thermbench/datasets`, scopes level2..level5, 64 x 64, Z = 1 | label = temperature, inputs = power, k and scalar channels; orientation **corrected** (spatial input channels swapped in-plane) for the clean base; as-stored for the natural-positive run |
Only non-test splits are read (train, val). No test split is opened. IC-ThermBench path is a CLI argument.

## 4. Fault matrix
Each fault is injected into copies of the clean base, at five severity levels, for R = 20 independent draws (draw d uses seed 20261104 + d: sample subset of 200 and injection sites).

| Check | Fault | Severity levels (weakest to strongest) |
|---|---|---|
| E001 | NaN pixels in one input channel of one sample | 1 px, 4 px, 16 px, 64 px, 1% of pixels |
| E001 | Inf pixels, same placement | same levels |
| E001 | one input channel resized to a different grid in one sample | 1 level (present / absent) |
| E002 | all-zero label | fraction of sampled cases affected: 0.2%, 0.5%, 1%, 2%, 5% |
| E002 | one negative label pixel | same fractions |
| E002 | one input channel set all-zero | same fractions |
| E003 | exact copy of a train sample into a second split | 1, 2, 5, 10, 20 copies |
| E003 | near-copy (input and label plus Gaussian noise, correlation at the E003 grid) | target correlation 0.99, 0.995, 0.999, 0.9995, 0.99999 (threshold 0.999) |
| E004 | a design moved into the other split (metadata only) | N = 20 designs: 1, 2, 3, 5, 8 moved; N = 5: 1 moved |
| E004 | a twin pair split across splits (ICCAD only) | 1, 2, 4, 8, 16 pairs |
| E005 | label transposed (square maps only) | fraction of samples affected 10%, 25%, 50%, 75%, 100% |
| E005 | label flipped up-down | same fractions |
| E005 | label transposed, 100% of samples, plus label noise | SNR 20, 10, 5, 2, 1 (signal std over noise std) |
Not injectable on a base: E004 on `icb` uses the case as design, only if the loader records it (section 6).

## 5. Metrics and criterion (fixed now)
- Detection = the target check returns `flagged` with an `error` or `warning` finding of the intended kind. An unrelated finding does not count.
- Per (check, fault, severity): detection rate over R = 20 draws, Wilson 95% interval. Report the weakest severity with rate >= 0.90 ("minimum detectable severity"), or "none".
- False-positive rate: each check on 20 random subsamples (200 samples, no injection) of each clean base. A false positive is an `error` or `warning` finding that is not in the
  known-findings list (section 7). Report the rate per check and base with a Wilson interval.
- **A (check, fault) pair is "validated"** if the detection rate is >= 0.90 at both of the two strongest severities AND the check's false-positive rate on every clean base is <= 0.05.
  Every other pair is reported as a limit. Fewer than R draws completing (e.g. a loader error) is reported, not hidden.
- No threshold or `Options` value may be changed after the first run. A change needs a dated amendment (section 9) and a rerun on seeds 20261204 + d.

## 6. IC-ThermBench loader
New `edalint.datasets.icthermbench`: inputs = power map and k map(s) (plus scalar channels as constant maps), label = temperature, primary input = power. Design = case: level5 from
`manifest.json`; levels 2-4 from the README layout (Case1-10 x 1500, contiguous) and flagged `case_source: readme`. Splits as the benchmark's own index split (72/8/20 train/val/test),
test held as a test split.

## 7. Known findings (listed before any false-positive run)
- IC-ThermBench: E002 `warning` for constant non-zero scalar channels (ambient_K, h_w_m2k, r_convec_k_per_w) in every sample.
- ICCAD'23 and N14: the existing findings in `results/dev/*_p3dev.md` and `results/n14/tables.md` (read and listed in the result file before the run).
- IC-ThermBench as stored: E005 expected to flag the power channel (the natural positive). Natural negative: the corrected base.

## 8. Outputs
`results/faults/faults_p8.json` (run-once, with git HEAD, code hash, library versions), `results/faults/tables_p8.md`, README section "What edalint does not detect".

## 9. Amendments
**A1 (2026-11-04, implementation clarifications; written before any injection, false-positive or IC-ThermBench check run; no threshold or criterion changes).**
1. E004 "design moved": for each of k designs ONE sample is reassigned to the other split, so the design occurs on both sides (moving a whole design would not create overlap). Clean layout = whole designs assigned to one side.
2. E004 design count: N = min(20, designs available in the pool); IC-ThermBench has 10 cases (5 in level5), so its N is reported per base. The N = 5 arm is run where the base has at least 5 designs.
3. Draws are taken from a pool of 600 evenly spaced non-test samples per base (read once; N14 tarball reads dominate), 200 per draw.
4. E003 near-copy: 5 near-copies per draw; noise is calibrated per input channel and the label so that the E003 descriptor correlation of each equals the target (bisection); the concatenated multi-channel correlation is then close to, not exactly, the target.
5. IC-ThermBench scopes level2..level5 are separate bases (`icb2`..`icb5`); the false-positive bound must hold on each.
6. Loader check, 2026-11-04: the train/val boundary of the benchmark's index split falls inside one case in every scope (Case8 on level2-4 by the README layout; Case19 on level5 by the manifest). Under design = case, E004 is expected to report this on the as-released data. It is reported as an observation about the benchmark split, not as an injected-fault result, and it enters the known-findings list for the false-positive run only for E004 on icb.
8. **A2 (2026-11-04, still before any run).** (a) Pre-existing defects: the train-only dev reports show real defects (CircuitNet-N14 IR drop: 1 of 80 samples has an all-zero label; ICCAD'23: 5 samples have a constant non-zero `pdn_density` map). A base that already triggers a check would make detection meaningless. Pool construction therefore removes samples with an all-zero or constant label, non-finite values, or negative label pixels, and the count is recorded as a known finding; the constant `pdn_density` map is kept (it does not match any needle below). (b) The E002 zero-channel needle is changed from `input channel` to `is all-zero in`, because the unrelated ICCAD finding "input channel 'pdn_density' is a constant non-zero map" would otherwise count as a detection. (c) Grouping for the clean two-split layout: ICCAD cases are grouped by `design_id` (twin partners share it), N14 by `design`, IC-ThermBench by case; the injected-fault design arms use the same groups. (d) Development uses only the in-memory synthetic base (key `synthetic`); no real base is run before the registered run.
10. **A3 (2026-11-04, after the registered IC-ThermBench run; written before the E003 code change and before any rerun).** Finding: E003's near-duplicate descriptor concatenates z-scored channels, and a constant channel contributes a zero vector, so a sample's self-similarity is (non-constant channels)/(channels). Verified on a synthetic pair: max input correlation 1.00 with no constant channel, 0.67 with one, 0.40 with three, against the threshold 0.999. On icb4/icb5 (3 constant scalar channels of 7) E003 near-duplicate detection is therefore impossible, and the registered result for `near_copy` on those bases (0/20 at every level) is a consequence of this defect. Registered results are not edited. Change: the input descriptor is averaged over non-constant channels only (a pair is compared on the channels that vary in at least one of the two samples; the dot product is divided by that channel count). No threshold changes (`dup_corr` stays 0.999). Rerun: the E003 faults and the E003 false-positive count on all bases with seeds 20261204 + d (20 draws), reported beside, not instead of, the original run. Success criterion of section 5 unchanged.
11. **A4 (2026-11-04, after the registered ICCAD'23 run; two harness defects, no check code involved).** (a) The N14 attempt produced no result: the pool screen used `label.min() < 0` and removed all 600 samples (rounding-level negatives). It now uses the E002 tolerance (`min < -max(1e-12, 1e-6 * max|label|)`). N14 is run with the registered checks (before the A3 change). (b) The E004 design-overlap injector reassigned one sample of a randomly chosen design; for a design with a single sample (ICCAD'23: the 100 fake cases are one design each) that cannot create an overlap, so the registered ICCAD'23 result for `design_overlap` (detection 0.00-0.35) measures the injector, not the check. The injector now chooses only designs with at least 2 samples, and k is capped at their number (ICCAD'23: 5 real designs). The registered ICCAD'23 `design_overlap` row stays in the file and is marked invalid; E004 `design_overlap` is rerun on iccad23 and n14ir with seeds 20261204 + d. (c) The runner gains `--seed0` and `--only` (restrict checks and faults) for the A3 and A4 reruns. (d) The summary script treats the ICCAD'23 constant `pdn_density` E002 warning as the known finding listed in section 7.
12. **A5 (2026-11-04, after the N14 attempt with the registered configuration ran out of memory; no N14 result existed).** N14 maps are 20 x 256 x 256 per sample, so a 600-sample pool with 200-sample copies per fault does not fit in RAM. For `n14ir` only: pool 300, 100 samples per draw, arrays held as float32 (`--float32`); 20 draws, seeds, faults, severities and the success criterion are unchanged. Consequences stated in advance: a draw of 100 samples gives E004 fewer designs per draw, and E005 reads the same 40 samples, so E005 detection at partial fractions is not directly comparable with the other bases.
13. **A6 (2026-11-04, freeze interaction found while implementing A3).** `protocol/FROZEN_n14.json` and `FROZEN_p7.json` bind to the sha256 of `src/edalint/**/*.py`. My first P8 commits put `faults.py` and `datasets/icthermbench.py` inside that package, which changed the hash and made the P7 freeze test fail (`test_real_repo_is_frozen_for_p7`); verified by restoring `checks.py` to c252323 and removing the two files, which reproduces the recorded hash `d3f6d2f6...` exactly. All P8 code now lives in `src/edalint_p8/` (outside the hashed package, like `edalint_p7`), `src/edalint/checks.py` is byte-identical to c252323, and the A3 correction is a variant check `edalint_p8.e003_fixed` used only by the reruns (`--e003-fixed`). The registered runs were produced with checks identical to c252323 (the P8 modules only inject and orchestrate). Merging the E003 correction into `edalint.checks` would change the hash again and needs an explicit re-freeze decision by the project owner; it is not done.
9. E002 zero-channel: the check reports a single all-zero input channel as `info` only, so this pair cannot reach the detection criterion by construction; it is run and reported as a limit.
14. **A7 (2026-11-04, after the A3/A4 reruns; decision by the project owner to merge).** The A3 correction is merged into `edalint.checks.e003_duplicates` (identical to the variant `edalint_p8.e003_fixed`, which stays as the record of what the reruns used; a test asserts the two agree). `code_hash()` moves from `d3f6d2f6...` to `a9900e6f...`. The freeze records `FROZEN_n14.json`, `FROZEN_p7.json` and `FROZEN.json` are NOT rewritten: they certify the code that produced the registered N14, P7 and P8 results (reproduce those at commit c252323). They no longer unlock test reads under the new code; P7 and N14 test reads are complete, and any new test read needs a new freeze. `protocol/CODE_HASH_LINEAGE.json` records both hashes and `tests/test_code_hash_lineage.py` pins the current one. Registered P8 result files are not edited; the E003 false-positive and detection numbers in `summary_p8r.md` are the ones that apply to the merged check.
