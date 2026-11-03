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
None.
