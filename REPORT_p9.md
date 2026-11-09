# P9: leakage audit and calibrated uncertainty, registered report

Protocol `protocol/protocol_p9.md` (frozen in `bc0858c`, freeze record with code hashes `f745cbd`). Registered results: `results/p9/leakage_*.json`,
`results/p9/uq_*.json`, `results/p9/verdicts_p9.json` and `tables_p9.md` (verdicts committed in `7cfa396`). Every number below comes from those
files. Development used training data only (`results/dev/*_p9dev.json`); nothing in it is a test number.

## 1. Verdicts

**Leakage (all four targets): nothing found.**

| check | N14 IR drop | N14 congestion | N14 DRC | ICCAD'23 |
|---|---|---|---|---|
| LK1 near-duplicate layouts across splits | not found (0 of 3,544 test rows) | not found (0 of 3,544) | not found (0 of 3,543) | not found |
| LK2 material proximity of a test design to training designs | none of 3 designs | none of 3 | none of 3 | n/a |
| LK3 test designs in published training lists | none (CircuitNet official N14/N28, CLDRoute N14/N28) | | | |

LK1 is a **lower bound**: the near-duplicate scan covered part of the training set (protocol section 5.1). LK2's same-design similarity for
zero-riscy (0.59 IR drop, 0.70 congestion, 0.40 DRC) stays below each task's line (q25 of the pooled reference: 0.74, 0.87, 0.50).

**Calibrated uncertainty** (`crc_norm`, nominal 0.9; a task-level statement needs the hypothesis TRUE for both ridge and gbm):

| hypothesis | N14 IR drop | N14 congestion | ICCAD'23 |
|---|---|---|---|
| U1 implementation validity (T_id coverage in band) | TRUE (both) | TRUE (both) | crc_norm TRUE; **crc_abs FALSE for both** (anomaly, see 3) |
| U2 naive calibration under-covers new designs | **FALSE** (1 of 3 designs below 0.85, 2 needed) | **FALSE** (ridge 1, gbm 0) | **TRUE** (hidden set below 0.85) |
| U3 cross-design calibration repairs it | **FALSE** | **FALSE** | **TRUE** |
| U4 raw bootstrap ensembles under-cover | **TRUE** | **TRUE** | **TRUE** |

## 2. Numbers behind the UQ verdicts

| target / model | T_id crc_norm | T_id ens_gauss (U4) | per design, C_id calibration / C_xd calibration | mean abs gap C_id -> C_xd | C_xd macro coverage |
|---|---|---|---|---|---|
| N14 IR drop ridge | 0.920 | 0.0003 | zero-riscy 0.928/0.803; openc910-1 0.900/0.726; Vortex-large 0.739/0.539 | 0.063 -> 0.211 | 0.689 |
| N14 IR drop gbm | 0.914 | 0.054 | zero-riscy 0.904/0.884; openc910-1 0.945/0.931; Vortex-large 0.800/0.745 | 0.050 -> 0.067 | 0.854 |
| N14 congestion ridge | 0.888 | 0.016 | zero-riscy 0.965/0.970; openc910-1 0.935/0.939; Vortex-large 0.786/0.795 | 0.071 -> 0.071 | 0.901 |
| N14 congestion gbm | 0.914 | 0.158 | zero-riscy 0.988/0.985; openc910-1 0.983/0.980; Vortex-large 0.866/0.858 | 0.068 -> 0.069 | 0.941 |
| ICCAD'23 ridge | 0.957 | 0.106 | hidden 0.644/0.963 | 0.256 -> 0.063 | 0.963 |
| ICCAD'23 gbm | 0.955 | 0.137 | hidden 0.734/0.949 | 0.166 -> 0.049 | 0.949 |

## 3. Reading

- **The benchmarks look clean** on every leakage check run, within the lower-bound caveat of LK1.
- **Raw ensembles are badly over-confident everywhere** (U4): Gaussian intervals from 5 bootstrap members cover 0.0003-0.16 of pixels at
  nominal 0.9. The members barely disagree relative to their error, so ensemble spread alone is not an uncertainty estimate here. This is the
  firmest result of P9.
- **On N14 the usual calibration mostly survives the design shift.** Only Vortex-large falls below 0.85 (IR drop 0.74 ridge,
  0.80 gbm; congestion ridge 0.79), and congestion gbm stays above it everywhere (Vortex-large 0.87). Calibrating on held-out
  designs does not help and, for IR drop ridge, makes it worse (macro 0.69), so U3 is FALSE for both N14 tasks.
- **On ICCAD'23 the shift to the hidden real cases breaks the usual calibration (0.64/0.73) and calibrating on held-out real cases repairs it
  (0.96/0.95).** The hidden set is 10 cases and the calibration set 10 visible real cases; treat the size of the repair loosely.
- **U1 anomaly:** on ICCAD'23 `crc_abs` covers 0.971 on T_id, above the band's upper edge 0.9625 (CRC bound for n = 15); `crc_norm` is inside.
  Over-coverage, not under-coverage: the guarantee side holds. With 15 calibration maps the absolute-width variant is coarse; it is reported, not
  explained away.
- These are statements about ridge/gbm baselines and the calibration machinery under design shift, not about deep models.

## 4. Run history and deviations (protocol sections 7-8)

- `uq_n14_ir_drop_ridge`: stopped by the host for low memory after calibration, before any test number; re-run once (deviation `2e906a9`).
- `uq_n14_ir_drop_gbm`: stopped twice at the same step (zero-riscy evaluation) after its first test line had been logged (T_id 0.914), most
  likely by memory pressure. Section 7 forbids a re-run after a partial test number; the project owner authorised re-runs (`fd795a1`, `99ad9b9`).
  The third attempt, run detached with nothing else on the machine, reproduced every logged value of the earlier attempts exactly (calibration
  lambdas and T_id 0.914) before completing; the run is seeded and unchanged, so no choice was made after seeing a test number.
- All other registered runs completed on the first attempt. Registered UQ runs were executed one at a time.
