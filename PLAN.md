# edalint — plan (lead, 2026-10-12)

**Goal.** An open audit of the public ML-for-EDA "map" benchmarks, plus a reusable linter. The question:

> How much of the published deep-learning skill on ICCAD'23 Problem C (static IR drop) and CircuitNet (IR drop,
> congestion, DRC) is reproduced by trivial, linear and physics baselines? And do the benchmarks have data
> faults (leakage, duplicates, degenerate samples, axis faults, metric artefacts) that inflate it?

**Split.** About 75% tool (`edalint`: loaders, checks, baseline zoo, report) and 25% research (a pre-registered audit).
**Why it is open.** The 2026-10-10 deep scoop check found no published audit of these benchmarks with linear or physics
baselines. GIF (arXiv 2604.09999) reports all-zero bounding boxes in CircuitNet-N14 as an aside.
See `notes/2026-10-10-project-scouting.md` in the thesis repo.

## Principles (carried over from fieldlint v3/v4)
- **Pre-register before results.** Protocol, baselines, metrics and verdict criteria are committed and frozen **before** any
  baseline is evaluated on any test split. The ICCAD'23 "hidden" cases are public in the repo, so the freeze is what makes
  them held-out. Nobody looks at hidden-case error before the freeze.
- **Run once.** A runner refuses to overwrite result files. Anything after the freeze is labelled POST HOC.
- **Findings on clean data are adjudicated.** A "fault" counts only if confirmed by documentation, by a physics or
  identity check stated in advance, or by the maintainers. Severity is benign or material.
- **Real data only.** Data is read-only after download, with sha256 recorded.
- **Honest framing.** An exact solve of the ICCAD'23 SPICE netlist reproduces the labels by construction, because the
  labels come from a solver. So the exact solve is reported as the *runtime* reference, not as a skill claim. The skill
  claims rest on the learned and linear baselines, which see only the contest's input maps.

## Phases

| # | Phase | Who (AGENTS.md) | Output | Depends on |
|---|---|---|---|---|
| P0 | Repo skeleton (`D:\Work\edalint`, data on `E:\edalint_data`), this plan | Opus | PLAN.md, .gitignore | none |
| P1 | Data acquisition and format inspection only (no statistics, no baselines) | Sonnet (Haiku for download checks) | `data_cards.md`, sha256 manifest, loaders' format notes | P0 |
| P2 | Literature table: published numbers to compare against (per paper: split, metric definition, MAE/F1/runtime, test cases) | Sonnet (Haiku first-pass search) | `docs/published_numbers.md` with citations | P0 (parallel with P1) |
| P3 | Tool build: package skeleton, loaders, checks, baseline zoo, report; tests on synthetic data | Sonnet | `src/edalint/`, `tests/` | P1 formats |
| P4 | Pre-registration (protocol, hypotheses, criteria), lead review, **freeze** | Sonnet drafts, Opus reviews | `protocol.md` frozen + hash of code | P1, P2, P3 |
| P5 | Registered runs (once) | Sonnet | `results/` | P4 |
| P6 | Report + independent number check | Sonnet writes, Haiku verifies | `REPORT.md` | P5 |
| P7 (optional) | Re-train one published model (e.g. a U-Net on ICCAD'23) on Kaggle T4 to get same-split numbers | Sonnet | `results/repro/` | P4, Kaggle quota |

## Datasets (priority order)
1. **ICCAD 2023 Problem C** (github.com/ASU-VDA-Lab/ML-for-IR-drop, BSD-3, about 650 MB on GitHub).
   - About 100 fake (BeGAN) training cases, 10 real training cases and 10 "hidden" real test cases.
   - Inputs: SPICE PDN netlist, current map, PDN density map, effective-distance map.
   - Label: IR-drop map.
   - Metrics: MAE, F1 at a 90%-of-max hotspot threshold, runtime.
2. **CircuitNet-N14** (HF `CircuitNet/CircuitNet`, BSD-3): only the IR_drop_features (~51 GB) and routability_features
   (~26 GB) subsets. Extract one design at a time.
3. **CircuitNet-N28** (Google Drive / Baidu): size unverified. Time-box the access check to 1 hour; skip if not
   reachable without an account.
4. **CircuitNet 3.0** (HF, 1 GB): check only whether it carries map tasks. Not a target unless it does.

## Checks (`edalint` rules, first draft; final list frozen in P4)
- **E001 integrity:** finite values, consistent shapes, every sample loads.
- **E002 degenerate samples:** all-zero or constant maps, all-zero bounding boxes (the GIF N14 finding), negative IR drop.
- **E003 duplicates:** exact and near-duplicate inputs and labels, within and across splits.
- **E004 split leakage:** the same design or netlist across train/test; random-versus-design split gap.
- **E005 axis/orientation:** label versus input correlation as stored against transposed or flipped.
- **E006 metric sanity:** hotspot base rate, F1 sensitivity to the threshold, empty-hotspot cases, MAE scale versus label range.
- **E007 baseline ladder** (informational): skill of trivial/linear/physics baselines against the published numbers.

## Baseline zoo
mean map → per-pixel linear on the input maps (ridge) → kNN on downsampled inputs → effective-distance regression
→ Green's-function / superposition from the current map → small classical models (gradient boosting on per-pixel
features) → **exact sparse solve** of the SPICE PDN (runtime reference only; ICCAD'23 only).

## Compute and constraints
- CPU-first: at most 2 CPU jobs at once, shared with other work on this machine. Data on E:.
- Kaggle T4 only for P7, and only if the weekly quota allows.
- No paid services. Downloads are free (GitHub, HF). No account-gated data unless the user approves.
- Commits: explicit paths, no AI attribution. Push only when the user asks. Stop processes by PID only.

## Pre-stated criteria idea (to be finalised in P4)
- **"Substantial share of skill reproduced":** the best non-deep baseline reaches ≥ X% of the best published model's
  improvement over the mean-map baseline, on the same split and metric.
- **"Material data fault":** any confirmed cross-split duplicate, leakage or degenerate-sample rate ≥ Y% on a test split.
- X and Y are fixed in P4 before any test-split evaluation.
