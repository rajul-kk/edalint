"""P9 linter checks L001-L003, same contract as edalint.checks (check(ds, opt) -> CheckResult; never raises on bad data).

They live outside src/edalint so that edalint.protocol.code_hash() does not move (CODE_HASH_LINEAGE.json); `all_checks()` returns the
union with edalint.checks.CHECKS. Merging them into edalint.checks is an owner decision (like amendment A7 of protocol_faults.md).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from edalint.checks import CHECKS, Options, _accessible
from edalint.core import CheckResult, MapDataset, skipped

from . import leakage as LK


@dataclass
class P9Options(Options):
    split_files: Dict[str, str] = field(default_factory=dict)      # L003: {split name: path of a split CSV / list file}
    nn_grid: int = 32


def _collect(ds: MapDataset, split: str, opt: P9Options):
    ids, X, Y, des = [], [], [], []
    for sid in ds.ids(split, opt.dup_max_samples):
        s = ds.get(sid)
        if s is None:
            continue
        x, y = LK.describe(s, opt.nn_grid)
        ids.append(sid)
        X.append(x)
        Y.append(y)
        des.append(ds.meta(sid).get('design', ''))
    if not ids:
        return ids, None, None, des
    return ids, np.stack(X), np.stack(Y), des


def _ref_and_queries(ds: MapDataset, opt: P9Options):
    acc = _accessible(ds, opt)
    acc = [s for s in acc if ds.splits.get(s)]
    ref = [s for s in acc if s not in ds.test_splits][:1]
    qs = [s for s in acc if s not in ref]
    return ref, qs


def l001_nn_near_duplicates(ds: MapDataset, opt: P9Options) -> CheckResult:
    """L001 nearest-neighbour near duplicates across splits (P9, design D1-D3).

    Why: E003 counts pairs above 0.999 among at most dup_max_samples per split; L001 gives, for EVERY query row of the other splits, its
    nearest reference (first non-test split) row under the E003 descriptor, the NN similarity distribution (input and label), counts at the
    primary 0.999 and the sensitivity thresholds 0.99 / 0.95, and confirms flagged pairs at full resolution (max abs difference <= 0.01 on
    every channel and the label). Only a confirmed pair is an error; a flagged, unconfirmed pair is a warning.
    Does NOT apply: needs two accessible splits (a test split only after the freeze); samples on different grids are never confirmed.
    """
    res = CheckResult('L001', 'nn-near-duplicates')
    ref, qs = _ref_and_queries(ds, opt)
    if not ref or not qs:
        return skipped('L001', 'nn-near-duplicates', f'needs a reference split and a query split, accessible: {_accessible(ds, opt)}')
    rid, RX, RY, _ = _collect(ds, ref[0], opt)
    if RX is None:
        return skipped('L001', 'nn-near-duplicates', 'empty reference split')
    for q in qs:
        qid, QX, QY, _ = _collect(ds, q, opt)
        if QX is None:
            continue
        SI = LK.similarity(QX, RX)
        SL = LK.label_similarity(QY, RY)
        recs = LK.nn_records(qid, rid, SI, SL)
        counts = LK.threshold_counts(recs)
        res.metrics[f'{q}_vs_{ref[0]}'] = {'counts': counts,
                                           'nn_input_sim_quantiles': np.quantile([r['input_sim'] for r in recs], [0, .5, .9, .99, 1]).tolist()}
        flagged = [r for r in recs if r['input_sim'] >= LK.DUP_CORR and r['label_sim'] >= LK.DUP_CORR]
        conf, unconf = [], []
        for r in flagged:
            c = LK.confirm_pair(ds.get(r['id']), ds.get(r['nn']))
            (conf if c['confirmed'] else unconf).append(f"{r['id']}~{r['nn']}")
        if conf:
            res.add('error', f'{len(conf)} {q} row(s) have a CONFIRMED near duplicate in {ref[0]} (descriptor >= {LK.DUP_CORR}, full-resolution '
                    f'max |diff| <= {LK.CONFIRM_MAXABS})', conf, n=len(conf))
        if unconf:
            res.add('warning', f'{len(unconf)} {q} row(s) have a flagged, unconfirmed near duplicate in {ref[0]} (descriptor >= {LK.DUP_CORR})',
                    unconf, n=len(unconf))
        io = counts[f'{LK.DUP_CORR:g}']['nn_input_only']
        if io:
            res.add('info', f'{io} {q} row(s) have a near-identical input in {ref[0]} with a different label (input >= {LK.DUP_CORR}, label below)')
    res.summary = '; '.join(f"{k}: {v['counts'][f'{LK.DUP_CORR:g}']['nn_pair_ge']} of {v['counts'][f'{LK.DUP_CORR:g}']['n_queries']} NN pairs >= {LK.DUP_CORR}"
                            for k, v in res.metrics.items() if isinstance(v, dict) and 'counts' in v)
    return res.finalise()


def l002_design_proximity(ds: MapDataset, opt: P9Options) -> CheckResult:
    """L002 design-family proximity (P9, design D4).

    Why: a design-level split still leaks if a test design is a variant of a train design (same RTL family, small/large configuration).
    For each query design: S_same / S_other = median NN input similarity of its rows to same-family / other-family reference rows. The
    reference scale is the within-design NN similarity of the reference designs (placement variants of one netlist, leave-one-out).
    Material (warning, provisional unless the family relation is documented) if S_same >= the 25th percentile of that reference and
    S_same - S_other >= 0.02.
    Does NOT apply: needs design metadata, at least one reference design with >= 2 rows, and a query split.
    """
    res = CheckResult('L002', 'design-proximity')
    ref, qs = _ref_and_queries(ds, opt)
    if not ref or not qs:
        return skipped('L002', 'design-proximity', 'needs a reference split and a query split')
    rid, RX, RY, rdes = _collect(ds, ref[0], opt)
    if RX is None:
        return skipped('L002', 'design-proximity', 'empty reference split')
    rdes = np.array(rdes)
    within = {}
    for d in sorted(set(rdes)):
        m = np.flatnonzero(rdes == d)
        if len(m) >= 2:
            S = LK.similarity(RX[m], RX[m])
            within[d] = LK.nearest(S, exclude_self=True)[1]
    if not within:
        return skipped('L002', 'design-proximity', 'no reference design with >= 2 rows')
    for q in qs:
        qid, QX, _, qdes = _collect(ds, q, opt)
        if QX is None:
            continue
        qdes = np.array(qdes)
        rows = {d: LK.similarity(QX[qdes == d], RX) for d in sorted(set(qdes))}
        prox = LK.design_proximity(rows, sorted(set(rdes)), sorted(rows), within, list(rdes))
        res.metrics[f'{q}_vs_{ref[0]}'] = prox
        for d, r in prox.items():
            if r['material']:
                res.add('warning', f"{q} design {d} is as close to same-family {ref[0]} design(s) {r['same_family_train_designs']} as placement "
                        f"variants of one design are to each other ({r['reason']})", design=d)
            elif r['S_same'] is not None:
                res.add('info', f'{q} design {d}: family proximity not material ({r["reason"]})', design=d)
    res.metrics['within_design_reference'] = {d: {'median': float(np.median(v)), 'q25': float(np.quantile(v, .25)), 'n': int(len(v))}
                                              for d, v in within.items()}
    res.summary = f'{sum(1 for f in res.findings if f.severity == "warning")} design(s) with material family proximity'
    return res.finalise()


def l003_published_splits(ds: Optional[MapDataset], opt: P9Options) -> CheckResult:
    """L003 design overlap in published split files (metadata only; P9 design D5).

    Why: a published split described as design-wise may still put one design on two sides. Exact design-name overlap between any two split
    files is an error (identity check on names); overlap of RTL families only is info.
    Does NOT apply: needs opt.split_files with at least two files.
    """
    if len(opt.split_files) < 2:
        return skipped('L003', 'published-splits', 'needs at least two split files (opt.split_files)')
    res = CheckResult('L003', 'published-splits')
    rows = {s: Path(p).read_text(encoding='utf-8', errors='replace').splitlines() for s, p in opt.split_files.items()}
    out = LK.split_file_overlap(rows)
    res.metrics.update(out)
    for p in out['pairs']:
        a, b = p['splits']
        if p['design_overlap']:
            res.add('error', f"design(s) {p['design_overlap']} occur in both {a} and {b} (rows {p['rows_in_overlap']})",
                    p['design_overlap'], splits=[a, b])
        fam_only = [f for f in p['family_overlap'] if f not in {LK.family(d) for d in p['design_overlap']}]
        if fam_only:
            res.add('info', f"RTL families {fam_only} on both sides of {a}/{b} (different design names)", splits=[a, b])
    res.summary = f"{sum(1 for p in out['pairs'] if p['design_overlap'])} split pair(s) with a shared design"
    return res.finalise()


P9_CHECKS: Dict[str, Callable] = {'L001': l001_nn_near_duplicates, 'L002': l002_design_proximity, 'L003': l003_published_splits}
P9_NAMES = {'L001': 'nn-near-duplicates', 'L002': 'design-proximity', 'L003': 'published-splits'}


def all_checks() -> Dict[str, Callable]:
    return {**CHECKS, **P9_CHECKS}
