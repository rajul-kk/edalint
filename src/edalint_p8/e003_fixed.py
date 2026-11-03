"""E003 with the amendment-A3 correction (protocol/protocol_faults.md section 9, item 10): the near-duplicate input correlation is computed on the
channels that vary in at least one of the two samples. Registered edalint.checks.e003_duplicates is NOT modified (the N14 and P7 freezes hash
src/edalint); this variant is what the P8 reruns use. Merging it into edalint.checks needs a re-freeze decision by the project owner.
"""
from collections import defaultdict

import numpy as np

from edalint.checks import Options, _accessible, _hash, _zdesc
from edalint.core import CheckResult, MapDataset, area_resize, skipped


def e003_duplicates(ds: MapDataset, opt: Options) -> CheckResult:
    """E003 exact and near-duplicate samples, within and across splits.

    Why: a held-out sample that is (nearly) a copy of a training sample measures memorisation, not generalisation; the ICCAD'23
    contest itself removed duplicates from the fake set in v4. Exact = byte-identical inputs (all channels) or labels; near =
    Pearson correlation >= dup_corr of the maps area-resized to dup_grid^2, for inputs AND labels. Channel-level exact duplicates
    (e.g. the same current map in two samples) are reported as info: ICCAD real "twin" cases share a current map by design.
    Does NOT apply: degenerate all-zero inputs are excluded (E002 reports them); near-duplicate scans use at most dup_max_samples
    evenly spaced samples per split. Cross-split comparison needs the test split, so it runs only after the freeze.
    """
    res = CheckResult('E003', 'duplicates')
    splits = _accessible(ds, opt)
    items = []                                   # (id, split, input_hash, label_hash, input_desc, label_desc)
    chan_hash = defaultdict(list)
    for split in splits:
        for sid in ds.ids(split, opt.dup_max_samples):
            s = ds.get(sid)
            if s is None:
                continue
            allzero = all(np.all(a == 0) for _, a in s.channels())
            ih = None if allzero else _hash(np.concatenate([a.ravel() for _, a in s.channels()]))
            for nm, a in s.inputs.items():
                if not np.all(a == 0):
                    chan_hash[(nm, _hash(a))].append(sid)
            idesc = np.stack([_zdesc(a if a.shape == s.shape else area_resize(a, s.shape), opt.dup_grid) for _, a in s.channels()])   # (C, g*g)
            items.append((sid, split, ih, _hash(s.label), idesc, _zdesc(s.label, opt.dup_grid)))
    n = len(items)
    res.metrics['n_checked'] = n
    if n < 2:
        return skipped('E003', 'duplicates', f'fewer than 2 samples in {splits}')
    spl = {i[0]: i[1] for i in items}
    for kind, col in (('input', 2), ('label', 3)):
        groups = defaultdict(list)
        for it in items:
            if it[col] is not None:
                groups[it[col]].append(it[0])
        dups = [g for g in groups.values() if len(g) > 1]
        res.metrics[f'exact_{kind}_groups'] = len(dups)
        for g in dups:
            cross = len({spl[i] for i in g}) > 1
            res.add('error' if cross else 'warning',
                    f'{len(g)} samples have byte-identical {kind}s ({"ACROSS splits " + str(sorted({spl[i] for i in g})) if cross else "within one split"})',
                    g, kind=kind, cross_split=cross)
    ch = [(k, v) for k, v in chan_hash.items() if len(v) > 1]
    if ch:
        names = sorted({k[0] for k, _ in ch})
        res.add('info', f'{len(ch)} channel-level exact duplicates across samples (channels {names}); expected when samples share a configuration '
                '(ICCAD twin cases share a current map, CircuitNet runs with the same macro placement share macro_region)',
                [i for _, v in ch[:1] for i in v], n_groups=len(ch), channels=names)
    # near duplicates
    ok = [i for i in items if np.any(i[4] != 0)]
    if len(ok) >= 2:
        X = np.stack([i[4] for i in ok])                                        # (n, C, g*g), z-scored per channel (constant channel = zeros)
        Y = np.stack([i[5] for i in ok]) / np.sqrt(len(ok[0][5]))
        # Amendment A3 of protocol_faults.md: a pair is compared on the channels that vary in at least one of the two samples. A constant
        # channel is a zero vector and used to dilute the correlation to (non-constant channels / channels), which made near-duplicates of
        # any input with constant channels undetectable.
        vary = (X != 0).any(-1)
        n_var = (vary[:, None, :] | vary[None, :, :]).sum(-1)
        Ci = np.einsum('icg,jcg->ij', X, X) / X.shape[-1] / np.maximum(n_var, 1)
        Cl = Y @ Y.T
        iu = np.triu_indices(len(ok), 1)
        both = (Ci[iu] >= opt.dup_corr) & (Cl[iu] >= opt.dup_corr)
        in_only = (Ci[iu] >= opt.dup_corr) & (Cl[iu] < opt.dup_corr)
        res.metrics.update({'n_near_dup_pairs': int(both.sum()), 'n_same_input_different_label_pairs': int(in_only.sum()),
                            'max_input_corr_offdiag': float(Ci[iu].max()), 'max_label_corr_offdiag': float(Cl[iu].max())})
        pairs = [(ok[a][0], ok[b][0]) for a, b in zip(iu[0][both], iu[1][both])]
        cross = [p for p in pairs if spl[p[0]] != spl[p[1]]]
        within = [p for p in pairs if spl[p[0]] == spl[p[1]]]
        if cross:
            res.add('error', f'{len(cross)} near-duplicate pair(s) ACROSS splits (input and label corr >= {opt.dup_corr})',
                    [f'{a}~{b}' for a, b in cross], n_pairs=len(cross))
        if within:
            res.add('warning', f'{len(within)} near-duplicate pair(s) within a split (input and label corr >= {opt.dup_corr})',
                    [f'{a}~{b}' for a, b in within], n_pairs=len(within))
        if in_only.any():
            pr = [f'{ok[a][0]}~{ok[b][0]}' for a, b in zip(iu[0][in_only], iu[1][in_only])]
            res.add('info', f'{int(in_only.sum())} pair(s) with near-identical inputs but different labels (label corr < {opt.dup_corr})', pr)
    res.summary = (f'{n} samples: {res.metrics.get("exact_input_groups", 0)} exact input-duplicate groups, '
                   f'{res.metrics.get("exact_label_groups", 0)} exact label-duplicate groups, {res.metrics.get("n_near_dup_pairs", 0)} near-duplicate pairs')
    return res.finalise()
