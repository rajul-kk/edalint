"""Checks E001-E007. Every check is `check(ds, opt) -> CheckResult`, carries a docstring stating (a) why it is justified and (b) when it
does NOT apply, and never raises on bad data: it reports. A check that cannot run returns status 'skipped' with a reason.

Label discipline: checks read only the splits named in `opt.split` (default 'train'). Test splits are reachable only when
protocol/FROZEN.json is valid (MapDataset.get enforces it). E004 uses metadata only (design ids, names) and may look across all
splits before the freeze.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import metrics as M
from .core import CheckResult, MapDataset, ProtocolError, Sample, area_resize, pearson, skipped


@dataclass
class Options:
    split: str = 'train'                    # split(s) the data checks read: a split name or 'all' (test needs FROZEN.json)
    max_samples: Optional[int] = 400        # evenly spaced subsample per split for per-sample checks (None = all)
    # E002
    neg_tol: float = 0.0                    # label below -neg_tol is negative
    all_zero_error_rate: float = 0.01       # a degenerate-sample rate at or above this is an error, below a warning
    # E003
    dup_max_samples: int = 800
    dup_grid: int = 32                      # near-duplicate descriptor: maps area-resized to dup_grid x dup_grid
    dup_corr: float = 0.999
    # E005
    orient_samples: int = 40
    orient_sigma: float = 2.0
    orient_grid: int = 64
    orient_consistency: float = 0.75        # primary channel: the alternative must win in at least this fraction of samples
    orient_margin: float = 0.10             # an alternative orientation must beat the stored one by this (median |corr|)
    # E004
    run_gap: bool = False                   # random-vs-design split gap: computed only when the protocol runs it
    gap_folds: int = 5
    gap_max_samples: int = 200
    # E006
    min_hotspot_px: int = 5                 # a case with fewer hotspot pixels makes F1 very noisy
    f1_fracs: Tuple[float, ...] = (0.85, 0.9, 0.95)
    f1_scales: Tuple[float, ...] = (0.95, 1.05)
    # E007
    fit_split: str = 'train'
    eval_split: Optional[str] = None
    baselines: Optional[Tuple[str, ...]] = None


def _short(x, n=5):
    return list(x)[:n]


def _accessible(ds: MapDataset, opt: Options) -> List[str]:
    """Split names the data checks may read: opt.split, or all (non-test unless frozen) when opt.split == 'all'."""
    if opt.split != 'all':
        return [opt.split]
    names = [s for s in ds.splits if s not in ds.test_splits and s not in ds.aux_splits]
    try:
        from .protocol import require_frozen
        require_frozen(ds.protocol_dir, what='checks on the test split')
        names += [s for s in ds.test_splits if s in ds.splits]
    except ProtocolError:
        pass
    return names


def _iter(ds: MapDataset, opt: Options, max_n: Optional[int]):
    """Samples of the accessible splits (opt.split), evenly subsampled per split to max_n; unloadable ones are skipped."""
    for split in _accessible(ds, opt):
        for sid in ds.ids(split, max_n):
            s = ds.get(sid)
            if s is not None:
                yield s


# -- E001 ------------------------------------------------------------------------------------------------------------------------
def e001_integrity(ds: MapDataset, opt: Options) -> CheckResult:
    """E001 every sample loads, values are finite, shapes are consistent.

    Why: every later check and every baseline indexes the maps pixel by pixel; a NaN/Inf silently poisons normalisation and losses,
    and inputs and label must share a grid. Note the authors' min-max `std()` divides by zero on a constant non-zero map (NaN/inf
    in `authors` mode); this check finds those.
    Does NOT apply: variable-resolution benchmarks legitimately vary in shape across samples (reported as info); inputs on a finer
    grid than the label (CircuitNet congestion_eGR_* in raw mode) are a warning, not an error, because Sample.stack() resizes them.
    """
    res = CheckResult('E001', 'integrity')
    bad_load, nonfinite, mismatch = [], [], defaultdict(list)
    shapes = defaultdict(list)
    n = 0
    for split in _accessible(ds, opt):
        for sid in ds.ids(split, opt.max_samples):
            s = ds.get(sid)
            if s is None:
                bad_load.append(f'{sid} ({ds.errors.get(sid, "?")})')
                continue
            n += 1
            shapes[s.shape].append(sid)
            if s.label.ndim != 2:
                res.add('error', f'label of {sid} is {s.label.ndim}D {s.label.shape}, expected 2D', [sid])
            if not np.isfinite(s.label).all():
                nonfinite.append((sid, 'label'))
            for nm, a in s.inputs.items():
                if not np.isfinite(a).all():
                    nonfinite.append((sid, nm))
                if tuple(a.shape[-2:]) != s.shape:
                    mismatch[nm].append((sid, tuple(a.shape[-2:]), s.shape))
    res.metrics['n_checked'] = n
    if bad_load:
        res.add('error', f'{len(bad_load)} sample(s) failed to load', bad_load, n_failed=len(bad_load))
    if nonfinite:
        chans = sorted({c for _, c in nonfinite})
        res.add('error', f'non-finite values in {len({s for s, _ in nonfinite})} sample(s), channels {chans}',
                [s for s, _ in nonfinite], channels=chans)
    for nm, lst in mismatch.items():
        res.add('warning', f"input '{nm}' is on a different grid from the label in {len(lst)} sample(s) (e.g. {lst[0][1]} vs {lst[0][2]}); "
                f'Sample.stack() area-resizes it', [x[0] for x in lst], channel=nm)
    if len(shapes) > 1:
        res.add('info', f'{len(shapes)} distinct grid shapes across {n} samples (min {min(shapes)}, max {max(shapes)})',
                n_shapes=len(shapes))
    unusable = [x for x in ds.skipped if str(x.get('reason', '')).startswith('tarball unusable')]
    if unusable:
        res.add('info', f'{len(unusable)} design tarball(s) skipped as incomplete / unverified: '
                + ', '.join(f"{x['design']} ({x['n_samples']} samples)" for x in unusable), n_designs=len(unusable))
    res.summary = f'{n} samples checked, {len(bad_load)} failed to load, {len({s for s, _ in nonfinite})} with non-finite values'
    return res.finalise()


# -- E002 ------------------------------------------------------------------------------------------------------------------------
def e002_degenerate(ds: MapDataset, opt: Options) -> CheckResult:
    """E002 degenerate samples: constant / all-zero maps, all-zero bounding boxes, negative or out-of-range labels, count mismatches.

    Why: an all-zero label or an all-zero input trains and scores nothing but still counts in the average; a sample whose
    bounding-box-derived inputs (RUDY*, cell_density) are all zero describes an empty layout (GIF, arXiv 2604.09999, reports this
    for CircuitNet-N14 as an aside); a negative IR drop is unphysical for a supply net (the ICCAD'23 contest removed such cases in v3);
    a label with no matching feature (or the reverse) means the CSV and the archives disagree (CircuitNet GitHub issues 31, 36, 42).
    Does NOT apply: eff_dist and pdn_density may legitimately contain zeros; a zero map is only counted as degenerate when ALL
    its pixels are equal. Bounding-box channels are the names in datasets.circuitnet.BBOX_CHANNELS (absent on ICCAD).
    """
    res = CheckResult('E002', 'degenerate')
    from .datasets.circuitnet import BBOX_CHANNELS
    const_label, zero_label, neg, oor, all_in_zero, bbox_zero = [], [], [], [], [], []
    chan_zero = defaultdict(int)
    chan_const_nonzero = defaultdict(list)
    n = 0
    lo, hi = ds.label_range if ds.label_range else (None, None)
    for split in _accessible(ds, opt):
        for sid in ds.ids(split, opt.max_samples):
            s = ds.get(sid)
            if s is None:
                continue
            n += 1
            L = s.label
            if np.all(L == L.flat[0]):
                (zero_label if abs(L.flat[0]) < 1e-6 * (hi if hi else 1.0) else const_label).append(sid)
            tol = max(opt.neg_tol, 1e-12, 1e-6 * (hi if hi else float(np.nanmax(np.abs(L)))))        # rounding noise of log/resize is not a negative label
            if np.nanmin(L) < -tol:
                neg.append(sid)
            if (hi is not None and np.nanmax(L) > hi + tol) or (lo is not None and lo > 0 and np.nanmin(L) < lo - tol):
                oor.append(sid)
            groups = defaultdict(list)                                           # channel group -> arrays (power_t has 20 slices)
            for nm, a in s.channels():
                groups[nm.split('[')[0]].append(a)
            zero_ch = []
            for base, arrs in groups.items():
                if all(np.all(a == 0) for a in arrs):
                    chan_zero[base] += 1
                    zero_ch.append(base)
                elif all(np.all(a == a.flat[0]) for a in arrs):
                    chan_const_nonzero[base].append(sid)
            if len(zero_ch) == len(groups):
                all_in_zero.append(sid)
            bb = [nm for nm in s.inputs if nm in BBOX_CHANNELS]
            if bb and all(np.all(s.inputs[nm] == 0) for nm in bb):
                bbox_zero.append(sid)
    res.metrics.update({'n_checked': n, 'n_zero_label': len(zero_label), 'n_const_label': len(const_label),
                        'n_negative_label': len(neg), 'n_out_of_range_label': len(oor), 'n_all_zero_inputs': len(all_in_zero),
                        'n_all_zero_bbox': len(bbox_zero), 'channel_all_zero_counts': dict(chan_zero)})

    def sev(k):
        return 'error' if n and k / n >= opt.all_zero_error_rate else 'warning'
    if zero_label:
        res.add(sev(len(zero_label)), f'{len(zero_label)}/{n} samples have an all-zero label', zero_label)
    if const_label:
        res.add(sev(len(const_label)), f'{len(const_label)}/{n} samples have a constant non-zero label', const_label)
    if neg:
        res.add(sev(len(neg)), f'{len(neg)}/{n} samples have negative label pixels (< -{opt.neg_tol})', neg)
    if oor:
        res.add(sev(len(oor)), f'{len(oor)}/{n} samples have label values outside the plausible range {ds.label_range}', oor)
    if all_in_zero:
        res.add(sev(len(all_in_zero)), f'{len(all_in_zero)}/{n} samples have every input channel all-zero', all_in_zero)
    if bbox_zero:
        res.add(sev(len(bbox_zero)), f'{len(bbox_zero)}/{n} samples have all bounding-box channels ({", ".join(BBOX_CHANNELS)}) all-zero',
                bbox_zero)
    for nm, k in sorted(chan_zero.items()):
        if k:
            res.add('info', f"input channel '{nm}' is all-zero in {k}/{n} samples", channel=nm, count=k)
    for nm, ids in chan_const_nonzero.items():
        res.add('warning', f"input channel '{nm}' is a constant non-zero map in {len(ids)} sample(s)", ids, channel=nm)
    mism = [x for x in ds.skipped if str(x.get('reason', '')).startswith('missing from tarball')]
    if mism:
        by = defaultdict(int)
        for x in mism:
            by[x['design']] += 1
        res.add('warning', f'{len(mism)} CSV-listed sample(s) lack their label or a feature map in the archive and were skipped '
                f'(per design: {dict(by)})', [x['id'] for x in mism], by_design=dict(by))
    unl = getattr(ds, 'unlisted', {})
    if any(unl.values()):
        res.add('info', f'archive members not listed in the split CSVs (per design): {dict((k, v) for k, v in unl.items() if v)}')
    res.summary = (f'{n} samples: {len(zero_label)} zero label, {len(neg)} negative, {len(oor)} out of range, '
                   f'{len(bbox_zero)} all-zero bbox, {len(mism)} count mismatches')
    return res.finalise()


# -- E003 ------------------------------------------------------------------------------------------------------------------------
def _hash(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes() + str(a.shape).encode()).hexdigest()


def _zdesc(a: np.ndarray, g: int) -> np.ndarray:
    d = area_resize(a, (g, g)).ravel()
    sd = d.std()
    return (d - d.mean()) / sd if sd > 0 else np.zeros_like(d)


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
            idesc = np.concatenate([_zdesc(a if a.shape == s.shape else area_resize(a, s.shape), opt.dup_grid) for _, a in s.channels()])
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
        X = np.stack([i[4] for i in ok]) / np.sqrt(len(ok[0][4]))
        Y = np.stack([i[5] for i in ok]) / np.sqrt(len(ok[0][5]))
        Ci, Cl = X @ X.T, Y @ Y.T
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


# -- E004 ------------------------------------------------------------------------------------------------------------------------
def _family(design: str) -> str:
    d = design.lower()
    for suf in ('-fpu', '-small', '-large', '-1', '-a', '-b'):
        d = d.replace(suf, '')
    return d


def e004_leakage(ds: MapDataset, opt: Options) -> CheckResult:
    """E004 split leakage: design overlap across splits, twin pairs across splits, related designs, and the random-vs-design split gap.

    Why: a model that has seen a design (or the other PDN of the same design) in training is not tested on an unseen design. ICCAD'23
    real cases come in twin pairs (same grid, same current-source count, different PDN); CircuitNet's official split is by design
    but puts small/large variants of one RTL (Vortex, nvdla) and RISCY / RISCY-FPU on different sides.
    Uses METADATA only (design ids, twin ids, names): safe to run on all splits before the freeze.
    The random-versus-design gap (how much easier a random split is than a design split) is computed ONLY if opt.run_gap is set
    by the protocol; it reads labels of the non-test splits.
    Does NOT apply: datasets without design metadata (a single split) are skipped.
    """
    res = CheckResult('E004', 'leakage')
    names = [s for s in ds.splits if s not in ds.aux_splits and ds.splits[s]]
    if len(names) < 2:
        return skipped('E004', 'leakage', f'needs at least two splits, have {names}')
    des = {s: {ds.meta(i).get('design', '') for i in ds.splits[s]} for s in names}
    ids = {s: set(ds.splits[s]) for s in names}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            sa, sb = names[a], names[b]
            if ids[sa] & ids[sb]:
                res.add('error', f'{len(ids[sa] & ids[sb])} sample id(s) occur in both {sa} and {sb}', _short(sorted(ids[sa] & ids[sb])))
            ov = des[sa] & des[sb]
            if ov:
                res.add('error', f'design(s) {sorted(ov)[:6]} occur in both {sa} and {sb} ({len(ov)} designs)', sorted(ov), splits=[sa, sb])
            fa = defaultdict(set)
            for d in des[sa]:
                fa[_family(d)].add(d)
            fb = defaultdict(set)
            for d in des[sb]:
                fb[_family(d)].add(d)
            rel = {f: (sorted(fa[f]), sorted(fb[f])) for f in fa if f in fb and not (fa[f] & fb[f])}
            if rel and ds.name.startswith('circuitnet'):
                res.add('info', f'related designs (same RTL family) on both sides of {sa}/{sb}: {rel}', splits=[sa, sb], families=rel)
    # twins
    twin_cross = []
    for s in names:
        for i in ds.splits[s]:
            t = ds.meta(i).get('twin_of')
            if t:
                for j in (t if isinstance(t, list) else [t]):
                    for s2 in names:
                        if s2 != s and j in ids[s2]:
                            twin_cross.append((i, j, s, s2))
    if twin_cross:
        res.add('error', f'{len(twin_cross)} twin pair link(s) cross splits (same grid and source count, different PDN)',
                [f'{a}<->{b} ({sa}/{sb})' for a, b, sa, sb in twin_cross], n=len(twin_cross))
    n_twin = sum(1 for s in names for i in ds.splits[s] if ds.meta(i).get('twin_of'))
    res.metrics.update({'designs_per_split': {s: len(v) for s, v in des.items()}, 'n_samples_with_twin': n_twin,
                        'twin_links_across_splits': len(twin_cross)})
    for s in names:
        grp = {ds.meta(i).get('design') for i in ds.splits[s] if ds.meta(i).get('twin_of')}
        if grp:
            res.metrics.setdefault('independent_designs_with_twins', {})[s] = len(grp)
    if opt.run_gap:
        res.metrics['random_vs_design_gap'] = _split_gap(ds, opt, res)
    else:
        res.metrics['random_vs_design_gap'] = 'not computed (run_gap is off: the gap is computed only when the protocol runs it)'
    res.summary = (f'splits {names}; designs per split {res.metrics["designs_per_split"]}; {len(twin_cross)} cross-split twin links')
    return res.finalise()


def _split_gap(ds: MapDataset, opt: Options, res: CheckResult) -> dict:
    from .baselines.zoo import PixelRidge, ScalarMean
    from .baselines.base import SampleStream
    pool = [s for s in ds.splits if s not in ds.test_splits and s not in ds.aux_splits]
    ids = [i for s in pool for i in ds.ids(s)]
    if len(ids) > opt.gap_max_samples:
        ids = [ids[j] for j in np.unique(np.linspace(0, len(ids) - 1, opt.gap_max_samples).round().astype(int))]
    designs = np.array([ds.meta(i).get('design', '') for i in ids])
    if len(set(designs)) < 2:
        return {'skipped': 'fewer than 2 designs in the pooled non-test splits'}
    rng = np.random.default_rng(0)
    k = min(opt.gap_folds, len(set(designs)), len(ids))
    rand_fold = rng.permutation(len(ids)) % k
    ud = sorted(set(designs))
    dmap = {d: j % k for j, d in enumerate(rng.permutation(ud))}
    des_fold = np.array([dmap[d] for d in designs])

    def cv(folds):
        errs, base = [], []
        for f in range(k):
            tr = [ids[j] for j in np.flatnonzero(folds != f)]
            te = [ids[j] for j in np.flatnonzero(folds == f)]
            if not tr or not te:
                continue
            r = PixelRidge().fit(SampleStream(ds, tr))
            m = ScalarMean().fit(SampleStream(ds, tr))
            for sid in te:
                s = ds.get(sid)
                errs.append(float(np.mean(np.abs(r.predict(s) - s.label))))
                base.append(float(np.mean(np.abs(m.predict(s) - s.label))))
        return float(np.mean(errs)), float(np.mean(base))
    (er, br), (ed, bd) = cv(rand_fold), cv(des_fold)
    out = {'folds': k, 'ridge_mae_random': er, 'ridge_mae_design': ed, 'scalar_mae_random': br, 'scalar_mae_design': bd,
           'skill_random': 1 - er / br if br else float('nan'), 'skill_design': 1 - ed / bd if bd else float('nan')}
    out['skill_gap'] = out['skill_random'] - out['skill_design']
    res.add('info', f"random-vs-design gap (ridge MAE skill over the scalar baseline): random {out['skill_random']:.3f}, "
            f"design {out['skill_design']:.3f}, gap {out['skill_gap']:.3f}")
    return out


# -- E005 ------------------------------------------------------------------------------------------------------------------------
def _dihedral(a: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    sq = a.shape[-1] == a.shape[-2]
    t = np.swapaxes(a, -1, -2)
    return {'stored': a, 'flip_ud': a[::-1, :], 'flip_lr': a[:, ::-1], 'rot180': a[::-1, ::-1],
            'transpose': t if sq else None, 'rot90': t[::-1, :] if sq else None, 'rot270': t[:, ::-1] if sq else None,
            'antitranspose': t[::-1, ::-1] if sq else None}


def e005_orientation(ds: MapDataset, opt: Options) -> CheckResult:
    """E005 axis / orientation: correlation of the label with each input channel as stored, transposed and flipped.

    Why: IR drop and congestion are driven by the local current / routing demand, so the label should correlate best with the input in
    the STORED orientation. If a transposed or flipped input correlates clearly better, the label and the inputs are stored in
    different axis conventions (fieldlint found such a fault in a public thermal benchmark). Maps are area-resized to a common grid and
    Gaussian-smoothed first; the comparison uses |corr| so the sign of the physics does not matter.
    Does NOT apply: non-square maps are tested only for the four flips; a channel that is unrelated to the label (median |corr| < 0.1
    in every orientation) cannot show a fault; a symmetric layout would also make orientations indistinguishable.
    """
    res = CheckResult('E005', 'orientation')
    from .core import box_mean
    from scipy.ndimage import gaussian_filter
    stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    n = 0
    g = opt.orient_grid
    for s in _iter(ds, opt, opt.orient_samples):
        if np.all(s.label == s.label.flat[0]):
            continue
        n += 1
        lab = gaussian_filter(area_resize(s.label, (g, g)), opt.orient_sigma)
        square = abs(s.shape[0] - s.shape[1]) <= 0.05 * max(s.shape)
        by = defaultdict(list)
        for nm, a in s.channels():
            by[nm.split('[')[0]].append(a)
        for nm, arrs in by.items():
            vals: Dict[str, List[float]] = defaultdict(list)
            for a in arrs[:4]:
                if np.all(a == a.flat[0]):
                    continue
                aa = gaussian_filter(area_resize(a if a.shape == s.shape else area_resize(a, s.shape), (g, g)), opt.orient_sigma)
                for vn, v in _dihedral(aa).items():
                    if v is None or (vn in ('transpose', 'rot90', 'rot270', 'antitranspose') and not square):
                        continue
                    c = pearson(lab, v)
                    if np.isfinite(c):
                        vals[vn].append(abs(c))
            for vn, cs in vals.items():
                stats[nm][vn].append(float(np.mean(cs)))
    if n == 0:
        return skipped('E005', 'orientation', 'no usable samples')
    table = {nm: {vn: float(np.median(v)) for vn, v in d.items()} for nm, d in stats.items()}
    res.metrics['n_checked'] = n
    res.metrics['median_abs_corr'] = table
    prim = ds.primary_input
    for nm, tb in table.items():
        st = tb.get('stored', float('nan'))
        alts = {k: v for k, v in tb.items() if k != 'stored'}
        if not alts or not np.isfinite(st):
            continue
        best = max(alts, key=alts.get)
        if alts[best] - st >= opt.orient_margin and alts[best] >= 0.1:
            k = min(len(stats[nm][best]), len(stats[nm]['stored']))
            frac = float(np.mean([stats[nm][best][j] > stats[nm]['stored'][j] for j in range(k)]))
            msg = (f"channel '{nm}': |corr| with the label is {st:.2f} as stored but {alts[best]:.2f} when '{best}' "
                   f'(better in {frac:.0%} of samples)')
            if nm == prim and frac >= opt.orient_consistency:
                res.add('error', msg + ' [primary channel]', channel=nm, best=best, stored=st, alt=alts[best])
            else:
                res.add('info', msg + ('' if nm == prim else ' [non-primary channel: informational]'), channel=nm, best=best,
                        stored=st, alt=alts[best])
    if prim in table:
        res.metrics['primary'] = {'channel': prim, 'stored': table[prim].get('stored'),
                                  'best_alternative': max((v for k, v in table[prim].items() if k != 'stored'), default=float('nan'))}
    res.summary = f'{n} samples; ' + ('; '.join(f"{prim}: stored {table[prim].get('stored', float('nan')):.2f}" for _ in [0] if prim in table) or 'primary channel absent')
    return res.finalise()


# -- E006 ------------------------------------------------------------------------------------------------------------------------
def _hot(L: np.ndarray, rule: dict) -> np.ndarray:
    k, v = rule['kind'], rule['value']
    if k == 'frac_of_max':
        return L > v * L.max()
    if k == 'abs':
        return L >= v
    return L >= np.quantile(L, 1 - v)


def e006_metric_sanity(ds: MapDataset, opt: Options) -> CheckResult:
    """E006 metric sanity: hotspot base rate, F1 at threshold perturbations, zero-hotspot predictions, how the official F1 treats them.

    Why: with the ICCAD rule (hotspot = pixel > 0.9 x max of the golden map) only a handful of pixels per case are positive (published
    base rates 0.01-0.5 %), so F1 is dominated by a few pixels, jumps with the exact threshold and is exactly 0 for a prediction
    that never exceeds it (the contest records F1 = 0 then). Three diagnostics on LABELS only (no model): (1) base rate per case;
    (2) F1 of two fixed pseudo-predictions - the label scaled by 0.95 / 1.05 and the label Gaussian-smoothed (sigma 1.5 px) - at
    thresholds 0.85 / 0.9 / 0.95 x max: a spread shows how fragile the score is to small amplitude or threshold changes; (3) the F1 of
    an all-zero prediction (0 whenever the golden map has a hotspot) and of a prediction equal to the golden map (1).
    Does NOT apply: the sensitivity table uses the frac_of_max rule (ICCAD); for CircuitNet only the base rate and the single-class
    (undefined ROC/PR AUC) counts are computed. Official-rule details: edalint.metrics module docstring.
    """
    res = CheckResult('E006', 'metric-sanity')
    from scipy.ndimage import gaussian_filter
    rule = ds.hotspot
    rates, counts, single_class, no_hot = [], [], [], []
    f1s = defaultdict(list)
    n = 0
    for s in _iter(ds, opt, opt.max_samples):
        L = s.label
        if not np.isfinite(L).all():
            continue
        n += 1
        h = _hot(L, rule)
        rates.append(float(h.mean()))
        counts.append(int(h.sum()))
        if h.sum() == 0:
            no_hot.append(s.id)
        if h.sum() == 0 or h.all():
            single_class.append(s.id)
        if rule['kind'] == 'frac_of_max' and L.max() > 0:
            smooth = gaussian_filter(L, 1.5)
            preds = {f'scale{sc}': L * sc for sc in opt.f1_scales}
            preds['smooth1.5'] = smooth
            for pn, p in preds.items():
                for fr in opt.f1_fracs:
                    f1s[(pn, fr)].append(M.f1_golden_max(p, L, frac=fr))
            f1s[('zero', 0.9)].append(M.f1_golden_max(np.zeros_like(L), L, frac=0.9))
            f1s[('golden', 0.9)].append(M.f1_golden_max(L, L, frac=0.9))
    if n == 0:
        return skipped('E006', 'metric-sanity', 'no usable samples')
    rates, counts = np.array(rates), np.array(counts)
    res.metrics.update({'n_checked': n, 'rule': rule, 'hotspot_base_rate': {'min': float(rates.min()), 'median': float(np.median(rates)),
                        'max': float(rates.max())}, 'hotspot_pixels': {'min': int(counts.min()), 'median': float(np.median(counts)),
                        'max': int(counts.max())}})
    few = [i for i, c in zip(range(len(counts)), counts) if 0 < c < opt.min_hotspot_px]
    if no_hot:
        res.add('error', f'{len(no_hot)}/{n} cases have NO hotspot under the rule {rule}: F1 is undefined (0/0) for them', no_hot)
    if (counts < opt.min_hotspot_px).sum() - len(no_hot) > 0:
        res.add('warning', f'{len(few)}/{n} cases have fewer than {opt.min_hotspot_px} hotspot pixels: their F1 is decided by a few pixels',
                n_cases=len(few))
    res.add('info', f'hotspot base rate (rule {rule}): min {rates.min():.2e}, median {np.median(rates):.2e}, max {rates.max():.2e}; '
            f'hotspot pixels per case: median {np.median(counts):.0f}')
    if single_class and rule['kind'] == 'abs':
        res.add('warning', f'{len(single_class)}/{n} cases have a single class at threshold {rule["value"]}: ROC/PR AUC undefined for them',
                single_class)
    if f1s:
        table = {f'{k[0]}@{k[1]}': float(np.nanmean(v)) for k, v in f1s.items()}
        res.metrics['mean_f1_pseudo_predictions'] = table
        for pn in {k[0] for k in f1s if k[0] not in ('zero', 'golden')}:
            vals = [table[f'{pn}@{fr}'] for fr in opt.f1_fracs]
            spread = max(vals) - min(vals)
            res.metrics.setdefault('f1_spread_over_threshold', {})[pn] = spread
            if spread > 0.15:
                res.add('warning', f"F1 of the pseudo-prediction '{pn}' moves by {spread:.2f} when the threshold changes among "
                        f'{list(opt.f1_fracs)} x max (values {["%.2f" % v for v in vals]})', spread=spread)
        z = table.get('zero@0.9')
        res.add('info', f'F1 of an all-zero prediction is {z:.2f} (the contest scores a prediction with no hotspot pixel as F1 = 0); '
                f'F1 of the label itself is {table.get("golden@0.9", float("nan")):.2f}')
    res.summary = (f'{n} cases; hotspot base rate median {np.median(rates):.2e}; {len(no_hot)} without hotspot')
    return res.finalise()


# -- E007 ------------------------------------------------------------------------------------------------------------------------
def e007_baseline_ladder(ds: MapDataset, opt: Options) -> CheckResult:
    """E007 (informational) baseline ladder: the zoo fitted on opt.fit_split and scored on opt.eval_split.

    Why: reports how much of the achievable score trivial, linear and physics baselines reach (the audit question). Never raises a
    severity above 'info'.
    Does NOT apply: it needs opt.eval_split (no default: the protocol chooses); evaluating a test split needs protocol/FROZEN.json.
    """
    if not opt.eval_split:
        return skipped('E007', 'baseline-ladder', 'no eval split given (informational check; pass eval_split, e.g. visible)')
    from .baselines import run_zoo
    res = CheckResult('E007', 'baseline-ladder')
    try:
        out = run_zoo(ds, opt.fit_split, opt.eval_split, names=opt.baselines, log=lambda *_: None)
    except ProtocolError as exc:
        return skipped('E007', 'baseline-ladder', str(exc))
    res.metrics['fit_split'], res.metrics['eval_split'] = opt.fit_split, opt.eval_split
    res.metrics['ladder'] = {k: (v['summary'] if 'summary' in v else {'skipped': v['skipped']}) for k, v in out.items()}
    for k, v in out.items():
        if 'summary' in v:
            sm = v['summary']
            keys = [x for x in ('mean_mae_mV', 'mean_f1_golden_max', 'mean_nrms', 'mean_ssim') if x in sm]
            res.add('info', f'{k}: ' + ', '.join(f'{x[5:]}={sm[x]:.4g}' for x in keys) + f' (n={sm["n"]}, {sm["mean_seconds"]:.2g} s/case)')
        else:
            res.add('info', f'{k}: skipped ({v["skipped"]})')
    res.summary = f'{sum("summary" in v for v in out.values())} baselines scored on {opt.eval_split}'
    return res.finalise()


CHECKS: Dict[str, Callable[[MapDataset, Options], CheckResult]] = {
    'E001': e001_integrity, 'E002': e002_degenerate, 'E003': e003_duplicates, 'E004': e004_leakage,
    'E005': e005_orientation, 'E006': e006_metric_sanity, 'E007': e007_baseline_ladder,
}
NAMES = {'E001': 'integrity', 'E002': 'degenerate', 'E003': 'duplicates', 'E004': 'leakage', 'E005': 'orientation',
         'E006': 'metric-sanity', 'E007': 'baseline-ladder'}
