"""P9 leakage primitives (protocol_p9.md section 4; design D1-D5).

Descriptor = the E003 descriptor: every input channel (incl. each power_t slice) area-resized to g x g and z-scored; a constant channel
is a zero vector. Similarity of two samples = sum over channels of the per-channel Pearson correlation, divided by the number of channels
that vary in at least one of the two (E003 after amendment A3/A7). Labels: Pearson of the g x g label descriptors.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from edalint.core import Sample, area_resize

DUP_CORR = 0.999                     # primary threshold (= edalint.checks.Options.dup_corr)
SENS_CORR = (0.99, 0.95)             # sensitivity thresholds (reported, not decisive)
CONFIRM_MAXABS = 0.01                # identity confirmation at full resolution (authors space, [0, 1] maps)
PROX_QUANTILE = 0.25                 # L002: S_same must reach this quantile of the within-design reference
PROX_MARGIN = 0.02                   # L002: and exceed S_other by this much


def zdesc(a: np.ndarray, g: int = 32) -> np.ndarray:
    d = area_resize(np.asarray(a, dtype=np.float64), (g, g)).ravel()
    sd = d.std()
    return (d - d.mean()) / sd if sd > 0 else np.zeros_like(d)


def describe(s: Sample, g: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """(input descriptor (C, g*g) float32, label descriptor (g*g,) float32), unit-norm per channel (so dot products are Pearson)."""
    chans = []
    for _, a in s.channels():
        a = a if a.shape == s.shape else area_resize(a, s.shape)
        chans.append(zdesc(a, g))
    X = np.stack(chans) / np.sqrt(g * g)
    y = zdesc(s.label, g) / np.sqrt(g * g)
    return X.astype(np.float32), y.astype(np.float32)


def similarity(Q: np.ndarray, R: np.ndarray, block: int = 256) -> np.ndarray:
    """Input similarity matrix (nq, nr) between descriptor stacks Q (nq, C, G) and R (nr, C, G): mean Pearson over the channels that vary
    in at least one of the pair. Blocked over queries to bound memory."""
    vq, vr = (Q != 0).any(-1), (R != 0).any(-1)                                           # (nq, C), (nr, C)
    out = np.empty((len(Q), len(R)), dtype=np.float32)
    for a in range(0, len(Q), block):
        q = Q[a:a + block]
        num = np.einsum('icg,jcg->ij', q, R, optimize=True)
        nv = (vq[a:a + block, None, :] | vr[None, :, :]).sum(-1)
        out[a:a + block] = num / np.maximum(nv, 1)
    return out


def label_similarity(qy: np.ndarray, ry: np.ndarray) -> np.ndarray:
    return qy @ ry.T


def nearest(S: np.ndarray, exclude_self: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Index and value of the row-wise maximum of S; with exclude_self (S square, same set), the diagonal is ignored."""
    S = S.copy()
    if exclude_self:
        np.fill_diagonal(S, -np.inf)
    j = S.argmax(1)
    return j, S[np.arange(len(S)), j]


def nn_records(q_ids: Sequence[str], r_ids: Sequence[str], SI: np.ndarray, SL: np.ndarray, exclude_self: bool = False) -> List[dict]:
    """Per query: its input-NN in the reference set, the input similarity and the label similarity of that pair, and the best label
    similarity among reference rows whose input similarity is >= DUP_CORR (None if none)."""
    j, v = nearest(SI, exclude_self)
    recs = []
    for k, qid in enumerate(q_ids):
        row = {'id': qid, 'nn': r_ids[j[k]], 'input_sim': float(v[k]), 'label_sim': float(SL[k, j[k]])}
        hit = SI[k] >= DUP_CORR
        if exclude_self:
            hit[k] = False
        row['n_input_ge_primary'] = int(hit.sum())
        both = hit & (SL[k] >= DUP_CORR)
        row['n_pair_ge_primary'] = int(both.sum())
        row['pairs_ge_primary'] = [r_ids[x] for x in np.flatnonzero(both)[:20]]
        recs.append(row)
    return recs


def threshold_counts(recs: Sequence[dict], SI_max_label: Optional[Sequence[float]] = None) -> Dict[str, dict]:
    """Counts of queries whose NN pair is a near duplicate (input AND label >= t), or input-only, for t in primary + sensitivity."""
    out = {}
    for t in (DUP_CORR,) + SENS_CORR:
        both = sum(1 for r in recs if r['input_sim'] >= t and r['label_sim'] >= t)
        inp = sum(1 for r in recs if r['input_sim'] >= t)
        out[f'{t:g}'] = {'n_queries': len(recs), 'nn_pair_ge': both, 'nn_input_ge': inp, 'nn_input_only': inp - both,
                         'frac_pair': both / len(recs) if recs else float('nan')}
    out['any_pair_ge_primary'] = sum(1 for r in recs if r['n_pair_ge_primary'] > 0)
    return out


def confirm_pair(a: Sample, b: Sample, tol: float = CONFIRM_MAXABS) -> dict:
    """Identity confirmation of a flagged pair at full resolution (D3): every input channel and the label differ by <= tol (max abs).
    Different grids (e.g. ICCAD'23 native sizes) cannot be confirmed: reported as not confirmed with the reason."""
    if a.shape != b.shape:
        return {'confirmed': False, 'reason': f'different grids {a.shape} vs {b.shape}'}
    ca, cb = dict(a.channels()), dict(b.channels())
    if set(ca) != set(cb):
        return {'confirmed': False, 'reason': 'different channel sets'}
    worst = {nm: float(np.max(np.abs(ca[nm].astype(np.float64) - cb[nm].astype(np.float64)))) for nm in ca}
    lab = float(np.max(np.abs(a.label.astype(np.float64) - b.label.astype(np.float64))))
    mx = max(max(worst.values()), lab)
    return {'confirmed': bool(mx <= tol), 'max_abs_input': max(worst.values()), 'max_abs_label': lab,
            'worst_channel': max(worst, key=worst.get)}


# -- design families ----------------------------------------------------------------------------------------------------------------
def family(design: str) -> str:
    """edalint.checks._family, then the pre-declared N14 family table again (so N28 names such as 'zero-riscy-a' map to 'riscy')."""
    from edalint.checks import N14_FAMILIES, _family
    f = _family(design)
    return N14_FAMILIES.get(f, f)


def design_proximity(test_rows: Dict[str, np.ndarray], train_designs: Sequence[str], test_designs: Sequence[str],
                     within: Dict[str, np.ndarray], train_design_of_col: Sequence[str]) -> Dict[str, dict]:
    """L002 per test design. test_rows[d] = (n_d, n_train) input-similarity matrix of design d's rows against all train rows (columns
    labelled by train_design_of_col); within[d'] = within-design NN similarities of train design d' (leave-one-out).
    S_same / S_other = median over d's rows of the NN similarity restricted to same-family / other-family train columns."""
    ref = np.concatenate([v for v in within.values() if len(v)]) if within else np.array([])
    q = float(np.quantile(ref, PROX_QUANTILE)) if len(ref) else float('nan')
    cols = np.array([family(x) for x in train_design_of_col])
    out = {}
    for d in test_designs:
        S = test_rows[d]
        f = family(d)
        same, other = cols == f, cols != f
        s_same = float(np.median(S[:, same].max(1))) if same.any() else None
        s_other = float(np.median(S[:, other].max(1))) if other.any() else None
        rec = {'family': f, 'n_rows': int(S.shape[0]), 'same_family_train_designs': sorted({x for x, c in zip(train_design_of_col, cols) if c == f}),
               'S_same': s_same, 'S_other': s_other, 'reference_q25_within_design': q,
               'reference_median_within_design': float(np.median(ref)) if len(ref) else float('nan')}
        if s_same is None:
            rec['material'] = False
            rec['reason'] = 'no same-family train design'
        else:
            rec['material'] = bool(s_same >= q and s_same - (s_other if s_other is not None else -np.inf) >= PROX_MARGIN)
            rec['reason'] = f'S_same {s_same:.4f} vs q25 {q:.4f}; S_same - S_other {s_same - (s_other or 0):.4f} vs {PROX_MARGIN}'
        out[d] = rec
    return out


# -- published split files (L003) -----------------------------------------------------------------------------------------------
_N14_RE = re.compile(r'^(?P<design>.+?)_freq_')


def design_from_row(row: str) -> str:
    """Design name of a split-CSV row (`feature/<name>.npy,label/<name>.npy` or an absolute path). N14 names: `<design>_freq_...`;
    N28 names: `<index>-<design>-<variant>-...` (design = RISCY, RISCY-FPU, zero-riscy; variant a/b kept, e.g. 'RISCY-a')."""
    first = row.strip().split(',')[0]
    name = first.replace('\\', '/').rsplit('/', 1)[-1]
    name = name[:-4] if name.endswith('.npy') else name
    m = _N14_RE.match(name)
    if m:
        return m.group('design')
    parts = name.split('-')
    if parts and parts[0].isdigit():
        # <index>-<design...>-<variant>-<macros>-c..: design tokens run until the single-letter variant token
        toks = parts[1:]
        for k, t in enumerate(toks):
            if len(t) == 1 and t.isalpha():
                return '-'.join(toks[:k + 1])
        return '-'.join(toks[:2])
    return name


def split_file_overlap(splits: Dict[str, Iterable[str]]) -> dict:
    """Design-name overlap between every pair of splits given as {split: rows}. Exact design overlap = confirmed (identity on names)."""
    des = {s: defaultdict(int) for s in splits}
    for s, rows in splits.items():
        for r in rows:
            if r.strip():
                des[s][design_from_row(r)] += 1
    names = list(splits)
    pairs = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            sa, sb = names[a], names[b]
            ov = sorted(set(des[sa]) & set(des[sb]))
            fam = sorted({family(x) for x in des[sa]} & {family(x) for x in des[sb]})
            pairs.append({'splits': [sa, sb], 'design_overlap': ov, 'rows_in_overlap': {s: sum(des[s][d] for d in ov) for s in (sa, sb)},
                          'family_overlap': fam})
    return {'designs': {s: dict(sorted(v.items())) for s, v in des.items()}, 'pairs': pairs}
