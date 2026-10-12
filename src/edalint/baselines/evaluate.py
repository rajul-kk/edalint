"""Evaluate baselines on a split. Refuses test splits unless the protocol is frozen (edalint.protocol.require_frozen)."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import metrics as M
from ..core import MapDataset
from ..protocol import require_frozen
from .base import BASELINES, Baseline, SampleStream, stream


def case_metrics(ds: MapDataset, pred: np.ndarray, gold: np.ndarray) -> Dict[str, float]:
    if ds.hotspot.get('kind') == 'frac_of_max':        # ICCAD-style metric family (MAE in mV, hotspot F1)
        return M.iccad_case_metrics(pred, gold)
    thr = ds.hotspot['value'] if ds.hotspot.get('kind') == 'abs' else None
    return M.circuitnet_case_metrics(pred, gold, ds.task, threshold=thr)


def guard_eval(ds: MapDataset, split: str):
    """Raise ProtocolError if evaluating on `split` would touch a test sample and no frozen protocol exists."""
    if ds.is_test_split(split):
        require_frozen(ds.protocol_dir, what=f'evaluating baselines on split {split!r}')


def evaluate(b: Baseline, ds: MapDataset, split: str, max_n: Optional[int] = None) -> List[dict]:
    """Per-case rows: id, design, group, shape, seconds, plus the metrics. `b` must already be fitted."""
    guard_eval(ds, split)
    rows = []
    for s in stream(ds, split, max_n):
        t0 = time.time()
        pred = b.predict(s)
        dt = time.time() - t0
        row = {'id': s.id, 'design': s.design, 'group': s.meta.get('group', ''), 'shape': list(s.shape), 'seconds': dt,
               'finite': bool(np.isfinite(pred).all())}
        row.update(case_metrics(ds, np.nan_to_num(pred), s.label))
        rows.append(row)
    return rows


def summarise(rows: Sequence[dict]) -> Dict[str, float]:
    keys = [k for k in rows[0] if isinstance(rows[0][k], float) and k != 'seconds'] if rows else []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out = {f'mean_{k}': float(np.nanmean([r[k] for r in rows])) for k in keys}
    out['mean_seconds'] = float(np.mean([r['seconds'] for r in rows])) if rows else float('nan')
    out['n'] = len(rows)
    return out


def run_zoo(ds: MapDataset, fit_split: str, eval_split: str, names: Optional[Sequence[str]] = None, max_fit: Optional[int] = None,
            max_eval: Optional[int] = None, log=print) -> Dict[str, dict]:
    """Fit every applicable baseline on fit_split and evaluate on eval_split. Returns {name: {rows, summary, fit_seconds}} or
    {name: {skipped: reason}}."""
    guard_eval(ds, eval_split)
    first = next(iter(stream(ds, eval_split, 1)), None)
    out: Dict[str, dict] = {}
    for name in (names or list(BASELINES)):
        if name not in BASELINES:
            out[name] = {'skipped': 'unknown baseline'}
            continue
        b = BASELINES[name]()
        ok, why = b.applicable(ds, first)
        if not ok:
            out[name] = {'skipped': why}
            log(f'  {name}: skipped ({why})')
            continue
        t0 = time.time()
        b.fit(stream(ds, fit_split, max_fit))
        fit_s = time.time() - t0
        rows = evaluate(b, ds, eval_split, max_eval)
        out[name] = {'rows': rows, 'summary': summarise(rows), 'fit_seconds': fit_s}
        log(f'  {name}: fit {fit_s:.1f}s, ' + ', '.join(f'{k}={v:.4g}' for k, v in out[name]['summary'].items() if k.startswith('mean_') and k != 'mean_seconds')[:200])
    return out
