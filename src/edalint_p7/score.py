"""P7 scorer: the N14 `_score_rows` row format and metrics for a model's predictions, with the id-set check of protocol_p7.md section 4.

Metrics are imported from edalint (not copied): `edalint.metrics` and `edalint.p4_n14.RocAcc / LABEL_THRESHOLD / PRIMARY`. Predictions are passed
through `np.nan_to_num` exactly as for the baselines. The scored id set must equal the registered `eval_ids` (results/n14/models_<task>.json);
any difference aborts. Nothing here opens a split: rows arrive as arguments.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from edalint import metrics as M
from edalint.core import ProtocolError
from edalint.datasets.circuitnet import design_of
from edalint.p4_n14 import LABEL_THRESHOLD, PRIMARY, RocAcc
from edalint.protocol import REPO_DIR

from .data import stack_inputs


def load_eval_ids(task: str, n14_dir=None) -> Dict[str, List[str]]:
    """`eval_ids` (design -> ids) of the registered N14 baseline run: ids only, no map or label value."""
    p = Path(n14_dir or REPO_DIR / 'results' / 'n14') / f'models_{task}.json'
    return {d: list(v) for d, v in json.loads(p.read_text(encoding='utf-8'))['eval_ids'].items()}


def check_scored_ids(scored: Sequence[str], expected_by_design: Dict[str, Sequence[str]]):
    exp = {i for v in expected_by_design.values() for i in v}
    got = list(scored)
    if len(set(got)) != len(got):
        raise ProtocolError(f'scorer aborted: {len(got) - len(set(got))} duplicate scored ids')
    miss, extra = exp - set(got), set(got) - exp
    if miss or extra:
        raise ProtocolError(f'scorer aborted: scored ids differ from eval_ids ({len(miss)} missing, e.g. {sorted(miss)[:3]}; '
                            f'{len(extra)} unexpected, e.g. {sorted(extra)[:3]})')


def score_predictions(task: str, items: Iterable[Tuple[str, str, np.ndarray, np.ndarray, float]],
                      expected_ids: Optional[Dict[str, Sequence[str]]]) -> Dict[str, Any]:
    """items: (id, design, prediction, label, seconds). Returns {'rows': [...], 'official_roc': {design: RocAcc dict}} in the format of
    the N14 models_<task>.json entries. `expected_ids` None is allowed only for development runs (train-only rows)."""
    thr = LABEL_THRESHOLD.get(task)
    rows: List[dict] = []
    acc: Dict[str, RocAcc] = {}
    for sid, des, raw, gold, dt in items:
        pred = np.nan_to_num(raw)
        row = {'id': sid, 'design': des, 'seconds': dt, 'nrms': M.nrms_official(pred, gold), 'ssim': M.ssim_official(pred, gold),
               'nrms_float': M.nrms(pred, gold), 'ssim_float': M.ssim(pred, gold), 'nmae': M.nmae(pred, gold), 'pearson': M.pearson_r(pred, gold)}
        if thr is not None:
            row['roc_auc_sample'] = M.roc_auc(pred, gold, thr)
            acc.setdefault(des, RocAcc()).add(pred, gold, thr)
        rows.append(row)
    if expected_ids is not None:
        check_scored_ids([r['id'] for r in rows], expected_ids)
    return {'primary': PRIMARY[task], 'rows': rows, 'official_roc': {d: a.to_dict() for d, a in acc.items()}}


def iter_dataset_rows(task: str, ds, ids_by_design: Dict[str, Sequence[str]]):
    """(id, design, x (C,H,W) float32, label (H,W)) from a MapDataset, design by design. Rows that fail to load are skipped (and so
    trip the id-set check)."""
    for des, ids in ids_by_design.items():
        for sid in ids:
            s = ds.get(sid)
            if s is None:
                continue
            yield sid, des, stack_inputs(task, s.inputs), s.label


def score_model(task: str, model, rows: Iterable[Tuple[str, str, np.ndarray, np.ndarray]], expected_ids, device: str = 'cpu'):
    """Run `model` (eval mode, batch 1, fp32) over (id, design, x, label) rows and score them."""
    import time
    from .train import predict

    def gen():
        for sid, des, x, y in rows:
            t0 = time.perf_counter()
            p = predict(task, model, x, device)
            yield sid, des, p, y, time.perf_counter() - t0
    return score_predictions(task, gen(), expected_ids)
