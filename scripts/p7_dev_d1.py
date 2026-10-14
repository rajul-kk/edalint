"""P7 D1 sanity run (DEVELOPMENT, TRAIN-ONLY, CPU). protocol_p7.md section 11, D1.

GPDL (congestion), Vortex-small and nvdla-small HELD OUT of training (both are official TRAIN designs), 2,000 iterations (cosine compressed to 2,000),
fp32, one seed. Training rows: the cached registered fit sample of the remaining train designs (RISCY, RISCY-FPU: 400 each). Pass criteria:
  (1) loss finite and the mean loss of the last 100 iterations is at least 30 % below the first 100;
  (2) official NRMS on the held-out rows is below the train-only mean_map (fitted on the same training rows) on the same held-out rows.
No test row is opened: open_train_rows gives an empty second split. Output: results/dev/p7_d1_gpdl_heldout_p7dev.json (run-once).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from edalint.baselines import stream                      # noqa: E402
from edalint.baselines.zoo import MeanMap                  # noqa: E402
from edalint.p4_n14 import PrimaryAcc, fit_selector        # noqa: E402
from edalint.runner import write_once                      # noqa: E402
from edalint_p7.data import MemorySource, check_disjoint, open_train_rows   # noqa: E402
from edalint_p7.score import load_eval_ids, score_predictions    # noqa: E402
from edalint_p7.train import predict, train                # noqa: E402

CACHE = 'E:/edalint_data/n14_registered_cache/congestion'
HELD = ('Vortex-small', 'nvdla-small')
FIT = ('RISCY', 'RISCY-FPU')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261101)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--out', default=str(REPO / 'results' / 'dev' / 'p7_d1_gpdl_heldout_p7dev.json'))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.perf_counter()
    task = 'congestion'
    ds = open_train_rows(task, designs=FIT + HELD, cache_dir=CACHE, select=fit_selector(400), stage=True)
    fit_ids, held_ids = [i for d in FIT for i in ds.ids(d)], [i for d in HELD for i in ds.ids(d)]
    check_disjoint(fit_ids + held_ids, load_eval_ids(task))
    report = {'n_fit': len(fit_ids), 'n_heldout': len(held_ids), 'disjoint_from_eval_ids': True}
    src = MemorySource.from_dataset(task, ds, fit_ids)
    print('train rows', len(src), 'held-out rows', len(held_ids), flush=True)
    rec = train(task, src, a.iters, a.seed, 'fp32', 'cpu', 'block100')
    model = rec.pop('_model')
    # held-out scoring with the P7 scorer (dev: no expected id set) and the train-only mean_map on the same rows
    held = MemorySource.from_dataset(task, ds, held_ids)

    def items():
        for k, sid in enumerate(held.ids):
            x, y = held.get(k)
            yield sid, held.designs[k], predict(task, model, x), y, 0.0
    sc = score_predictions(task, items(), None)
    mm = MeanMap().fit(stream(ds, fit_ids))
    acc_m, acc_g = PrimaryAcc(task), PrimaryAcc(task)
    for k, sid in enumerate(held.ids):
        s = ds.get(sid)
        acc_m.add(np.nan_to_num(mm.predict(s)), s.label)
    for r in sc['rows']:
        acc_g.nrms.append(r['nrms'])
    nrms_gpdl, nrms_mm = acc_g.value(), acc_m.value()
    by = {}
    for r in sc['rows']:
        by.setdefault(r['design'], []).append(r['nrms'])
    curve = rec['loss_per_100_iters']
    drop = 1 - curve[-1] / curve[0] if curve and np.isfinite(curve).all() else float('nan')
    verdict = {'loss_finite': bool(np.isfinite(curve).all()), 'loss_first_block': curve[0], 'loss_last_block': curve[-1], 'loss_drop_fraction': drop,
               'loss_drop_ge_30pct': bool(drop >= 0.30), 'heldout_nrms_gpdl': nrms_gpdl, 'heldout_nrms_mean_map': nrms_mm,
               'gpdl_beats_mean_map': bool(nrms_gpdl < nrms_mm),
               'per_design_nrms_gpdl': {d: float(np.mean(v)) for d, v in by.items()}}
    verdict['pass'] = bool(verdict['loss_finite'] and verdict['loss_drop_ge_30pct'] and verdict['gpdl_beats_mean_map'])
    out = {'label': 'DEVELOPMENT, train-only (D1 sanity run, protocol_p7.md section 11); not a result', 'held_out_designs': list(HELD),
           'train_designs': list(FIT), 'train_id_report': report, 'n_heldout_rows': len(held.ids), 'train': rec, 'verdict': verdict,
           'seconds_total': time.perf_counter() - t0}
    write_once(a.out, out, dataset='p7')
    print(verdict)


if __name__ == '__main__':
    main()
