"""P4 development for CircuitNet-N14 (TRAIN data only): baseline hyperparameters by leave-one-design-out CV on the 4 train designs.

The split CSV given to the dataset has an EMPTY test file, so no test row can be listed, staged or read.
Folds: each train design (RISCY, RISCY-FPU, Vortex-small, nvdla-small) is held out in turn; the fit set is the other three designs. Every
design contributes at most K=100 rows (the first 100 of the seeded order of edalint.p4_n14, a prefix of the registered K=400 fit sample).
Criterion (set before the run, protocol_n14.md section 5.2): best design-macro mean over the 4 held-out designs of the task's PRIMARY metric
(congestion NRMS lower, DRC SSIM higher, IR drop official ROC area higher); ties go to the first listed option.
Grids (reduced, fixed): ridge alpha {1e-4, 1e-2, 1}; kNN k {1, 3, 5}; GBM {100 iter depth 3; 300 iter depth 6}; Green edges {coarse, medium} (IR drop only).

  python -I scripts/p4_dev_n14_hparams.py --tasks ir_drop --out <scratch>/cv_ir.json
  python -I scripts/p4_dev_n14_hparams.py --merge a.json b.json --out results/dev/n14_hparam_cv.json

Round 2 (lead decisions of 2026-10-13, after the first round): the primary NRMS / SSIM are now the authors' (clamp to [0, 1], uint8 quantisation), and the ridge
grid is extended upward to {10, 100, 1000}. Congestion and DRC are re-run in full (new metric definitions); IR drop (ROC metric unchanged) runs only the new ridge
penalties and is merged with round 1 (`--ir-ridge-only`, `--prev`):
  python -I scripts/p4_dev_n14_hparams.py --tasks congestion,drc --ridge-ext --out <scratch>/cv2_cd.json
  python -I scripts/p4_dev_n14_hparams.py --tasks ir_drop --ir-ridge-only --out <scratch>/cv2_ir.json
  python -I scripts/p4_dev_n14_hparams.py --merge cv2_ir.json cv2_cd.json --prev results/dev/n14_hparam_cv.json --out results/dev/n14_hparam_cv_round2.json
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import numpy as np

from edalint import p4_n14 as P
from edalint.baselines import stream
from edalint.baselines.zoo import GREEN_EDGES, GreenSuperposition, KNNMap, PixelGBM, PixelRidge
from edalint.datasets.circuitnet import N14Dataset, packaged_split_csvs
from edalint.runner import write_once

K_CV = 100
CACHE = 'E:/edalint_data/n14_cv_cache'


def grid(task, ridge_ext=False, ir_ridge_only=False):
    alphas = (1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0) if ridge_ext else (1e-4, 1e-2, 1.0)
    if ir_ridge_only:
        return {'ridge': [(f'alpha={a:g}', (lambda a=a: PixelRidge(alpha=a))) for a in (10.0, 100.0, 1000.0)]}
    g = {'ridge': [(f'alpha={a:g}', (lambda a=a: PixelRidge(alpha=a))) for a in alphas],
         'knn': [(f'k={k}', (lambda k=k: KNNMap(k=k))) for k in (1, 3, 5)],
         'gbm': [('iter100_depth3', lambda: PixelGBM(max_iter=100, max_leaf_nodes=None, max_depth=3)),
                 ('iter300_depth6', lambda: PixelGBM(max_iter=300, max_leaf_nodes=None, max_depth=6))]}
    if task == 'ir_drop':
        g['green'] = [(f'edges={k}', (lambda k=k: GreenSuperposition(edges=GREEN_EDGES[k]))) for k in ('coarse', 'medium')]
    return g


def run_task(task, ridge_ext=False, ir_ridge_only=False):
    tmp = Path(tempfile.mkdtemp(prefix='n14cv_'))
    (tmp / 'test_empty.csv').write_text('')
    ds = N14Dataset(task=task, mode='authors', cache_dir=f'{CACHE}/{task}', split_csvs=(str(packaged_split_csvs()[0]), str(tmp / 'test_empty.csv')),
                    select=P.fit_selector(K_CV, P.SEED), stage=True)
    assert not ds.ids('test'), 'the test split must be empty in development'
    by_design = P._by_design(ds, ds.ids('train'))
    designs = sorted(by_design)
    print(task, {d: len(v) for d, v in by_design.items()}, 'skipped', len(ds.skipped), flush=True)
    rows, secs = {}, {}
    for held in designs:
        fit_ids = [i for d in designs if d != held for i in by_design[d]]
        for fam, opts in grid(task, ridge_ext, ir_ridge_only).items():
            for label, mk in opts:
                t0 = time.time()
                b = mk().fit(stream(ds, fit_ids))
                acc = P.PrimaryAcc(task)
                for sid in by_design[held]:
                    s = ds.get(sid)
                    acc.add(np.nan_to_num(b.predict(s)), s.label)
                rows.setdefault((fam, label), {})[held] = acc.value()
                secs[(fam, label, held)] = time.time() - t0
                print(task, held, fam, label, round(rows[(fam, label)][held], 4), round(secs[(fam, label, held)], 1), 's', flush=True)
    higher = P.PRIMARY[task][1]
    summary, choice = {}, {}
    for fam, opts in grid(task, ridge_ext, ir_ridge_only).items():
        summary[fam] = {label: {'design_macro': P.macro(rows[(fam, label)]), 'per_design': rows[(fam, label)]} for label, _ in opts}
        pick = None
        for label, _ in opts:                                                    # first listed wins ties
            v = summary[fam][label]['design_macro']
            if np.isfinite(v) and (pick is None or (v > summary[fam][pick]['design_macro'] if higher else v < summary[fam][pick]['design_macro'])):
                pick = label
        choice[fam] = pick
    return {'task': task, 'metric': P.PRIMARY[task][0], 'higher_is_better': higher, 'folds': designs, 'k_per_design': K_CV,
            'n_per_design': {d: len(v) for d, v in by_design.items()}, 'summary': summary, 'choice': choice,
            'fit_seconds_total': float(sum(secs.values())), 'partial': bool(ir_ridge_only), 'ridge_ext': bool(ridge_ext or ir_ridge_only)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default='')
    ap.add_argument('--merge', nargs='*')
    ap.add_argument('--prev')
    ap.add_argument('--ridge-ext', action='store_true')
    ap.add_argument('--ir-ridge-only', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.merge:
        parts = [json.loads(Path(p).read_text()) for p in a.merge]
        res = {}
        for p in parts:
            res.update(p['tasks'])
        if a.prev:                                              # partial tasks (IR-drop ridge extension) are merged into round 1
            prev = json.loads(Path(a.prev).read_text())['tasks']
            for t, v in list(res.items()):
                if v.get('partial'):
                    higher = v['higher_is_better']
                    summ = {f: dict(o) for f, o in prev[t]['summary'].items()}
                    summ['ridge'].update(v['summary']['ridge'])
                    choice = {}
                    for fam, opts in summ.items():
                        pick = None
                        for label in opts:                       # insertion order: round-1 options first, so ties keep the first listed
                            x = opts[label]['design_macro']
                            if np.isfinite(x) and (pick is None or (x > opts[pick]['design_macro'] if higher else x < opts[pick]['design_macro'])):
                                pick = label
                        choice[fam] = pick
                    res[t] = {**prev[t], 'summary': summ, 'choice': choice, 'partial': False, 'ridge_ext': True,
                              'note': 'ridge options 10, 100, 1000 from round 2; every other option from round 1 (same metric, same folds)'}
        write_once(a.out, {'label': 'development (train only), P4 N14', 'criterion': __doc__.split('Criterion')[1].split('Grids')[0].strip(), 'tasks': res})
    else:
        res = {t: run_task(t, a.ridge_ext, a.ir_ridge_only) for t in a.tasks.split(',')}
        write_once(a.out, {'label': 'development (train only), P4 N14 (part)', 'tasks': res})
