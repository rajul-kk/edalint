"""P4 development (TRAIN data only): baseline hyperparameters by leave-one-TWIN-PAIR-out CV on the 10 visible real cases.

Folds: each of the 5 visible real designs (a twin pair) is held out; the fit set is 100 fake + the other 4 visible designs (8 real).
Selection criterion (set before the run): lowest mean MAE (mV) over the 10 held-out cases (pooled over folds); ties go to the first
listed option. F1 is reported but not used (it is ~0 for every non-solver baseline on the visible cases).
Grids: ridge alpha {1e-6,1e-4,1e-2,1}; green edge sets {coarse,medium,fine}; knn k {1,3,5}; gbm {(100 iter, depth 3), (300 iter, depth 6)}.

  python scripts/p4_dev_hparams.py results/dev/iccad23_hparam_cv_p4dev.json
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import numpy as np

from edalint.baselines import evaluate, stream
from edalint.baselines.zoo import GREEN_EDGES, GreenSuperposition, KNNMap, PixelGBM, PixelRidge
from edalint.core import ListDataset
from edalint.datasets import ICCAD23Dataset
from edalint.runner import write_once

out = Path(sys.argv[1])
ds = ICCAD23Dataset()
fake = ds.ids('fake')
vis = ds.ids('visible')
assert len(fake) == 100 and len(vis) == 10
mem = ListDataset([ds.get(i) for i in fake + vis], name='iccad23', test_splits=(), hotspot=ds.hotspot, label_range=ds.label_range)
grid = {
    'ridge': [(f'alpha={a:g}', (lambda a=a: PixelRidge(alpha=a))) for a in (1e-6, 1e-4, 1e-2, 1.0)],
    'green': [(f'edges={k}', (lambda k=k: GreenSuperposition(edges=GREEN_EDGES[k]))) for k in ('coarse', 'medium', 'fine')],
    'knn': [(f'k={k}', (lambda k=k: KNNMap(k=k))) for k in (1, 3, 5)],
    'gbm': [('iter100_depth3', lambda: PixelGBM(max_iter=100, max_leaf_nodes=None, max_depth=3)),
            ('iter300_depth6', lambda: PixelGBM(max_iter=300, max_leaf_nodes=None, max_depth=6))],
}
designs = sorted({ds.meta(i)['design_id'] for i in vis})
rows = {}
for d in designs:
    held = [i for i in vis if ds.meta(i)['design_id'] == d]
    fit_ids = fake + [i for i in vis if i not in held]
    mem.splits['fit'], mem.splits['held'] = fit_ids, held
    for fam, opts in grid.items():
        for label, mk in opts:
            t0 = time.time()
            b = mk().fit(stream(mem, fit_ids))
            r = evaluate(b, mem, 'held')
            for x in r:
                x['fold'] = d
            rows.setdefault((fam, label), []).extend(r)
            print(d, fam, label, [round(x['mae_mV'], 4) for x in r], round(time.time() - t0, 1), 's', flush=True)
summary, choice = {}, {}
for fam, opts in grid.items():
    summary[fam] = {}
    for label, _ in opts:
        r = rows[(fam, label)]
        summary[fam][label] = {'mean_mae_mV': float(np.mean([x['mae_mV'] for x in r])), 'mean_f1_golden_max': float(np.nanmean([x['f1_golden_max'] for x in r])),
                               'mean_f1_top10pct': float(np.nanmean([x['f1_top10pct'] for x in r])), 'n': len(r)}
    choice[fam] = min(summary[fam], key=lambda k: summary[fam][k]['mean_mae_mV'])
print(json_ := {f: (choice[f], summary[f]) for f in grid})
write_once(out, {'label': 'development (train only), P4', 'folds': designs, 'criterion': 'lowest mean MAE (mV) over the 10 held-out visible cases',
                 'summary': summary, 'choice': choice, 'rows': {f'{k[0]}|{k[1]}': v for k, v in rows.items()}})
