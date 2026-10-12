"""Development run (TRAIN data only): leave-one-visible-design-out for the ICCAD'23 baselines.

For each of the 5 visible real designs (a twin pair), fit every learned baseline on `n_fake` fake cases + the other 4 visible designs
(8 real cases) and score on the held-out pair. Only train data is read; the hidden cases are never touched. Reported as
"development (train only)" and used to size the P4 protocol (does adding real cases to the fit change the ladder?).

  python scripts/p3_dev_iccad_loo.py results/dev/iccad23_loo_visible_p3dev.json [n_fake]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import numpy as np

from edalint.baselines import BASELINES, evaluate, summarise
from edalint.core import ListDataset
from edalint.datasets import ICCAD23Dataset
from edalint.runner import write_once

out = Path(sys.argv[1])
n_fake = int(sys.argv[2]) if len(sys.argv) > 2 else 60
ds = ICCAD23Dataset()
fake_ids = ds.ids('fake', n_fake)
vis = ds.ids('visible')
mem = ListDataset([ds.get(i) for i in fake_ids + vis], name='iccad23', test_splits=(), hotspot=ds.hotspot, label_range=ds.label_range)
designs = sorted({ds.meta(i)['design_id'] for i in vis})
names = ['mean_map', 'scalar', 'ridge', 'gbm', 'knn', 'effdist', 'green']
res = {}
for d in designs:
    held = [i for i in vis if ds.meta(i)['design_id'] == d]
    fit_ids = fake_ids + [i for i in vis if i not in held]
    mem.splits['fit'] = fit_ids
    mem.splits['held'] = held
    for n in names:
        b = BASELINES[n]()
        b.fit(__import__('edalint.baselines', fromlist=['stream']).stream(mem, fit_ids))
        rows = evaluate(b, mem, 'held')
        res.setdefault(n, []).extend(rows)
        print(d, n, [round(r['mae_mV'], 3) for r in rows], [round(r['f1_golden_max'], 3) for r in rows], flush=True)
summary = {n: summarise(r) for n, r in res.items()}
write_once(out, {'label': 'development (train only)', 'protocol': f'leave-one-visible-design-out; fit = {n_fake} fake + 8 other visible real cases',
                 'fake_ids': fake_ids, 'summary': summary, 'rows': res})
for n, s in summary.items():
    print(n, {k: round(v, 4) for k, v in s.items() if k in ('mean_mae_mV', 'mean_f1_golden_max', 'mean_f1_top10pct')})
