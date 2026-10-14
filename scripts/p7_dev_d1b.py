"""P7 D1 follow-up (b) (DEVELOPMENT, TRAIN-ONLY, no training): per-design reference NRMS on the D1 held-out rows, to read the D1 result.
Same rows as scripts/p7_dev_d1.py (fit RISCY + RISCY-FPU, held out Vortex-small + nvdla-small, registered fit-sample cache); no test row opened.
Reports per held-out design: zero, scalar mean, mean_map fitted on the 800 fit rows, and an ORACLE mean_map fitted on the held-out design itself
(optimistic bound: how much a model that knew the design's average map could gain), plus label statistics. Not a result.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from edalint.baselines import stream                      # noqa: E402
from edalint.baselines.zoo import MeanMap, ScalarMean, Zero  # noqa: E402
from edalint.p4_n14 import PrimaryAcc, fit_selector        # noqa: E402
from edalint_p7.data import check_disjoint, open_train_rows  # noqa: E402
from edalint_p7.score import load_eval_ids                 # noqa: E402

CACHE = 'E:/edalint_data/n14_registered_cache/congestion'
HELD, FIT = ('Vortex-small', 'nvdla-small'), ('RISCY', 'RISCY-FPU')
task = 'congestion'
ds = open_train_rows(task, designs=FIT + HELD, cache_dir=CACHE, select=fit_selector(400), stage=True)
fit_ids = [i for d in FIT for i in ds.ids(d)]
check_disjoint(fit_ids + [i for d in HELD for i in ds.ids(d)], load_eval_ids(task))
models = {'zero': Zero().fit(stream(ds, fit_ids)), 'scalar': ScalarMean().fit(stream(ds, fit_ids)), 'mean_map': MeanMap().fit(stream(ds, fit_ids))}
out = {'label': 'DEVELOPMENT, train-only; not a result', 'n_fit': len(fit_ids), 'designs': {}}
for d in HELD:
    ids = ds.ids(d)
    accs = {k: PrimaryAcc(task) for k in list(models) + ['oracle_mean_map']}
    oracle = MeanMap().fit(stream(ds, ids))
    ys = []
    for i in ids:
        s = ds.get(i)
        ys.append(s.label)
        for k, m in models.items():
            accs[k].add(np.nan_to_num(m.predict(s)), s.label)
        accs['oracle_mean_map'].add(np.nan_to_num(oracle.predict(s)), s.label)
    y = np.stack(ys)
    out['designs'][d] = {'n': len(ids), 'nrms': {k: a.value() for k, a in accs.items()},
                         'label_mean': float(y.mean()), 'label_std_pixels': float(y.std()), 'label_frac_gt_0.1': float((y > 0.1).mean()),
                         'label_row_to_row_std_of_row_mean': float(y.reshape(len(ids), -1).mean(1).std())}
Path(REPO / 'results' / 'dev' / 'p7_d1b_heldout_refs_p7dev.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
