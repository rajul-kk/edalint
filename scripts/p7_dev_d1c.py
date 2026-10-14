"""P7 D1 follow-up (c) (DEVELOPMENT, TRAIN-ONLY, no training): the D1 training loss (100 x MSE) against reference predictors on the same 800 fit rows."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from edalint.baselines import stream                      # noqa: E402
from edalint.baselines.zoo import MeanMap, ScalarMean       # noqa: E402
from edalint.p4_n14 import fit_selector                    # noqa: E402
from edalint_p7.data import open_train_rows                # noqa: E402

CACHE = 'E:/edalint_data/n14_registered_cache/congestion'
HELD, FIT = ('Vortex-small', 'nvdla-small'), ('RISCY', 'RISCY-FPU')
ds = open_train_rows('congestion', designs=FIT + HELD, cache_dir=CACHE, select=fit_selector(400), stage=True)
fit_ids = [i for d in FIT for i in ds.ids(d)]
mm, sc = MeanMap().fit(stream(ds, fit_ids)), ScalarMean().fit(stream(ds, fit_ids))
out = {'label': 'DEVELOPMENT, train-only; not a result', 'unit': '100 x MSE (the training loss)', 'fit': {}, 'heldout': {}}
for grp, designs in (('fit', FIT), ('heldout', HELD)):
    for d in designs:
        e = {'mean_map': [], 'scalar': []}
        for i in ds.ids(d):
            s = ds.get(i)
            e['mean_map'].append(float(np.mean((np.nan_to_num(mm.predict(s)) - s.label) ** 2)))
            e['scalar'].append(float(np.mean((np.nan_to_num(sc.predict(s)) - s.label) ** 2)))
        out[grp][d] = {k: 100 * float(np.mean(v)) for k, v in e.items()}
Path(REPO / 'results' / 'dev' / 'p7_d1c_train_loss_refs_p7dev.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
