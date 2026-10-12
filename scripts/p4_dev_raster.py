"""P4 development (TRAIN data only): choose the rasterisation of the exact solver.

Compares cubic / linear / nearest interpolation from m1 node voltages to the pixel grid on 10 visible real + 5 fake train cases
(fake = 5 evenly spaced ids of the fake split). For each case and method: MAE against the stored label (mV) and the end-to-end time
(parse + solve + rasterise). Solve-only time (the sparse factorisation + back-substitution) is recorded separately.
Rule (set by the lead before the run): freeze the fastest method whose MAE is <= 0.01 mV on EVERY one of these 15 train cases.

  python scripts/p4_dev_raster.py results/dev/iccad23_spice_raster_p4dev.json
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import numpy as np

from edalint.datasets import ICCAD23Dataset
from edalint.runner import write_once
from edalint.spice import exact_ir_drop

out = Path(sys.argv[1])
ds = ICCAD23Dataset()
ids = ds.ids('fake', 5) + ds.ids('visible')
assert not any(ds.is_test_split(ds.meta(i)['split']) for i in ids)
rows = []
for sid in ids:
    s = ds.get(sid)
    for method in ('cubic', 'linear', 'nearest'):
        t0 = time.perf_counter()
        m, info = exact_ir_drop(s.meta['netlist'], s.shape, method='direct', interp=method)
        dt = time.perf_counter() - t0
        e = np.abs(m - s.label) * 1e3
        rows.append({'id': sid, 'shape': list(s.shape), 'interp': method, 'mae_mV': float(e.mean()), 'max_abs_mV': float(e.max()),
                     'seconds_end_to_end': dt, 'seconds_parse': info['seconds_parse'], 'seconds_solve': info['seconds_solve'],
                     'seconds_assemble_and_prepare': info['seconds_assemble'] + info['seconds_prepare'],
                     'seconds_rasterize': info['seconds_rasterize']})
        print(sid, s.shape, method, round(rows[-1]['mae_mV'], 5), round(dt, 2), round(info['seconds_rasterize'], 2), flush=True)
summ = {}
for method in ('cubic', 'linear', 'nearest'):
    r = [x for x in rows if x['interp'] == method]
    summ[method] = {'mean_mae_mV': float(np.mean([x['mae_mV'] for x in r])), 'max_mae_mV': float(np.max([x['mae_mV'] for x in r])),
                    'n_cases_mae_gt_0.01': int(sum(x['mae_mV'] > 0.01 for x in r)),
                    'median_end_to_end_s': float(np.median([x['seconds_end_to_end'] for x in r])),
                    'median_rasterize_s': float(np.median([x['seconds_rasterize'] for x in r])),
                    'median_solve_only_s': float(np.median([x['seconds_solve'] for x in r]))}
ok = [m for m in summ if summ[m]['max_mae_mV'] <= 0.01]
choice = min(ok, key=lambda m: summ[m]['median_end_to_end_s']) if ok else None
print(summ, 'CHOICE', choice)
write_once(out, {'label': 'development (train only), P4', 'ids': ids, 'rule': 'fastest method with MAE <= 0.01 mV on every case',
                 'summary': summ, 'choice': choice, 'rows': rows})
