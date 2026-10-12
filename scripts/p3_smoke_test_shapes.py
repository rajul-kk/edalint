"""Loader smoke test on TEST files: shapes and dtypes only (no value, statistic, metric or comparison is computed or printed).

  python scripts/p3_smoke_test_shapes.py results/dev/smoke_test_shapes_p3dev.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from edalint.datasets import ICCAD23Dataset, N14Dataset
from edalint.runner import write_once

res = {}
ds = ICCAD23Dataset()
res['iccad23_hidden'] = {}
for sid in ds.ids('test'):
    s = ds.load_unguarded(sid)                           # loading smoke test only
    res['iccad23_hidden'][sid] = {'shape': list(s.shape), 'inputs': {k: list(v.shape) for k, v in s.inputs.items()},
                                  'dtype': str(s.label.dtype), 'design_id': ds.meta(sid)['design_id'], 'twin_of': ds.meta(sid)['twin_of']}
    del s
print('iccad23 hidden: loaded', len(res['iccad23_hidden']), 'cases', flush=True)
for task in ('drc',):
    n14 = N14Dataset(task=task, stage=False)
    res[f'n14_{task}_test_designs'] = n14.smoke_test_shapes(per_design=1)
    print(task, [(r['design'], r['n_members']) for r in res[f'n14_{task}_test_designs']], flush=True)
    res[f'n14_{task}_skipped'] = n14.skipped
write_once(sys.argv[1], res)
