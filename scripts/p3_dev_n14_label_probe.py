"""Development probe (TRAIN designs only): raw CircuitNet-N14 IR_drop label statistics, to size the authors' clip at 50.

Streams the IR_drop_features tarball of each design given on the command line and records, per sample, the raw label maximum and the
fraction of pixels that are zero / above the authors' clip (50). Writes a run-once JSON.

  python scripts/p3_dev_n14_label_probe.py results/dev/n14_ir_label_clip_probe_p3dev.json Vortex-small nvdla-small
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import numpy as np

from edalint.datasets.circuitnet import DEFAULT_ROOT, FEATURE_DIRS, stream_tarball
from edalint.runner import write_once

out, designs = Path(sys.argv[1]), sys.argv[2:]
res = {}
for d in designs:
    tar = Path(DEFAULT_ROOT) / 'CircuitNet-N14' / 'IR_drop_features' / f'{d}.tar.gz'
    rows = []

    def h(f, n, a, rows=rows):
        rows.append({'name': n, 'shape': list(a.shape), 'max': float(a.max()), 'frac_gt50': float((a > 50).mean()),
                     'frac_zero': float((a == 0).mean()), 'mean': float(a.mean())})
    stream_tarball(tar, d, {FEATURE_DIRS['IR_drop']: 'IR_drop'}, None, h)
    m = np.array([r['max'] for r in rows])
    g = np.array([r['frac_gt50'] for r in rows])
    z = np.array([r['frac_zero'] for r in rows])
    print(d, 'n', len(rows), 'label max min/median/max', m.min(), np.median(m), m.max(), '| mean fraction of pixels > 50:', g.mean(),
          '| samples with any pixel > 50:', int((g > 0).sum()), '| all-zero labels:', int((z == 1).sum()), '| median zero fraction', np.median(z))
    res[d] = rows
write_once(out, {'label': 'development (train only)', 'probe': 'raw IR_drop label statistics vs the authors clip at 50', 'designs': res})
