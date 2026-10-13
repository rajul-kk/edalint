"""TRAIN-only evidence for two metric findings of protocol_n14.md (the split CSV given to the dataset has an empty test file):

 1. E006 / ROC: the authors' ROC area (utils/metrics.py 175-221, 299-337) has no (0, 0) anchor and drops thresholds at which nothing is predicted positive,
    so a constant predictor scores 0, not 0.5. Shown for IR drop with zero / scalar / mean_map / constant 0.5 predictors, official vs (0, 0)-anchored area vs the
    mean per-sample rank AUC.
 2. The authors' SSIM and NRMS are computed on maps clamped to [0, 1] and quantised to uint8 (utils/metrics.py 35-55 `input_converter`, 339-368 `tensor2img`).
    Shown for DRC and congestion: official versus float SSIM / NRMS for zero / scalar / mean_map, the share of labels that quantise to all-zero, the share with a
    positive pixel at the DRC threshold 0.1.

Fit set: RISCY and RISCY-FPU (first 100 of the seeded order each); evaluated on the other two train designs (Vortex-small, nvdla-small).

  python -I scripts/p4_dev_n14_metric_checks.py results/dev/n14_metric_checks.json
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import numpy as np

from edalint import metrics as M
from edalint import p4_n14 as P
from edalint.baselines import stream
from edalint.baselines.zoo import MeanMap, ScalarMean, Zero
from edalint.datasets.circuitnet import N14Dataset, packaged_split_csvs
from edalint.runner import write_once

CACHE = 'E:/edalint_data/n14_cv_cache'


class Const:
    def __init__(self, v):
        self.v = v

    def fit(self, train):
        return self

    def predict(self, s):
        return np.full(s.shape, self.v)


def load(task):
    tmp = Path(tempfile.mkdtemp(prefix='n14mc_'))
    (tmp / 'test_empty.csv').write_text('')
    ds = N14Dataset(task=task, mode='authors', cache_dir=f'{CACHE}/{task}', split_csvs=(str(packaged_split_csvs()[0]), str(tmp / 'test_empty.csv')),
                    select=P.fit_selector(P.K_GAP, P.SEED), stage=True)
    assert not ds.ids('test')
    by = P._by_design(ds, ds.ids('train'))
    fit = [i for d in ('RISCY', 'RISCY-FPU') for i in by[d]]
    ev = [i for d in ('Vortex-small', 'nvdla-small') for i in by[d]]
    return ds, fit, ev


def main(out):
    res = {'label': 'development (train only)', 'fit_designs': ['RISCY', 'RISCY-FPU'], 'eval_designs': ['Vortex-small', 'nvdla-small']}
    # 1. ROC anchor
    ds, fit, ev = load('ir_drop')
    thr = P.LABEL_THRESHOLD['ir_drop']
    preds = {'zero': Zero(), 'scalar': ScalarMean().fit(stream(ds, fit)), 'mean_map': MeanMap().fit(stream(ds, fit)), 'constant_0.5': Const(0.5)}
    roc = {}
    for n, b in preds.items():
        acc, rank = P.RocAcc(), []
        for sid in ev:
            s = ds.get(sid)
            p = np.nan_to_num(b.predict(s))
            acc.add(p, s.label, thr)
            rank.append(M.roc_auc(p, s.label, thr))
        roc[n] = {'official_area': acc.area(), 'anchored_area': acc.area(anchored=True), 'mean_per_sample_rank_auc': float(np.nanmean(rank)),
                  'n_samples': acc.n_samples, 'n_skipped_no_positive': acc.n_no_positive}
        print('ir', n, roc[n], flush=True)
    res['ir_drop_roc'] = {'threshold': thr, 'baselines': roc}
    # 2. SSIM / NRMS definitions
    for task in ('drc', 'congestion'):
        ds, fit, ev = load(task)
        bl = {'zero': Zero(), 'scalar': ScalarMean().fit(stream(ds, fit)), 'mean_map': MeanMap().fit(stream(ds, fit))}
        rows = {}
        labels_q0 = labels_pos = 0
        for sid in ev:
            L = ds.get(sid).label
            labels_q0 += int(not M.to_u8(L).any())
            if task == 'drc':
                labels_pos += int((L >= P.LABEL_THRESHOLD['drc']).any())
        for n, b in bl.items():
            v = {k: [] for k in ('ssim_official', 'ssim_float', 'nrms_official', 'nrms_float')}
            for sid in ev:
                s = ds.get(sid)
                p = np.nan_to_num(b.predict(s))
                v['ssim_official'].append(M.ssim_official(p, s.label))
                v['ssim_float'].append(M.ssim(p, s.label))
                v['nrms_official'].append(M.nrms_official(p, s.label))
                v['nrms_float'].append(M.nrms(p, s.label))
            rows[n] = {k: {'mean_finite': float(np.mean([x for x in a if np.isfinite(x)])) if any(np.isfinite(a)) else None,
                           'n_not_finite': int(sum(not np.isfinite(x) for x in a)), 'n_exactly_1': int(sum(x == 1.0 for x in a))} for k, a in v.items()}
            print(task, n, {k: x['mean_finite'] for k, x in rows[n].items()}, flush=True)
        res[task] = {'n_eval': len(ev), 'labels_quantising_to_all_zero': labels_q0,
                     'labels_with_pixel_at_threshold_0.1': labels_pos if task == 'drc' else None, 'baselines': rows}
    write_once(out, res)


if __name__ == '__main__':
    main(sys.argv[1])
