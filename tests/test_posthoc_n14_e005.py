"""scripts/posthoc_n14_e005.py: a transposed channel is flagged SUSPECT, a correctly oriented one is OK (synthetic maps)."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from edalint.checks import Options
from edalint.core import Sample

spec = importlib.util.spec_from_file_location('posthoc_n14_e005', Path(__file__).resolve().parent.parent / 'scripts' / 'posthoc_n14_e005.py')
PH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PH)


def samples(transpose, n=30, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x = gaussian_filter(rng.random((64, 64)), 3) + 0.02 * np.arange(64)[:, None]      # asymmetric smooth map
        lab = x + 0.01 * rng.random((64, 64))
        out.append(Sample(f's{i}', 'train', 'd', {'ch': x.T.copy() if transpose else x}, lab))
    return out


def test_transposed_flagged():
    opt = Options(orient_samples=None)
    s = PH.summarise(PH.per_sample_corr(samples(True), ['ch'], opt)['ch'])
    assert s['best_orientation'] == 'transpose' and s['suspect']


def test_stored_ok():
    opt = Options(orient_samples=None)
    s = PH.summarise(PH.per_sample_corr(samples(False), ['ch'], opt)['ch'])
    assert s['best_orientation'] == 'stored' and not s['suspect']
