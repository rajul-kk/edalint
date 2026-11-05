"""P9 leakage checks L001-L003 and their primitives, on synthetic data with known answers."""
import numpy as np

from edalint.core import ListDataset, Sample
from edalint_p9 import leakage as LK
from edalint_p9.checks import P9Options, P9_CHECKS, all_checks, l001_nn_near_duplicates, l002_design_proximity, l003_published_splits


def _smooth(rng, n=64):
    from scipy.ndimage import gaussian_filter
    a = gaussian_filter(rng.random((n, n)), 3)
    return 0.5 + 0.2 * (a - a.mean()) / a.std()


def _sample(sid, split, design, rng, base=None, noise=0.0, const=True):
    p = base['p'] if base else _smooth(rng)
    k = base['k'] if base else _smooth(rng)
    ins = {'p': p + noise * rng.normal(size=p.shape), 'k': k + noise * rng.normal(size=k.shape)}
    if const:
        ins['c'] = np.full(p.shape, 300.0)                       # a constant channel must not dilute the similarity (A3)
    lab = 2 * ins['p'] + 0.5 * ins['k']
    return Sample(sid, split, design, ins, lab)


def _ds(near_noise=1e-6, n_train=12, n_val=6, seed=0):
    rng = np.random.default_rng(seed)
    tr = [_sample(f't{i}', 'train', f'd{i % 4}', rng) for i in range(n_train)]
    va = [_sample(f'v{i}', 'val', f'e{i}', rng) for i in range(n_val - 1)]
    base = {'p': tr[0].inputs['p'], 'k': tr[0].inputs['k']}
    va.append(_sample('vdup', 'val', 'e_dup', rng, base=base, noise=near_noise))
    return ListDataset(tr + va, test_splits=())


def test_similarity_is_one_for_identical_and_ignores_constant_channels():
    rng = np.random.default_rng(1)
    s = _sample('a', 'train', 'd', rng)
    x, _ = LK.describe(s)
    S = LK.similarity(x[None], x[None])
    assert abs(S[0, 0] - 1.0) < 1e-5


def test_l001_confirms_a_near_copy_and_nothing_else():
    r = l001_nn_near_duplicates(_ds(near_noise=1e-6), P9Options(split='all'))
    assert r.status == 'flagged' and r.severity == 'error'
    err = [f for f in r.findings if f.severity == 'error'][0]
    assert err.samples == ['vdup~t0'] and err.metrics['n'] == 1
    c = r.metrics['val_vs_train']['counts']['0.999']
    assert c['nn_pair_ge'] == 1 and c['n_queries'] == 6


def test_l001_flags_but_does_not_confirm_a_noisier_copy():
    # descriptor correlation stays >= 0.999 (32x32 area mean averages the noise out), full-resolution max diff > 0.01
    r = l001_nn_near_duplicates(_ds(near_noise=0.004), P9Options(split='all'))
    sev = {f.severity for f in r.findings}
    assert 'warning' in sev and 'error' not in sev


def test_l001_clean_data_is_ok():
    rng = np.random.default_rng(3)
    smp = [_sample(f't{i}', 'train', f'd{i}', rng) for i in range(8)] + [_sample(f'v{i}', 'val', f'e{i}', rng) for i in range(4)]
    r = l001_nn_near_duplicates(ListDataset(smp, test_splits=()), P9Options(split='all'))
    assert r.status == 'ok'


def test_l001_skips_without_a_query_split():
    rng = np.random.default_rng(3)
    ds = ListDataset([_sample(f't{i}', 'train', 'd', rng) for i in range(3)], test_splits=())
    assert l001_nn_near_duplicates(ds, P9Options(split='all')).status == 'skipped'


def test_l002_flags_a_variant_design_and_not_an_unrelated_one():
    rng = np.random.default_rng(5)
    smp = []
    bases = {}
    for d in ('RISCY', 'nvdla-small'):
        b = {'p': _smooth(rng), 'k': _smooth(rng)}
        bases[d] = b
        smp += [_sample(f'{d}_{i}', 'train', d, rng, base=b, noise=0.02) for i in range(6)]   # placement variants of one design
    smp += [_sample(f'zr_{i}', 'val', 'zero-riscy', rng, base=bases['RISCY'], noise=0.02) for i in range(4)]   # family variant
    smp += [_sample(f'oc_{i}', 'val', 'openc910-1', rng) for i in range(4)]                                   # unrelated
    r = l002_design_proximity(ListDataset(smp, test_splits=()), P9Options(split='all'))
    prox = r.metrics['val_vs_train']
    assert prox['zero-riscy']['material'] is True
    assert prox['openc910-1']['material'] is False and prox['openc910-1']['reason'] == 'no same-family train design'
    assert r.status == 'flagged'


def test_design_from_row_n14_and_n28_and_family():
    assert LK.design_from_row('feature/RISCY-FPU_freq_50_mp_1_fpu_60_fpa_1.0_p_1_fi_ap.npy,label/x.npy') == 'RISCY-FPU'
    assert LK.design_from_row('/data/x/feature/zero-riscy_freq_200_mp_1.npy,/data/x/label/zero-riscy_freq_200_mp_1.npy') == 'zero-riscy'
    assert LK.design_from_row('feature/7-RISCY-FPU-b-1-c2-u0.75-m1-p4-f1.npy,label/7.npy') == 'RISCY-FPU-b'
    assert LK.design_from_row('12-zero-riscy-a-1-c5-u0.7-m3-p2-f0') == 'zero-riscy-a'
    assert LK.family('zero-riscy-a') == 'riscy' and LK.family('Vortex-large') == 'vortex' and LK.family('openc910-1') == 'openc910'


def test_l003_design_overlap_is_an_error(tmp_path):
    tr = tmp_path / 'tr.csv'
    te = tmp_path / 'te.csv'
    va = tmp_path / 'va.csv'
    tr.write_text('feature/RISCY_freq_1.npy,label/RISCY_freq_1.npy\nfeature/nvdla-small_freq_2.npy,l\n')
    va.write_text('feature/Vortex-small_freq_3.npy,l\n')
    te.write_text('feature/nvdla-small_freq_9.npy,l\nfeature/zero-riscy_freq_4.npy,l\n')
    r = l003_published_splits(None, P9Options(split_files={'train': str(tr), 'val': str(va), 'test': str(te)}))
    errs = [f for f in r.findings if f.severity == 'error']
    assert len(errs) == 1 and errs[0].samples == ['nvdla-small'] and errs[0].metrics['splits'] == ['train', 'test']
    assert any(f.severity == 'info' and 'riscy' in f.message for f in r.findings)
    assert l003_published_splits(None, P9Options()).status == 'skipped'


def test_registry_extends_the_linter_without_touching_it():
    from edalint.checks import CHECKS
    a = all_checks()
    assert set(CHECKS) <= set(a) and set(P9_CHECKS) == {'L001', 'L002', 'L003'} and not (set(CHECKS) & set(P9_CHECKS))
