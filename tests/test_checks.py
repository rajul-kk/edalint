"""Every check: clean synthetic data passes, and an injected fault is detected."""
import numpy as np
import pytest

from edalint.checks import (CHECKS, Options, e001_integrity, e002_degenerate, e003_duplicates, e004_leakage, e005_orientation,
                            e006_metric_sanity, e007_baseline_ladder)
from edalint.core import ListDataset, Sample
from edalint.datasets.synthetic import make_synthetic
from edalint.runner import run_checks

OPT = Options(split='train')


def codes(res):
    return [(f.severity, f.message) for f in res.findings]


def has(res, severity, text):
    return any(f.severity == severity and text in f.message for f in res.findings)


@pytest.fixture(scope='module')
def clean():
    return make_synthetic()


def test_clean_data_has_no_errors(clean):
    rep = run_checks(clean, None, OPT)
    errs = [(r.code, f.message) for r in rep.results for f in r.findings if f.severity == 'error']
    assert errs == []
    assert [r.status for r in rep.results if r.code in ('E001', 'E002', 'E003', 'E004', 'E005')] == ['ok'] * 5


def test_result_structure(clean):
    r = e001_integrity(clean, OPT).to_dict()
    assert set(r) == {'code', 'name', 'status', 'severity', 'findings', 'summary', 'metrics', 'skip_reason'}
    assert all(CHECKS[c].__doc__ and 'Does NOT apply' in CHECKS[c].__doc__ for c in CHECKS)


# -- E001 ------------------------------------------------------------------------------------------------------------------------
def test_e001_nan_detected():
    r = e001_integrity(make_synthetic(n_nan=1), OPT)
    assert r.status == 'flagged' and r.severity == 'error' and has(r, 'error', 'non-finite')


def test_e001_load_failure_detected(clean):
    class Broken(ListDataset):
        def _load(self, sid):
            if sid == 'trdesign0_0':
                raise OSError('truncated file')
            return super()._load(sid)
    ds = Broken(list(clean._samples.values()), hotspot=clean.hotspot)
    r = e001_integrity(ds, OPT)
    assert has(r, 'error', 'failed to load') and 'truncated file' in r.findings[0].samples[0]


def test_e001_grid_mismatch_between_input_and_label():
    s = make_synthetic()
    smp = s._samples['trdesign0_0']
    smp.inputs['eff_dist'] = np.zeros((20, 20))
    r = e001_integrity(s, OPT)
    assert has(r, 'warning', "input 'eff_dist' is on a different grid")


# -- E002 ------------------------------------------------------------------------------------------------------------------------
def test_e002_zero_label_negative_and_range():
    r = e002_degenerate(make_synthetic(n_zero_labels=2, n_negative=1), OPT)
    assert has(r, 'error', 'all-zero label') and has(r, 'error', 'negative label')
    assert r.metrics['n_zero_label'] == 2 and r.metrics['n_negative_label'] == 1


def test_e002_out_of_range_label():
    s = make_synthetic()
    s._samples['trdesign0_0'].label = s._samples['trdesign0_0'].label * 1e4          # > VDD
    r = e002_degenerate(s, OPT)
    assert has(r, 'warning', 'outside the plausible range') or has(r, 'error', 'outside the plausible range')


def test_e002_all_zero_bounding_boxes_and_inputs():
    samples = []
    rng = np.random.default_rng(0)
    for i in range(6):
        z = np.zeros((16, 16))
        a = rng.random((16, 16))
        inputs = {'macro_region': a, 'RUDY': z.copy() if i < 2 else a, 'cell_density': z.copy() if i < 2 else a}
        samples.append(Sample(f's{i}', 'train', f'd{i}', inputs, rng.random((16, 16))))
    ds = ListDataset(samples, task='drc', hotspot={'kind': 'abs', 'value': 0.1}, label_range=(0, 1))
    r = e002_degenerate(ds, OPT)
    assert r.metrics['n_all_zero_bbox'] == 2 and has(r, 'error', 'bounding-box')
    z_all = ListDataset([Sample('x', 'train', 'd', {'a': np.zeros((8, 8))}, np.ones((8, 8)) * 0.5)], hotspot={'kind': 'abs', 'value': 0.1})
    assert has(e002_degenerate(z_all, OPT), 'error', 'every input channel all-zero')


def test_e002_constant_nonzero_channel_and_count_mismatch(clean):
    s = make_synthetic()
    s._samples['trdesign0_0'].inputs['pdn_density'] = np.full((40, 40), 2.0)
    r = e002_degenerate(s, OPT)
    assert has(r, 'warning', 'constant non-zero')
    s.skipped.append({'id': 'x', 'design': 'D', 'split': 'train', 'reason': 'missing from tarball: DRC_all'})
    r = e002_degenerate(s, OPT)
    assert has(r, 'warning', 'lack their label or a feature map')


# -- E003 ------------------------------------------------------------------------------------------------------------------------
def test_e003_exact_and_near_duplicates_within_split():
    r = e003_duplicates(make_synthetic(n_duplicates=2), OPT)
    assert has(r, 'warning', 'byte-identical inputs') and has(r, 'warning', 'byte-identical labels')
    assert r.metrics['n_near_dup_pairs'] >= 2


def test_e003_near_duplicate_with_noise():
    ds = make_synthetic()
    a = ds._samples['trdesign0_0']
    rng = np.random.default_rng(0)
    b = Sample('noisy', 'train', 'noisyd', {k: v * (1 + 1e-5 * rng.standard_normal(v.shape)) for k, v in a.inputs.items()},
               a.label * (1 + 1e-5 * rng.standard_normal(a.label.shape)))
    ds2 = ListDataset(list(ds._samples.values()) + [b], hotspot=ds.hotspot, label_range=ds.label_range)
    r = e003_duplicates(ds2, OPT)
    assert r.metrics['n_near_dup_pairs'] >= 1 and r.metrics['exact_input_groups'] == 0


def test_e003_cross_split_duplicate_is_an_error_only_when_both_splits_are_read():
    ds = make_synthetic()
    t = ds._samples['trdesign0_0']
    ds._samples['te_copy'] = Sample('te_copy', 'test', 'tedupe', {k: v.copy() for k, v in t.inputs.items()}, t.label.copy())
    ds.splits['test'].append('te_copy')
    ds._meta['te_copy'] = {'split': 'test', 'design': 'tedupe'}
    # test split excluded before the freeze: nothing to find, nothing read
    assert e003_duplicates(ds, Options(split='all')).status == 'ok'
    # pretend the freeze happened: no test splits are guarded
    ds.test_splits = ()
    r = e003_duplicates(ds, Options(split='all'))
    assert has(r, 'error', 'ACROSS splits')


def test_e003_clean_has_no_duplicates(clean):
    r = e003_duplicates(clean, OPT)
    assert r.status == 'ok' and r.metrics['n_near_dup_pairs'] == 0


# -- E004 ------------------------------------------------------------------------------------------------------------------------
def test_e004_design_overlap_detected():
    r = e004_leakage(make_synthetic(leak_design=True), Options(split='all'))
    assert has(r, 'error', 'occur in both train and test')


def test_e004_twin_pair_across_splits_detected():
    r = e004_leakage(make_synthetic(cross_twin=True), Options(split='all'))
    assert has(r, 'error', 'twin pair link')
    assert r.metrics['twin_links_across_splits'] == 2


def test_e004_clean_and_independent_design_counts(clean):
    r = e004_leakage(clean, Options(split='all'))
    assert r.status == 'ok'
    assert r.metrics['independent_designs_with_twins'] == {'train': 6, 'test': 2}
    assert 'not computed' in r.metrics['random_vs_design_gap']


def test_e004_gap_only_when_protocol_runs_it(clean):
    r = e004_leakage(clean, Options(split='all', run_gap=True, gap_folds=3, gap_max_samples=12))
    g = r.metrics['random_vs_design_gap']
    assert isinstance(g, dict) and 'skill_gap' in g and np.isfinite(g['skill_random'])


def test_e004_related_designs_family_info():
    from edalint.checks import _family
    assert _family('Vortex-small') == _family('Vortex-large') == 'vortex'
    assert _family('RISCY-FPU') == _family('RISCY')
    assert _family('zero-riscy') != _family('RISCY')


# -- E005 ------------------------------------------------------------------------------------------------------------------------
def test_e005_transposed_labels_detected():
    r = e005_orientation(make_synthetic(transpose_train_labels=True), OPT)
    assert r.status == 'flagged' and has(r, 'error', "channel 'current'") and 'transpose' in r.findings[0].message
    assert r.metrics['median_abs_corr']['current']['transpose'] > r.metrics['median_abs_corr']['current']['stored'] + 0.3


def test_e005_flipped_labels_detected():
    ds = make_synthetic()
    for s in ds._samples.values():
        s.label = np.ascontiguousarray(s.label[::-1, :])
    r = e005_orientation(ds, Options(split='all'))
    assert has(r, 'error', 'flip_ud')


def test_e005_clean_is_ok(clean):
    r = e005_orientation(clean, OPT)
    assert r.status == 'ok' and r.metrics['primary']['stored'] > 0.5


# -- E006 ------------------------------------------------------------------------------------------------------------------------
def test_e006_no_hotspot_case_detected():
    r = e006_metric_sanity(make_synthetic(no_hotspot_label=True), OPT)
    assert has(r, 'error', 'NO hotspot')


def test_e006_sensitivity_and_zero_prediction(clean):
    r = e006_metric_sanity(clean, OPT)
    t = r.metrics['mean_f1_pseudo_predictions']
    assert t['zero@0.9'] == 0.0 and t['golden@0.9'] == 1.0
    assert r.metrics['f1_spread_over_threshold']['scale0.95'] > 0.15            # F1 collapses when the threshold moves
    assert 0 < r.metrics['hotspot_base_rate']['median'] < 0.2


def test_e006_few_hotspot_pixels_warning():
    rng = np.random.default_rng(0)
    samples = []
    for i in range(4):
        L = rng.random((30, 30)) * 0.1
        L[5, 5] = 1.0                                           # one hotspot pixel
        samples.append(Sample(f's{i}', 'train', f'd{i}', {'current': rng.random((30, 30))}, L))
    r = e006_metric_sanity(ListDataset(samples, hotspot={'kind': 'frac_of_max', 'value': 0.9}), OPT)
    assert has(r, 'warning', 'fewer than 5 hotspot pixels')


def test_e006_single_class_for_abs_rule():
    rng = np.random.default_rng(0)
    s = [Sample('a', 'train', 'd', {'x': rng.random((20, 20))}, np.zeros((20, 20))),
         Sample('b', 'train', 'e', {'x': rng.random((20, 20))}, rng.random((20, 20)))]
    r = e006_metric_sanity(ListDataset(s, task='drc', hotspot={'kind': 'abs', 'value': 0.5}), OPT)
    assert has(r, 'warning', 'single class')


# -- E007 ------------------------------------------------------------------------------------------------------------------------
def test_e007_skips_without_eval_split_and_runs_on_train(clean):
    assert e007_baseline_ladder(clean, OPT).status == 'skipped'
    r = e007_baseline_ladder(clean, Options(fit_split='train', eval_split='train', baselines=('zero', 'scalar', 'ridge', 'green')))
    assert r.status == 'ok' and set(r.metrics['ladder']) == {'zero', 'scalar', 'ridge', 'green'}
    assert all(f.severity == 'info' for f in r.findings)


def test_e007_refuses_test_split_without_freeze(clean):
    r = e007_baseline_ladder(clean, Options(fit_split='train', eval_split='test', baselines=('zero',)))
    assert r.status == 'skipped' and 'refused' in r.skip_reason


# -- runner --------------------------------------------------------------------------------------------------------------------
def test_run_checks_refuses_named_test_split(clean):
    from edalint.core import ProtocolError
    with pytest.raises(ProtocolError):
        run_checks(clean, ['E001'], Options(split='test'))


def test_run_checks_split_all_reads_only_non_test_before_freeze(clean):
    rep = run_checks(clean, ['E001', 'E004'], Options(split='all'))
    assert rep.results[0].metrics['n_checked'] == 12 and rep.results[1].status == 'ok'


def test_check_crash_is_reported_not_raised(clean, monkeypatch):
    from edalint import runner

    def boom(ds, opt):
        raise RuntimeError('bug')
    monkeypatch.setitem(runner.CHECKS, 'E002', boom)
    rep = run_checks(clean, ['E001', 'E002'], OPT)
    assert rep.results[1].status == 'flagged' and 'crashed' in rep.results[1].findings[0].message


def test_report_text_markdown_json(clean):
    rep = run_checks(clean, None, OPT)
    assert 'E001' in rep.text() and rep.markdown().startswith('# edalint report')
    d = rep.to_dict()
    assert d['label'] == 'development (train only)' and len(d['checks']) == 6


def test_e002_rounding_noise_is_not_a_negative_label_and_slices_count_once():
    rng = np.random.default_rng(0)
    L = rng.random((20, 20))
    L[0, 0] = -8.9e-16                                   # log10/resize rounding in CircuitNet authors mode
    ok = Sample('a', 'train', 'd', {'power_t': np.zeros((20, 20, 20)), 'power_all': rng.random((20, 20))}, L)
    zl = Sample('b', 'train', 'e', {'power_t': np.zeros((20, 20, 20)), 'power_all': rng.random((20, 20))}, np.full((20, 20), -8.9e-16))
    r = e002_degenerate(ListDataset([ok, zl], hotspot={'kind': 'abs', 'value': 0.5}), OPT)
    assert r.metrics['n_negative_label'] == 0
    assert r.metrics['n_zero_label'] == 1                # a constant at rounding noise is an all-zero label
    assert r.metrics['channel_all_zero_counts'] == {'power_t': 2}          # per sample, not per slice
