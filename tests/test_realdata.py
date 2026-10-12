"""Real-data tests (TRAIN data and file-shape smoke tests only). Skipped when the data is absent."""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.realdata

ICCAD = Path(os.environ.get('EDALINT_ICCAD23', r'E:\edalint_data\iccad23\repo'))
N14 = Path(os.environ.get('EDALINT_N14', r'E:\edalint_data\circuitnet_n14'))


@pytest.fixture(scope='module')
def iccad():
    if not (ICCAD / 'benchmarks' / 'real-circuit-data' / 'testcase1').is_dir():
        pytest.skip('ICCAD23 data not present')
    from edalint.datasets import ICCAD23Dataset
    return ICCAD23Dataset(root=str(ICCAD))


def test_iccad_structure_matches_the_data_card(iccad):
    assert {k: len(v) for k, v in iccad.splits.items()} == {'train': 110, 'test': 10, 'fake': 100, 'visible': 10}
    hidden = sorted(int(i.split('testcase')[1]) for i in iccad.splits['test'])
    assert hidden == [7, 8, 9, 10, 13, 14, 15, 16, 19, 20]
    # file-layout shapes of the hidden cases (no values read): grid sides from the data card
    sides = {i: iccad.shape_of(i)[0] for i in iccad.splits['test']}
    assert sorted(set(sides.values())) == [257, 489, 601, 835, 870]
    designs = {iccad.meta(i)['design_id'] for i in iccad.splits['test']}
    assert len(designs) == 5                                          # 10 hidden cases = 5 designs
    assert all(iccad.meta(i)['twin_of'] for i in iccad.splits['visible'])


def test_iccad_visible_case_loads_at_native_size(iccad):
    s = iccad.get('real/testcase11')
    assert s.shape == (204, 204) and all(v.shape == (204, 204) for v in s.inputs.values())
    assert np.isfinite(s.label).all() and s.label.min() > 0


def test_exact_solver_reproduces_train_labels(iccad):
    from edalint.spice import compare_to_label, exact_ir_drop
    for sid in ('real/testcase11', 'real/testcase1'):
        s = iccad.get(sid)
        pred, info = exact_ir_drop(s.meta['netlist'], s.shape)
        c = compare_to_label(pred, s.label)
        assert c['mae_mV'] < 0.02 and c['pearson'] > 0.999 and info['rel_residual'] < 1e-10
    s = iccad.get('fake/current_map01')
    pred, _ = exact_ir_drop(s.meta['netlist'], s.shape)
    assert compare_to_label(pred, s.label)['mae_mV'] < 0.01


def test_n14_vortex_small_routability_train_subset():
    tar = N14 / 'CircuitNet-N14' / 'routability_features' / 'Vortex-small.tar.gz'
    if not tar.is_file():
        pytest.skip('N14 Vortex-small routability tarball not present')
    from edalint.datasets import N14Dataset
    ds = N14Dataset(task='drc', root=str(N14), designs=['Vortex-small'], max_per_design=2)
    if not ds.splits['train']:
        pytest.skip('Vortex-small not usable (see ds.skipped)')
    s = ds.get(ds.ids('train')[0])
    assert len(s.inputs) == 9 and s.label.shape == (256, 256) and 0 <= s.label.min() and s.label.max() <= 1
