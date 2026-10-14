"""Tests for the Kaggle entry (scripts/p7_kernel_entry.py) bundle locator and the bundle file list (scripts/p7_make_bundle.py). Nothing is run or uploaded."""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bundle(root: Path, name='edalint-p7-code-frozen'):
    (root / name / 'protocol').mkdir(parents=True)
    (root / name / 'protocol' / 'FROZEN_p7.json').write_text('{}')
    return root / name


def test_find_bundle_accepts_exactly_the_pinned_one(tmp_path):
    e = _load('p7_kernel_entry')
    b = _bundle(tmp_path)
    (tmp_path / 'edalint-p7-train-ir').mkdir()                      # a data input is not a bundle candidate
    assert e.find_bundle(tmp_path) == b


def test_find_bundle_refuses_missing_or_second_candidate(tmp_path):
    e = _load('p7_kernel_entry')
    with pytest.raises(SystemExit, match='not attached'):
        e.find_bundle(tmp_path)
    _bundle(tmp_path)
    _bundle(tmp_path, 'edalint-p7-code-frozen-old')
    with pytest.raises(SystemExit, match='more than one'):
        e.find_bundle(tmp_path)


def test_entry_stages_and_usage(capsys):
    e = _load('p7_kernel_entry')
    assert set(e.STAGES) == {'train', 'score'} and e.main([]) == 2


def test_bundle_file_list_is_exactly_the_registered_set(tmp_path):
    m = _load('p7_make_bundle')
    r = tmp_path
    for rel in ('src/edalint/a.py', 'src/edalint/data/split.csv', 'src/edalint/__pycache__/a.pyc', 'src/edalint_p7/b.py', 'src/edalint_p7/__pycache__/b.pyc',
                'scripts/p7_run_train.py', 'scripts/p7_run_score.py', 'scripts/p7_dev_x.py', 'scripts/p1_make_manifest.py', 'scripts/p7_kernel_entry.py',
                'protocol/FROZEN_p7.json', 'protocol/protocol_p7.md', 'protocol/protocol_n14.md', 'results/n14/models_ir_drop.json',
                'results/n14/models_congestion.json'):
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text('x')
    got = [p.relative_to(r).as_posix() for p in m.bundle_files(r)]
    assert got == ['protocol/FROZEN_p7.json', 'protocol/protocol_p7.md', 'results/n14/models_ir_drop.json', 'scripts/p7_run_score.py',
                   'scripts/p7_run_train.py', 'src/edalint/a.py', 'src/edalint/data/split.csv', 'src/edalint_p7/b.py']
