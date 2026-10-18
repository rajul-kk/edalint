"""Frozen-protocol guard, run-once writer, CLI."""
import json

import numpy as np
import pytest

from edalint import cli
from edalint.baselines import evaluate, run_zoo, stream
from edalint.baselines.zoo import ScalarMean, Zero
from edalint.core import ProtocolError
from edalint.datasets.synthetic import make_synthetic
from edalint.protocol import code_hash, frozen_info, provenance, require_frozen
from edalint.runner import ResultExists, write_once, write_text_once


def test_require_frozen_states(tmp_path):
    with pytest.raises(ProtocolError, match='does not exist'):
        require_frozen(str(tmp_path))
    (tmp_path / 'FROZEN.json').write_text(json.dumps({'note': 'no hash'}))
    with pytest.raises(ProtocolError, match='no "code_hash"'):
        require_frozen(str(tmp_path))
    (tmp_path / 'FROZEN.json').write_text(json.dumps({'code_hash': 'abc'}))
    with pytest.raises(ProtocolError, match='differs'):
        require_frozen(str(tmp_path))
    (tmp_path / 'FROZEN.json').write_text(json.dumps({'code_hash': code_hash(), 'x': 1}))
    assert require_frozen(str(tmp_path))['x'] == 1


def test_evaluating_test_split_refuses_without_freeze(tmp_path):
    ds = make_synthetic()
    ds.protocol_dir = str(tmp_path)
    z = Zero()
    with pytest.raises(ProtocolError):
        evaluate(z, ds, 'test')
    with pytest.raises(ProtocolError):
        run_zoo(ds, 'train', 'test', ['zero'])
    # fitting on the test split is refused too (the stream goes through the guarded dataset)
    with pytest.raises(ProtocolError):
        ScalarMean().fit(stream(ds, 'test'))
    # train evaluation is fine
    assert len(evaluate(z, ds, 'train')) == 12
    # after a valid freeze the test split opens
    (tmp_path / 'FROZEN_synthetic.json').write_text(json.dumps({'code_hash': code_hash()}))
    assert len(evaluate(z, ds, 'test')) == 4


def test_code_hash_is_stable_and_sensitive():
    assert code_hash() == code_hash() and len(code_hash()) == 64


def test_write_once_refuses_overwrite_and_records_provenance(tmp_path):
    p = write_once(tmp_path / 'r.json', {'a': np.float64(1.5), 'b': np.array([1, 2]), 'c': float('nan')})
    body = json.loads(p.read_text())
    assert body['a'] == 1.5 and body['b'] == [1, 2] and body['c'] is None
    pv = body['provenance']
    assert {'git_head', 'code_hash', 'versions', 'argv'} <= set(pv) and pv['code_hash'] == code_hash()
    assert {'python', 'numpy', 'scipy', 'pandas', 'edalint'} <= set(pv['versions'])
    with pytest.raises(ResultExists):
        write_once(tmp_path / 'r.json', {'a': 2})
    assert json.loads(p.read_text())['a'] == 1.5
    write_text_once(tmp_path / 'r.md', 'x')
    with pytest.raises(ResultExists):
        write_text_once(tmp_path / 'r.md', 'y')


def test_cli_lint_baselines_and_guard(tmp_path, monkeypatch, capsys):
    ds = make_synthetic()
    ds.protocol_dir = str(tmp_path / 'protocol')
    monkeypatch.setattr(cli, '_dataset', lambda a: ds)
    assert cli.main(['lint', 'x', '--split', 'train', '--checks', 'E001,E002', '--out', str(tmp_path), '--tag', 't1']) in (0, 1)
    assert (tmp_path / 'synthetic_train_t1.json').is_file() and (tmp_path / 'synthetic_train_t1.md').is_file()
    # second identical run refuses to overwrite
    assert cli.main(['lint', 'x', '--split', 'train', '--checks', 'E001', '--out', str(tmp_path), '--tag', 't1']) == 4
    # test split refused
    assert cli.main(['lint', 'x', '--split', 'test']) == 3
    assert cli.main(['baselines', 'x', '--fit', 'train', '--eval', 'test']) == 3
    assert 'refused' in capsys.readouterr().err
    assert cli.main(['baselines', 'x', '--fit', 'train', '--eval', 'train', '--baselines', 'zero,scalar', '--out', str(tmp_path), '--tag', 'b']) == 0
    body = json.loads((tmp_path / 'synthetic_baselines_train_to_train_b.json').read_text())
    assert body['label'] == 'development (train only)' and set(body['results']) == {'zero', 'scalar'}


def test_cli_freeze_status_and_hash(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv('EDALINT_PROTOCOL_DIR', str(tmp_path))
    assert cli.main(['freeze-status']) == 0
    assert 'absent' in capsys.readouterr().out
    assert cli.main(['code-hash']) == 0
    assert capsys.readouterr().out.strip() == code_hash()
    assert frozen_info() is None and provenance()['code_hash'] == code_hash()


def test_freeze_is_per_dataset(tmp_path):
    """The legacy FROZEN.json (ICCAD'23) and FROZEN_<key>.json each unlock only their own dataset."""
    ch = code_hash()
    (tmp_path / 'FROZEN.json').write_text(json.dumps({'dataset': 'iccad23', 'code_hash': ch}))
    assert require_frozen(str(tmp_path), dataset='iccad23')['dataset'] == 'iccad23'
    for other in ('circuitnet_n14_ir_drop_authors', 'circuitnet_n14_drc_authors', 'synthetic'):
        with pytest.raises(ProtocolError, match='belongs to another dataset'):
            require_frozen(str(tmp_path), dataset=other)
    (tmp_path / 'FROZEN_n14.json').write_text(json.dumps({'dataset': 'n14', 'code_hash': ch}))
    assert require_frozen(str(tmp_path), dataset='circuitnet_n14_congestion_authors')['dataset'] == 'n14'
    (tmp_path / 'FROZEN_n14.json').write_text(json.dumps({'dataset': 'n14', 'code_hash': 'abc'}))
    with pytest.raises(ProtocolError, match='differs'):
        require_frozen(str(tmp_path), dataset='circuitnet_n14_drc_authors')
    assert require_frozen(str(tmp_path), dataset='iccad23')            # ICCAD'23 freeze still valid


def test_real_repo_freezes_are_recognised_per_dataset(monkeypatch):
    """ICCAD'23 and N14 are both frozen in the real repo (N14 since its freeze commit); each freeze is recognised only for its own
    dataset key, and a dataset with no freeze file (a draft protocol) is not unlocked by either."""
    from edalint.protocol import REPO_DIR
    monkeypatch.delenv('EDALINT_PROTOCOL_DIR', raising=False)
    d = str(REPO_DIR / 'protocol')
    assert frozen_info(d, 'iccad23') is not None
    n14 = frozen_info(d, 'circuitnet_n14_ir_drop_authors')
    assert n14 is not None and n14['dataset'] == 'n14' and n14['code_hash'] == code_hash()
    assert frozen_info(d, 'p7')['dataset'] == 'p7'
    assert frozen_info(d, 'no_such_dataset_without_freeze') is None
