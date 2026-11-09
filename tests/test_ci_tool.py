"""The CI side of edalint: folder datasets, configuration, fail-on exit codes, baselines, SARIF / JUnit / GitHub output."""
import csv
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from edalint import ci
from edalint.cli import main
from edalint.config import ConfigError, find_config, fails
from edalint.core import Finding, ProtocolError
from edalint.datasets import FolderDataset, open_dataset
from edalint.datasets.synthetic import make_synthetic


def write_folder(root, **faults):
    ds = make_synthetic(**faults)
    rows = []
    for sid in ds.ids('all'):
        s = ds._samples[sid]
        d = root / s.split
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f'{sid}.npz', label=s.label, **s.inputs)
        rows.append({'id': f'{s.split}/{sid}', 'design': s.design, 'twin_of': ''})
    with open(root / 'samples.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['id', 'design', 'twin_of'])
        w.writeheader()
        w.writerows(rows)
    return root


@pytest.fixture
def clean(tmp_path):
    return write_folder(tmp_path / 'clean')


@pytest.fixture
def faulty(tmp_path):
    return write_folder(tmp_path / 'faulty', n_duplicates=1, leak_design=True, n_nan=1)


def run(capsys, *argv):
    code = main(['lint', *map(str, argv)])
    return code, capsys.readouterr()


# -- folder dataset -----------------------------------------------------------------------------------------------------------------
def test_folder_dataset_reads_splits_designs_and_arrays(faulty):
    ds = open_dataset(str(faulty))
    assert isinstance(ds, FolderDataset) and set(ds.splits) == {'train', 'test'}
    sid = ds.ids('train')[0]
    s = ds.get(sid)
    assert s.design == ds.meta(sid)['design'] and set(s.inputs) == {'current', 'eff_dist', 'pdn_density'}
    assert ds.test_splits == ()                                  # nothing guarded unless configured
    assert ds.primary_input == 'current'


def test_folder_without_csv_uses_sample_ids_and_root_split(tmp_path):
    for i in range(3):
        np.savez(tmp_path / f's{i}.npz', label=np.random.rand(8, 8), x=np.random.rand(8, 8))
    ds = FolderDataset(tmp_path)
    assert list(ds.splits) == ['all-samples'] and ds.meta('s0')['design'] == 's0'


def test_folder_rejects_label_only_and_missing_label(tmp_path, capsys):
    np.savez(tmp_path / 'a.npz', label=np.zeros((4, 4)))
    with pytest.raises(ValueError, match='only the label'):
        FolderDataset(tmp_path)
    assert run(capsys, tmp_path)[0] == 5
    np.savez(tmp_path / 'a.npz', y=np.zeros((4, 4)), x=np.zeros((4, 4)))
    with pytest.raises(ValueError, match='no label array'):
        FolderDataset(tmp_path)


def test_folder_test_split_is_guarded_only_when_configured(faulty):
    ds = FolderDataset(faulty, test_splits=['test'])
    with pytest.raises(ProtocolError):
        ds.get(ds.ids('test')[0])


def test_builtin_names_win_over_a_folder_of_the_same_name(tmp_path, monkeypatch):
    (tmp_path / 'iccad23').mkdir()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match='ICCAD23'):        # the benchmark loader, not the folder reader
        open_dataset('iccad23', root=str(tmp_path / 'nowhere'))
    with pytest.raises(ValueError, match='no .npz files'):          # folder: forces the folder reader
        open_dataset('folder:iccad23')


# -- exit codes, select / ignore -------------------------------------------------------------------------------------------------
def test_exit_codes_follow_fail_on(faulty, capsys):
    assert run(capsys, faulty, '--no-config')[0] == 2
    assert run(capsys, faulty, '--no-config', '--fail-on', 'never')[0] == 0
    assert run(capsys, faulty, '--no-config', '--checks', 'E006', '--fail-on', 'error')[0] == 0
    assert run(capsys, faulty, '--no-config', '--checks', 'E006')[0] == 1
    assert run(capsys, faulty, '--no-config', '--checks', 'E005', '--fail-on', 'info')[0] == 0
    assert fails('info', 'info') and not fails('none', 'info') and not fails('error', 'never')


def test_clean_data_passes_with_e006_ignored(clean, capsys):
    code, out = run(capsys, clean, '--no-config', '--ignore', 'E006')
    assert code == 0, out.out
    assert 'E006' not in out.out


def test_unknown_check_and_split_are_usage_errors(faulty, capsys):
    assert run(capsys, faulty, '--no-config', '--checks', 'E999')[0] == 5
    assert run(capsys, faulty, '--no-config', '--split', 'nope')[0] == 5
    assert run(capsys, faulty, '--no-config', '--ignore', 'E001,E002,E003,E004,E005,E006')[0] == 5


# -- configuration --------------------------------------------------------------------------------------------------------------
def test_config_file_sets_dataset_checks_and_fail_on(faulty, tmp_path, capsys, monkeypatch):
    proj = tmp_path / 'proj'
    proj.mkdir()
    (proj / 'edalint.toml').write_text(f'dataset = "../faulty"\nselect = ["E004", "E006"]\nfail-on = "error"\n'
                                       '[options]\nf1_fracs = [0.9]\n')
    monkeypatch.chdir(proj)
    code, out = run(capsys, '--format', 'json')
    rep = json.loads(out.out)
    assert [c['code'] for c in rep['checks']] == ['E004', 'E006'] and rep['fail_on'] == 'error'
    assert code == rep['exit_code'] == 2                       # E004: a test design is also in train


def test_pyproject_tool_table_and_unknown_keys(tmp_path):
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "x"\n[tool.edalint]\nfail-on = "info"\n')
    cfg, path = find_config(cwd=tmp_path)
    assert cfg == {'fail-on': 'info'} and path.name == 'pyproject.toml'
    (tmp_path / 'edalint.toml').write_text('fail_on = "info"\n')
    with pytest.raises(ConfigError, match='unknown key'):
        find_config(cwd=tmp_path)
    (tmp_path / 'edalint.toml').write_text('[options]\nnot_an_option = 1\n')
    with pytest.raises(ConfigError, match=r'\[options\]'):
        find_config(cwd=tmp_path)
    (tmp_path / 'edalint.toml').write_text('fail-on = "sometimes"\n')
    with pytest.raises(ConfigError, match='fail-on'):
        find_config(cwd=tmp_path)
    with pytest.raises(ConfigError, match='does not exist'):
        find_config(str(tmp_path / 'missing.toml'))


def test_config_in_the_dataset_folder_is_found(faulty, capsys, tmp_path, monkeypatch):
    (faulty / 'edalint.toml').write_text('select = ["E001"]\n')
    monkeypatch.chdir(tmp_path)
    _, out = run(capsys, faulty, '--format', 'json')
    assert [c['code'] for c in json.loads(out.out)['checks']] == ['E001']


# -- baseline -------------------------------------------------------------------------------------------------------------------
def test_baseline_round_trip_hides_only_accepted_findings(faulty, tmp_path, capsys):
    base = tmp_path / 'base.json'
    assert run(capsys, faulty, '--no-config', '--write-baseline', base)[0] == 0
    assert run(capsys, faulty, '--no-config', '--baseline', base)[0] == 0
    data = json.loads(base.read_text())
    data['findings'] = [f for f in data['findings'] if f['code'] != 'E004']
    base.write_text(json.dumps(data))
    code, out = run(capsys, faulty, '--no-config', '--baseline', base, '--format', 'json')
    left = [f for c in json.loads(out.out)['checks'] for f in c['findings']]
    assert code == 2 and {f['code'] for f in left} == {'E004'}
    assert run(capsys, faulty, '--no-config', '--baseline', tmp_path / 'missing.json')[0] == 5


def test_fingerprint_changes_with_the_message():
    a = Finding('E002', 'error', '3 samples have zero labels')
    b = Finding('E002', 'error', '4 samples have zero labels')
    assert ci.fingerprint('d', a) != ci.fingerprint('d', b) and ci.fingerprint('d', a) != ci.fingerprint('e', a)


# -- SARIF / JUnit / GitHub ------------------------------------------------------------------------------------------------------
def test_sarif_report(faulty, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(faulty.parent)
    out = tmp_path / 'r' / 'edalint.sarif'
    run(capsys, faulty.name, '--no-config', '--sarif', out)
    d = json.loads(out.read_text())
    assert d['version'] == '2.1.0'
    run_ = d['runs'][0]
    ids = [r['id'] for r in run_['tool']['driver']['rules']]
    assert ids == sorted(ids) and 'E004' in ids
    res = run_['results']
    assert res and all(r['level'] in ('error', 'warning', 'note') for r in res)
    assert all(ids[r['ruleIndex']] == r['ruleId'] for r in res)
    assert all(r['locations'] for r in res)                    # a sample file, else samples.csv
    uris = {r['locations'][0]['physicalLocation']['artifactLocation']['uri'] for r in res}
    assert any(u.endswith('.npz') for u in uris) and all('\\' not in u for u in uris)
    assert 'faulty/samples.csv' in uris                         # E004 names designs, not samples


def test_junit_report(faulty, tmp_path, capsys):
    out = tmp_path / 'junit.xml'
    run(capsys, faulty, '--no-config', '--junit', out, '--fail-on', 'error')
    suite = ET.parse(out).getroot()
    assert suite.tag == 'testsuite' and suite.get('tests') == '6'
    failed = {c.get('name').split()[0] for c in suite if c.find('failure') is not None}
    assert failed == {'E001', 'E003', 'E004'}                    # E006 has warnings only, below fail-on error


def test_github_annotations_escape_and_point_at_files(faulty, capsys, monkeypatch):
    monkeypatch.chdir(faulty.parent)
    _, out = run(capsys, faulty.name, '--no-config', '--format', 'github')
    lines = out.out.strip().splitlines()
    assert lines and all(l.startswith(('::error ', '::warning ', '::notice ')) for l in lines)
    assert any('file=faulty/train/' in l for l in lines)
    assert ci._gh_escape('a:b,c\n%', prop=True) == 'a%3Ab%2Cc%0A%25'


def test_version_and_checks_listing(capsys):
    import edalint
    with pytest.raises(SystemExit) as e:
        main(['--version'])
    assert e.value.code == 0 and edalint.__version__ in capsys.readouterr().out
    assert main(['checks']) == 0
    assert capsys.readouterr().out.count('\nE0') == 6
