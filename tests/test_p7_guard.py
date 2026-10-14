"""P7 pre-freeze tests: p7_code_hash recipe, freeze-guard states, open_test_rows, kernel-start check, write_once, protocol text.
No test row is read: the dataset factory is a stub, protocol and repo directories are temporary."""
import hashlib
import json
import re
from pathlib import Path

import pytest

from edalint.core import ProtocolError
from edalint.protocol import code_hash
from edalint.runner import ResultExists, write_once
from edalint_p7 import guard
from edalint_p7.guard import check_p7_freeze, kernel_start_check, open_test_rows, p7_code_hash

REPO = Path(__file__).resolve().parents[1]


def make_repo(root: Path, body=b'x = 1\n'):
    (root / 'src' / 'edalint_p7' / '__pycache__').mkdir(parents=True)
    (root / 'scripts').mkdir()
    (root / 'src' / 'edalint_p7' / '__init__.py').write_bytes(b'"""p7"""\n')
    (root / 'src' / 'edalint_p7' / 'a.py').write_bytes(body)
    (root / 'scripts' / 'p7_run_x.py').write_bytes(b'print(1)\n')
    (root / 'scripts' / 'p7_dev_y.py').write_bytes(b'print(2)\n')
    (root / 'protocol').mkdir()
    return root


def write_freeze(root: Path, **over):
    rec = {'dataset': 'p7', 'code_hash': code_hash(), 'p7_code_hash': p7_code_hash(root)}
    rec.update(over)
    rec = {k: v for k, v in rec.items() if v is not None}
    (root / 'protocol' / 'FROZEN_p7.json').write_text(json.dumps(rec))
    return rec


# -- hash recipe -----------------------------------------------------------------------------------------------------------------------
def test_hash_matches_the_documented_recipe(tmp_path):
    r = make_repo(tmp_path)
    h = hashlib.sha256()
    for rel in ('scripts/p7_run_x.py', 'src/edalint_p7/__init__.py', 'src/edalint_p7/a.py'):          # sorted posix paths
        h.update(rel.encode() + b'\0' + (r / rel).read_bytes() + b'\0')
    assert p7_code_hash(r) == h.hexdigest()


def test_hash_changes_on_a_one_byte_edit(tmp_path):
    r = make_repo(tmp_path)
    before = p7_code_hash(r)
    (r / 'src' / 'edalint_p7' / 'a.py').write_bytes(b'x = 2\n')
    assert p7_code_hash(r) != before
    (r / 'src' / 'edalint_p7' / 'a.py').write_bytes(b'x = 1\n')
    assert p7_code_hash(r) == before
    (r / 'scripts' / 'p7_run_x.py').write_bytes(b'print(2)\n')
    assert p7_code_hash(r) != before


def test_hash_equal_under_crlf_and_ignores_pycache_and_dev_scripts(tmp_path):
    r = make_repo(tmp_path)
    before = p7_code_hash(r)
    (r / 'src' / 'edalint_p7' / 'a.py').write_bytes(b'x = 1\r\n')
    assert p7_code_hash(r) == before
    (r / 'src' / 'edalint_p7' / '__pycache__' / 'junk.py').write_bytes(b'garbage')
    (r / 'scripts' / 'p7_dev_y.py').write_bytes(b'print(99)\n')
    (r / 'scripts' / 'p7_dev_new.py').write_bytes(b'print(100)\n')
    (r / 'scripts' / 'other.py').write_bytes(b'print(5)\n')
    assert p7_code_hash(r) == before


def test_real_repo_hash_is_defined_and_includes_the_new_modules():
    names = {p.relative_to(guard.REPO_ROOT).as_posix() for p in guard.p7_files()}
    assert {'src/edalint_p7/guard.py', 'src/edalint_p7/train.py', 'src/edalint_p7/recipe.py', 'scripts/p7_run_train.py'} <= names
    assert not any('p7_dev' in n for n in names) and re.fullmatch('[0-9a-f]{64}', p7_code_hash())


# -- guard states ----------------------------------------------------------------------------------------------------------------------
def test_refuses_without_freeze_file(tmp_path):
    r = make_repo(tmp_path)
    with pytest.raises(ProtocolError, match='does not exist'):
        check_p7_freeze(str(r / 'protocol'), r)


def test_refuses_legacy_freeze_of_another_dataset(tmp_path):
    r = make_repo(tmp_path)
    (r / 'protocol' / 'FROZEN.json').write_text(json.dumps({'dataset': 'iccad23', 'code_hash': code_hash(), 'p7_code_hash': p7_code_hash(r)}))
    with pytest.raises(ProtocolError, match='another dataset'):
        check_p7_freeze(str(r / 'protocol'), r)


def test_refuses_n14_freeze_alone(tmp_path):
    r = make_repo(tmp_path)
    (r / 'protocol' / 'FROZEN_n14.json').write_text(json.dumps({'dataset': 'n14', 'code_hash': code_hash()}))
    with pytest.raises(ProtocolError):
        check_p7_freeze(str(r / 'protocol'), r)


def test_refuses_wrong_dataset_field(tmp_path):
    r = make_repo(tmp_path)
    write_freeze(r, dataset='n14')
    with pytest.raises(ProtocolError, match="not 'p7'"):
        check_p7_freeze(str(r / 'protocol'), r)


def test_refuses_missing_code_hash_field(tmp_path):
    r = make_repo(tmp_path)
    write_freeze(r, code_hash=None)
    with pytest.raises(ProtocolError, match='no "code_hash"'):
        check_p7_freeze(str(r / 'protocol'), r)


def test_refuses_old_field_name_edalint_code_hash(tmp_path):
    r = make_repo(tmp_path)
    write_freeze(r, code_hash=None, edalint_code_hash=code_hash())
    with pytest.raises(ProtocolError, match='no "code_hash"'):
        check_p7_freeze(str(r / 'protocol'), r)


def test_refuses_edalint_hash_mismatch(tmp_path):
    r = make_repo(tmp_path)
    write_freeze(r, code_hash='0' * 64)
    with pytest.raises(ProtocolError, match='code hash differs'):
        check_p7_freeze(str(r / 'protocol'), r)


def test_refuses_missing_p7_hash_field(tmp_path):
    r = make_repo(tmp_path)
    write_freeze(r, p7_code_hash=None)
    with pytest.raises(ProtocolError, match='p7_code_hash'):
        check_p7_freeze(str(r / 'protocol'), r)


def test_refuses_p7_hash_mismatch_and_edit_after_freeze(tmp_path):
    r = make_repo(tmp_path)
    write_freeze(r, p7_code_hash='f' * 64)
    with pytest.raises(ProtocolError, match='P7 code hash differs'):
        check_p7_freeze(str(r / 'protocol'), r)
    write_freeze(r)
    check_p7_freeze(str(r / 'protocol'), r)                                            # accepted ...
    (r / 'src' / 'edalint_p7' / 'a.py').write_bytes(b'x = 3\n')                         # ... until the code is edited
    with pytest.raises(ProtocolError, match='P7 code hash differs'):
        check_p7_freeze(str(r / 'protocol'), r)


def test_accepts_when_everything_matches(tmp_path):
    r = make_repo(tmp_path)
    rec = write_freeze(r, frozen_commit='abc')
    assert check_p7_freeze(str(r / 'protocol'), r)['frozen_commit'] == rec['frozen_commit']


def test_real_repo_is_not_frozen_for_p7():
    with pytest.raises(ProtocolError):
        check_p7_freeze(str(REPO / 'protocol'), REPO)               # no protocol/FROZEN_p7.json in this commit; legacy FROZEN.json is iccad23's


# -- open_test_rows ------------------------------------------------------------------------------------------------------------------
class _FakeDS:
    def __init__(self, **kw):
        self.kw = kw
        self._wanted = {('test', 'zero-riscy'): ['a'], ('train', 'RISCY'): ['b'], ('test', 'Vortex-large'): ['c']}
        self.staged = []

    def stage_design(self, des, splits):
        self.staged.append((des, tuple(splits)))

    def ids(self, split):
        return ['zero-riscy_1', 'Vortex-large_1', 'zero-riscy_2'] if split == 'test' else []

    def meta(self, sid):
        return {'design': sid.rsplit('_', 1)[0]}


def _boom(**kw):
    raise AssertionError('the dataset factory must not be called before the guard passes')


def test_open_test_rows_refuses_before_the_freeze(tmp_path):
    r = make_repo(tmp_path)
    with pytest.raises(ProtocolError):
        open_test_rows('congestion', protocol_dir=str(r / 'protocol'), repo_root=r, _factory=_boom)
    write_freeze(r, p7_code_hash='1' * 64)
    with pytest.raises(ProtocolError):
        open_test_rows('congestion', protocol_dir=str(r / 'protocol'), repo_root=r, _factory=_boom)


def test_open_test_rows_default_arguments_refuse_in_this_checkout():
    with pytest.raises(ProtocolError):
        open_test_rows('ir_drop', _factory=_boom)               # default protocol dir (empty via conftest), default repo root


def test_open_test_rows_stages_test_designs_only_after_the_guard(tmp_path):
    r = make_repo(tmp_path)
    write_freeze(r)
    tr = open_test_rows('congestion', protocol_dir=str(r / 'protocol'), repo_root=r, _factory=_FakeDS)
    assert tr.ds.staged == [('Vortex-large', ('test',)), ('zero-riscy', ('test',))]       # no train staging
    assert tr.ids_by_design == {'Vortex-large': ['Vortex-large_1'], 'zero-riscy': ['zero-riscy_1', 'zero-riscy_2']}
    assert tr.ds.kw['stage'] is False and tr.ds.kw['task'] == 'congestion'


# -- kernel start --------------------------------------------------------------------------------------------------------------------
def test_kernel_start_check(tmp_path):
    r = make_repo(tmp_path)
    with pytest.raises(ProtocolError):
        kernel_start_check(r)                                   # bundle without a freeze file
    write_freeze(r)
    assert kernel_start_check(r)['dataset'] == 'p7'
    (r / 'scripts' / 'p7_run_x.py').write_bytes(b'print(3)\n')
    with pytest.raises(ProtocolError, match='P7 code hash differs'):
        kernel_start_check(r)


def test_run_scripts_exit_2_before_the_freeze(tmp_path, capsys):
    import importlib.util
    for name, args in (('p7_run_train', ['--task', 'congestion', '--shards', 'x', '--iters', '1', '--seed', '1', '--bundle', str(make_repo(tmp_path / 'b'))]),
                       ('p7_run_score', ['--task', 'congestion', '--ckpt', 'x', '--seed', '1', '--bundle', str(make_repo(tmp_path / 'c'))])):
        spec = importlib.util.spec_from_file_location(name, REPO / 'scripts' / f'{name}.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.main(args) == 2
        assert 'refused' in capsys.readouterr().out


# -- write_once ----------------------------------------------------------------------------------------------------------------------
def test_write_once_exclusive_for_p7(tmp_path):
    p = tmp_path / 'results' / 'p7' / 'train_x.json'
    write_once(p, {'a': 1}, with_provenance=False, dataset='p7')
    with pytest.raises(ResultExists):
        write_once(p, {'a': 2}, with_provenance=False, dataset='p7')
    assert json.loads(p.read_text())['a'] == 1
    write_once(tmp_path / 'with_prov.json', {'a': 1}, dataset='p7')                       # with provenance, not frozen: frozen_protocol None
    assert json.loads((tmp_path / 'with_prov.json').read_text())['provenance']['frozen_protocol'] is None


# -- protocol text ---------------------------------------------------------------------------------------------------------------------
def test_protocol_section_6_names_the_freeze_fields_correctly():
    text = (REPO / 'protocol' / 'protocol_p7.md').read_text(encoding='utf-8')
    sec = text[text.index('## 6.'):text.index('## 7.')]
    assert '"code_hash"' in sec and '"p7_code_hash"' in sec
    assert 'edalint_code_hash' not in sec
    assert 'scripts/p7_run_*.py' in sec and 'scripts/p7_*.py' not in sec
    assert 'NOT FROZEN' in text.splitlines()[2] and not (REPO / 'protocol' / 'FROZEN_p7.json').exists()
