"""P9 guard: test data only after FROZEN_p9.json with both hashes; static check that nothing else opens test data."""
import json
import re
from pathlib import Path

import pytest

from edalint.core import ProtocolError
from edalint.protocol import code_hash
from edalint_p9 import data as D
from edalint_p9 import guard as G

REPO = Path(__file__).resolve().parents[1]


def _freeze(tmp_path, **over):
    rec = {'dataset': 'p9', 'code_hash': code_hash(), 'p9_code_hash': G.p9_code_hash()}
    rec.update(over)
    (tmp_path / 'FROZEN_p9.json').write_text(json.dumps(rec))
    return tmp_path


def test_refuses_without_freeze(tmp_path):
    with pytest.raises(ProtocolError):
        G.check_p9_freeze(tmp_path)
    with pytest.raises(ProtocolError):
        G.open_n14_test('ir_drop', protocol_dir=tmp_path)


def test_refuses_on_either_hash_mismatch(tmp_path):
    with pytest.raises(ProtocolError, match='P9 code hash'):
        G.check_p9_freeze(_freeze(tmp_path, p9_code_hash='0' * 64))
    with pytest.raises(ProtocolError, match='code hash differs'):
        G.check_p9_freeze(_freeze(tmp_path, code_hash='0' * 64))
    with pytest.raises(ProtocolError, match='p9_code_hash'):
        G.check_p9_freeze(_freeze(tmp_path, p9_code_hash=''))


def test_accepts_a_valid_freeze_and_unlocks_test_designs(tmp_path):
    cache, info = G.open_n14_test('ir_drop', protocol_dir=_freeze(tmp_path), root=tmp_path / 'nocache')
    assert info['dataset'] == 'p9' and cache._allow_test
    assert cache.listing('zero-riscy') == []                         # empty cache root: nothing to list, but not refused


def test_cache_refuses_test_designs_without_the_guard(tmp_path):
    c = D.N14Cache('congestion', root=tmp_path)
    with pytest.raises(ProtocolError):
        c.listing('zero-riscy')
    with pytest.raises(ProtocolError):
        c.load('Vortex-large', 'x')
    assert c.listing('RISCY') == []
    assert D.N14Cache('congestion', root=tmp_path, allow_test=True)._allow_test is False    # only the guard's token unlocks


def test_hash_covers_package_and_run_scripts_only():
    names = [p.relative_to(REPO).as_posix() for p in G.p9_files()]
    assert all(n.startswith('src/edalint_p9/') or re.match(r'scripts/p9_run_\w+\.py$', n) for n in names)
    assert 'src/edalint_p9/guard.py' in names


def test_static_only_the_guard_opens_test_data():
    """No P9 file other than guard.py passes the guard token or calls the ICCAD'23 unguarded loaders; dev scripts never import the openers."""
    files = list((REPO / 'src' / 'edalint_p9').glob('*.py')) + list((REPO / 'scripts').glob('p9_*.py'))
    for p in files:
        t = p.read_text(encoding='utf-8')
        if p.name in ('guard.py', 'data.py'):
            continue
        assert '_GUARD_TOKEN' not in t, p
        assert 'load_unguarded' not in t and '._load(' not in t, p
        if p.name.startswith('p9_dev_'):
            assert 'open_n14_test' not in t and 'open_iccad_hidden' not in t, p
