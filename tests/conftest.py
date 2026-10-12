import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import pytest


@pytest.fixture(autouse=True)
def isolated_freeze_state(tmp_path_factory, monkeypatch):
    """Point the default protocol directory at an empty temp dir, so the repository's real freeze files never affect a test.
    Tests that need a freeze file pass an explicit protocol_dir / write into their own tmp_path."""
    monkeypatch.setenv('EDALINT_PROTOCOL_DIR', str(tmp_path_factory.mktemp('protocol_empty')))
