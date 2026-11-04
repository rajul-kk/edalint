"""The hash lineage file pins the current code hash, so any later edit of src/edalint must be recorded on purpose; the registered freezes keep
the hash of the code that produced their results; the merged E003 equals the P8 variant."""
import json
from pathlib import Path

import numpy as np

from edalint.checks import Options, e003_duplicates
from edalint.core import ListDataset, Sample
from edalint.protocol import REPO_DIR, code_hash
from edalint_p8.e003_fixed import e003_duplicates as e003_variant

REGISTERED = 'd3f6d2f67218794687d736ab2d8fe0ab095cced40ff8bd250bf36191a7f5a2b7'
LINEAGE = json.loads((Path(REPO_DIR) / 'protocol' / 'CODE_HASH_LINEAGE.json').read_text(encoding='utf-8'))['lineage']


def test_lineage_ends_at_the_current_hash():
    assert LINEAGE[-1]['code_hash'] == code_hash(), 'src/edalint changed: append an entry to protocol/CODE_HASH_LINEAGE.json'
    assert LINEAGE[0]['code_hash'] == REGISTERED


def test_registered_freezes_keep_the_registered_hash():
    for name in ('FROZEN_n14.json', 'FROZEN_p7.json'):
        assert json.loads((Path(REPO_DIR) / 'protocol' / name).read_text(encoding='utf-8'))['code_hash'] == REGISTERED
    assert code_hash() != REGISTERED          # so the registered freezes no longer unlock test reads under this code


def test_merged_e003_equals_the_p8_variant():
    rng = np.random.default_rng(4)
    samples = []
    for i in range(12):
        ins = {'p': rng.random((32, 32)), 'k': rng.random((32, 32)), 'c': np.full((32, 32), 300.0)}
        samples.append(Sample(f's{i}', 'train' if i < 8 else 'val', f'd{i}', ins, ins['p'] * 2 + rng.normal(scale=1e-3, size=(32, 32))))
    samples.append(Sample('dup', 'val', 'dd', {k: v + 1e-5 * rng.normal(size=v.shape) * (v.std() > 0) for k, v in samples[0].inputs.items()},
                          samples[0].label + 1e-5 * rng.normal(size=(32, 32))))
    ds = ListDataset(samples, test_splits=())
    a, b = e003_duplicates(ds, Options(split='all')), e003_variant(ds, Options(split='all'))
    assert a.metrics == b.metrics and a.status == b.status == 'flagged'
