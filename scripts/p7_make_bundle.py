"""Assemble the P7 code bundle for the Kaggle dataset `edalint-p7-code-frozen` (NOT part of the p7 code hash; run only at the freeze commit).

  python scripts/p7_make_bundle.py --out E:/edalint_data/p7_bundle

Copies exactly: src/edalint/** and src/edalint_p7/** (no __pycache__), scripts/p7_run_*.py, protocol/FROZEN_p7.json, protocol/protocol_p7.md,
results/n14/models_ir_drop.json. Refuses unless the P7 freeze checks pass (both code hashes against protocol/FROZEN_p7.json), refuses a non-empty
--out, and writes <out>/MANIFEST.json with the sha256 of every file and of the p7 / edalint code hashes. It uploads nothing.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from edalint.core import ProtocolError                      # noqa: E402
from edalint.protocol import code_hash                      # noqa: E402
from edalint_p7.guard import check_p7_freeze, p7_code_hash   # noqa: E402

SINGLE = ['protocol/FROZEN_p7.json', 'protocol/protocol_p7.md', 'results/n14/models_ir_drop.json']


def bundle_files(repo: Path):
    files = []
    for tree in ('src/edalint', 'src/edalint_p7'):
        files += [p for p in (repo / tree).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    files += sorted((repo / 'scripts').glob('p7_run_*.py'))
    files += [repo / f for f in SINGLE]
    return sorted(files, key=lambda p: p.relative_to(repo).as_posix())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    try:
        check_p7_freeze(REPO / 'protocol', REPO, what='bundling P7 code')
    except ProtocolError as exc:
        print(exc)
        return 2
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        print(f'refused: {out} is not empty')
        return 2
    missing = [f for f in SINGLE if not (REPO / f).is_file()]
    if missing:
        print(f'refused: missing {missing}')
        return 2
    man = {}
    for p in bundle_files(REPO):
        rel = p.relative_to(REPO).as_posix()
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
        man[rel] = hashlib.sha256(dst.read_bytes()).hexdigest()
    (out / 'MANIFEST.json').write_text(json.dumps({'p7_code_hash': p7_code_hash(REPO), 'code_hash': code_hash(), 'n_files': len(man), 'files': man}, indent=1),
                                       encoding='utf-8')
    print(f'bundle of {len(man)} files in {out}; p7_code_hash {p7_code_hash(REPO)[:12]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
