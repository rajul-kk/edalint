"""P7 exporter: compressed float32 .npz shards (one design per shard) + manifest with sha256 per shard.

  python scripts/p7_run_export.py --task congestion --out E:/edalint_data/p7_export/congestion_train        (TRAIN rows, allowed any time)
  python scripts/p7_run_export.py --task congestion --out E:/edalint_data/p7_export/congestion_eval --test (post-freeze only: goes through
                                                                                                           edalint_p7.guard.open_test_rows)
Before exporting train rows it checks that they are disjoint (by id and by design) from the registered eval ids and that their count equals
the official train count less the loader-skipped rows.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from edalint.core import ProtocolError                                     # noqa: E402
from edalint_p7.data import check_train_ids, export_rows, open_train_rows, train_names   # noqa: E402
from edalint_p7.guard import open_test_rows                                # noqa: E402
from edalint_p7.score import load_eval_ids                                 # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['ir_drop', 'congestion'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache-dir', default=None, help='staging cache for the authors-space member maps (default E:/edalint_data/p7_cache/<task>)')
    ap.add_argument('--root', default=None)
    ap.add_argument('--shard-rows', type=int, default=128)
    ap.add_argument('--test', action='store_true', help='export the N14 evaluation rows (needs the P7 freeze)')
    a = ap.parse_args(argv)
    cache = a.cache_dir or f'E:/edalint_data/p7_cache/{a.task}'
    if a.test:
        try:
            tr = open_test_rows(a.task, cache_dir=cache + '_eval', root=a.root)
        except ProtocolError as exc:
            print(exc)
            return 2
        ids = [i for v in tr.ids_by_design.values() for i in v]
        man = export_rows(a.task, tr.ds, ids, a.out, a.shard_rows, label='POST-FREEZE evaluation rows')
    else:
        ds = open_train_rows(a.task, cache_dir=cache, root=a.root)
        ids = ds.ids('train')
        skipped = [s['id'] for s in ds.skipped if 'id' in s]
        report = check_train_ids(ids, load_eval_ids(a.task), train_names(), skipped)
        print(json.dumps(report))
        man = export_rows(a.task, ds, ids, a.out, a.shard_rows, label='train only')
        man['train_id_report'] = report
        (Path(a.out) / 'manifest.json').write_text(json.dumps(man, indent=1), encoding='utf-8')
    print(f'exported {man["n_rows"]} rows in {len(man["shards"])} shards to {a.out}; non-finite rows: {len(man["nonfinite_ids"])}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
