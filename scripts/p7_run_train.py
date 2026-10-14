"""P7 registered training run (one kernel = one task, arm and seed). FIRST ACTION: kernel_start_check (both code hashes against
protocol/FROZEN_p7.json in the bundle); it refuses before the freeze. Reads exported train shards only.

  python scripts/p7_run_train.py --task congestion --shards DIR --iters 50000 --seed 20261101 --arm A --precision fp32 --device cuda --out results/p7
Writes <out>/train_<task>_<arm>_s<seed>.json (run-once) and <out>/ckpt_<task>_<arm>_s<seed>.pt.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from edalint.core import ProtocolError                                      # noqa: E402
from edalint.runner import write_once                                       # noqa: E402
from edalint_p7.data import ShardSource, SubsetSource, check_train_ids, train_names   # noqa: E402
from edalint_p7.guard import kernel_start_check, p7_code_hash               # noqa: E402
from edalint_p7.score import load_eval_ids                                  # noqa: E402
from edalint_p7.train import train                                          # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['ir_drop', 'congestion'])
    ap.add_argument('--shards', required=True)
    ap.add_argument('--iters', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--arm', default='A', choices=['A', 'B'])
    ap.add_argument('--precision', default='fp32', choices=['fp32', 'fp16'])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--sampler', default='epoch', choices=['epoch', 'block100'])
    ap.add_argument('--bundle', default=str(REPO))
    ap.add_argument('--out', default=str(REPO / 'results' / 'p7'))
    ap.add_argument('--n14-dir', default=None, help='dir with models_<task>.json (eval_ids, fit_ids); default results/n14 of the bundle')
    a = ap.parse_args(argv)
    try:
        info = kernel_start_check(a.bundle)
    except ProtocolError as exc:
        print(exc)
        return 2
    n14 = a.n14_dir or str(Path(a.bundle) / 'results' / 'n14')
    src = ShardSource(a.shards)
    ids = src.ids
    check_train_ids(ids, load_eval_ids(a.task, n14), train_names(), [s['id'] for s in src.manifest['loader_skipped'] if 'id' in s])
    if a.arm == 'B':
        have = set(src.ids)
        ids = [i for i in json.loads((Path(n14) / f'models_{a.task}.json').read_text(encoding='utf-8'))['fit_ids'] if i in have]   # excluded non-finite rows drop out
        src = SubsetSource(src, ids)
    out = Path(a.out)
    tag = f'{a.task}_{a.arm}_s{a.seed}'
    rec = train(a.task, src, a.iters, a.seed, a.precision, a.device, a.sampler, ckpt_path=str(out / f'ckpt_{tag}.pt'))
    rec.pop('_model')
    rec.update({'label': 'REGISTERED TRAINING (post-freeze)', 'arm': a.arm, 'p7_code_hash': p7_code_hash(a.bundle), 'frozen_commit': info.get('frozen_commit')})
    write_once(out / f'train_{tag}.json', rec, dataset='p7')
    return 0


if __name__ == '__main__':
    sys.exit(main())
