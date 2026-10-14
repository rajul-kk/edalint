"""P7 registered scoring: a trained checkpoint on ALL official N14 evaluation rows (3,544), through guard.open_test_rows only.

  python scripts/p7_run_score.py --task congestion --ckpt results/p7/ckpt_congestion_A_s20261101.pt --arm A --seed 20261101
Refuses before the P7 freeze (open_test_rows). Aborts if the scored id set differs from eval_ids of results/n14/models_<task>.json.
Writes results/p7/models_<task>_<arm>_s<seed>.json (run-once; same row format as the N14 files, plus official_roc per design).
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from edalint.core import ProtocolError                      # noqa: E402
from edalint.runner import write_once                       # noqa: E402
from edalint_p7.guard import kernel_start_check, open_test_rows, p7_code_hash   # noqa: E402
from edalint_p7.score import iter_dataset_rows, load_eval_ids, score_model       # noqa: E402
from edalint_p7.train import build_model                    # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=['ir_drop', 'congestion'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arm', default='A', choices=['A', 'B'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--bundle', default=str(REPO))
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--root', default=None)
    ap.add_argument('--out', default=str(REPO / 'results' / 'p7'))
    a = ap.parse_args(argv)
    try:
        kernel_start_check(a.bundle)
        tr = open_test_rows(a.task, cache_dir=a.cache_dir or f'E:/edalint_data/p7_cache/{a.task}_eval', root=a.root)
    except ProtocolError as exc:
        print(exc)
        return 2
    model = build_model(a.task)
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu')['state_dict'])
    model.to(a.device)
    res = score_model(a.task, model, iter_dataset_rows(a.task, tr.ds, tr.ids_by_design), load_eval_ids(a.task, Path(a.bundle) / 'results' / 'n14'), a.device)
    res.update({'label': 'REGISTERED SCORING (post-freeze)', 'task': a.task, 'arm': a.arm, 'seed': a.seed, 'checkpoint': a.ckpt,
                'p7_code_hash': p7_code_hash(a.bundle), 'eval_ids': tr.ids_by_design, 'skipped_by_loader': tr.ds.skipped})
    write_once(Path(a.out) / f'models_{a.task}_{a.arm}_s{a.seed}.json', res, dataset='p7')
    return 0


if __name__ == '__main__':
    sys.exit(main())
