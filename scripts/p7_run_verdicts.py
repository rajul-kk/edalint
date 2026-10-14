"""P7 registered verdicts: H-A / H-B / H-C for IR drop from the scored files results/p7/models_ir_drop_A_s<seed>.json.

  python scripts/p7_run_verdicts.py
Refuses before the P7 freeze (or on a code-hash mismatch). Reads only the scored row files and the registered N14 result files; writes
results/p7/verdicts_p7.json and results/p7/tables_p7.md (run-once).
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from edalint.core import ProtocolError                      # noqa: E402
from edalint.runner import ResultExists                     # noqa: E402
from edalint_p7.verdicts import SEEDS, run_verdicts         # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', default=str(REPO / 'results' / 'p7'))
    ap.add_argument('--n14-dir', default=str(REPO / 'results' / 'n14'))
    ap.add_argument('--bundle', default=str(REPO))
    a = ap.parse_args(argv)
    try:
        v = run_verdicts(a.models_dir, a.n14_dir, Path(a.bundle) / 'protocol', a.bundle, seeds=SEEDS)
    except (ProtocolError, ResultExists) as exc:
        print(exc)
        return 2
    print('P7 IR drop verdict:', v['verdict_label'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
