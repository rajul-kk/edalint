"""Thin Kaggle entry for the registered P7 runs (NOT part of the p7 code hash; it only locates the bundle and calls the hashed scripts).

  python p7_kernel_entry.py train  -- --shards /kaggle/input/edalint-p7-train-ir --iters 30000 --seed 20261101 --arm A --precision fp16 --device cuda --out /kaggle/working
  python p7_kernel_entry.py score  -- --task ir_drop --ckpt /kaggle/working/ckpt_ir_drop_A_s20261101.pt --seed 20261101 --device cuda --out /kaggle/working

The bundle is the ONE attached dataset pinned below (built by scripts/p7_make_bundle.py: src/edalint, src/edalint_p7, scripts/p7_run_*.py,
protocol/FROZEN_p7.json, protocol/protocol_p7.md, results/n14/models_ir_drop.json). It refuses if the pinned directory is missing or if more than one
attached input looks like a P7 bundle. It then runs <bundle>/scripts/p7_run_<stage>.py with --bundle <bundle>; those scripts call
`kernel_start_check` first, which recomputes both code hashes against the FROZEN_p7.json in the bundle and refuses on any mismatch.
For `train`, `--task ir_drop` and `--bundle` are added here; for `score`, `--bundle` is added.
"""
import subprocess
import sys
from pathlib import Path

PINNED = 'edalint-p7-code-frozen'
INPUT_ROOT = Path('/kaggle/input')
STAGES = {'train': 'p7_run_train.py', 'score': 'p7_run_score.py'}


def find_bundle(input_root=INPUT_ROOT, pinned: str = PINNED) -> Path:
    """The pinned bundle directory. Raises SystemExit (message) if it is absent, or if any other attached input also carries
    protocol/FROZEN_p7.json or a scripts/p7_run_train.py (two candidates are refused, not resolved)."""
    root = Path(input_root)
    want = root / pinned
    if not (want / 'protocol' / 'FROZEN_p7.json').is_file():
        raise SystemExit(f'refused: pinned bundle {want} (with protocol/FROZEN_p7.json) is not attached')
    others = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name != pinned
              and ((d / 'protocol' / 'FROZEN_p7.json').is_file() or (d / 'scripts' / 'p7_run_train.py').is_file())]
    if others:
        raise SystemExit(f'refused: more than one P7 bundle candidate attached: {[pinned] + [d.name for d in others]}')
    return want


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in STAGES:
        print(f'usage: p7_kernel_entry.py {{{"|".join(STAGES)}}} -- <args of the p7_run script>')
        return 2
    stage, rest = argv[0], [a for a in argv[1:] if a != '--']
    bundle = find_bundle()
    cmd = [sys.executable, str(bundle / 'scripts' / STAGES[stage]), '--bundle', str(bundle)]
    if stage == 'train':
        cmd += ['--task', 'ir_drop']
    return subprocess.call(cmd + rest)


if __name__ == '__main__':
    sys.exit(main())
