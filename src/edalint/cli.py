"""edalint command line.

  edalint lint <dataset> [--split train] [--checks E001,E002] [--out DIR --tag TAG]
  edalint baselines <dataset> --fit train --eval <split> [--baselines zero,ridge] [--out DIR --tag TAG]
  edalint freeze-status | code-hash

Datasets: iccad23, n14-ir_drop, n14-drc, n14-congestion. Evaluating (or linting) a TEST split is refused unless
protocol/FROZEN.json exists and its code_hash equals this code's hash. Results written with --out are run-once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .checks import Options
from .core import ProtocolError
from .protocol import code_hash, frozen_info, protocol_dir
from .runner import ResultExists, parse_codes, run_checks, write_once, write_text_once


def _dataset(args):
    from .datasets import open_dataset
    kw = {}
    if args.data_root:
        kw['root'] = args.data_root
    if args.protocol_dir:
        kw['protocol_dir'] = args.protocol_dir
    name = args.dataset.lower()
    if name.startswith('n14'):
        kw.update(mode=args.mode, cache_dir=args.cache_dir, max_per_design=args.max_per_design,
                  designs=args.designs.split(',') if args.designs else None)
    return open_dataset(name, **kw)


def _common(p):
    p.add_argument('dataset')
    p.add_argument('--data-root')
    p.add_argument('--protocol-dir', help='directory holding FROZEN.json (default: <repo>/protocol or $EDALINT_PROTOCOL_DIR)')
    p.add_argument('--mode', default='authors', choices=['authors', 'raw'], help='N14 only')
    p.add_argument('--cache-dir', help='N14: stage maps on disk here instead of RAM')
    p.add_argument('--max-per-design', type=int, help='N14: evenly spaced subsample per design (RAM guard)')
    p.add_argument('--designs', help='N14: comma list of designs')
    p.add_argument('--out', help='write <dataset>_<tag>.json/.md here (run-once: refuses to overwrite)')
    p.add_argument('--tag', default='run')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='edalint', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', action='version', version=f'edalint {__version__}')
    sub = ap.add_subparsers(dest='cmd', required=True)
    lp = sub.add_parser('lint', help='run checks E001-E006 (E007 with --checks E007 --eval)')
    _common(lp)
    lp.add_argument('--split', default='train')
    lp.add_argument('--checks')
    lp.add_argument('--max-samples', type=int, default=400)
    lp.add_argument('--run-gap', action='store_true', help='E004: compute the random-vs-design split gap (protocol decision)')
    lp.add_argument('--fit', default='train', help='E007 fit split')
    lp.add_argument('--eval', help='E007 eval split')
    bp = sub.add_parser('baselines', help='fit baselines on --fit, score them on --eval')
    _common(bp)
    bp.add_argument('--fit', default='train')
    bp.add_argument('--eval', required=True)
    bp.add_argument('--baselines')
    bp.add_argument('--max-fit', type=int)
    bp.add_argument('--max-eval', type=int)
    sub.add_parser('freeze-status')
    sub.add_parser('code-hash')
    a = ap.parse_args(argv)

    if a.cmd == 'code-hash':
        print(code_hash())
        return 0
    if a.cmd == 'freeze-status':
        info = frozen_info()
        print(f'protocol dir: {protocol_dir()}')
        print('FROZEN.json: ' + ('absent (no test split can be read)' if info is None else
              f"present, code_hash {'matches' if info.get('code_hash') == code_hash() else 'DIFFERS from'} this code"))
        return 0
    try:
        ds = _dataset(a)
        if a.cmd == 'lint':
            opt = Options(split=a.split, max_samples=a.max_samples, run_gap=a.run_gap, fit_split=a.fit, eval_split=a.eval)
            rep = run_checks(ds, parse_codes(a.checks), opt)
            print(rep.text())
            if a.out:
                base = Path(a.out) / f'{ds.name}_{a.split}_{a.tag}'
                payload = rep.to_dict()
                payload['loader'] = {'skipped': getattr(ds, 'skipped', [])}
                write_once(str(base) + '.json', payload)
                write_text_once(str(base) + '.md', rep.markdown())
                print(f'wrote {base}.json / .md')
            return rep.exit_code
        from .baselines import run_zoo
        names = a.baselines.split(',') if a.baselines else None
        out = run_zoo(ds, a.fit, a.eval, names, a.max_fit, a.max_eval)
        if a.out:
            base = Path(a.out) / f'{ds.name}_baselines_{a.fit}_to_{a.eval}_{a.tag}'
            write_once(str(base) + '.json', {'dataset': ds.name, 'fit': a.fit, 'eval': a.eval,
                                             'label': 'development (train only)' if not ds.is_test_split(a.eval) else 'POST-FREEZE',
                                             'results': out, 'loader': {'skipped': getattr(ds, 'skipped', [])}})
            print(f'wrote {base}.json')
        return 0
    except ProtocolError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ResultExists as exc:
        print(str(exc), file=sys.stderr)
        return 4


if __name__ == '__main__':
    sys.exit(main())
