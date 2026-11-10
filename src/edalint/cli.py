"""edalint command line.

  edalint lint [<dataset>] [--split S] [--checks E001,E002] [--ignore E005] [--fail-on warning] [--baseline FILE]
               [--format text|json|markdown|sarif|junit|github] [--sarif F] [--junit F] [--json F] [--markdown F] [--out DIR --tag TAG]
  edalint baselines <dataset> --fit train --eval <split> [--baselines zero,ridge] [--out DIR --tag TAG]
  edalint checks | freeze-status | code-hash

<dataset>: a folder of .npz files (your own data, see edalint.datasets.folder), or a built-in benchmark: iccad23, n14-ir_drop, n14-drc,
n14-congestion. Settings come from edalint.toml or [tool.edalint] in pyproject.toml (see edalint.config); flags override them.

Exit codes of lint: 0 nothing at or above --fail-on (default warning); 1 worst finding is a warning (or info with --fail-on info);
2 worst finding is an error; 3 refused by the frozen-protocol guard; 4 a run-once --out file exists; 5 bad configuration or dataset.
Reading a TEST split that a dataset guards needs a freeze record whose code_hash equals this code's hash. --out files are run-once
(never overwritten); --sarif / --junit / --json / --markdown files are CI artefacts and are overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, find_config, fails, options_from
from .core import ProtocolError
from .protocol import code_hash, protocol_dir
from .runner import EXIT_CODES, ResultExists, _jsonable, parse_codes, run_checks, write_once, write_text_once

BUILTIN = ('iccad23', 'n14', 'n28')


def _is_builtin(name: str) -> bool:
    n = name.lower()
    return n == 'iccad23' or n.startswith(BUILTIN[1:])


def _dataset(args):
    from .datasets import open_dataset
    if not _is_builtin(args.dataset):
        kw = dict(getattr(args, 'folder_format', None) or {})
        if args.protocol_dir:
            kw['protocol_dir'] = args.protocol_dir
        return open_dataset(args.dataset if args.dataset.startswith('folder:') else 'folder:' + args.dataset, **kw)
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
    p.add_argument('dataset', nargs='?', help='folder of .npz files, or iccad23 / n14-<task> (default: dataset in the config)')
    p.add_argument('--data-root', help='built-in benchmarks: where the raw data lives')
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
    lp.add_argument('--split', help="split(s) to read (default: config, else 'all' for a folder, 'train' for a benchmark)")
    lp.add_argument('--checks', help='comma list of checks to run (config: select)')
    lp.add_argument('--ignore', help='comma list of checks not to run')
    lp.add_argument('--fail-on', choices=['error', 'warning', 'info', 'never'], help='lowest severity that fails the run (default warning)')
    lp.add_argument('--config', help='edalint.toml or pyproject.toml to read (default: search, see edalint.config)')
    lp.add_argument('--no-config', action='store_true', help='ignore configuration files')
    lp.add_argument('--baseline', help='JSON of accepted findings to leave out (written by --write-baseline)')
    lp.add_argument('--write-baseline', metavar='FILE', help='write every finding of this run to FILE as accepted, exit 0')
    lp.add_argument('--format', default='text', choices=['text', 'json', 'markdown', 'sarif', 'junit', 'github'],
                    help='what to print on stdout')
    for f in ('sarif', 'junit', 'json', 'markdown'):
        lp.add_argument(f'--{f}', metavar='FILE', help=f'also write the report as {f} to FILE (overwritten)')
    lp.add_argument('--max-samples', type=int, help='evenly spaced subsample per split for per-sample checks (default 400)')
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
    sub.add_parser('checks', help='list the checks')
    sub.add_parser('freeze-status')
    sub.add_parser('code-hash')
    a = ap.parse_args(argv)

    if a.cmd == 'checks':
        from .checks import CHECKS, NAMES
        for c in sorted(CHECKS):
            print(f'{c} {NAMES[c]:<16} {(CHECKS[c].__doc__ or "").strip().splitlines()[0].removeprefix(c).strip()}')
        return 0
    if a.cmd == 'code-hash':
        print(code_hash())
        return 0
    if a.cmd == 'freeze-status':
        print(f'protocol dir: {protocol_dir()}')
        files = [f for f in [protocol_dir() / 'FROZEN.json'] + sorted(protocol_dir().glob('FROZEN_*.json')) if f.is_file()]
        if not files:
            print('freeze files: absent (no test split can be read)')
        for f in files:
            info = json.loads(f.read_text(encoding='utf-8'))
            print(f"{f.name}: dataset {info.get('dataset', '?')}, code_hash "
                  f"{'matches' if info.get('code_hash') == code_hash() else 'DIFFERS from'} this code")
        return 0
    try:
        if a.cmd == 'lint':
            return _lint(a)
        if not a.dataset:
            raise ConfigError('baselines needs a dataset')
        ds = _dataset(a)
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
    except (ConfigError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f'edalint: {exc}', file=sys.stderr)
        return 5


def exit_code(worst: str, fail_on: str) -> int:
    """0 if nothing reaches fail_on, else 1 for warning (or info), 2 for error."""
    return max(1, EXIT_CODES[worst]) if fails(worst, fail_on) else 0


def _lint(a) -> int:
    from . import ci
    cfg, cfg_path = ({}, None) if a.no_config else find_config(a.config, a.dataset)
    if not a.dataset:
        if not cfg.get('dataset'):
            raise ConfigError('no dataset: name one on the command line or set dataset = "..." in edalint.toml')
        d = str(cfg['dataset'])
        rel = cfg_path.parent / d if cfg_path and not _is_builtin(d) and not d.startswith('folder:') else None
        a.dataset = str(rel) if rel is not None else d
    a.folder_format = cfg.get('dataset-format')
    ds = _dataset(a)
    is_folder = hasattr(ds, 'root')
    split = a.split or cfg.get('split') or ('all' if is_folder else 'train')
    max_n = a.max_samples if a.max_samples is not None else cfg.get('max-samples', 400)
    opt = options_from(cfg, split, max_n)
    opt.run_gap = a.run_gap or opt.run_gap
    opt.fit_split, opt.eval_split = a.fit, a.eval or opt.eval_split
    select = parse_codes(a.checks or (','.join(cfg['select']) if cfg.get('select') else None))
    ignore_spec = a.ignore or (','.join(cfg['ignore']) if cfg.get('ignore') else None)
    ignore = set(parse_codes(ignore_spec)) if ignore_spec else set()
    codes = [c for c in select if c not in ignore]
    if not codes:
        raise ConfigError('every check is ignored; nothing to run')
    fail_on = a.fail_on or cfg.get('fail-on', 'warning')
    rep = run_checks(ds, codes, opt)

    if a.write_baseline:
        p = ci.write_baseline(rep, a.write_baseline)
        print(rep.text())
        print(f'wrote baseline {p} ({sum(len(r.findings) for r in rep.results)} findings accepted)')
        return 0
    baseline = a.baseline or cfg.get('baseline')
    if baseline:
        bp = Path(baseline) if a.baseline or not cfg_path else cfg_path.parent / baseline
        if bp.is_file():
            n = ci.apply_baseline(rep, bp)
            print(f'baseline {bp}: {n} accepted finding(s) not reported', file=sys.stderr)
        elif a.baseline:
            raise ConfigError(f'baseline file {bp} does not exist')

    fallback = None
    if is_folder:
        table = ds.root / 'samples.csv'
        fallback = ci._uri(str(cfg_path if cfg_path else table if table.is_file() else ds.root))
    code = exit_code(rep.worst, fail_on)

    def as_json():
        d = rep.to_dict()
        d.update(fail_on=fail_on, exit_code=code)
        return json.dumps(_jsonable(d), indent=1)

    render = {'text': rep.text, 'markdown': rep.markdown, 'json': as_json,
              'sarif': lambda: json.dumps(ci.sarif(rep, ds, fallback), indent=1),
              'junit': lambda: ci.junit(rep, fail_on), 'github': lambda: ci.github(rep, ds, fallback)}
    shown = render[a.format]()
    if shown:
        print(shown)
    for f in ('sarif', 'junit', 'json', 'markdown'):
        target = getattr(a, f)
        if target:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            text = render[f]()
            Path(target).write_text(text if text.endswith('\n') else text + '\n', encoding='utf-8')
    if a.out:
        base = Path(a.out) / f'{ds.name}_{split}_{a.tag}'
        payload = rep.to_dict()
        payload['loader'] = {'skipped': getattr(ds, 'skipped', [])}
        write_once(str(base) + '.json', payload)
        write_text_once(str(base) + '.md', rep.markdown())
        print(f'wrote {base}.json / .md')
    if a.format == 'text' and fail_on != 'warning':
        print(f'  fail-on {fail_on}: exit code {code}')
    return code


if __name__ == '__main__':
    sys.exit(main())
