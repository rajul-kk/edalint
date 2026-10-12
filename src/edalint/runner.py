"""Run checks over a dataset; build the report (text, Markdown, JSON); run-once results writer.

The writer is modelled on fieldlint's run_v3.py: a result file that exists is NEVER overwritten (open with mode 'x'); every file
records the git HEAD, the edalint code hash, library versions, the command line and (when present) the frozen-protocol hash.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .checks import CHECKS, NAMES, Options
from .core import CheckResult, MapDataset, ProtocolError, skipped
from .protocol import frozen_info, provenance

SEVERITY_RANK = {'none': -1, 'info': 0, 'warning': 1, 'error': 2}
EXIT_CODES = {'none': 0, 'info': 0, 'warning': 1, 'error': 2}


class ResultExists(FileExistsError):
    """A run-once result file already exists."""


def _jsonable(o: Any):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, Path):
        return str(o)
    return o


def write_once(path, payload: Dict[str, Any], with_provenance: bool = True) -> Path:
    """Write JSON to `path`; raise ResultExists if it exists (no overwrite, no append). Adds provenance (git HEAD, code hash,
    versions, argv, frozen protocol if any)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    if with_provenance:
        body['provenance'] = provenance()
        fz = frozen_info()
        body['provenance']['frozen_protocol'] = fz if fz is not None else None
    try:
        with open(p, 'x', encoding='utf-8') as fh:
            json.dump(_jsonable(body), fh, indent=1)
    except FileExistsError as exc:
        raise ResultExists(f'{p} already exists; result files are run-once and are never overwritten') from exc
    return p


def write_text_once(path, text: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(p, 'x', encoding='utf-8') as fh:
            fh.write(text)
    except FileExistsError as exc:
        raise ResultExists(f'{p} already exists; result files are run-once and are never overwritten') from exc
    return p


@dataclass
class Report:
    dataset: str
    split: str
    n_samples: int
    results: List[CheckResult] = field(default_factory=list)
    seconds: float = 0.0
    label: str = 'development (train only)'

    @property
    def worst(self) -> str:
        sev = [f.severity for r in self.results for f in r.findings]
        return max(sev, key=SEVERITY_RANK.get) if sev else 'none'

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.worst]

    def to_dict(self):
        return {'tool': 'edalint', 'dataset': self.dataset, 'split': self.split, 'label': self.label, 'n_samples': self.n_samples,
                'worst_severity': self.worst, 'exit_code': self.exit_code, 'seconds': round(self.seconds, 2),
                'checks': [r.to_dict() for r in self.results]}

    def text(self) -> str:
        out = [f'edalint: {self.dataset} / {self.split} ({self.n_samples} samples) [{self.label}]']
        for r in self.results:
            tag = {'ok': 'ok     ', 'flagged': 'FLAGGED', 'skipped': 'skipped'}[r.status]
            out.append(f'  {r.code} {r.name:<16} {tag}  {r.summary}')
            for f in r.findings:
                ids = f'  [e.g. {", ".join(f.samples[:3])}]' if f.samples else ''
                out.append(f'      {f.severity.upper():<7} {f.message}{ids}')
        c = {s: sum(f.severity == s for r in self.results for f in r.findings) for s in ('error', 'warning', 'info')}
        out.append(f'  result: {c["error"]} error(s), {c["warning"]} warning(s), {c["info"]} info; exit code {self.exit_code}')
        return '\n'.join(out)

    def markdown(self) -> str:
        out = [f'# edalint report: {self.dataset} / {self.split}', '',
               f'*{self.label}.* {self.n_samples} samples, {self.seconds:.1f} s, worst severity **{self.worst}**.', '',
               '| check | status | summary |', '|---|---|---|']
        for r in self.results:
            out.append(f'| {r.code} {r.name} | {r.status} | {r.summary.replace("|", "/")} |')
        for r in self.results:
            out += ['', f'## {r.code} {r.name}', '', f'Status: **{r.status}**' + (f' ({r.skip_reason})' if r.skip_reason else '') + f'. {r.summary}']
            for f in r.findings:
                ex = f' (e.g. `{"`, `".join(f.samples[:3])}`)' if f.samples else ''
                out.append(f'- **{f.severity}**: {f.message}{ex}')
            if r.metrics:
                out += ['', '```json', json.dumps(_jsonable(r.metrics), indent=1)[:6000], '```']
        return '\n'.join(out) + '\n'


def parse_codes(spec: Optional[str]) -> List[str]:
    if not spec:
        return [c for c in CHECKS if c != 'E007']
    codes = [c.strip().upper() for c in spec.split(',') if c.strip()]
    bad = [c for c in codes if c not in CHECKS]
    if bad:
        raise ValueError(f'unknown check(s) {bad}; available {sorted(CHECKS)}')
    return codes


def run_checks(ds: MapDataset, codes: Optional[Iterable[str]] = None, opt: Optional[Options] = None) -> Report:
    """Run the selected checks (default E001-E006). Refuses (ProtocolError) to read a test split without FROZEN.json.
    E001 runs first; a check that crashes is reported as an error finding instead of hiding the others."""
    opt = opt or Options()
    named_test = opt.split != 'all' and ds.is_test_split(opt.split)
    if named_test or (opt.eval_split and ds.is_test_split(opt.eval_split) and 'E007' in (codes or [])):
        from .protocol import require_frozen
        require_frozen(ds.protocol_dir, what=f'running checks on split {opt.split!r}')
    # split 'all' without a freeze silently means "every non-test split" (E004 reads metadata of all splits regardless)
    codes = list(codes) if codes else [c for c in CHECKS if c != 'E007']
    codes = sorted(codes)
    label = 'development (train only)' if not named_test else 'POST-FREEZE'
    rep = Report(ds.name, opt.split, len(ds.ids(opt.split)) if opt.split != 'all' else len(ds), label=label)
    t0 = time.time()
    for c in codes:
        try:
            r = CHECKS[c](ds, opt)
        except ProtocolError as exc:
            r = skipped(c, NAMES[c], f'refused: {exc}')
        except Exception as exc:                                    # noqa: BLE001 - a check bug must not hide the other checks
            r = CheckResult(c, NAMES[c], 'flagged', 'error', summary='check crashed')
            r.add('error', f'check crashed: {type(exc).__name__}: {exc}')
        rep.results.append(r)
    rep.seconds = time.time() - t0
    return rep
