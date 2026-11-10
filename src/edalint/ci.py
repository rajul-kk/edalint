"""CI outputs of a lint report: SARIF 2.1.0 (GitHub code scanning), JUnit XML, GitHub Actions annotations, and the baseline of
accepted findings.

A finding's fingerprint is sha256 over (dataset name, check code, severity, message). The message states counts and sample ids, so a
finding whose count or affected samples change is a new finding and is not hidden by an old baseline.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from xml.sax.saxutils import escape, quoteattr

from . import __version__
from .checks import CHECKS, NAMES
from .core import Finding
from .runner import Report

URI = 'https://github.com/rajul-kk/edalint'
SARIF_LEVEL = {'error': 'error', 'warning': 'warning', 'info': 'note'}


def fingerprint(dataset: str, f: Finding) -> str:
    return hashlib.sha256('\0'.join((dataset, f.code, f.severity, f.message)).encode()).hexdigest()[:32]


# -- baseline -----------------------------------------------------------------------------------------------------------------------
def write_baseline(rep: Report, path) -> Path:
    items = sorted({(fingerprint(rep.dataset, f), f.code, f.severity, f.message) for r in rep.results for f in r.findings})
    body = {'tool': 'edalint', 'version': __version__, 'dataset': rep.dataset,
            'note': 'Accepted findings: edalint lint --baseline FILE does not report these again. Delete an entry to see it again.',
            'findings': [{'fingerprint': h, 'code': c, 'severity': s, 'message': m} for h, c, s, m in items]}
    p = Path(path)
    p.write_text(json.dumps(body, indent=1) + '\n', encoding='utf-8')
    return p


def apply_baseline(rep: Report, path) -> int:
    """Drop the findings listed in the baseline file from `rep` (statuses recomputed). Returns how many were dropped."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    known = {x['fingerprint'] for x in data.get('findings', [])}
    dropped = 0
    for r in rep.results:
        keep = [f for f in r.findings if fingerprint(rep.dataset, f) not in known]
        dropped += len(r.findings) - len(keep)
        r.findings = keep
        r.finalise()
    return dropped


# -- locations ----------------------------------------------------------------------------------------------------------------------
def _uri(p: str) -> str:
    try:
        rel = os.path.relpath(p)
    except ValueError:                                     # another drive on Windows
        rel = p
    return Path(rel).as_posix()


def finding_paths(ds, f: Finding) -> List[str]:
    """Files of the samples a finding names (folder datasets know them; built-in benchmarks do not)."""
    out = []
    for sid in f.samples:
        sid = str(sid).split(' (')[0]                       # E001 lists 'id (error text)'
        p = ds.meta(sid).get('path') if ds is not None else None
        if p:
            out.append(_uri(p))
    return out


def _check_doc(code: str) -> str:
    doc = (CHECKS[code].__doc__ or '').strip().splitlines()
    return doc[0] if doc else code


# -- SARIF --------------------------------------------------------------------------------------------------------------------------
def sarif(rep: Report, ds=None, fallback_uri: Optional[str] = None) -> Dict:
    """SARIF 2.1.0. Each result points at the first affected sample file, else at `fallback_uri` (the config file or the dataset's
    samples.csv), else it has no location."""
    codes = sorted(CHECKS)
    rules = [{'id': c, 'name': NAMES[c], 'shortDescription': {'text': _check_doc(c)}, 'helpUri': f'{URI}#checks'} for c in codes]
    results = []
    for r in rep.results:
        for f in r.findings:
            paths = finding_paths(ds, f)
            res = {'ruleId': f.code, 'ruleIndex': codes.index(f.code), 'level': SARIF_LEVEL[f.severity],
                   'message': {'text': f.message + (f' (e.g. {", ".join(map(str, f.samples[:3]))})' if f.samples else '')},
                   'partialFingerprints': {'edalintFinding/v1': fingerprint(rep.dataset, f)}}
            where = paths[:1] or ([fallback_uri] if fallback_uri else [])
            if where:
                res['locations'] = [{'physicalLocation': {'artifactLocation': {'uri': where[0]}}}]
            if len(paths) > 1:
                res['relatedLocations'] = [{'id': i, 'physicalLocation': {'artifactLocation': {'uri': p}}}
                                           for i, p in enumerate(paths[1:10], 1)]
            results.append(res)
    return {'$schema': 'https://json.schemastore.org/sarif-2.1.0.json', 'version': '2.1.0',
            'runs': [{'tool': {'driver': {'name': 'edalint', 'version': __version__, 'informationUri': URI, 'rules': rules}},
                      'automationDetails': {'id': f'edalint/{rep.dataset}/{rep.split}/'},
                      'results': results}]}


# -- JUnit --------------------------------------------------------------------------------------------------------------------------
def junit(rep: Report, fail_on: str = 'warning') -> str:
    """One <testcase> per check. A check fails if it has a finding at or above `fail_on`; skipped checks are <skipped/>."""
    from .config import fails
    cases, n_fail, n_skip = [], 0, 0
    for r in rep.results:
        name = quoteattr(f'{r.code} {r.name}')
        body = ''
        if r.status == 'skipped':
            n_skip += 1
            body = f'<skipped message={quoteattr(r.skip_reason or r.summary)}/>'
        else:
            bad = [f for f in r.findings if fails(f.severity, fail_on)]
            if bad:
                n_fail += 1
                text = '\n'.join(f'{f.severity.upper()}: {f.message}' for f in bad)
                body = f'<failure message={quoteattr(bad[0].message[:300])} type={quoteattr(bad[0].severity)}>{escape(text)}</failure>'
            other = [f for f in r.findings if f not in bad]
            if other:
                body += '<system-out>' + escape('\n'.join(f'{f.severity.upper()}: {f.message}' for f in other)) + '</system-out>'
        cases.append(f'  <testcase classname={quoteattr("edalint." + rep.dataset)} name={name} time="0">{body}</testcase>')
    head = (f'<testsuite name={quoteattr(f"edalint {rep.dataset}/{rep.split}")} tests="{len(rep.results)}" failures="{n_fail}" '
            f'errors="0" skipped="{n_skip}" time="{rep.seconds:.2f}">')
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + head + '\n' + '\n'.join(cases) + '\n</testsuite>\n'


# -- GitHub Actions annotations -----------------------------------------------------------------------------------------------------
def _gh_escape(s: str, prop: bool = False) -> str:
    s = s.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
    return s.replace(':', '%3A').replace(',', '%2C') if prop else s


def github(rep: Report, ds=None, fallback_uri: Optional[str] = None) -> str:
    """Workflow commands (::error / ::warning / ::notice) that GitHub shows as annotations on the run and the PR."""
    cmd = {'error': 'error', 'warning': 'warning', 'info': 'notice'}
    out = []
    for r in rep.results:
        for f in r.findings:
            where = (finding_paths(ds, f) or ([fallback_uri] if fallback_uri else []))[:1]
            props = [f'title={_gh_escape(f"edalint {f.code} {r.name}", True)}'] + [f'file={_gh_escape(w, True)}' for w in where]
            ex = f' (e.g. {", ".join(map(str, f.samples[:3]))})' if f.samples else ''
            out.append(f'::{cmd[f.severity]} {",".join(props)}::{_gh_escape(f.message + ex)}')
    return '\n'.join(out)
