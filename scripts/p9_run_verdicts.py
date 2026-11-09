"""P9 verdicts (protocol_p9.md section 6): reads results/p9/*.json, writes results/p9/verdicts_p9.json and tables_p9.md (run-once)."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from edalint.runner import write_once, write_text_once      # noqa: E402
from edalint_p9 import verdicts as V                         # noqa: E402
from edalint_p9.guard import check_p9_freeze, p9_code_hash   # noqa: E402

R = REPO / 'results' / 'p9'


def load(name):
    p = R / f'{name}.json'
    return json.loads(p.read_text(encoding='utf-8'))['result'] if p.is_file() else None


def main() -> int:
    check_p9_freeze(what='P9 verdicts')
    out = {'label': 'REGISTERED VERDICTS (mechanical)', 'p9_code_hash': p9_code_hash(), 'missing': []}
    lines = ['# P9 verdicts (mechanical, from results/p9/*.json)', '']
    for t in ('ir_drop', 'congestion', 'drc'):
        r = load(f'leakage_n14_{t}')
        if r is None:
            out['missing'].append(f'leakage_n14_{t}')
            continue
        rows = {d: sum(1 for x in r['L001']['rows'] if x['id'].startswith(d + '/')) for d in r['query_designs']}
        out[f'LK1_n14_{t}'] = V.lk1(r, r['n_queries'])
        out[f'LK2_n14_{t}'] = V.lk2(r, rows)
        lines.append(f"- LK1 N14 {t}: {out[f'LK1_n14_{t}']['verdict']} ({out[f'LK1_n14_{t}']['n_confirmed_rows']} confirmed of {r['n_queries']} test rows, "
                     f"{out[f'LK1_n14_{t}']['n_flagged']} flagged)")
        for d, v in out[f'LK2_n14_{t}'].items():
            lines.append(f"- LK2 N14 {t} {d}: proximity material {v['proximity_material']}, severity {v['severity']} "
                         f"(S_same {v['S_same']}, S_other {v['S_other']:.4f}, q25 {v['q25_reference']:.4f}, share {v['share_of_test_rows']:.3f})")
    r = load('leakage_iccad23')
    if r is None:
        out['missing'].append('leakage_iccad23')
    else:
        out['LK1_iccad23'] = V.lk1(r, r['n_queries'])
        lines.append(f"- LK1 ICCAD'23: {out['LK1_iccad23']['verdict']} ({out['LK1_iccad23']['n_flagged']} flagged)")
    r = load('leakage_splits')
    if r is None:
        out['missing'].append('leakage_splits')
    else:
        out['LK3'] = V.lk3(r)
        for k, v in out['LK3'].items():
            lines.append(f"- LK3 {k}: {v['verdict']} {v['overlaps_with_test']}")
    lines += ['', '| target | model | U1 abs/norm | U2 | U3 | U4 | T_id cov | per-design coverage C_id / C_xd |', '|---|---|---|---|---|---|---|---|']
    for target, keys in ((f'n14_{t}', ['zero-riscy', 'Vortex-large', 'openc910-1']) for t in ('ir_drop', 'congestion')):
        for m in ('ridge', 'gbm'):
            _uq_row(out, lines, f'uq_{target}_{m}', keys, target, m)
    for m in ('ridge', 'gbm'):
        _uq_row(out, lines, f'uq_iccad23_{m}', ['hidden'], 'iccad23', m)
    write_once(R / 'verdicts_p9.json', out, dataset='p9')
    write_text_once(R / 'tables_p9.md', '\n'.join(lines) + '\n')
    return 0


def _uq_row(out, lines, name, keys, target, m):
    r = load(name)
    if r is None:
        out['missing'].append(name)
        return
    v = V.uq(r, keys)
    out[name] = v
    per = '; '.join(f"{d} {v['per_design'][d]['id']:.3f}/{v['per_design'][d]['xd']:.3f}" for d in keys)
    lines.append(f"| {target} | {m} | {v['U1_id_validity']['crc_abs']}/{v['U1_id_validity']['crc_norm']} | {v['U2_ood_undercoverage']['verdict']} | "
                 f"{v['U3_xd_repairs']['verdict']} | {v['U4_raw_ensemble_undercovers']['verdict']} | {v['T_id']['crc_norm_id']:.3f} | {per} |")


if __name__ == '__main__':
    sys.exit(main())
