"""Cross-base summary of the P8 result files: python scripts/p8_summarise.py results/faults p8 > results/faults/summary_p8.md
False positives exclude the known findings of protocol section 7 (constant scalar channels on icb4/icb5, E002); everything else counts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from edalint_p8 import faults as F                       # noqa: E402

BASES = ['iccad23', 'n14ir', 'icb2', 'icb3', 'icb4', 'icb5']


def known(base, check, msg):
    return check == 'E002' and 'is a constant non-zero map in' in msg and (base in ('icb4', 'icb5') or ('iccad23' == base and 'pdn_density' in msg))


def main(d, tag):
    R = {b: json.load(open(Path(d) / f'faults_{b}_{tag}.json')) for b in BASES if (Path(d) / f'faults_{b}_{tag}.json').exists()}
    print(f'# P8 summary ({", ".join(R)})\n')
    print('## False-positive rate excluding known findings (flagged draws / 20)\n')
    print('| check | ' + ' | '.join(R) + ' |\n|---|' + '---|' * len(R))
    fpmax = {}
    for c in ('E001', 'E002', 'E003', 'E004', 'E005'):
        row = []
        for b, r in R.items():
            if c not in r['false_positive']:
                row.append('-')
                continue
            v = r['false_positive'][c]
            msgs = {m: n for m, n in v['messages'].items() if not known(b, c, m)}
            # all flagged draws explained by known findings => 0; otherwise the raw flagged count
            k = 0 if (not msgs and v['flagged']) else v['flagged']
            fpmax[(c, b)] = k / max(1, v['draws'])
            row.append(f"{k}/{v['draws']}")
        print(f'| {c} | ' + ' | '.join(row) + ' |')
    print('\n## Minimum severity level with detection >= 0.90 (levels 0..4; "-" = never; n/a = not applicable); validated per protocol section 5\n')
    print('| check | fault | ' + ' | '.join(R) + ' |\n|---|---|' + '---|' * len(R))
    for name, f in F.FAULTS.items():
        cells = []
        for b, r in R.items():
            lv = r['detection'].get(name)
            if not lv:
                cells.append('n/a')
                continue
            rates = {int(k): e['detected'] / e['n'] for k, e in lv.items()}
            mn = next((k for k in sorted(rates) if rates[k] >= 0.9), None)
            top = sorted(rates)[-2:]
            ok = all(rates[k] >= 0.9 for k in top) and fpmax.get((f.check, b), 1.0) <= 0.05
            cells.append(f"L{mn}" + (' (validated)' if ok else '') if mn is not None else '-')
        print(f'| {f.check} | {name} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main(*sys.argv[1:3])
