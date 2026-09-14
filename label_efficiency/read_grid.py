#!/usr/bin/env python3
"""Summarise a soft-argmax tuning sweep.

Selection is on VALIDATION PCK@10 -- the held-out sessions decide nothing, so
the sweep cannot leak into the number the paper reports. The test column is
printed only so the eventual winner can be read off in one place; do not pick
by it.

    python3 read_grid.py --runs runs
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def parse(tag):
    """softargmax_f25_s0_t0.5_w20 -> (25, 0, 0.5, 20.0)"""
    parts = tag.split('_')
    frac = int(parts[1][1:])
    seed = int(parts[2][1:])
    tau, w = 1.0, 5.0
    for p in parts[3:]:
        if p.startswith('t'):
            tau = float(p[1:])
        elif p.startswith('w'):
            w = float(p[1:])
    return frac, seed, tau, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='runs')
    a = ap.parse_args()

    by = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(a.runs, 'softargmax_*.json'))):
        d = json.load(open(f))
        tag = os.path.splitext(os.path.basename(f))[0]
        if '_lr' in tag:          # learning-rate variants belong to the probe
            continue
        frac, seed, tau, w = parse(tag)
        by[(tau, w, frac)][seed] = (d['results']['val']['overall']['pck10'],
                                    d['results']['test']['overall']['pck10'])

    fracs = sorted({k[2] for k in by})
    print(f'{"tau":>5} {"w":>5} ' +
          ' '.join(f'| f{f:<3} val    test   n' for f in fracs))
    rows = []
    for tau, w in sorted({(k[0], k[1]) for k in by}):
        line = f'{tau:>5g} {w:>5g} '
        vals = []
        for f in fracs:
            runs = by.get((tau, w, f), {})
            if not runs:
                line += '|  --     --    0 '
                continue
            v = sum(r[0] for r in runs.values()) / len(runs)
            t = sum(r[1] for r in runs.values()) / len(runs)
            vals.append(v)
            line += f'| {v:6.3f} {t:6.3f} {len(runs):>2d} '
        print(line + ('   <- original' if (tau, w) == (1.0, 5.0) else ''))
        if vals:
            rows.append((sum(vals) / len(vals), tau, w))

    if rows:
        best = max(rows)
        print(f'\nbest validation PCK@10 (mean over budgets): '
              f'tau={best[1]:g}, coordinate weight={best[2]:g}')
        orig = [r for r in rows if (r[1], r[2]) == (1.0, 5.0)]
        if orig and (best[1], best[2]) != (1.0, 5.0):
            print(f'  original setting: {orig[0][0]:.3f}  ->  '
                  f'tuned: {best[0]:.3f}   '
                  f'({best[0] - orig[0][0]:+.3f} on validation)')
            print(f'\nConfirm on three seeds:\n'
                  f'  WINNER="{best[1]:g} {best[2]:g}" ./grid_softargmax.sh '
                  f'/path/to/splits.json')
        elif orig:
            print('  the original setting already wins the sweep')


if __name__ == '__main__':
    main()
