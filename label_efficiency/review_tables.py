#!/usr/bin/env python3
"""Turn the review_eval.py outputs into the two tables the reviewers asked for.

    python3 review_tables.py results/review/*.json --out results/review

Produces

  table_sessions.tex   held-out results per individual session, per head,
                       at each head's deployed operating point.
  table_matched.tex    the heads compared at matched false-detection rates
                       instead of at their default thresholds.

Protocol for the matched table, which is the part that is easy to get wrong:
the operating point is chosen on the VALIDATION split -- the threshold whose
validation false-detection rate is closest to the target -- and the numbers
reported are the HELD-OUT ones at that threshold. Choosing the threshold on
the held-out split would report the best of 19 tries as if it were a single
measurement.

Spread is the sample standard deviation over seeds (ddof = 1), the same
convention as the label-efficiency table in the paper. The runs read here are
fresh retrains, so their rows will not reproduce that table digit for digit;
they belong in a table of their own and the caption should say so.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

METRICS = ('recall', 'pck10', 'pck20', 'mean_px', 'median_px', 'fp_per_image')


def agg(rows, key):
    """Mean and sample SD of one metric over seeds."""
    v = [r[key] for r in rows if r.get(key) is not None]
    if not v:
        return float('nan'), float('nan')
    return float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0)


def at(rows, thr):
    return next((r for r in rows if abs(r['threshold'] - thr) < 1e-9), None)


def nearest_fp(rows, target):
    """The row whose false-detection rate is closest to the target.

    A head cannot always reach a given rate: if even its loosest threshold
    produces fewer false detections than the target, the closest row is still
    far from it. Reporting that row as a matched comparison would be the same
    error the matching is meant to remove, so the caller checks how close the
    achieved rate actually is -- see TOL below."""
    return min(rows, key=lambda r: abs(r['fp_per_image'] - target))


# How far the achieved false-detection rate may sit from the target before the
# row stops counting as matched: 15 % of the target, or 0.02 per image,
# whichever is larger.
def within_tolerance(achieved, target):
    return abs(achieved - target) <= max(0.15 * target, 0.02)


def fmt(m, s, nd=3):
    if np.isnan(m):
        return '--'
    return f'{m:.{nd}f} $\\pm$ {s:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--out', default='.')
    ap.add_argument('--targets', default='',
                    help='comma-separated false-detections-per-image targets. '
                         'Default: each head\'s own deployed validation rate, '
                         'so the comparison is run in both directions.')
    a = ap.parse_args()

    paths = [p for f in a.files for p in sorted(glob.glob(f))]
    runs = []
    for p in paths:
        d = json.load(open(p))
        d['_file'] = os.path.basename(p)
        runs.append(d)
    if not runs:
        raise SystemExit('no review_eval.py json found')

    by_head = defaultdict(list)
    for d in runs:
        by_head[d['head']].append(d)
    print('runs read:')
    for h, g in sorted(by_head.items()):
        print(f'  {h:<11} {len(g)} seed(s): '
              f'{", ".join(sorted(x["_file"] for x in g))}')
        if len(g) < 3:
            print(f'    note: fewer than 3 seeds; the spread for {h} is '
                  f'not comparable with the paper\'s three-seed spreads.')
    os.makedirs(a.out, exist_ok=True)

    # ---------------------------------------------------------- per session
    sessions = sorted({s for d in runs for s in d['per_session']})
    if 'unassigned' in sessions:
        print('\nWARNING: an "unassigned" group is present, meaning some '
              'frames matched no session name. Fix session_of in '
              'review_eval.py before reporting this table.')

    print('\nheld-out results per session, at each head\'s deployed '
          'threshold (mean +- SD over seeds)')
    lines = []
    for head in sorted(by_head):
        g = by_head[head]
        thr = g[0]['deployed_threshold']
        for s in sessions:
            rows = [at(d['per_session'][s], thr) for d in g
                    if s in d['per_session']]
            rows = [r for r in rows if r]
            if not rows:
                continue
            n = rows[0]['n_images']
            rc, rcs = agg(rows, 'recall')
            pk, pks = agg(rows, 'pck10')
            p2, p2s = agg(rows, 'pck20')
            mp, mps = agg(rows, 'mean_px')
            print(f'  {head:<11} {s:<20} n={n:>4}  recall {rc:.3f}+-{rcs:.3f}'
                  f'  PCK@10 {pk:.3f}+-{pks:.3f}  mean {mp:.2f}+-{mps:.2f} px')
            # TeX needs the underscores in the session names escaped. Built
            # outside the f-string: a backslash in an f-string expression is a
            # syntax error before Python 3.12.
            esc = s.replace('_', '\\_')
            eol = ' \\\\'
            lines.append(f'{head} & {esc} & {n} & '
                         f'{fmt(rc, rcs)} & {fmt(pk, pks)} & '
                         f'{fmt(p2, p2s)} & {fmt(mp, mps, 2)}' + eol)

    tex = ['% generated by review_tables.py -- do not edit by hand',
           '\\begin{tabularx}{\\linewidth}{llcCCCC}', '\\toprule',
           '\\textbf{Head} & \\textbf{Session} & \\textbf{Frames} & '
           '\\textbf{Recall} & \\textbf{PCK@10} & \\textbf{PCK@20} & '
           '\\textbf{Mean error (px)} \\\\', '\\midrule',
           *lines, '\\bottomrule', '\\end{tabularx}']
    open(os.path.join(a.out, 'table_sessions.tex'), 'w').write(
        '\n'.join(tex) + '\n')

    # -------------------------------------------------------- matched rates
    if a.targets:
        targets = [float(t) for t in a.targets.split(',')]
    else:
        targets = []
        for head in sorted(by_head):
            g = by_head[head]
            thr = g[0]['deployed_threshold']
            v = [at(d['sweep']['val'], thr)['fp_per_image'] for d in g]
            targets.append(round(float(np.mean(v)), 3))
        targets = sorted(set(targets))

    print(f'\nmatched false-detection comparison. Targets (per image): '
          f'{targets}')
    print('the threshold is chosen on validation, the numbers are held-out')
    mlines, unmatched = [], []
    for target in targets:
        for head in sorted(by_head):
            g = by_head[head]
            chosen, trs = [], []
            for d in g:
                vrow = nearest_fp(d['sweep']['val'], target)
                trow = at(d['sweep']['test'], vrow['threshold'])
                chosen.append(trow)
                trs.append(vrow['threshold'])
            vfp, _ = agg([nearest_fp(d['sweep']['val'], target) for d in g],
                         'fp_per_image')
            fp, fps = agg(chosen, 'fp_per_image')
            rc, rcs = agg(chosen, 'recall')
            pk, pks = agg(chosen, 'pck10')
            p2, p2s = agg(chosen, 'pck20')
            mp, mps = agg(chosen, 'mean_px')
            tr = '/'.join(f'{t:g}' for t in trs)
            ok = within_tolerance(vfp, target)
            if not ok:
                unmatched.append((target, head, vfp))
            flag = '' if ok else '  <- NOT MATCHED'
            print(f'  target {target:.3f}  {head:<11} thr {tr:<16} '
                  f'val fp/img {vfp:.3f}  test fp/img {fp:.3f}  '
                  f'recall {rc:.3f}+-{rcs:.3f}  '
                  f'PCK@10 {pk:.3f}+-{pks:.3f}{flag}')
            mark = '' if ok else '$^{\\dagger}$'
            mlines.append(f'{target:.2f} & {head}{mark} & {tr} & {vfp:.3f} & '
                          f'{fmt(fp, fps)} & {fmt(rc, rcs)} & '
                          f'{fmt(pk, pks)} & {fmt(p2, p2s)} & '
                          f'{fmt(mp, mps, 2)}' + ' \\\\')
        mlines.append('\\midrule')
    if mlines and mlines[-1] == '\\midrule':
        mlines.pop()

    if unmatched:
        print('\nWARNING: the rows marked NOT MATCHED did not reach their '
              'target rate. The head cannot produce that many false '
              'detections at any threshold on the grid, so that pair is not '
              'a matched comparison and must not be presented as one -- '
              'report it as the head\'s reachable range instead, or drop '
              'that target. Rows carrying a dagger in the LaTeX table:')
        for t, h, v in unmatched:
            print(f'  target {t:.3f}  {h}  reached only {v:.3f}')

    tex = ['% generated by review_tables.py -- do not edit by hand',
           # four l + five C = the nine columns of the header row below
           '\\begin{tabularx}{\\linewidth}{llllCCCCC}', '\\toprule',
           '\\textbf{Target FD/img} & \\textbf{Head} & '
           '\\textbf{Threshold} & \\textbf{Val.\\ FD/img} & '
           '\\textbf{FD/img} & \\textbf{Recall} & \\textbf{PCK@10} & '
           '\\textbf{PCK@20} & \\textbf{Mean error (px)} \\\\', '\\midrule',
           *mlines, '\\bottomrule', '\\end{tabularx}']
    if unmatched:
        tex.append('% dagger: this head did not reach the target rate at any')
        tex.append('% threshold on the grid -- not a matched comparison.')
    open(os.path.join(a.out, 'table_matched.tex'), 'w').write(
        '\n'.join(tex) + '\n')

    summary = dict(targets=targets, sessions=sessions,
                   files=[d['_file'] for d in runs])
    with open(os.path.join(a.out, 'review_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)

    print(f'\nwrote {a.out}/table_sessions.tex, {a.out}/table_matched.tex '
          f'and {a.out}/review_summary.json')


if __name__ == '__main__':
    main()
