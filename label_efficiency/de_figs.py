#!/usr/bin/env python3
"""Aggregate the runs into the manuscript figure and table.

Produces:
  fig_label_efficiency.pdf  (a) PCK@10 vs labels  (b) mean error vs labels
                            (c) recall vs labels  (d) label efficiency ratio
  table_label_efficiency.tex
  summary.json

Usage: python3 de_figs.py --runs runs --out .
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

HEAD_LABEL = {'heatmap': 'Heatmap (deployed)',
              'softargmax': 'Soft-argmax',
              'coord': 'Coordinate regression'}
STYLE = {'heatmap': dict(color='#1f4e79', marker='o', ls='-'),
         'softargmax': dict(color='#3f7f3f', marker='^', ls='-.'),
         'coord': dict(color='#c0504d', marker='s', ls='--')}
HEAD_ORDER = ('heatmap', 'softargmax', 'coord')


def collect(runs_dir, split='test'):
    """{head: {fraction: {metric: [values over seeds]}}}"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    n_train = {}
    for f in sorted(glob.glob(os.path.join(runs_dir, '*.json'))):
        r = json.load(open(f))
        if 'results' not in r:
            continue
        o = r['results'][split]['overall']
        d = data[r['head']][r['fraction']]
        for k in ('pck5', 'pck10', 'pck20', 'mean_px', 'median_px', 'recall'):
            if o.get(k) is not None:
                d[k].append(o[k])
        # Precision is not stored directly: it is the fraction of emitted
        # instances that matched a ground-truth landmark of the same class.
        if o.get('n_det') is not None and o.get('n_fp') is not None:
            denom = o['n_det'] + o['n_fp']
            if denom:
                d['precision'].append(o['n_det'] / denom)
        d['_seeds'].append(r['seed'])
        n_train[r['fraction']] = r['n_train']
    return data, n_train


def ms(vals):
    a = np.array(vals, float)
    return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0


def interp_labels_for(fracs, n_train, means, target):
    """Number of labels at which a curve first reaches `target`, by linear
    interpolation between the bracketing fractions. Returns None if the
    target is never reached."""
    xs = [n_train[f] for f in fracs]
    for i in range(len(fracs)):
        if means[i] >= target:
            if i == 0:
                return xs[0]
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = means[i - 1], means[i]
            if y1 == y0:
                return x1
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--out', default='.')
    ap.add_argument('--split', default='test', choices=['test', 'val'])
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data, n_train = collect(a.runs, a.split)
    if not data:
        raise SystemExit(f'no run files found in {a.runs}')
    heads = [h for h in HEAD_ORDER if h in data]
    fracs = sorted(n_train)
    xs = [n_train[f] for f in fracs]

    fig, axgrid = plt.subplots(2, 2, figsize=(8.6, 6.2))
    axes = axgrid.ravel()
    panels = [('pck10', 'PCK@10 px', axes[0]),
              ('mean_px', 'Mean localization error (px)', axes[1]),
              ('recall', 'Detection recall', axes[2])]

    summary = {}
    for head in heads:
        summary[head] = {}
        for key, ylab, ax in panels:
            m, s = [], []
            for f in fracs:
                mu, sd = ms(data[head][f][key])
                m.append(mu)
                s.append(sd)
            summary[head][key] = dict(fractions=fracs, n_labels=xs,
                                      mean=m, std=s)
            m, s = np.array(m), np.array(s)
            ax.plot(xs, m, label=HEAD_LABEL[head], lw=1.8, ms=5, **STYLE[head])
            ax.fill_between(xs, m - s, m + s, alpha=0.18,
                            color=STYLE[head]['color'], lw=0)
            ax.set_xlabel('Labeled training images')
            ax.set_ylabel(ylab)
            ax.set_xscale('log')
            ax.set_xticks(xs)
            ax.set_xticklabels([str(v) for v in xs])
            ax.grid(alpha=0.3, which='both')

    # Retained for the text even though panel (d) now shows something else.
    for head in heads:
        m = summary[head]['pck10']['mean']
        summary[head]['labels_to_reach'] = {
            str(t): interp_labels_for(fracs, n_train, m, t)
            for t in (0.5, 0.6, 0.7, 0.8, 0.9)}

    # (d) the generalization gap: held-out frames of a training session
    # against entirely unseen sessions. This is the quantity a random frame
    # split would hide, and it does not shrink as labels are added.
    ax = axes[3]
    other = 'val' if a.split == 'test' else 'test'
    data_o, _ = collect(a.runs, other)
    head = heads[0]
    for split_name, src, sty in (
            (f'{other} (frames of training sessions)', data_o,
             dict(color='#7f7f7f', marker='^', ls=':')),
            (f'{a.split} (unseen sessions)', data,
             dict(color=STYLE[head]['color'], marker='o', ls='-'))):
        m = [ms(src[head][f]['pck10'])[0] for f in fracs]
        ax.plot(xs, m, label=split_name, lw=1.8, ms=5, **sty)
    m_in = np.array([ms(data_o[head][f]['pck10'])[0] for f in fracs])
    m_out = np.array([ms(data[head][f]['pck10'])[0] for f in fracs])
    ax.fill_between(xs, m_out, m_in, color='#7f7f7f', alpha=0.15, lw=0)
    for x, lo, hi in zip(xs, m_out, m_in):
        ax.annotate(f'{hi - lo:+.2f}', (x, (lo + hi) / 2), fontsize=7,
                    ha='center', va='center', color='#444444')
    # The panel plots one head; the summary records all three, because the
    # paper quotes the gap of the two baselines at the full budget as well.
    summary['generalization_gap'] = dict(plotted_head=head, n_labels=xs)
    for h in heads:
        i = np.array([ms(data_o[h][f]['pck10'])[0] for f in fracs])
        t = np.array([ms(data[h][f]['pck10'])[0] for f in fracs])
        summary['generalization_gap'][h] = dict(
            inside=list(i), outside=list(t), gap=list(i - t))
    ax.set_xlabel('Labeled training images')
    ax.set_ylabel(f'PCK@10 px, {HEAD_LABEL[head].split(" (")[0].lower()}')
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in xs])
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7, loc='lower right')

    for ax, letter in zip(axes, 'abcd'):
        ax.set_title(f'({letter})', loc='left', fontsize=11)
    axes[0].legend(fontsize=7.5, loc='lower right')
    fig.tight_layout()
    os.makedirs(a.out, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, f'fig_label_efficiency.{ext}'), dpi=200)

    # ------------------------------------------------ LaTeX table
    lines = [
        r'\begin{table}[H]', r'\caption{Landmark detection accuracy on the '
        r'held-out sessions as a function of the number of labeled training '
        r'images. Mean $\pm$ sample standard deviation over three seeds. '
        r'Recall is over landmark instances present in the frame, precision '
        r'over emitted instances.'
        r'\label{tab:label_efficiency}}',
        r'\begin{tabularx}{\textwidth}{lCCCCCC}', r'\toprule',
        r'\textbf{Head} & \textbf{Labels} & \textbf{Recall} & '
        r'\textbf{Precision} & '
        r'\textbf{Mean err.\ (px)} & \textbf{PCK@10} & \textbf{PCK@20} \\',
        r'\midrule']
    for head in heads:
        for j, f in enumerate(fracs):
            d = data[head][f]
            row = [HEAD_LABEL[head] if j == 0 else '',
                   f'{n_train[f]} ({f}\\%)']
            for k in ('recall', 'precision', 'mean_px', 'pck10', 'pck20'):
                mu, sd = ms(d[k])
                fmt = '{:.2f} $\\pm$ {:.2f}' if k == 'mean_px' \
                    else '{:.3f} $\\pm$ {:.3f}'
                row.append(fmt.format(mu, sd))
            lines.append(' & '.join(row) + r' \\')
        lines.append(r'\midrule' if head != heads[-1] else '')
    lines += [r'\bottomrule', r'\end{tabularx}', r'\end{table}']
    with open(os.path.join(a.out, 'table_label_efficiency.tex'), 'w') as fh:
        fh.write('\n'.join(l for l in lines if l) + '\n')

    with open(os.path.join(a.out, 'summary.json'), 'w') as fh:
        json.dump(dict(split=a.split, n_train=n_train, summary=summary), fh,
                  indent=1)

    print('runs found: ' + ', '.join(
        f'{h}={sum(len(data[h][f]["_seeds"]) for f in fracs)}' for h in heads))
    for head in heads:
        m = summary[head]['pck10']['mean']
        print(f'{head:>8} PCK@10: ' +
              '  '.join(f'{f}%={v:.3f}' for f, v in zip(fracs, m)))
    print('wrote fig_label_efficiency.pdf, table_label_efficiency.tex, '
          'summary.json')


if __name__ == '__main__':
    main()
