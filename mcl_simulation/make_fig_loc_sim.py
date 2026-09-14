#!/usr/bin/env python3
"""Figure 6 -- accuracy of the MCL filter under the nominal configuration.

Panels:
  (a) position and heading RMSE versus range measurement noise   (noise_*.json)
  (b) position RMSE versus particle count, with the per-update   (part_*.json +
      time measured on the robot's onboard computer               timing_*.json)
  (c) convergence from uniform initialization, median and IQR    (noise_*.json)
  (d) estimated versus ground-truth trajectory                   (traj_*.json)

Panel (b) deliberately mixes two sources. The accuracy comes from the stored
simulation runs, which are machine-independent because the seeds are fixed. The
per-update time comes from `time_onboard.py`, run on the robot, because that is
the machine the claim is about. The `ms` field inside `part_*.json` is *not*
used: it was recorded on the desktop machine under an environment that is no
longer reconstructible, and it disagrees with the current measurement by roughly
half. See the README.

Usage:
    cd results && python3 ../make_fig_loc_sim.py
    cd results && python3 ../make_fig_loc_sim.py --timing timing_host-i7-9750H.json
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'font.family': 'serif',
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.1,
                     'figure.dpi': 200})

RATE = 20.0
SKIP_S = 5.0


def load(pat):
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f'no files match {pat} -- run this from results/')
    return [json.load(open(f)) for f in files]


def steady(vals):
    """RMSE after the first SKIP_S seconds, matching the paper's definition."""
    a = np.asarray(vals, dtype=float)[int(SKIP_S * RATE):]
    a = a[~np.isnan(a)]
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else np.nan


def by_key(runs, key):
    g = {}
    for r in runs:
        g.setdefault(r[key], []).append(r)
    return dict(sorted(g.items()))


def stacked(runs, field):
    """Per-timestep matrix over runs, truncated to the shortest run."""
    n = min(len(r[field]) for r in runs)
    return np.array([r[field][:n] for r in runs], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timing', default='timing_nuc8i7beh.json',
                    help='per-update times from time_onboard.py')
    ap.add_argument('--out', default='fig_localization_sim')
    a = ap.parse_args()

    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0))
    (a1, a2), (a3, a4) = ax

    # ------------------------------------------------------------------ (a)
    noise = by_key(load('noise_*.json'), 'std')
    xs = list(noise)
    pos = [[steady(r['errs']) * 100 for r in rs] for rs in noise.values()]
    head = [[np.degrees(steady(r['herrs'])) for r in rs] for rs in noise.values()]
    # Population SD (ddof=0), the convention used by make_figs2.py and by the
    # spreads quoted in the manuscript. Do not change to ddof=1: the error bars
    # would no longer match the numbers in the text.
    pm, ps = [np.mean(v) for v in pos], [np.std(v) for v in pos]
    hm = [np.mean(v) for v in head]

    a1.errorbar(xs, pm, yerr=ps, marker='o', ms=3, capsize=2, color='C0')
    a1.set_xlabel(r'Range noise $\sigma$ (m)')
    a1.set_ylabel('Position RMSE (cm)', color='C0')
    a1.tick_params(axis='y', labelcolor='C0')
    a1b = a1.twinx()
    a1b.plot(xs, hm, marker='s', ms=3, color='C3', ls='--')
    a1b.set_ylabel(r'Heading RMSE ($^\circ$)', color='C3')
    a1b.tick_params(axis='y', labelcolor='C3')
    a1b.set_ylim(0, max(5, max(hm) * 1.5))
    a1.set_title('(a) Sensor noise', loc='left')

    # ------------------------------------------------------------------ (b)
    part = by_key(load('part_*.json'), 'N')
    ns = list(part)
    ppos = [[steady(r['errs']) * 100 for r in rs] for rs in part.values()]
    pm2, ps2 = [np.mean(v) for v in ppos], [np.std(v) for v in ppos]

    if not os.path.exists(a.timing):
        raise SystemExit(f'{a.timing} not found -- run time_onboard.py first')
    tj = json.load(open(a.timing))
    tms = [tj['by_particles'][str(n)]['ms_mean'] for n in ns]
    tsd = [tj['by_particles'][str(n)]['ms_sd'] for n in ns]
    cpu = tj['machine'].get('cpu', tj['tag'])

    # Set the log scale before plotting. Switching afterwards keeps the limits
    # autoscaled in linear space, whose lower edge is below the smallest N, and
    # the axis then stretches over four decades of empty space.
    a2.set_xscale('log')
    a2.errorbar(ns, pm2, yerr=ps2, marker='o', ms=3, capsize=2, color='C0')
    a2.set_xlim(min(ns) / 1.6, max(ns) * 1.6)
    a2.set_xticks(ns)
    a2.set_xticklabels([f'{n:,}' for n in ns])
    a2.set_xticks([], minor=True)
    a2.set_xlabel('Particles $N$')
    a2.set_ylabel('Position RMSE (cm)', color='C0')
    a2.tick_params(axis='y', labelcolor='C0')
    a2b = a2.twinx()
    a2b.errorbar(ns, tms, yerr=tsd, marker='s', ms=3, capsize=2,
                 color='C3', ls='--')
    a2b.set_ylabel('Update time (ms)', color='C3')
    a2b.tick_params(axis='y', labelcolor='C3')
    a2b.axhline(1000.0 / RATE, color='0.6', lw=0.7, ls=':')
    a2b.set_ylim(0, 1000.0 / RATE * 1.05)
    a2.set_title('(b) Particle count', loc='left')

    # ------------------------------------------------------------------ (c)
    for std, color, label in ((0.1, 'C0', r'$\sigma=0.1$ m'),
                              (0.4, 'C1', r'$\sigma=0.4$ m')):
        if std not in noise:
            continue
        m = stacked(noise[std], 'errs') * 100
        t = np.arange(m.shape[1]) / RATE
        med = np.median(m, axis=0)
        lo, hi = np.percentile(m, 25, axis=0), np.percentile(m, 75, axis=0)
        a3.plot(t, med, color=color, label=label)
        a3.fill_between(t, lo, hi, color=color, alpha=0.2, lw=0)
    a3.axhline(30, color='0.4', lw=0.7, ls='--')
    a3.set_xlabel('Time (s)')
    a3.set_ylabel('Position error (cm)')
    a3.set_yscale('log')
    a3.legend(frameon=False, loc='upper right')
    a3.set_title('(c) Convergence', loc='left')

    # ------------------------------------------------------------------ (d)
    tr = load('traj_*.json')[0]
    true = np.array(tr['true'])
    est = np.array(tr['est'])
    fw, fh = tr['fw'], tr['fh']
    a4.add_patch(plt.Rectangle((-fw / 2, -fh / 2), fw, fh,
                               fill=False, ec='0.7', lw=0.7))
    lm = np.array(list(tr['landmarks'].values()))
    a4.plot(lm[:, 0], lm[:, 1], '.', color='0.6', ms=3)
    a4.plot(true[:, 0], true[:, 1], color='0.2', lw=1.0, label='ground truth')
    a4.plot(est[:, 0], est[:, 1], color='C3', lw=0.8, alpha=0.85,
            label='estimate')
    a4.set_aspect('equal')
    a4.set_xlabel('$x$ (m)')
    a4.set_ylabel('$y$ (m)')
    a4.legend(frameon=False, loc='upper right', fontsize=7)
    a4.set_title('(d) Trajectory', loc='left')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{a.out}.{ext}', bbox_inches='tight')

    print(f'timing source: {a.timing}  ({cpu})\n')
    print('=== NOISE SWEEP ===')
    for s, m, sd, h in zip(xs, pm, ps, hm):
        print(f'  sigma {s:<5} | pos {m:5.1f} +- {sd:4.1f} cm | heading {h:5.1f} deg')
    print('=== PARTICLE COUNT ===')
    for n, m, sd, t, ts in zip(ns, pm2, ps2, tms, tsd):
        print(f'  N {n:6d} | pos {m:5.1f} +- {sd:4.1f} cm | {t:5.2f} +- {ts:4.2f} ms')
    print(f'\nwrote {a.out}.pdf and {a.out}.png')


if __name__ == '__main__':
    main()
