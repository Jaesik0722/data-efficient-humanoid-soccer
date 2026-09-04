#!/usr/bin/env python3
"""Summarise the physical static-localization test and draw the paper figure.

Usage:
    python3 analyze_static_test.py static_test.json [--sim-rmse 6.2] [--out fig_static_test.pdf]
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'font.family': 'serif',
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.1,
                     'figure.dpi': 200})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('json')
    ap.add_argument('--sim-rmse', type=float, default=6.2,
                    help='simulation position RMSE in cm, for comparison')
    ap.add_argument('--out', default='fig_static_test.pdf')
    args = ap.parse_args()

    d = json.load(open(args.json))
    R = d['results']
    if not R:
        raise SystemExit('no results in file')

    e = np.array([r['err_xy'] for r in R]) * 100.0          # cm
    th = np.degrees([r['err_theta'] for r in R])
    nd = np.array([r['mean_detections'] for r in R])
    conv = np.array([r['converged'] for r in R])
    rmse = float(np.sqrt(np.mean(e ** 2)))
    th_rmse = float(np.sqrt(np.mean(th ** 2)))

    print(f"poses            : {len(R)}")
    print(f"backend          : {d['backend']}   range mode: {d['range_mode']}")
    print(f"position RMSE    : {rmse:.1f} cm "
          f"(mean {e.mean():.1f}, median {np.median(e):.1f}, max {e.max():.1f})")
    print(f"heading RMSE     : {th_rmse:.1f} deg")
    print(f"converged        : {conv.sum()}/{len(R)}")
    print(f"detections/frame : {nd.mean():.2f} +- {nd.std():.2f}")
    print(f"\nfor the paper: {rmse:.1f} cm physical vs {args.sim_rmse:.1f} cm simulation "
          f"(ratio {rmse/args.sim_rmse:.2f}x)")

    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))

    a = ax[0]
    gt = np.array([r['gt'] for r in R])
    es = np.array([r['est'] for r in R])
    a.scatter(gt[:, 0], gt[:, 1], marker='o', s=26, facecolors='none',
              edgecolors='k', label='Ground truth')
    a.scatter(es[:, 0], es[:, 1], marker='x', s=26, color='tab:red',
              label='MCL estimate')
    for g, s in zip(gt, es):
        a.plot([g[0], s[0]], [g[1], s[1]], '-', color='gray', lw=0.6)
    a.set_aspect('equal'); a.set_xlabel('x (m)'); a.set_ylabel('y (m)')
    a.set_title('(a) Surveyed vs. estimated poses', fontsize=8)
    a.legend(fontsize=6.5)

    a = ax[1]
    a.hist(e, bins=min(12, max(4, len(e) // 2)), color='tab:blue', alpha=0.75,
           edgecolor='white', linewidth=0.4)
    a.axvline(rmse, color='tab:red', ls='--', lw=1, label=f'RMSE {rmse:.1f} cm')
    a.axvline(args.sim_rmse, color='tab:green', ls=':', lw=1.2,
              label=f'simulation {args.sim_rmse:.1f} cm')
    a.set_xlabel('Position error (cm)'); a.set_ylabel('Count')
    a.set_title(f'(b) Error distribution (n={len(R)})', fontsize=8)
    a.legend(fontsize=6.5)

    a = ax[2]
    a.scatter(nd, e, s=26, color='tab:purple')
    if len(nd) > 2 and nd.std() > 1e-6:
        k = np.polyfit(nd, e, 1)
        xs = np.linspace(nd.min(), nd.max(), 20)
        a.plot(xs, np.polyval(k, xs), '--', color='gray', lw=0.9)
    a.set_xlabel('Detected landmarks per frame')
    a.set_ylabel('Position error (cm)')
    a.set_title('(c) Error vs. observability', fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches='tight')
    fig.savefig(args.out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
