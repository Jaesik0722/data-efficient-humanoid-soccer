#!/usr/bin/env python3
"""Read the soft-argmax probes side by side.

    python3 read_probe.py                      # every probe json in this directory
    python3 read_probe.py a.json b.json        # or the ones named

Compares on localization error, not on the loss: the losses of different tau
are not on the same scale, so only accuracy is comparable across settings.

Rows from different step budgets are NOT comparable and the reader says so.
The 800-step and 3000-step learning-rate probes disagree with each other, and
the 3000-step one disagrees with the full 6000-step protocol; a probe is a
screen for configurations that fail outright, not a verdict on the winner.
"""
import glob
import json
import os
import sys


def rows(path):
    d = json.load(open(path))
    tau = d.get('tau', 1.0)
    w = d.get('coord_weight', 5.0)
    for r in d['results']:
        yield dict(src=os.path.basename(path), steps=d.get('steps'),
                   tau=tau, w=w, lr=r['lr'],
                   mean_px=r.get('mean_px'), pck10=r.get('pck10'),
                   recall=r.get('recall'), loss=r.get('val_loss'))


def main():
    files = sys.argv[1:] or sorted(
        glob.glob('lr_probe*.json') + glob.glob('probe_*.json')
        + glob.glob('tau_probe*.json'))
    if not files:
        raise SystemExit('no probe json found here')
    rs = [r for f in files for r in rows(f)]

    budgets = sorted({r['steps'] for r in rs if r['steps']})
    if len(budgets) > 1:
        print(f'WARNING: mixing step budgets {budgets}. Rows from different '
              f'budgets are not comparable; a shorter budget can rank an '
              f'undertrained head incorrectly.\n')

    print(f'{"tau":>5} {"w":>4} {"lr":>8} {"steps":>6} {"mean px":>9} '
          f'{"PCK@10":>8} {"recall":>8}   source')
    for r in sorted(rs, key=lambda r: (r['tau'], r['w'], r['lr'])):
        mp = f"{r['mean_px']:9.1f}" if r['mean_px'] else '      nan'
        print(f"{r['tau']:>5g} {r['w']:>4g} {r['lr']:>8g} "
              f"{(r['steps'] or 0):>6d} {mp} "
              f"{(r['pck10'] or 0):8.3f} {(r['recall'] or 0):8.3f}   {r['src']}")

    for steps in budgets:
        g = [r for r in rs if r['steps'] == steps]
        base = [r for r in g if r['tau'] == 1.0 and r['w'] == 5.0
                and abs(r['lr'] - 1e-3) < 1e-12]
        if not base:
            continue
        b = base[0]
        print(f"\nat {steps} steps, against the study's setting "
              f"(lr 1e-3, tau 1): mean {b['mean_px']:.1f} px, "
              f"PCK@10 {b['pck10']:.3f}")

        lrs = [r for r in g if r['tau'] == 1.0 and r['w'] == 5.0
               and abs(r['lr'] - 1e-3) >= 1e-12]
        for r in sorted(lrs, key=lambda r: r['lr']):
            d = (r['mean_px'] or 9e9) - b['mean_px']
            print(f"  lr {r['lr']:<8g} {r['mean_px']:6.1f} px "
                  f"({d:+.1f} vs baseline; negative is better)")

        taus = [r for r in g if r['w'] == 5.0
                and abs(r['lr'] - 1e-3) < 1e-12 and r['tau'] != 1.0]
        for r in sorted(taus, key=lambda r: r['tau']):
            d = (r['mean_px'] or 9e9) - b['mean_px']
            print(f"  tau {r['tau']:<7g} {r['mean_px']:6.1f} px "
                  f"({d:+.1f} vs baseline; negative is better)")

    print("\nA probe screens out configurations that fail outright. It does "
          "not settle\nwhich of the survivors is best: the 800- and 3000-step "
          "probes disagree about\nthe learning rate, and the 3000-step one "
          "disagrees with the full protocol.\nThe decisions reported in the "
          "paper come from 6000-step runs with the cosine\nschedule -- the "
          "softargmax_f100_s0_* files in this directory.")


if __name__ == '__main__':
    main()
