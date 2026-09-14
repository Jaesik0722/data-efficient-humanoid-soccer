#!/usr/bin/env python3
"""Per-update filter timing, measured on a named machine.

The accuracy results in this directory are machine-independent: the seeds are
fixed, so the same run reproduces anywhere. The per-update time is not, and the
paper reports it as evidence that the filter is affordable next to two Edge TPU
inferences on the robot. That claim is about the robot's onboard computer, so
the measurement has to be taken there.

Run this on BOTH machines with the same script, so the two numbers time the same
region of code and differ only in the hardware:

    export ROBINION_REPO=/path/to/ros-robinion2
    python3 time_onboard.py --tag host-i7-9750H
    python3 time_onboard.py --tag nuc8i7beh          # on the robot

Each run writes timing_<tag>.json. Accuracy is reported too, but only as a
sanity check that the filter behaved the same on both machines; the paper's
accuracy figures stay with the existing runs in results/.

The sweep matches the configuration behind Figure 6(b): sigma = 0.1 m, 69 deg
field of view, 20 Hz, 300 steps, IMU on, six seeds per particle count.
"""
import argparse
import json
import platform
import sys

import numpy as np

import mcl_experiments as X

PARTICLE_COUNTS = [500, 1000, 2000, 5000, 10000]
SEEDS = [200, 201, 202, 203, 204, 205]


def machine():
    """Whatever the interpreter can tell us about where this ran."""
    info = {
        'node': platform.node(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'machine': platform.machine(),
        'python': sys.version.split()[0],
        'numpy': np.__version__,
    }
    # platform.processor() is often empty on Linux; /proc/cpuinfo is not.
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                info['cpu'] = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    try:
        import os
        info['cpu_count'] = os.cpu_count()
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True,
                    help="short name for this machine, e.g. 'nuc8i7beh'")
    ap.add_argument('--counts', type=int, nargs='+', default=PARTICLE_COUNTS)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    host = machine()
    print(f"tag       {a.tag}")
    print(f"cpu       {host.get('cpu', host.get('processor') or 'unknown')}")
    print(f"numpy     {host['numpy']}  python {host['python']}\n")

    rows = {}
    for N in a.counts:
        ms, rmse = [], []
        for s in a.seeds:
            r = X.run_trial(s, N=N, sensor_std=0.1)
            ms.append(r['ms'])
            rmse.append(X.steady_rmse(r['errs']))
            print(f"  N={N:6d} seed={s}  {r['ms']:6.2f} ms")
        rows[N] = {
            'n_seeds': len(a.seeds),
            'ms_mean': float(np.mean(ms)),
            'ms_sd': float(np.std(ms, ddof=1)) if len(ms) > 1 else 0.0,
            'ms_all': ms,
            'rmse_cm_mean': float(np.mean(rmse) * 100),
        }

    out = a.out or f'timing_{a.tag}.json'
    json.dump({'tag': a.tag, 'machine': host, 'sensor_std': 0.1,
               'by_particles': rows}, open(out, 'w'), indent=1)

    print(f'\n{"N":>7} {"ms/update":>11} {"sd":>7} {"RMSE cm":>9}')
    for N, d in rows.items():
        print(f'{N:7d} {d["ms_mean"]:11.2f} {d["ms_sd"]:7.2f} {d["rmse_cm_mean"]:9.1f}')
    print(f'\nsaved {out}')


if __name__ == '__main__':
    main()
