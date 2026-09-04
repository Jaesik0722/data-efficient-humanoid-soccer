#!/usr/bin/env python3
"""Build figures for the extended MCL evaluation (observability, IMU ablation,
baseline comparison, robustness) from parts/*.json produced by mcl_experiments.py."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'font.family': 'serif',
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.1,
                     'figure.dpi': 200})

RATE = 20.0


def load(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(pat))]


def steady(vals, skip_s=5.0, rate=RATE):
    a = np.asarray(vals, dtype=float)[int(skip_s * rate):]
    a = a[~np.isnan(a)]
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else np.nan


def group(runs, key):
    """Group runs by a config value.

    JSON has no tuple type, so a config entry written as a tuple -- motion_std,
    for instance -- comes back as a list and cannot be used as a dict key. Any
    list is therefore converted back to a tuple here.
    """
    g = {}
    for r in runs:
        k = r['cfg'][key]
        g.setdefault(tuple(k) if isinstance(k, list) else k, []).append(r)
    return dict(sorted(g.items()))


# ---------------------------------------------------------------- observability
obs = group(load('parts/obs_*.json'), 'fov_deg')
obs_x, obs_rmse, obs_sd, obs_nvis = [], [], [], []
for fov, rs in obs.items():
    obs_x.append(fov)
    v = [steady(r['errs']) for r in rs]
    obs_rmse.append(np.mean(v)); obs_sd.append(np.std(v))
    obs_nvis.append(np.mean([np.mean(r['nvis']) for r in rs]))

# ------------------------------------------------------------------ IMU ablation
imu = group(load('parts/imu_*.json'), 'imu')
imu_stats = {}
for on, rs in imu.items():
    imu_stats[bool(on)] = dict(
        pos=[steady(r['errs']) for r in rs],
        hdg=[steady(r['herrs']) for r in rs],
        curve_pos=np.median([r['errs'] for r in rs], axis=0),
        curve_hdg=np.median([r['herrs'] for r in rs], axis=0))

# ------------------------------------------------------------------- baseline
base = load('parts/base_*.json')
mcl_v = [steady(r['errs']) for r in base]
tri_v = [steady(r['base_errs']) for r in base]
# availability: fraction of frames where trilateration was solvable
avail = np.mean([np.mean(~np.isnan(np.array(r['base_errs'], dtype=float))) for r in base])

# ------------------------------------------------------------------ robustness
odom = group(load('parts/odom_*.json'), 'motion_std')
odom_x = [k[0] if isinstance(k, (list, tuple)) else k for k in odom]
odom_m = [np.mean([steady(r['errs']) for r in rs]) for rs in odom.values()]
odom_s = [np.std([steady(r['errs']) for r in rs]) for rs in odom.values()]

rate = group(load('parts/rate_*.json'), 'rate')
rate_x = list(rate)
rate_m, rate_s, rate_ms = [], [], []
for hz, rs in rate.items():
    v = [steady(r['errs'], rate=hz) for r in rs]
    rate_m.append(np.mean(v)); rate_s.append(np.std(v))
    rate_ms.append(np.mean([r['ms'] for r in rs]))

kid = load('parts/kidnap_*.json')
KID_T = 7.5
kid_curves = np.array([r['errs'] for r in kid])
kid_med = np.median(kid_curves, axis=0)
kid_lo = np.percentile(kid_curves, 25, axis=0)
kid_hi = np.percentile(kid_curves, 75, axis=0)


def recovery_time(e, thr=0.3, hold=10, start=int(KID_T * RATE)):
    ok = np.asarray(e) < thr
    for i in range(start, len(ok) - hold):
        if ok[i:i + hold].all():
            return (i - start) / RATE
    return np.nan


rec = [recovery_time(r['errs']) for r in kid]
rec = [v for v in rec if not np.isnan(v)]

# ------------------------------------------------------------------- figure A
figA, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))

a = ax[0]
a.errorbar(obs_nvis, np.array(obs_rmse) * 100, yerr=np.array(obs_sd) * 100,
           marker='o', capsize=3)
for nv, rm, fov in zip(obs_nvis, obs_rmse, obs_x):
    a.annotate(f"{int(fov)}$^\\circ$", (nv, rm * 100), textcoords='offset points',
               xytext=(3, 5), fontsize=6, color='gray')
a.set_xlabel('Mean number of visible landmarks')
a.set_ylabel('Position RMSE (cm)')
a.set_title('(a) Observability (FOV sweep)', fontsize=8)

a = ax[1]
lbl = ['With IMU', 'Range only']
pos_data = [np.array(imu_stats[True]['hdg']), np.array(imu_stats[False]['hdg'])]
bp = a.boxplot([np.degrees(v) for v in pos_data], labels=lbl, widths=0.55,
               patch_artist=True)
for p, c in zip(bp['boxes'], ['tab:blue', 'tab:red']):
    p.set_facecolor(c); p.set_alpha(0.35)
a.set_ylabel('Heading RMSE (deg)')
a.set_yscale('log')
a.set_title('(b) IMU correction ablation', fontsize=8)

a = ax[2]
bp = a.boxplot([np.array(mcl_v) * 100, np.array(tri_v) * 100],
               labels=['MCL', 'LS trilat.'], widths=0.55, patch_artist=True)
for p, c in zip(bp['boxes'], ['tab:blue', 'tab:orange']):
    p.set_facecolor(c); p.set_alpha(0.35)
a.set_ylabel('Position RMSE (cm)')
a.set_yscale('log')
a.set_title(f'(c) Baseline comparison', fontsize=8)
a.text(0.5, 0.02, f'trilateration solvable in {avail*100:.0f}% of frames',
       transform=a.transAxes, ha='center', fontsize=6, color='gray')
figA.tight_layout()
figA.savefig('fig_mcl_analysis.pdf', bbox_inches='tight')
figA.savefig('fig_mcl_analysis.png', bbox_inches='tight', dpi=200)

# ------------------------------------------------------------------- figure B
figB, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))

a = ax[0]
a.errorbar(odom_x, np.array(odom_m) * 100, yerr=np.array(odom_s) * 100,
           marker='o', capsize=3, color='tab:purple')
a.set_xlabel('Odometry noise $\\sigma_d,\\sigma_\\theta$')
a.set_ylabel('Position RMSE (cm)')
a.set_title('(a) Odometry noise', fontsize=8)

a = ax[1]
a.errorbar(rate_x, np.array(rate_m) * 100, yerr=np.array(rate_s) * 100,
           marker='o', capsize=3, color='tab:blue', label='RMSE')
a.set_xlabel('Filter update rate (Hz)')
a.set_ylabel('Position RMSE (cm)')
a2 = a.twinx()
a2.plot(rate_x, rate_ms, marker='^', color='tab:green', label='Update time')
a2.set_ylabel('Update time (ms)', color='tab:green')
a2.tick_params(axis='y', colors='tab:green')
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=6.5, loc='upper center')
a.set_title('(b) Update rate', fontsize=8)

a = ax[2]
t = np.arange(len(kid_med)) / RATE
a.plot(t, kid_med, color='tab:red', label='Median')
a.fill_between(t, kid_lo, kid_hi, color='tab:red', alpha=0.25, label='IQR')
a.axvline(KID_T, ls=':', color='k', lw=0.9)
a.text(KID_T + 0.2, a.get_ylim()[1] * 0.5, 'kidnap', fontsize=6.5, rotation=90)
a.axhline(0.3, ls='--', lw=0.8, color='gray')
a.set_yscale('log')
a.set_xlabel('Time (s)'); a.set_ylabel('Position error (m)')
a.set_title('(c) Kidnapped-robot recovery', fontsize=8)
a.legend(fontsize=6.5, loc='upper right')
figB.tight_layout()
figB.savefig('fig_mcl_robustness.pdf', bbox_inches='tight')
figB.savefig('fig_mcl_robustness.png', bbox_inches='tight', dpi=200)

# ---------------------------------------------------------------------- report
# The manuscript quotes the MEDIAN across seeds for the observability sweep
# (Section 5.3: "the median position RMSE improves ... from 9.6 cm ... to
# 3.2 cm"), so both summaries are printed to make the comparison unambiguous.
print("=== OBSERVABILITY ===   (the paper quotes the median column)")
for fov, nv, rm, sd in zip(obs_x, obs_nvis, obs_rmse, obs_sd):
    med = np.median([steady(r['errs']) for r in obs[fov]])
    print(f"  FOV {fov:5.1f} deg | visible {nv:.2f} | median {med*100:4.1f} cm"
          f" | mean {rm*100:.1f} +- {sd*100:.1f} cm")
print("=== IMU ABLATION ===")
for on in [True, False]:
    s = imu_stats[on]
    print(f"  IMU={'on ' if on else 'off'} | pos {np.mean(s['pos'])*100:.1f} cm | "
          f"heading {np.degrees(np.mean(s['hdg'])):.1f} deg")
print("=== BASELINE ===")
print(f"  MCL {np.mean(mcl_v)*100:.1f} +- {np.std(mcl_v)*100:.1f} cm | "
      f"trilateration {np.nanmean(tri_v)*100:.1f} +- {np.nanstd(tri_v)*100:.1f} cm | "
      f"solvable {avail*100:.1f}% of frames")
print("=== ODOMETRY NOISE ===")
for xx, mm, ss in zip(odom_x, odom_m, odom_s):
    print(f"  sigma {xx:.2f} | RMSE {mm*100:.1f} +- {ss*100:.1f} cm")
print("=== UPDATE RATE ===")
for xx, mm, ss, ms in zip(rate_x, rate_m, rate_s, rate_ms):
    print(f"  {xx:.0f} Hz | RMSE {mm*100:.1f} +- {ss*100:.1f} cm | {ms:.1f} ms/update")
print("=== KIDNAP RECOVERY ===")
print(f"  recovered {len(rec)}/{len(kid)} runs | median {np.median(rec):.1f} s | "
      f"max {np.max(rec):.1f} s")
