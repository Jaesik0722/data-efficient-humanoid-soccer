#!/usr/bin/env python3
"""Final analysis and figure for the field-localization measurement.

Three conventions relate the hand annotation to the visualiser output. None of
them is chosen by minimising the error; all three are fixed in advance:

  panel <-> robot   read off the recording (left panel shows the left robot).

  vertical sign     the visualiser maps field metres to panel pixels as
                    s = sigma * p + o with a single POSITIVE sigma on both
                    axes and no negation, so panel rows increase with field y.
                    YFLIP = +1 follows from the renderer, not from the data.

  field ends        which physical goal the robots' map calls y = +1.85 m was
                    fixed when the field was set up and is not recorded in the
                    configuration file. It is determined here from goal-area
                    membership and the SIGN of each robot's own estimate: each
                    robot sits in a goal area throughout the interval, and in
                    30 of 30 observations the sign of its estimated y is
                    opposite to the sign of its annotated y. ROT = -1.
                    Only signs are used, never an error magnitude.

The error under the opposite choice of field ends is reported afterwards as a
consistency check, not as the basis for the choice.

    python3 gt_report.py --field field_points.json --robots robot_points.json \
                         --viz viz_tracks.json
"""
import argparse, json
import cv2
import numpy as np

FPS = 29.97
YFLIP = +1             # visualiser applies one positive scale to both axes
SETTLED = 336          # frame after which the second robot stands still
ASSIGNMENT_NOTE = 'left visualiser panel = left robot (read off the recording)'


def fit(i, w):
    return cv2.findHomography(np.float32(i), np.float32(w), 0)[0]


def apply(H, p):
    q = H @ np.array([p[0], p[1], 1.0])
    return q[:2] / q[2]


def reference(path):
    d = json.load(open(path))
    names = sorted(d['clicks'])
    img = [d['clicks'][n] for n in names]
    wld = [d['field'][n] for n in names]
    H = fit(img, wld)
    loo = [np.linalg.norm(apply(fit([img[j] for j in range(len(names)) if j != i],
                                    [wld[j] for j in range(len(names)) if j != i]),
                                img[i]) - wld[i]) for i in range(len(names))]
    return H, float(np.median(loo)), len(names)


def pair(H, robots_path, viz_path):
    rows = json.load(open(robots_path))['rows']
    viz = json.load(open(viz_path))['tracks']
    est = {k: {int(p['frame']): (p['x'], p['y']) for p in viz[str(k)]} for k in (0, 1)}
    return [(r['frame'], apply(H, r['left']), apply(H, r['right']),
             np.array(est[0][r['frame']]), np.array(est[1][r['frame']]))
            for r in rows if r['frame'] in est[0] and r['frame'] in est[1]]


def errors(obs, rot, yf):
    eL = np.array([np.linalg.norm(gl * rot * np.array([1., yf]) - e0)
                   for _, gl, _, e0, _ in obs])
    eR = np.array([np.linalg.norm(gr * rot * np.array([1., yf]) - e1)
                   for _, _, gr, _, e1 in obs])
    return eL, eR


def field_ends_from_signs(obs):
    """Determine ROT from the sign of each estimate, never from its magnitude.

    Returns (rot, n_opposite, n_total). Each robot stands in a goal area for
    the whole interval, so the sign of its y is unambiguous in both frames; if
    the two frames disagree on that sign the field ends are interchanged.
    """
    opp = 0
    for _, gl, gr, e0, e1 in obs:
        opp += (np.sign(gl[1]) != np.sign(e0[1]))
        opp += (np.sign(gr[1]) != np.sign(e1[1]))
    n = 2 * len(obs)
    return (-1 if opp > n / 2 else 1), int(opp), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--field', default='field_points.json')
    ap.add_argument('--robots', default='robot_points.json')
    ap.add_argument('--viz', default='viz_tracks.json')
    ap.add_argument('--fig', default='fig_absolute_gt.pdf')
    a = ap.parse_args()

    H, loo, nlm = reference(a.field)
    obs = pair(H, a.robots, a.viz)
    yf = YFLIP                                   # fixed by the renderer
    rot, n_opp, n_tot = field_ends_from_signs(obs)   # fixed by signs alone
    eL, eR = errors(obs, rot, yf)
    alt = np.concatenate(errors(obs, -rot, yf))  # consistency check only
    runner = float(np.median(alt))
    e = np.concatenate([eL, eR])
    med = float(np.median(e))

    gL = np.array([o[1] * rot * np.array([1., yf]) for o in obs])
    gR = np.array([o[2] * rot * np.array([1., yf]) for o in obs])
    sL = np.array([o[3] for o in obs])
    sR = np.array([o[4] for o in obs])
    frames = [o[0] for o in obs]

    late = np.array(frames) >= SETTLED
    G, S = np.vstack([gL, gR]), np.vstack([sL, sR])
    bias = (S - G).mean(0)
    med_nb = float(np.median(np.linalg.norm(S - G - bias, axis=1)))

    rep = dict(
        assignment=ASSIGNMENT_NOTE, rot=rot, yflip=yf,
        landmarks=nlm, reference_loo_m=loo,
        frames=len(obs), frame_range=[frames[0], frames[-1]],
        n_pairs=int(e.size),
        median_m=med, mean_m=float(e.mean()),
        rmse_m=float(np.sqrt((e ** 2).mean())),
        p90_m=float(np.percentile(e, 90)), max_m=float(e.max()),
        alt_field_ends_median_m=runner, alt_ratio=runner / med,
        ends_sign_disagreement=f'{n_opp}/{n_tot}',
        moving_robot=dict(
            median_m=float(np.median(eL)),
            path_gt_m=float(np.abs(np.diff(gL, axis=0)).sum()),
            path_est_m=float(np.abs(np.diff(sL, axis=0)).sum()),
            corr_x=float(np.corrcoef(gL[:, 0], sL[:, 0])[0, 1]),
            corr_y=float(np.corrcoef(gL[:, 1], sL[:, 1])[0, 1]),
            bias_m=[float(v) for v in (sL - gL).mean(0)],
            median_after_bias_m=float(np.median(
                np.linalg.norm(sL - gL - (sL - gL).mean(0), axis=1)))),
        idle_robot=dict(
            median_m=float(np.median(eR)),
            min_m=float(eR.min()), max_m=float(eR.max()),
            settled_from_frame=SETTLED,
            settled_seconds=float((frames[-1] - SETTLED) / FPS),
            settled_true_max_excursion_m=float(np.linalg.norm(
                gR[late] - gR[late].mean(0), axis=1).max()),
            settled_est_sd_m=[float(sR[late, 0].std()), float(sR[late, 1].std())],
            settled_est_path_m=float(np.abs(np.diff(sR[late], axis=0)).sum()),
            early_move_m=float(np.linalg.norm(gR[late][0] - gR[0])),
            gt_sd_m=[float(gR[:, 0].std()), float(gR[:, 1].std())],
            est_sd_m=[float(sR[:, 0].std()), float(sR[:, 1].std())],
            corr_x=float(np.corrcoef(gR[:, 0], sR[:, 0])[0, 1]),
            corr_y=float(np.corrcoef(gR[:, 1], sR[:, 1])[0, 1])),
        global_bias_m=[float(v) for v in bias],
        median_after_global_bias_m=med_nb)
    json.dump(rep, open('absolute_gt_report.json', 'w'), indent=1)

    for k, v in rep.items():
        print(f'{k:28s} {v}')

    figure(a.fig, gL, sL, gR, sR, eL, eR, frames)
    print(f'\nsaved {a.fig} and absolute_gt_report.json')


def figure(path, gL, sL, gR, sR, eL, eR, frames):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif',
                         'axes.linewidth': 0.6})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0),
                           gridspec_kw=dict(width_ratios=[1.05, 1]))

    p = ax[0]
    ln = dict(color='0.55', lw=0.7, zorder=1)
    p.add_patch(plt.Rectangle((-1.45, -1.85), 2.95, 3.70, fill=False, **ln))
    p.plot([-1.45, 1.50], [0, 0], **ln)
    p.add_patch(plt.Circle((0, 0), 0.45, fill=False, **ln))
    for sy in (1, -1):
        p.add_patch(plt.Rectangle((-1.00, 0.95 * sy if sy > 0 else -1.85),
                                  2.00, 0.90, fill=False, **ln))
        p.plot([-0.65, 0.65], [1.98 * sy] * 2, color='0.35', lw=2.4, zorder=1)
        p.plot([-0.65, -0.65], [1.85 * sy, 1.98 * sy], color='0.35', lw=1.0, zorder=1)
        p.plot([0.65, 0.65], [1.85 * sy, 1.98 * sy], color='0.35', lw=1.0, zorder=1)

    for g, s, c, nm in ((gL, sL, '#1f77b4', 'robot A (walking)'),
                        (gR, sR, '#d62728', 'robot B (stationary)')):
        for a_, b_ in zip(g, s):
            p.plot([a_[0], b_[0]], [a_[1], b_[1]], color=c, lw=0.5,
                   alpha=0.35, zorder=2)
        p.plot(g[:, 0], g[:, 1], '-o', color=c, ms=2.6, lw=1.2, zorder=3,
               label=f'{nm} true')
        p.plot(s[:, 0], s[:, 1], '--s', color=c, ms=2.6, lw=1.0, zorder=3,
               mfc='none', label=f'{nm} est.')
    p.set_aspect('equal')
    p.set_xlim(-1.75, 1.80)
    p.set_ylim(-2.15, 3.35)
    p.set_xlabel('field $x$ (m)')
    p.set_ylabel('field $y$ (m)')
    p.legend(fontsize=5.4, loc='upper left', framealpha=0.0, handlelength=1.9, borderpad=0.4)
    p.set_title('(a) hand-annotated position vs. filter estimate', fontsize=8)

    q = ax[1]
    q.plot(frames, eL * 100, '-o', color='#1f77b4', ms=3, lw=1.2,
           label='robot A (walking)')
    q.plot(frames, eR * 100, '-s', color='#d62728', ms=3, lw=1.2,
           label='robot B (stationary)')
    q.axhline(np.median(np.concatenate([eL, eR])) * 100, color='0.3',
              ls=':', lw=1.0, label='pooled median')
    q.set_xlabel('video frame')
    q.set_ylabel('position error (cm)')
    q.set_ylim(0, 120)
    q.legend(fontsize=6.2, loc='upper left', framealpha=0.9)
    q.grid(alpha=0.25, lw=0.4)
    q.set_title('(b) error over the measured interval', fontsize=8)

    fig.tight_layout(pad=0.5)
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'), dpi=190)


if __name__ == '__main__':
    main()
