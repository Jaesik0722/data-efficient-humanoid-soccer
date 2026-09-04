#!/usr/bin/env python3
"""Measure the localization error of the deployed system from demonstration video.

The estimated pose is read from the on-screen localization visualiser and the
true pose from the field view, both in the same frame, so no instrumentation of
the robot is required. The field-view homography is fitted to the painted lines
of mini_soccer_field.field; its residual is reported as the measurement
uncertainty.

Usage: python3 measure_localization.py VIDEO --viz viz_tracks.json --H H_final.npy
"""
import argparse, json
import cv2
import numpy as np

XC, YG = 1.475, 1.85


def field_mask(shape, Hi, margin=0.10):
    c = np.float32([[[-XC - margin, -YG - margin]], [[XC + margin, -YG - margin]],
                    [[XC + margin, YG + margin]], [[-XC - margin, YG + margin]]])
    poly = cv2.perspectiveTransform(c, Hi).reshape(-1, 2)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.int32(poly)], 255)
    return m, poly


def robot_feet(img, fmask):
    """Ground-contact points of the dark robot bodies standing on the turf."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 85))
    dark = cv2.bitwise_and(dark, fmask)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
    H_img = img.shape[0]
    out = []
    for i in range(1, n):
        a = st[i, 4]
        if not (6000 <= a <= 160000):
            continue
        ys, xs = np.nonzero(lab == i)
        ymax = int(ys.max())
        if ymax >= H_img - 3:          # blob runs off the bottom: foot unknown
            continue
        sel = xs[ys > ymax - 15]
        out.append((float(np.median(sel)), float(ymax), int(a)))
    out.sort(key=lambda t: -t[2])
    return out[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--viz', required=True)
    ap.add_argument('--H', required=True)
    ap.add_argument('--start', type=int, default=120)
    ap.add_argument('--end', type=int, default=620)
    ap.add_argument('--step', type=int, default=4)
    ap.add_argument('--out', default='localization_video.json')
    a = ap.parse_args()

    H = np.load(a.H)
    Hi = np.linalg.inv(H)
    viz = json.load(open(a.viz))['tracks']
    est = {int(p['frame']): (p['x'], p['y']) for p in viz['0']}
    est2 = {int(p['frame']): (p['x'], p['y']) for p in viz['1']}

    cap = cv2.VideoCapture(a.video)
    fmask = None
    rows = []
    f = -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        f += 1
        if f < a.start:
            continue
        if f > a.end:
            break
        if (f - a.start) % a.step:
            continue
        if fmask is None:
            fmask, _ = field_mask(img.shape, Hi)
        feet = robot_feet(img, fmask)
        if len(feet) < 2 or f not in est or f not in est2:
            continue
        pts = np.float32([[[fx, fy]] for fx, fy, _ in feet])
        w = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        rows.append(dict(frame=f, true=[list(map(float, p)) for p in w],
                         est=[list(est[f]), list(est2[f])]))
    cap.release()

    if not rows:
        raise SystemExit('no paired frames')

    # resolve the two-fold y ambiguity and the robot/panel assignment globally
    best = None
    for yflip in (1, -1):
        for perm in ((0, 1), (1, 0)):
            errs = []
            for r in rows:
                for i, j in enumerate(perm):
                    t = np.array(r['true'][i]) * np.array([1.0, yflip])
                    e = np.array(r['est'][j])
                    errs.append(float(np.linalg.norm(t - e)))
            m = float(np.median(errs))
            if best is None or m < best[0]:
                best = (m, yflip, perm, errs)
    med, yflip, perm, errs = best
    errs = np.array(errs)
    print(f'paired frames      : {len(rows)}  ({len(errs)} robot observations)')
    print(f'resolved y flip    : {yflip},  panel assignment {perm}')
    print(f'localization error : median {med*100:.1f} cm, mean {errs.mean()*100:.1f} cm, '
          f'RMSE {np.sqrt((errs**2).mean())*100:.1f} cm')
    print(f'                     p90 {np.percentile(errs,90)*100:.1f} cm, '
          f'max {errs.max()*100:.1f} cm')

    json.dump(dict(n_frames=len(rows), n_obs=len(errs), yflip=yflip,
                   assignment=list(perm),
                   median_m=med, mean_m=float(errs.mean()),
                   rmse_m=float(np.sqrt((errs**2).mean())),
                   p90_m=float(np.percentile(errs, 90)),
                   errors_m=[float(e) for e in errs], rows=rows), open(a.out, 'w'))
    print(f'saved {a.out}')


if __name__ == '__main__':
    main()
