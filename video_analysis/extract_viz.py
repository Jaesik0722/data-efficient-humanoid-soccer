#!/usr/bin/env python3
"""Extract the localization estimates drawn by robinion_localization_viz.py
from a recorded demonstration video.

The visualiser maps field metres to panel pixels with
    sim = span * real + origin        (span_x == span_y, origin at panel centre)
so two landmarks with known field coordinates are enough to invert it. The two
centre-line junctions are used, which are unambiguous and always drawn.

Usage:
    python3 extract_viz.py VIDEO [--start 110] [--end 840] [--step 1] [--out viz_tracks.json]
"""
import argparse, json, sys
import cv2
import numpy as np

# centre-line landmarks of mini_soccer_field.field
X_RIGHT_JUNCTION = -1.45      # right_center_junction
X_LEFT_JUNCTION = 1.50        # left_center_junction

ROI = (0, 380, 1300, 1920)    # y0, y1, x0, x1 of the visualiser inset


def _runs(vec, thr, minlen):
    on = vec > thr
    out, s = [], None
    for i, b in enumerate(on):
        if b and s is None:
            s = i
        if not b and s is not None:
            if i - s > minlen:
                out.append((s, i))
            s = None
    if s is not None and len(on) - s > minlen:
        out.append((s, len(on)))
    return out


def panel_boxes(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (38, 70, 50), (80, 255, 255))
    cs = _runs(g.sum(0) / 255, 40, 40)
    rs = _runs(g.sum(1) / 255, 40, 40)
    if len(cs) != 2 or not rs:
        return None
    return [(x0, rs[0][0], x1, rs[0][1]) for x0, x1 in cs]


def calibrate(sub):
    """Return (scale_px_per_m, origin_x, origin_y) from the centre-line dots."""
    hs = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    red = (cv2.inRange(hs, (0, 100, 80), (10, 255, 255)) |
           cv2.inRange(hs, (168, 100, 80), (180, 255, 255)))
    n, _, st, ce = cv2.connectedComponentsWithStats(red)
    h = sub.shape[0]
    mid = sorted([(ce[i][0], ce[i][1]) for i in range(1, n)
                  if 1 <= st[i, 4] <= 40 and abs(ce[i][1] - h / 2) < h * 0.08])
    if len(mid) < 2:
        return None
    xl, xr = mid[0][0], mid[-1][0]
    s = (xr - xl) / (X_LEFT_JUNCTION - X_RIGHT_JUNCTION)
    if not (10 < s < 500):
        return None
    ox = xl - s * X_RIGHT_JUNCTION
    oy = (mid[0][1] + mid[-1][1]) / 2.0
    return s, ox, oy


def pose_marker(sub):
    """Yellow robot marker: centroid and area."""
    hs = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    y = cv2.inRange(hs, (22, 110, 140), (36, 255, 255))
    y = cv2.morphologyEx(y, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, st, ce = cv2.connectedComponentsWithStats(y)
    best = None
    for i in range(1, n):
        a = st[i, 4]
        if 20 <= a <= 400 and (best is None or a > best[2]):
            best = (ce[i][0], ce[i][1], a)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--start', type=int, default=110)
    ap.add_argument('--end', type=int, default=840)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--out', default='viz_tracks.json')
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit('cannot open video')
    y0, y1, x0, x1 = ROI

    tracks = {0: [], 1: []}
    cals = {0: [], 1: []}
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
        roi = img[y0:y1, x0:x1]
        boxes = panel_boxes(roi)
        if not boxes:
            continue
        for pi, (bx0, by0, bx1, by1) in enumerate(boxes):
            sub = roi[by0:by1, bx0:bx1]
            c = calibrate(sub)
            if not c:
                continue
            cals[pi].append(c)
            m = pose_marker(sub)
            if not m:
                continue
            s, ox, oy = c
            tracks[pi].append(dict(frame=f, x=float((m[0] - ox) / s),
                                   y=float((m[1] - oy) / s), area=float(m[2])))
    cap.release()

    meta = {}
    for pi in (0, 1):
        if cals[pi]:
            arr = np.array(cals[pi])
            meta[f'panel{pi}'] = dict(
                scale_px_per_m=float(arr[:, 0].mean()),
                scale_std=float(arr[:, 0].std()),
                origin_x=float(arr[:, 1].mean()),
                origin_y=float(arr[:, 2].mean()),
                calib_frames=len(arr), pose_frames=len(tracks[pi]))
            print(f"panel{pi}: scale {arr[:,0].mean():.2f} +- {arr[:,0].std():.2f} px/m, "
                  f"{len(tracks[pi])} poses")
    json.dump(dict(video=a.video, meta=meta,
                   tracks={str(k): v for k, v in tracks.items()}),
              open(a.out, 'w'))
    print(f'saved {a.out}')


if __name__ == '__main__':
    main()
