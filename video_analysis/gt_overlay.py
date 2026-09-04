#!/usr/bin/env python3
"""Draw the field model onto the frame using the homography fitted from the
clicks, so that the clicks can be checked by eye before anything is measured.

If the drawn lines sit on the painted lines, the fit is good. If they drift,
the clicks that are wrong are usually obvious: the drawn model will pivot away
from the paint in the direction of the bad point.

    python3 gt_overlay.py --field field_points.json --video localization.mp4
"""
import argparse
import json

import cv2
import numpy as np

# outline, halfway line and the two goal areas, in field metres
XN, XF, YA, YB = -1.45, 1.50, -1.85, 1.85
SEGMENTS = [
    [(XN, YA), (XF, YA)], [(XF, YA), (XF, YB)],
    [(XF, YB), (XN, YB)], [(XN, YB), (XN, YA)],
    [(XN, 0.0), (XF, 0.0)],
    [(-1.0, YB), (-1.0, 0.95)], [(-1.0, 0.95), (1.0, 0.95)], [(1.0, 0.95), (1.0, YB)],
    [(-1.0, YA), (-1.0, -0.95)], [(-1.0, -0.95), (1.0, -0.95)], [(1.0, -0.95), (1.0, YA)],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--field', default='field_points.json')
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', default='overlay_check.jpg')
    ap.add_argument('--circle', type=float, default=0.48,
                    help='centre-circle radius in metres, for reference only; '
                         'it is not used in the fit')
    a = ap.parse_args()

    d = json.load(open(a.field))
    names = sorted(d['clicks'])
    if len(names) < 4:
        raise SystemExit(f'only {len(names)} clicks; at least 4 are needed')
    img_pts = np.float32([d['clicks'][n] for n in names])
    world = np.float32([d['field'][n] for n in names])
    H, _ = cv2.findHomography(img_pts, world, 0)
    Hi = np.linalg.inv(H)

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, d.get('frame', 300))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit('could not read the frame')

    def to_img(p):
        q = Hi @ np.array([p[0], p[1], 1.0])
        return int(round(q[0] / q[2])), int(round(q[1] / q[2]))

    for p, q in SEGMENTS:
        cv2.line(img, to_img(p), to_img(q), (0, 0, 255), 2, cv2.LINE_AA)
    ring = [to_img((a.circle * np.cos(t), a.circle * np.sin(t)))
            for t in np.linspace(0, 2 * np.pi, 90)]
    cv2.polylines(img, [np.int32(ring)], True, (255, 0, 255), 2, cv2.LINE_AA)

    for n in names:
        u, v = int(d['clicks'][n][0]), int(d['clicks'][n][1])
        cv2.circle(img, (u, v), 7, (0, 255, 0), -1)
        cv2.circle(img, (u, v), 7, (0, 0, 0), 1)
        cv2.putText(img, n.replace('front', 'F').replace('back', 'B'),
                    (u + 10, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite(a.out, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'wrote {a.out}')
    print('Red lines are the field model drawn through your clicks. If they sit '
          'on the painted lines, the clicks are consistent.')
    print('The magenta circle uses an assumed radius and is only a sanity cue; '
          'it is not part of the fit.')


if __name__ == '__main__':
    main()
