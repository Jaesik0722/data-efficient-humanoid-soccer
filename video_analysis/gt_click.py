#!/usr/bin/env python3
"""Manual annotation for the absolute-localization measurement.

Two passes, both driven by clicking on the recorded demonstration:

  field   click the field landmarks that are visible in one static frame.
          These re-fit the image-to-ground homography from correspondences
          spread over the whole frame, rather than from the few far-side lines
          the automatic fit had to rely on.

  robots  step through frames and click the ground-contact point of each
          robot. This replaces the dark-blob detector, whose failures are the
          most likely reason the first attempt at this measurement produced an
          implausible number.

Every click is refined: the first click opens a 6x magnified inset, and the
second click inside it sets the point, so the achievable precision is about
one pixel of the original frame.

    python3 gt_click.py --mode field  --video localization.mp4 --frame 300
    python3 gt_click.py --mode robots --video localization.mp4 --n 20
"""
import argparse
import json
import os

import cv2
import numpy as np

VERSION = 'v2 (named arguments)'

# Landmark coordinates from robinion_localization/config/mini_soccer_field.field.
# Only the ones that can actually be seen are clicked; the rest are skipped.
FIELD = {
    'front_right_field_corner':  (-1.45,  1.85),
    'front_right_area_baseline': (-1.00,  1.85),
    'front_right_area_corner':   (-1.00,  0.95),
    'front_right_goal_post':     (-0.65,  1.85),
    'front_left_goal_post':      ( 0.65,  1.85),
    'front_left_area_baseline':  ( 1.00,  1.85),
    'front_left_area_corner':    ( 1.00,  0.95),
    'front_left_field_corner':   ( 1.50,  1.85),
    'left_center_junction':      ( 1.50,  0.00),
    'right_center_junction':     (-1.45,  0.00),
    'field_center':              ( 0.00,  0.00),
    'back_right_field_corner':   ( 1.50, -1.85),
    'back_right_area_baseline':  ( 1.00, -1.85),
    'back_right_area_corner':    ( 1.00, -0.95),
    'back_right_goal_post':      ( 0.65, -1.85),
    'back_left_goal_post':       (-0.65, -1.85),
    'back_left_area_baseline':   (-1.00, -1.85),
    'back_left_area_corner':     (-1.00, -0.95),
    'back_left_field_corner':    (-1.45, -1.85),
}

WIN = 'click  (click once to zoom, click again to set | s skip | u undo | q quit)'
ZOOM = 6
HALF = 90            # half-size of the magnified region, in source pixels


def grab(video, frame_idx):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f'could not read frame {frame_idx}')
    return img


class Picker:
    """Click once for a coarse position, again in the magnified inset to set it."""

    def __init__(self, img, scale=0.62):
        self.img = img
        self.scale = scale
        self.small = cv2.resize(img, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
        self.coarse = None
        self.point = None
        self.skip = False
        self.undo = False

    def _draw(self, banner):
        if self.coarse is None:
            view = self.small.copy()
        else:
            cx, cy = self.coarse
            x0 = int(np.clip(cx - HALF, 0, self.img.shape[1] - 2 * HALF))
            y0 = int(np.clip(cy - HALF, 0, self.img.shape[0] - 2 * HALF))
            patch = self.img[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF]
            view = cv2.resize(patch, None, fx=ZOOM, fy=ZOOM,
                              interpolation=cv2.INTER_NEAREST)
            self._origin = (x0, y0)
            h, w = view.shape[:2]
            cv2.line(view, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
            cv2.line(view, (0, h // 2), (w, h // 2), (0, 255, 255), 1)
        cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(view, banner, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN, view)

    def _on_mouse(self, ev, x, y, flags, _):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        if self.coarse is None:
            self.coarse = (x / self.scale, y / self.scale)
        else:
            ox, oy = self._origin
            self.point = (ox + x / ZOOM, oy + y / ZOOM)

    def run(self, banner):
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self._on_mouse)
        while True:
            self._draw(banner)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('q'):
                raise SystemExit('aborted')
            if k == ord('s'):
                self.skip = True
                return None
            if k == ord('u'):
                if self.coarse is not None and self.point is None:
                    self.coarse = None          # back out of the zoom
                else:
                    self.undo = True
                    return None
            if self.point is not None:
                return self.point


def do_field(a):
    img = grab(a.video, a.frame)
    out = {}
    names = list(FIELD)
    i = 0
    while i < len(names):
        n = names[i]
        p = Picker(img, a.scale)
        banner = (f'[{i+1}/{len(names)}] {n}  {FIELD[n]} m'
                  f'   -- s = not visible, u = back')
        pt = p.run(banner)
        if p.undo:
            i = max(0, i - 1)
            out.pop(names[i], None)
            continue
        if pt is not None:
            out[n] = [round(pt[0], 2), round(pt[1], 2)]
            print(f'  {n:28s} -> {out[n]}')
        i += 1
    cv2.destroyAllWindows()
    rec = dict(video=os.path.basename(a.video), frame=a.frame,
               field=FIELD, clicks=out)
    json.dump(rec, open(a.out, 'w'), indent=1)
    print(f'\n{len(out)} landmarks clicked -> {a.out}')
    if len(out) < 6:
        print('WARNING: fewer than 6 points. The fit will be poorly '
              'constrained; try to include points near the bottom of the '
              'frame as well as the far side.')


def do_robots(a):
    viz = json.load(open(a.viz))['tracks']
    common = sorted(set(int(p['frame']) for p in viz['0']) &
                    set(int(p['frame']) for p in viz['1']))
    if a.start:
        common = [f for f in common if f >= a.start]
    if a.end:
        common = [f for f in common if f <= a.end]
    step = max(1, len(common) // a.n)
    frames = common[::step][:a.n]
    print(f'{len(frames)} frames selected from {len(common)} with estimates '
          f'for both robots')

    cap = cv2.VideoCapture(a.video)
    out = []
    for k, f in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        row = {'frame': f}
        for side in ('left', 'right'):
            p = Picker(img, a.scale)
            banner = (f'[{k+1}/{len(frames)}] frame {f}: click the '
                      f'{side.upper()} robot where it meets the ground '
                      f'(midpoint between the feet)  -- s = skip')
            pt = p.run(banner)
            if pt is not None:
                row[side] = [round(pt[0], 2), round(pt[1], 2)]
        if 'left' in row and 'right' in row:
            out.append(row)
            print(f'  frame {f}: {row["left"]}  {row["right"]}')
    cap.release()
    cv2.destroyAllWindows()
    json.dump(dict(video=os.path.basename(a.video), rows=out),
              open(a.out, 'w'), indent=1)
    print(f'\n{len(out)} frames annotated -> {a.out}')


def main():
    ap = argparse.ArgumentParser(
        description='Manual annotation for the absolute-localization measurement.')
    ap.add_argument('--mode', required=True, choices=['field', 'robots'],
                    help="'field' clicks the landmarks; 'robots' clicks the "
                         "robot ground-contact points")
    ap.add_argument('--video', required=True, help='path to the recording')
    ap.add_argument('--frame', type=int, default=300,
                    help='[field] which frame to annotate')
    ap.add_argument('--viz', default='viz_tracks.json', help='[robots]')
    ap.add_argument('--n', type=int, default=20,
                    help='[robots] how many frames to annotate')
    ap.add_argument('--start', type=int, default=120, help='[robots]')
    ap.add_argument('--end', type=int, default=620, help='[robots]')
    ap.add_argument('--scale', type=float, default=0.62,
                    help='display scale of the first, coarse view')
    ap.add_argument('--out', default=None,
                    help='output JSON (defaults per mode)')
    a = ap.parse_args()

    print(f'gt_click.py {VERSION}')
    if not os.path.exists(a.video):
        raise SystemExit(f'video not found: {a.video}\n'
                         f'  (run this from the folder that contains it, or '
                         f'give the full path)')
    if a.out is None:
        a.out = 'field_points.json' if a.mode == 'field' else 'robot_points.json'
    (do_field if a.mode == 'field' else do_robots)(a)


if __name__ == '__main__':
    main()
