#!/usr/bin/env python3
"""Glass-to-glass latency from EXISTING footage, by frame counting.

Use this when the live system is no longer runnable but you have recordings in
which BOTH the real world AND the gamer-station monitor are visible in the same
frame. Because a single recording camera observes both, its own capture latency
cancels out, so the frame difference is a valid glass-to-glass measurement.

  latency_ms = (screen_frame - event_frame) / video_fps * 1000

Workflow
--------
1) Inspect the clip and list candidate events (sharp onsets work best):
       python3 video_latency_frames.py info match.mp4

2) Export the frames around a candidate event so you can step through them:
       python3 video_latency_frames.py dump match.mp4 --start 12.0 --end 13.5 \
           --out frames_ev1
   Open frames_ev1/ in any image viewer and use the arrow keys. File names carry
   the frame index, e.g. f_000361_t12.033s.png

3) Note two frame indices per event:
       event_frame  : first frame in which the REAL robot/ball starts to move
       screen_frame : first frame in which that motion appears ON THE MONITOR
   Write one line per event into g2g_frames.csv:

       # event_frame, screen_frame, video_fps
       361, 364, 29.97
       902, 906, 29.97

4) Compute the result (also handled by analyze_measurements.py):
       python3 video_latency_frames.py calc g2g_frames.csv

Good events to look for
-----------------------
  * a kick: the swing leg leaving the ground
  * the ball being struck and starting to roll
  * the robot starting to walk from a standstill
  * a hand or object entering the robot camera's view
  * any light switching on/off in view of the robot camera

Precision
---------
One frame of quantisation. At 30 fps that is 33 ms per frame, so collect at
least 10-15 events and report the mean with its standard error. If any clip was
recorded in slow motion (120/240 fps), prefer it: the resolution improves to
8/4 ms.
"""
import argparse, csv, os, sys
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit('this script needs opencv:  pip install opencv-python')


def cmd_info(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f'cannot open {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'file       : {args.video}')
    print(f'resolution : {w}x{h}')
    print(f'frame rate : {fps:.3f} fps   -> {1000.0/fps:.1f} ms per frame')
    print(f'frames     : {n}  ({n/fps:.1f} s)')
    print(f'\nquantisation of the latency measurement: +-{1000.0/fps:.1f} ms per event')
    if fps < 50:
        print('note: this is standard-rate footage; collect 10-15 events so the '
              'mean is precise enough.')
    cap.release()


def cmd_dump(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f'cannot open {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(args.out, exist_ok=True)
    f0 = int(args.start * fps)
    f1 = int(args.end * fps) if args.end else f0 + int(2 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    n = 0
    for idx in range(f0, f1 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        cv2.imwrite(os.path.join(args.out, f'f_{idx:06d}_t{idx/fps:07.3f}s.png'), frame)
        n += 1
    cap.release()
    print(f'wrote {n} frames ({f0}..{f1}) to {args.out}/  at {fps:.3f} fps')
    print('step through them and note the event frame and the screen frame.')


def read_pairs(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p for p in line.replace(',', ' ').split() if p]
            if len(parts) < 2:
                continue
            ev, sc = float(parts[0]), float(parts[1])
            fps = float(parts[2]) if len(parts) > 2 else 30.0
            rows.append((ev, sc, fps))
    return rows


def cmd_calc(args):
    rows = read_pairs(args.csv)
    if not rows:
        sys.exit('no usable rows found')
    vals = np.array([(sc - ev) / fps * 1000.0 for ev, sc, fps in rows])
    fps0 = rows[0][2]
    sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    print(f'events         : {len(vals)}')
    print(f'video fps      : {fps0:.2f}  (quantisation +-{1000.0/fps0:.1f} ms)')
    print(f'latency        : {vals.mean():.1f} +- {vals.std(ddof=1) if len(vals)>1 else 0:.1f} ms '
          f'(SD),  SEM {sem:.1f} ms')
    print(f'median         : {np.median(vals):.1f} ms   '
          f'range {vals.min():.1f}..{vals.max():.1f} ms')
    print(f'\nfor the paper  : {vals.mean():.0f} +- {max(sem, 1000.0/fps0/2):.0f} ms')
    out = os.path.splitext(args.csv)[0] + '_result.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['event_frame', 'screen_frame', 'video_fps', 'latency_ms'])
        for (ev, sc, fps), v in zip(rows, vals):
            w.writerow([ev, sc, fps, f'{v:.2f}'])
    print(f'saved {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('info', help='show frame rate and measurement resolution')
    p.add_argument('video'); p.set_defaults(func=cmd_info)

    p = sub.add_parser('dump', help='export frames around an event')
    p.add_argument('video')
    p.add_argument('--start', type=float, required=True, help='seconds')
    p.add_argument('--end', type=float, default=None, help='seconds')
    p.add_argument('--out', default='frames')
    p.add_argument('--scale', type=float, default=1.0)
    p.set_defaults(func=cmd_dump)

    p = sub.add_parser('calc', help='compute latency from g2g_frames.csv')
    p.add_argument('csv'); p.set_defaults(func=cmd_calc)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
