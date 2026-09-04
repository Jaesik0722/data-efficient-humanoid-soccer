#!/usr/bin/env python3
"""Capture RGB-D frames at surveyed field poses -- NO ROS REQUIRED.

RUN ON: any laptop with the RealSense camera plugged in over USB. The camera
should stay mounted on the robot at its normal height and tilt; only the USB
cable goes to the laptop. The robot's on-board computer is not used.

For each surveyed pose you place the robot, type the ground-truth pose, and the
script records a burst of aligned RGB + depth frames plus the camera intrinsics.

Usage:
    python3 capture_poses.py --out captures --frames 30

    # then, at the prompt, for each pose enter:  x y theta_deg
    # e.g.   -1.0  0.5  90
    # enter 'q' to finish.

Output layout:
    captures/
        intrinsics.json
        pose_000/  meta.json  rgb_000.png ... depth_000.npy ...
        pose_001/  ...
"""
import argparse, json, os, sys
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit('needs pyrealsense2:  pip install pyrealsense2')
try:
    import cv2
except ImportError:
    sys.exit('needs opencv:  pip install opencv-python')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='captures')
    ap.add_argument('--frames', type=int, default=30,
                    help='frames recorded per pose')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    json.dump(dict(fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy,
                   width=intr.width, height=intr.height,
                   model=str(intr.model), coeffs=list(intr.coeffs),
                   depth_scale=depth_scale),
              open(os.path.join(args.out, 'intrinsics.json'), 'w'), indent=2)
    print(f'intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} '
          f'ppx={intr.ppx:.1f} ppy={intr.ppy:.1f} depth_scale={depth_scale}')

    # let auto-exposure settle
    for _ in range(30):
        pipeline.wait_for_frames()

    idx = 0
    try:
        while True:
            print('\n--- next pose ---')
            print('place the robot, then enter the surveyed pose as: x y theta_deg')
            print("(x,y in metres in the field frame, theta in degrees, 'q' to quit)")
            line = input('> ').strip()
            if line.lower() in ('q', 'quit', 'exit', ''):
                break
            try:
                x, y, th = (float(v) for v in line.replace(',', ' ').split())
            except ValueError:
                print('could not parse; expected three numbers')
                continue

            d = os.path.join(args.out, f'pose_{idx:03d}')
            os.makedirs(d, exist_ok=True)
            print(f'recording {args.frames} frames ... hold still')
            n = 0
            while n < args.frames:
                frames = align.process(pipeline.wait_for_frames())
                c = frames.get_color_frame()
                z = frames.get_depth_frame()
                if not c or not z:
                    continue
                cv2.imwrite(os.path.join(d, f'rgb_{n:03d}.png'),
                            np.asanyarray(c.get_data()))
                np.save(os.path.join(d, f'depth_{n:03d}.npy'),
                        np.asanyarray(z.get_data()))
                n += 1
            json.dump(dict(pose_index=idx, x=x, y=y, theta_deg=th,
                           theta_rad=np.radians(th), frames=n),
                      open(os.path.join(d, 'meta.json'), 'w'), indent=2)
            print(f'saved {n} frames to {d}  (GT: x={x} y={y} theta={th} deg)')
            idx += 1
    finally:
        pipeline.stop()

    print(f'\ndone: {idx} poses recorded in {args.out}/')
    print('next: python3 offline_localize.py --captures %s ...' % args.out)


if __name__ == '__main__':
    main()
