#!/usr/bin/env python3
"""Offline perception + localization on captured RGB-D frames -- NO ROS REQUIRED.

Runs the SAME components as the deployed robot:
  * the same quantized heatmap network (tflite, Edge TPU or CPU),
  * the same blob decoding (findBlobCenters from the released repository),
  * the same Monte Carlo Localization implementation (mcl.py),
  * the same field/landmark configuration file,
so the result is an end-to-end validation of the real perception-to-pose
pipeline rather than a re-implementation.

Geometry note
-------------
The landmarks lie on the ground while the camera sits about 0.5-0.9 m above it,
so the raw depth reading is NOT the ground-plane distance that the particle
filter expects. This script back-projects the detected pixel with the camera
intrinsics, rotates it into the robot frame using the camera height and pitch,
and takes the horizontal component. Use --range-mode raw to reproduce the
uncorrected behaviour for comparison.

Usage:
    python3 offline_localize.py \
        --captures captures \
        --model /path/to/model_tpu.tflite \
        --field /path/to/mini_soccer_field.field \
        --repo  /path/to/ros-robinion2 \
        --cam-height 0.62 --cam-pitch -20 \
        --out static_test.json
"""
import argparse, glob, json, os, sys
import numpy as np

CLASS_INFO = {0: 'right_field_corner', 1: 'right_area_baseline',
              2: 'right_area_corner', 3: 'right_goal_post',
              4: 'left_goal_post', 5: 'left_area_baseline',
              6: 'left_area_corner', 7: 'left_field_corner',
              8: 'right_center_junction', 9: 'field_center',
              10: 'left_center_junction', 11: 'ball'}


def load_repo(repo):
    sys.path.insert(0, os.path.join(repo, 'landmark_detection/scripts'))
    sys.path.insert(0, os.path.join(repo, 'robinion_localization/scripts'))
    import utils as det_utils
    import mcl as mcl_mod
    mcl_mod.random = np.random.random          # mcl.py calls random() unimported
    from field import Field
    return det_utils, mcl_mod, Field


def make_interpreter(model, device):
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
    if device != 'cpu':
        try:
            it = tflite.Interpreter(model, experimental_delegates=[
                tflite.load_delegate('libedgetpu.so.1', {'device': device})])
            it.allocate_tensors()
            return it, f'edgetpu({device})'
        except Exception as e:                                   # noqa: BLE001
            print(f'[warn] Edge TPU unavailable ({e}); using CPU')
    it = tflite.Interpreter(model)
    it.allocate_tensors()
    return it, 'cpu'


def detect(interp, det_utils, bgr, threshold, min_score):
    """Replicates inference_node.py: resize 224x224, invoke, dequantize, decode."""
    import cv2
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    scale, zero = out['quantization']
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    net_h, net_w = int(inp['shape'][1]), int(inp['shape'][2])
    resized = cv2.resize(rgb, (net_w, net_h))
    interp.set_tensor(inp['index'], np.expand_dims(resized, 0).astype(inp['dtype']))
    interp.invoke()
    data = interp.get_tensor(out['index']).astype(np.float32)
    if scale:
        data = (data - zero) * scale
    heatmap = np.squeeze(data)
    return det_utils.findBlobCenters(heatmap, minimum_score=min_score,
                                     threshold=threshold, resize_fix=True)


def pixel_to_ground_range(u, v, depth_m, K, cam_h, cam_pitch, mode):
    """Ground-plane distance from the robot to a landmark seen at pixel (u,v)."""
    if depth_m <= 0.05 or depth_m > 12.0:
        return None, None
    if mode == 'raw':                       # reproduce the deployed behaviour
        return float(depth_m), None
    fx, fy, cx, cy = K
    # camera-frame point (x right, y down, z forward)
    pc = np.array([(u - cx) / fx, (v - cy) / fy, 1.0]) * depth_m
    if mode == 'euclidean':
        return float(np.linalg.norm(pc)), None
    # ground mode: rotate by camera pitch into a robot frame with z up
    p = np.radians(cam_pitch)
    # forward (X), left (Y), up (Z) in the robot frame
    X = pc[2] * np.cos(p) + pc[1] * np.sin(p)
    Y = -pc[0]
    Z = cam_h - (pc[1] * np.cos(p) - pc[2] * np.sin(p))
    rng = float(np.hypot(X, Y))
    bearing = float(np.arctan2(Y, X))
    return rng, bearing


def localize(mcl_mod, Field, field, observations, n_particles, sensor_std,
             heading_prior, heading_std, iters, seed):
    """Run MCL on a static robot given repeated landmark range observations."""
    rng = np.random.default_rng(seed)
    P = mcl_mod.create_uniform_particles(
        (-field.width / 2, field.width / 2),
        (-field.height / 2, field.height / 2), (0, 2 * np.pi), n_particles)
    W = np.ones(n_particles) / n_particles
    trace = []
    for k in range(iters):
        z = observations[k % len(observations)]
        mcl_mod.predict(P, [0.0, 0.0, 0.0], [0.01, 0.01], 1.0)   # static jitter
        if z:
            mcl_mod.updateBasedOnDist(P, W, z, sensor_std, field)
        if heading_prior is not None:
            mcl_mod.updateBasedOnOrientation(P, W, heading_prior % (2 * np.pi),
                                             heading_std)
        if mcl_mod.neff(W) < n_particles / 2:
            mcl_mod.simple_resample(P, W)
        m, _ = mcl_mod.estimate(P, W)
        trace.append([float(m[0]), float(m[1]), float(m[2])])
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captures', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--field', required=True)
    ap.add_argument('--repo', required=True, help='path to ros-robinion2 checkout')
    ap.add_argument('--device', default='usb:0', help="Edge TPU device, or 'cpu'")
    ap.add_argument('--cam-height', type=float, default=0.62,
                    help='camera optical centre height above the ground (m)')
    ap.add_argument('--cam-pitch', type=float, default=-20.0,
                    help='camera pitch in degrees, negative = looking down')
    ap.add_argument('--range-mode', default='ground',
                    choices=['ground', 'euclidean', 'raw'])
    ap.add_argument('--particles', type=int, default=5000)
    ap.add_argument('--sensor-std', type=float, default=0.1)
    ap.add_argument('--heading-source', default='placement',
                    choices=['placement', 'none'],
                    help="'placement' supplies the surveyed heading as the prior "
                         "that the IMU provides on the robot; 'none' runs "
                         "range-only")
    ap.add_argument('--heading-std', type=float, default=0.1)
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--threshold', type=float, default=0.01)
    ap.add_argument('--min-score', type=float, default=7.5)
    ap.add_argument('--out', default='static_test.json')
    args = ap.parse_args()

    import cv2
    det_utils, mcl_mod, Field = load_repo(args.repo)
    field = Field(args.field)
    known = set(field.landmarks.keys())
    interp, backend = make_interpreter(args.model, args.device)
    print(f'backend: {backend}')

    intr = json.load(open(os.path.join(args.captures, 'intrinsics.json')))
    K = (intr['fx'], intr['fy'], intr['ppx'], intr['ppy'])
    dscale = intr.get('depth_scale', 0.001)

    results = []
    for d in sorted(glob.glob(os.path.join(args.captures, 'pose_*'))):
        meta = json.load(open(os.path.join(d, 'meta.json')))
        gt = np.array([meta['x'], meta['y'], meta['theta_rad']])
        rgbs = sorted(glob.glob(os.path.join(d, 'rgb_*.png')))
        obs, det_counts, per_class = [], [], {}
        for f in rgbs:
            depth = np.load(f.replace('rgb_', 'depth_').replace('.png', '.npy'))
            blobs = detect(interp, det_utils, cv2.imread(f),
                           args.threshold, args.min_score)
            z = {}
            for cl, (rx, ry), score in blobs:
                name = CLASS_INFO[cl]
                if name not in known:            # 'ball' is not a map landmark
                    continue
                u = min(depth.shape[1] - 1, int(rx * depth.shape[1] + 0.5))
                v = min(depth.shape[0] - 1, int(ry * depth.shape[0] + 0.5))
                dm = float(depth[v, u]) * dscale
                rng, _ = pixel_to_ground_range(u, v, dm, K, args.cam_height,
                                               args.cam_pitch, args.range_mode)
                if rng is None:
                    continue
                z[name] = rng
                per_class[name] = per_class.get(name, 0) + 1
            obs.append(z)
            det_counts.append(len(z))
        if not any(obs):
            print(f'{os.path.basename(d)}: NO usable detections - skipped')
            continue

        hp = gt[2] if args.heading_source == 'placement' else None
        trace = localize(mcl_mod, Field, field, obs, args.particles,
                         args.sensor_std, hp, args.heading_std, args.iters,
                         seed=meta['pose_index'])
        est = np.array(trace[-1])
        err_xy = float(np.hypot(est[0] - gt[0], est[1] - gt[1]))
        err_th = float(abs((est[2] - gt[2] + np.pi) % (2 * np.pi) - np.pi))
        errs = [float(np.hypot(t[0] - gt[0], t[1] - gt[1])) for t in trace]
        conv = next((i for i, e in enumerate(errs)
                     if all(v < 0.3 for v in errs[i:i + 5])), None)
        results.append(dict(pose=meta['pose_index'], gt=gt.tolist(),
                            est=est.tolist(), err_xy=err_xy, err_theta=err_th,
                            mean_detections=float(np.mean(det_counts)),
                            per_class=per_class, converged=conv is not None,
                            convergence_iter=conv, trace=trace))
        print(f"{os.path.basename(d)}: {np.mean(det_counts):.1f} landmarks/frame | "
              f"err {err_xy*100:.1f} cm, {np.degrees(err_th):.1f} deg | "
              f"{'converged' if conv is not None else 'NOT converged'}")

    summary = dict(backend=backend, range_mode=args.range_mode,
                   heading_source=args.heading_source,
                   particles=args.particles, sensor_std=args.sensor_std,
                   cam_height=args.cam_height, cam_pitch=args.cam_pitch,
                   model=os.path.basename(args.model), results=results)
    json.dump(summary, open(args.out, 'w'), indent=2)

    if results:
        e = np.array([r['err_xy'] for r in results])
        t = np.array([r['err_theta'] for r in results])
        print(f"\n=== {len(results)} poses ===")
        print(f"position RMSE : {np.sqrt(np.mean(e**2))*100:.1f} cm "
              f"(mean {e.mean()*100:.1f}, median {np.median(e)*100:.1f}, "
              f"max {e.max()*100:.1f})")
        print(f"heading  RMSE : {np.degrees(np.sqrt(np.mean(t**2))):.1f} deg")
        print(f"converged     : {sum(r['converged'] for r in results)}/{len(results)}")
        print(f"detections    : {np.mean([r['mean_detections'] for r in results]):.2f} "
              f"landmarks per frame")
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
