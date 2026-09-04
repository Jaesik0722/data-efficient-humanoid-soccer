#!/usr/bin/env python3
"""Benchmark a TFLite model on the Edge TPU.

RUN ON: the robot (onboard computer with the Edge TPU attached).

Measures three things:
  1. inference-only latency          (interpreter.invoke)
  2. full-pipeline latency           (resize + set_tensor + invoke + get_tensor)
  3. the resulting frames per second for both

Usage:
    python3 bench_edgetpu.py <model.tflite> [--device usb:1] [--runs 500]
                            [--label landmark] [--out results_fps.json]

Examples:
    python3 bench_edgetpu.py ~/catkin_ws/src/ros-robinion2/landmark_detection/model/model_tpu.tflite \
        --label landmark --out fps_landmark.json
    python3 bench_edgetpu.py ~/catkin_ws/src/ros-robinion2/landmark_detection/model/ball_model.tflite \
        --label ball --out fps_ball.json
"""
import argparse, json, time, platform, os
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:                                    # full TF fallback
    import tensorflow.lite as tflite                   # type: ignore


def percentile_stats(t_ms):
    return dict(mean=float(np.mean(t_ms)), std=float(np.std(t_ms)),
                median=float(np.median(t_ms)),
                p95=float(np.percentile(t_ms, 95)),
                min=float(np.min(t_ms)), max=float(np.max(t_ms)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--device', default='usb:1',
                    help='Edge TPU device string; try usb:0 if usb:1 fails')
    ap.add_argument('--runs', type=int, default=500)
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--label', default='model')
    ap.add_argument('--src-res', default='640x480',
                    help='camera resolution feeding the model, for the resize step')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    try:
        interpreter = tflite.Interpreter(
            args.model,
            experimental_delegates=[tflite.load_delegate('libedgetpu.so.1',
                                                         {'device': args.device})])
        backend = f'edgetpu({args.device})'
    except Exception as e:                              # noqa: BLE001
        print(f'[warn] Edge TPU delegate failed ({e}); falling back to CPU')
        interpreter = tflite.Interpreter(args.model)
        backend = 'cpu'

    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    outs = interpreter.get_output_details()
    shape = [int(v) for v in inp['shape']]
    dtype = inp['dtype']

    sw, sh = (int(v) for v in args.src_res.lower().split('x'))
    src_frame = np.random.randint(0, 255, size=(sh, sw, 3), dtype=np.uint8)
    net_h, net_w = shape[1], shape[2]

    try:
        import cv2
        def preprocess(frame):
            img = cv2.resize(frame, (net_w, net_h))
            return np.expand_dims(img, 0).astype(dtype)
    except ImportError:
        print('[warn] cv2 not available; pipeline timing will skip the resize step')
        def preprocess(frame):
            return np.zeros(shape, dtype=dtype)

    # ---------------------------------------------------------------- warm-up
    dummy = preprocess(src_frame)
    for _ in range(args.warmup):
        interpreter.set_tensor(inp['index'], dummy)
        interpreter.invoke()

    # ------------------------------------------------------- inference only
    t_inf = []
    for _ in range(args.runs):
        interpreter.set_tensor(inp['index'], dummy)
        t0 = time.perf_counter()
        interpreter.invoke()
        t_inf.append((time.perf_counter() - t0) * 1000.0)

    # -------------------------------------------------------- full pipeline
    t_pipe = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        x = preprocess(src_frame)
        interpreter.set_tensor(inp['index'], x)
        interpreter.invoke()
        for o in outs:
            interpreter.get_tensor(o['index'])
        t_pipe.append((time.perf_counter() - t0) * 1000.0)

    inf, pipe = percentile_stats(t_inf), percentile_stats(t_pipe)
    res = dict(label=args.label, model=os.path.basename(args.model),
               backend=backend, runs=args.runs,
               input_shape=shape, source_resolution=[sw, sh],
               host=platform.node(), machine=platform.machine(),
               inference_ms=inf, pipeline_ms=pipe,
               fps_inference=1000.0 / inf['mean'],
               fps_pipeline=1000.0 / pipe['mean'])

    print(f"\n=== {args.label}  ({os.path.basename(args.model)}, {backend}) ===")
    print(f"input shape        : {shape}")
    print(f"inference only     : {inf['mean']:.2f} +- {inf['std']:.2f} ms "
          f"(median {inf['median']:.2f}, p95 {inf['p95']:.2f})  ->  "
          f"{res['fps_inference']:.1f} fps")
    print(f"full pipeline      : {pipe['mean']:.2f} +- {pipe['std']:.2f} ms "
          f"(median {pipe['median']:.2f}, p95 {pipe['p95']:.2f})  ->  "
          f"{res['fps_pipeline']:.1f} fps")

    if args.out:
        json.dump(res, open(args.out, 'w'), indent=2)
        print(f"saved {args.out}")


if __name__ == '__main__':
    main()
