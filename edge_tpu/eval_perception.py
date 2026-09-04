#!/usr/bin/env python3
"""Evaluate the landmark heatmap network against the annotated dataset.

Point-like landmarks are evaluated with the metrics appropriate to keypoints
rather than boxes: detection rate, mean pixel error, and PCK at several
thresholds. Results are reported per recording session so that a
session-independent split can be applied.

Usage:
    python3 eval_perception.py --dataset "Localization Dataset" \
        --model model_1000_epochs_quant.tflite --repo ros-robinion2 \
        --out perception_eval.json
"""
import argparse, glob, json, os, sys
import numpy as np

CLASS_INFO = {0: 'right_field_corner', 1: 'right_area_baseline',
              2: 'right_area_corner', 3: 'right_goal_post',
              4: 'left_goal_post', 5: 'left_area_baseline',
              6: 'left_area_corner', 7: 'left_field_corner',
              8: 'right_center_junction', 9: 'field_center',
              10: 'left_center_junction', 11: 'ball'}


def load_labels(path):
    """YOLO format: cls cx cy w h (normalised). Returns {cls: (cx, cy)}."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        p = line.split()
        if len(p) >= 5:
            out[int(p[0])] = (float(p[1]), float(p[2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--threshold', type=float, default=0.01)
    ap.add_argument('--min-score', type=float, default=7.5)
    ap.add_argument('--img-w', type=int, default=640)
    ap.add_argument('--img-h', type=int, default=480)
    ap.add_argument('--limit', type=int, default=0, help='cap images per session')
    ap.add_argument('--out', default='perception_eval.json')
    args = ap.parse_args()

    import cv2
    import tflite_runtime.interpreter as tflite
    sys.path.insert(0, os.path.join(args.repo, 'landmark_detection/scripts'))
    import utils as det_utils

    interp = tflite.Interpreter(args.model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    scale, zero = out_d['quantization']
    net_h, net_w = int(inp['shape'][1]), int(inp['shape'][2])

    sessions = {}
    for d in sorted(glob.glob(os.path.join(args.dataset, '*'))):
        if not os.path.isdir(d):
            continue
        imgs = sorted(glob.glob(os.path.join(d, '*.jpg')))
        if imgs:
            sessions[os.path.basename(d)] = imgs
        for sub in sorted(glob.glob(os.path.join(d, '*'))):
            if os.path.isdir(sub):
                sub_imgs = sorted(glob.glob(os.path.join(sub, '*.jpg')))
                if sub_imgs:
                    key = f'{os.path.basename(d)}/{os.path.basename(sub)}'
                    sessions[key] = sub_imgs

    report = {}
    for name, imgs in sessions.items():
        if args.limit:
            imgs = imgs[:args.limit]
        # per class: tp (gt present & detected), fn, fp, pixel errors
        stat = {c: dict(gt=0, det=0, fp=0, err=[]) for c in CLASS_INFO}
        n_img = 0
        for f in imgs:
            gt = load_labels(os.path.splitext(f)[0] + '.txt')
            img = cv2.imread(f)
            if img is None:
                continue
            n_img += 1
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (net_w, net_h))
            interp.set_tensor(inp['index'],
                              np.expand_dims(resized, 0).astype(inp['dtype']))
            interp.invoke()
            data = interp.get_tensor(out_d['index']).astype(np.float32)
            if scale:
                data = (data - zero) * scale
            blobs = det_utils.findBlobCenters(np.squeeze(data),
                                              minimum_score=args.min_score,
                                              threshold=args.threshold,
                                              resize_fix=True)
            pred = {cl: pos for cl, pos, _ in blobs}
            for c in CLASS_INFO:
                has_gt, has_pred = c in gt, c in pred
                if has_gt:
                    stat[c]['gt'] += 1
                    if has_pred:
                        stat[c]['det'] += 1
                        dx = (pred[c][0] - gt[c][0]) * args.img_w
                        dy = (pred[c][1] - gt[c][1]) * args.img_h
                        stat[c]['err'].append(float(np.hypot(dx, dy)))
                elif has_pred:
                    stat[c]['fp'] += 1

        per_class, all_err = {}, []
        tot_gt = tot_det = tot_fp = 0
        for c, s in stat.items():
            if s['gt'] == 0 and s['fp'] == 0:
                continue
            e = np.array(s['err']) if s['err'] else np.array([])
            per_class[CLASS_INFO[c]] = dict(
                gt=s['gt'], detected=s['det'], false_pos=s['fp'],
                recall=(s['det'] / s['gt']) if s['gt'] else None,
                mean_px_err=float(e.mean()) if len(e) else None,
                median_px_err=float(np.median(e)) if len(e) else None,
                pck10=float((e <= 10).mean()) if len(e) else None,
                pck20=float((e <= 20).mean()) if len(e) else None)
            all_err += list(e)
            tot_gt += s['gt']; tot_det += s['det']; tot_fp += s['fp']

        E = np.array(all_err)
        report[name] = dict(
            images=n_img, gt_instances=tot_gt, detected=tot_det,
            false_positives=tot_fp,
            recall=(tot_det / tot_gt) if tot_gt else None,
            mean_px_err=float(E.mean()) if len(E) else None,
            median_px_err=float(np.median(E)) if len(E) else None,
            pck10=float((E <= 10).mean()) if len(E) else None,
            pck20=float((E <= 20).mean()) if len(E) else None,
            per_class=per_class)
        r = report[name]
        print(f"{name:34s} n={n_img:4d}  recall {100*(r['recall'] or 0):5.1f}%  "
              f"px err {r['mean_px_err'] or float('nan'):6.2f}  "
              f"PCK@10 {100*(r['pck10'] or 0):5.1f}%  FP {tot_fp}")

    json.dump(dict(model=os.path.basename(args.model), backend='cpu-int8',
                   threshold=args.threshold, min_score=args.min_score,
                   sessions=report), open(args.out, 'w'), indent=2)
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
