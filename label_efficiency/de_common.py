#!/usr/bin/env python3
"""Shared pieces for the label-efficiency study.

Dataset handling, session-independent splitting, nested label-fraction
sampling, heatmap encoding/decoding, and the keypoint metrics. Everything
here is deliberately free of TensorFlow so that the split can be inspected
and checked on a machine without a GPU.
"""
import glob
import json
import os

import numpy as np

# ---------------------------------------------------------------- classes

CLASS_NAMES = [
    'right_field_corner',      # 0
    'right_area_baseline',     # 1
    'right_area_corner',       # 2
    'right_goal_post',         # 3
    'left_goal_post',          # 4
    'left_area_baseline',      # 5
    'left_area_corner',        # 6
    'left_field_corner',       # 7
    'right_center_junction',   # 8
    'field_center',            # 9
    'left_center_junction',    # 10
    'ball',                    # 11
]
N_CLASSES = len(CLASS_NAMES)

# Left/right mirror pairs. A horizontal flip is only a valid augmentation if
# the class indices are swapped as well -- otherwise it teaches the network
# that a left goal post looks like a right goal post.
MIRROR = {0: 7, 7: 0, 1: 5, 5: 1, 2: 6, 6: 2, 3: 4, 4: 3,
          8: 10, 10: 8, 9: 9, 11: 11}

# Geometry of the deployed network.
INPUT_SIZE = 224          # network input, square
HEATMAP_SIZE = 56         # output grid, stride 4
IMG_W, IMG_H = 640, 480   # native camera resolution the labels refer to

# Sessions that are never used for training. Held out whole, so that the
# reported numbers describe generalisation to an unseen recording session
# rather than to unseen frames of a session the network has already seen.
DEFAULT_TEST_SESSIONS = ('Dataset4', 'Dataset1_Jeehyun')

# Directories inside the dataset archive that must not be used: these are
# transformed copies of frames that also appear elsewhere, so including them
# would leak training frames into the test set.
EXCLUDE_DIRS = ('new_labels_dataset', 'ybat-master')


# ---------------------------------------------------------------- loading

def load_label(path):
    """Read one YOLO-format label file.

    Returns {class_index: (cx, cy)} with coordinates normalised to [0, 1].
    Box width and height are ignored: these are point-like landmarks and the
    annotation tool was simply the only one available.
    """
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(parts[0])
            if 0 <= c < N_CLASSES:
                out[c] = (float(parts[1]), float(parts[2]))
    return out


def scan_dataset(root):
    """Return {session_name: [(image_path, label_path), ...]}.

    Only immediate subdirectories of `root` are treated as sessions, and only
    images that have a matching .txt label are kept.
    """
    sessions = {}
    for d in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        if name in EXCLUDE_DIRS:
            continue
        pairs = []
        for img in sorted(glob.glob(os.path.join(d, '*.jpg'))):
            lbl = os.path.splitext(img)[0] + '.txt'
            if os.path.exists(lbl):
                pairs.append((img, lbl))
        if pairs:
            sessions[name] = pairs
    return sessions


# ---------------------------------------------------------------- splitting

def build_splits(root, test_sessions=DEFAULT_TEST_SESSIONS, val_frac=0.15,
                 fractions=(0.10, 0.25, 0.50, 1.00), seed=0):
    """Build the session-independent split and the nested label subsets.

    The fraction subsets are *nested*: the 10% subset is contained in the 25%
    subset and so on. Without nesting, a non-monotone curve could be an
    artefact of which frames happened to be drawn rather than of the amount
    of supervision, and the curve would not be interpretable.

    Sampling is stratified by session, so every fraction preserves the
    session composition of the full training pool.
    """
    sessions = scan_dataset(root)
    missing = [s for s in test_sessions if s not in sessions]
    if missing:
        raise SystemExit(f'test sessions not found in dataset: {missing}')

    test = []
    for s in test_sessions:
        test += [p[0] for p in sessions[s]]

    rng = np.random.RandomState(seed)
    train_pool, val = [], []
    for name, pairs in sessions.items():
        if name in test_sessions:
            continue
        idx = rng.permutation(len(pairs))
        n_val = max(1, int(round(val_frac * len(pairs))))
        val += [pairs[i][0] for i in idx[:n_val]]
        train_pool += [pairs[i][0] for i in idx[n_val:]]

    # Nested subsets, stratified by session.
    by_session = {}
    for p in train_pool:
        by_session.setdefault(os.path.basename(os.path.dirname(p)), []).append(p)
    order = []
    for name in sorted(by_session):
        files = sorted(by_session[name])
        order.append([files[i] for i in rng.permutation(len(files))])

    subsets = {}
    for f in sorted(fractions):
        picked = []
        for files in order:
            picked += files[:int(round(f * len(files)))]
        subsets[f'{int(round(f * 100))}'] = sorted(picked)

    return dict(root=os.path.abspath(root), seed=seed,
                test_sessions=list(test_sessions),
                train_sessions=sorted(set(sessions) - set(test_sessions)),
                counts={k: len(v) for k, v in
                        dict(test=test, val=val, train_pool=train_pool).items()},
                fractions={k: len(v) for k, v in subsets.items()},
                test=sorted(test), val=sorted(val), subsets=subsets)


def load_splits(path):
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- heatmaps

def make_heatmap(label, size=HEATMAP_SIZE, sigma=1.5):
    """Encode {class: (cx, cy)} as a (size, size, N_CLASSES) Gaussian map."""
    hm = np.zeros((size, size, N_CLASSES), np.float32)
    if not label:
        return hm
    ax = np.arange(size, dtype=np.float32) + 0.5
    for c, (cx, cy) in label.items():
        mx, my = cx * size, cy * size
        if not (0 <= mx < size and 0 <= my < size):
            continue
        gx = np.exp(-((ax - mx) ** 2) / (2 * sigma ** 2))
        gy = np.exp(-((ax - my) ** 2) / (2 * sigma ** 2))
        hm[:, :, c] = np.maximum(hm[:, :, c], np.outer(gy, gx))
    return hm


def decode_heatmap(hm, threshold=0.30, min_score=1.0):
    """Decode a predicted heatmap back to {class: (cx, cy)}.

    For each channel the peak is located, the connected region above
    `threshold` around it is taken, and the intensity-weighted centroid of
    that region is returned. The centroid gives sub-cell precision, which
    matters because one cell is 4 input pixels (about 11 native pixels).

    NOTE: no `resize_fix` offset is applied. The +3.5-cell shift present in
    the deployed code is a bug; applying it here would move every prediction
    by roughly 42 px in x and 29 px in y.
    """
    out = {}
    size = hm.shape[0]
    for c in range(hm.shape[2]):
        ch = hm[:, :, c]
        peak = float(ch.max())
        if peak < threshold:
            continue
        r, k = np.unravel_index(int(ch.argmax()), ch.shape)
        mask = ch >= threshold
        # Keep only the component containing the peak (flood fill, 4-connected).
        comp = np.zeros_like(mask)
        stack = [(r, k)]
        comp[r, k] = True
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < size and 0 <= nx < size and mask[ny, nx] \
                        and not comp[ny, nx]:
                    comp[ny, nx] = True
                    stack.append((ny, nx))
        w = ch * comp
        total = float(w.sum())
        if total < min_score:
            continue
        ys, xs = np.nonzero(comp)
        cy = float((w[ys, xs] * (ys + 0.5)).sum() / total) / size
        cx = float((w[ys, xs] * (xs + 0.5)).sum() / total) / size
        out[c] = (cx, cy)
    return out


# ---------------------------------------------------------------- metrics

def evaluate(preds, gts, img_w=IMG_W, img_h=IMG_H, pck=(5, 10, 20)):
    """Keypoint metrics, pooled over images and reported per class.

    preds, gts: lists of {class: (cx, cy)}, one entry per image, aligned.

    Pixel errors are computed in the native 640x480 frame so that the numbers
    are directly comparable with those already reported in the manuscript.
    PCK is computed over ground-truth instances, counting a missed detection
    as incorrect -- otherwise a model can score well by predicting only the
    easy landmarks.
    """
    per_class = {c: dict(gt=0, det=0, fp=0, err=[]) for c in range(N_CLASSES)}
    for pred, gt in zip(preds, gts):
        for c in range(N_CLASSES):
            has_gt, has_pred = c in gt, c in pred
            if has_gt:
                per_class[c]['gt'] += 1
                if has_pred:
                    per_class[c]['det'] += 1
                    dx = (pred[c][0] - gt[c][0]) * img_w
                    dy = (pred[c][1] - gt[c][1]) * img_h
                    per_class[c]['err'].append(float(np.hypot(dx, dy)))
            elif has_pred:
                per_class[c]['fp'] += 1

    report, all_err = {}, []
    tot_gt = tot_det = tot_fp = 0
    pck_hit = {t: 0 for t in pck}
    for c, s in per_class.items():
        if s['gt'] == 0 and s['fp'] == 0:
            continue
        e = np.array(s['err'], np.float64)
        tot_gt += s['gt']
        tot_det += s['det']
        tot_fp += s['fp']
        all_err += list(e)
        entry = dict(n_gt=s['gt'], n_det=s['det'], n_fp=s['fp'],
                     recall=s['det'] / s['gt'] if s['gt'] else None,
                     mean_px=float(e.mean()) if e.size else None,
                     median_px=float(np.median(e)) if e.size else None)
        for t in pck:
            hit = int((e <= t).sum())
            pck_hit[t] += hit
            entry[f'pck{t}'] = hit / s['gt'] if s['gt'] else None
        report[CLASS_NAMES[c]] = entry

    e = np.array(all_err, np.float64)
    overall = dict(
        n_images=len(gts), n_gt=tot_gt, n_det=tot_det, n_fp=tot_fp,
        recall=tot_det / tot_gt if tot_gt else None,
        mean_px=float(e.mean()) if e.size else None,
        median_px=float(np.median(e)) if e.size else None,
        p90_px=float(np.percentile(e, 90)) if e.size else None,
    )
    for t in pck:
        overall[f'pck{t}'] = pck_hit[t] / tot_gt if tot_gt else None
    return dict(overall=overall, per_class=report)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Build and inspect the splits.')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', default='splits.json')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    sp = build_splits(a.dataset, seed=a.seed)
    with open(a.out, 'w') as fh:
        json.dump(sp, fh, indent=1)

    print(f'train sessions : {", ".join(sp["train_sessions"])}')
    print(f'test sessions  : {", ".join(sp["test_sessions"])}  (held out whole)')
    print(f'train pool     : {sp["counts"]["train_pool"]} images')
    print(f'validation     : {sp["counts"]["val"]} images')
    print(f'test           : {sp["counts"]["test"]} images')
    print('label fractions:')
    for k in sorted(sp['fractions'], key=int):
        print(f'  {k:>3}%  ->  {sp["fractions"][k]:>4} images')
    print(f'saved {a.out}')
