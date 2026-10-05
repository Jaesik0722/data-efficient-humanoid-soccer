#!/usr/bin/env python3
"""Per-session and detection-threshold analysis of one trained run.

Answers two reviewer questions that the stored run JSONs cannot:

  * how the held-out result splits across the individual held-out sessions,
    rather than only pooled over them;
  * how the heads compare when their false-detection rates are matched,
    rather than at whatever rate their default decoding happens to produce.

Both need the model's predictions, not just the summary numbers, so this
script runs on a model saved with `de_train.py --save-model`. It changes
nothing in the training path: the decoding here is `predict_labels` from
de_train, with the one threshold it hard-codes exposed as a sweep instead.

    python3 review_eval.py --splits splits.json \
            --model models/heatmap_f100_s0.keras --head heatmap \
            --out results/review/heatmap_f100_s0.json

What the swept threshold means differs by head, and that is the point:

  heatmap     the peak threshold of `de_common.decode_heatmap` (deployed
              value 0.30). `min_score` is held at its deployed 1.0.
  softargmax  the cutoff on the sigmoid of the visibility logit (deployed
              value 0.50).
  coord       the same visibility cutoff.

They are not comparable as numbers. They are comparable through the false
detections per image that each produces, which is what the matched
comparison in review_tables.py aligns on.
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import de_common as C
import de_train as T


def load_with_paths(paths):
    """Like de_train.load_pairs, but returns the paths that were actually
    read. load_pairs silently drops an unreadable image, which would put the
    labels out of step with the paths and so with the session tags."""
    import cv2
    xs, labels, kept = [], [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (C.INPUT_SIZE, C.INPUT_SIZE),
                         interpolation=cv2.INTER_AREA)
        xs.append(img.astype(np.uint8))
        labels.append(C.load_label(os.path.splitext(p)[0] + '.txt'))
        kept.append(p)
    if not xs:
        raise SystemExit(
            'not one image could be read. The paths in splits.json are '
            'relative, so run this from the directory they are relative to, '
            'or symlink the dataset next to splits.json.')
    if len(kept) < len(paths):
        print(f'  note: {len(paths) - len(kept)} of {len(paths)} images '
              f'could not be read and are excluded')
    return np.stack(xs), labels, kept


def session_of(path, sessions):
    """The session a frame belongs to, taken from its path.

    Matches against the session names splits.json records, longest first, so
    that 'Dataset1_Jeehyun' is not swallowed by a prefix like 'Dataset1'."""
    parts = set(os.path.normpath(path).split(os.sep))
    for s in sorted(sessions, key=len, reverse=True):
        if s in parts:
            return s
    for s in sorted(sessions, key=len, reverse=True):
        if s in path:
            return s
    return 'unassigned'


def predict_raw(model, x, head, batch=32):
    """Forward the whole split once. Decoding is cheap, the forward pass is
    not, so the sweep re-decodes these outputs instead of re-predicting."""
    out = []
    for i in range(0, len(x), batch):
        out.append(model.predict(x[i:i + batch].astype(np.float32) / 255.0,
                                 verbose=0))
    p = np.concatenate(out, axis=0)
    if head == 'heatmap':
        p = 1.0 / (1.0 + np.exp(-p))       # the model emits logits
    return p


def decode(p, head, thr, min_score=1.0):
    """The decoding of de_train.predict_labels, with its threshold exposed."""
    out = []
    for k in range(len(p)):
        if head == 'heatmap':
            out.append(C.decode_heatmap(p[k], thr, min_score))
        else:
            d = {}
            vis = 1.0 / (1.0 + np.exp(-p[k][:, 0]))
            xy = p[k][:, 1:]
            if head == 'coord':            # squashed into [0, 1] by the loss
                xy = 1.0 / (1.0 + np.exp(-xy))
            for c in range(C.N_CLASSES):
                if vis[c] >= thr:
                    d[c] = (float(xy[c, 0]), float(xy[c, 1]))
            out.append(d)
    return out


def overall(pred, labels):
    """C.evaluate's pooled block, plus the false detections per image, which
    is the quantity the matched comparison aligns on."""
    o = dict(C.evaluate(pred, labels)['overall'])
    o['fp_per_image'] = o['n_fp'] / max(o['n_images'], 1)
    return o


DEFAULT_THR = {'heatmap': 0.30, 'softargmax': 0.50, 'coord': 0.50}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', required=True)
    ap.add_argument('--model', required=True,
                    help='a .keras file written by de_train.py --save-model')
    ap.add_argument('--head', required=True,
                    choices=['heatmap', 'softargmax', 'coord'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--thresholds', default='0.05:0.95:0.05',
                    help='lo:hi:step of the swept decoding threshold')
    a = ap.parse_args()

    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)

    lo, hi, step = (float(v) for v in a.thresholds.split(':'))
    grid = sorted({round(float(t), 4)
                   for t in np.arange(lo, hi + 1e-9, step)}
                  | {DEFAULT_THR[a.head]})

    sp = C.load_splits(a.splits)
    sessions = sp.get('test_sessions', [])

    model = keras.models.load_model(
        a.model, custom_objects={'SoftArgmax2D': T.SoftArgmax2D},
        compile=False)
    print(f'loaded {a.model}')

    rec = dict(model=os.path.basename(a.model), head=a.head,
               deployed_threshold=DEFAULT_THR[a.head],
               test_sessions=sessions, splits=os.path.abspath(a.splits),
               thresholds=grid, sweep={}, per_session={})

    for split in ('val', 'test'):
        print(f'{split}: reading {len(sp[split])} images')
        x, labels, kept = load_with_paths(sp[split])
        print(f'{split}: forward pass')
        p = predict_raw(model, x, a.head, a.batch)

        rec['sweep'][split] = []
        for thr in grid:
            pred = decode(p, a.head, thr)
            o = overall(pred, labels)
            o['threshold'] = thr
            rec['sweep'][split].append(o)
            mark = '  <- deployed' if thr == DEFAULT_THR[a.head] else ''
            print(f'  {split} thr {thr:.2f}  recall {o["recall"]:.3f}  '
                  f'PCK@10 {o["pck10"]:.3f}  '
                  f'fp/img {o["fp_per_image"]:.3f}{mark}')

        if split == 'test' and sessions:
            tags = [session_of(q, sessions) for q in kept]
            found = sorted(set(tags))
            print(f'  sessions present: {found}')
            if 'unassigned' in found:
                print('  WARNING: some frames matched no session name from '
                      'splits.json; their rows are grouped as "unassigned" '
                      'and should not be reported.')
            for s in found:
                idx = [i for i, t in enumerate(tags) if t == s]
                sub = [labels[i] for i in idx]
                rows = []
                for thr in grid:
                    pred = decode(p[idx], a.head, thr)
                    o = overall(pred, sub)
                    o['threshold'] = thr
                    rows.append(o)
                rec['per_session'][s] = rows
                d = next(r for r in rows
                         if r['threshold'] == DEFAULT_THR[a.head])
                print(f'  session {s:<20} n={d["n_images"]:>4}  '
                      f'recall {d["recall"]:.3f}  PCK@10 {d["pck10"]:.3f}  '
                      f'mean {d["mean_px"]:.2f} px')

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(f'saved {a.out}')


if __name__ == '__main__':
    main()
