#!/usr/bin/env python3
"""Find a learning rate and step budget that actually converge.

The heatmap target is 99.3 % background, so the loss scale is deceptive: a
network that has learned nothing except "output zero everywhere" already
reaches an MSE of about 7.4e-4, while an untrained network sits at 0.25.
Any run whose validation loss is still far above the all-zeros baseline has
not begun to solve the task, whatever the loss curve looks like.

This script trains the full-label subset for a short budget at several
learning rates and reports, for each, the validation loss relative to that
baseline plus the actual keypoint accuracy. Run it before committing to the
24-run study.

    python tune_lr.py --splits splits.json --steps 800
"""
import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

import de_common as C
import de_train as T


def zeros_baseline(paths, limit=400):
    """MSE obtained by predicting an empty heatmap: the floor that means
    'has learned nothing useful'."""
    tot, n = 0.0, 0
    for p in paths[:limit]:
        hm = C.make_heatmap(C.load_label(os.path.splitext(p)[0] + '.txt'))
        tot += float((hm ** 2).mean())
        n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', required=True)
    ap.add_argument('--head', choices=['heatmap', 'softargmax', 'coord'],
                    default='heatmap')
    ap.add_argument('--fraction', default='100')
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lrs', default='3e-4,1e-3,3e-3')
    ap.add_argument('--pos-weight', type=float, default=100.0)
    ap.add_argument('--tau', type=float, default=1.0,
                    help='softargmax only: spatial-softmax temperature')
    ap.add_argument('--coord-weight', type=float, default=T.COORD_WEIGHT,
                    help='softargmax and coord only: coordinate-term weight')
    ap.add_argument('--cosine', action='store_true',
                    help='decay the rate to zero over the budget')
    ap.add_argument('--out', default='lr_probe.json')
    a = ap.parse_args()

    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)

    sp = C.load_splits(a.splits)
    train_paths = sp['subsets'][a.fraction]
    base = zeros_baseline(train_paths)
    print(f'all-zeros baseline (val MSE of a useless model): {base:.6f}')
    print(f'initialisation      (sigmoid outputs 0.5)      : 0.250000')
    if a.head != 'heatmap':
        print('  (that baseline describes the heatmap target; for this head '
              'the verdict below rests on accuracy, not on the loss)')
    if a.head == 'softargmax':
        print(f'  tau {a.tau:g}, coordinate weight {a.coord_weight:g}')
    print()

    xtr, ytr, _ = T.load_pairs(train_paths, a.head)
    xva, yva, lva = T.load_pairs(sp['val'], a.head)
    val = T.make_ds(xva, yva, a.head, a.batch, False)

    results = []
    for lr in [float(v) for v in a.lrs.split(',')]:
        tf.keras.utils.set_random_seed(0)
        model = {'heatmap': T.build_heatmap_model,
                 'softargmax': lambda: T.build_softargmax_model(a.tau),
                 'coord': T.build_coord_model}[a.head]()
        sched = keras.optimizers.schedules.CosineDecay(lr, a.steps) \
            if a.cosine else lr
        model.compile(
            optimizer=keras.optimizers.Adam(sched, clipnorm=1.0),
            loss={'heatmap': lambda: T.heatmap_loss(a.pos_weight),
                  'softargmax': lambda: T.softargmax_loss_fn(a.coord_weight),
                  'coord': lambda: T.coord_loss_fn(a.coord_weight)}[a.head]())
        ds = T.make_ds(xtr, ytr, a.head, a.batch, True, False, 0)

        t0 = time.time()
        curve = []
        for done in range(0, a.steps, max(a.steps // 4, 1)):
            model.fit(ds, steps_per_epoch=max(a.steps // 4, 1), epochs=1,
                      verbose=0)
            curve.append(float(model.evaluate(val, verbose=0)))
        dt = time.time() - t0
        vl = curve[-1]

        pred = T.predict_labels(model, xva, a.head)
        ev = C.evaluate(pred, lva)['overall']
        ratio = vl / base if base else float('nan')
        # The loss is a weighted cross-entropy and is not comparable across
        # settings in absolute terms, so accuracy decides. A mean error near
        # 250 px means the decoder is returning the image centre, i.e. the
        # model has learned nothing; recall 0 means it predicts nothing at all.
        if ev['recall'] == 0:
            verdict = 'collapsed: predicts nothing'
        elif (ev['mean_px'] or 999) > 100:
            verdict = 'not learning: predictions are near the image centre'
        elif (ev['pck10'] or 0) > 0.3:
            verdict = 'learning'
        else:
            verdict = 'weak'
        print(f'lr {lr:<8g} loss {vl:.6f}   -> {verdict}')
        print(f'{"":<12}recall {ev["recall"]:.3f}  '
              f'PCK@10 {ev["pck10"]:.3f}  '
              f'mean {ev["mean_px"] if ev["mean_px"] else float("nan"):.1f} px'
              f'   [{dt/60:.1f} min]')
        print(f'{"":<12}curve {" -> ".join(f"{c:.4f}" for c in curve)}\n')
        results.append(dict(lr=lr, val_loss=vl, curve=curve,
                            ratio_to_zeros=ratio, seconds=dt,
                            recall=ev['recall'], pck10=ev['pck10'],
                            mean_px=ev['mean_px']))

    json.dump(dict(head=a.head, fraction=a.fraction, steps=a.steps,
                   cosine=a.cosine, zeros_baseline=base,
                   tau=a.tau, coord_weight=a.coord_weight, results=results),
              open(a.out, 'w'), indent=1)

    best = max(results, key=lambda r: (r['pck10'] or 0))
    print(f'best at this budget: lr={best["lr"]:g}, '
          f'PCK@10 {best["pck10"]:.3f}')
    if (best['pck10'] or 0) < 0.4:
        print('\nStill weak. Either the budget is too small -- rerun with '
              '--steps 3000 -- or the rate needs to go higher. Send me '
              f'{a.out} and I will read the curves.')
    print(f'saved {a.out}')


if __name__ == '__main__':
    main()
