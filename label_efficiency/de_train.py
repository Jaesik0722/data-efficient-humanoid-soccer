#!/usr/bin/env python3
"""Train and evaluate one point of the label-efficiency curve.

One invocation = one (head, fraction, seed) run. Both heads share the same
ImageNet-initialised MobileNet-V2 backbone, the same data, the same
augmentation and the same optimisation budget, so any difference between the
curves is attributable to the output representation and not to the setup.

  heatmap     -- the deployed representation: a 56x56 Gaussian map per class,
                 decoded by taking the weighted centroid of the peak region.
  softargmax  -- the same spatial decoder, but the coordinate is the
                 expectation of a spatial softmax and is supervised directly,
                 with no target heatmap (integral regression).
  coord       -- the baseline: global pooling followed by a dense head that
                 regresses (visible, x, y) per class directly.

The three differ only in how a coordinate is produced from the backbone;
softargmax and coord additionally share a loss, so the gap between them
isolates the value of keeping the spatial map.

Optimisation is budgeted in *steps*, not epochs. With epochs, the 10% run
would receive a tenth of the gradient updates of the 100% run and the curve
would confound "less data" with "less training".

Usage:
    python3 de_train.py --splits splits.json --head heatmap --fraction 25 --seed 0
"""
import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

import de_common as C

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------- models

def build_backbone(trainable=True):
    inp = keras.Input(shape=(C.INPUT_SIZE, C.INPUT_SIZE, 3), dtype=tf.float32)
    x = keras.applications.mobilenet_v2.preprocess_input(inp * 255.0)
    base = keras.applications.MobileNetV2(
        alpha=1.0, include_top=False, weights='imagenet', input_tensor=x)
    base.trainable = trainable
    return inp, base


def build_heatmap_model():
    """MobileNet-V2 -> 7x7x1280 -> three bilinear upsamples -> 56x56x12."""
    inp, base = build_backbone()
    head = keras.Sequential([
        keras.layers.Conv2D(16, 1, padding='same', activation='relu'),
        keras.layers.UpSampling2D(2, interpolation='bilinear'),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.UpSampling2D(2, interpolation='bilinear'),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.UpSampling2D(2, interpolation='bilinear'),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        # Linear, not sigmoid: the model emits logits and the loss below
        # applies the sigmoid internally. Pairing a sigmoid with MSE, as the
        # original script does, makes the gradient vanish as soon as the
        # output saturates -- and with a target that is 99.3 % background it
        # saturates almost immediately, at either "all zero" or "all one".
        # That is the divergence seen in the learning-rate probe.
        keras.layers.Conv2D(C.N_CLASSES, 1, padding='same'),
    ], name='heatmap_head')
    return keras.Model(inp, head(base.output), name='heatmap')


def heatmap_loss(alpha=100.0):
    """Positive-weighted binary cross-entropy over the heatmap.

    Cross-entropy has gradient (sigma(z) - t), with no sigma'(z) factor, so it
    does not stall when the output saturates. The weight 1 + alpha*t
    counteracts the class imbalance: the mean target value is 0.0015, so with
    alpha = 100 the landmark pixels carry about 13 % of the loss instead of
    0.15 %. Without it the trivial "predict nothing" solution is a strong
    local optimum -- the lr = 3e-3 run fell into exactly that, reaching the
    all-zeros baseline with a recall of 0.
    """
    def loss(y_true, y_pred):
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                      logits=y_pred)
        w = 1.0 + alpha * y_true
        return tf.reduce_sum(bce * w) / tf.reduce_sum(w)
    return loss


class SoftArgmax2D(keras.layers.Layer):
    """Decode a per-class spatial logit map into a coordinate and a score.

    A spatial softmax turns each channel into a distribution over the output
    grid and the coordinate is its expectation, which is differentiable end to
    end -- the integral / soft-argmax formulation. Detection confidence is the
    log-partition of the same channel, rescaled by a learned per-class affine
    so that it can be trained against a visibility label; this adds 2 scalars
    per class and no spatial parameters.
    """

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.score_scale = self.add_weight(
            'score_scale', (c,), initializer='ones', trainable=True)
        self.score_bias = self.add_weight(
            'score_bias', (c,), initializer='zeros', trainable=True)

    def call(self, logits):
        h, w = int(logits.shape[1]), int(logits.shape[2])
        flat = tf.reshape(logits, (-1, h * w, int(logits.shape[-1])))
        p = tf.nn.softmax(flat, axis=1)

        ys = (tf.range(h, dtype=tf.float32) + 0.5) / float(h)
        xs = (tf.range(w, dtype=tf.float32) + 0.5) / float(w)
        gy, gx = tf.meshgrid(ys, xs, indexing='ij')
        gx = tf.reshape(gx, (1, h * w, 1))
        gy = tf.reshape(gy, (1, h * w, 1))

        x = tf.reduce_sum(p * gx, axis=1)
        y = tf.reduce_sum(p * gy, axis=1)
        vis = tf.reduce_logsumexp(flat, axis=1) * self.score_scale \
            + self.score_bias
        return tf.stack([vis, x, y], axis=-1)


def build_softargmax_model():
    """The intermediate representation: the heatmap decoder of the deployed
    network, but decoded by soft-argmax and supervised on coordinates rather
    than on a synthesised target map. Everything up to the final 1x1
    convolution is identical to `build_heatmap_model`."""
    inp, base = build_backbone()
    head = keras.Sequential([
        keras.layers.Conv2D(16, 1, padding='same', activation='relu'),
        keras.layers.UpSampling2D(2, interpolation='bilinear'),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.UpSampling2D(2, interpolation='bilinear'),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.UpSampling2D(2, interpolation='bilinear'),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(C.N_CLASSES, 1, padding='same'),
    ], name='softargmax_head')
    return keras.Model(inp, SoftArgmax2D()(head(base.output)),
                       name='softargmax')


def build_coord_model():
    """Same backbone, but the spatial map is pooled away and the coordinates
    are regressed. Output is (N_CLASSES, 3): visibility logit, x, y."""
    inp, base = build_backbone()
    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(C.N_CLASSES * 3)(x)
    out = keras.layers.Reshape((C.N_CLASSES, 3))(x)
    return keras.Model(inp, out, name='coord')


def coord_loss(y_true, y_pred):
    """Binary cross-entropy on visibility plus MSE on the coordinates of the
    landmarks that are actually present. Coordinates of absent landmarks carry
    no information and must not contribute a gradient."""
    vis_t = y_true[..., 0]
    vis_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(vis_t, y_pred[..., 0]))
    xy_err = tf.reduce_sum(tf.square(y_true[..., 1:] - tf.sigmoid(y_pred[..., 1:])),
                           axis=-1)
    denom = tf.maximum(tf.reduce_sum(vis_t), 1.0)
    xy_loss = tf.reduce_sum(xy_err * vis_t) / denom
    return vis_loss + 5.0 * xy_loss


def softargmax_loss(y_true, y_pred):
    """Identical in form and weighting to `coord_loss`; the only difference is
    that the soft-argmax already emits coordinates in [0, 1], so no squashing
    is applied. Keeping the loss the same is what makes the comparison between
    the two heads a comparison of representations."""
    vis_t = y_true[..., 0]
    vis_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(vis_t, y_pred[..., 0]))
    xy_err = tf.reduce_sum(tf.square(y_true[..., 1:] - y_pred[..., 1:]),
                           axis=-1)
    denom = tf.maximum(tf.reduce_sum(vis_t), 1.0)
    xy_loss = tf.reduce_sum(xy_err * vis_t) / denom
    return vis_loss + 5.0 * xy_loss


# ---------------------------------------------------------------- data

def encode_target(label, head):
    if head == 'heatmap':
        return C.make_heatmap(label)
    # 'coord' and 'softargmax' share the target encoding, which is what makes
    # their losses -- and therefore the comparison -- directly comparable.
    t = np.zeros((C.N_CLASSES, 3), np.float32)
    for c, (cx, cy) in label.items():
        t[c] = (1.0, cx, cy)
    return t


def load_pairs(paths, head):
    """Read every image and target into memory. The whole dataset is ~1900
    frames at 224x224, i.e. under 300 MB as float32 -- small enough that
    reading from disk every epoch would only add noise to the timing."""
    import cv2
    xs, ys, labels = [], [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (C.INPUT_SIZE, C.INPUT_SIZE),
                         interpolation=cv2.INTER_AREA)
        lab = C.load_label(os.path.splitext(p)[0] + '.txt')
        xs.append(img.astype(np.uint8))
        ys.append(encode_target(lab, head))
        labels.append(lab)
    return np.stack(xs), np.stack(ys), labels


def augment(x, y, head, flip=False):
    """Photometric augmentation, applied to a whole batch at once.

    The colour jitter of the original training script is reproduced with
    vectorised arithmetic rather than `tf.image.random_*`: the random factors
    have shape (B, 1, 1, 1), so each image still gets its own value, but the
    expensive per-example RGB<->HSV conversions are avoided. Those conversions
    were the data-pipeline bottleneck.

    Geometric augmentation is deliberately limited: a horizontal flip has to
    swap the left/right class pairs, and a crop or shift would have to move
    the targets as well.
    """
    b = tf.shape(x)[0]
    r = lambda lo, hi: tf.random.uniform((b, 1, 1, 1), lo, hi)

    x = x + r(-0.25, 0.25)                                  # brightness
    mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - mean) * r(0.75, 1.35) + mean                   # contrast
    grey = tf.reduce_sum(x * tf.constant([0.299, 0.587, 0.114]),
                         axis=-1, keepdims=True)
    x = grey + (x - grey) * r(0.70, 1.40)                   # saturation
    x = tf.clip_by_value(x, 0.0, 1.0)

    if flip:
        do = tf.random.uniform([]) < 0.5
        perm = tf.constant([C.MIRROR[i] for i in range(C.N_CLASSES)], tf.int32)
        if head == 'heatmap':
            y = tf.cond(do, lambda: tf.gather(tf.image.flip_left_right(y),
                                              perm, axis=-1), lambda: y)
        else:
            f = tf.gather(y, perm, axis=1)
            f = tf.stack([f[..., 0], 1.0 - f[..., 1], f[..., 2]], axis=-1)
            y = tf.cond(do, lambda: f, lambda: y)
        x = tf.cond(do, lambda: tf.image.flip_left_right(x), lambda: x)
    return x, y


def make_ds(x, y, head, batch, training, flip=False, seed=0):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b),
                num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(x), 2048), seed=seed,
                        reshuffle_each_iteration=True).repeat().batch(batch)
        ds = ds.map(lambda a, b: augment(a, b, head, flip),
                    num_parallel_calls=AUTOTUNE)
        return ds.prefetch(AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


# ---------------------------------------------------------------- predict

def predict_labels(model, x, head, batch=32, threshold=0.30, min_score=1.0):
    out = []
    for i in range(0, len(x), batch):
        p = model.predict(x[i:i + batch].astype(np.float32) / 255.0, verbose=0)
        if head == 'heatmap':
            p = 1.0 / (1.0 + np.exp(-p))      # the model emits logits
        for k in range(len(p)):
            if head == 'heatmap':
                out.append(C.decode_heatmap(p[k], threshold, min_score))
            else:
                d = {}
                vis = 1.0 / (1.0 + np.exp(-p[k][:, 0]))
                xy = p[k][:, 1:]
                if head == 'coord':          # squashed into [0, 1] by the loss
                    xy = 1.0 / (1.0 + np.exp(-xy))
                for c in range(C.N_CLASSES):
                    if vis[c] >= 0.5:
                        d[c] = (float(xy[c, 0]), float(xy[c, 1]))
                out.append(d)
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', required=True)
    ap.add_argument('--head', choices=['heatmap', 'softargmax', 'coord'],
                    default='heatmap')
    ap.add_argument('--fraction', default='100',
                    help='key into splits["subsets"]: 10, 25, 50 or 100')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=6000,
                    help='gradient steps; identical for every fraction')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--cosine', action='store_true', default=True,
                    help='cosine-decay the learning rate over the budget')
    ap.add_argument('--no-cosine', dest='cosine', action='store_false')
    ap.add_argument('--pos-weight', type=float, default=100.0,
                    help='alpha in the 1 + alpha*target loss weighting')
    ap.add_argument('--eval-every', type=int, default=200)
    ap.add_argument('--patience', type=int, default=10,
                    help='evaluations without improvement before stopping')
    ap.add_argument('--flip', action='store_true',
                    help='mirror augmentation with left/right class swap')
    ap.add_argument('--outdir', default='runs')
    ap.add_argument('--tf-verbose', type=int, default=1, choices=[0, 1, 2],
                    help="Keras progress bar: 1 shows it, 0 is silent. Leave "
                         "at 1 for the first run so you can see it is alive")
    a = ap.parse_args()

    tf.keras.utils.set_random_seed(a.seed)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)

    sp = C.load_splits(a.splits)
    if a.fraction not in sp['subsets']:
        raise SystemExit(f'fraction {a.fraction} not in {list(sp["subsets"])}')
    train_paths = sp['subsets'][a.fraction]

    tag = f'{a.head}_f{a.fraction}_s{a.seed}'
    os.makedirs(a.outdir, exist_ok=True)
    print(f'[{tag}] train {len(train_paths)}  val {len(sp["val"])}  '
          f'test {len(sp["test"])}')

    print(f'[{tag}] reading images...', flush=True)
    xtr, ytr, _ = load_pairs(train_paths, a.head)
    xva, yva, lva = load_pairs(sp['val'], a.head)
    xte, _, lte = load_pairs(sp['test'], a.head)
    print(f'[{tag}] building model (first run downloads the 9.4 MB ImageNet '
          f'weights once, then caches them)', flush=True)

    builders = {'heatmap': build_heatmap_model,
                'softargmax': build_softargmax_model,
                'coord': build_coord_model}
    losses = {'heatmap': lambda: heatmap_loss(a.pos_weight),
              'softargmax': lambda: softargmax_loss,
              'coord': lambda: coord_loss}
    model = builders[a.head]()
    loss = losses[a.head]()
    sched = keras.optimizers.schedules.CosineDecay(a.lr, a.steps) \
        if a.cosine else a.lr
    model.compile(optimizer=keras.optimizers.Adam(sched, clipnorm=1.0),
                  loss=loss)

    ds = make_ds(xtr, ytr, a.head, a.batch, True, a.flip, a.seed)
    val = make_ds(xva, yva, a.head, a.batch, False)

    print(f'[{tag}] training: {a.steps} steps, evaluating every '
          f'{a.eval_every}. The first chunk also traces the graph, so it is '
          f'slower than the rest.', flush=True)
    best, best_w, bad, history = np.inf, None, 0, []
    t0 = time.time()
    done = 0
    while done < a.steps:
        chunk = min(a.eval_every, a.steps - done)
        model.fit(ds, steps_per_epoch=chunk, epochs=1, verbose=a.tf_verbose)
        done += chunk
        vl = float(model.evaluate(val, verbose=0))
        history.append(dict(step=done, val_loss=vl))
        if len(history) == 1:
            rate = done / (time.time() - t0)
            print(f'[{tag}] {rate:.1f} steps/s -> about '
                  f'{a.steps / rate / 60:.0f} min for this run',
                  flush=True)
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_w = [w.copy() for w in model.get_weights()]
        else:
            bad += 1
        print(f'[{tag}] step {done:>5}/{a.steps}  val {vl:.5f}'
              f'{"  *" if bad == 0 else ""}')
        if bad >= a.patience:
            print(f'[{tag}] early stop at step {done}')
            break
    if best_w is not None:
        model.set_weights(best_w)
    train_time = time.time() - t0

    res = {}
    for name, x, lab in (('val', xva, lva), ('test', xte, lte)):
        res[name] = C.evaluate(predict_labels(model, x, a.head), lab)
        o = res[name]['overall']
        print(f'[{tag}] {name:>4}: recall {o["recall"]:.3f}  '
              f'mean {o["mean_px"]:.2f} px  PCK@10 {o["pck10"]:.3f}  '
              f'PCK@20 {o["pck20"]:.3f}')

    rec = dict(tag=tag, head=a.head, fraction=int(a.fraction), seed=a.seed,
               n_train=len(train_paths), n_val=len(sp['val']),
               n_test=len(sp['test']), steps_run=done, steps_budget=a.steps,
               best_val_loss=best, train_seconds=train_time,
               flip=a.flip, lr=a.lr, batch=a.batch, cosine=a.cosine,
               pos_weight=a.pos_weight, loss='weighted_bce_logits',
               test_sessions=sp['test_sessions'], history=history, results=res)
    path = os.path.join(a.outdir, tag + '.json')
    with open(path, 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(f'[{tag}] saved {path}  ({train_time / 60:.1f} min)')


if __name__ == '__main__':
    main()
