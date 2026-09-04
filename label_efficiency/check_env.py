#!/usr/bin/env python3
"""Verify the environment before committing hours of GPU time.

Checks, in order, the things that actually go wrong: the GPU not being
visible to TensorFlow, a numpy/TF version clash, the three model heads failing
to build, a training step failing, the dataset path being wrong, and the
decoder disagreeing with the encoder. Then it times a few steps and
extrapolates the cost of the full study.

    python check_env.py --dataset "/path/to/Localization Dataset"
"""
import argparse
import sys
import time

FAIL = []


def ok(msg):
    print(f'  [ ok ] {msg}')


def bad(msg):
    print(f'  [FAIL] {msg}')
    FAIL.append(msg)


def warn(msg):
    print(f'  [warn] {msg}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', help='path to the "Localization Dataset" folder')
    ap.add_argument('--steps', type=int, default=20,
                    help='timed training steps per head')
    ap.add_argument('--budget', type=int, default=6000,
                    help='step budget assumed for the extrapolation')
    a = ap.parse_args()

    # ---------------------------------------------------------- versions
    print('\n1. Versions')
    import numpy as np
    print(f'  python {sys.version.split()[0]}, numpy {np.__version__}')
    try:
        import tensorflow as tf
    except ImportError as e:
        tf = None
        bad(f'tensorflow not importable: {e}')
        warn('continuing with the checks that do not need TensorFlow, so that '
             'the dataset can still be verified')
    if tf is not None:
        # tf.keras is a lazily-loaded shim in TF 2.15 and does not carry a
        # __version__; ask the real keras package, and fall back quietly.
        try:
            import keras as _keras
            kv = _keras.__version__
        except Exception:                                   # noqa: BLE001
            kv = getattr(tf.keras, '__version__', 'unknown')
        print(f'  tensorflow {tf.__version__}, keras {kv}')
        if int(np.__version__.split('.')[0]) >= 2 and tf.__version__ < '2.16':
            bad('numpy 2.x with TensorFlow < 2.16 -- pin numpy==1.26.4')
        else:
            ok('numpy / tensorflow versions are compatible')
        if tf.__version__ >= '2.16':
            warn('TF >= 2.16 uses Keras 3; this code was written against '
                 'Keras 2. If the model fails to build, use TF 2.15.1 as '
                 'pinned in environment.yml.')

    # ---------------------------------------------------------- GPU
    print('\n2. GPU')
    gpus = []
    if tf is None:
        warn('skipped (no TensorFlow)')
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            bad('no GPU visible to TensorFlow -- the full study needs days '
                'on CPU')
            print('         check: nvidia-smi, and that tensorflow[and-cuda] '
                  'was installed')
        else:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            ok(f'{len(gpus)} GPU(s): ' + ', '.join(g.name for g in gpus))
            try:
                d = tf.config.experimental.get_device_details(gpus[0])
                if d.get('device_name'):
                    print(f'         {d["device_name"]} '
                          f'(compute {d.get("compute_capability")})')
            except Exception:
                pass

    # ---------------------------------------------------------- encoder
    print('\n3. Heatmap encode/decode round trip')
    import de_common as C
    rng = np.random.RandomState(0)
    errs = []
    for _ in range(200):
        lab = {int(c): (rng.uniform(.05, .95), rng.uniform(.05, .95))
               for c in rng.choice(C.N_CLASSES, 5, replace=False)}
        dec = C.decode_heatmap(C.make_heatmap(lab))
        for c, (x, y) in lab.items():
            if c in dec:
                errs.append(np.hypot((dec[c][0] - x) * C.IMG_W,
                                     (dec[c][1] - y) * C.IMG_H))
    errs = np.array(errs)
    recall = len(errs) / 1000.0
    if recall > 0.99 and errs.mean() < 2.0:
        ok(f'recall {recall:.3f}, mean {errs.mean():.2f} px '
           f'(well below the ~8 px model error)')
    else:
        bad(f'round trip degraded: recall {recall:.3f}, '
            f'mean {errs.mean():.2f} px')

    # ---------------------------------------------------------- dataset
    print('\n4. Dataset')
    if not a.dataset:
        warn('no --dataset given; skipping. Pass it to verify the split.')
    else:
        try:
            sp = C.build_splits(a.dataset)
        except SystemExit as e:
            bad(str(e))
            sp = None
        except Exception as e:                              # noqa: BLE001
            bad(f'could not read the dataset -- {type(e).__name__}: {e}')
            sp = None
        if sp:
            c = sp['counts']
            ok(f'train pool {c["train_pool"]}, val {c["val"]}, '
               f'test {c["test"]}')
            print('         fractions: ' + ', '.join(
                f'{k}%={v}' for k, v in sorted(sp['fractions'].items(),
                                               key=lambda kv: int(kv[0]))))
            s = sp['subsets']
            nested = all(set(s[x]) <= set(s[y]) for x, y in
                         (('10', '25'), ('25', '50'), ('50', '100')))
            leak = (set(sp['test']) | set(sp['val'])) & set(s['100'])
            ok('label subsets are nested') if nested else \
                bad('label subsets are NOT nested')
            ok('no train/val/test leakage') if not leak else \
                bad(f'{len(leak)} images appear in both train and val/test')
            try:
                import cv2
            except ImportError as e:
                bad(f'opencv not importable: {e}')
                cv2 = None
            if cv2 is not None:
                n_img = sum(1 for p in s['10'][:20]
                            if cv2.imread(p) is not None)
                ok(f'read {n_img}/20 sample images with OpenCV') \
                    if n_img == 20 else \
                    bad(f'only {n_img}/20 sample images were readable')

    # ---------------------------------------------------------- models
    print('\n5. Model construction and one training step')
    timings = {}
    if tf is None:
        warn('skipped (no TensorFlow)')
        return summarise()
    import de_train as T
    builders = {'heatmap': T.build_heatmap_model,
                'softargmax': T.build_softargmax_model,
                'coord': T.build_coord_model}
    for head in ('heatmap', 'softargmax', 'coord'):
        try:
            model = builders[head]()
        except Exception as e:                              # noqa: BLE001
            bad(f'{head}: model failed to build -- {type(e).__name__}: {e}')
            continue
        n = model.count_params()
        loss = {'heatmap': T.heatmap_loss(),
                'softargmax': T.softargmax_loss,
                'coord': T.coord_loss}[head]
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss)

        x = rng.randint(0, 255, (a.steps * 2, C.INPUT_SIZE, C.INPUT_SIZE, 3),
                        dtype=np.uint8)
        labs = [{int(c): (rng.uniform(.1, .9), rng.uniform(.1, .9))
                 for c in rng.choice(C.N_CLASSES, 4, replace=False)}
                for _ in range(len(x))]
        y = np.stack([T.encode_target(l, head) for l in labs])
        ds = T.make_ds(x, y, head, 16, True, False, 0)

        try:
            model.fit(ds, steps_per_epoch=2, epochs=1, verbose=0)   # warm up
            t0 = time.time()
            model.fit(ds, steps_per_epoch=a.steps, epochs=1, verbose=0)
            dt = (time.time() - t0) / a.steps
        except Exception as e:                              # noqa: BLE001
            bad(f'{head}: training step failed -- {type(e).__name__}: {e}')
            continue
        timings[head] = dt
        ok(f'{head}: {n/1e6:.2f} M parameters, {dt*1000:.0f} ms/step')

        try:
            pred = T.predict_labels(model, x[:8], head)
            ok(f'{head}: decoder produced '
               f'{sum(len(p) for p in pred)} points from 8 images '
               f'(untrained, so any number is fine)')
        except Exception as e:                              # noqa: BLE001
            bad(f'{head}: prediction failed -- {type(e).__name__}: {e}')

    # ---------------------------------------------------------- estimate
    if timings:
        print('\n6. Projected cost of the full study (36 runs)')
        per_run = {h: t * a.budget for h, t in timings.items()}
        total = 12 * sum(per_run.values()) / 3600   # 4 fractions x 3 seeds
        for h, s in per_run.items():
            print(f'  {h:>8}: {s/60:5.1f} min per run at {a.budget} steps')
        print(f'  total   : {total:5.1f} h if no run stops early '
              f'(early stopping usually cuts this substantially)')
        if total > 24:
            warn('over a day of compute -- consider --steps 3000, or run the '
                 'heatmap arm first')

    return summarise()


def summarise():
    print()
    if FAIL:
        print(f'{len(FAIL)} problem(s) found:')
        for f in FAIL:
            print(f'  - {f}')
        print('\nFix these before running run_all.sh.')
        return 1
    print('Environment looks good. Next:\n'
          '  ./run_all.sh "/path/to/Localization Dataset"')
    return 0


if __name__ == '__main__':
    sys.exit(main())
