#!/usr/bin/env python3
"""float32 vs int8: does post-training quantization change the detection metrics?

The label-efficiency study in this directory evaluates float32 models, whereas
the network that runs on the accelerator is the same architecture quantized to
int8. Reviewers reasonably ask whether the accuracy carries across. This script
answers that directly: it converts a trained model with full integer
post-training quantization, evaluates both versions on the same held-out
sessions with the same metric code (`de_common.evaluate`), and reports the
difference.

    python3 de_train.py --splits splits.json --head heatmap --fraction 25 \
            --seed 0 --outdir runs --save-model m_f25.keras
    python3 quantize_eval.py --model m_f25.keras --splits splits.json \
            --fraction 25 --out quant_f25.json

Add --compile to also run the Edge TPU compiler and capture its on-chip /
off-chip allocation report, which is what settles the parameter-caching
question rather than the model file size.
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import tensorflow as tf

import de_common as C
import de_train as T


def representative_dataset(x, n=200):
    """int8 calibration needs real inputs; a random subset of the training pool."""
    idx = np.random.RandomState(0).choice(len(x), min(n, len(x)), replace=False)

    def gen():
        for i in idx:
            yield [x[i:i + 1].astype(np.float32) / 255.0]
    return gen


def to_int8(model, x_calib, path):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset(x_calib)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    with open(path, 'wb') as fh:
        fh.write(blob)
    return path


def predict_tflite(path, x, head, threshold=0.30, min_score=1.0):
    """Same decoding as de_train.predict_labels, but through the int8 graph.

    The output tensor is dequantized with its own scale and zero point before
    the sigmoid, so the comparison is of the quantized weights and activations
    rather than of a different decoding rule.
    """
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    o_scale, o_zero = out['quantization']
    res = []
    for i in range(len(x)):
        interp.set_tensor(inp['index'], x[i:i + 1].astype(np.uint8))
        interp.invoke()
        p = interp.get_tensor(out['index'])[0].astype(np.float32)
        if o_scale:
            p = (p - o_zero) * o_scale
        if head == 'heatmap':
            res.append(C.decode_heatmap(1.0 / (1.0 + np.exp(-p)), threshold, min_score))
        else:
            d, vis = {}, 1.0 / (1.0 + np.exp(-p[:, 0]))
            xy = p[:, 1:]
            if head == 'coord':
                xy = 1.0 / (1.0 + np.exp(-xy))
            for c in range(C.N_CLASSES):
                if vis[c] >= 0.5:
                    d[c] = (float(xy[c, 0]), float(xy[c, 1]))
            res.append(d)
    return res


def compiler_report(tflite_path):
    """`edgetpu_compiler -s` prints the on-chip and off-chip parameter split.

    This is the measurement that decides whether a model's parameters are
    cached, which the file size alone does not.
    """
    try:
        r = subprocess.run(['edgetpu_compiler', '-s', tflite_path],
                           capture_output=True, text=True, timeout=600)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {'error': f'edgetpu_compiler not run: {e}'}
    txt = r.stdout + r.stderr
    grab = lambda pat: (re.search(pat, txt).group(1) if re.search(pat, txt) else None)
    return {
        'on_chip_used':   grab(r'On-chip memory used for caching model parameters:\s*(.+)'),
        'on_chip_remain': grab(r'On-chip memory remaining[^:]*:\s*(.+)'),
        'off_chip_used':  grab(r'Off-chip memory used for streaming uncached model parameters:\s*(.+)'),
        'ops_on_tpu':     grab(r'Number of operations that will run on Edge TPU:\s*(\d+)'),
        'ops_on_cpu':     grab(r'Number of operations that will run on CPU:\s*(\d+)'),
        'raw': txt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Keras model from --save-model')
    ap.add_argument('--splits', required=True)
    ap.add_argument('--head', default='heatmap',
                    choices=['heatmap', 'softargmax', 'coord'])
    ap.add_argument('--tflite', default=None, help='where to write the int8 model')
    ap.add_argument('--compile', action='store_true',
                    help='also run edgetpu_compiler -s and record the allocation')
    ap.add_argument('--fraction', default=None,
                    help="label budget used to train this model ('25', '100', ...); "
                         "its images calibrate the int8 conversion. Defaults to the "
                         "largest available subset.")
    ap.add_argument('--out', default='quantize_eval.json')
    a = ap.parse_args()

    sp = json.load(open(a.splits))
    model = tf.keras.models.load_model(a.model, compile=False)
    subsets = sp['subsets']
    frac = a.fraction or max(subsets, key=lambda k: len(subsets[k]))
    if frac not in subsets:
        raise SystemExit(f'--fraction {frac} not in splits.json; have {sorted(subsets)}')
    print(f'calibrating on the {frac}% subset')
    xtr, _, _ = T.load_pairs(subsets[frac], a.head)
    xte, _, lte = T.load_pairs(sp['test'], a.head)
    print(f'calibration pool {len(xtr)} images | test {len(xte)} images')

    tfl = a.tflite or os.path.splitext(a.model)[0] + '_int8.tflite'
    to_int8(model, xtr, tfl)
    print(f'wrote {tfl}  ({os.path.getsize(tfl)/1e6:.2f} MB)')

    rep = {'model': a.model, 'tflite': tfl, 'head': a.head,
           'tflite_bytes': os.path.getsize(tfl), 'n_test': len(xte)}
    pred_f = T.predict_labels(model, xte, a.head)
    pred_q = predict_tflite(tfl, xte, a.head)
    rep['float32'] = C.evaluate(pred_f, lte)['overall']
    rep['int8'] = C.evaluate(pred_q, lte)['overall']

    # Per-session breakdown. Reviewers ask whether the cross-session shortfall is
    # a property of both held-out sessions or driven by one of them; the answer
    # costs nothing here because the predictions are already in hand.
    sess = [os.path.basename(os.path.dirname(p)) for p in sp['test']]
    rep['per_session'] = {}
    for name in sorted(set(sess)):
        idx = [i for i, sn in enumerate(sess) if sn == name]
        rep['per_session'][name] = {
            'n_images': len(idx),
            'float32': C.evaluate([pred_f[i] for i in idx], [lte[i] for i in idx])['overall'],
            'int8': C.evaluate([pred_q[i] for i in idx], [lte[i] for i in idx])['overall'],
        }
    if a.compile:
        rep['edgetpu_compiler'] = compiler_report(tfl)

    print(f'\n{"metric":12s} {"float32":>9} {"int8":>9} {"delta":>9}')
    for k in ('recall', 'mean_px', 'median_px', 'pck5', 'pck10', 'pck20'):
        f, q = rep['float32'][k], rep['int8'][k]
        print(f'{k:12s} {f:9.3f} {q:9.3f} {q - f:+9.3f}')
    for lbl, d in (('float32', rep['float32']), ('int8', rep['int8'])):
        prec = d['n_det'] / (d['n_det'] + d['n_fp']) if d['n_det'] + d['n_fp'] else float('nan')
        print(f'{"precision":12s} {lbl:>9}: {prec:.3f}')
    if a.compile and 'error' not in rep.get('edgetpu_compiler', {}):
        c = rep['edgetpu_compiler']
        print(f"\non-chip used {c['on_chip_used']} | off-chip used {c['off_chip_used']}"
              f" | ops TPU/CPU {c['ops_on_tpu']}/{c['ops_on_cpu']}")

    print(f'\n{"session":24s} {"n":>5} {"recall":>8} {"PCK@10":>8}  (float32)')
    for name, d in rep['per_session'].items():
        o = d['float32']
        print(f'{name:24s} {d["n_images"]:5d} {o["recall"]:8.3f} {o["pck10"]:8.3f}')

    json.dump(rep, open(a.out, 'w'), indent=1)
    print(f'\nsaved {a.out}')


if __name__ == '__main__':
    main()
