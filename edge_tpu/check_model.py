#!/usr/bin/env python3
"""Check whether a .tflite model is actually compiled for the Edge TPU.

A model that was quantized but never passed through edgetpu_compiler will load
fine with the Edge TPU delegate and then run every operator on the CPU, which
typically costs an order of magnitude in speed. This tells you which case you
are in, and what to do about it.

Usage:
    python3 check_model.py model_a.tflite model_b.tflite ...
    python3 check_model.py /path/to/models/*.tflite
"""
import os
import sys


def describe(path):
    try:
        blob = open(path, 'rb').read()
    except IOError as e:
        print(f'{path}: cannot read ({e})')
        return None

    has_op = b'edgetpu-custom-op' in blob
    size_mb = len(blob) / 1e6

    shape = out_shape = None
    ops = {}
    try:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite            # type: ignore
        it = tflite.Interpreter(path)
        shape = [int(v) for v in it.get_input_details()[0]['shape']]
        out_shape = [[int(v) for v in d['shape']] for d in it.get_output_details()]
        try:                                            # private, best effort
            for d in it._get_ops_details():
                ops[d['op_name']] = ops.get(d['op_name'], 0) + 1
        except Exception:                               # noqa: BLE001
            pass
    except Exception as e:                              # noqa: BLE001
        print(f'  [warn] could not introspect: {str(e)[:70]}')

    print(f'\n=== {os.path.basename(path)} ===')
    print(f'  size            : {size_mb:.2f} MB')
    print(f'  input shape     : {shape}')
    print(f'  output shape(s) : {out_shape}')
    print(f'  edgetpu op      : {"YES" if has_op else "NO"}')
    if ops:
        total = sum(ops.values())
        tpu = ops.get('edgetpu-custom-op', 0)
        cpu = total - tpu
        print(f'  operators       : {total} total '
              f'({tpu} edgetpu-custom-op, {cpu} other)')
        if cpu and has_op:
            others = {k: v for k, v in ops.items() if k != 'edgetpu-custom-op'}
            print(f'                    partially compiled; on CPU: {others}')

    ONCHIP_MB = 8.0        # Edge TPU on-chip SRAM for cached model parameters
    if has_op and size_mb > ONCHIP_MB:
        print(f'  VERDICT         : compiled, but {size_mb:.2f} MB EXCEEDS the '
              f'{ONCHIP_MB:.0f} MB on-chip SRAM.')
        print('                    Parameters are streamed from host memory on every')
        print('                    inference, which typically costs ~10x in latency.')
        print('  next            : confirm the split with')
        print(f'                    edgetpu_compiler -s "{path}"')
        print('                    and look at "Off-chip memory used for streaming".')
    elif has_op:
        print(f'  VERDICT         : compiled and {size_mb:.2f} MB fits within the '
              f'{ONCHIP_MB:.0f} MB on-chip SRAM -> expect full speed')
    else:
        print('  VERDICT         : NOT compiled -> it will run entirely on the CPU')
        stem = path[:-7] if path.endswith('.tflite') else path
        cand = stem + '_edgetpu.tflite'
        if os.path.exists(cand):
            print(f'  next            : use the compiled file that already exists:')
            print(f'                    {cand}')
        else:
            print('  next            : compile it once, then benchmark the output:')
            print(f'                    edgetpu_compiler "{path}"')
            print(f'                    -> creates {os.path.basename(cand)}')
    return has_op


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    for p in paths:
        describe(p)
    print('\nAlso list the folder to see whether a compiled version exists:')
    print(f'  ls -la {os.path.dirname(os.path.abspath(paths[0])) or "."}')


if __name__ == '__main__':
    main()
