#!/usr/bin/env python3
"""Turn the raw robot measurements into paper-ready numbers and a figure.

RUN ON: any machine (no ROS needed). Put all measurement files in one folder.

Expected inputs (whichever exist):
    fps_landmark.json     from bench_edgetpu.py
    fps_ball.json         from bench_edgetpu.py
    cmd_latency.json      from latency_ping_station.py
    stream_latency.json   from measure_stream.py
    g2g.csv               glass-to-glass photo readings, one per line:
                          camera_display_ms,station_display_ms

Usage:
    python3 analyze_measurements.py [--dir .] [--out fig_latency.pdf]
"""
import argparse, glob, json, os
import numpy as np


def load(path):
    try:
        return json.load(open(path))
    except (IOError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='.')
    ap.add_argument('--out', default='fig_latency.pdf')
    args = ap.parse_args()
    d = args.dir

    fps_files = sorted(glob.glob(os.path.join(d, 'fps_*.json')))
    fps = [load(f) for f in fps_files]
    fps = [f for f in fps if f]
    cmd = load(os.path.join(d, 'cmd_latency.json'))
    stream = load(os.path.join(d, 'stream_latency.json'))

    g2g = None
    g2g_path = os.path.join(d, 'g2g.csv')
    if os.path.exists(g2g_path):
        vals = []
        for line in open(g2g_path):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p for p in line.replace(',', ' ').split() if p]
            if len(parts) >= 2:
                a, b = float(parts[0]), float(parts[1])
                delta = a - b
                if delta < 0:            # timer wrapped at 10 s
                    delta += 10000.0
                vals.append(delta)
        if vals:
            g2g = np.array(vals)

    print('=' * 62)
    print('MEASUREMENT SUMMARY'.center(62))
    print('=' * 62)

    if fps:
        print('\n--- Edge TPU inference ---')
        for f in fps:
            print(f"  {f['label']:<10s} {f['inference_ms']['mean']:6.2f} +- "
                  f"{f['inference_ms']['std']:4.2f} ms  "
                  f"({f['fps_inference']:6.1f} fps inference, "
                  f"{f['fps_pipeline']:6.1f} fps pipeline)  [{f['backend']}]")
        tot = sum(f['pipeline_ms']['mean'] for f in fps)
        print(f"  combined perception pipeline: {tot:.2f} ms -> {1000.0/tot:.1f} fps "
              f"if run sequentially on one accelerator")

    if stream:
        print('\n--- Video stream ---')
        print(f"  frame rate      : {stream['fps']:.1f} fps")
        print(f"  transport delay : {stream['transport_ms']['mean']:.1f} +- "
              f"{stream['transport_ms']['std']:.1f} ms "
              f"(p95 {stream['transport_ms']['p95']:.1f})")
        print(f"  bandwidth       : {stream['bitrate_mbps']:.2f} Mbit/s "
              f"({stream['frame_kb']['mean']:.1f} kB/frame)")

    if cmd:
        print('\n--- Command path ---')
        print(f"  one-way latency : {cmd['oneway_ms']['mean']:.2f} +- "
              f"{cmd['oneway_ms']['std']:.2f} ms "
              f"(p95 {cmd['oneway_ms']['p95']:.2f})")
        print(f"  packet loss     : {cmd['loss_pct']:.1f}%")
        print(f"  clock offset    : {cmd['clock_offset_ms']['median']:.2f} ms")

    if g2g is not None:
        print('\n--- Glass-to-glass (photo method) ---')
        print(f"  n = {len(g2g)}  {g2g.mean():.1f} +- {g2g.std():.1f} ms "
              f"(median {np.median(g2g):.1f}, min {g2g.min():.1f}, max {g2g.max():.1f})")

    if g2g is not None and cmd:
        loop = g2g.mean() + cmd['oneway_ms']['mean']
        print(f"\n  >>> closed-loop delay T_loop = {loop:.1f} ms "
              f"(video {g2g.mean():.1f} + command {cmd['oneway_ms']['mean']:.1f})")

    # ------------------------------------------------------------------ figure
    panels = sum(x is not None for x in
                 [g2g, cmd, stream if stream else None]) + (1 if fps else 0)
    if panels == 0:
        print('\nno data to plot')
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif',
                         'axes.linewidth': 0.6, 'figure.dpi': 200})

    fig, axs = plt.subplots(1, panels, figsize=(2.4 * panels, 2.35))
    if panels == 1:
        axs = [axs]
    i = 0

    if g2g is not None:
        a = axs[i]; i += 1
        a.hist(g2g, bins=min(20, max(5, len(g2g) // 3)), color='tab:blue',
               alpha=0.75, edgecolor='white', linewidth=0.4)
        a.axvline(g2g.mean(), color='tab:red', ls='--', lw=1,
                  label=f'mean {g2g.mean():.0f} ms')
        a.set_xlabel('Glass-to-glass latency (ms)'); a.set_ylabel('Count')
        a.set_title(f'(a) Video latency (n={len(g2g)})', fontsize=8)
        a.legend(fontsize=6.5)

    if cmd is not None:
        a = axs[i]; i += 1
        ow = np.array([s['oneway_ms'] for s in cmd['samples']])
        xs = np.sort(ow); ys = np.arange(1, len(xs) + 1) / len(xs) * 100
        a.plot(xs, ys, color='tab:green')
        a.axvline(np.percentile(ow, 95), ls='--', lw=0.9, color='gray',
                  label=f'p95 {np.percentile(ow, 95):.1f} ms')
        a.set_xlabel('Command latency (ms)'); a.set_ylabel('CDF (%)')
        a.set_title(f'(b) Command path (n={len(ow)})', fontsize=8)
        a.legend(fontsize=6.5, loc='lower right')

    if stream is not None:
        a = axs[i]; i += 1
        if stream.get('delays_ms'):
            v = np.array(stream['delays_ms'])
            a.hist(v, bins=25, color='tab:orange', alpha=0.75,
                   edgecolor='white', linewidth=0.4)
            a.set_xlabel('Transport delay (ms)'); a.set_ylabel('Count')
        a.set_title(f"(c) Stream: {stream['fps']:.0f} fps", fontsize=8)

    if fps:
        a = axs[i]; i += 1
        labels = [f['label'] for f in fps]
        inf = [f['inference_ms']['mean'] for f in fps]
        pipe = [f['pipeline_ms']['mean'] - f['inference_ms']['mean'] for f in fps]
        x = np.arange(len(labels))
        a.bar(x, inf, 0.55, label='Inference', color='tab:blue')
        a.bar(x, pipe, 0.55, bottom=inf, label='Pre/post', color='tab:cyan')
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=7)
        a.set_ylabel('Time per frame (ms)')
        a.set_title('(d) Edge TPU inference', fontsize=8)
        a.legend(fontsize=6.5)

    fig.tight_layout()
    fig.savefig(os.path.join(d, args.out), bbox_inches='tight')
    fig.savefig(os.path.join(d, args.out.replace('.pdf', '.png')),
                bbox_inches='tight', dpi=200)
    print(f"\nsaved {args.out}")


if __name__ == '__main__':
    main()
