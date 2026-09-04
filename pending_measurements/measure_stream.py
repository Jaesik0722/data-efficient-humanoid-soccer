#!/usr/bin/env python3
"""Video-stream transport latency and frame rate -- STATION SIDE.

RUN ON: the gamer station, while the robot is streaming (i.e. with the normal
camera pipeline running on the robot).

For every received frame it records
    transport delay = t_receive(station clock) - header.stamp(robot clock)
which still contains the clock offset between the two machines. Run
latency_ping_station.py first and pass its result with --offset-json so the
delay is corrected:

    corrected_delay = raw_delay + clock_offset      (offset = robot - station)

This measures capture -> encode -> network -> receive. It does NOT include
decode, rendering and monitor display lag, so it is a LOWER BOUND on the
glass-to-glass latency; use the photo method (see README) for the true
end-to-end figure and report both.

Usage:
    python3 measure_stream.py --topic /camera/color/image_raw/compressed \
        --seconds 60 --offset-json cmd_latency.json --out stream_latency.json
"""
import argparse, json, time
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, Image


class StreamMeter:
    def __init__(self, topic, msg_type):
        self.delays = []
        self.arrivals = []
        self.sizes = []
        rospy.Subscriber(topic, msg_type, self._cb, queue_size=1,
                         buff_size=2 ** 24)

    def _cb(self, msg):
        now = time.time()
        stamp = msg.header.stamp.to_sec()
        if stamp > 0:
            self.delays.append((now - stamp) * 1000.0)
        self.arrivals.append(now)
        data = getattr(msg, 'data', b'')
        self.sizes.append(len(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/camera/color/image_raw/compressed')
    ap.add_argument('--seconds', type=float, default=60.0)
    ap.add_argument('--offset-json', default=None,
                    help='cmd_latency.json produced by latency_ping_station.py')
    ap.add_argument('--out', default='stream_latency.json')
    args, _ = ap.parse_known_args()

    msg_type = CompressedImage if 'compressed' in args.topic else Image
    rospy.init_node('measure_stream', anonymous=True)
    m = StreamMeter(args.topic, msg_type)

    rospy.loginfo(f'listening on {args.topic} for {args.seconds:.0f} s ...')
    t_end = time.time() + args.seconds
    while time.time() < t_end and not rospy.is_shutdown():
        time.sleep(0.2)

    if len(m.arrivals) < 2:
        print(f'ERROR: no frames received on {args.topic}.')
        print('       Check the topic name with:  rostopic list | grep image')
        return

    offset_ms = 0.0
    if args.offset_json:
        try:
            offset_ms = json.load(open(args.offset_json))['clock_offset_ms']['median']
            print(f'applying clock offset correction: {offset_ms:+.2f} ms')
        except Exception as e:                          # noqa: BLE001
            print(f'[warn] could not read offset ({e}); reporting raw delays')

    raw = np.array(m.delays) if m.delays else np.array([np.nan])
    corr = raw + offset_ms
    iat = np.diff(np.array(m.arrivals)) * 1000.0
    fps = 1000.0 / np.mean(iat)
    sizes = np.array(m.sizes, dtype=float)

    res = dict(
        topic=args.topic, duration_s=args.seconds, frames=len(m.arrivals),
        fps=float(fps),
        interarrival_ms=dict(mean=float(iat.mean()), std=float(iat.std()),
                             p95=float(np.percentile(iat, 95)),
                             max=float(iat.max())),
        transport_ms_raw=dict(mean=float(np.nanmean(raw)),
                              std=float(np.nanstd(raw)),
                              median=float(np.nanmedian(raw))),
        clock_offset_applied_ms=offset_ms,
        transport_ms=dict(mean=float(np.nanmean(corr)),
                          std=float(np.nanstd(corr)),
                          median=float(np.nanmedian(corr)),
                          p95=float(np.nanpercentile(corr, 95))),
        frame_kb=dict(mean=float(sizes.mean() / 1024.0),
                      max=float(sizes.max() / 1024.0)),
        bitrate_mbps=float(sizes.sum() * 8 / args.seconds / 1e6),
        delays_ms=[float(v) for v in corr[:2000]])
    json.dump(res, open(args.out, 'w'), indent=2)

    print('\n=== VIDEO STREAM ===')
    print(f"frames received : {len(m.arrivals)} in {args.seconds:.0f} s")
    print(f"frame rate      : {fps:.1f} fps "
          f"(inter-arrival {iat.mean():.1f} +- {iat.std():.1f} ms, "
          f"p95 {np.percentile(iat, 95):.1f})")
    print(f"transport delay : {np.nanmean(corr):.1f} +- {np.nanstd(corr):.1f} ms "
          f"(median {np.nanmedian(corr):.1f}, p95 {np.nanpercentile(corr, 95):.1f})")
    print(f"frame size      : {sizes.mean()/1024:.1f} kB avg  ->  "
          f"{res['bitrate_mbps']:.2f} Mbit/s")
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
