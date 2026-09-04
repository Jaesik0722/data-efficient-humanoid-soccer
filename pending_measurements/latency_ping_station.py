#!/usr/bin/env python3
"""Command-path latency and clock-offset measurement -- STATION SIDE.

RUN ON: the gamer station (the PC the player uses).
REQUIRES: latency_echo_robot.py running on the robot at the same time.

Uses the NTP round-trip algorithm, so NO clock synchronisation is needed:

    t0 = station sends ping
    t1 = robot receives ping      (robot clock)
    t2 = robot sends pong         (robot clock)
    t3 = station receives pong

    RTT    = (t3 - t0) - (t2 - t1)
    offset = ((t1 - t0) + (t2 - t3)) / 2       # robot_clock - station_clock

The one-way command latency over the WLAN is RTT/2. The offset is written to
the results file so that the video-latency measurement (which compares clocks
across the two machines) can be corrected.

Usage:
    python3 latency_ping_station.py [--count 300] [--rate 10] [--out cmd_latency.json]
"""
import argparse, json, time
import numpy as np
import rospy
from std_msgs.msg import String


class Pinger:
    def __init__(self, count, rate):
        self.count = count
        self.rate = rate
        self.samples = []
        self.pending = {}
        self.pub = rospy.Publisher('/latency/ping', String, queue_size=10)
        rospy.Subscriber('/latency/pong', String, self._pong_cb, queue_size=50)

    def _pong_cb(self, msg):
        t3 = time.time()
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        seq = d.get('seq')
        if seq not in self.pending:
            return
        t0 = self.pending.pop(seq)
        t1, t2 = d['t1'], d['t2']
        rtt = (t3 - t0) - (t2 - t1)
        offset = ((t1 - t0) + (t2 - t3)) / 2.0
        self.samples.append(dict(seq=seq, rtt_ms=rtt * 1000.0,
                                 oneway_ms=rtt * 500.0,
                                 offset_ms=offset * 1000.0,
                                 robot_proc_ms=(t2 - t1) * 1000.0))

    def run(self):
        r = rospy.Rate(self.rate)
        rospy.loginfo('waiting 2 s for the echo node to connect...')
        time.sleep(2.0)
        for seq in range(self.count):
            if rospy.is_shutdown():
                break
            t0 = time.time()
            self.pending[seq] = t0
            self.pub.publish(String(json.dumps({'seq': seq, 't0': t0})))
            if seq % 50 == 0:
                rospy.loginfo(f'sent {seq}/{self.count}, got {len(self.samples)} replies')
            r.sleep()
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=300)
    ap.add_argument('--rate', type=float, default=10.0)
    ap.add_argument('--out', default='cmd_latency.json')
    args, _ = ap.parse_known_args()

    rospy.init_node('latency_ping_station', anonymous=True)
    p = Pinger(args.count, args.rate)
    p.run()

    if not p.samples:
        print('ERROR: no replies received. Is latency_echo_robot.py running on the robot,')
        print('       and are ROS_MASTER_URI / ROS_IP set correctly on both machines?')
        return

    ow = np.array([s['oneway_ms'] for s in p.samples])
    rtt = np.array([s['rtt_ms'] for s in p.samples])
    off = np.array([s['offset_ms'] for s in p.samples])
    res = dict(
        n_sent=args.count, n_received=len(p.samples),
        loss_pct=100.0 * (1.0 - len(p.samples) / float(args.count)),
        oneway_ms=dict(mean=float(ow.mean()), std=float(ow.std()),
                       median=float(np.median(ow)),
                       p95=float(np.percentile(ow, 95)), max=float(ow.max())),
        rtt_ms=dict(mean=float(rtt.mean()), std=float(rtt.std()),
                    median=float(np.median(rtt))),
        clock_offset_ms=dict(mean=float(off.mean()), std=float(off.std()),
                             median=float(np.median(off))),
        samples=p.samples)
    json.dump(res, open(args.out, 'w'), indent=2)

    print('\n=== COMMAND-PATH LATENCY (station <-> robot) ===')
    print(f"received      : {len(p.samples)}/{args.count} "
          f"({res['loss_pct']:.1f}% loss)")
    print(f"one-way       : {ow.mean():.2f} +- {ow.std():.2f} ms "
          f"(median {np.median(ow):.2f}, p95 {np.percentile(ow, 95):.2f})")
    print(f"round trip    : {rtt.mean():.2f} +- {rtt.std():.2f} ms")
    print(f"clock offset  : {off.mean():.2f} ms  (robot clock - station clock)")
    print(f"saved {args.out}")


if __name__ == '__main__':
    main()
