#!/usr/bin/env python3
"""Command-path latency measurement -- ROBOT SIDE (echo responder).

RUN ON: the robot. Start this BEFORE latency_ping_station.py on the station.
It simply echoes every ping back, stamped with the robot's receive and send
times, so the station can compute round-trip time and clock offset.

Usage:
    python3 latency_echo_robot.py
"""
import json, time
import rospy
from std_msgs.msg import String


class Echo:
    def __init__(self):
        self.pub = rospy.Publisher('/latency/pong', String, queue_size=10)
        rospy.Subscriber('/latency/ping', String, self._cb, queue_size=50)
        self.n = 0

    def _cb(self, msg):
        t1 = time.time()
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        t2 = time.time()
        self.pub.publish(String(json.dumps({'seq': d['seq'], 't1': t1, 't2': t2})))
        self.n += 1
        if self.n % 50 == 0:
            rospy.loginfo(f'echoed {self.n} pings')


if __name__ == '__main__':
    rospy.init_node('latency_echo_robot')
    Echo()
    rospy.loginfo('echo responder ready on /latency/ping -> /latency/pong')
    rospy.spin()
