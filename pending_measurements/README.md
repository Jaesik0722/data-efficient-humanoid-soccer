# Measurements the paper reports as outstanding

**Nothing in this directory produced a number that appears in the paper.** These are
working tools for the measurements the Limitations section lists as not yet carried out.
They are included so that the future work is concrete rather than aspirational.

## Teleoperation latency (Limitations, third item)

The paper defines the end-to-end delay but does not report it, because measuring it needs
the complete streaming and control stack running, which the platform does not currently
permit. The tooling is ready:

| script | measures |
|---|---|
| `latency_ping_station.py` | command-path latency and clock offset, station side |
| `latency_echo_robot.py` | the responder that runs on the robot |
| `measure_stream.py` | video transport latency and frame rate, station side |
| `video_latency_frames.py` | glass-to-glass latency by frame counting from footage |
| `analyze_measurements.py` | turns the raw logs into paper-ready numbers and a figure |

## Surveyed-position validation (Limitations, first item)

Section 5.7 measures field accuracy against a hand-annotated reference. The stronger
experiment — placing the robot at surveyed positions and comparing — is what would
resolve the reference-frame offset that the hand annotation cannot separate from
estimator error.

| script | does |
|---|---|
| `capture_poses.py` | captures RGB-D frames at surveyed field poses (no ROS needed) |
| `offline_localize.py` | runs perception + localization offline on those frames |
| `analyze_static_test.py` | summarises the test and draws the figure |
