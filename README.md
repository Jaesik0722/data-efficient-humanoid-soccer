# Data-Efficient Visual Perception under Edge-Computing Constraints

Experiment and analysis code for

> **Data-Efficient Visual Perception under Edge-Computing Constraints: Application to
> Autonomous Localization and Humanoid Robot Soccer**
> Jaesik Jeong et al., *Electronics* (under review).

The result files are included, so **every figure in the paper regenerates in under a
minute — no GPU, no dataset, no robot.**

```bash
pip install -r requirements.txt

cd label_efficiency && python3 de_figs.py --runs results/runs --out .   # Table 2, Fig 5
cd ../mcl_simulation/results && python3 ../make_figs2.py                # Fig 6-8
cd ../../video_analysis && python3 gt_report.py \
    --field results/field_points.json --robots results/robot_points.json \
    --viz results/viz_tracks.json                                       # Fig 10
```

## Where each result comes from

| Paper | Command | Needs |
|---|---|---|
| Table 2, Figure 5 — label efficiency | `label_efficiency/de_figs.py` | stored results |
| the 36 training runs | `label_efficiency/run_all.sh` | GPU + dataset, ~14.5 h |
| Figures 6–8 — MCL simulation | `mcl_simulation/make_figs2.py` | stored results |
| the simulation runs | `mcl_simulation/mcl_experiments.py` | platform repo |
| Figure 9 — estimator stability | `video_analysis/measure_localization.py` | the recording |
| Figure 10 — field accuracy | `video_analysis/gt_report.py` | stored annotations |
| the annotations | `video_analysis/gt_click.py` | the recording |
| Section 3 — inference timing | `edge_tpu/bench_edgetpu.py` | Coral Edge TPU |
| teleoperation latency | *not measured* — see `pending_measurements/` |

Each directory has its own README with the expected output and the design notes.

## Layout

```
label_efficiency/     Section 5.1   three output heads under one training budget
mcl_simulation/       Sections 5.2-5.5   filter evaluation, 6 experiment families
video_analysis/       Sections 5.6-5.7   on-robot estimates recovered from a recording
edge_tpu/             Section 3     inference timing and detection accuracy
pending_measurements/ tooling for the measurements the paper lists as outstanding
```

## Not included here

**The robot platform.** The ROS stack, the MCL implementation and the visualiser live in
the platform repository:

```
https://github.com/<user>/ros-robinion2
```

`mcl_simulation/` imports `mcl.py` from there at run time (`export ROBINION_REPO=...`)
rather than vendoring a copy, so the simulation always evaluates the deployed filter. Two
known issues in that repository affect results reported here and are described in the
paper's Limitations: the `resize_fix` path in `utils.py`, and a missing `random` import in
`mcl.py` that `mcl_experiments.py` works around.

**The labelled images and the demonstration video.** The 1193-image dataset and the
recording are not redistributed; the recording belongs to a third party. Both are
available from the corresponding author on request. Neither is needed to reproduce a
figure — only to retrain or re-annotate.
