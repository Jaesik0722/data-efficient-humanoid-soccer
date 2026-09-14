# Changelog

What changed in this repository during peer review of the paper. Earlier entries are
listed first within each round.

## Round 3 — baseline tuning

**Added `--tau` and `--coord-weight` to `de_train.py`.** The `SoftArgmax2D` decoder
previously normalized its output map with an unscaled spatial softmax and exposed no
temperature at all. `tau=1` is that original behaviour and reproduces the stored runs
exactly; the parameter exists so that the temperature can be varied rather than only
discussed. `coord_loss` and `softargmax_loss` became factories parameterised by the
coordinate weight, with the fixed-weight versions kept under their old names.

**`tune_lr.py` accepts `--head softargmax`.** The probe previously covered the heatmap and
pooled-regression heads only. All three are now probed under one protocol.

**Added `results/tuning/`** — the probes and the three full-budget checks behind Table 2,
with a README relating each file to a row of that table.

**Added `tune_softargmax.sh`, `grid_softargmax.sh`, `read_probe.py`, `read_grid.py`** —
one-command wrappers for the probe and an optional τ × coordinate-weight sweep, and the
readers that summarise them. Selection in both readers is on validation only.

**Run tags carry non-default hyperparameters.** `softargmax_f100_s0_t0.5.json` and the
like, so a sweep cannot overwrite one of the 36 runs of the main study.

**`de_figs.py` reports precision**, computed from the stored detection counts, and records
the validation-to-held-out gap for all three heads rather than for the plotted one only.
Both are now quoted in the paper.

## Round 2 — timing and figure provenance

**Added `mcl_simulation/time_onboard.py`.** The filter's per-update time had been measured
on the desktop machine while the paper attributed it to the robot's onboard computer. The
script measures it with one code path on either machine; both result files are in
`mcl_simulation/results/`. The onboard figures are the ones the paper now reports.

**Added `mcl_simulation/make_fig_loc_sim.py`.** Figure 6 had no generating script in the
repository, and the top-level README claimed that `make_figs2.py` produced Figures 6–8
when it produces 7–8 only. Two defects in the new script were fixed before use: it
computed run-to-run spread with the sample standard deviation while the rest of the
simulation code and the quoted numbers use the population one, and it set the logarithmic
x-axis after plotting, which left the axis stretched over four empty decades.

**Added `label_efficiency/quantize_eval.py`** and `results/quantization/`, comparing
float32 against int8 at a fixed checkpoint. Those checkpoints are separate retrains and
differ from the 36-run study on every metric, so the two tables are not comparable row by
row; `results/retrain_seed0/` holds them.

**Added the Edge TPU compiler report** for the landmark network,
`edge_tpu/results/compiler_report_landmark.txt`. The ball detector's could not be
recovered: the deployed `.tflite` is already compiled and the float32 source does not
compile.

## Round 1 — release

First public version: the label-efficiency study, the MCL simulation, the video analysis,
the Edge TPU benchmarks, and the tooling for the measurements the paper lists as
outstanding.
