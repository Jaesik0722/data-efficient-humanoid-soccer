# MCL simulation — Sections 5.2–5.5, Figures 6–8

Evaluates the Monte Carlo localization filter under controlled conditions.

The scripts import `mcl.py` and `field.py` **from the platform repository at run time**
rather than vendoring a copy, so the simulation is always evaluated against the filter
that is actually deployed on the robot and cannot drift from it.

## Regenerate the figures from the stored results

```bash
cd results && python3 ../make_figs2.py
```

Writes `fig_mcl_analysis.pdf` (Figure 7) and `fig_mcl_robustness.pdf` (Figure 8), and
prints every number quoted in Sections 5.3–5.5:

```
=== OBSERVABILITY ===   (the paper quotes the median column)
  FOV  30.0 deg | visible 1.68 | median  9.6 cm | mean 9.8 +- 2.5 cm
  FOV  69.0 deg | visible 3.53 | median  6.2 cm | mean 6.1 +- 1.0 cm
  FOV 360.0 deg | visible 19.00 | median  3.2 cm | mean 3.3 +- 0.1 cm
=== IMU ABLATION ===
  IMU=on  | pos 7.9 cm | heading 1.4 deg
  IMU=off | pos 9.5 cm | heading 78.6 deg
=== BASELINE ===
  MCL 7.1 +- 2.0 cm | trilateration 22.1 +- 7.0 cm | solvable 59.7% of frames
=== ODOMETRY NOISE ===
  sigma 0.02 | RMSE 5.7 +- 2.0 cm      sigma 0.20 | RMSE 35.3 +- 8.8 cm
=== UPDATE RATE ===
  10 Hz | RMSE 7.7 cm | 3.0 ms/update  30 Hz | RMSE 7.2 cm | 3.3 ms/update
=== KIDNAP RECOVERY ===
  recovered 8/8 runs | median 1.4 s | max 2.6 s
```

The observability sweep is quoted in the paper as the **median** over seeds, so both
summaries are printed; 6.2 (median) and 6.1 (mean) are the same result.

## Re-run the simulation

```bash
export ROBINION_REPO=/path/to/ros-robinion2
for fam in obs imu base odom rate kidnap; do
  for seed in 0 1 2 3 4 5; do python3 mcl_experiments.py $fam $seed; done
done
```

Six families, six seeds; results land in `parts/`. Roughly a minute per run.

| family | what it sweeps | figure |
|---|---|---|
| `obs` | camera field of view, 30°–360° | 7(a) |
| `imu` | IMU heading correction on/off | 7(b) |
| `base` | MCL vs least-squares trilateration | 7(c) |
| `odom` | odometry noise in the motion model | 8(a) |
| `rate` | filter update rate | 8(b) |
| `kidnap` | recovery from a kidnapped-robot event | 8(c) |

## Figure 6

```bash
cd results && python3 ../make_fig_loc_sim.py
```

Reads `noise_*.json`, `part_*.json` and `traj_*.json` for the accuracy panels, and
`timing_nuc8i7beh.json` for the per-update times in panel (b).

## Per-update timing

Accuracy is machine-independent here — the seeds are fixed, so a run reproduces anywhere.
The per-update time is not, and the paper reports it as evidence that the filter is
affordable next to two Edge TPU inferences **on the robot**. That claim is about the
robot's onboard computer, so the measurement is taken there:

```bash
export ROBINION_REPO=/path/to/ros-robinion2
python3 time_onboard.py --tag nuc8i7beh          # on the robot
python3 time_onboard.py --tag host-i7-9750H      # desktop, for comparison
```

Both files are in `results/`. Five particle counts, six seeds, σ = 0.1 m, 69°, 20 Hz.

| N | desktop i7-9750H | onboard i7-8559U |
|---:|---:|---:|
| 500 | 2.18 ± 0.30 | 2.64 ± 0.44 |
| 5,000 | 4.51 ± 0.47 | 4.50 ± 0.54 |
| 10,000 | 6.67 ± 0.52 | 6.55 ± 0.61 |

The two machines are indistinguishable at the particle counts that matter, so the cost of
the filter does not depend on which of them runs it.

**These times are higher than the `ms` fields stored in `part_*.json` and `parts/rate_*.json`,
by roughly half.** Those were recorded on the desktop machine at an earlier date, under a
Python and numpy build that is no longer reconstructible, and the script that produced
`part_*.json` was not kept. The figures reported in the paper are the ones measured with
`time_onboard.py`, which ships here and can be re-run on either machine.

## Note on the platform code

`mcl.py` calls `random()` without importing `random`. `mcl_experiments.py` binds
`numpy.random.random` before use rather than patching the platform repository; the issue
is listed in the paper's Limitations.
