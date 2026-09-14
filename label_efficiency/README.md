# Label efficiency — Section 5.1, Tables 2–3, Figure 5

How far the landmark detector's accuracy falls as labels are removed, for three output
representations trained under one common budget.

## Regenerate Table 3 and Figure 5 from the stored runs

```bash
python3 de_figs.py --runs results/runs --out .
```

```
runs found: heatmap=12, softargmax=12, coord=12
 heatmap PCK@10: 10%=0.225  25%=0.436  50%=0.548  100%=0.605
softargmax PCK@10: 10%=0.153  25%=0.257  50%=0.275  100%=0.257
   coord PCK@10: 10%=0.018  25%=0.041  50%=0.066  100%=0.073
wrote fig_label_efficiency.pdf, table_label_efficiency.tex, summary.json
```

`table_label_efficiency.tex` is the table that goes into the manuscript, unedited.
`summary.json` also carries the validation-to-held-out gap for all three heads, of which
the figure plots one.

## Regenerate Table 2 — the soft-argmax tuning checks

```bash
python3 read_probe.py results/tuning/lr_probe_softargmax_3000.json \
                      results/tuning/tau_probe_0.5_3000.json \
                      results/tuning/tau_probe_2.0_3000.json
```

The full-budget runs behind Table 2 are the three `softargmax_f100_s0_*.json` files in
`results/tuning/`; their baseline is `results/runs/softargmax_f100_s0.json`, the same
configuration at the same seed. See [`results/tuning/README.md`](results/tuning/README.md).

## Re-run the study

```bash
conda env create -f environment.yml && conda activate robinion-de
python3 check_env.py --dataset "/path/to/Localization Dataset"   # before committing 14 h
./run_all.sh "/path/to/Localization Dataset"
```

36 runs (3 heads × 4 label budgets × 3 seeds), about 14.5 h on one GPU.

A fresh run writes to `runs/`, which is kept separate from the shipped `results/runs/` so
that re-running never overwrites the results the paper was built on. `run_all.sh` skips
anything already present in its output directory, so an interrupted job restarts cleanly.
To compare a new run against the paper's:

```bash
python3 de_figs.py --runs runs         --out /tmp/new
python3 de_figs.py --runs results/runs --out /tmp/paper
```

## What the design controls for

**The splits are session-disjoint.** Two whole recording sessions are held out. Frames
from one session are near-duplicates of each other, so a random frame split would put
near-identical images on both sides and report a number that says nothing about a new
session.

**The label budgets are nested.** The 10 % subset is contained in the 25 % subset, and so
on, so the curves differ by which labels were added rather than by which happened to be
drawn.

**The budget is equal optimisation steps, not equal epochs.** Under equal epochs the 10 %
arm would receive a twelfth of the updates and the experiment would measure training
budget instead of label efficiency. Early stopping applies on top, so 6000 is a *maximum*
and each run's `steps_run` records what it actually used.

**All three heads share a backbone and an initialisation.** MobileNet-V2, ImageNet
weights, trained rather than frozen, 224×224 input. Only the output representation
differs:

| head | output | decoding |
|---|---|---|
| `heatmap` | 12 channels at 56×56 | threshold, flood fill, weighted centroid |
| `softargmax` | same, plus a spatial-softmax expectation | differentiable, integral regression |
| `coord` | global average pooling → dense | direct coordinate regression |

**The two coordinate-supervised heads share a loss.** `softargmax` and `coord` use the
same target encoding and the same objective, with the same coordinate weight of 5, so the
interval between them isolates the spatial decoder. Only `heatmap` differs in objective,
for the reason below.

## Tuning, and what was and was not searched

At the learning rate inherited from the platform's own schedule the heatmap head does not
train at all. The cause is not the rate itself: a sigmoid + MSE objective saturates on
targets that are 99.3 % background, so the gradient vanishes before anything is learned.
The released configuration uses **positive-weighted BCE on the logits** (`--pos-weight
100`), a cosine schedule from 1e-3, and gradient clipping.

`tune_lr.py` is the probe that established the rate. It trains each head at 3e-4, 1e-3 and
3e-3 for 800 steps on the full label set from a fixed seed, and judges by keypoint accuracy
rather than by the loss, whose scale is not comparable across heads. All three heads were
probed; 1e-3 wins for each.

Because 800 steps is short enough to rank an undertrained head incorrectly — and the
soft-argmax head is the slowest of the three to converge — the choice was re-examined for
that head at the full protocol, together with the softmax temperature of its decoder. The
`SoftArgmax2D` layer takes a `--tau`; `tau=1` is the unscaled softmax of the original
formulation and reproduces the earlier runs exactly. Results are Table 2 of the paper and
`results/tuning/`.

Two notes that the stored files make checkable:

* The 800-step and 3000-step probes disagree about the learning rate. At 800 steps 1e-3
  looks best; at 3000 steps 3e-4 does. At the full protocol — 6000 steps with the cosine
  schedule — 1e-3 wins clearly (0.733 against 0.497 validation PCK@10). The short probe is
  a screen, not a verdict.
* A lower validation loss does not imply better localization here. Both the 3e-4 and the
  `tau=0.5` configuration reach a lower best validation loss than the configuration used,
  and both localize worse. This is why the probe judges on accuracy.

The coordinate-loss weight was not searched; it is fixed at 5 for both coordinate-
supervised heads, so it is a property they share rather than an asymmetry between them.

**Within a run**, the evaluated checkpoint is the one with the lowest validation loss;
**across runs**, a setting is judged by that checkpoint's validation PCK@10. Whether
selecting each run's checkpoint by PCK instead would preserve the ordering was not tested:
per-checkpoint accuracy is not recorded during training, only per-checkpoint loss, so
answering it needs a retrain rather than a re-read of the stored runs.

## Files

| | |
|---|---|
| `de_common.py` | splits, heatmap encoding/decoding, PCK evaluation |
| `de_train.py` | the three heads, augmentation, training loop |
| `de_figs.py` | Figure 5 and Table 3 |
| `quantize_eval.py` | float32 vs int8 at a fixed checkpoint (Table 4) |
| `tune_lr.py` | learning-rate and temperature probe |
| `tune_softargmax.sh` | the soft-argmax learning-rate probe, one command |
| `grid_softargmax.sh` | optional τ × coordinate-weight sweep, resumable |
| `read_probe.py` | reads the probe files side by side |
| `read_grid.py` | summarises a sweep, selecting on validation only |
| `check_env.py` | verifies TensorFlow, Keras and CUDA before a long job |
| `run_all.sh` | the full grid, resumable |
| `environment.yml` | pinned; the Keras 2 / Keras 3 split makes the TF version load-bearing |
| `results/runs/` | the 36 result files |
| `results/tuning/` | the probes and full-budget checks behind Table 2 |
| `results/quantization/`, `results/retrain_seed0/` | Table 4 and the checkpoints it used |
| `results/summary.json` | aggregated |

One run, `softargmax_f10_s0`, was initially written by a short smoke test that
`run_all.sh` then skipped as complete. It was re-run at the full budget; the stored file
is the corrected one, and every run's `steps_budget` field can be checked against 6000.
