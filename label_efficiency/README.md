# Label efficiency — Section 5.1, Table 2, Figure 5

How far the landmark detector's accuracy falls as labels are removed, for three output
representations trained under one common budget.

## Regenerate Table 2 and Figure 5 from the stored runs

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

## Re-run the study

```bash
conda env create -f environment.yml && conda activate de
python3 check_env.py                       # do this before committing 14 hours
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
budget instead of label efficiency.

**All three heads share a backbone and an initialisation.** MobileNet-V2, ImageNet
weights, 224×224 input. Only the output representation differs:

| head | output | decoding |
|---|---|---|
| `heatmap` | 12 channels at 56×56 | threshold, flood fill, weighted centroid |
| `softargmax` | same, plus a spatial-softmax expectation | differentiable, integral regression |
| `coord` | global average pooling → dense | direct coordinate regression |

## The learning rate, and why `tune_lr.py` exists

At the learning rate inherited from the platform's own schedule the heatmap head does not
train at all. The cause is not the rate itself: a sigmoid + MSE objective saturates on
targets that are 99.3 % background, so the gradient vanishes before anything is learned.
The released configuration uses **positive-weighted BCE on the logits** (`--pos-weight
100`), a cosine schedule from 1e-3, and gradient clipping. `tune_lr.py` is the probe that
established this; its output is kept so the choice is auditable rather than asserted.

## Files

| | |
|---|---|
| `de_common.py` | splits, heatmap encoding/decoding, PCK evaluation |
| `de_train.py` | the three heads, augmentation, training loop |
| `de_figs.py` | Figure 5 and Table 2 |
| `tune_lr.py` | learning-rate probe |
| `check_env.py` | verifies TensorFlow, Keras and CUDA before a long job |
| `run_all.sh` | the full grid, resumable |
| `environment.yml` | pinned; the Keras 2 / Keras 3 split makes the TF version load-bearing |
| `results/runs/` | 36 result files |
| `results/summary.json` | aggregated |

One run, `softargmax_f10_s0`, was initially written by a short smoke test that
`run_all.sh` then skipped as complete. It was re-run at the full budget; the stored file
is the corrected one, and every run's `steps_budget` field can be checked against 6000.
