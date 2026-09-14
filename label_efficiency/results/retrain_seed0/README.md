# Reproduction check, seed 0

These two runs were retrained with `--save-model` so that the checkpoints could be
quantized (see `EXPERIMENTS_TODO.md`). They are kept because they double as a
reproduction check of the study: on a different day and a fresh GPU session,

| run | original | retrained |
|---|---|---|
| heatmap_f25_s0  PCK@10 | 0.430 | 0.445 |
| heatmap_f100_s0 PCK@10 | 0.636 | 0.614 |

Both differences are inside the seed-to-seed spread reported in the paper
(±0.005 at 25 %, ±0.030 at 100 %). The results in `../runs/` remain the ones the
paper reports; these are not used in any figure or table.
