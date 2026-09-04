#!/usr/bin/env bash
# Full label-efficiency study: 3 heads x 4 label fractions x 3 seeds = 36 runs.
#
#   ./run_all.sh "/path/to/Localization Dataset"
#
# Output goes to runs/ by default. The results the paper was built on live in
# results/runs/ and are never touched, so a fresh run cannot overwrite them.
#
# Runs are independent; if the machine is interrupted, re-running the script
# skips whatever is already in runs/ and continues.
set -euo pipefail

DATASET="${1:?usage: run_all.sh /path/to/Localization\ Dataset}"
STEPS="${STEPS:-6000}"
OUTDIR="${OUTDIR:-runs}"

python3 de_common.py --dataset "$DATASET" --out splits.json

mkdir -p "$OUTDIR"
for head in heatmap softargmax coord; do
  for frac in 10 25 50 100; do
    for seed in 0 1 2; do
      tag="${head}_f${frac}_s${seed}"
      if [ -f "$OUTDIR/$tag.json" ]; then
        echo "== skip $tag (already done)"
        continue
      fi
      echo "== $tag"
      python3 de_train.py --splits splits.json --head "$head" \
              --fraction "$frac" --seed "$seed" --steps "$STEPS" \
              --outdir "$OUTDIR"
    done
  done
done

python3 de_figs.py --runs "$OUTDIR" --out .
echo
echo "Done. Send me the whole $OUTDIR/ directory (36 small JSON files)."
