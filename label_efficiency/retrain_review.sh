#!/usr/bin/env bash
# Retrain the full-budget heatmap and soft-argmax runs, keeping the model this
# time. The 36 runs of the study were trained without --save-model, so their
# weights are gone and neither the per-session breakdown nor the matched
# false-detection comparison can be computed from the stored summaries.
#
# Six runs: 2 heads x 3 seeds, fraction 100, every other setting at the
# study's default. About 26 min each, so roughly 2.6 h in total.
#
#   bash retrain_review.sh                  # all six
#   bash retrain_review.sh heatmap 0        # just one, to check the setup
#
# Finished runs are skipped, so the script is safe to re-run after an
# interruption. Nothing is written into runs/ -- the outputs land in
# runs_review/ and models/ so that the 36 runs of the study stay untouched.
set -euo pipefail

SPLITS=${SPLITS:-splits.json}
OUTDIR=${OUTDIR:-runs_review}
MODELDIR=${MODELDIR:-models}
LOGDIR=${LOGDIR:-logs_review}

HEADS=(heatmap softargmax)
SEEDS=(0 1 2)
if [ $# -ge 1 ]; then HEADS=("$1"); fi
if [ $# -ge 2 ]; then SEEDS=("$2"); fi

mkdir -p "$OUTDIR" "$MODELDIR" "$LOGDIR"

if [ ! -f "$SPLITS" ]; then
  echo "no $SPLITS here. Run this from the label_efficiency directory, or"
  echo "set SPLITS=/path/to/splits.json. The paths inside splits.json are"
  echo "relative, so the dataset has to be reachable from the working"
  echo "directory -- a symlink next to splits.json is the usual fix."
  exit 1
fi

for head in "${HEADS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tag="${head}_f100_s${seed}"
    model="$MODELDIR/$tag.keras"
    if [ -f "$model" ] && [ -f "$OUTDIR/$tag.json" ]; then
      echo "== $tag already done, skipping"
      continue
    fi
    echo "== $tag  ($(date '+%H:%M:%S'))"
    python3 de_train.py \
      --splits "$SPLITS" \
      --head "$head" \
      --fraction 100 \
      --seed "$seed" \
      --outdir "$OUTDIR" \
      --save-model "$model" \
      --tf-verbose 0 \
      2>&1 | tee "$LOGDIR/$tag.log"
  done
done

echo
echo "done. models in $MODELDIR/, run summaries in $OUTDIR/"
ls -la "$MODELDIR"
