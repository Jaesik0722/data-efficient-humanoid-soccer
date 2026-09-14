#!/usr/bin/env bash
# Step 2 (optional) of the soft-argmax fairness experiment: search the two
# hyperparameters the paper admits were never searched -- the spatial-softmax
# temperature and the coordinate-loss weight -- select on validation, and
# re-evaluate the winner on the untouched test sessions.
#
#   ./grid_softargmax.sh /path/to/splits.json
#
# Two stages, because selecting on one seed and confirming on three is what
# keeps the cost to one night:
#
#   stage A  9 settings x 2 budgets x seed 0        = 18 runs   ~8 h
#   stage B  winning setting x 2 budgets x 3 seeds  =  6 runs   ~2.5 h
#
# Runs land in runs/ under names carrying their tau and weight, e.g.
# softargmax_f25_s0_t0.5_w20.json, so none of the 36 runs of the main study is
# overwritten and an interrupted sweep resumes where it stopped.
#
# Stage B is NOT run automatically: read stage A first, then call this script
# with WINNER set, e.g.   WINNER="0.5 20" ./grid_softargmax.sh splits.json
set -euo pipefail

SPLITS="${1:?usage: grid_softargmax.sh /path/to/splits.json}"
STEPS="${STEPS:-6000}"
OUTDIR="${OUTDIR:-runs}"
TAUS="${TAUS:-0.5 1.0 2.0}"
WEIGHTS="${WEIGHTS:-1 5 20}"
mkdir -p "$OUTDIR"

run() {   # run <tau> <weight> <fraction> <seed>
  local tau=$1 w=$2 frac=$3 seed=$4
  local tag="softargmax_f${frac}_s${seed}"
  [ "$tau" != "1.0" ] && tag="${tag}_t${tau}"
  [ "$w" != "5" ] && tag="${tag}_w${w}"
  if [ -f "$OUTDIR/$tag.json" ]; then
    echo "== skip $tag (already done)"; return
  fi
  echo "== $tag"
  python3 de_train.py --splits "$SPLITS" --head softargmax \
          --fraction "$frac" --seed "$seed" --steps "$STEPS" \
          --tau "$tau" --coord-weight "$w" --outdir "$OUTDIR" --tf-verbose 0
}

if [ -n "${WINNER:-}" ]; then
  read -r tau w <<<"$WINNER"
  echo "Stage B: confirming tau=$tau weight=$w on three seeds"
  for frac in 25 100; do for seed in 0 1 2; do run "$tau" "$w" "$frac" "$seed"; done; done
else
  echo "Stage A: 3 temperatures x 3 coordinate weights, seed 0, budgets 25 and 100"
  for tau in $TAUS; do for w in $WEIGHTS; do for frac in 25 100; do
    run "$tau" "$w" "$frac" 0
  done; done; done
  echo
  echo "Stage A complete. Pick the setting with the best *validation* PCK@10:"
  echo "    python3 read_grid.py --runs $OUTDIR"
  echo "then rerun with e.g.  WINNER=\"0.5 20\" $0 $SPLITS"
fi
