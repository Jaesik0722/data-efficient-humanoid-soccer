#!/usr/bin/env bash
# Step 1 of the soft-argmax fairness experiment: give the soft-argmax head the
# same learning-rate probe that the heatmap and coordinate heads already had.
#
#   ./tune_softargmax.sh /path/to/splits.json
#
# Three rates x 800 steps on the full label set, seed 0 -- identical protocol
# to the probe reported in Section 5.1. About 10 minutes on the machine that
# produced the 36 runs in runs/.
#
# Writes lr_probe_softargmax.json. Nothing in runs/ is touched.
set -euo pipefail

SPLITS="${1:?usage: tune_softargmax.sh /path/to/splits.json}"
STEPS="${STEPS:-800}"
LRS="${LRS:-3e-4,1e-3,3e-3}"

python3 tune_lr.py --splits "$SPLITS" --head softargmax \
        --fraction 100 --steps "$STEPS" --lrs "$LRS" \
        --out lr_probe_softargmax.json

echo
echo "Done. Send lr_probe_softargmax.json."
echo "If the best rate is 1e-3, the paper can simply say all three heads were"
echo "probed identically. If it is not, step 2 becomes worth running."
