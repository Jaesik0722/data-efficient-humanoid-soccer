# float32 vs int8, and the per-session breakdown

Produced by `../../quantize_eval.py` from the seed-0 checkpoints retrained with
`--save-model` (see `../retrain_seed0/`). Behind Section 5.1.4 of the paper.

| | float32 | int8 | delta |
|---|---|---|---|
| 298 labels, PCK@10 | 0.445 | 0.443 | -0.003 |
| 298 labels, recall | 0.801 | 0.799 | -0.002 |
| 1193 labels, PCK@10 | 0.614 | 0.617 | +0.003 |
| 1193 labels, recall | 0.855 | 0.854 | -0.001 |

Per-session PCK@10 (float32):

| session | n | 298 labels | 1193 labels |
|---|---|---|---|
| Dataset1_Jeehyun | 98 | 0.364 | 0.639 |
| Dataset4 | 394 | 0.465 | 0.608 |

Reproduce:

    python3 de_train.py --splits splits.json --head heatmap --fraction 25 \
            --seed 0 --outdir runs_q --save-model m_f25.keras
    python3 quantize_eval.py --model m_f25.keras --splits splits.json \
            --fraction 25 --out quant_f25.json
