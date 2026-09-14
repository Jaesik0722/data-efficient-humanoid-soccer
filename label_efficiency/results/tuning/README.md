# Soft-argmax tuning checks — Table 2

Raw output of the checks reported in Table 2 of the paper, plus the two short probes that
led to them. Nothing here entered the main comparison; these files exist so that the claim
"no tested alternative improved on the setting used" can be checked rather than taken on
trust.

Every file is a single seed (0) at the full label budget (1193 images). Settings were
compared on the **validation** split; the held-out sessions took no part in the selection.

## The full-protocol checks — the four rows of Table 2

| file | learning rate | τ | validation PCK@10 | held-out PCK@10 | best val. loss |
|---|---|---|---|---|---|
| `../runs/softargmax_f100_s0.json` | 1e-3 | 1 | 0.733 | 0.291 | 0.081 |
| `softargmax_f100_s0_lr0.0003.json` | 3e-4 | 1 | 0.497 | 0.124 | 0.078 |
| `softargmax_f100_s0_t0.5.json` | 1e-3 | 0.5 | 0.682 | 0.282 | 0.062 |
| `softargmax_f100_s0_t2.json` | 1e-3 | 2 | 0.594 | 0.203 | 0.081 |

The first row is the configuration used throughout the study, at the same seed, and lives
with the other 36 runs rather than here. Over three seeds it reaches 0.257 ± 0.032 PCK@10
on the held-out sessions, so held-out differences of that order are inside the run-to-run
spread.

Each was produced by one command, for example:

```bash
python3 ../../de_train.py --splits splits.json --head softargmax \
        --fraction 100 --seed 0 --steps 6000 --tau 0.5 --outdir .
```

`de_train.py` appends a suffix to the tag whenever τ, the coordinate weight or the
learning rate is not the default, so a sweep can never overwrite a run of the main study.

## The short probes

| file | budget | varied | note |
|---|---|---|---|
| `lr_probe_softargmax_800.json` | 800 steps | 3e-4 / 1e-3 / 3e-3 | the protocol the other two heads received; selects 1e-3 |
| `lr_probe_softargmax_3000.json` | 3000 steps | 3e-4 / 1e-3 / 3e-3 | selects 3e-4 — reversed by the full-protocol check above |
| `tau_probe_0.5_3000.json` | 3000 steps | τ = 0.5 at 1e-3 | |
| `tau_probe_2.0_3000.json` | 3000 steps | τ = 2 at 1e-3 | |

The probes run without the cosine schedule, which is why their absolute numbers are far
below the full runs and why the 3000-step ranking of learning rates does not survive. Read
them for shape, not for a decision; the decisions in Table 2 come from the full protocol.

```bash
python3 ../../read_probe.py lr_probe_softargmax_3000.json \
                            tau_probe_0.5_3000.json tau_probe_2.0_3000.json
```
