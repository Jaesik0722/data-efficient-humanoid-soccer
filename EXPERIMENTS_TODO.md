# Reviewer-requested experiments — both complete

## Experiment 1 — float32 vs int8 ✅

Retrained the 25 % and 100 % configurations with the checkpoint retained, converted to
int8 with each model's own training subset as the calibration set, and evaluated both on
the same held-out sessions with the same metric code.

| | float32 | int8 | Δ |
|---|---|---|---|
| 298 labels, PCK@10 | 0.445 | 0.443 | −0.003 |
| 298 labels, recall | 0.801 | 0.799 | −0.002 |
| 1193 labels, PCK@10 | 0.614 | 0.617 | +0.003 |
| 1193 labels, recall | 0.855 | 0.854 | −0.001 |

Every difference is an order of magnitude below the seed-to-seed spread. **The
label-efficiency result transfers to the deployed representation.** Now Section 5.1.4.

The same run gave the per-session breakdown a second reviewer asked for, at no extra cost:

| session | n | 298 labels | 1193 labels |
|---|---|---|---|
| Dataset1_Jeehyun | 98 | 0.364 | 0.639 |
| Dataset4 | 394 | 0.465 | 0.608 |

At the full budget the two sessions are close, so the cross-session shortfall is not the
artifact of one difficult recording. At the smaller budget they diverge.

Raw output: `label_efficiency/results/quantization/`

## Experiment 2 — Edge TPU parameter allocation ✅ (landmark network)

`edgetpu_compiler -s model_1000_epochs_quant.tflite`:

| | |
|---|---|
| On-chip used for parameters | **2.77 MiB** |
| On-chip remaining | **4.36 MiB** |
| Off-chip streamed | **5 KiB** (0.2 %) |
| Operations Edge TPU / CPU | **84 / 0** |

Three consequences, all now in Section 4.1.1 and the Conclusions:

1. The model is cached essentially in full — measured, not inferred from file size.
2. Memory available for parameters is **7.13 MiB**, not the nominal 8 MiB. This was the
   reviewer's specific objection.
3. **No CPU fallback**, so the 2.8× USB 2.0 slowdown of a model that streams almost
   nothing is isolated as the cost of moving input and output tensors.

The ball detector's own report could not be recovered: the deployed `.tflite` is already
compiled, the float32 source cannot be compiled, and the TFLite interpreter segfaults on
it. The paper says so, and notes that its 416×416 input is 3.4× the bytes of the landmark
network's, so its 8.2× slowdown cannot be attributed to parameter streaming alone.

## Experiment 3 — per-update filter timing on the robot ✅

A reviewer asked which machine produced the filter's per-update time, since the paper
attributed it to the onboard computer. It had in fact been measured on the desktop machine.
`time_onboard.py` now measures it with one script on either machine:

| N | desktop i7-9750H | onboard i7-8559U (NUC8i7BEH) |
|---:|---:|---:|
| 500 | 2.18 ± 0.30 | 2.64 ± 0.44 |
| 5,000 | 4.51 ± 0.47 | 4.50 ± 0.54 |
| 10,000 | 6.67 ± 0.52 | 6.55 ± 0.61 |

The two are indistinguishable where it matters, and 6.6 ms is 13 % of the 50 ms budget at
20 Hz, so the affordability claim now rests on a measurement taken on the deployed hardware.

**These are about 1.5× the times stored in `part_*.json`.** Those were recorded earlier on
the desktop machine and the script that produced them was not kept, so the discrepancy
cannot be traced. The paper reports the `time_onboard.py` figures.

Raw output: `mcl_simulation/results/timing_*.json`

---

## Nothing left to run.

Remaining items before submission are author checks, not experiments:

- ORCID identifiers (main.tex line 29)
- IJHR platform-paper citation (line 218), once accepted
- CRediT roles confirmed with co-authors (line 986)
- Publish `github.com/Jaesik0722/data-efficient-humanoid-soccer` — the Introduction
  footnote and the Data Availability statement both point at it
