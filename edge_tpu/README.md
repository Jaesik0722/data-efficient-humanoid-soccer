# Edge TPU — inference timing and detection accuracy

Numbers reported in Section 3 of the paper. Reproducing these needs the physical Coral
accelerator and the quantised models; the stored results are in `results/`.

## Timing

```bash
python3 check_model.py model_tpu.tflite            # is it really compiled for the TPU?
python3 bench_edgetpu.py model_tpu.tflite --label landmark --runs 500 --out results/fps_landmark.json
python3 bench_edgetpu.py quant.tflite     --label ball     --runs 500 --out results/fps_ball.json
```

| model | input | per inference | paper |
|---|---|---|---|
| landmark heatmap | 224×224 | 5.18 ± 0.44 ms (193 fps) | Section 3 |
| ball detector | 416×416 | 19.71 ± 0.29 ms (51 fps) | Section 3 |

`check_model.py` is worth running first: a `.tflite` that was quantised but never passed
through the Edge TPU compiler still loads and still returns correct output, just an order
of magnitude slower, which is easy to miss in a benchmark.

## Detection accuracy

```bash
python3 eval_perception.py --dataset "/path/to/Localization Dataset" \
        --model model_quant.tflite --repo /path/to/ros-robinion2 \
        --out results/perception_corrected.json
```

Evaluates the deployed quantised model with the same decoding path the robot uses.
`resize_fix` in the platform repository changes this result and is discussed in the
paper's Limitations; the stored file records which setting produced it.
