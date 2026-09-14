# Edge TPU — inference timing and detection accuracy

Numbers reported in Section 4.1.1 of the paper. Reproducing these needs the physical Coral
accelerator and the quantised models; the stored results are in `results/`.

## Timing

```bash
python3 check_model.py model_tpu.tflite            # is it really compiled for the TPU?
python3 bench_edgetpu.py model_tpu.tflite --label landmark --runs 500 --out results/fps_landmark.json
python3 bench_edgetpu.py quant.tflite     --label ball     --runs 500 --out results/fps_ball.json
```

| model | input | per inference |
|---|---|---|
| landmark heatmap | 224×224 | 5.18 ± 0.44 ms (193 fps) |
| ball detector | 416×416 | 19.71 ± 0.29 ms (51 fps) |

Each model is benchmarked on its own accelerator, one at a time. The paper's 48 fps is the
slower of the two and therefore assumes the pair runs concurrently, which follows from the
separate accelerators but was not measured with both executing together.

`check_model.py` is worth running first: a `.tflite` that was quantised but never passed
through the Edge TPU compiler still loads and still returns correct output, just an order
of magnitude slower, which is easy to miss in a benchmark.

## Parameter allocation

`results/compiler_report_landmark.txt` is the `edgetpu_compiler -s` report for the
landmark network: 2.77 MiB of on-chip memory used for parameters against 4.36 MiB
remaining, 5 KiB streamed off-chip, and no operation falling back to the CPU. It is quoted
in the paper as evidence that the network's bandwidth sensitivity is not parameter
streaming.

The ball detector has no counterpart here. Its deployed `.tflite` is already compiled, so
the compiler will not re-report it, and the float32 source does not compile; the paper says
so rather than inferring the allocation from file size.

## Detection accuracy

```bash
python3 eval_perception.py --dataset "/path/to/Localization Dataset" \
        --model model_quant.tflite --repo /path/to/ros-robinion2 \
        --out results/perception_corrected.json
```

Evaluates the deployed quantised model with the same decoding path the robot uses.
`resize_fix` in the platform repository changes this result and is discussed in the
paper's Limitations; the stored file records which setting produced it.
