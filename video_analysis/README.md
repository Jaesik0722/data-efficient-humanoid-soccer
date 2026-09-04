# Video analysis — Sections 5.6–5.7, Figures 9–10

Two measurements on the deployed robots, both recovered from a recorded demonstration in
which the localization visualiser happens to be in frame alongside the field. That makes
it possible to read out the pose each robot believed it occupied, frame by frame, without
instrumenting the robot.

* **Section 5.6** — how *stable* the estimate is (precision).
* **Section 5.7** — how *close* it is to the truth (accuracy), against a reference
  annotated by hand.

The recording itself is not redistributed here. Everything the analysis consumes — the
extracted tracks and the annotations — is in `results/`.

## Regenerate Figure 10 from the stored annotations

```bash
python3 gt_report.py --field results/field_points.json \
                     --robots results/robot_points.json \
                     --viz results/viz_tracks.json
```

```
landmarks                    10
reference_loo_m              0.0344       <- 3.4 cm; the gate, checked first
frames / n_pairs             15 / 30
median_m                     0.4565       <- 45.7 cm
ends_sign_disagreement       30/30        <- frame convention, from signs alone
moving_robot   median 31.0 cm | corr x 0.98 y 0.95 | after removing offset 11.3 cm
idle_robot     median 75.7 cm | true excursion 6.5 cm vs estimate wandering 2.5 m
```

## Two things this measurement does deliberately

**The reference is gated before the answer is looked at.** `gt_report.py` reports the
leave-one-out error of the homography first — each landmark predicted by a fit that
excluded it, which is the situation the robots are in — against a 5 cm threshold fixed in
advance. The fit residual on the points used in the fit is optimistic by construction and
is printed for reference only. Had the gate failed, the paper would have kept the existing
limitation.

**No frame convention is chosen by minimising the error.** Three conventions relate the
annotation to the visualiser, and each is fixed independently:

| convention | fixed by |
|---|---|
| which panel is which robot | read off the recording; hard-coded |
| sign of the panel's vertical axis | the renderer applies one *positive* scale to both axes, so `YFLIP = +1` follows from the code |
| orientation of the map vs the field | `field_ends_from_signs()` — compares only the **sign** of each robot's annotated *y* against the sign of its own estimate, while the robot stands in a goal area. They disagree in 30 of 30 observations, so the ends are interchanged (`rot = -1`). |

No error magnitude enters any of the three. The error under the opposite choice of ends
(160.7 cm, 3.5× larger) is printed afterwards as a consistency check only.

## Producing the inputs from a recording

```bash
python3 extract_viz.py VIDEO.mp4                  # -> viz_tracks.json  (Section 5.6)
python3 measure_localization.py                   # -> jitter.json, Figure 9

python3 gt_click.py --mode field --video VIDEO.mp4 --frame 200
python3 gt_overlay.py --field field_points.json --video VIDEO.mp4   # look at it
python3 gt_click.py --mode robots --video VIDEO.mp4 --n 20
```

`gt_click.py` refines every click through a 6× magnified inset, so the achievable
precision is about one pixel of the 1920×1080 frame. Keep `field_landmarks.png` open
while clicking — the camera looks *across* the field, so the halfway line runs
bottom-to-top in the image and the two goals are at the left and right edges. `s` skips a
landmark that is not visible, `u` goes back.

`gt_overlay.py` draws the field model back through the fitted homography onto the frame.
If the red lines sit on the painted lines across the whole frame, the clicks are
consistent; if they drift somewhere, a click near that area is wrong.

## Files

| | |
|---|---|
| `extract_viz.py` | inverts the visualiser's affine mapping using the two centre-line junctions |
| `measure_localization.py` | stability during stationary intervals (Figure 9) |
| `gt_click.py` | manual annotation, two passes |
| `gt_overlay.py` | visual check on the homography |
| `gt_report.py` | gate, conventions, error, Figure 10 |
| `field_landmarks.png` | landmark names and layout, for the annotator |
| `results/viz_tracks.json` | 258 recovered pose estimates per robot |
| `results/field_points.json` | the 10 clicked landmarks |
| `results/robot_points.json` | robot ground-contact points in 15 frames |
| `results/H_manual.npy` | the fitted reference homography |
| `results/absolute_gt_report.json` | every figure quoted in Section 5.7 |
