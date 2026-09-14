**The labelled images and the demonstration video.** The 1896-image dataset and the
recording are too large to keep in the repository; both are available from the
corresponding author on request. Neither is needed to reproduce a figure — only to
retrain or re-annotate.

**Trained weights.** The `.tflite` and `.h5` files are not redistributed. `edge_tpu/`
documents what each script expects.

## Reproducibility notes

**Splits.** `de_common.py` writes the paths into `splits.json` exactly as they were given
on the command line. Passing a relative `--dataset` therefore produces a `splits.json`
that only resolves from the directory it was created in. Use an absolute path, or keep the
working directory fixed. `splits.json` is not committed; regenerating it from the same
dataset with the default seed reproduces the same split, which can be checked against the
counts recorded in any run file (120 / 298 / 597 / 1193 training, 211 validation, 492
held out).

**Standard deviations.** The label-efficiency table and figure report the sample standard
deviation over three seeds; the MCL figures report the population standard deviation over
six seeds. Each is internally consistent with the numbers quoted in its own section of the
paper.

## Citing and license

See [`CITATION.cff`](CITATION.cff). MIT — see [`LICENSE`](LICENSE); the platform
repository and the datasets are covered by their own terms.
[`CHANGELOG.md`](CHANGELOG.md) records what was added during peer review.
