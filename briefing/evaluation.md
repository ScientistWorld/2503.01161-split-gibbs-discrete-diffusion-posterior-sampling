# Evaluation

Register each method as `scripts/samplers/NAME.sh`. Evaluation calls it from `/home/user` as:

```sh
bash scripts/samplers/NAME.sh PROBLEM_JSON OUTPUT_DIR
```

Write `OUTPUT_DIR/samples.npz`, containing an integer `samples` array of shape exactly `(128, 1024)`, with every token 0 or 1. Images are flattened in row order. Boolean, float, nonfinite, negative, oversized and empty outputs are invalid. Other output files and reported metrics are ignored. `data/image_problem.py` provides an optional public input loader and model interface.

Run `bash scripts/evaluate_train.sh` for development. It runs every registered method and writes `scoring/scores_train.json`. `bash scripts/evaluate.sh` additionally copies these numbers to `scoring/scores.json`. `bash scripts/evaluate_test.sh` takes no arguments, runs every registered method on unseen inputs, and writes `scoring/scores_test.json`. Only numbers under each experiment's `results` are ranked. Fresh evaluation always runs the procedure; saved samples can be rescored explicitly with `--rescore`. Public logs, inputs and outputs live under `checkpoints/train/METHOD/EXPERIMENT/`. Hidden run artifacts stay under `eval/test/results/runs/METHOD/EXPERIMENT/`; a hidden sampler can read its own run directory, while other hidden artifacts remain inaccessible.

The evaluator computes three quantities from the raw images:

- `psnr`: mean image PSNR in dB, data range 1. Exact recovery uses an MSE floor of one pixel, so the finite best value is 30.103 dB.
- `accuracy`: fraction for which a fixed, evaluator-owned classifier assigns the same class to reconstruction and underlying image. This is classifier consistency as measured in the paper, rather than accuracy against a method-supplied label. Both images use the same 0/128 classifier preprocessing.
- `measurement_error`: fraction of logical measurement bits violated. Lower is better; this measures data consistency independently of visual quality.

A run must reach PSNR ≥6 dB, classifier consistency ≥20%, and measurement error ≤10% in both experiments. A crash, malformed output, impossible number or failed gate makes every metric of every experiment its worst value: PSNR 0, accuracy 0, measurement error 1. Failure cannot be compensated by another metric, and no failed entry is left empty. The evaluator owns all inputs, targets, dimensions, thresholds and comparison models. Test targets are constructed in evaluator memory after inference; the inference process cannot read private file contents in `eval/test/`. Directory enumeration remains available for package discovery.

Rank improvement is the sum over experiments and metrics of `experiment_weight × abs(coefficient) × directed_improvement / population_std`. Directed improvement is positive in the metric's `higher_is_better` direction relative to the stored reference. The standard deviation is computed across the attempts being ranked on that metric; a zero-variance metric contributes zero. Missing or impossible entries first receive all worst values. Coefficient signs and measurement units do not change priorities. `eval/ranking.py` implements this calculation for audits; scalar audit gains are not additional ranked metrics.

All methods see the same 128 inputs and fixed sampling seeds. The public and hidden evaluators are independent implementations with different source partitions and private generation seeds. Inference uses fixed iteration counts, runs serially and has no evaluator wall-clock cutoff. Download data and weights before a job; hidden evaluation never trains a model.

For recovery from a lost worker, `--resume` explicitly reuses a saved raw array only when its input JSON matches and its shape, dtype and tokens validate; every metric is recomputed. It runs missing or malformed outputs. Use this option only when method code and resources are unchanged. Default evaluation runs every procedure afresh. Scores and completion flags never authorize reuse.
