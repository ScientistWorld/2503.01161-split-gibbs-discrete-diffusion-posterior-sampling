# Scoring Targets

Every experiment is scored on the hidden test by `bash scripts/evaluate_test.sh` (no arguments). It runs each sampler registered in `scripts/samplers/` on held-out instances with exactly 50,000 samples per instance.

| Experiment | Weight | What is sampled |
|---|---:|---|
| `synthetic_d2` | 0.20 | Table 2 task, `D=2`, coordinatewise observation |
| `synthetic_d5` | 0.20 | Table 2 task, `D=5`, coordinatewise observation |
| `synthetic_d10` | 0.25 | Table 2 task, `D=10`, coordinatewise observation |
| `synthetic_chain_d10` | 0.35 | `D=10`, pairwise distances between neighbouring coordinates (coupled, multimodal posterior) |

Metrics per experiment (lower is better; coefficient in parentheses):

- `hellinger` (-1.0, primary): first-two-coordinate posterior marginal, as in Table 2. Benefit metric.
- `total_variation` (-0.6): same marginal, as in Table 2. Benefit metric.
- `all_pairs_hellinger` (-0.5): mean over all coordinate pairs. Constraint metric: a method must sample the whole joint posterior, not trade the other coordinates for the reported pair.

Hard constraints, checked by the evaluator. Breaking any one fails the method on every experiment, with all distances set to 1.0:

- exactly `num_samples` samples of shape `[num_samples, D]`;
- integer dtype, tokens in `[0, 49]`;
- the sampler exits successfully on every instance.

Hard constraints checked outside the evaluator are listed in `scoring/CONSTRAINTS.md`. They include gradient-free sampling and no analytic posterior.

Reference values: `sgdd` (the paper's method at its synthetic settings) on the hidden test, recorded in `scoring/scores_test.json` and `scoring/reference.json`. The paper's Table 2 numbers (10,000 samples) are kept unchanged in `scoring/reference.json` under `_paper_*` keys. They are not on this gym's 50,000-sample scale.
