# Scoring Targets

Future agents should optimize posterior sample quality on these experiments:

- `synthetic_d2`: synthetic posterior recovery with latent dimension 2.
- `synthetic_d5`: synthetic posterior recovery with latent dimension 5.
- `synthetic_d10`: synthetic posterior recovery with latent dimension 10.

For each experiment, `scripts/evaluate.sh` reports:

- `hellinger`: lower is better.
- `total_variation`: lower is better.
- `coordinate_marginal_tv`: lower is better; checks the average one-coordinate marginal across all latent coordinates so improvements cannot overfit only the displayed two-coordinate heatmap.
- `uses_likelihood_gradient`: must remain `0` for methods claiming the paper's gradient-free setting.

The generated artifact format is method-agnostic: write `samples.npz` under `checkpoints/<method>_synthetic_d<D>/` with a `samples` array of integer categorical states. Each run should also write `metadata.json` with `uses_likelihood_gradient` set to `0` or `1`; the evaluator requires this for methods other than the included SGDD and direct-MH implementations.

The included `direct_mh` baseline is a sanity check for the evaluator and is written under `checkpoints/direct_mh_synthetic_d<D>/`. It is not a paper Table 2 baseline; the paper baseline numbers remain in `scoring/reference.json`. Standard reproduction jobs score this sanity baseline in `scoring/sanity_scores.json`; manual evaluations can do the same with `INCLUDE_SANITY_BASELINES=1 SCORES_OUTPUT=scoring/sanity_scores.json`.
