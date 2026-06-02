# Scoring Targets

Future agents should optimize posterior sample quality on these experiments:

- `synthetic_d5`: synthetic posterior recovery with latent dimension 5.
- `synthetic_d10`: synthetic posterior recovery with latent dimension 10.

For each experiment, `scripts/evaluate.sh` reports:

- `hellinger`: lower is better.
- `total_variation`: lower is better.
- `uses_likelihood_gradient`: must remain `0` for methods claiming the paper's gradient-free setting.

The generated artifact format is fixed: write `samples.npz` under `checkpoints/sgdd_synthetic_d<D>/` with a `samples` array of integer categorical states.

The included `direct_mh` baseline is a sanity check for the evaluator and is written under `checkpoints/direct_mh_synthetic_d<D>/`. It is not a paper Table 2 baseline; the paper baseline numbers remain in `scoring/reference.json`. To include this sanity baseline in a manual evaluation run, set `INCLUDE_SANITY_BASELINES=1`.
