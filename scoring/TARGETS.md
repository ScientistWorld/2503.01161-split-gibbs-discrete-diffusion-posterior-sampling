# Scoring Targets

Future agents should optimize posterior sample quality on these experiments:

- `synthetic_d2`: synthetic posterior recovery with latent dimension 2.
- `synthetic_d5`: synthetic posterior recovery with latent dimension 5.
- `synthetic_d10`: synthetic posterior recovery with latent dimension 10.

For each experiment, `scripts/evaluate.sh` reports:

- `hellinger`: lower is better.
- `total_variation`: lower is better.
- `uses_likelihood_gradient`: must remain `0` for methods claiming the paper's gradient-free setting.

The generated artifact format is fixed: write `samples.npz` under `checkpoints/sgdd_synthetic_d<D>/` with a `samples` array of integer categorical states.
