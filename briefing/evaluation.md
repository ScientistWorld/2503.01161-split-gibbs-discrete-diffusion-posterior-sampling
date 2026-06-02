# Evaluation

The benchmark evaluates two active synthetic dimensions: `D=5` and `D=10`. For each dimension, a method writes integer samples to:

`checkpoints/sgdd_synthetic_d<D>/samples.npz`

The file must contain a `samples` array with shape `[num_samples, D]` and entries in `[0, 49]`. `scripts/evaluate.sh` calls only `eval/` code, computes the exact first-two-coordinate posterior marginal from the public problem definition in `data/`, and writes `scoring/scores.json`.

Metrics:

- `hellinger`: Hellinger distance between empirical and exact first-two-coordinate marginals. Lower is better.
- `total_variation`: Total variation distance between empirical and exact first-two-coordinate marginals. Lower is better.
- `uses_likelihood_gradient`: constraint metric indicating whether the method relies on likelihood gradients. The target method is gradient-free, so lower is better.

The main comparison is against the corresponding rows of the paper's Table 2 synthetic posterior-recovery results.
