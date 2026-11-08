# Constraints

## Fixed problem (owned by `eval/`)

- The prior (discretized Gaussian, std 2, grid `[-5, 5]`, 50 tokens), the forward models, `sigma = 3`, the observations, the seeds and the sample count (50,000 per instance) are defined in `eval/train/` and `eval/test/`. Methods receive them through the problem file.
- `data/synthetic_sgdd.py` only parses the problem file for samplers. Editing it changes what your sampler sees, not the target it is scored against.
- The test instances are hidden. Samplers are run on them at test time, and outputs from earlier runs are never reused.

## Output validity (enforced; any violation fails the method on every experiment)

- `samples.npz` with an integer array `samples` of shape exactly `[num_samples, dim]`;
- every token in `[0, 49]`;
- the sampler exits with status 0 on every instance, within the time limit.

## Eligibility rules (checked by review of the method, not by the evaluator)

- **Gradient-free.** The paper's claim is posterior sampling without likelihood gradients. Do not use gradients of the likelihood with respect to a continuous relaxation of the tokens. A metadata flag is not scored and does not satisfy this rule.
- **No analytic posterior.** Use the prior only through its diffusion-model interface (the noised marginals / concrete score), and the likelihood only as a black-box `log p(y|x)` on full token sequences. Do not compute posterior marginals or normalizers in closed form and sample from them, for example by exploiting the factorization of the coordinatewise likelihood or by a dynamic program over the L1 sum. Exact-posterior samples score at the bound but are not a sampling method.
- **No target leakage.** Do not read `eval/`, score files, or other methods' outputs inside a sampler.
- **i.i.d.-style samples.** Do not deterministically allocate tokens to match histogram cells (stratified or quota tricks). The metric assumes posterior draws.
- Use the seed in the problem file, so runs are reproducible and comparisons are paired.
