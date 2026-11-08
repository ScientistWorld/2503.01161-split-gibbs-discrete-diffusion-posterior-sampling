# Method

The paper proposes Split Gibbs Discrete Diffusion Posterior Sampling (SGDD). It samples a discrete posterior by augmenting the latent variable with an auxiliary state `z`, coupled to `x` through a Hamming-distance potential (Eq. 10), and alternates two conditional updates under an annealed coupling level `eta_k`:

- a likelihood step (Eq. 13): `T` Metropolis-Hastings steps on `z` targeting `p(y|z) exp(-D(x, z; eta_k))`, using single-site uniform proposals;
- a prior step (Eq. 11): `x ~ p_0(x) (beta/(1-beta))^d(x, z)`. This is sampled by running the reverse uniform-kernel discrete diffusion from `sigma = eta_k` down to 0 with an `H`-step Euler sampler and the closed-form concrete score of the synthetic prior.

The noise level follows the paper's geometric schedule from `eta_max = 20` to `eta_min = 1e-4`. Settings are the paper's synthetic values (Appendix C.3, Table 7): `K = 10` iterations, `T = 10` MH steps and `H = 20` Euler steps.

Implementation: `method/sgdd_synthetic.py`, registered as `scripts/samplers/sgdd.sh`. In each Euler (sub)step every token jumps at most once, using rates computed from the state at the start of the step. Sub-steps are added only when a jump probability would exceed 0.95. An exact Eq. 11 prior sampler (`--prior-sampler exact`) is kept for diagnostics only.

Comparators, all registered under `scripts/samplers/`:

- `svdd_pm` and `smc` (`baseline/paper_baselines.py`): the paper's SVDD-PM (M=20 candidates, 3 Monte Carlo value samples) and SMC (groups of 20 particles) baselines (Appendix C.2), with the closed-form synthetic reverse kernel and denoiser and 32 reverse steps.
- `direct_mh` (`baseline/direct_mh_synthetic.py`): single-site Metropolis-Hastings on `p_0(x) p(y|x)` without split Gibbs, 100 steps.
