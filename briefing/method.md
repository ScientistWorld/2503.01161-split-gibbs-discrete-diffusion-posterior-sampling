# Method

The paper proposes Split Gibbs Discrete Diffusion Posterior Sampling (SGDD). It samples a discrete posterior by augmenting the latent variable with an intermediate noised state and alternating two conditional updates:

- a likelihood step that samples the intermediate state under the observation likelihood and a discrete corruption kernel;
- a prior step that samples the clean state using the discrete diffusion model's posterior denoising distribution.

For the synthetic reproduction, the prior score is available analytically because the prior is a factorized discretized Gaussian. The implementation keeps the paper's split Gibbs computation: a Metropolis-Hastings likelihood update for the intermediate state followed by an exact categorical prior-denoising update using the Hamming corruption kernel. The noise level follows the paper's geometric schedule from `eta_min=1e-4` to `eta_max=20`, with `T=10`, `K=10`, and `H=20` for the synthetic task.

The implementation lives in `method/sgdd_synthetic.py`. The author repository was inspected for details of the synthetic operator and evaluator, but the benchmark uses a compact implementation tailored to this reproducible environment.
