# Overview

This workspace reproduces the core synthetic result from "Split Gibbs Discrete Diffusion Posterior Sampling." The paper addresses posterior sampling in discrete inverse problems without requiring likelihood gradients.

The implemented benchmark runs SGDD on the paper's synthetic categorical posterior-recovery task for `D=2`, `D=5`, and `D=10`, then evaluates generated samples against the exact posterior marginal using Hellinger and total variation distances.

---

# Problem

The task is posterior sampling for discrete inverse problems. Given a discrete latent vector `x` with a known prior and an observation `y` produced by a likelihood model, produce samples from the posterior distribution `p(x | y)`.

The reusable benchmark in this workspace focuses on the paper's synthetic posterior-recovery setting. Latents are categorical vectors of dimension `D` with 50 possible values per coordinate. The prior is a discretized Gaussian over each coordinate, and the observation is produced by a non-invertible coordinate-wise transform. Evaluation compares the empirical first-two-coordinate marginal from generated samples against the exact posterior marginal.

Success means producing posterior samples whose empirical distribution is close to the exact posterior while preserving the intended discrete sampling setting. Evaluation code is method-agnostic and reads generated sample artifacts from `checkpoints/`.

---

# Method

The paper proposes Split Gibbs Discrete Diffusion Posterior Sampling (SGDD). It samples a discrete posterior by augmenting the latent variable with an intermediate noised state and alternating two conditional updates:

- a likelihood step that samples the intermediate state under the observation likelihood and a discrete corruption kernel;
- a prior step that samples the clean state using the discrete diffusion model's posterior denoising distribution.

For the synthetic reproduction, the prior score is available analytically because the prior is a factorized discretized Gaussian. The implementation keeps the paper's split Gibbs computation: a Metropolis-Hastings likelihood update for the intermediate state followed by an exact categorical prior-denoising update using the Hamming corruption kernel. The noise level follows the paper's geometric schedule from `eta_min=1e-4` to `eta_max=20`; the runnable default matches the released SGDD configuration with 50 split-Gibbs steps and 200 likelihood MH updates per step.

The implementation lives in `method/sgdd_synthetic.py`. The author repository was inspected for details of the synthetic operator and evaluator, but the benchmark uses a compact implementation tailored to this reproducible environment.

---

# Evaluation

The benchmark evaluates three synthetic dimensions: `D=2`, `D=5`, and `D=10`. For each dimension, a method writes integer samples to:

`checkpoints/sgdd_synthetic_d<D>/samples.npz`

The file must contain a `samples` array with shape `[num_samples, D]` and entries in `[0, 49]`. `scripts/evaluate.sh` calls only `eval/` code, computes the exact first-two-coordinate posterior marginal from the public problem definition in `data/`, and writes `scoring/scores.json`.

Metrics:

- `hellinger`: Hellinger distance between empirical and exact first-two-coordinate marginals. Lower is better.
- `total_variation`: Total variation distance between empirical and exact first-two-coordinate marginals. Lower is better.
- `uses_likelihood_gradient`: constraint metric indicating whether the method relies on likelihood gradients. The target method is gradient-free, so lower is better.

The main comparison is against the paper's Table 2 synthetic posterior-recovery results.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 00:00] - none
- Read the paper and identified SGDD's synthetic posterior-recovery experiment as the minimum core-claim target.
- Implemented the synthetic problem, SGDD sampler, and method-agnostic evaluator.
- Prepared a CPU reproduction job to validate the implementation end-to-end.
- Fixed the first container portability failure by switching to a Docker Hub base image.

### [2026-06-02 21:05] - core_claim_plus
- Ran the SGDD synthetic posterior sampler for `D=2`, `D=5`, and `D=10` inside the compute container.
- Validation passed with method/evaluation separation intact.
- Reproduced the paper's gradient-free posterior recovery claim on all three synthetic Table 2 settings.

</details>


---

# Reproduction Milestones

**Current: core_claim**
