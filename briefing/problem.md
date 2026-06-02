# Problem

The task is posterior sampling for discrete inverse problems. Given a discrete latent vector `x` with a known prior and an observation `y` produced by a likelihood model, produce samples from the posterior distribution `p(x | y)`.

The reusable benchmark in this workspace focuses on the paper's synthetic posterior-recovery setting. Latents are categorical vectors of dimension `D` with 50 possible values per coordinate. The prior is a discretized Gaussian over each coordinate, and the observation is produced by a non-invertible coordinate-wise transform. Evaluation compares the empirical first-two-coordinate marginal from generated samples against the exact posterior marginal.

Success means producing posterior samples whose empirical distribution is close to the exact posterior while preserving the intended discrete sampling setting. Evaluation code is method-agnostic and reads generated sample artifacts from `checkpoints/`.
