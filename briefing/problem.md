# Problem

The task is posterior sampling for discrete inverse problems. A latent vector `x` of `D` categorical tokens (50 values per coordinate) has a known prior `p(x)`, and an observation `y` is generated through a known forward model and noise level. A sampler must return draws from the posterior `p(x | y) ∝ p(x) p(y | x)`.

The prior is the same on every coordinate: a Gaussian with standard deviation 2 discretized on 50 evenly spaced points in `[-5, 5]`, independent across coordinates. It is the clean-data distribution of a discrete diffusion model with a uniform transition kernel, so its noised marginals `p_sigma` (and therefore the model's concrete score) are available in closed form through `data/synthetic_sgdd.py`. The likelihood is `log p(y | x) = -loss(G(x), y) / sigma^2` with `sigma = 3`, where `c = 24.5` is the centre token, and the forward model is one of:

- `coordinatewise`: `G(x)_d = |x_d - c|`, one observation per coordinate, loss = mean over coordinates of the squared residual. The posterior is bimodal on each coordinate.
- `l1_norm`: `G(x) = sum_d |x_d - c|`, a single scalar observation, loss = squared residual. The observation lies in the prior's upper tail, so the posterior couples all coordinates.

There are four experiments: `coordinatewise` with `D = 2, 5, 10`, and `l1_norm` with `D = 10`. Each experiment is scored on problem instances that differ only in the observation `y` and the seed. The train instances are public. The test instances use different, hidden observations of the same kind.

A sampler receives a problem file describing one instance, including the number of samples to return and a seed. It must return exactly that many samples. Success means the samples' joint distribution is close to the exact posterior, both on the first two coordinates and on every other pair of coordinates.
