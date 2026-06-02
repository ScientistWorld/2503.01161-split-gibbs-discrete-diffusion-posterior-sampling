# Constraints

The problem definition must remain fixed:

- categorical vocabulary size is 50;
- active latent dimensions are `D=5` and `D=10`;
- the discretized Gaussian prior, SingleModal observation, and likelihood scale are defined in `data/synthetic_sgdd.py`;
- evaluation is the first-two-coordinate posterior marginal against the exact target.

Methods must not use the exact posterior target during sampling. It is reserved for evaluation only.

For comparisons to the paper's core claim, methods must not rely on gradients of the likelihood with respect to continuous relaxations of the categorical state. Such methods can be studied separately, but they violate the `uses_likelihood_gradient=0` constraint.
