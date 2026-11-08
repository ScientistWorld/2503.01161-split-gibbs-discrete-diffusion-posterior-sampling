# Overview

This workspace is a gym built from "Split Gibbs Discrete Diffusion Posterior Sampling" (SGDD). The paper addresses posterior sampling in discrete-state spaces with a discrete diffusion prior, without likelihood gradients.

The gym uses the paper's synthetic posterior-recovery task, where the exact posterior is computable: `D = 2, 5, 10` with the coordinatewise observation of Table 2, plus a coupled `D = 10` task with pairwise relative observations between neighbouring coordinates. This is a categorical analogue of the paper's pairwise logical forward models (Eq. 15). Methods are samplers. The evaluators run them on public train instances and on hidden test instances with a fixed sample budget, then score the samples' pair marginals against the exact posterior.
