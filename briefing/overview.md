# Overview

This workspace reproduces the core synthetic result from "Split Gibbs Discrete Diffusion Posterior Sampling." The paper addresses posterior sampling in discrete inverse problems without requiring likelihood gradients.

The implemented benchmark runs SGDD on the paper's synthetic categorical posterior-recovery task for `D=2`, `D=5`, and `D=10`, then evaluates generated samples against the exact posterior marginal using Hellinger and total variation distances.
