# Overview

This gym studies reconstruction from discrete, incomplete measurements, using the MNIST XOR/AND image problems in *Split Gibbs Discrete Diffusion Posterior Sampling*. It evaluates whole reconstructed images and measurement consistency, with unseen instances supplied at test time.

Start with `briefing/problem.md` and `briefing/evaluation.md`. `scripts/download.sh` fetches all public resources. `environment/setup.sh` declares Python dependencies. The ordinary `scripts/run.sh` reproduces the reference and included comparators on both evaluation slices in a serialized GPU job.

The former small Gaussian and chain experiments remain only as historical reproduction/audit source. They no longer affect ranking: their analytical posteriors made them unsuitable as a scientific improvement benchmark.
