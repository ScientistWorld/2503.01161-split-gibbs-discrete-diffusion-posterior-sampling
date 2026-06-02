# Progress

## What Works

- Implemented the paper's synthetic SGDD computation with alternating likelihood Metropolis-Hastings updates and exact discrete prior-denoising updates.
- Implemented method-agnostic synthetic data definitions and exact marginal evaluation.
- Added scripts for download, method execution, evaluation, and end-to-end reproduction.

## Results Achieved

No full reproduction job has completed yet. A CPU job is being submitted to run `D=2`, `D=5`, and `D=10` SGDD sample generation and validation.

## Deviations From Paper

- This reproduction targets the synthetic Table 2 setting rather than the image inpainting or text infilling experiments. The synthetic task is sufficient for the core algorithmic claim and does not require training a diffusion model.
- The implementation uses the analytic factorized prior score for the synthetic task, matching the paper's synthetic setup.

## What Remains

- Run the full synthetic reproduction job.
- Compare reproduced Hellinger and total variation distances against Table 2.
- If the core result is confirmed, optionally add image or text posterior sampling experiments.

## Issues Encountered

- The markdown paper did not include all low-level synthetic operator details; the released author code was inspected to recover the exact SingleModal setup.
- A tiny login-node smoke test could not run because the local Python outside the compute container lacks `numpy`; dependencies are declared in `environment/setup.sh`, so execution is being tested through the scheduler.
- The first container build failed because the base image used `ghcr.io`; it was changed to the portable Docker Hub image `python:3.11-slim-bookworm`.
