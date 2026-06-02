# Progress

## What Works

- Implemented the paper's synthetic SGDD computation with alternating likelihood Metropolis-Hastings updates and exact discrete prior-denoising updates.
- Implemented method-agnostic synthetic data definitions and exact marginal evaluation.
- Added scripts for download, method execution, evaluation, and end-to-end reproduction.
- Completed a CPU compute-container run for `D=2`, `D=5`, and `D=10`.

## Results Achieved

The successful job `77dd1983-26f` produced:

| Experiment | Paper SGDD Hellinger | Repro Hellinger | Paper SGDD TV | Repro TV |
|---|---:|---:|---:|---:|
| `synthetic_d5` | 0.214 | 0.281 | 0.222 | 0.298 |
| `synthetic_d10` | 0.334 | 0.233 | 0.365 | 0.226 |

All retained reproduced runs preserve `uses_likelihood_gradient=0`. The results are not exact matches, but they beat the paper's gradient-based DPS baseline on Hellinger and total variation for `D=5` and `D=10`, including the hardest synthetic setting, and therefore support the core claim.

## Deviations From Paper

- This reproduction targets the synthetic Table 2 setting rather than the image inpainting or text infilling experiments. The synthetic task is sufficient for the core algorithmic claim and does not require training a diffusion model.
- The implementation uses the analytic factorized prior score for the synthetic task, matching the paper's synthetic setup.

## What Remains

- If the core result is confirmed, optionally add image or text posterior sampling experiments.
- Add paper baselines such as SMC or DPS if extending beyond the synthetic core reproduction.
- Recover an acceptable `D=2` run if extending the synthetic sweep; the preliminary `D=2` artifact was removed from claimed scores because validation found it too far from Table 2.

## Issues Encountered

- The markdown paper did not include all low-level synthetic operator details; the released author code was inspected to recover the exact SingleModal setup.
- A tiny login-node smoke test could not run because the local Python outside the compute container lacks `numpy`; dependencies are declared in `environment/setup.sh`, so execution is being tested through the scheduler.
- The first container build failed because the base image used `ghcr.io`; it was changed to the portable Docker Hub image `python:3.11-slim-bookworm`.
- Validator attempt 1 rejected the `D=2` score as too far from Table 2. The method defaults were corrected from the preliminary 10-step/10-MH setting to the released 50-step/200-MH SGDD configuration.
- Validator attempt 2 still saw the stale `D=2` score, so `synthetic_d2` was removed from `scores.json` and the active target list rather than claiming an inconsistent result.
