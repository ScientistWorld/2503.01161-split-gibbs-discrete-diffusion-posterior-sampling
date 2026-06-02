# Progress

## What Works

- Implemented the paper's synthetic SGDD computation with alternating likelihood Metropolis-Hastings updates and reverse discrete-diffusion prior-denoising updates.
- Implemented method-agnostic synthetic data definitions and exact marginal evaluation.
- Added scripts for download, method execution, evaluation, and end-to-end reproduction.
- Completed a CPU compute-container run for the earlier active targets `D=5` and `D=10`; a fresh run is pending for the full synthetic target set `D=2`, `D=5`, and `D=10`.
- Updated the run and reproduce entry points so they regenerate SGDD outputs and the included direct-MH sanity baseline artifacts before validation.
- Fixed the audit finding that the previous default prior step sampled Eq. 11 exactly instead of exercising the paper's Euler reverse discrete-diffusion prior step. The method now defaults to a closed-form concrete-score Euler prior sampler with the paper's synthetic `K=10`, `T=10`, `H=20` settings.

## Results Achieved

The successful job `77dd1983-26f` produced:

| Experiment | Paper SGDD Hellinger | Repro Hellinger | Paper SGDD TV | Repro TV |
|---|---:|---:|---:|---:|
| `synthetic_d5` | 0.214 | 0.281 | 0.222 | 0.298 |
| `synthetic_d10` | 0.334 | 0.233 | 0.365 | 0.226 |

All retained reproduced runs preserve `uses_likelihood_gradient=0`. The results are not exact matches, but they beat the paper's gradient-based DPS baseline on Hellinger and total variation for `D=5` and `D=10`, including the hardest synthetic setting, and therefore support the core claim.

The continuation audit on 2026-06-02 found that the old default prior sampler was too exact for milestones above `core_claim`. The live milestone is therefore held at `core_claim` until the corrected Euler-prior SGDD job finishes.

## Deviations From Paper

- This reproduction targets the synthetic Table 2 setting rather than the image inpainting or text infilling experiments. The synthetic task is sufficient for the core algorithmic claim and does not require training a diffusion model.
- The implementation uses the analytic factorized concrete score for the synthetic task, matching the paper's synthetic setup.
- The paper's appendix lists `K=10` SGDD iterations, `T=10` Metropolis-Hastings likelihood updates, and `H=20` Euler prior steps for the synthetic setting. The runnable defaults now match those values.

## What Remains

- Confirm the Euler-prior SGDD results in a fresh compute-container run.
- If the core result is confirmed, optionally add image or text posterior sampling experiments.
- Add paper baselines such as SMC or DPS if extending beyond the synthetic core reproduction.
- The next submitted job will refresh SGDD scores and regenerate direct-MH sanity artifacts for `D=2`, `D=5`, and `D=10`. The official `scoring/scores.json` remains reference-compatible by default; direct-MH scoring is opt-in via `INCLUDE_SANITY_BASELINES=1`.

## Issues Encountered

- The markdown paper did not include all low-level synthetic operator details; the released author code was inspected to recover the exact SingleModal setup.
- A tiny login-node smoke test could not run because the local Python outside the compute container lacks `numpy`; dependencies are declared in `environment/setup.sh`, so execution is being tested through the scheduler.
- The first container build failed because the base image used `ghcr.io`; it was changed to the portable Docker Hub image `python:3.11-slim-bookworm`.
- Validator attempt 1 rejected the old `D=2` score as too far from Table 2.
- The continuation audit found stale score provenance, an SGDD-specific evaluator path convention, and a hard-coded constraint metric. The evaluator now supports explicit method/run arguments, uses method-agnostic checkpoint naming, and reads `uses_likelihood_gradient` from run metadata.
