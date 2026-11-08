> Iteration 11 (hardening) replaced the evaluation design and added baselines and a coupled chain experiment; current state and results are in `HARDENING.md`. The notes below describe earlier iterations.

# Progress

## What Works

- Implemented the paper's synthetic SGDD computation with alternating likelihood Metropolis-Hastings updates and reverse discrete-diffusion prior-denoising updates.
- Implemented method-agnostic synthetic data definitions and exact marginal evaluation.
- Added scripts for download, method execution, evaluation, and end-to-end reproduction.
- Completed CPU compute-container runs for the full synthetic target set `D=2`, `D=5`, and `D=10`.
- Updated the run and reproduce entry points so they regenerate SGDD outputs and the included direct-MH sanity baseline artifacts before validation.
- Confirmed job `4f4fee48-166` passes the full validation path with SGDD-only official scores and separate direct-MH sanity scores.
- Fixed the audit finding that the previous default prior step sampled Eq. 11 exactly instead of exercising the paper's Euler reverse discrete-diffusion prior step. The method now defaults to a closed-form concrete-score Euler prior sampler with the paper's synthetic `K=10`, `T=10`, `H=20` settings.

## Results Achieved

The successful corrected job `cb3d9782-268` produced:

| Experiment | Paper SGDD Hellinger | Repro Hellinger | Paper SGDD TV | Repro TV |
|---|---:|---:|---:|---:|
| `synthetic_d2` | 0.149 | 0.162 | 0.125 | 0.132 |
| `synthetic_d5` | 0.214 | 0.165 | 0.222 | 0.152 |
| `synthetic_d10` | 0.334 | 0.181 | 0.365 | 0.163 |

All retained reproduced SGDD runs preserve `uses_likelihood_gradient=0`. The corrected Euler-prior implementation is close to the paper's `D=2` score and improves on the paper's SGDD Hellinger and total variation for `D=5` and `D=10`, including the hardest synthetic setting. This restores the synthetic `core_claim_plus` reproduction.

## Deviations From Paper

- This reproduction targets the synthetic Table 2 setting rather than the image inpainting or text infilling experiments. The synthetic task is sufficient for the core algorithmic claim and does not require training a diffusion model.
- The implementation uses the analytic factorized concrete score for the synthetic task, matching the paper's synthetic setup.
- The paper's appendix lists `K=10` SGDD iterations, `T=10` Metropolis-Hastings likelihood updates, and `H=20` Euler prior steps for the synthetic setting. The runnable defaults now match those values.

## What Remains

- The reusable synthetic reproduction environment is complete through `core_claim_plus`.
- If extending beyond the synthetic core, add image or text posterior sampling experiments.
- Add paper baselines such as SMC or DPS if extending beyond the synthetic core reproduction.
- Paper baselines such as SMC or DPS remain paper-reported references rather than reproduced implementations.

## Issues Encountered

- The markdown paper did not include all low-level synthetic operator details; the released author code was inspected to recover the exact SingleModal setup.
- A tiny login-node smoke test could not run because the local Python outside the compute container lacks `numpy`; dependencies are declared in `environment/setup.sh`, so execution is being tested through the scheduler.
- The first container build failed because the base image used `ghcr.io`; it was changed to the portable Docker Hub image `python:3.11-slim-bookworm`.
- Validator attempt 1 rejected the old `D=2` score as too far from Table 2; the corrected Euler-prior job now lands near the paper's reported `D=2` result.
- The continuation audit found stale score provenance, an SGDD-specific evaluator path convention, and a hard-coded constraint metric. The evaluator now supports explicit method/run arguments, uses method-agnostic checkpoint naming, and reads `uses_likelihood_gradient` from run metadata.
- Job `af393454-f57` correctly computed direct-MH sanity metrics, but validation failed because `direct_mh` is not a paper-reported reference method. The standard score file now remains SGDD-only and sanity metrics are written to `scoring/sanity_scores.json`.
- Job `4f4fee48-166` confirmed the corrected scoring split passes `python validate.py --compare`.
