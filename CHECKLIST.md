# Checklist

- [x] Read the paper and identify the core reproducible claim.
- [x] Inspect the released author code for synthetic experiment details.
- [x] Implement method-agnostic synthetic problem definitions.
- [x] Implement SGDD for the synthetic categorical posterior.
- [x] Implement reusable evaluation that does not import `method/`.
- [x] Fill `scoring/reference.json` from paper-reported numbers.
- [x] Write environment and script entry points.
- [x] Complete a full reproduction job.
- [x] Update reproduced scores and milestone after validation cleanup.
- [x] Audit inherited implementation and restore `core_claim_plus`.
- [x] Add direct-MH sanity baseline generation while keeping default scores paper-compatible.
- [x] Replace exact synthetic prior default with Euler reverse discrete-diffusion prior sampler.
- [x] Make evaluation paths and gradient-constraint metadata method-agnostic.
- [x] Validate corrected Euler-prior SGDD on `D=2`, `D=5`, and `D=10`.
- [x] Include generated direct-MH sanity artifacts in the standard scoring path.
