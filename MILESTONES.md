# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-02 21:15] - core_claim
- Completed a CPU compute-container run of SGDD on `synthetic_d5` and `synthetic_d10`.
- Validation passed and reproduced SGDD beats the paper's reported synthetic baselines on the active dimensions while preserving `uses_likelihood_gradient=0`.

### [2026-06-02 21:25] - core_claim_plus
- Audited the existing implementation against the paper: the method uses split Gibbs alternation, the Hamming potential, Metropolis-Hastings likelihood updates, and a closed-form synthetic prior-denoising step.
- Prepared a follow-up run that also evaluates the direct-MH sanity baseline for the active target dimensions.

## Stop Justification
- Not stopped; submitting a follow-up CPU job to refresh method and baseline scores.
