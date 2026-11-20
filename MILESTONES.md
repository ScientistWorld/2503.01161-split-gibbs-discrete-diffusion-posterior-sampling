# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-02 21:15] - core_claim
- Completed a CPU compute-container run of SGDD on `synthetic_d5` and `synthetic_d10`.
- Validation passed and reproduced SGDD beats the paper's reported synthetic baselines on the active dimensions while preserving `uses_likelihood_gradient=0`.

### [2026-06-02 21:25] - core_claim_plus
- Audited the existing implementation against the paper: the method uses split Gibbs alternation, the Hamming potential, Metropolis-Hastings likelihood updates, and a closed-form synthetic prior-denoising step.
- Prepared a follow-up run that also evaluates the direct-MH sanity baseline for the active target dimensions.

### [2026-06-02 22:05] - core_claim
- Continuation audit found that the old default prior step sampled the synthetic Eq. 11 conditional exactly instead of running the paper's Euler reverse discrete-diffusion prior sampler.
- Replaced the default prior step with a closed-form concrete-score Euler CTMC sampler and restored the paper's synthetic `K=10`, `T=10`, `H=20` defaults.
- Fixed evaluator issues: method-agnostic artifact paths, explicit run arguments, and metadata-based `uses_likelihood_gradient`.

### [2026-06-02 22:35] - core_claim_plus
- Corrected CPU compute-container job `cb3d9782-268` completed on `synthetic_d2`, `synthetic_d5`, and `synthetic_d10`.
- SGDD preserves `uses_likelihood_gradient=0`, is close to the paper's `D=2` result, and improves over the paper's SGDD results on `D=5` and `D=10`.
- Updated the standard run path to evaluate the generated direct-MH sanity baseline separately from the paper-compatible `scoring/scores.json`.

### [2026-06-02 22:45] - core_claim_plus
- Job `af393454-f57` failed validation only because the direct-MH sanity baseline was placed in `scoring/scores.json`, where the validator expects methods from `reference.json`.
- Moved direct-MH sanity evaluation to `scoring/sanity_scores.json` and restored `scoring/scores.json` as the official paper-comparison output.

### [2026-06-02 22:55] - core_claim_plus
- Job `4f4fee48-166` completed successfully and passed `python validate.py --compare`.
- Official `scoring/scores.json` contains the corrected SGDD results for all three synthetic Table 2 dimensions.
- Optional direct-MH sanity metrics are regenerated separately in `scoring/sanity_scores.json`.

### [2026-06-02 23:00] - core_claim_plus
- Final verification job `56df88b2-3db` completed successfully on a CPU node.
- Regenerated auxiliary train/test scores after deterministic split cleanup.
- `python validate.py --compare` passed with no validation warnings.

### [2026-11-08 09:00] - core_claim_plus
- Hardening (iteration 11): test-time evaluation of every registered sampler on hidden observations, exactly 50,000 samples per instance, validity gates, an all-pairs constraint metric, and the self-reported gradient flag removed from scoring.
- Fixed the SGDD Euler prior step so each token jumps at most once per sub-step. Made the paper's SVDD-PM and SMC baselines executable; SGDD beats both on every experiment.
- Added the coupled, multimodal `synthetic_chain_d10` experiment, where the reference is far from the exact-sample bound. Details and audit results are in `HARDENING.md`.

## Current status
- The active image method runs on the public slice. Hidden evaluation and headroom validation remain in progress; historical synthetic milestones above do not establish the image gym's milestone.

### [2026-11-17] - none
- Prepared a replacement image gym using the original pretrained SEDD model and paper SGDD settings after REVIEW.md found structural problems in the old synthetic benchmark.
- Active image reproduction and final hardening claims await the bundled compute job; historical synthetic milestones above are not claims about the new task.

### [2026-11-20] - method_runs
- Job `620f95a2-5f2` completed all public-slice reference and comparator inference. SGDD achieved XOR PSNR 17.335 dB / class consistency 85.16%, and sparse AND 12.425 dB / 65.63%, with measurement error below 0.25% on both.
- Hidden inference failed before sampling: the read boundary prevented package discovery from the workspace directory. Added directory-list permission while retaining file-content denial for hidden resources and other hidden runs.
- Fixed launcher restoration to replace the script atomically. The resumed bundled job rescores the completed training outputs and reruns all hidden reference, comparator, honest-variant and shortcut procedures.
