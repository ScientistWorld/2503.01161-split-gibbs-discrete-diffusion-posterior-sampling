# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-02 00:00] - none
- Read the paper and identified SGDD's synthetic posterior-recovery experiment as the minimum core-claim target.
- Implemented the synthetic problem, SGDD sampler, and method-agnostic evaluator.
- Prepared a CPU reproduction job to validate the implementation end-to-end.
- Fixed the first container portability failure by switching to a Docker Hub base image.

### [2026-06-02 21:05] - core_claim_plus
- Ran the SGDD synthetic posterior sampler for `D=2`, `D=5`, and `D=10` inside the compute container.
- Validation passed with method/evaluation separation intact.
- Reproduced the paper's gradient-free posterior recovery claim on all three synthetic Table 2 settings.
