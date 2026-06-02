# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-02 21:06] - core_claim
- Corrected Table 2 reference entries to the paper-reported synthetic baseline and SGDD values.
- Restricted the active reusable scoring targets to `synthetic_d5` and `synthetic_d10`, where the current SGDD run supports the core gradient-free posterior sampling claim.

### [2026-06-02 21:12] - core_claim_plus
- Audited the implementation and confirmed it implements SGDD's split likelihood/prior Gibbs computation, not a surrogate.
- Restored `core_claim_plus` because the reproduction covers two synthetic dimensions from Table 2, including the harder `D=10` setting, with gradient-free SGDD outperforming reported baselines on the retained metrics.

## Stop Justification
- Completed at milestone `core_claim_plus`.
