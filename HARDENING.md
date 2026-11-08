# Hardening report

**Bottom line.**

- **Ranking.** Better results now score higher, far beyond seed noise: the seed spread is 0.0011, and better variants gain +0.05 to +0.22.
- **Shortcuts.** Every major shortcut in `REDTEAM.md` now scores below the reference or fails outright, except one. A sampler that computes the exact posterior analytically from the problem file still reaches the bound. This is structural to a closed-form synthetic prior and is left to the shortcut grader.
- **Headroom.** The reference is below the bound on every scored metric. I added a coupled experiment (`synthetic_chain_d10`) where neither long Metropolis–Hastings nor longer SGDD gets close to the bound.

All numbers come from jobs `cd871d51-488` (round 1) and `c4b7c79e-ce6` (round 2). Every score below is from exactly `bash scripts/evaluate_test.sh` with no arguments, run in a fresh process. Variants and shortcuts were registered temporarily as `scripts/samplers/<name>.sh` (`redteam/hardening/run_audit.sh`). Gains use the final `scoring/reference.json` weights and coefficients, relative to the reference `sgdd` (gain 0). Raw files are in `redteam/hardening/results/` (final) and `redteam/hardening/results_round1/` (first design).

## What changed in the gym

| Problem in REDTEAM.md | Fix |
|---|---|
| Test answers rebuildable from public `data/` (`exact_posterior_first2`) | `data/` now only parses problem files. Exact targets are computed separately in `eval/train/` and `eval/test/`, with no shared code. Test observations and seeds exist only in `eval/test/`. Samplers are **run at test time** on those hidden instances and write into a temp dir the evaluator owns. |
| Problem definition imported from mutable `data/` | Prior, forward models, noise levels, observations, seeds and sample count are constants in each evaluator. They are handed to samplers as a JSON problem file. |
| Sample count was free; 10× samples gave +0.205 | Exactly 50,000 samples per instance is required. Any other count fails the method. |
| Self-reported `uses_likelihood_gradient` was scored and unbounded | Removed from scoring. Gradient-free sampling is an eligibility rule in `scoring/CONSTRAINTS.md`. Nothing a sampler writes about itself is read. |
| Impossible tokens wrapped by NumPy; empty output crashed the test | Integer dtype, exact shape and the `[0, 49]` range are enforced. Any violation, crash or missing file fails the method on every experiment, with all distances set to 1.0, the worst possible value. |
| Only the first coordinate pair was checked | New constraint metric `all_pairs_hellinger`: the mean Hellinger distance over every coordinate pair, coefficient −0.5. |
| `_run_dirs` in the train file chose which artifact got scored | There is no registry. The test discovers `scripts/samplers/*.sh` and runs each one fresh. |
| 5,000-sample test half; noise floor ≈ reference (D5/D10 indistinguishable from perfect) | 50,000 samples per instance, two hidden instances per experiment. The iid floor is now 0.061 / 0.074 / 0.084 Hellinger for D = 2 / 5 / 10, against reference values of 0.093 / 0.092 / 0.099. |
| Simple direct-MH baseline beat the reference | Added executable paper baselines SVDD-PM and SMC (Appendix C.2). Both score far below SGDD. Direct MH is kept as a strong comparator: with the exact synthetic prior it is a good sampler, not a weak baseline. |
| Euler prior step could move a token several times per sub-step | Fixed so each token jumps at most once per (sub-)step, matching the paper's Euler sampler. Measured effect is +0.0013, at the noise level, as the red team found. |
| No headroom beyond what a simple variant reaches | Added the `synthetic_chain_d10` experiment (see the headroom section). |

The SGDD reference itself is unchanged apart from the Euler fix. It still uses the paper's synthetic settings K=10, T=10, H=20, geometric η from 20 to 1e-4, and the paper's Eq. 10 potential. I checked that the authors' code uses the same `β/(1−β)` per-mismatch factor.

## Do better results score higher? (variants before → after)

"Before" is the red-team run on the old test; "after" is the final test. All are gains against the reference.

| Variant | Before | After | Note |
|---|---:|---:|---|
| Reference, other seeds (spread) | 0.0063 | **0.0011** | seeds +1, +2 score +0.00106 and +0.00098 |
| 10× samples | +0.205 | **failed (−1.594)** | now a constraint violation, not an improvement |
| Likelihood updates removed (T=0) | −0.262 | **−0.463** | |
| Fixed η=0.5 instead of annealing (paper App. B.1 ablation) | — | **−0.226** | |
| Pre-fix Euler step (old reference) | −0.005 | **+0.0013** | ≈ seed spread; the fix does not change quality |
| Direct MH, 100 steps | +0.025 | **+0.053** | beats the reference on D=2/5/10 and on the chain first pair |
| Direct MH, 1000 steps | — | **+0.218** | at the bound on D=2/5/10; far from it on the chain |
| SGDD, T=100 MH steps | — | **+0.094** | worse on D=2/5/10, much better on the chain |
| SGDD, K=50, T=50 | — | **+0.108** | worse on D=2/5/10, best SGDD on the chain |
| SGDD with kernel-matched Hamming potential `β/((N−1)(1−β))` | — | **−0.212** | tested hypothesis; worse at paper settings |
| same, K=50, T=50 | — | **+0.045** | below SGDD K=50 with the paper's potential |
| Paper baseline SMC (M=20) | — | **−0.288** | below the reference on every experiment |
| Paper baseline SVDD-PM (M=20) | — | **−0.622** | below the reference on every experiment |

Apart from the Euler fix, every effect is 40× to 500× the seed spread. Every comparison is paired: identical problem files and seeds.

Train and test agree for honest methods. For `sgdd`, Hellinger is 0.095 / 0.092 / 0.097 / 0.528 on train against 0.093 / 0.092 / 0.099 / 0.527 on test, with the same pattern for the baselines.

**The reproduction shows the paper's claim in the paper's direction.** On the Table 2 tasks, SGDD's Hellinger is 0.093 / 0.092 / 0.099 for D = 2 / 5 / 10. SMC gets 0.203 / 0.240 / 0.269 and SVDD-PM gets 0.530 / 0.480 / 0.446. On the chain task, SGDD gets 0.527 against SMC 0.673 and SVDD-PM 0.656.

**Where the paper's method falls short.**

- **Direct MH.** Not a paper baseline, but single-site MH with the exact synthetic prior beats SGDD at matched likelihood-evaluation counts on the coordinatewise tasks.
- **More compute hurts on the Table 2 tasks.** With more MH steps or iterations, SGDD gets *worse* there, though better on the chain. The likelihood step's Eq. 10 potential and the diffusion prior step do not describe the same joint distribution: they differ by a factor of N−1 per mismatch. That could explain it, but the kernel-matched variant above did not fix it at paper settings, so the cause remains open.
- These are research questions for a future agent, not gym defects.

## Shortcuts: fix and score before → after

| Shortcut (red-team name) | Before | Fix | After |
|---|---:|---|---:|
| Rebuild test answers from public `data/` | +0.273 | Exact targets live only in `eval/`; test instances hidden. Answers rebuilt for the public train instance (`shortcut_cached_train`) no longer transfer. | **−0.051** |
| Impossible constraint value (G=−10) | +0.751 | flag not read | **−1.594** |
| Compensate a binary constraint violation (G=1) | +0.073 | Flag not scored; gradient use is an eligibility rule. The run only pays through the analytic exact posterior below. | +0.384 (via exact posterior) |
| Replace the fixed problem (edit `data/` defaults) | +0.351 | evaluator owns the problem | **−1.594** |
| Move the submitted artifact (`_run_dirs`) | +0.025 | No registry: each sampler is run fresh into an evaluator-owned temp dir. The closest analogue, submitting another method's samples, just scores as that method. | not replayable |
| Invalid full posterior (copied coordinates) | +0.025 | `all_pairs_hellinger` | **−0.087** |
| Impossible categories (wrap-around, exact tie) | 0.000 | token-range gate | **failed, −1.594** |
| Forged reported scores | −1.249 | scores recomputed | −1.594 |
| Constant outputs | −1.249 | — | −1.594 |
| Empty outputs | crash, no score | shape gate | **failed, −1.594** |
| Return more samples | +0.205 | exact-count gate | **failed, −1.594** |

**Shortcuts left open, with reasons.**

- **Analytic exact posterior** (`shortcut_exact_posterior`, +0.384, at the bound).
  - The sampler receives the problem file, so it can compute the posterior exactly: by per-coordinate enumeration, or by forward-filter/backward-sample on the chain.
  - This cannot be closed by output checks: the synthetic prior is closed-form, and any interface that lets SGDD run (concrete score plus likelihood) determines the posterior.
  - It is forbidden in `scoring/CONSTRAINTS.md` and `scoring/DIRECTION.md`, and left to the shortcut grader.
  - Rebuilding *precomputed* answers no longer works (−0.051).
- **Gradient use.** Not measurable from samples, so it is an eligibility rule.
- **Stratified or quota sampling.** Matching histogram cells instead of drawing samples could go below the iid floor. It needs near-exact posterior knowledge, so it is covered by the rule above. Grader.
- **Reading `eval/test/` at test time.** A sampler shares the filesystem with the test. Forbidden in `CONSTRAINTS.md`; grader.

## Reference score before and after

The reference has gain 0 by definition in both tests. Its raw values:

| | D=2 H / TV | D=5 H / TV | D=10 H / TV | chain H / TV / all-pairs |
|---|---|---|---|---|
| Old test (5k-sample half, public instance) | 0.209 / 0.170 | 0.220 / 0.196 | 0.266 / 0.235 | — |
| New test (50k samples, hidden instances) | 0.093 / 0.085 | 0.092 / 0.087 | 0.099 / 0.096 | 0.527 / 0.536 / 0.466 |

The paper's Table 2 numbers (10k samples) are kept unchanged under `_paper_*` in `scoring/reference.json`. They are not on the gym's 50k-sample scale.

## Headroom check

The bound is the score of exact iid posterior samples at 50,000 samples per instance, measured with the analytic shortcut. **The reference sits at the bound of no scored metric:**

| Experiment | Reference H / TV / all-pairs | Bound H / TV / all-pairs | Best non-shortcut variant |
|---|---|---|---|
| synthetic_d2 | 0.093 / 0.085 / 0.093 | 0.061 / 0.046 / 0.061 | direct MH 1000: 0.062 (at bound) |
| synthetic_d5 | 0.092 / 0.087 / 0.092 | 0.074 / 0.059 / 0.075 | direct MH: 0.075 (at bound) |
| synthetic_d10 | 0.099 / 0.096 / 0.098 | 0.084 / 0.067 / 0.084 | direct MH 1000: 0.084 (at bound) |
| synthetic_chain_d10 | 0.527 / 0.536 / 0.466 | 0.033 / 0.024 / 0.062 | SGDD K=50,T=50: 0.173 / 0.178 / 0.331 |

**The three Table 2 experiments are reachable by a simple variant.** Direct MH reaches the bound there, so on its own their headroom (+0.033 in total) only rewards running the comparator.

**What I added, `synthetic_chain_d10`.**

- **Definition.** D=10, observations `|x_{d+1} − x_d|` of a latent drawn from the prior, σ=1, summed loss. This is the categorical analogue of the paper's pairwise XOR/AND forward models (Eq. 15), with the exact posterior computable by transfer matrices.
- **Correctness.** Exact targets match brute-force enumeration to 1e-16, checked at D=3 and D=4 in `redteam/hardening/smoke.py`.
- **Difficulty.** The posterior is multimodal (sign and shift ambiguity). Single-site moves cannot cross between modes: direct MH at 1000 steps stays at 0.223.
- **Headroom.** The bound is +0.351 of weighted gain above the reference on this experiment alone. The best variant tried reaches +0.223 there, so the gap is not closed.
- **First attempt, dropped.** Round 1 added a scalar L1-norm observation instead (`G(x)=‖f(x)‖₁`, the paper's stated synthetic operator; `results_round1/`). It was saturated: SVDD-PM 0.080 and direct MH 1000 0.081 against a bound of 0.081. I dropped it.

**Weights.**

- Changed from 0.34 / 0.33 / 0.33 to **0.20 / 0.20 / 0.25 / 0.35** (D2 / D5 / D10 / chain).
- The paper says all methods do well at D=2, and that its largest gains come with growing dimension and hard pairwise forward models (XOR/AND, Table 4).
- The chain experiment is the only one where the test still separates good from excellent samplers.
- Coefficients: Hellinger −1.0 and TV −0.6 are kept, `all_pairs_hellinger` −0.5 is added as the constraint, and the gradient flag (−0.2) is removed.

## Not fixed / limitations, stated plainly

- The analytic-posterior shortcut (above) pays and can only be caught by review.
- No compute budget is enforced: only the sample count is fixed, and wall-clock time is not scored. Longer runs of honest samplers are allowed, and this is how long direct MH saturates the Table 2 experiments.
- The paper's DPS baseline is not implemented. Its Table 2 numbers remain as static `_paper_dps` entries.
- The SVDD-PM and SMC baselines use 32 reverse steps, chosen for test cost. The paper does not state the synthetic step count.
- The hidden test re-runs every registered sampler. With the four shipped samplers it takes roughly 10 minutes on one CPU core, mostly SVDD-PM.
- `redteam/audit.py` (the previous iteration's harness) targets the old artifact interface and no longer runs. Its results are kept as historical evidence.
