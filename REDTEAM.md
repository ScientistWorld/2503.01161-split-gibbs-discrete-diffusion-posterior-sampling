Honest improvements and ablations move the score in the right direction, but this gym does not reliably rank scientific contribution. Increasing the reference sample count improves its score well beyond seed variation, and removing likelihood updates clearly hurts it. However, the reproduced method loses to its simple direct-MH sanity baseline, and seven of ten shortcut checks pay, including one exact tie.

Scores below are improvements over the original hidden-test reference, whose score is **0**. For each dimension the contribution is its experiment weight times `-(H-H_ref) - 0.6*(TV-TV_ref) - 0.2*(G-G_ref)`, using weights `0.34, 0.33, 0.33` for D=2,5,10. H is Hellinger distance, TV is total variation, and G is the gradient-use flag. Larger aggregate gains rank higher. The raw reference and coefficient definitions are preserved in [reference_test.json](redteam/reference_test.json) and [ranking_reference.json](redteam/ranking_reference.json).

The first checks used the reference algorithm and settings (10 split iterations, 10 MH steps, 20 Euler steps, vocabulary 50). Only the stated feature changed. Each distance cell is H / TV; all four rows declare G=0 in every dimension.

| Variant | Samples per dimension | D=2 H / TV | D=5 H / TV | D=10 H / TV | Test-score gain |
|---|---:|---:|---:|---:|---:|
| [Reference rerun, seed 0](redteam/reference_seed0/scores_test.json) | 10,000 | 0.208911 / 0.169945 | 0.220315 / 0.195579 | 0.265944 / 0.235191 | +0.000000000 |
| [Honest improvement: 10x samples](redteam/more_samples/scores_test.json) | 100,000 | 0.094171 / 0.086772 | 0.090566 / 0.085462 | 0.095626 / 0.090852 | +0.205382523 |
| [Weaker: likelihood updates removed](redteam/weaker_no_likelihood/scores_test.json) | 10,000 | 0.500465 / 0.545205 | 0.343258 / 0.342320 | 0.294743 / 0.272740 | -0.262245874 |
| [Honest fix: frozen-state Euler transition](redteam/corrected_euler/scores_test.json) | 10,000 | 0.203476 / 0.168305 | 0.222984 / 0.202629 | 0.255849 / 0.224208 | +0.005411403 |

The larger-sample gain and ablation loss are both much larger than the reference seed spread. The Euler fix prevents an entry from being updated repeatedly within one Euler step when its new value is encountered later in the reference loop. Its measured gain is smaller than the seed spread, so this audit does **not** establish a quality improvement from that fix. The large gain from additional samples also shows that sample count strongly affects this benchmark; it does not enforce a common sampling budget.

Three fresh reference runs at seeds 0, 1, and 2 gave the following test-score gains. Seed 0 also exactly reproduced the original cached test scores. The observed aggregate maximum-minus-minimum spread is **0.006327212**; gains below that spread are noise under this check. Three seeds provide an empirical spread, not a confidence interval.

| Reference seed | Test-score gain |
|---|---:|
| [0](redteam/reference_seed0/scores_test.json) | +0.000000000 |
| [1](redteam/reference_seed1/scores_test.json) | +0.003371518 |
| [2](redteam/reference_seed2/scores_test.json) | +0.006327212 |

| Dimension | Hellinger spread | Total-variation spread |
|---|---:|---:|
| D=2 | 0.007110904 | 0.006455747 |
| D=5 | 0.007868982 | 0.010196499 |
| D=10 | 0.014712281 | 0.016135911 |

The reproduction check fails against the executable sanity baseline: the included direct-MH sampler, rerun with its original 10,000 samples, 100 steps, and seed 1, beats the reference on **both distances in every dimension**, with aggregate gain **+0.024854759**, beyond the reference seed spread. This simple baseline therefore does not sit below the reference as a useful scientific starting point.

| Dimension | Reference H / TV | Direct-MH baseline H / TV |
|---|---:|---:|
| D=2 | 0.208911 / 0.169945 | 0.185063 / 0.149294 |
| D=5 | 0.220315 / 0.195579 | 0.220075 / 0.190171 |
| D=10 | 0.265944 / 0.235191 | 0.246137 / 0.210709 |

Against the *static* paper-baseline rows supplied in `scoring/reference.json`, the reference does beat SVDD-PM (both M values) and SMC on both distances in all three dimensions. It does not beat the supplied gradient-using DPS Hellinger result at D=2 (0.208911 versus 0.182). The provided runnable baseline is direct MH, not those paper baselines, so the static comparison does not establish a reproduced head-to-head victory over them. No paper contents or hidden-test information were used to construct candidate methods.

The reference can be beaten and is not saturated on either distance. Every scored entry already at its intended bound is listed here:

| Experiment | Metric at bound | Reference value | Intended best value |
|---|---|---:|---:|
| synthetic_d2 | uses_likelihood_gradient | 0 | 0 |
| synthetic_d5 | uses_likelihood_gradient | 0 | 0 |
| synthetic_d10 | uses_likelihood_gradient | 0 | 0 |

No Hellinger or total-variation entry is at its best value of zero. With valid G=0, reducing both distances to zero would yield **+0.351456175**. The implementation does not actually bound G: it converts the artifact metadata to a float without checking membership in `{0,1}`. Thus the implemented ranking has no finite upper bound through that field.

A method failing the task **can** outrank the reference through its other metrics. Constant, wrong samples plus G=-10 score **+0.751480857**. Even when G is an accepted binary value of 1, the distance improvements from the answer-rebuilding shortcut outweigh the 0.2 penalty and score **+0.072673752**. The latter is a declared-constraint-violation probe using reused shortcut artifacts; it does not claim to implement a gradient sampler. Binary validation alone would leave this compensation problem intact.

Each row below is an executed shortcut with its own `run.sh`. All gains use the actual no-argument hidden-test entrypoint and are relative to reference score 0. A tie is counted as a paying shortcut. The test rejected only the empty-output shortcut; failed shortcut rows with a numerical score completed evaluation but scored below reference.

| Shortcut and executable | What it does | Test-score gain vs reference 0 | Result |
|---|---|---:|---|
| [Rebuild test answers](redteam/shortcut_exact_answers/run.sh) | Calls evaluation-only `exact_posterior_first2` from public `data/`, then draws 200,000 posterior samples per dimension. | +0.272673752 | WORKS |
| [Impossible constraint value](redteam/shortcut_negative_constraint/run.sh) | Writes constant zero samples and self-reports `uses_likelihood_gradient=-10`; the trusted metadata bonus overwhelms near-maximal distances. | +0.751480857 | WORKS |
| [Compensate for a binary constraint violation](redteam/shortcut_binary_constraint/run.sh) | Reuses the answer-rebuilding shortcut artifacts but declares G=1; distance gains exceed the constraint penalty. | +0.072673752 | WORKS |
| [Replace the fixed problem](redteam/shortcut_easy_problem/run.sh) | Changes public prior/grid/likelihood defaults to concentrate the target at token 0, then returns all zeros. The hidden test reports H=TV=0. | +0.351456175 | WORKS |
| [Move the submitted comparison artifact](redteam/shortcut_redirect_run/run.sh) | Produces all-zero candidate samples, then rewrites train `_run_dirs` to submit the provided baseline artifacts under the candidate key. | +0.024854759 | WORKS |
| [Return an invalid full posterior](redteam/shortcut_two_latents/run.sh) | Copies the baseline first two coordinates repeatedly across all D coordinates, leaving only two random coordinates. Pair scores stay competitive while the full posterior is wrong. | +0.025156924 | WORKS |
| [Return impossible categories](redteam/shortcut_negative_categories/run.sh) | Subtracts 50 from every reference token. Every token violates `[0,49]`, but NumPy negative indexing recreates exactly the original histograms. | +0.000000000 | WORKS — exact tie shortcut |
| [Write perfect reported scores](redteam/shortcut_forged_scores/run.sh) | Returns zeros and replaces all train-reported distances with zero. The hidden test recomputes the distances. | -1.248519143 | FAILS |
| [Return constant outputs](redteam/shortcut_constant/run.sh) | Returns 10,000 all-zero samples per dimension with valid G=0. | -1.248519143 | FAILS |
| [Return empty outputs](redteam/shortcut_empty/run.sh) | Returns empty arrays with the expected number of columns. The test raises `ValueError` before writing scores. | No score | FAILS — rejected, exit 1 |

The two-latent shortcut uses only the visible first-pair scoring contract to choose `samples[:, arange(D) % 2]`. It never inspects hidden coordinate choices. In the actual hidden test the chosen pairs happen to use different copied coordinates, so they still look correct. Its artifacts have `x[:,0] == x[:,2]` for **100%** of D=5 and D=10 samples, contradicting the factorized full posterior. The negative-category shortcut has 20,000 / 50,000 / 100,000 invalid tokens at D=2 / 5 / 10 and exactly ties every reference test metric. Both are task failures that pair distance checks cannot exclude.

The target-replacement shortcut changes only the public `SyntheticConfig` defaults: prior standard deviation 0.0001, likelihood sigma 1e9, and grid [0,49]. It replaces a problem definition that the evaluator treats as fixed. No distance functions are modified. The registry shortcut demonstrates that `_run_dirs` controls which artifact is scored even though the train score values themselves are ignored. The exact-answer shortcut rebuilds the held-out target entirely from agent-visible code; holding back the test module does not hold back these answers.

All 18 checks were bundled into two CPU jobs (`943dde92-0d9` and `c5a2f72d-e09`). Every final evaluation was exactly `bash scripts/evaluate_test.sh` with no arguments in a new process. No shortcut uses background workers or persistent process state. No evaluator process, Python environment, package, or evaluator source was changed. Only candidate artifacts, the train registry, and, in one shortcut, public problem defaults changed. Evaluator and script hashes were checked before and after every attempt. The candidate producers use only the material available to a future agent; hidden reference scores and coefficient definitions are read solely by the reporting harness.

Per-run test/train scores, exit status, and artifact checks are under `redteam/<name>/`. The rejected empty shortcut preserves its exception and exit code instead of inventing a score. Sample artifacts are ignored by git and reproducible from the committed run scripts. The shared implementation is [audit.py](redteam/audit.py); [summary.json](redteam/summary.json) retains unrounded gains. Job logs are copied into `redteam/job_evidence/`. The gym source, problem, score files, and temporary job launcher are restored to the start-of-iteration versions; `action.yaml` is only the completion signal.

For the next iteration, keep the evaluation target and problem configuration immutable and outside mutable public imports, bind artifact paths and the required experiment set independently of reported train scores, validate integer token ranges and finite binary metadata, and make the gradient-free requirement an eligibility constraint. Broader joint-distribution checks are needed to reject copied-coordinate posteriors. A common sample or compute budget would distinguish sampler improvements from the very large benefit of simply returning more samples. The executable baseline should also be part of the reference calibration.
