# Final hardening

The previous claim that the gym rewarded scientific progress was too strong. `REVIEW.md` found direct MH ahead of the reference (+0.052958, and +0.218386 with more steps), and an analytic posterior at the bound (+0.384). Those are structural failures of the small synthetic task. They are addressed together here by moving the scored gym to the paper's real, full-resolution MNIST inverse problems and original pretrained model. The completed serial audit (`cba2826c-061`, exit 0) now ranks the tested stronger variant above the reference beyond paired uncertainty, the weaker variant below it, and every replayed shortcut below it. The stronger variant gains +0.160067 (paired 95% interval [+0.031717, +0.291394]); the independent seed change gains +0.016194. This establishes measured headroom, with the limitations listed below.

## Changes decided from the reports

The active experiments are now the paper's XOR problem (γ=0.5) and its sparse AND study (γ=2, Appendix B.3). The 1,024-bit image dimension and original 90M SEDD EMA checkpoint are preserved. The reference uses the paper's K/T/H values, the Hamming coupling, discrete likelihood MH, and an annealed reverse-diffusion prior. This is the authors' released inference procedure, including its analytic predictor and final denoiser; the paper calls the prior solver Euler. It is not an end-to-end reconstruction surrogate or a small substitute prior.

Test images are real held-out MNIST with modest affine and elastic deformation. Images, private generation seeds, deformation details and answer-generation code stay in `eval/test/`. The method receives only measurements and the public input contract. An unprivileged Linux Landlock policy denies private file-content reads below `eval/test/` during the sampler subprocess, including the source images, generator and other methods’ reconstructions. Only that sampler’s own run directory is readable. This also closes copying an earlier reference output from the same evaluation job. Targets used for scoring are constructed in evaluator memory after inference returns. Public train uses a separate source partition and seed, independently implemented with no train/test imports. All methods see the same inputs within a comparison.

The original factorized/chain analytic posterior is no longer the task. Enumerating all 2^1024 binary images is infeasible, and a logical solution to the public measurements does not determine the hidden image. The audit also searches all 60,000 public MNIST training templates with its own copy of the measurement objective, and tests a short logical solver, rather than relying on a declared search counter. Public-image deformations make a finite source catalogue approximate. More sophisticated learned reconstruction and optimization remain expressible through the unrestricted sampler interface.

The evaluator recomputes whole-image PSNR, the paper's frozen-classifier consistency, and logical measurement error from raw binary reconstructions. It validates integer dtype, exact count/shape and category range before conversion. Method metadata, claimed metrics and train artifact registries are ignored. It owns every measurement, threshold, dimension, source partition, comparison classifier and operating point. Its classifier is loaded and hash-checked before inference, and targets do not import `data/` or `method/`.

Any crash, malformed output, impossible number or failed task gate receives PSNR 0, accuracy 0 and measurement error 1 in **every** scored experiment. Gates are fixed in advance: PSNR ≥6 dB, accuracy ≥20%, measurement error ≤10%. There are no missing-value escapes or auxiliary bonuses. Positive coefficients in `reference.json` are interpreted through `higher_is_better`; absolute coefficient magnitude is used, and improvement is divided by the per-metric population standard deviation across attempts. Failed runs cannot compensate their way above a solution. `eval/ranking.py` and the bundled correctness checks implement these rules.

No wall-clock timeout remains in the active method or evaluator path. All canonical methods and audit alternatives run serially with fixed inference counts and seeds. The frozen diffusion backbone and evaluation classifier are shared. Paired comparisons use the same 128 input images, including the same image indices across XOR and AND; the report includes paired bootstrap intervals and an independent reference sampling seed.

## Before/after evidence

Before values below are from `REDTEAM.md`, the previous `HARDENING.md`, and `REVIEW.md`; their raw records are retained in `redteam/final_hardening/before/` and `redteam/hardening/results/`. The old and new scalar scales differ, so before/after gains diagnose whether something pays rather than comparing their numerical sizes.

| Major finding or shortcut | Before gain | Fix | Final gain |
|---|---:|---|---:|
| Direct MH outranks reference | +0.052958; long +0.218386 | Original learned image prior and full paper inference. Direct MH rerun at the full K×T likelihood-step budget. | −4.516171 |
| Factorized/chain closed-form posterior | +0.384, at bound | Real images and learned prior replace the soluble target. Old exact producer errors; new logical and template solvers also tested. | old exact −4.516171; rule −0.670145; template −0.522951 |
| Copy previous sampler's reconstruction | Exact reference tie | Hidden artifacts inaccessible except current sampler directory. | −4.516171 |
| Public answer rebuilding | +0.272674 | Private deformed instances and generator; no public posterior answer helper. Old producer and replacement logical/template solvers tested. | −4.516171; −0.670145; −0.522951 |
| Impossible self-reported constraint value | +0.751481 | Ignore flags and reported metrics; forged constant output fails raw-image gates. | −4.516171 |
| Binary flag compensation | +0.072674; exact posterior later +0.384 | Remove flag rewards and analytic target; failed constraints receive all worst values. Gradient eligibility requires source review. | forged −4.516171; rule −0.670145 |
| Replace mutable public target | +0.351456 | Evaluator-owned in-memory problem ignores tampering with the input copy. | −4.516171 |
| Redirect `_run_dirs` registry | +0.024855 | Evaluator never reads method-written registries; forged replay rewrites the old registry. | −4.516171 |
| Copy only two latent coordinates | +0.025157 | Whole-image metrics and logical constraints; replay repeats two pixels. | −4.516171 |
| Negative category wraparound | 0, exact tie | Validate raw integer categories before conversion. | −4.516171 |
| More outputs improve histogram estimate | +0.205383 | Exact output count of one image per observation. | −4.516171 |
| Empty output avoids scores | Crash / missing score | Every metric gets its worst value. Float and NaN replays also fail. | −4.516171 each |
| Forged reported metrics / constant output | −1.248519 | Recompute every quantity from raw images. | −4.516171 each |

Honest alternatives are twice as many split iterations (`longer`), a different sampling seed (`seed1`), and fixed η=0.5 (`weaker`, the Appendix B.1 ablation). A predeclared fallback doubles both iterations and prior resolution (`longer_accurate`) if the first improvement does not clear the paired uncertainty/seed check. All changes are inference variants of the paper's procedure, on identical inputs. Before, longer SGDD gained +0.108 and fixed η lost −0.226, but direct MH and the analytic solver invalidated that headroom interpretation. After hardening, longer SGDD gains +0.126059, the fixed-coupling variant loses −1.670260, and the fallback gains +0.160067. Only the fallback clears the paired interval and observed seed-change check; it remains far from every metric bound.

## Headroom, weights and resource choices

The old three factorized experiments and the chain experiment are retired from ranking, not merely reweighted. Sparse AND is the added scored stress test from the paper's empirical study of missing information. XOR weight is 0.4; sparse AND weight is 0.6, emphasizing the learned prior's contribution when measurements are incomplete. PSNR coefficient 1.0 and classifier consistency 0.7 measure the paper's claimed image quality; measurement error 0.3 records the accompanying consistency constraint, enforced by a gate as well. No self-reported constraint flag or runtime is rewarded.

PSNR's finite best value is 30.103 dB (a one-pixel MSE floor for exact recovery); best accuracy is 1 and best measurement error is 0. The job reports which reference metrics, if any, reach these bounds. A positive gain from a tested honest variant beyond paired uncertainty and sampling-seed change is required: merely being below a bound does not establish headroom.

The paper uses 1,000 images; this test uses 128 to fit the final job and reduce paired uncertainty relative to a tiny test. Deformations prevent a public-source lookup answer key. The particle comparators retain M=20, MC=3 and their selection/resampling steps. The SVDD comparator is explicitly named `svdd_mc`: it estimates the Eq.55 soft value, rather than claiming the paper’s SVDD-PM posterior-mean approximation. Both use the released analytic prior kernel with one-step conditional draws (the small-H setting in Appendix B.2), and the release’s squared-noise likelihood convention. They use 16 reverse time steps rather than 128 because the full reference, both slices, alternatives and shortcut searches share a six-hour allocation. This baseline scaling can weaken these comparators, and is disclosed; direct MH is independently included at the reference's full likelihood-step budget. The reference K/T/H and pretrained model are not scaled down. No training is performed by the hidden test.

Published Table 2 values remain unchanged in the archived underscore-prefixed reference object. Original Table 4 PSNR/accuracy values are also stored under `_paper_table4_unchanged`; none is a target to fit on the altered test distribution. Active reference results are replaced only by the new canonical test measurements. `scores_train.json`, `scores_test.json` and `scores.json` contain only the reference and the three shipped comparators; shortcut scores remain in this report and audit records.

## Remaining scope and limitations

- Gradient use by arbitrary code cannot be inferred from an image. No flag claims to prove it. The reference is inference-only and gradient-free; the separate shortcut grader checks future attempts' eligibility.
- We do not defend against reading another process's memory, tampering with evaluator code, or changing the Python environment. Test-file reads are blocked during inference; broader malicious code review belongs to the separate grader.
- Private likelihood computation is permitted and explicitly exercised by template search and logical solving. No evaluator query counter is claimed to bind it. Exhaustive image enumeration is not feasible; richer legitimate reconstruction procedures remain allowed.
- This is an image reconstruction gym, not a calibration test for an entire posterior distribution. PSNR, classifier consistency and measurement fit match the paper's image study; a correct point estimate is an honest alternative on these quantities.
- The original DPS baseline and the paper's DNA/music/FFHQ studies are not reproduced here. The old synthetic code and red-team records are historical only.
- Bit-for-bit reproducibility across different accelerator architectures is not promised. Seeds and model parameters are fixed; no score depends on elapsed time or a wall-clock deadline.
- Exact restored-source lookup after optimizing arbitrary continuous deformations is not exhaustively ruled out. The finite-catalogue shortcut is tested. The tested logical and finite-catalogue procedures lose to both the reference and the honest improvement. Optimized continuous source lookup was not tested; this remains a limitation, rather than a proved exclusion of every possible search procedure.

## First bundled job and resumed verification

Job `620f95a2-5f2` completed the public-slice reference and all three comparators, then failed hidden inference before any samples were produced. Python package discovery requires directory enumeration at the workspace root; the original read policy granted public child directories but omitted the root's directory-list permission. Every hidden attempt therefore raised `ModuleNotFoundError`. Those tied, failed hidden scores do not establish scientific validity or headroom.

The policy now allows directory enumeration while continuing to deny hidden file contents, except the current sampler's own directory. The bundled regression check now imports the actual public method and audit packages under that policy, in addition to checking private-file denial. The audit refuses to freeze a failed reference. Launcher restoration now replaces its file atomically, avoiding the exit-127 shell error caused by overwriting a script still being read.

Completed public SGDD measurements were XOR PSNR 17.334907 dB, classifier consistency 0.851562, error 0.002136; sparse AND PSNR 12.424918 dB, consistency 0.656250, error 0.002407. Subsequent jobs recomputed these metrics from saved raw training reconstructions because neither the training procedure nor its evaluator changed. Hidden reference and comparator inference was rerun from the corrected interface; final evidence follows below.


## Worker loss and raw-output recovery

Job `ddc4395a-9bc` verified package discovery and private-file denial in the compute container. The hidden SGDD reference completed: XOR PSNR 17.978222 dB, consistency 0.859375, error 0.001770; sparse AND PSNR 12.520452 dB, consistency 0.632812, error 0.002026. Direct MH and SMC completed, and SVDD-MC completed XOR. The worker node then died during SVDD-MC's sparse AND inference (`Worker lost ... died`), before any honest variant or shortcut replay. This was infrastructure loss, not a method timeout or a numerical failure.

The recovery patch changes no model, data, method, metric, threshold or operating point. An explicit `--resume` option validates the existing raw array and exact input JSON before reuse, then recomputes every metric; it runs missing or malformed outputs. Default evaluation still runs every on-disk method afresh. Every registered method is initialized to all worst values before inference, so an interrupted procedure cannot disappear from the rank. Partial metric and paired-point files are now saved atomically after each method. The audit persists the original registry so a worker loss during temporary variant registration can also resume safely. No method-written score, flag or verdict is used to authorize recovery. Code and resources must remain unchanged to use it.

Job `cba2826c-061` reused the completed raw reference/comparator inference, finished SVDD-MC, and completed every predeclared honest variant and shortcut replay, including the conditional fallback. All metrics were recomputed. The reference and its comparators ran serially with nothing computing beside them; no scientific setting changed during recovery. The job passed bundled correctness checks and `python validate.py --compare`. The validator emitted two heuristic output-path warnings; both evaluators explicitly load `OUTPUT_DIR/samples.npz` in `load_valid_samples`, and malformed-output replays demonstrate that those raw files determine the score. The canonical four-method registry, both official split files, and end-to-end `scripts/run.sh` were restored.


## Final bundled job measurements

These gains use the final metric directions, coefficient magnitudes and population standard deviations across all attempts in this audit. Before/after scalar gains use different experiments and are not numerically comparable. Only the canonical four methods are retained in the official split score files.

| Method | Gain over final reference | Outcome |
|---|---:|---|
| longer_accurate | +0.160067 | valid |
| longer | +0.126059 | valid |
| seed1 | +0.016194 | valid |
| sgdd | +0.000000 | valid |
| shortcut_template | -0.522951 | valid |
| shortcut_rule | -0.670145 | valid |
| weaker | -1.670260 | valid |
| direct_mh | -4.516171 | failed, all worst values |
| shortcut_constant | -4.516171 | failed, all worst values |
| shortcut_copy_reference | -4.516171 | failed, all worst values |
| shortcut_empty | -4.516171 | failed, all worst values |
| shortcut_extra | -4.516171 | failed, all worst values |
| shortcut_float | -4.516171 | failed, all worst values |
| shortcut_forged | -4.516171 | failed, all worst values |
| shortcut_nan | -4.516171 | failed, all worst values |
| shortcut_negative | -4.516171 | failed, all worst values |
| shortcut_old_exact | -4.516171 | failed, all worst values |
| shortcut_tamper_problem | -4.516171 | failed, all worst values |
| shortcut_two_latents | -4.516171 | failed, all worst values |
| smc | -4.516171 | failed, all worst values |
| svdd_mc | -4.516171 | failed, all worst values |

Paired uncertainty (4,000 bootstrap resamples of the same image indices across every metric and experiment):

| Variant | Gain | Paired 95% interval | Mean PSNR changes (XOR; AND), dB |
|---|---:|---|---|
| longer | +0.126059 | [-0.006014, +0.256227] | mnist_xor: +0.6211; mnist_and_sparse: +0.4378 |
| longer_accurate | +0.160067 | [+0.031717, +0.291394] | mnist_xor: +0.7798; mnist_and_sparse: +0.4648 |
| seed1 | +0.016194 | [-0.138235, +0.156835] | mnist_xor: -0.1699; mnist_and_sparse: +0.0340 |
| weaker | -1.670260 | [-1.878400, -1.452214] | mnist_xor: -6.8691; mnist_and_sparse: -4.1003 |

The independent sampling-seed change has absolute gain 0.016194. Variants with a positive paired interval and a gain exceeding that seed change: longer_accurate.

Simple comparators below reference: True. Fixed-coupling weaker variant below reference: True. Replayed shortcuts below reference: True.

Canonical reference, before: gain 0 by definition on the archived synthetic task. After: gain 0 by definition on the final image task. Final raw quantities:

| Experiment | PSNR, dB | Class consistency | Measurement error |
|---|---:|---:|---:|
| mnist_xor | 17.978222 | 0.859375 | 0.001770 |
| mnist_and_sparse | 12.520452 | 0.632812 | 0.002026 |


The retired reference raw values were as follows (lower was better for all three distances). They are archived evidence, not comparable units to the new image metrics:

| Retired experiment | Hellinger | Total variation | All-pairs Hellinger |
|---|---:|---:|---:|
| synthetic_chain_d10 | 0.526595 | 0.536155 | 0.466180 |
| synthetic_d10 | 0.099147 | 0.096288 | 0.098118 |
| synthetic_d2 | 0.093243 | 0.085355 | 0.093243 |
| synthetic_d5 | 0.091508 | 0.087064 | 0.092157 |

Reference metrics at their best bound: none. The sparse AND experiment is the new headroom stress test; below-bound numbers alone are not counted as demonstrated headroom.

Shortcut replay results (the old analytic producer now errors because its factorized/chain model does not describe the new prior; logical and template solvers separately probe replacements):

| Replay | Final gain |
|---|---:|
| shortcut_constant | -4.516171 |
| shortcut_copy_reference | -4.516171 |
| shortcut_empty | -4.516171 |
| shortcut_extra | -4.516171 |
| shortcut_float | -4.516171 |
| shortcut_forged | -4.516171 |
| shortcut_nan | -4.516171 |
| shortcut_negative | -4.516171 |
| shortcut_old_exact | -4.516171 |
| shortcut_rule | -0.670145 |
| shortcut_tamper_problem | -4.516171 |
| shortcut_template | -0.522951 |
| shortcut_two_latents | -4.516171 |

Public/hidden reference comparison, with the same metrics and gates:

| Experiment | Train PSNR | Test PSNR | Train accuracy | Test accuracy | Train measurement error | Test measurement error |
|---|---:|---:|---:|---:|---:|---:|
| mnist_xor | 17.334907 | 17.978222 | 0.851562 | 0.859375 | 0.002136 | 0.001770 |
| mnist_and_sparse | 12.424918 | 12.520452 | 0.656250 | 0.632812 | 0.002407 | 0.002026 |


Raw comparator measurements before gate propagation show that the reference advantage is not created only by the failure policy. SMC and SVDD-MC miss the XOR consistency gate; direct MH misses reconstruction gates. Their official results are therefore all worst values, including sparse AND. Particle-comparator scaling is disclosed above.

| Experiment | Method | Raw PSNR | Raw class consistency | Raw measurement error |
|---|---|---:|---:|---:|
| mnist_xor | direct_mh | 3.053832 | 0.085938 | 0.016434 |
| mnist_xor | smc | 11.538274 | 0.562500 | 0.128326 |
| mnist_xor | svdd_mc | 11.497371 | 0.515625 | 0.129593 |
| mnist_and_sparse | direct_mh | 5.951310 | 0.125000 | 0.000000 |
| mnist_and_sparse | smc | 10.456002 | 0.453125 | 0.010029 |
| mnist_and_sparse | svdd_mc | 10.547057 | 0.507812 | 0.009968 |

Detailed measurements, failure explanations and paired image measurements are in `redteam/final_hardening/results/`. No timing measurements enter these scores. Training/test comparison is retained in `canonical_train.json` and `canonical_test.json`.
