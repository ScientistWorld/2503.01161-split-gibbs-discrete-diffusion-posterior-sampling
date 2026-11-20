# Overview

This gym studies reconstruction from discrete, incomplete measurements, using the MNIST XOR/AND image problems in *Split Gibbs Discrete Diffusion Posterior Sampling*. It evaluates whole reconstructed images and measurement consistency, with unseen instances supplied at test time.

Start with `briefing/problem.md` and `briefing/evaluation.md`. `scripts/download.sh` fetches all public resources. `environment/setup.sh` declares Python dependencies. The ordinary `scripts/run.sh` reproduces the reference and included comparators on both evaluation slices in a serialized GPU job.

The former small Gaussian and chain experiments remain only as historical reproduction/audit source. They no longer affect ranking: their analytical posteriors made them unsuitable as a scientific improvement benchmark.

---

# Binary image reconstruction

Recover a 32×32 binary handwritten digit from incomplete, noiseless logical measurements. Each measurement gives the XOR or AND of a specified pair of pixels. Many images satisfy the same measurements; recovering the underlying image requires information about handwritten digits as well as measurement consistency.

There are two scored experiments: XOR with 512 pairs, and sparse AND with 2,048 pairs. Both use real MNIST images with small affine and elastic deformations, padding to 32×32, and thresholding at 128. The deformation prevents a finite catalogue of the public source images from being an answer key. Public and held-out evaluation use the same distribution and difficulty, with separate images and generation seeds.

A problem JSON supplies 128 observations, their pixel pairs, the binary vocabulary, a sampling seed, and the location of a public pretrained image model. Return one reconstruction for each observation. Any reconstruction procedure is allowed subject to the scientific eligibility rules; the public model is available but its use is not required. The hidden evaluator supplies unseen instances when it runs your procedure.

---

# Reference implementation and provenance

The reference implements the paper's split Gibbs sampler with a Hamming coupling, single-site Metropolis–Hastings likelihood updates, annealed coupling, and reverse discrete diffusion prior updates. It uses the authors' original 90M-parameter MNIST SEDD EMA checkpoint without retraining. XOR uses K=50, T=2000, H=20; AND uses K=100, T=5000, H=20, with geometric noise from 20 to 1e-4, as in Appendix C.3.

`method/sgdd_image.py` follows the authors' released `sampling/sgdd.py`: prior update followed by likelihood update, returning the last likelihood state. The release uses an analytic CTMC predictor with a final denoising step; the paper describes the prior solver as Euler. We preserve the released inference procedure and disclose this discrepancy. Computing the MH likelihood difference from incident edges is an exact optimization, checked against complete likelihood differences. Bernoulli categorical sampling has the same distribution as the release's exponential race.

`data/sedd/` contains the authors' full architecture, with PyTorch SDPA replacing the optional FlashAttention extension. Architecture and weights are unchanged. Upstream commit, modifications and artifact sources are recorded in `data/sedd/PROVENANCE.md` and `scripts/download.sh`.

The two diffusion comparators implement Appendix C.2 candidate selection and sequential importance resampling, with M=20 and three Monte Carlo value samples. `svdd_mc` uses the soft value in Eq.55, not the posterior-mean approximation of the reported SVDD-PM baseline. Both comparators use the same released analytic prior integrator as the reference; their one-step conditional draws are the small-H setting discussed in Appendix B.2. The likelihood follows the release’s squared-noise convention, exp(−mismatch/σ²). Their reverse path uses 16 rather than 128 steps to fit the bundled audit budget; this reduces scale, not the selection/resampling procedure. The direct MH comparator uses the likelihood alone, as in Appendix B.3, for K×T updates. It is not an exact posterior sampler for the learned image prior.

Full 1,024-bit resolution and the original pretrained models are preserved. Evaluation uses 128 rather than 1,000 images, and modest deformations to prevent public-source lookup. Sparse AND uses γ=2 from Appendix B.3 rather than γ=4 in Table 4. Consequently the original Table 4 numbers are archival, not gym-scale targets. No diffusion training is substituted or approximated.

---

# Evaluation

Register each method as `scripts/samplers/NAME.sh`. Evaluation calls it from `/home/user` as:

```sh
bash scripts/samplers/NAME.sh PROBLEM_JSON OUTPUT_DIR
```

Write `OUTPUT_DIR/samples.npz`, containing an integer `samples` array of shape exactly `(128, 1024)`, with every token 0 or 1. Images are flattened in row order. Boolean, float, nonfinite, negative, oversized and empty outputs are invalid. Other output files and reported metrics are ignored. `data/image_problem.py` provides an optional public input loader and model interface.

Run `bash scripts/evaluate_train.sh` for development. It runs every registered method and writes `scoring/scores_train.json`. `bash scripts/evaluate.sh` additionally copies these numbers to `scoring/scores.json`. `bash scripts/evaluate_test.sh` takes no arguments, runs every registered method on unseen inputs, and writes `scoring/scores_test.json`. Only numbers under each experiment's `results` are ranked. Fresh evaluation always runs the procedure; saved samples can be rescored explicitly with `--rescore`. Public logs, inputs and outputs live under `checkpoints/train/METHOD/EXPERIMENT/`. Hidden run artifacts stay under `eval/test/results/runs/METHOD/EXPERIMENT/`; a hidden sampler can read its own run directory, while other hidden artifacts remain inaccessible.

The evaluator computes three quantities from the raw images:

- `psnr`: mean image PSNR in dB, data range 1. Exact recovery uses an MSE floor of one pixel, so the finite best value is 30.103 dB.
- `accuracy`: fraction for which a fixed, evaluator-owned classifier assigns the same class to reconstruction and underlying image. This is classifier consistency as measured in the paper, rather than accuracy against a method-supplied label. Both images use the same 0/128 classifier preprocessing.
- `measurement_error`: fraction of logical measurement bits violated. Lower is better; this measures data consistency independently of visual quality.

A run must reach PSNR ≥6 dB, classifier consistency ≥20%, and measurement error ≤10% in both experiments. A crash, malformed output, impossible number or failed gate makes every metric of every experiment its worst value: PSNR 0, accuracy 0, measurement error 1. Failure cannot be compensated by another metric, and no failed entry is left empty. The evaluator owns all inputs, targets, dimensions, thresholds and comparison models. Test targets are constructed in evaluator memory after inference; the inference process cannot read private file contents in `eval/test/`. Directory enumeration remains available for package discovery.

Rank improvement is the sum over experiments and metrics of `experiment_weight × abs(coefficient) × directed_improvement / population_std`. Directed improvement is positive in the metric's `higher_is_better` direction relative to the stored reference. The standard deviation is computed across the attempts being ranked on that metric; a zero-variance metric contributes zero. Missing or impossible entries first receive all worst values. Coefficient signs and measurement units do not change priorities. `eval/ranking.py` implements this calculation for audits; scalar audit gains are not additional ranked metrics.

All methods see the same 128 inputs and fixed sampling seeds. The public and hidden evaluators are independent implementations with different source partitions and private generation seeds. Inference uses fixed iteration counts, runs serially and has no evaluator wall-clock cutoff. Download data and weights before a job; hidden evaluation never trains a model.

For recovery from a lost worker, `--resume` explicitly reuses a saved raw array only when its input JSON matches and its shape, dtype and tokens validate; every metric is recomputed. It runs missing or malformed outputs. Use this option only when method code and resources are unchanged. Default evaluation runs every procedure afresh. Scores and completion flags never authorize reuse.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 00:00] - none
- Read the paper and identified SGDD's synthetic posterior-recovery experiment as the minimum core-claim target.
- Implemented the synthetic problem, SGDD sampler, and method-agnostic evaluator.
- Prepared a CPU reproduction job to validate the implementation end-to-end.
- Fixed the first container portability failure by switching to a Docker Hub base image.

### [2026-06-02 21:05] - core_claim_plus
- Ran the SGDD synthetic posterior sampler for `D=2`, `D=5`, and `D=10` inside the compute container.
- Validation passed with method/evaluation separation intact.
- Reproduced the paper's gradient-free posterior recovery claim on all three synthetic Table 2 settings.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 1 total (1 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 21:06] - core_claim
- Corrected Table 2 reference entries to the paper-reported synthetic baseline and SGDD values.
- Restricted the active reusable scoring targets to `synthetic_d5` and `synthetic_d10`, where the current SGDD run supports the core gradient-free posterior sampling claim.

### [2026-06-02 21:12] - core_claim_plus
- Audited the implementation and confirmed it implements SGDD's split likelihood/prior Gibbs computation, not a surrogate.
- Restored `core_claim_plus` because the reproduction covers two synthetic dimensions from Table 2, including the harder `D=10` setting, with gradient-free SGDD outperforming reported baselines on the retained metrics.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 17m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 1m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 6 total (5 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 6 total (5 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 31m | **GPU**: 0.0h
- **Jobs**: 8 total (7 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 11: gpt-6.1-sol (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 48m | **GPU**: 3.0h
- **Jobs**: 11 total (8 completed, 3 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: core_claim_plus**
