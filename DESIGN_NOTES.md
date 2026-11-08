> Superseded by `HARDENING.md` (iteration 11): the evaluation design described below (sample-split train/test, artifact registry, self-reported constraint flag) was replaced.

# Train/Test Evaluation Design Notes

Checked the split evaluators, wrapper scripts, and scoring schema for shortcut risk.

Changes made:

- Added a slice-specific coordinate-pair marginal score path. The visible train evaluator scores the first two coordinates; the held-out test evaluator scores a different coordinate pair for `D>2` while keeping the same metric keys. This makes a method that only tunes the visible first-two marginal less likely to transfer cleanly, while a genuine posterior sampler should remain stable because the synthetic target is exchangeable across coordinates.
- Added `coordinate_marginal_tv`, the mean one-coordinate posterior marginal total variation across all latent coordinates. This triangulates posterior quality beyond a single pairwise marginal and catches methods that put the right mass on one visible pair but neglect nuisance coordinates.
- Added required `description` fields to every nonzero metric entry in `scoring/reference.json`.
- Confirmed `scripts/evaluate_test.sh` invokes `eval/test/` and writes `scoring/scores_test.json`; `scripts/evaluate_train.sh` invokes `eval/train/` and writes `scoring/scores_train.json`.
- Confirmed the split evaluators read method artifacts from `samples.npz` and `metadata.json`; they do not regenerate the method inline.

Runtime concerns for the runner iteration:

- Existing `scoring/scores.json` was produced before `coordinate_marginal_tv` existed, so the next real evaluation run must regenerate scores before comparing against the updated reference schema.
- `coordinate_marginal_tv` may be low-variance for strong samplers because the synthetic posterior factorizes, but it still helps identify coordinate-neglect shortcuts.
- `synthetic_d2` cannot use a different held-out coordinate pair, so its train/test distinction comes only from the sample split and added marginal triangulation.
