# Evaluation

## Sampler contract

A method is any executable script registered as `scripts/samplers/<name>.sh`. The evaluators call it as

```
bash scripts/samplers/<name>.sh PROBLEM_JSON OUTPUT_DIR
```

from `/home/user`. `PROBLEM_JSON` describes one problem instance (`dim`, `vocab_size`, `num_samples`, `seed`, the prior, the likelihood type, `sigma` and the observation). `data/synthetic_sgdd.py` parses it (`load_problem`) and provides the prior marginals and `log_likelihood`. The sampler must write `OUTPUT_DIR/samples.npz` containing an integer array `samples` of shape exactly `[num_samples, dim]` with tokens in `[0, 49]`. Anything else the sampler writes is ignored.

## Train and test

- `bash scripts/evaluate_train.sh` runs every registered sampler on the public train instances in `eval/train/` and writes `scoring/scores_train.json` (`bash scripts/evaluate.sh` writes the same to `scoring/scores.json`). Use `--methods NAME ...` to score selected samplers only. Problem files, sampler logs and outputs are kept under `checkpoints/train/<method>/<instance>/`.
- `bash scripts/evaluate_test.sh` (no arguments) runs every registered sampler on hidden test instances. These are the same experiments with different observations and seeds, and the same sample count. It writes `scoring/scores_test.json`. Samplers are run fresh at test time; precomputed outputs are never read.

Every instance asks for 50,000 samples, and all methods receive identical problem files.

## Metrics (per experiment, averaged over its instances; lower is better)

- `hellinger`: Hellinger distance between the empirical joint histogram of coordinates 0 and 1 and their exact posterior marginal (50×50 cells).
- `total_variation`: total variation distance on the same marginal.
- `all_pairs_hellinger`: mean Hellinger distance over every coordinate pair `(i, j)`. This checks that the whole joint posterior is sampled, not only the reported pair.

The evaluators compute exact posterior marginals from their own copy of the problem definition. They do not use `data/`, `method/` or anything a sampler reports.

## Validity

A sampler that crashes, times out, writes no `samples.npz`, returns a non-integer array, a different number of samples, or any token outside `[0, 49]` on any instance fails. All its distances on every experiment are then set to 1.0, the worst possible value.

## Ranking

A method's score is `sum_experiment weight * sum_metric coefficient * (value - value_reference)`, with weights and coefficients in `scoring/reference.json`. The reference is `sgdd` on the hidden test. The included comparators are `direct_mh`, `svdd_pm` and `smc`.
