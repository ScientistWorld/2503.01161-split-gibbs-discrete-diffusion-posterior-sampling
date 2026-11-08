"""Direct single-site Metropolis-Hastings on p_0(x) p(y|x), without split Gibbs.

This sampler uses the prior's clean marginal (the sigma -> 0 concrete score)
and the black-box likelihood. On the synthetic task the prior is exact, which
makes long-run MCMC a strong comparator rather than a weak baseline.

    python -m baseline.direct_mh_synthetic --problem PROBLEM.json --output-dir OUT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.synthetic_sgdd import SyntheticProblem, load_problem, log_likelihood, prior_probs_1d, save_samples


def run_direct_mh(problem: SyntheticProblem, steps: int, seed_offset: int = 0) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(problem.seed + seed_offset)
    log_prior = np.log(prior_probs_1d(problem) + 1e-300)
    n = problem.num_samples
    x = rng.integers(0, problem.vocab_size, size=(n, problem.dim), dtype=np.int64)
    current = log_likelihood(x, problem) + log_prior[x].sum(axis=1)
    rows = np.arange(n)
    accepted = 0
    for _ in range(steps):
        proposal = x.copy()
        cols = rng.integers(0, problem.dim, size=n)
        proposal[rows, cols] = rng.integers(0, problem.vocab_size, size=n)
        score = log_likelihood(proposal, problem) + log_prior[proposal].sum(axis=1)
        accept = np.log(rng.random(n) + 1e-300) < np.minimum(score - current, 0.0)
        x[accept] = proposal[accept]
        current[accept] = score[accept]
        accepted += int(accept.sum())
    return x, {"method": "direct_mh", "steps": steps, "acceptance": accepted / max(n * steps, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    problem = load_problem(args.problem)
    samples, meta = run_direct_mh(problem, args.steps, args.seed_offset)
    save_samples(args.output_dir, samples, meta)


if __name__ == "__main__":
    main()
