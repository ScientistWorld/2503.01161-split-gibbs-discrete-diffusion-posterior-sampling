"""Direct Metropolis-Hastings baseline without a diffusion prior step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.synthetic_sgdd import SyntheticConfig, log_likelihood, prior_probs_1d


def run_direct_mh(config: SyntheticConfig, num_samples: int, steps: int, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    prior = prior_probs_1d(config)
    x = rng.integers(0, config.vocab_size, size=(num_samples, config.dim), dtype=np.int64)
    current = log_likelihood(x, config) + np.log(prior[x] + 1e-300).sum(axis=1)
    accepted = 0

    for _ in range(steps):
        proposal = x.copy()
        rows = np.arange(num_samples)
        cols = rng.integers(0, config.dim, size=num_samples)
        proposal[rows, cols] = rng.integers(0, config.vocab_size, size=num_samples)
        score = log_likelihood(proposal, config) + np.log(prior[proposal] + 1e-300).sum(axis=1)
        accept = np.log(rng.random(num_samples) + 1e-300) < np.minimum(score - current, 0.0)
        x[accept] = proposal[accept]
        current[accept] = score[accept]
        accepted += int(accept.sum())

    return x, {"acceptance": accepted / max(num_samples * steps, 1), "steps": steps, "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/direct_mh_synthetic_d2"))
    args = parser.parse_args()

    config = SyntheticConfig(dim=args.dim, vocab_size=args.vocab_size)
    samples, meta = run_direct_mh(config, args.num_samples, args.steps, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "samples.npz", samples=samples)
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
