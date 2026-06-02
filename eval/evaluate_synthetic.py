"""Evaluate generated synthetic samples against the exact posterior marginal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.synthetic_sgdd import SyntheticConfig, empirical_first2, exact_posterior_first2, hellinger, total_variation


def evaluate_run(run_dir: Path, config: SyntheticConfig) -> dict[str, float]:
    artifact = np.load(run_dir / "samples.npz")
    samples = artifact["samples"]
    target = exact_posterior_first2(config)
    empirical = empirical_first2(samples, config.vocab_size)
    return {
        "hellinger": hellinger(empirical, target),
        "total_variation": total_variation(empirical, target),
        "uses_likelihood_gradient": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", nargs=3, metavar=("EXPERIMENT", "METHOD", "RUN_DIR"), required=True)
    parser.add_argument("--output", type=Path, default=Path("scoring/scores.json"))
    parser.add_argument("--vocab-size", type=int, default=50)
    args = parser.parse_args()

    scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {"experiments": {}}
    for experiment, method, run_dir in args.run:
        dim = int(experiment.rsplit("_d", 1)[1])
        config = SyntheticConfig(dim=dim, vocab_size=args.vocab_size)
        metrics = evaluate_run(Path(run_dir), config)
        scores["experiments"].setdefault(experiment, {"results": {}})
        scores["experiments"][experiment]["results"][method] = metrics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
