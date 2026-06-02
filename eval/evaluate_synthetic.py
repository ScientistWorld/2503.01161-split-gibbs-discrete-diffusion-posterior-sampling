"""Evaluate generated synthetic samples against the exact posterior marginal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.synthetic_sgdd import SyntheticConfig, empirical_first2, exact_posterior_first2, hellinger, total_variation


def _read_constraint_metadata(run_dir: Path, method: str) -> float:
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if "uses_likelihood_gradient" in metadata:
            return float(metadata["uses_likelihood_gradient"])
    if method in {"sgdd", "direct_mh"}:
        return 0.0
    raise ValueError(
        f"{run_dir} is missing metadata.json with uses_likelihood_gradient; "
        "future methods must declare this constraint metric explicitly."
    )


def _mean_coordinate_tv(samples: np.ndarray, target_1d: np.ndarray, vocab_size: int) -> float:
    distances = []
    for coord in range(samples.shape[1]):
        counts = np.bincount(samples[:, coord], minlength=vocab_size).astype(np.float64)
        empirical = counts / max(len(samples), 1)
        distances.append(total_variation(empirical, target_1d))
    return float(np.mean(distances))


def evaluate_run(run_dir: Path, config: SyntheticConfig, method: str) -> dict[str, float]:
    artifact = np.load(run_dir / "samples.npz")
    samples = artifact["samples"]
    target = exact_posterior_first2(config)
    empirical = empirical_first2(samples, config.vocab_size)
    target_1d = target.sum(axis=1)
    return {
        "hellinger": hellinger(empirical, target),
        "total_variation": total_variation(empirical, target),
        "coordinate_marginal_tv": _mean_coordinate_tv(samples, target_1d, config.vocab_size),
        "uses_likelihood_gradient": _read_constraint_metadata(run_dir, method),
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
        metrics = evaluate_run(Path(run_dir), config, method)
        scores["experiments"].setdefault(experiment, {"results": {}})
        scores["experiments"][experiment]["results"][method] = metrics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
