"""Train-slice scoring for generated synthetic samples.

This evaluator is method-agnostic: it reads generated sample artifacts and
metadata from checkpoint directories, then scores a deterministic visible half
of each sample set against exact posterior marginals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.synthetic_sgdd import SyntheticConfig, exact_posterior_first2, hellinger, total_variation


SPLIT_KEY = 250301161
SLICE_NAME = "train"


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


def _sample_slice(samples: np.ndarray) -> np.ndarray:
    indices = np.argsort((np.arange(len(samples), dtype=np.int64) * 1103515245 + SPLIT_KEY) % (2**31))
    keep = indices[: len(indices) // 2]
    return samples[np.sort(keep)]


def _coordinate_pair(config: SyntheticConfig) -> tuple[int, int]:
    if config.dim < 2:
        raise ValueError("Synthetic evaluation requires at least two coordinates")
    return (0, 1)


def _empirical_pair(samples: np.ndarray, vocab_size: int, pair: tuple[int, int]) -> np.ndarray:
    hist = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    np.add.at(hist, (samples[:, pair[0]], samples[:, pair[1]]), 1.0)
    return hist / max(len(samples), 1)


def evaluate_run(run_dir: Path, config: SyntheticConfig, method: str) -> dict[str, float]:
    artifact = np.load(run_dir / "samples.npz")
    samples = artifact["samples"]
    samples = _sample_slice(samples)
    if len(samples) == 0:
        raise ValueError(f"{run_dir}/samples.npz has no samples in the {SLICE_NAME} slice")
    if samples.ndim != 2 or samples.shape[1] != config.dim:
        raise ValueError(f"{run_dir}/samples.npz has samples with shape {samples.shape}, expected (*, {config.dim})")
    target = exact_posterior_first2(config)
    empirical = _empirical_pair(samples, config.vocab_size, _coordinate_pair(config))
    return {
        "hellinger": hellinger(empirical, target),
        "total_variation": total_variation(empirical, target),
        "uses_likelihood_gradient": _read_constraint_metadata(run_dir, method),
    }


def _discover_runs(checkpoints_dir: Path, dims: list[int]) -> list[tuple[str, str, str]]:
    runs: list[tuple[str, str, str]] = []
    for dim in dims:
        suffix = f"_synthetic_d{dim}"
        for sample_path in sorted(checkpoints_dir.glob(f"*{suffix}/samples.npz")):
            run_dir = sample_path.parent
            method = run_dir.name[: -len(suffix)]
            if method:
                runs.append((f"synthetic_d{dim}", method, str(run_dir)))
    return runs


def _runs_from_scores(scores_path: Path, checkpoints_dir: Path) -> list[tuple[str, str, str]]:
    payload = json.loads(scores_path.read_text())
    runs: list[tuple[str, str, str]] = []
    for experiment, experiment_payload in sorted(payload.get("experiments", {}).items()):
        dim = int(experiment.rsplit("_d", 1)[1])
        results = experiment_payload.get("results", {})
        for method in sorted(results):
            run_dir = checkpoints_dir / f"{method}_synthetic_d{dim}"
            runs.append((experiment, method, str(run_dir)))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", nargs=3, metavar=("EXPERIMENT", "METHOD", "RUN_DIR"))
    parser.add_argument("--output", type=Path, default=Path("scoring/scores_train.json"))
    parser.add_argument("--vocab-size", type=int, default=50)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 5, 10])
    parser.add_argument("--from-scores", type=Path, default=Path("scoring/scores.json"))
    args = parser.parse_args()

    if args.run:
        runs = args.run
    elif args.from_scores.exists():
        runs = _runs_from_scores(args.from_scores, args.checkpoints_dir)
    else:
        runs = _discover_runs(args.checkpoints_dir, args.dims)
    if not runs:
        raise SystemExit(f"No generated sample artifacts found under {args.checkpoints_dir}.")

    scores: dict[str, object] = {"slice": SLICE_NAME, "experiments": {}}
    experiments = scores["experiments"]
    assert isinstance(experiments, dict)
    for experiment, method, run_dir in runs:
        dim = int(experiment.rsplit("_d", 1)[1])
        config = SyntheticConfig(dim=dim, vocab_size=args.vocab_size)
        metrics = evaluate_run(Path(run_dir), config, method)
        experiments.setdefault(experiment, {"_slice": SLICE_NAME, "_run_dirs": {}, "results": {}})
        experiments[experiment]["_run_dirs"][method] = str(run_dir)
        experiments[experiment]["results"][method] = metrics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
