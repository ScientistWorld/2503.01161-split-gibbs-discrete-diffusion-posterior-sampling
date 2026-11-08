"""Visible train evaluation for the synthetic discrete posterior-sampling gym.

Runs every method registered as ``scripts/samplers/<name>.sh`` on the public
train problem instances below and scores the samples against exact posterior
pair marginals that this module computes from its own constants. Sampler
contract:

    bash scripts/samplers/<name>.sh PROBLEM_JSON OUTPUT_DIR

The sampler must write ``OUTPUT_DIR/samples.npz`` with an integer array
``samples`` of shape exactly ``(num_samples, dim)`` and tokens in
``[0, vocab_size)``. Any invalid output fails the method on every experiment
(distances set to 1.0). Problem files and sampler outputs are kept under
``checkpoints/train/<method>/<instance>/`` for inspection.

The held-out test runs the same samplers on different, hidden instances of the
same experiments with the same sample count.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SLICE_NAME = "train"
NUM_SAMPLES = 50000
VOCAB_SIZE = 50
PRIOR = {"type": "discretized_gaussian", "std": 2.0, "grid_min": -5.0, "grid_max": 5.0}
LIKELIHOOD_SIGMA = 3.0
SAMPLER_TIMEOUT_SECONDS = 3600
METRICS = ("hellinger", "total_variation", "all_pairs_hellinger")

# Public train instances. The paper's synthetic observation is y = |x - c| with
# x the discretization of 5/3, i.e. y = 8.5 on every coordinate.
TRAIN_INSTANCES = (
    {"experiment": "synthetic_d2", "instance_id": "train_d2", "dim": 2, "type": "coordinatewise", "observation": [8.5] * 2, "seed": 1002},
    {"experiment": "synthetic_d5", "instance_id": "train_d5", "dim": 5, "type": "coordinatewise", "observation": [8.5] * 5, "seed": 1005},
    {"experiment": "synthetic_d10", "instance_id": "train_d10", "dim": 10, "type": "coordinatewise", "observation": [8.5] * 10, "seed": 1010},
    {"experiment": "synthetic_l1_d10", "instance_id": "train_l1_d10", "dim": 10, "type": "l1_norm", "observation": 115.0, "seed": 1110},
)


class SamplerFailure(Exception):
    pass


def problem_spec(instance: dict, num_samples: int) -> dict:
    return {
        "instance_id": instance["instance_id"],
        "dim": instance["dim"],
        "vocab_size": VOCAB_SIZE,
        "num_samples": num_samples,
        "seed": instance["seed"],
        "prior": dict(PRIOR),
        "likelihood": {"type": instance["type"], "sigma": LIKELIHOOD_SIGMA, "observation": instance["observation"]},
    }


def stable_logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    top = np.max(x, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    return np.squeeze(top + np.log(np.exp(x - top).sum(axis=axis, keepdims=True)), axis=axis)


def softmax_from_logits(logits: np.ndarray) -> np.ndarray:
    weights = np.exp(logits - np.max(logits))
    return weights / weights.sum()


def discretized_gaussian(spec: dict) -> np.ndarray:
    grid = np.linspace(spec["prior"]["grid_min"], spec["prior"]["grid_max"], spec["vocab_size"])
    return softmax_from_logits(-0.5 * (grid / spec["prior"]["std"]) ** 2)


def exact_pair_marginals(spec: dict) -> dict[tuple[int, int], np.ndarray]:
    """Exact posterior marginal p(x_i, x_j | y) for every pair i < j."""
    dim, vocab = spec["dim"], spec["vocab_size"]
    sigma2 = spec["likelihood"]["sigma"] ** 2
    kind = spec["likelihood"]["type"]
    prior = discretized_gaussian(spec)
    log_prior = np.log(prior)
    radius = np.abs(np.arange(vocab) - (vocab - 1) / 2.0)
    pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    if kind == "coordinatewise":
        y = np.broadcast_to(np.asarray(spec["likelihood"]["observation"], dtype=np.float64), (dim,))
        single = [softmax_from_logits(log_prior - (radius - y[d]) ** 2 / (dim * sigma2)) for d in range(dim)]
        return {(i, j): np.outer(single[i], single[j]) for (i, j) in pairs}
    if kind == "l1_norm":
        y = float(spec["likelihood"]["observation"])
        base = radius.min()
        level = np.rint(radius - base).astype(np.int64)
        level_prob = np.bincount(level, weights=prior)
        others = np.ones(1)
        for _ in range(dim - 2):
            others = np.convolve(others, level_prob)
        with np.errstate(divide="ignore"):
            log_others = np.log(others)
        both = np.arange(2 * len(level_prob) - 1)
        norm = both[:, None] + np.arange(len(others))[None, :] + dim * base
        log_factor = stable_logsumexp(log_others[None, :] - (norm - y) ** 2 / sigma2, axis=1)
        joint = softmax_from_logits(log_prior[:, None] + log_prior[None, :] + log_factor[level[:, None] + level[None, :]])
        return {pair: joint for pair in pairs}
    raise ValueError(f"unknown likelihood type {kind}")


def histogram2d(samples: np.ndarray, i: int, j: int, vocab: int) -> np.ndarray:
    counts = np.bincount(samples[:, i] * vocab + samples[:, j], minlength=vocab * vocab)
    return (counts / len(samples)).reshape(vocab, vocab)


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, 1.0 - float(np.sqrt(p * q).sum()))))


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def metrics_for(samples: np.ndarray, spec: dict, marginals: dict) -> dict[str, float]:
    vocab = spec["vocab_size"]
    first = histogram2d(samples, 0, 1, vocab)
    per_pair = [hellinger_distance(histogram2d(samples, i, j, vocab), q) for (i, j), q in marginals.items()]
    return {
        "hellinger": hellinger_distance(first, marginals[(0, 1)]),
        "total_variation": tv_distance(first, marginals[(0, 1)]),
        "all_pairs_hellinger": float(np.mean(per_pair)),
    }


def read_samples(out_dir: Path, spec: dict) -> np.ndarray:
    path = out_dir / "samples.npz"
    if not path.is_file():
        raise SamplerFailure("no samples.npz written")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "samples" not in archive.files:
                raise SamplerFailure("samples.npz has no 'samples' array")
            samples = archive["samples"]
    except SamplerFailure:
        raise
    except Exception as exc:
        raise SamplerFailure(f"unreadable samples.npz ({type(exc).__name__})") from None
    want = (spec["num_samples"], spec["dim"])
    if samples.dtype == np.bool_ or not np.issubdtype(samples.dtype, np.integer):
        raise SamplerFailure(f"samples must be integer tokens, got dtype {samples.dtype}")
    if samples.shape != want:
        raise SamplerFailure(f"samples have shape {samples.shape}, expected exactly {want}")
    if samples.min() < 0 or samples.max() >= spec["vocab_size"]:
        raise SamplerFailure(f"tokens outside [0, {spec['vocab_size'] - 1}]")
    return samples.astype(np.int64)


def invoke(script: Path, spec: dict, run_dir: Path) -> np.ndarray:
    run_dir.mkdir(parents=True, exist_ok=True)
    problem_path = run_dir / "problem.json"
    problem_path.write_text(json.dumps(spec, indent=2) + "\n")
    out_dir = run_dir / "output"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir()
    log_file = run_dir / "sampler.log"
    with log_file.open("w") as log:
        try:
            result = subprocess.run(
                ["bash", str(script), str(problem_path), str(out_dir)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=SAMPLER_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            raise SamplerFailure("sampler timed out") from None
    if result.returncode != 0:
        print(log_file.read_text(errors="replace")[-2000:], file=sys.stderr)
        raise SamplerFailure(f"sampler exited with code {result.returncode}")
    return read_samples(out_dir, spec)


def registered_samplers(samplers_dir: Path) -> dict[str, Path]:
    return {
        script.stem: script
        for script in sorted(samplers_dir.glob("*.sh"))
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]*", script.stem)
    }


def run_all(samplers: dict[str, Path], num_samples: int, run_root: Path) -> dict:
    specs = [(inst["experiment"], problem_spec(inst, num_samples)) for inst in TRAIN_INSTANCES]
    marginals = {spec["instance_id"]: exact_pair_marginals(spec) for _, spec in specs}
    scores: dict = {"slice": SLICE_NAME, "num_samples": num_samples, "experiments": {}}
    failed: dict[str, str] = {}
    collected: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for name, script in samplers.items():
        print(f"[{SLICE_NAME}] {name}", flush=True)
        collected[name] = {}
        for experiment, spec in specs:
            try:
                samples = invoke(script, spec, run_root / name / spec["instance_id"])
            except SamplerFailure as exc:
                failed[name] = f"{spec['instance_id']}: {exc}"
                print(f"  FAILED {failed[name]}", flush=True)
                break
            collected[name].setdefault(experiment, {})[spec["instance_id"]] = metrics_for(
                samples, spec, marginals[spec["instance_id"]]
            )
    for experiment in sorted({exp for exp, _ in specs}):
        block = {"_slice": SLICE_NAME, "_failures": {}, "_per_instance": {}, "results": {}}
        for name in samplers:
            if name in failed:
                block["_failures"][name] = failed[name]
                block["results"][name] = {metric: 1.0 for metric in METRICS}
                continue
            rows = collected[name][experiment]
            block["_per_instance"][name] = rows
            block["results"][name] = {
                metric: float(np.mean([row[metric] for row in rows.values()])) for metric in METRICS
            }
        scores["experiments"][experiment] = block
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "scoring" / "scores_train.json")
    parser.add_argument("--samplers-dir", type=Path, default=ROOT / "scripts" / "samplers")
    parser.add_argument("--methods", nargs="+", default=None, help="score only these registered samplers")
    parser.add_argument("--run-root", type=Path, default=ROOT / "checkpoints" / "train")
    args = parser.parse_args()

    samplers = registered_samplers(args.samplers_dir)
    if args.methods:
        missing = sorted(set(args.methods) - set(samplers))
        if missing:
            raise SystemExit(f"Not registered under {args.samplers_dir}: {missing}")
        samplers = {name: samplers[name] for name in args.methods}
    if not samplers:
        raise SystemExit(f"No samplers registered under {args.samplers_dir}")
    scores = run_all(samplers, NUM_SAMPLES, args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
