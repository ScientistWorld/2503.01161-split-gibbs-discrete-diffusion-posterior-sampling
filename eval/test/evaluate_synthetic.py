"""Hidden-test evaluation for the synthetic discrete posterior-sampling gym.

Every method registered as ``scripts/samplers/<name>.sh`` is run at test time on
problem instances that exist only in this file (hidden observations and
seeds). Each sampler is called as

    bash scripts/samplers/<name>.sh PROBLEM_JSON OUTPUT_DIR

and must write ``OUTPUT_DIR/samples.npz`` holding an integer array ``samples``
of shape ``(num_samples, dim)`` with tokens in ``[0, vocab_size)``. All methods
see identical problem files (paired comparison). This module builds the
problem and its exact posterior marginals from its own constants; it does not
import ``data/`` or ``method/`` and never reads anything a sampler reports
about itself.

A method whose output is missing, unreadable, of the wrong size or dtype, or
contains an impossible token on any instance is scored as failed on every
experiment (each distance set to its worst value, 1.0).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SLICE_NAME = "test"
NUM_SAMPLES = 50000
VOCAB_SIZE = 50
PRIOR = {"type": "discretized_gaussian", "std": 2.0, "grid_min": -5.0, "grid_max": 5.0}
LIKELIHOOD_SIGMA = 3.0
SAMPLER_TIMEOUT_SECONDS = 3600
WORST = {"hellinger": 1.0, "total_variation": 1.0, "all_pairs_hellinger": 1.0}

# Hidden problem instances: (experiment, instance id, dim, likelihood type, observation, seed).
INSTANCES = (
    ("synthetic_d2", "t2a", 2, "coordinatewise", [9.5, 7.5], 61031),
    ("synthetic_d2", "t2b", 2, "coordinatewise", [7.5, 8.5], 74471),
    ("synthetic_d5", "t5a", 5, "coordinatewise", [8.5, 9.5, 7.5, 9.5, 8.5], 52183),
    ("synthetic_d5", "t5b", 5, "coordinatewise", [7.5, 7.5, 9.5, 8.5, 9.5], 90907),
    ("synthetic_d10", "t10a", 10, "coordinatewise", [9.5, 8.5, 7.5, 8.5, 9.5, 7.5, 8.5, 9.5, 8.5, 7.5], 38261),
    ("synthetic_d10", "t10b", 10, "coordinatewise", [7.5, 9.5, 8.5, 9.5, 7.5, 8.5, 9.5, 7.5, 8.5, 9.5], 27449),
    ("synthetic_l1_d10", "tl1a", 10, "l1_norm", 111.0, 83117),
    ("synthetic_l1_d10", "tl1b", 10, "l1_norm", 119.0, 45989),
)


class InvalidOutput(Exception):
    pass


def make_spec(instance: tuple, num_samples: int) -> dict:
    _, instance_id, dim, kind, observation, seed = instance
    return {
        "instance_id": instance_id,
        "dim": dim,
        "vocab_size": VOCAB_SIZE,
        "num_samples": num_samples,
        "seed": seed,
        "prior": dict(PRIOR),
        "likelihood": {"type": kind, "sigma": LIKELIHOOD_SIGMA, "observation": observation},
    }


# ---------------------------------------------------------------- exact targets


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    peak = np.max(values, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return np.squeeze(peak + np.log(np.sum(np.exp(values - peak), axis=axis, keepdims=True)), axis=axis)


def _prior_1d(spec: dict) -> np.ndarray:
    prior = spec["prior"]
    grid = np.linspace(prior["grid_min"], prior["grid_max"], spec["vocab_size"])
    logits = -0.5 * (grid / prior["std"]) ** 2
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()


def _normalize_log(log_probs: np.ndarray) -> np.ndarray:
    probs = np.exp(log_probs - log_probs.max())
    return probs / probs.sum()


def pair_targets(spec: dict) -> dict[tuple[int, int], np.ndarray]:
    """Exact posterior marginal of every coordinate pair (i < j)."""
    dim, vocab = spec["dim"], spec["vocab_size"]
    lik = spec["likelihood"]
    sigma2 = lik["sigma"] ** 2
    prior = _prior_1d(spec)
    log_prior = np.log(prior)
    dist = np.abs(np.arange(vocab) - (vocab - 1) / 2.0)
    pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
    if lik["type"] == "coordinatewise":
        obs = np.broadcast_to(np.asarray(lik["observation"], dtype=np.float64), (dim,))
        marginals = [
            _normalize_log(log_prior - (dist - obs[d]) ** 2 / (dim * sigma2)) for d in range(dim)
        ]
        return {(i, j): np.outer(marginals[i], marginals[j]) for i, j in pairs}
    if lik["type"] == "l1_norm":
        y = float(lik["observation"])
        offset = dist.min()
        steps = np.rint(dist - offset).astype(np.int64)
        step_prior = np.bincount(steps, weights=prior)
        rest = np.array([1.0])
        for _ in range(dim - 2):
            rest = np.convolve(rest, step_prior)
        with np.errstate(divide="ignore"):
            log_rest = np.log(rest)
        pair_steps = np.arange(2 * len(step_prior) - 1)
        totals = pair_steps[:, None] + np.arange(len(rest))[None, :] + dim * offset
        log_g = _logsumexp(log_rest[None, :] - (totals - y) ** 2 / sigma2, axis=1)
        log_pair = log_prior[:, None] + log_prior[None, :] + log_g[steps[:, None] + steps[None, :]]
        target = _normalize_log(log_pair)
        return {pair: target for pair in pairs}
    raise ValueError(f"unknown likelihood type {lik['type']}")


# ---------------------------------------------------------------- metrics


def _pair_histogram(samples: np.ndarray, pair: tuple[int, int], vocab: int) -> np.ndarray:
    codes = samples[:, pair[0]] * vocab + samples[:, pair[1]]
    return (np.bincount(codes, minlength=vocab * vocab) / len(samples)).reshape(vocab, vocab)


def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, 1.0 - float(np.sqrt(p * q).sum()))))


def _total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def score_samples(samples: np.ndarray, spec: dict, targets: dict) -> dict[str, float]:
    vocab = spec["vocab_size"]
    first = _pair_histogram(samples, (0, 1), vocab)
    all_pairs = [_hellinger(_pair_histogram(samples, pair, vocab), target) for pair, target in targets.items()]
    return {
        "hellinger": _hellinger(first, targets[(0, 1)]),
        "total_variation": _total_variation(first, targets[(0, 1)]),
        "all_pairs_hellinger": float(np.mean(all_pairs)),
    }


# ---------------------------------------------------------------- running samplers


def load_valid_samples(output_dir: Path, spec: dict) -> np.ndarray:
    path = output_dir / "samples.npz"
    if not path.is_file():
        raise InvalidOutput("no samples.npz written")
    try:
        with np.load(path, allow_pickle=False) as payload:
            if "samples" not in payload.files:
                raise InvalidOutput("samples.npz has no 'samples' array")
            samples = payload["samples"]
    except InvalidOutput:
        raise
    except Exception as exc:  # unreadable, pickled, or not an .npz archive
        raise InvalidOutput(f"unreadable samples.npz ({type(exc).__name__})") from None
    expected = (spec["num_samples"], spec["dim"])
    if samples.dtype == np.bool_ or not np.issubdtype(samples.dtype, np.integer):
        raise InvalidOutput(f"samples must be integer tokens, got dtype {samples.dtype}")
    if samples.shape != expected:
        raise InvalidOutput(f"samples have shape {samples.shape}, expected exactly {expected}")
    if samples.min() < 0 or samples.max() >= spec["vocab_size"]:
        raise InvalidOutput(f"tokens outside [0, {spec['vocab_size'] - 1}]")
    return samples.astype(np.int64)


def run_sampler(script: Path, spec: dict, instance_dir: Path) -> np.ndarray:
    instance_dir.mkdir(parents=True, exist_ok=True)
    spec_path = instance_dir / "problem.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    output_dir = instance_dir / "output"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir()
    log_path = instance_dir / "sampler.log"
    with log_path.open("w") as log:
        try:
            proc = subprocess.run(
                ["bash", str(script), str(spec_path), str(output_dir)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=SAMPLER_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            raise InvalidOutput("sampler timed out") from None
    if proc.returncode != 0:
        tail = log_path.read_text(errors="replace")[-2000:]
        print(f"  sampler exited with {proc.returncode}; log tail:\n{tail}", file=sys.stderr)
        raise InvalidOutput(f"sampler exited with code {proc.returncode}")
    return load_valid_samples(output_dir, spec)


def discover_samplers(samplers_dir: Path) -> dict[str, Path]:
    found = {}
    for script in sorted(samplers_dir.glob("*.sh")):
        name = script.stem
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]*", name):
            found[name] = script
    return found


def evaluate(samplers: dict[str, Path], num_samples: int, workdir: Path, instances=INSTANCES) -> dict:
    specs = [(inst[0], make_spec(inst, num_samples)) for inst in instances]
    targets = {spec["instance_id"]: pair_targets(spec) for _, spec in specs}
    experiments = sorted({exp for exp, _ in specs})
    per_method: dict[str, dict[str, list[dict[str, float]]]] = {}
    failures: dict[str, str] = {}
    for method, script in samplers.items():
        print(f"[{SLICE_NAME}] {method}", flush=True)
        per_method[method] = {exp: [] for exp in experiments}
        for exp, spec in specs:
            try:
                samples = run_sampler(script, spec, workdir / method / spec["instance_id"])
            except InvalidOutput as exc:
                failures[method] = f"{spec['instance_id']}: {exc}"
                print(f"  FAILED {failures[method]}", flush=True)
                break
            per_method[method][exp].append(score_samples(samples, spec, targets[spec["instance_id"]]))
    scores: dict = {"slice": SLICE_NAME, "num_samples": num_samples, "experiments": {}}
    for exp in experiments:
        count = sum(1 for e, _ in specs if e == exp)
        entry = {"_slice": SLICE_NAME, "_instances": count, "_failures": {}, "results": {}}
        for method in samplers:
            if method in failures:
                entry["_failures"][method] = failures[method]
                entry["results"][method] = dict(WORST)
                continue
            rows = per_method[method][exp]
            entry["results"][method] = {key: float(np.mean([row[key] for row in rows])) for key in WORST}
        scores["experiments"][exp] = entry
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "scoring" / "scores_test.json")
    parser.add_argument("--samplers-dir", type=Path, default=ROOT / "scripts" / "samplers")
    args = parser.parse_args()

    samplers = discover_samplers(args.samplers_dir)
    if not samplers:
        raise SystemExit(f"No samplers registered under {args.samplers_dir}")
    workdir = Path(tempfile.mkdtemp(prefix="synthetic_test_"))
    try:
        scores = evaluate(samplers, NUM_SAMPLES, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
