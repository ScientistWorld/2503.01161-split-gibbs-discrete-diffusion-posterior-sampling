"""Public problem definition for the synthetic discrete posterior-sampling task.

This module is method-agnostic. It turns a problem specification (a JSON file
written by the evaluator, see ``load_problem``) into the two ingredients a
posterior sampler is allowed to use:

* the prior over each categorical coordinate (a discretized Gaussian), whose
  uniform-kernel noised marginals ``noised_prior_probs`` give the closed-form
  concrete score of the synthetic discrete diffusion prior;
* the observation log-likelihood ``log_likelihood`` of full token sequences.

It deliberately contains no posterior, no evaluation targets and no metrics.
The evaluator builds its own copy of the problem from the specification, so
editing this file changes what a sampler sees, not what it is scored against.

Specification fields (all written by the evaluator):

``instance_id``      opaque name of the problem instance
``dim``              number of categorical coordinates D
``vocab_size``       number of categories per coordinate N
``num_samples``      exact number of samples the sampler must return
``seed``             seed the sampler should use (all methods get the same one)
``prior``            {"type": "discretized_gaussian", "std", "grid_min", "grid_max"}
``likelihood``       {"type": "coordinatewise" | "chain", "sigma", "observation"}

``coordinatewise``: observation is a length-D vector y and
    log p(y|x) = -mean_d (|x_d - c| - y_d)^2 / sigma^2
    where c = (N - 1) / 2 is the centre token;
``chain``: observation is a length-(D-1) vector y of distances between
    neighbouring coordinates and
    log p(y|x) = -sum_d (|x_{d+1} - x_d| - y_d)^2 / sigma^2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LIKELIHOOD_TYPES = ("coordinatewise", "chain")


@dataclass(frozen=True)
class SyntheticProblem:
    instance_id: str
    dim: int
    vocab_size: int
    num_samples: int
    seed: int
    prior_std: float
    grid_min: float
    grid_max: float
    likelihood_type: str
    likelihood_sigma: float
    observation: tuple[float, ...]

    @property
    def center(self) -> float:
        return (self.vocab_size - 1) / 2.0


def problem_from_spec(spec: dict) -> SyntheticProblem:
    prior = spec["prior"]
    likelihood = spec["likelihood"]
    if prior.get("type") != "discretized_gaussian":
        raise ValueError(f"Unsupported prior type: {prior.get('type')}")
    kind = likelihood["type"]
    if kind not in LIKELIHOOD_TYPES:
        raise ValueError(f"Unsupported likelihood type: {kind}")
    dim = int(spec["dim"])
    obs = np.atleast_1d(np.asarray(likelihood["observation"], dtype=np.float64))
    if kind == "coordinatewise":
        if obs.size == 1:
            obs = np.full(dim, float(obs[0]))
        if obs.size != dim:
            raise ValueError("coordinatewise observation must have one value per coordinate")
    elif obs.size != dim - 1:
        raise ValueError("chain observation must have one value per neighbouring pair (dim - 1)")
    return SyntheticProblem(
        instance_id=str(spec.get("instance_id", "instance")),
        dim=dim,
        vocab_size=int(spec["vocab_size"]),
        num_samples=int(spec["num_samples"]),
        seed=int(spec["seed"]),
        prior_std=float(prior["std"]),
        grid_min=float(prior["grid_min"]),
        grid_max=float(prior["grid_max"]),
        likelihood_type=kind,
        likelihood_sigma=float(likelihood["sigma"]),
        observation=tuple(float(v) for v in obs),
    )


def load_problem(path: str | Path) -> SyntheticProblem:
    return problem_from_spec(json.loads(Path(path).read_text()))


def grid_values(problem: SyntheticProblem) -> np.ndarray:
    return np.linspace(problem.grid_min, problem.grid_max, problem.vocab_size, dtype=np.float64)


def prior_probs_1d(problem: SyntheticProblem) -> np.ndarray:
    """Prior probability of each token for one coordinate (coordinates are i.i.d.)."""
    logits = -0.5 * (grid_values(problem) / problem.prior_std) ** 2
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()


def noised_prior_probs(prior_1d: np.ndarray, sigma: float) -> np.ndarray:
    """Uniform-kernel forward marginal p_sigma = e^-sigma p_0 + (1 - e^-sigma) / N.

    Ratios p_sigma(v) / p_sigma(c) are the concrete score of the synthetic
    discrete diffusion prior at noise level sigma.
    """
    keep = np.exp(-sigma)
    return keep * prior_1d + (1.0 - keep) / len(prior_1d)


def forward_operator(tokens: np.ndarray, problem: SyntheticProblem) -> np.ndarray:
    """G(x): |x_d - c| per coordinate, or |x_{d+1} - x_d| per neighbouring pair for chain."""
    values = np.asarray(tokens, dtype=np.float64)
    if problem.likelihood_type == "chain":
        return np.abs(np.diff(values, axis=-1))
    return np.abs(values - problem.center)


def log_likelihood(tokens: np.ndarray, problem: SyntheticProblem) -> np.ndarray:
    """log p(y | x) for an array of token sequences with trailing axis D."""
    g = forward_operator(tokens, problem)
    if problem.likelihood_type == "chain":
        loss = ((g - np.asarray(problem.observation)) ** 2).sum(axis=-1)
    else:
        loss = ((g - np.asarray(problem.observation)) ** 2).mean(axis=-1)
    return -loss / problem.likelihood_sigma**2


def save_samples(output_dir: str | Path, samples: np.ndarray, metadata: dict | None = None) -> None:
    """Write the sampler output contract: ``samples.npz`` with an integer array."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "samples.npz", samples=np.asarray(samples, dtype=np.int64))
    if metadata is not None:
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
