"""Public, method-agnostic interface to the gym's posterior-sampling problems.

A problem file (JSON, written by the evaluator) describes one experiment: a
batch of observations ``y_1..y_I`` of binary 14x14 images, each produced from
an unknown image drawn from the prior, and the number of posterior samples to
return per observation. ``load_problem`` parses it into the two ingredients a
sampler may use:

* ``BernoulliMixturePrior`` - the clean-data distribution p_0 of a discrete
  diffusion model with the uniform transition kernel on {0, 1}. Use it through
  its diffusion-model interface: ``noised_probs`` / ``log_prob(x, sigma)``,
  the concrete score ``flip_ratios(x, sigma)`` = p_sigma(x with bit d flipped)
  / p_sigma(x), the reverse kernel ``reverse_sample(x_t, sigma_t, sigma_s)``
  (``sigma_s = 0`` is the denoiser p(x_0 | x_t)) and unconditional ``sample``.
* ``PairwiseLikelihood`` - the black-box likelihood. The forward model takes P
  fixed pixel pairs (i_p, j_p) and returns x_i XOR x_j (``xor``) or x_i AND x_j
  (``and``) for each pair (paper Eq. 15). Each measured bit is flipped
  independently with probability ``flip_prob``, so
  log p(y | x) = P log(1 - eps) - log((1 - eps) / eps) * #mismatches(G(x), y).

Problem file fields: ``experiment``, ``dim`` (196), ``image_shape``,
``vocab_size`` (2), ``num_observations`` (I), ``num_samples`` (S, per
observation), ``seed``, ``prior`` {``type``: ``bernoulli_mixture``,
``weights`` [M], ``probs`` [M][D]} and ``likelihood`` {``type``: ``xor`` |
``and``, ``pairs`` [P][2], ``flip_prob``, ``observations`` [I][P]}.

A sampler returns an integer array of shape exactly (I, S, D) with values in
{0, 1}; ``save_samples`` writes it in the expected format. This module holds no
evaluation targets and no metrics; the evaluators keep their own copy of the
prior and generate their own instances.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    peak = np.max(values, axis=axis, keepdims=True)
    out = peak + np.log(np.sum(np.exp(values - peak), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


@dataclass(frozen=True)
class BernoulliMixturePrior:
    """p_0(x) = sum_m w_m prod_d theta_md^x_d (1 - theta_md)^(1 - x_d), x in {0,1}^D."""

    weights: np.ndarray  # (M,)
    probs: np.ndarray  # (M, D)

    vocab_size: int = 2

    @property
    def dim(self) -> int:
        return int(self.probs.shape[1])

    @property
    def num_components(self) -> int:
        return int(self.probs.shape[0])

    def noised_probs(self, sigma: float) -> np.ndarray:
        """P(x_sigma,d = 1 | component m) under the uniform kernel: e^-s theta + (1 - e^-s) / 2."""
        keep = float(np.exp(-sigma))
        return keep * self.probs + (1.0 - keep) / 2.0

    def component_log_joint(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """log w_m + log p_sigma(x | m) for every chain and component, shape (C, M)."""
        q1 = self.noised_probs(sigma)
        log_q1, log_q0 = np.log(q1), np.log1p(-q1)
        xf = np.asarray(x, dtype=np.float64)
        return xf @ (log_q1 - log_q0).T + log_q0.sum(1) + np.log(self.weights)

    def log_prob(self, x: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        """log p_sigma(x), shape (C,)."""
        return logsumexp(self.component_log_joint(x, sigma), axis=1)

    def responsibilities(self, x: np.ndarray, sigma: float) -> np.ndarray:
        log_joint = self.component_log_joint(x, sigma)
        log_joint -= log_joint.max(1, keepdims=True)
        resp = np.exp(log_joint)
        return resp / resp.sum(1, keepdims=True)

    def flip_ratios(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """Concrete score: p_sigma(x with bit d flipped) / p_sigma(x), shape (C, D)."""
        q1 = self.noised_probs(sigma)
        resp = self.responsibilities(x, sigma)
        up = resp @ (q1 / (1.0 - q1))  # x_d = 0 -> 1
        down = resp @ ((1.0 - q1) / q1)  # x_d = 1 -> 0
        return np.where(np.asarray(x) == 1, down, up)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        comp = rng.choice(self.num_components, size=n, p=self.weights)
        return (rng.random((n, self.dim)) < self.probs[comp]).astype(np.uint8)

    def reverse_sample(
        self, x_t: np.ndarray, sigma_t: float, sigma_s: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Exact draw x_s ~ p(x_s | x_t) for 0 <= sigma_s < sigma_t (sigma_s = 0: denoiser)."""
        x_t = np.asarray(x_t)
        resp = self.responsibilities(x_t, sigma_t)
        cdf = np.cumsum(resp, axis=1)
        u = rng.random((len(x_t), 1)) * cdf[:, -1:]
        comp = np.minimum((u > cdf).sum(1), self.num_components - 1)
        q_s = self.noised_probs(sigma_s)[comp]
        q_t = self.noised_probs(sigma_t)[comp]
        keep = float(np.exp(-(sigma_t - sigma_s)))
        stay = keep + (1.0 - keep) / 2.0  # k(c | c)
        move = (1.0 - keep) / 2.0  # k(c | 1 - c)
        # P(x_s = 1 | x_t = c, m) = q_s(1) k(c | 1) / q_t(c)
        k_given_one = np.where(x_t == 1, stay, move)
        q_t_c = np.where(x_t == 1, q_t, 1.0 - q_t)
        p_one = q_s * k_given_one / q_t_c
        return (rng.random(x_t.shape) < p_one).astype(np.uint8)


@dataclass(frozen=True)
class PairwiseLikelihood:
    kind: str  # "xor" | "and"
    pairs: np.ndarray  # (P, 2)
    flip_prob: float
    observations: np.ndarray  # (I, P) uint8

    @property
    def num_pairs(self) -> int:
        return int(self.pairs.shape[0])

    @property
    def strength(self) -> float:
        """log((1 - eps) / eps): log-likelihood cost of one mismatched measurement."""
        return float(np.log1p(-self.flip_prob) - np.log(self.flip_prob))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """G(x) for chains x of shape (C, D) -> (C, P) uint8."""
        a = x[:, self.pairs[:, 0]]
        b = x[:, self.pairs[:, 1]]
        return (a ^ b) if self.kind == "xor" else (a & b)

    def mismatches(self, x: np.ndarray, obs_index: np.ndarray) -> np.ndarray:
        return self.forward(x) != self.observations[obs_index]

    def log_likelihood(self, x: np.ndarray, obs_index: np.ndarray) -> np.ndarray:
        """log p(y_obs | x) for chains x (C, D) whose observation is obs_index (C,)."""
        mm = self.mismatches(x, obs_index).sum(1)
        return self.num_pairs * np.log1p(-self.flip_prob) - self.strength * mm

    def incidence(self, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Padded incidence lists for single-site updates.

        Returns (pair_index, partner, valid), each (D, max_degree): the pairs
        containing pixel d, the other pixel of each pair, and a validity mask.
        Padding points to the dummy pair index P and partner 0.
        """
        lists: list[list[tuple[int, int]]] = [[] for _ in range(dim)]
        for p, (i, j) in enumerate(self.pairs.tolist()):
            lists[i].append((p, j))
            lists[j].append((p, i))
        width = max(1, max(len(entries) for entries in lists))
        pair_index = np.full((dim, width), self.num_pairs, dtype=np.int64)
        partner = np.zeros((dim, width), dtype=np.int64)
        valid = np.zeros((dim, width), dtype=bool)
        for d, entries in enumerate(lists):
            for k, (p, other) in enumerate(entries):
                pair_index[d, k], partner[d, k], valid[d, k] = p, other, True
        return pair_index, partner, valid


@dataclass(frozen=True)
class Problem:
    experiment: str
    dim: int
    image_shape: tuple[int, int]
    num_observations: int
    num_samples: int
    seed: int
    prior: BernoulliMixturePrior
    likelihood: PairwiseLikelihood

    def chain_observation_index(self) -> np.ndarray:
        """Observation index of each of the I * S chains, observation-major order."""
        return np.repeat(np.arange(self.num_observations), self.num_samples)


def problem_from_spec(spec: dict) -> Problem:
    prior_spec, lik = spec["prior"], spec["likelihood"]
    if prior_spec.get("type") != "bernoulli_mixture":
        raise ValueError(f"unsupported prior type {prior_spec.get('type')}")
    if lik.get("type") not in ("xor", "and"):
        raise ValueError(f"unsupported likelihood type {lik.get('type')}")
    weights = np.asarray(prior_spec["weights"], dtype=np.float64)
    probs = np.asarray(prior_spec["probs"], dtype=np.float64)
    dim = int(spec["dim"])
    if probs.shape != (len(weights), dim):
        raise ValueError("prior probs must have shape (num_components, dim)")
    observations = np.asarray(lik["observations"], dtype=np.uint8)
    pairs = np.asarray(lik["pairs"], dtype=np.int64)
    if observations.shape != (int(spec["num_observations"]), len(pairs)):
        raise ValueError("observations must have shape (num_observations, num_pairs)")
    return Problem(
        experiment=str(spec["experiment"]),
        dim=dim,
        image_shape=tuple(int(v) for v in spec["image_shape"]),
        num_observations=int(spec["num_observations"]),
        num_samples=int(spec["num_samples"]),
        seed=int(spec["seed"]),
        prior=BernoulliMixturePrior(weights=weights / weights.sum(), probs=probs),
        likelihood=PairwiseLikelihood(
            kind=str(lik["type"]),
            pairs=pairs,
            flip_prob=float(lik["flip_prob"]),
            observations=observations,
        ),
    )


def load_problem(path: str | Path) -> Problem:
    return problem_from_spec(json.loads(Path(path).read_text()))


def save_samples(output_dir: str | Path, samples: np.ndarray, metadata: dict | None = None) -> None:
    """Write the sampler output: ``samples.npz`` with an integer array of shape (I, S, D)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "samples.npz", samples=np.asarray(samples, dtype=np.int8))
    if metadata is not None:
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
