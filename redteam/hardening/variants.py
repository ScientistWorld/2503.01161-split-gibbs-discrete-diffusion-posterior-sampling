"""Variant and shortcut samplers used only by the hardening audit.

Each entry follows the public sampler contract (problem JSON in, samples.npz
out) and is registered temporarily as ``scripts/samplers/<name>.sh`` so that it
is scored by the unmodified, no-argument hidden test. Shortcuts use only what a
future agent can see: the problem file, ``data/``, ``method/``, ``baseline/``.

    python redteam/hardening/variants.py NAME PROBLEM_JSON OUTPUT_DIR
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import method.sgdd_synthetic as sgdd  # noqa: E402
from baseline.direct_mh_synthetic import run_direct_mh  # noqa: E402
from data.synthetic_sgdd import (  # noqa: E402
    SyntheticProblem,
    load_problem,
    noised_prior_probs,
    prior_probs_1d,
    save_samples,
)


# --------------------------------------------------------------------- variants


def legacy_euler_prior_step(z, eta, prior_1d, euler_steps, rng):
    """The pre-hardening reference Euler step (updates x in place while looping
    over token values, so a token can jump several times in one sub-step)."""
    vocab_size = len(prior_1d)
    x = z.copy()
    if euler_steps <= 0 or eta <= 0.0:
        return x
    delta = eta / euler_steps
    values = np.arange(vocab_size)
    for step in range(euler_steps):
        sigma = eta * (1.0 - step / euler_steps)
        p_sigma = noised_prior_probs(prior_1d, sigma)
        exit_rates = (1.0 - p_sigma) / (vocab_size * np.maximum(p_sigma, 1e-300))
        substeps = max(1, int(np.ceil(delta * float(exit_rates.max()) / 0.95)))
        sub_delta = delta / substeps
        for _ in range(substeps):
            for current_value in range(vocab_size):
                rows, cols = np.where(x == current_value)
                if len(rows) == 0:
                    continue
                rates = p_sigma / (vocab_size * max(p_sigma[current_value], 1e-300))
                rates[current_value] = 0.0
                probs = sub_delta * rates
                probs[current_value] = 1.0 - probs.sum()
                probs /= probs.sum()
                x[rows, cols] = rng.choice(values, size=len(rows), p=probs)
    return x


def legacy_euler(problem):
    sgdd.euler_prior_step = legacy_euler_prior_step
    return sgdd.run_sgdd(problem)


# --------------------------------------------------------------------- exact sampler


def _logsumexp(values, axis):
    peak = np.max(values, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return np.squeeze(peak + np.log(np.exp(values - peak).sum(axis=axis, keepdims=True)), axis=axis)


def _gumbel_argmax(logits, rng):
    return np.argmax(logits - np.log(-np.log(rng.random(logits.shape) + 1e-300) + 1e-300), axis=-1)


def exact_posterior_samples(problem: SyntheticProblem, rng) -> np.ndarray:
    """SHORTCUT: compute the posterior analytically from the problem file and
    sample it exactly (per-coordinate enumeration, or forward filtering /
    backward sampling on the chain). Possible only because the synthetic prior
    is closed-form."""
    vocab, dim, n = problem.vocab_size, problem.dim, problem.num_samples
    p0 = prior_probs_1d(problem)
    log_p0 = np.log(p0)
    sigma2 = problem.likelihood_sigma**2
    out = np.empty((n, dim), dtype=np.int64)
    if problem.likelihood_type == "coordinatewise":
        radius = np.abs(np.arange(vocab) - problem.center)
        for d in range(dim):
            logits = log_p0 - (radius - problem.observation[d]) ** 2 / (dim * sigma2)
            out[:, d] = _gumbel_argmax(np.broadcast_to(logits, (n, vocab)), rng)
        return out
    y = np.asarray(problem.observation)
    gaps = np.abs(np.arange(vocab)[:, None] - np.arange(vocab)[None, :])
    log_links = [-((gaps - y[d]) ** 2) / sigma2 for d in range(dim - 1)]
    log_alpha = [log_p0]
    for d in range(dim - 1):
        msg = _logsumexp(log_alpha[-1][:, None] + log_links[d], axis=0) + log_p0
        log_alpha.append(msg - msg.max())
    out[:, dim - 1] = _gumbel_argmax(np.broadcast_to(log_alpha[dim - 1], (n, vocab)), rng)
    for d in range(dim - 2, -1, -1):
        logits = log_alpha[d][None, :] + log_links[d][:, out[:, d + 1]].T
        out[:, d] = _gumbel_argmax(logits, rng)
    return out


def kernel_consistent_weight(eta: float, vocab_size: int) -> float:
    """VARIANT: Hamming potential matched to the uniform-kernel transition
    probability of one specific token, log(beta / ((N-1)(1-beta))), i.e. the
    coupling the diffusion prior step actually samples, instead of Eq. 10's
    log(beta / (1-beta))."""
    return sgdd_paper_weight(eta, vocab_size) - float(np.log(vocab_size - 1.0))


sgdd_paper_weight = sgdd.hamming_log_transition_weight


# --------------------------------------------------------------------- registry


def run(name: str, problem: SyntheticProblem, output_dir: Path) -> None:
    rng = np.random.default_rng(problem.seed + 7)
    n, dim = problem.num_samples, problem.dim
    meta = None
    if name == "legacy_euler":
        samples, meta = legacy_euler(problem)
    elif name in ("consistent_potential", "consistent_potential_long"):
        sgdd.hamming_log_transition_weight = kernel_consistent_weight
        if name == "consistent_potential":
            samples, meta = sgdd.run_sgdd(problem)
        else:
            samples, meta = sgdd.run_sgdd(problem, iterations=50, mh_steps=50)
    elif name == "exact_posterior":
        samples = exact_posterior_samples(problem, rng)
    elif name == "binary_flag":
        samples = exact_posterior_samples(problem, rng)
        meta = {"uses_likelihood_gradient": 1}
    elif name == "cached_train":
        # SHORTCUT: answers rebuilt from the public train instance of the same
        # experiment shape, submitted regardless of the problem actually given.
        public_obs = [8.5] * dim if problem.likelihood_type == "coordinatewise" else [15, 11, 8, 20, 12, 12, 5, 6, 10]
        public = dataclasses.replace(problem, observation=tuple(public_obs))
        samples = exact_posterior_samples(public, rng)
    elif name == "two_latents":
        base, _ = run_direct_mh(problem, steps=100)
        samples = base[:, np.arange(dim) % 2]
    elif name == "negative_categories":
        base, _ = sgdd.run_sgdd(problem)
        samples = base - problem.vocab_size
    elif name == "more_samples":
        samples, meta = sgdd.run_sgdd(dataclasses.replace(problem, num_samples=2 * n))
    elif name == "constant":
        samples = np.zeros((n, dim), dtype=np.int64)
    elif name == "empty":
        samples = np.zeros((0, dim), dtype=np.int64)
    elif name == "negative_flag":
        samples = np.zeros((n, dim), dtype=np.int64)
        meta = {"uses_likelihood_gradient": -10}
    elif name == "forged_scores":
        samples = np.zeros((n, dim), dtype=np.int64)
        meta = {"hellinger": 0.0, "total_variation": 0.0, "all_pairs_hellinger": 0.0, "uses_likelihood_gradient": 0}
        forged = {"experiments": {}}
        for exp in ("synthetic_d2", "synthetic_d5", "synthetic_d10", "synthetic_chain_d10"):
            forged["experiments"][exp] = {"results": {"shortcut_forged_scores": {k: 0.0 for k in ("hellinger", "total_variation", "all_pairs_hellinger")}}}
        for target in ("scoring/scores_test.json", "scoring/scores_train.json"):
            (ROOT / target).write_text(json.dumps(forged, indent=2) + "\n")
    else:
        raise SystemExit(f"unknown variant {name}")
    save_samples(output_dir, samples, meta)


if __name__ == "__main__":
    variant, problem_path, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    run(variant, load_problem(problem_path), out)
