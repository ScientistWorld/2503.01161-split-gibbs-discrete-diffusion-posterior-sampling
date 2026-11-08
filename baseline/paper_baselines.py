"""Executable versions of two Table 2 baselines from the paper (Appendix C.2).

Both run the unconditional reverse discrete diffusion of the synthetic prior
and steer it with the likelihood; neither uses the split Gibbs relaxation.

* ``svdd_pm``  SVDD-PM (C.2.2): at every reverse step draw M candidate next
  states from the unconditional reverse kernel, score each candidate by the
  value v(x_t) = log E_{x0 ~ p(x0|x_t)} p(y|x0) estimated with 3 Monte Carlo
  denoised samples, and keep the best (beta = infinity, i.e. alpha = 0).
* ``smc``      SMC (C.2.3): groups of M particles are propagated with the
  unconditional reverse kernel and resampled within their group with weights
  p^(y|x_s) / p^(y|x_t), where p^(y|x_t) is the same Monte Carlo estimate.

The synthetic prior factorizes over coordinates, so the reverse kernel
p(x_s | x_t) and the denoiser p(x_0 | x_t) of the uniform-kernel diffusion are
available in closed form (the synthetic analogue of the pretrained model).

    python -m baseline.paper_baselines svdd_pm --problem PROBLEM.json --output-dir OUT
    python -m baseline.paper_baselines smc     --problem PROBLEM.json --output-dir OUT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.synthetic_sgdd import (
    SyntheticProblem,
    load_problem,
    log_likelihood,
    noised_prior_probs,
    prior_probs_1d,
    save_samples,
)


def sigma_levels(steps: int, sigma_max: float = 20.0, sigma_min: float = 1e-4) -> np.ndarray:
    """Geometric noise levels from sigma_max to sigma_min, followed by sigma = 0."""
    k = np.arange(steps, dtype=np.float64)
    denom = max(steps - 1, 1)
    levels = sigma_max ** (1.0 - k / denom) * sigma_min ** (k / denom)
    return np.append(levels, 0.0)


def sample_categorical(probs: np.ndarray, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    cdf = np.cumsum(probs)
    u = rng.random(shape) * cdf[-1]
    return np.minimum(np.searchsorted(cdf, u, side="right"), len(probs) - 1)


def reverse_kernel(
    x_t: np.ndarray, sigma_t: float, sigma_s: float, prior_1d: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Exact per-coordinate draw x_s ~ p(x_s | x_t) for sigma_s < sigma_t.

    p(x_s = v | x_t = c) is proportional to p_s(v) [e^-(t-s) 1{v=c} + (1 - e^-(t-s)) / N],
    a mixture of "stay at c" and a fresh draw from p_s. With sigma_s = 0 this is
    the denoiser p(x_0 | x_t).
    """
    vocab_size = len(prior_1d)
    p_s = noised_prior_probs(prior_1d, sigma_s)
    keep = float(np.exp(-(sigma_t - sigma_s)))
    stay_weight = p_s[x_t] * keep
    stay_prob = stay_weight / (stay_weight + (1.0 - keep) / vocab_size)
    stay = rng.random(x_t.shape) < stay_prob
    fresh = sample_categorical(p_s, x_t.shape, rng)
    return np.where(stay, x_t, fresh)


def log_mean_exp(values: np.ndarray, axis: int = 0) -> np.ndarray:
    peak = values.max(axis=axis, keepdims=True)
    return np.squeeze(peak + np.log(np.exp(values - peak).mean(axis=axis, keepdims=True)), axis=axis)


def estimated_log_value(
    x_t: np.ndarray, sigma_t: float, problem: SyntheticProblem, prior_1d: np.ndarray, mc_samples: int, rng
) -> np.ndarray:
    """log p^(y | x_t) = log mean_j p(y | x0_j), x0_j ~ p(x0 | x_t)."""
    if sigma_t <= 0.0:
        return log_likelihood(x_t, problem)
    draws = np.stack(
        [log_likelihood(reverse_kernel(x_t, sigma_t, 0.0, prior_1d, rng), problem) for _ in range(mc_samples)]
    )
    return log_mean_exp(draws, axis=0)


def run_svdd_pm(
    problem: SyntheticProblem,
    num_candidates: int = 20,
    steps: int = 32,
    mc_samples: int = 3,
    chunk: int = 10000,
    seed_offset: int = 0,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(problem.seed + seed_offset)
    prior_1d = prior_probs_1d(problem)
    sigmas = sigma_levels(steps)
    out = np.empty((problem.num_samples, problem.dim), dtype=np.int64)
    for start in range(0, problem.num_samples, chunk):
        n = min(chunk, problem.num_samples - start)
        x = sample_categorical(noised_prior_probs(prior_1d, sigmas[0]), (n, problem.dim), rng)
        cols = np.arange(n)
        for k in range(len(sigmas) - 1):
            sigma_t, sigma_s = float(sigmas[k]), float(sigmas[k + 1])
            parents = np.broadcast_to(x, (num_candidates, n, problem.dim))
            candidates = reverse_kernel(parents, sigma_t, sigma_s, prior_1d, rng)
            values = estimated_log_value(candidates, sigma_s, problem, prior_1d, mc_samples, rng)
            x = candidates[np.argmax(values, axis=0), cols]
        out[start : start + n] = x
    meta = {"method": "svdd_pm", "num_candidates": num_candidates, "steps": steps, "mc_samples": mc_samples}
    return out, meta


def run_smc(
    problem: SyntheticProblem,
    particles: int = 20,
    steps: int = 32,
    mc_samples: int = 3,
    beta: float = 1.0,
    seed_offset: int = 0,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(problem.seed + seed_offset)
    prior_1d = prior_probs_1d(problem)
    sigmas = sigma_levels(steps)
    groups = -(-problem.num_samples // particles)
    total = groups * particles
    x = sample_categorical(noised_prior_probs(prior_1d, sigmas[0]), (total, problem.dim), rng)
    log_value = estimated_log_value(x, float(sigmas[0]), problem, prior_1d, mc_samples, rng)
    offsets = (np.arange(groups) * particles)[:, None]
    for k in range(len(sigmas) - 1):
        sigma_t, sigma_s = float(sigmas[k]), float(sigmas[k + 1])
        x_new = reverse_kernel(x, sigma_t, sigma_s, prior_1d, rng)
        log_value_new = estimated_log_value(x_new, sigma_s, problem, prior_1d, mc_samples, rng)
        log_w = (beta * (log_value_new - log_value)).reshape(groups, particles)
        w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
        cdf = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
        u = rng.random((groups, particles))
        idx = np.minimum((u[:, :, None] > cdf[:, None, :]).sum(axis=-1), particles - 1)
        flat = (offsets + idx).ravel()
        x = x_new[flat]
        log_value = log_value_new[flat]
    meta = {"method": "smc", "particles": particles, "steps": steps, "mc_samples": mc_samples, "beta": beta}
    return x[: problem.num_samples], meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", choices=("svdd_pm", "smc"))
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=20, help="M: candidates (SVDD-PM) or particles (SMC)")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--mc-samples", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    problem = load_problem(args.problem)
    if args.baseline == "svdd_pm":
        samples, meta = run_svdd_pm(problem, args.particles, args.steps, args.mc_samples, seed_offset=args.seed_offset)
    else:
        samples, meta = run_smc(problem, args.particles, args.steps, args.mc_samples, seed_offset=args.seed_offset)
    save_samples(args.output_dir, samples, meta)


if __name__ == "__main__":
    main()
