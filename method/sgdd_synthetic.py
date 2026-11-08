"""Split Gibbs Discrete Diffusion Posterior Sampling (SGDD) on the synthetic task.

Algorithm 1 of the paper: starting from uniform tokens, alternate for
k = 0..K-1 with an annealed noise level eta_k

* likelihood step (Eq. 13): T Metropolis-Hastings steps on z targeting
  p(y|z) exp(-D(x, z; eta_k)) with the Hamming potential of Eq. 10;
* prior step (Eq. 11): x ~ p_0(x) (beta/(1-beta))^d(z, x), sampled by running
  the reverse uniform-kernel discrete diffusion from sigma = eta_k down to 0
  with an H-step Euler sampler and the closed-form synthetic concrete score.

Usage (sampler contract used by the evaluators):

    python -m method.sgdd_synthetic --problem PROBLEM.json --output-dir OUT
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


def eta_schedule(num_iterations: int, eta_min: float = 1e-4, eta_max: float = 20.0) -> np.ndarray:
    """Geometric annealing eta_k = eta_min^(k/K) eta_max^(1-k/K) (Appendix C.3)."""
    k = np.arange(num_iterations, dtype=np.float64)
    return eta_min ** (k / num_iterations) * eta_max ** (1.0 - k / num_iterations)


def hamming_log_transition_weight(eta: float, vocab_size: int) -> float:
    """log(beta/(1-beta)) = -D(x, z; eta) per Hamming mismatch (Eq. 10/11)."""
    beta = (vocab_size - 1.0) / vocab_size * (1.0 - np.exp(-eta))
    beta = np.clip(beta, 1e-12, 1.0 - 1e-12)
    return float(np.log(beta) - np.log1p(-beta))


def exact_prior_step(z: np.ndarray, eta: float, prior_1d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Diagnostic only: sample Eq. 11 exactly instead of with the Euler sampler."""
    vocab_size = len(prior_1d)
    match_bonus = -hamming_log_transition_weight(eta, vocab_size)
    base_log = np.log(prior_1d + 1e-300)
    out = np.empty_like(z)
    for value in range(vocab_size):
        mask = z == value
        count = int(mask.sum())
        if count == 0:
            continue
        logits = base_log.copy()
        logits[value] += match_bonus
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        out[mask] = rng.choice(vocab_size, size=count, p=probs)
    return out


def euler_prior_step(
    z: np.ndarray,
    eta: float,
    prior_1d: np.ndarray,
    euler_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Partial reverse discrete-diffusion sampler for Eq. 11 (H Euler steps).

    For the uniform forward kernel the reverse CTMC rate from token c to v at
    noise level sigma is p_sigma(v) / (N p_sigma(c)). Each Euler step moves a
    token with probability dt * exit_rate(c) and, if it moves, draws the new
    token from p_sigma restricted to v != c. Every transition is computed from
    the state at the start of the (sub)step, so a token jumps at most once per
    (sub)step. Steps whose largest jump probability would exceed 0.95 are split
    into equal sub-steps so the Euler probabilities stay valid.
    """
    vocab_size = len(prior_1d)
    x = z.copy()
    if euler_steps <= 0 or eta <= 0.0:
        return x
    delta = eta / euler_steps
    for step in range(euler_steps):
        sigma = eta * (1.0 - step / euler_steps)
        p_sigma = noised_prior_probs(prior_1d, sigma)
        exit_rates = (1.0 - p_sigma) / (vocab_size * np.maximum(p_sigma, 1e-300))
        substeps = max(1, int(np.ceil(delta * float(exit_rates.max()) / 0.95)))
        sub_delta = delta / substeps
        cdf = np.cumsum(p_sigma)
        for _ in range(substeps):
            move = rng.random(x.shape) < sub_delta * exit_rates[x]
            current = x[move]
            if current.size == 0:
                continue
            p_cur = p_sigma[current]
            u = rng.random(current.size) * (1.0 - p_cur)
            u = u + np.where(u >= cdf[current] - p_cur, p_cur, 0.0)
            x[move] = np.minimum(np.searchsorted(cdf, u, side="right"), vocab_size - 1)
    return x


def likelihood_step_mh(
    x: np.ndarray,
    eta: float,
    problem: SyntheticProblem,
    mh_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Metropolis-Hastings likelihood step from Eq. 13 (single-site uniform proposals)."""
    z = x.copy()
    current_ll = log_likelihood(z, problem)
    current_hamming = (z != x).sum(axis=1)
    log_mismatch = hamming_log_transition_weight(eta, problem.vocab_size)
    rows = np.arange(len(z))
    accepted = 0
    total = 0
    for _ in range(mh_steps):
        cols = rng.integers(0, problem.dim, size=len(z))
        vals = rng.integers(0, problem.vocab_size, size=len(z))
        proposal = z.copy()
        proposal[rows, cols] = vals
        proposal_ll = log_likelihood(proposal, problem)
        proposal_hamming = (proposal != x).sum(axis=1)
        log_accept = proposal_ll - current_ll + (proposal_hamming - current_hamming) * log_mismatch
        accept = np.log(rng.random(len(z)) + 1e-300) < np.minimum(log_accept, 0.0)
        z[accept] = proposal[accept]
        current_ll[accept] = proposal_ll[accept]
        current_hamming[accept] = proposal_hamming[accept]
        accepted += int(accept.sum())
        total += len(z)
    return z, accepted / max(total, 1)


def run_sgdd(
    problem: SyntheticProblem,
    iterations: int = 10,
    mh_steps: int = 10,
    euler_steps: int = 20,
    prior_sampler: str = "euler",
    eta_min: float = 1e-4,
    eta_max: float = 20.0,
    fixed_eta: float | None = None,
    seed_offset: int = 0,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(problem.seed + seed_offset)
    prior_1d = prior_probs_1d(problem)
    x = rng.integers(0, problem.vocab_size, size=(problem.num_samples, problem.dim), dtype=np.int64)
    if fixed_eta is None:
        etas = eta_schedule(iterations, eta_min, eta_max)
    else:
        etas = np.full(iterations, float(fixed_eta))
    accept_rates = []
    for eta in etas:
        z, accept_rate = likelihood_step_mh(x, float(eta), problem, mh_steps, rng)
        if prior_sampler == "exact":
            x = exact_prior_step(z, float(eta), prior_1d, rng)
        elif prior_sampler == "euler":
            x = euler_prior_step(z, float(eta), prior_1d, euler_steps, rng)
        else:
            raise ValueError(f"Unknown prior sampler: {prior_sampler}")
        accept_rates.append(accept_rate)
    meta = {
        "method": "sgdd",
        "iterations": int(iterations),
        "mh_steps": int(mh_steps),
        "euler_steps": int(euler_steps),
        "prior_sampler": prior_sampler,
        "eta_schedule": "annealed" if fixed_eta is None else f"fixed:{fixed_eta}",
        "mean_mh_acceptance": float(np.mean(accept_rates)) if accept_rates else 0.0,
        "seed": int(problem.seed + seed_offset),
    }
    return x, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--mh-steps", type=int, default=10)
    parser.add_argument("--euler-steps", type=int, default=20)
    parser.add_argument("--prior-sampler", choices=("euler", "exact"), default="euler")
    parser.add_argument("--eta-min", type=float, default=1e-4)
    parser.add_argument("--eta-max", type=float, default=20.0)
    parser.add_argument("--fixed-eta", type=float, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    problem = load_problem(args.problem)
    samples, meta = run_sgdd(
        problem,
        iterations=args.iterations,
        mh_steps=args.mh_steps,
        euler_steps=args.euler_steps,
        prior_sampler=args.prior_sampler,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        fixed_eta=args.fixed_eta,
        seed_offset=args.seed_offset,
    )
    save_samples(args.output_dir, samples, meta)


if __name__ == "__main__":
    main()
