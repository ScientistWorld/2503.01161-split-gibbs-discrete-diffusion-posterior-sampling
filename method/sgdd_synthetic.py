"""Split Gibbs Discrete Diffusion Posterior Sampling on the synthetic task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.synthetic_sgdd import SyntheticConfig, log_likelihood, prior_probs_1d


def eta_schedule(num_iterations: int, eta_min: float = 1e-4, eta_max: float = 20.0) -> np.ndarray:
    k = np.arange(num_iterations, dtype=np.float64)
    return eta_min ** (k / num_iterations) * eta_max ** (1.0 - k / num_iterations)


def hamming_log_transition_weight(eta: float, vocab_size: int) -> float:
    """Log beta/(1-beta), equivalent to -D per Hamming mismatch."""
    beta = (vocab_size - 1.0) / vocab_size * (1.0 - np.exp(-eta))
    beta = np.clip(beta, 1e-12, 1.0 - 1e-12)
    return float(np.log(beta) - np.log1p(-beta))


def exact_prior_step(z: np.ndarray, eta: float, prior_1d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample x ~ p0(x) exp(-D(x,z;eta)).

    For the factorized synthetic prior and uniform-kernel corruption process,
    this is the exact discrete diffusion denoising distribution p(x0 | x_eta=z).
    """
    vocab_size = len(prior_1d)
    match_bonus = -hamming_log_transition_weight(eta, vocab_size)
    base_log = np.log(prior_1d + 1e-300)
    out = np.empty_like(z)
    for value in range(vocab_size):
        rows, cols = np.where(z == value)
        if len(rows) == 0:
            continue
        logits = base_log.copy()
        logits[value] += match_bonus
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        out[rows, cols] = rng.choice(vocab_size, size=len(rows), p=probs)
    return out


def noised_prior_probs(prior_1d: np.ndarray, sigma: float) -> np.ndarray:
    """Forward uniform-kernel marginal p_sigma for the synthetic prior."""
    vocab_size = len(prior_1d)
    keep = np.exp(-sigma)
    return keep * prior_1d + (1.0 - keep) / vocab_size


def euler_prior_step(
    z: np.ndarray,
    eta: float,
    prior_1d: np.ndarray,
    euler_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Partial reverse discrete-diffusion sampler for Eq. 11.

    The synthetic experiment has a closed-form concrete score. For the uniform
    forward kernel, the reverse CTMC rate from current token c to token v at
    noise level sigma is Q[v,c] = p_sigma(v) / (N * p_sigma(c)). We integrate
    this reverse chain from sigma=eta to sigma=0 with the paper's Euler sampler.
    """
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
                count = len(rows)
                if count == 0:
                    continue

                rates = p_sigma / (vocab_size * max(p_sigma[current_value], 1e-300))
                rates[current_value] = 0.0
                transition_probs = sub_delta * rates
                exit_prob = float(transition_probs.sum())

                probs = transition_probs.copy()
                probs[current_value] = 1.0 - exit_prob
                probs /= probs.sum()
                x[rows, cols] = rng.choice(values, size=count, p=probs)
    return x


def likelihood_step_mh(
    x: np.ndarray,
    eta: float,
    config: SyntheticConfig,
    mh_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Metropolis-Hastings likelihood step from Eq. 13."""
    z = x.copy()
    current_ll = log_likelihood(z, config)
    current_hamming = (z != x).sum(axis=1)
    log_mismatch = hamming_log_transition_weight(eta, config.vocab_size)
    accepted = 0
    total = 0

    for _ in range(mh_steps):
        proposal = z.copy()
        rows = np.arange(len(z))
        cols = rng.integers(0, config.dim, size=len(z))
        vals = rng.integers(0, config.vocab_size, size=len(z))
        proposal[rows, cols] = vals

        proposal_ll = log_likelihood(proposal, config)
        proposal_hamming = (proposal != x).sum(axis=1)
        log_accept = proposal_ll - current_ll
        log_accept += (proposal_hamming - current_hamming) * log_mismatch
        accept = np.log(rng.random(len(z)) + 1e-300) < np.minimum(log_accept, 0.0)

        z[accept] = proposal[accept]
        current_ll[accept] = proposal_ll[accept]
        current_hamming[accept] = proposal_hamming[accept]
        accepted += int(accept.sum())
        total += len(z)

    return z, accepted / max(total, 1)


def run_sgdd(
    config: SyntheticConfig,
    num_samples: int,
    iterations: int,
    mh_steps: int,
    euler_steps: int,
    seed: int,
    prior_sampler: str,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    prior_1d = prior_probs_1d(config)
    x = rng.integers(0, config.vocab_size, size=(num_samples, config.dim), dtype=np.int64)
    accept_rates = []

    for eta in eta_schedule(iterations):
        z, accept_rate = likelihood_step_mh(x, eta, config, mh_steps, rng)
        if prior_sampler == "exact":
            x = exact_prior_step(z, eta, prior_1d, rng)
        elif prior_sampler == "euler":
            x = euler_prior_step(z, eta, prior_1d, euler_steps, rng)
        else:
            raise ValueError(f"Unknown prior sampler: {prior_sampler}")
        accept_rates.append(accept_rate)

    meta = {
        "mean_mh_acceptance": float(np.mean(accept_rates)),
        "iterations": int(iterations),
        "mh_steps": int(mh_steps),
        "euler_steps": int(euler_steps),
        "num_samples": int(num_samples),
        "prior_sampler": prior_sampler,
        "uses_likelihood_gradient": 0.0,
        "seed": int(seed),
    }
    return x, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--mh-steps", type=int, default=10)
    parser.add_argument("--euler-steps", type=int, default=20)
    parser.add_argument("--prior-sampler", choices=("euler", "exact"), default="euler")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/sgdd_synthetic_d2"))
    args = parser.parse_args()

    config = SyntheticConfig(dim=args.dim, vocab_size=args.vocab_size)
    samples, meta = run_sgdd(
        config,
        args.num_samples,
        args.iterations,
        args.mh_steps,
        args.euler_steps,
        args.seed,
        args.prior_sampler,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "samples.npz", samples=samples)
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
