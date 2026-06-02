"""Synthetic posterior recovery problem used by SGDD.

This module is method-agnostic. It defines the finite categorical Gaussian
prior, the SingleModal inverse observation from the authors' code, and exact
posterior marginals for evaluating generated samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticConfig:
    dim: int = 2
    vocab_size: int = 50
    prior_std: float = 2.0
    likelihood_sigma: float = 3.0
    grid_min: float = -5.0
    grid_max: float = 5.0


def grid_values(config: SyntheticConfig) -> np.ndarray:
    return np.linspace(config.grid_min, config.grid_max, config.vocab_size, dtype=np.float64)


def prior_logits_1d(config: SyntheticConfig) -> np.ndarray:
    grid = grid_values(config)
    return -0.5 * (grid / config.prior_std) ** 2


def normalize_log_probs(logits: np.ndarray, axis: int | None = None) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    probs = np.exp(shifted)
    return probs / probs.sum(axis=axis, keepdims=True)


def prior_probs_1d(config: SyntheticConfig) -> np.ndarray:
    return normalize_log_probs(prior_logits_1d(config))


def center_index(config: SyntheticConfig) -> float:
    return (config.vocab_size - 1) / 2.0


def observation_value(config: SyntheticConfig) -> float:
    """Scalar paper-style synthetic observation y from x = discretize(5/3)."""
    x_obs = int(((5.0 / 3.0 - config.grid_min) / (config.grid_max - config.grid_min)) * config.vocab_size)
    x_obs = min(max(x_obs, 0), config.vocab_size - 1)
    return float(abs(x_obs - center_index(config)))


def observation(config: SyntheticConfig) -> np.ndarray:
    """Vector observation used by the coordinate-wise SingleModal operator."""
    return np.full(config.dim, observation_value(config), dtype=np.float64)


def singlemodal_loss_per_dim(tokens: np.ndarray, config: SyntheticConfig) -> np.ndarray:
    """Per-coordinate squared residual for the synthetic SingleModal operator."""
    y = observation_value(config)
    return (np.abs(tokens.astype(np.float64) - center_index(config)) - y) ** 2


def log_likelihood(tokens: np.ndarray, config: SyntheticConfig) -> np.ndarray:
    """Log p(y|x), matching the authors' BaseOperator scaling."""
    losses = singlemodal_loss_per_dim(tokens, config).mean(axis=-1)
    return -losses / (config.likelihood_sigma**2)


def exact_posterior_first2(config: SyntheticConfig) -> np.ndarray:
    """Exact posterior marginal over the first two dimensions.

    The synthetic prior is factorized and the SingleModal likelihood is a mean
    of coordinate-wise losses, so the posterior also factorizes. The likelihood
    contribution is scaled by 1 / D because the authors' synthetic loss uses a
    mean over dimensions.
    """
    prior = prior_probs_1d(config)
    tokens = np.arange(config.vocab_size)
    per_dim_loglike = -singlemodal_loss_per_dim(tokens, config)
    per_dim_loglike /= config.dim * (config.likelihood_sigma**2)
    marginal_1d = normalize_log_probs(np.log(prior + 1e-300) + per_dim_loglike)
    return np.outer(marginal_1d, marginal_1d)


def empirical_first2(samples: np.ndarray, vocab_size: int) -> np.ndarray:
    hist = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    np.add.at(hist, (samples[:, 0], samples[:, 1]), 1.0)
    return hist / max(len(samples), 1)


def hellinger(empirical: np.ndarray, target: np.ndarray) -> float:
    affinity = np.sqrt(empirical * target).sum()
    return float(np.sqrt(max(0.0, 1.0 - affinity)))


def total_variation(empirical: np.ndarray, target: np.ndarray) -> float:
    return float(0.5 * np.abs(empirical - target).sum())
