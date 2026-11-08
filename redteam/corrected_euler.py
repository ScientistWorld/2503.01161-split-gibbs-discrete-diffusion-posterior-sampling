"""A genuine method fix: apply each Euler transition to a frozen input state."""
import numpy as np


def euler_prior_step(z, eta, prior_1d, euler_steps, rng):
    vocab_size = len(prior_1d)
    x = z.copy()
    if euler_steps <= 0 or eta <= 0:
        return x
    delta = eta / euler_steps
    for step in range(euler_steps):
        sigma = eta * (1 - step / euler_steps)
        keep = np.exp(-sigma)
        p = keep * prior_1d + (1 - keep) / vocab_size
        exit_rates = (1 - p) / (vocab_size * np.maximum(p, 1e-300))
        substeps = max(1, int(np.ceil(delta * exit_rates.max() / .95)))
        cdf = np.cumsum(p)
        for _ in range(substeps):
            # Every entry moves at most once in this Euler transition. The
            # reference revisits entries after changing their categorical value.
            old = x.copy()
            move = rng.random(old.shape) < (delta / substeps) * exit_rates[old]
            current = old[move]
            u = rng.random(len(current)) * (1 - p[current])
            u += np.where(u >= cdf[current] - p[current], p[current], 0.)
            x[move] = np.minimum(np.searchsorted(cdf, u), vocab_size - 1)
    return x
