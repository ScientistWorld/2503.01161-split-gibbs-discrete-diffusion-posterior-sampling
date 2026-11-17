"""Build the gym's discrete diffusion prior: a Bernoulli mixture fitted to MNIST.

The paper's image experiments (Section 4.4) use binarized MNIST with a
pretrained uniform-kernel discrete diffusion model as the prior. This gym
replaces the neural network by a mixture of independent Bernoulli images
fitted to the same data, because a product mixture keeps every quantity the
diffusion model provides exact: the noised marginals p_sigma, the concrete
score p_sigma(x~)/p_sigma(x), the reverse kernel p(x_s | x_t) and unconditional
samples (see ``data/discrete_problem.py``).

Preprocessing: 28x28 MNIST training images are average-pooled 2x2 to 14x14
(D = 196 pixels) and binarized at intensity 0.35. The mixture has 30
components, initialised from three random splits of every digit class and
refined with 80 EM iterations (Beta(1,1) pseudo-counts); component pixel
probabilities are finally clipped to [0.01, 0.99] and rounded to 6 decimals.

    python -m data.mnist_prior --mnist-dir data/downloads/mnist --output data/mnist14_mob_prior.json

The output is deterministic. The committed file is the prior every evaluator
uses (``eval/train/`` and ``eval/test/`` keep their own identical copies).
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
from pathlib import Path

import numpy as np


IMAGE_SIDE = 14
THRESHOLD = 0.35
COMPONENTS_PER_CLASS = 3
EM_ITERATIONS = 80
PSEUDO_COUNT = 1.0
THETA_CLIP = 0.01
SEED = 20250301


def load_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    magic = struct.unpack(">I", raw[:4])[0]
    if magic == 2051:
        n, rows, cols = struct.unpack(">III", raw[4:16])
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows, cols)
    if magic == 2049:
        (n,) = struct.unpack(">I", raw[4:8])
        return np.frombuffer(raw, dtype=np.uint8, offset=8).reshape(n)
    raise ValueError(f"{path}: unknown IDX magic {magic}")


def binarize(images: np.ndarray) -> np.ndarray:
    """28x28 uint8 images -> (n, 196) {0,1} after 2x2 average pooling."""
    x = images.astype(np.float64) / 255.0
    n = x.shape[0]
    pooled = x.reshape(n, IMAGE_SIDE, 2, IMAGE_SIDE, 2).mean(axis=(2, 4))
    return (pooled > THRESHOLD).astype(np.uint8).reshape(n, IMAGE_SIDE * IMAGE_SIDE)


def mixture_log_lik(x: np.ndarray, weights: np.ndarray, theta: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float64)
    return xf @ np.log(theta).T + (1.0 - xf) @ np.log1p(-theta).T + np.log(weights)


def fit(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(SEED)
    xf = x.astype(np.float64)
    theta = []
    for digit in range(10):
        idx = rng.permutation(np.flatnonzero(labels == digit))
        for part in np.array_split(idx, COMPONENTS_PER_CLASS):
            theta.append((xf[part].sum(0) + PSEUDO_COUNT) / (len(part) + 2 * PSEUDO_COUNT))
    theta = np.asarray(theta)
    weights = np.full(len(theta), 1.0 / len(theta))
    history = []
    for _ in range(EM_ITERATIONS):
        log_joint = mixture_log_lik(x, weights, theta)
        peak = log_joint.max(1, keepdims=True)
        resp = np.exp(log_joint - peak)
        norm = resp.sum(1, keepdims=True)
        history.append(float(np.mean(peak[:, 0] + np.log(norm[:, 0]))))
        resp /= norm
        counts = resp.sum(0)
        theta = (resp.T @ xf + PSEUDO_COUNT) / (counts[:, None] + 2 * PSEUDO_COUNT)
        weights = (counts + 1.0) / (len(x) + len(counts))
    theta = np.round(np.clip(theta, THETA_CLIP, 1.0 - THETA_CLIP), 6)
    weights = np.round(weights / weights.sum(), 8)
    weights = weights / weights.sum()
    order = np.argsort(-weights, kind="stable")
    return weights[order], theta[order], history


def mean_log_prob(x: np.ndarray, weights: np.ndarray, theta: np.ndarray) -> float:
    log_joint = mixture_log_lik(x, weights, theta)
    peak = log_joint.max(1, keepdims=True)
    return float(np.mean(peak[:, 0] + np.log(np.exp(log_joint - peak).sum(1))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-dir", type=Path, default=Path("data/downloads/mnist"))
    parser.add_argument("--output", type=Path, default=Path("data/mnist14_mob_prior.json"))
    args = parser.parse_args()

    train_x = binarize(load_idx(args.mnist_dir / "train-images-idx3-ubyte.gz"))
    train_y = load_idx(args.mnist_dir / "train-labels-idx1-ubyte.gz")
    test_x = binarize(load_idx(args.mnist_dir / "t10k-images-idx3-ubyte.gz"))
    weights, theta, history = fit(train_x, train_y)
    payload = {
        "name": "mnist14_bernoulli_mixture",
        "description": (
            "Mixture of independent Bernoulli images fitted by EM to binarized 14x14 MNIST "
            "(2x2 average pooling, threshold 0.35). Clean-data distribution p_0 of a uniform-kernel "
            "discrete diffusion model over {0,1}^196."
        ),
        "image_shape": [IMAGE_SIDE, IMAGE_SIDE],
        "vocab_size": 2,
        "num_components": int(len(weights)),
        "weights": [float(w) for w in weights],
        "probs": [[float(v) for v in row] for row in theta],
        "fit": {
            "train_images": int(len(train_x)),
            "mean_ink_fraction": float(train_x.mean()),
            "train_mean_log_prob_nats": mean_log_prob(train_x, weights, theta),
            "test_mean_log_prob_nats": mean_log_prob(test_x, weights, theta),
            "em_first_last_log_lik": [history[0], history[-1]],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload) + "\n")
    print(json.dumps(payload["fit"], indent=2))


if __name__ == "__main__":
    main()
