"""Fast pre-flight checks for the hardening job (tiny sample counts).

1. Both evaluators' exact pair marginals agree with brute-force enumeration.
2. The audit's exact sampler matches those targets.
3. Every shipped and audit sampler runs through both evaluators at N=300;
   only the shortcuts expected to be rejected may fail.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import eval.test.evaluate_synthetic as te  # noqa: E402
import eval.train.evaluate_synthetic as tr  # noqa: E402
from data.synthetic_sgdd import problem_from_spec  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variants  # noqa: E402

EXPECTED_FAILURES = {"shortcut_negative_categories", "shortcut_more_samples", "shortcut_empty"}


def brute_force_pairs(spec):
    vocab, dim = spec["vocab_size"], spec["dim"]
    grids = np.indices((vocab,) * dim, dtype=np.int16).reshape(dim, -1).T
    log_w = np.log(te._prior_1d(spec))[grids].sum(axis=1)
    dist = np.abs(grids - (vocab - 1) / 2.0)
    lik = spec["likelihood"]
    if lik["type"] == "l1_norm":
        loss = (dist.sum(axis=1) - lik["observation"]) ** 2
    else:
        loss = ((dist - np.asarray(lik["observation"])) ** 2).mean(axis=1)
    log_w = log_w - loss / lik["sigma"] ** 2
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    out = {}
    for i in range(dim):
        for j in range(i + 1, dim):
            m = np.bincount(grids[:, i].astype(np.int64) * vocab + grids[:, j], weights=w, minlength=vocab * vocab)
            out[(i, j)] = m.reshape(vocab, vocab)
    return out


def check_targets():
    cases = [
        ("coordinatewise", 3, [7.5, 9.5, 8.5]),
        ("l1_norm", 3, 40.0),
        ("l1_norm", 4, 47.0),
    ]
    for kind, dim, obs in cases:
        spec = {
            "instance_id": "smoke", "dim": dim, "vocab_size": 50, "num_samples": 10, "seed": 0,
            "prior": dict(te.PRIOR), "likelihood": {"type": kind, "sigma": 3.0, "observation": obs},
        }
        truth = brute_force_pairs(spec)
        for label, got in (("test", te.pair_targets(spec)), ("train", tr.exact_pair_marginals(spec))):
            err = max(float(np.abs(got[p] - truth[p]).max()) for p in truth)
            print(f"target check {kind} D={dim} {label}: max abs error {err:.2e}", flush=True)
            if err > 1e-9:
                raise SystemExit(f"exact target mismatch for {kind} D={dim} ({label})")
        spec["num_samples"] = 100000
        samples = variants.exact_posterior_samples(problem_from_spec(spec), np.random.default_rng(0))
        h = te._hellinger(te._pair_histogram(samples, (0, 1), 50), truth[(0, 1)])
        print(f"exact sampler {kind} D={dim}: pair Hellinger {h:.4f} at N=100000", flush=True)
        if h > 0.08:
            raise SystemExit("exact sampler does not match the exact target")


def check_samplers():
    samplers = te.discover_samplers(ROOT / "scripts" / "samplers")
    audit = te.discover_samplers(Path(__file__).resolve().parent / "samplers")
    audit.pop("shortcut_easy_problem")
    samplers.update(audit)
    work = Path(tempfile.mkdtemp(prefix="smoke_"))
    try:
        train = tr.run_all(samplers, 300, work / "train")
        test = te.evaluate(samplers, 300, work / "test")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    for label, scores in (("train", train), ("test", test)):
        failed = set()
        for exp, block in scores["experiments"].items():
            failed |= set(block["_failures"])
            for method, metrics in block["results"].items():
                for key, value in metrics.items():
                    if not (0.0 <= value <= 1.0):
                        raise SystemExit(f"{label} {exp} {method} {key}={value} out of range")
        print(f"{label}: failed samplers {sorted(failed)}", flush=True)
        if failed != EXPECTED_FAILURES:
            raise SystemExit(f"{label}: unexpected failure set {sorted(failed)}")
        print(f"{label} smoke sgdd: {scores['experiments']['synthetic_l1_d10']['results']['sgdd']}", flush=True)


if __name__ == "__main__":
    check_targets()
    check_samplers()
    print("SMOKE OK", flush=True)
