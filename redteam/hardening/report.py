"""Rank every method in a hidden-test score file against the reference method.

gain(method) = sum_exp weight_exp * sum_metric coefficient_metric * (value - value_ref)
using the weights and coefficients in scoring/reference.json. Positive is better.

    python redteam/hardening/report.py SCORES_TEST.json OUTPUT.json [REFERENCE_SCORES.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    scores = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2])
    ref_scores = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else scores
    reference = json.loads((ROOT / "scoring" / "reference.json").read_text())["experiments"]
    methods = sorted(next(iter(scores["experiments"].values()))["results"])
    summary = {}
    for method in methods:
        total, per_exp, failures = 0.0, {}, {}
        for exp, definition in reference.items():
            block = scores["experiments"][exp]
            values = block["results"][method]
            ref = ref_scores["experiments"][exp]["results"]["sgdd"]
            contribution = sum(
                rule["coefficient"] * (values[m] - ref[m]) for m, rule in definition["metrics"].items()
            )
            per_exp[exp] = {"weighted_gain": definition["weight"] * contribution, **values}
            total += definition["weight"] * contribution
            if method in block.get("_failures", {}):
                failures[exp] = block["_failures"][method]
        summary[method] = {"gain_vs_reference": total, "failed": bool(failures), "failure": failures, "experiments": per_exp}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{'method':32s} {'gain':>10s}  " + "  ".join(f"{e[10:]:>18s}" for e in reference))
    for method, row in sorted(summary.items(), key=lambda kv: -kv[1]["gain_vs_reference"]):
        cells = "  ".join(
            f"{row['experiments'][e]['hellinger']:.4f}/{row['experiments'][e]['all_pairs_hellinger']:.4f}".rjust(18)
            for e in reference
        )
        flag = " FAILED" if row["failed"] else ""
        print(f"{method:32s} {row['gain_vs_reference']:+10.5f}  {cells}{flag}")


if __name__ == "__main__":
    main()
