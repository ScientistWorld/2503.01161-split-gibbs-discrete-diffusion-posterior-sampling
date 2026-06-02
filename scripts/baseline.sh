#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

NAME="${1:-direct_mh}"
DIM="${2:-2}"

case "${NAME}" in
  direct_mh)
    python -m baseline.direct_mh_synthetic \
      --dim "${DIM}" \
      --vocab-size 50 \
      --num-samples "${NUM_SAMPLES:-10000}" \
      --steps 100 \
      --seed 1 \
      --output-dir "checkpoints/direct_mh_synthetic_d${DIM}"
    ;;
  *)
    echo "Unknown baseline: ${NAME}" >&2
    exit 2
    ;;
esac

bash scripts/evaluate.sh
