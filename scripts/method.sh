#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

DIM="${1:-2}"
RUN_DIR="checkpoints/sgdd_synthetic_d${DIM}"

python -m method.sgdd_synthetic \
  --dim "${DIM}" \
  --vocab-size 50 \
  --num-samples "${NUM_SAMPLES:-10000}" \
  --iterations "${ITERATIONS:-10}" \
  --mh-steps "${MH_STEPS:-10}" \
  --euler-steps "${EULER_STEPS:-20}" \
  --prior-sampler "${PRIOR_SAMPLER:-euler}" \
  --seed 0 \
  --output-dir "${RUN_DIR}"

bash scripts/evaluate.sh
