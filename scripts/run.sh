#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

DIMS="${RUN_DIMS:-2 5 10}"

if [ "${RUN_GENERATE:-1}" = "1" ]; then
  for DIM in ${DIMS}; do
    bash scripts/method.sh "${DIM}"
  done

  for DIM in ${DIMS}; do
    bash scripts/baseline.sh direct_mh "${DIM}"
  done
fi

INCLUDE_SANITY_BASELINES=1 bash scripts/evaluate.sh
python validate.py --compare
