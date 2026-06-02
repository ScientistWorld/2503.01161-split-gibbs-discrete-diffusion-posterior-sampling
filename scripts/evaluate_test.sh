#!/bin/bash
# Held-out test-slice evaluation. Mirrors scripts/evaluate.sh's method-agnostic
# contract, but writes scoring/scores_test.json and scores eval/test/.

set -e

cd /home/user

mkdir -p scoring
OUTPUT="${SCORES_OUTPUT:-scoring/scores_test.json}"

if [ "$#" -gt 0 ]; then
  if [ "$1" = "--run" ]; then
    python -m eval.test.evaluate_synthetic "$@" --output "${OUTPUT}"
    exit 0
  fi
  if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: bash scripts/evaluate_test.sh METHOD RUN_DIR [EXPERIMENT]" >&2
    echo "   or: bash scripts/evaluate_test.sh --run EXPERIMENT METHOD RUN_DIR [--run ...]" >&2
    exit 2
  fi
  METHOD="$1"
  RUN_DIR="$2"
  if [ "$#" -eq 3 ]; then
    EXPERIMENT="$3"
  else
    DIM_FROM_PATH="$(printf '%s\n' "${RUN_DIR}" | sed -n 's/.*_d\([0-9][0-9]*\).*/\1/p' | tail -1)"
    if [ -z "${DIM_FROM_PATH}" ]; then
      echo "Could not infer experiment from ${RUN_DIR}; pass EXPERIMENT explicitly." >&2
      exit 2
    fi
    EXPERIMENT="synthetic_d${DIM_FROM_PATH}"
  fi
  python -m eval.test.evaluate_synthetic --run "${EXPERIMENT}" "${METHOD}" "${RUN_DIR}" --output "${OUTPUT}"
  exit 0
fi

python -m eval.test.evaluate_synthetic --from-train scoring/scores_train.json --output "${OUTPUT}"
