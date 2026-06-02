#!/bin/bash
# Visible train-slice evaluation. Mirrors scripts/evaluate.sh's method-agnostic
# contract, but writes scoring/scores_train.json and scores eval/train/.

set -e

cd /home/user

mkdir -p scoring
OUTPUT="${SCORES_OUTPUT:-scoring/scores_train.json}"

if [ "$#" -gt 0 ]; then
  if [ "$#" -eq 1 ] && [ "$1" != "--run" ]; then
    python -m eval.train.evaluate_synthetic --checkpoints-dir "$1" --output "${OUTPUT}"
    exit 0
  fi
  if [ "$1" = "--run" ]; then
    python -m eval.train.evaluate_synthetic "$@" --output "${OUTPUT}"
    exit 0
  fi
  if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: bash scripts/evaluate_train.sh [CHECKPOINTS_DIR]" >&2
    echo "Usage: bash scripts/evaluate_train.sh METHOD RUN_DIR [EXPERIMENT]" >&2
    echo "   or: bash scripts/evaluate_train.sh --run EXPERIMENT METHOD RUN_DIR [--run ...]" >&2
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
  python -m eval.train.evaluate_synthetic --run "${EXPERIMENT}" "${METHOD}" "${RUN_DIR}" --output "${OUTPUT}"
  exit 0
fi

python -m eval.train.evaluate_synthetic --output "${OUTPUT}"
