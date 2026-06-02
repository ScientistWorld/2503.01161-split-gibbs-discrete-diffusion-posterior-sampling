#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"imagenet_classification": {"top1_accuracy": 82.1, "top5_accuracy": 94.8}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Example usage:
#   ./evaluate.sh <method_and_run_name>                             # evaluate with defaults
#   ./evaluate.sh <method_and_run_name> checkpoints/epoch_10.pt     # evaluate a specific checkpoint
#
# Called from run.sh after training, or independently.

set -e

cd /home/user

mkdir -p scoring

if [ "$#" -gt 0 ]; then
  if [ "$1" = "--run" ]; then
    python -m eval.evaluate_synthetic "$@" --output scoring/scores.json
    exit 0
  fi
  if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: bash scripts/evaluate.sh METHOD RUN_DIR [EXPERIMENT]" >&2
    echo "   or: bash scripts/evaluate.sh --run EXPERIMENT METHOD RUN_DIR [--run ...]" >&2
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
  python -m eval.evaluate_synthetic --run "${EXPERIMENT}" "${METHOD}" "${RUN_DIR}" --output scoring/scores.json
  exit 0
fi

RUN_ARGS=()
EVAL_DIMS="${EVAL_DIMS:-2 5 10}"
INCLUDE_SANITY_BASELINES="${INCLUDE_SANITY_BASELINES:-0}"
EVAL_METHODS="${EVAL_METHODS:-sgdd}"

for DIM in ${EVAL_DIMS}; do
  METHODS="${EVAL_METHODS}"
  if [ "${INCLUDE_SANITY_BASELINES}" = "1" ]; then
    METHODS="${METHODS} direct_mh"
  fi
  for METHOD in ${METHODS}; do
    RUN_DIR="checkpoints/${METHOD}_synthetic_d${DIM}"
    if [ -f "${RUN_DIR}/samples.npz" ]; then
      RUN_ARGS+=(--run "synthetic_d${DIM}" "${METHOD}" "${RUN_DIR}")
    fi
  done
done

if [ "${#RUN_ARGS[@]}" -eq 0 ]; then
  echo "No generated sample artifacts found under checkpoints/." >&2
  exit 1
fi

python -m eval.evaluate_synthetic "${RUN_ARGS[@]}" --output scoring/scores.json
