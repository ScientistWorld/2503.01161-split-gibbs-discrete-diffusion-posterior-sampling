#!/bin/bash
# Standard evaluation entrypoint: the visible train evaluation (eval/train/),
# written to scoring/scores.json. It runs every sampler registered as
# scripts/samplers/<name>.sh. A sampler is any script called as
#   bash scripts/samplers/<name>.sh PROBLEM_JSON OUTPUT_DIR
# that writes OUTPUT_DIR/samples.npz (integer array "samples", shape exactly
# [num_samples, dim], tokens in [0, vocab_size)). Uses only eval/ code.
set -euo pipefail
cd /home/user
mkdir -p scoring
python -m eval.train.evaluate_synthetic --output scoring/scores.json "$@"
