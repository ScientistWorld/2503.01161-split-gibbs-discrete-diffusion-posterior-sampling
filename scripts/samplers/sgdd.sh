#!/bin/bash
# Reference method: SGDD with the paper's synthetic settings (K=10, T=10, H=20).
# Sampler contract: bash scripts/samplers/<name>.sh PROBLEM_JSON OUTPUT_DIR
set -euo pipefail
cd /home/user
python -m method.sgdd_synthetic --problem "$1" --output-dir "$2"
