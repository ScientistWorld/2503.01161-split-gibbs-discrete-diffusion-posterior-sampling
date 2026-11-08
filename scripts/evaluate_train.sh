#!/bin/bash
# Visible train evaluation. With no arguments it runs every sampler registered
# as scripts/samplers/<name>.sh on the public train instances in eval/train/
# and writes scoring/scores_train.json. Problem files and sampler outputs are
# kept under checkpoints/train/<method>/<instance>/.
#   bash scripts/evaluate_train.sh --methods my_sampler   # score selected samplers only
set -euo pipefail
cd /home/user
mkdir -p scoring
python -m eval.train.evaluate_synthetic "$@"
