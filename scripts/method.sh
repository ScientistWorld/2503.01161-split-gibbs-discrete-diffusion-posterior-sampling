#!/bin/bash
# Run the paper's method (SGDD, scripts/samplers/sgdd.sh) on the public train
# instances and score it. Outputs: checkpoints/train/sgdd/<instance>/.
set -euo pipefail
cd /home/user
python -m eval.train.evaluate_synthetic --methods sgdd --output checkpoints/train/scores_sgdd.json
cat checkpoints/train/scores_sgdd.json
