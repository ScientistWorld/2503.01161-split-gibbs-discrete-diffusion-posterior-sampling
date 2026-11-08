#!/bin/bash
# Reproduce the gym end to end (CPU only, about 15 minutes on one core):
# run every registered sampler (SGDD reference and baselines) on the public
# train instances and on the hidden test instances, then validate.
set -euo pipefail
cd /home/user
bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
bash scripts/evaluate_test.sh
python validate.py --compare
