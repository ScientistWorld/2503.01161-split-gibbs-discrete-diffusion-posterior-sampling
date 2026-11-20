#!/bin/bash
# Reproduce the reference and shipped comparators end to end. One GPU, serial.
# Download public artifacts with scripts/download.sh before submitting the job.
set -euo pipefail
cd /home/user
bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
bash scripts/evaluate_test.sh
python validate.py --compare
