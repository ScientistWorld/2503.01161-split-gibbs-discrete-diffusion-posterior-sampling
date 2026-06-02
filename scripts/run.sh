#!/bin/bash

set -e

cd /home/user

bash scripts/evaluate.sh
bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
python validate.py --compare
