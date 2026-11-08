#!/bin/bash
# Reproduce the gym's reference scores end to end (CPU only, one core):
# every registered sampler (SGDD reference, direct_mh, svdd_pm, smc) is run on
# the public train instances and on the hidden test instances, then validated.
# The hardening audit of variants and shortcuts is redteam/hardening/run_job.sh.
set -euo pipefail
cd /home/user
bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
bash scripts/evaluate_test.sh
python validate.py --compare
