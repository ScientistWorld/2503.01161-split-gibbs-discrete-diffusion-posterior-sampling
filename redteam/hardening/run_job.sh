#!/bin/bash
# Hardening job (was scripts/run.sh for jobs cd871d51-488 and c4b7c79e-ce6): smoke checks, canonical reference scores, then the audit of
# variants and shortcuts through the no-argument hidden test.
set -uo pipefail
cd /home/user

python redteam/hardening/smoke.py || { echo "SMOKE FAILED"; exit 1; }

bash scripts/evaluate_train.sh || { echo "TRAIN EVAL FAILED"; exit 1; }
cp scoring/scores_train.json scoring/scores.json
bash scripts/evaluate_test.sh || { echo "TEST EVAL FAILED"; exit 1; }
mkdir -p redteam/hardening/results
cp scoring/scores_test.json redteam/hardening/results/canonical_test.json
cp scoring/scores_train.json redteam/hardening/results/canonical_train.json
python redteam/hardening/report.py scoring/scores_test.json redteam/hardening/results/canonical_summary.json

bash redteam/hardening/run_audit.sh
python redteam/hardening/report.py redteam/hardening/results/test_all.json redteam/hardening/results/audit_summary.json
python redteam/hardening/report.py redteam/hardening/results/test_easy_problem.json \
  redteam/hardening/results/easy_problem_summary.json redteam/hardening/results/canonical_test.json

# The audit's forged-score shortcut overwrites score files; restore canonical ones.
cp redteam/hardening/results/canonical_test.json scoring/scores_test.json
cp redteam/hardening/results/canonical_train.json scoring/scores_train.json
cp redteam/hardening/results/canonical_train.json scoring/scores.json
python validate.py --compare || echo "VALIDATE FAILED (reference.json gym-scale values are filled after this job)"
