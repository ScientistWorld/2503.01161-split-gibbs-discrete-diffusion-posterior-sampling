#!/bin/bash
# One serialized GPU job: final reference/baselines and the predeclared audit.
set -euo pipefail
cd /home/user
restore_launcher() { cp scripts/reproduce.sh scripts/run.sh; }
trap restore_launcher EXIT
python -m redteam.final_hardening.checks
python -m redteam.final_hardening.audit_job
python validate.py --compare
