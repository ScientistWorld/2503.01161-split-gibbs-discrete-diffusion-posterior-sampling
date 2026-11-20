#!/bin/bash
# One serialized GPU job: final reference/baselines and the predeclared audit.
set -euo pipefail
cd /home/user
restore_launcher() {
    # Replacing the inode avoids changing a script while bash is reading it.
    local launcher
    launcher="$(mktemp scripts/.run.XXXXXX)"
    cp scripts/reproduce.sh "$launcher"
    chmod +x "$launcher"
    mv -f "$launcher" scripts/run.sh
}
trap restore_launcher EXIT
python -m redteam.final_hardening.checks
python -m redteam.final_hardening.audit_job "$@"
python validate.py --compare
