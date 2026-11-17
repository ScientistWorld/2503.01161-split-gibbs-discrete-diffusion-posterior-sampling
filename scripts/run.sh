#!/bin/bash
# Final bundled hardening check; restores the ordinary reproduction launcher.
set -euo pipefail
cd /home/user
bash redteam/final_hardening/run_job.sh
