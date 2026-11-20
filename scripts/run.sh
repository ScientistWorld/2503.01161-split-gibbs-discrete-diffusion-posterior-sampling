#!/bin/bash
# Resume the bundled audit after fixing hidden Python package discovery.
# The completed training outputs are rescored; hidden methods all run afresh.
set -euo pipefail
cd /home/user
exec bash redteam/final_hardening/run_job.sh --rescore-train
