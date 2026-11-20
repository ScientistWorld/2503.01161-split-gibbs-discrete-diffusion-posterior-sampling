#!/bin/bash
# Resume after worker loss. Rescore completed raw outputs; run missing work.
set -euo pipefail
cd /home/user
exec bash redteam/final_hardening/run_job.sh --rescore-train --resume-test
