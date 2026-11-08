#!/bin/bash
# Paper baseline: SMC with groups of M=20 particles, 3 Monte Carlo likelihood samples.
set -euo pipefail
cd /home/user
python -m baseline.paper_baselines smc --problem "$1" --output-dir "$2" --particles 20
