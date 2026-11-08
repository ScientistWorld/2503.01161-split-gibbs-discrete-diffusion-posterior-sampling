#!/bin/bash
# Paper baseline: SVDD-PM with M=20 candidates, 3 Monte Carlo value samples.
set -euo pipefail
cd /home/user
python -m baseline.paper_baselines svdd_pm --problem "$1" --output-dir "$2" --particles 20
