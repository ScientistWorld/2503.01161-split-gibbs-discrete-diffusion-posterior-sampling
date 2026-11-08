#!/bin/bash
# Comparator: direct single-site Metropolis-Hastings on prior x likelihood (100 sweeps of single-site moves).
set -euo pipefail
cd /home/user
python -m baseline.direct_mh_synthetic --problem "$1" --output-dir "$2" --steps 100
