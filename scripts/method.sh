#!/bin/bash
# Public sampler interface: PROBLEM_JSON OUTPUT_DIR.
set -euo pipefail
cd /home/user
bash scripts/samplers/sgdd.sh "$@"
