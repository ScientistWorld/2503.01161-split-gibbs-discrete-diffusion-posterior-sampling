#!/bin/bash
# Hardening-audit sampler (registered temporarily in scripts/samplers/ during the audit).
set -euo pipefail
cd /home/user
python -m method.sgdd_synthetic --problem "$1" --output-dir "$2" --mh-steps 100
