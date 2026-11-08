#!/bin/bash
# Hardening-audit sampler (registered temporarily in scripts/samplers/ during the audit).
set -euo pipefail
cd /home/user
python -m baseline.direct_mh_synthetic --problem "$1" --output-dir "$2" --steps 1000
