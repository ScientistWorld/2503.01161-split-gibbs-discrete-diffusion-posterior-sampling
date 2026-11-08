#!/bin/bash
# Hardening-audit sampler (registered temporarily in scripts/samplers/ during the audit).
set -euo pipefail
cd /home/user
python redteam/hardening/variants.py exact_posterior "$1" "$2"
