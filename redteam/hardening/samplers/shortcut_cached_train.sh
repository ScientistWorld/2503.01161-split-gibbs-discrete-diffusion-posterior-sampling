#!/bin/bash
# Hardening-audit sampler (registered temporarily in scripts/samplers/ during the audit).
set -euo pipefail
cd /home/user
python redteam/hardening/variants.py cached_train "$1" "$2"
