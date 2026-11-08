#!/bin/bash
# Held-out test evaluation. With no arguments it runs every sampler registered
# as scripts/samplers/<name>.sh on the hidden problem instances defined in
# eval/test/ (fixed observations, seeds and sample count) and writes
# scoring/scores_test.json. Samplers are run fresh; nothing on disk is reused.
set -euo pipefail
cd /home/user
mkdir -p scoring
python -m eval.test.evaluate_synthetic "$@"
