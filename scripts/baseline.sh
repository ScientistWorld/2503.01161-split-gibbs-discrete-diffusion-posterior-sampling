#!/bin/bash
# Run baselines on the public train instances and score them.
#   bash scripts/baseline.sh                # direct_mh svdd_pm smc
#   bash scripts/baseline.sh svdd_pm smc    # selected baselines
# svdd_pm and smc are the paper's SVDD-PM and SMC baselines (Appendix C.2);
# direct_mh is direct Metropolis-Hastings on prior x likelihood.
set -euo pipefail
cd /home/user
if [ "$#" -gt 0 ]; then
  BASELINES=("$@")
else
  BASELINES=(direct_mh svdd_pm smc)
fi
python -m eval.train.evaluate_synthetic --methods "${BASELINES[@]}" --output checkpoints/train/scores_baselines.json
cat checkpoints/train/scores_baselines.json
