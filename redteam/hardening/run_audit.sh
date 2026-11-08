#!/bin/bash
# Hardening audit: every variant and shortcut is registered as a sampler and
# scored by exactly `bash scripts/evaluate_test.sh` with no arguments.
# Run from a job after the canonical scores were produced; it restores the
# sampler registry and data/ afterwards. Results: redteam/hardening/results/.
set -uo pipefail
cd /home/user

HD=redteam/hardening
RES=$HD/results
HELD=$RES/.held_samplers
mkdir -p "$RES" "$HELD"
cp data/synthetic_sgdd.py "$RES/.data_backup.py"

registered=()
restore() {
  for name in "${registered[@]}"; do rm -f "scripts/samplers/$name"; done
  registered=()
  if compgen -G "$HELD/*.sh" > /dev/null; then mv "$HELD"/*.sh scripts/samplers/; fi
  cp "$RES/.data_backup.py" data/synthetic_sgdd.py
}
trap restore EXIT

# Stage A: shipped samplers + all variants and shortcuts in one hidden-test run.
for script in "$HD"/samplers/*.sh; do
  name=$(basename "$script")
  [ "$name" = "shortcut_easy_problem.sh" ] && continue
  cp "$script" "scripts/samplers/$name"
  registered+=("$name")
done
echo "== Stage A: $(ls scripts/samplers | wc -l) registered samplers"
rm -f scoring/scores_test.json
bash scripts/evaluate_test.sh
echo "Stage A exit code: $?"
cp scoring/scores_test.json "$RES/test_all.json"
restore

# Stage B: replace the public problem definition, then run only that shortcut.
mv scripts/samplers/*.sh "$HELD"/
cp "$HD/samplers/shortcut_easy_problem.sh" scripts/samplers/
registered+=("shortcut_easy_problem.sh")
sed -i \
  -e 's/prior_std=float(prior\["std"\])/prior_std=0.0001/' \
  -e 's/grid_min=float(prior\["grid_min"\])/grid_min=0.0/' \
  -e 's/grid_max=float(prior\["grid_max"\])/grid_max=49.0/' \
  -e 's/likelihood_sigma=float(likelihood\["sigma"\])/likelihood_sigma=1e9/' \
  data/synthetic_sgdd.py
grep -c "prior_std=0.0001\|grid_min=0.0\|grid_max=49.0\|likelihood_sigma=1e9" data/synthetic_sgdd.py
echo "== Stage B: easy-problem shortcut"
rm -f scoring/scores_test.json
bash scripts/evaluate_test.sh
echo "Stage B exit code: $?"
cp scoring/scores_test.json "$RES/test_easy_problem.json"
restore
trap - EXIT
rm -f "$RES/.data_backup.py"
rmdir "$HELD" 2>/dev/null || true
