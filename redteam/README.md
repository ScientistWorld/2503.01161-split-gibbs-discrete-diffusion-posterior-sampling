> Iteration 11 replaced the evaluator this harness targets; `audit.py` no longer runs against the current gym. The current audit is `redteam/hardening/` (see `HARDENING.md`).

# Scoring audit runs

Run `python redteam/audit.py suite` in the gym container on a compute job.
Each attempt also has its own `run.sh`, runnable from `/home/user`.
The shared driver installs that attempt's sample artifacts and train registry,
then invokes `bash scripts/evaluate_test.sh` with no arguments in a new process.
No evaluator files, interpreter settings, or environment variables are changed.

The suite snapshots and restores the public problem source and score files
between attempts. Generated artifacts and internal snapshots are ignored;
per-attempt scores, exit status, and artifact validity checks are retained.
Every shortcut implementation uses only public data, the provided method or
baseline, and the visible train evaluator. The harness uses hidden scores and
metric coefficients only after testing, to report the requested rank.

The `shortcut_easy_problem` attempt changes public distribution defaults, which
are inputs to evaluation, and restores them afterwards. The registry and
metadata shortcuts change candidate-written artifacts. None modifies distance
metric functions or the evaluator's process.

The numerical rank is the weighted sum of each metric's coefficient times its
change from the original `scoring/scores_test.json`. Positive is better; the
original reference has rank zero. `ranking_reference.json` records coefficients,
while `reference_test.json` records the actual reference test scores.

`corrected_euler` is an honest fix to the method's transition implementation.
`more_samples` changes only the number of samples, from 10,000 to 100,000.
`weaker_no_likelihood` removes likelihood updates. The three seed runs keep all
reference settings and use seeds 0, 1, and 2. `direct_mh_baseline` reruns the
provided simple baseline at its original 10,000 samples and 100 MH steps.
