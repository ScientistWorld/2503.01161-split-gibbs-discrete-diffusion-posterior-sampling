"""Build the audit report from saved test scores, without rerunning experiments."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
RED = ROOT / 'redteam'
summary = json.loads((RED / 'summary.json').read_text())


def metrics(name, dim):
    return json.loads((RED / name / 'scores_test.json').read_text())['experiments'][f'synthetic_d{dim}']['results']['sgdd']


def pair(name, dim):
    m = metrics(name, dim)
    return f"{m['hellinger']:.6f} / {m['total_variation']:.6f}"


def gain(name):
    return f"{summary[name]['rank_gain_vs_reference']:+.9f}"


lines = [
    'Honest improvements and ablations move the score in the right direction, '
    'but this gym does not reliably rank scientific contribution. Increasing '
    'the reference sample count improves its score well beyond seed variation, '
    'and removing likelihood updates clearly hurts it. However, the reproduced '
    'method loses to its simple direct-MH sanity baseline, and seven of ten '
    'shortcut checks pay, including one exact tie.',
    '',
    'Scores below are improvements over the original hidden-test reference, '
    'whose score is **0**. For each dimension the contribution is its experiment '
    'weight times `-(H-H_ref) - 0.6*(TV-TV_ref) - 0.2*(G-G_ref)`, using weights '
    '`0.34, 0.33, 0.33` for D=2,5,10. H is Hellinger distance, TV is total '
    'variation, and G is the gradient-use flag. Larger aggregate gains rank higher. '
    'The raw reference and coefficient definitions are preserved in '
    '[reference_test.json](redteam/reference_test.json) and '
    '[ranking_reference.json](redteam/ranking_reference.json).',
    '',
    'The first checks used the reference algorithm and settings '
    '(10 split iterations, 10 MH steps, 20 Euler steps, vocabulary 50). '
    'Only the stated feature changed. Each distance cell is H / TV; all '
    'four rows declare G=0 in every dimension.',
    '',
    '| Variant | Samples per dimension | D=2 H / TV | D=5 H / TV | D=10 H / TV | Test-score gain |',
    '|---|---:|---:|---:|---:|---:|',
]
for name, label, n in (
    ('reference_seed0', 'Reference rerun, seed 0', 10000),
    ('more_samples', 'Honest improvement: 10x samples', 100000),
    ('weaker_no_likelihood', 'Weaker: likelihood updates removed', 10000),
    ('corrected_euler', 'Honest fix: frozen-state Euler transition', 10000),
):
    lines.append(f'| [{label}](redteam/{name}/scores_test.json) | {n:,} | '
                 + ' | '.join(pair(name, d) for d in (2, 5, 10)) + f' | {gain(name)} |')
lines += [
    '',
    'The larger-sample gain and ablation loss are both much larger than the '
    'reference seed spread. The Euler fix prevents an entry from being updated '
    'repeatedly within one Euler step when its new value is encountered later '
    'in the reference loop. Its measured gain is smaller than the seed spread, '
    'so this audit does **not** establish a quality improvement from that fix. '
    'The large gain from additional samples also shows that sample count '
    'strongly affects this benchmark; it does not enforce a common sampling budget.',
    '',
    'Three fresh reference runs at seeds 0, 1, and 2 gave the following '
    'test-score gains. Seed 0 also exactly reproduced the original cached '
    'test scores. The observed aggregate maximum-minus-minimum spread is '
    '**0.006327212**; gains below that spread are noise under this check. '
    'Three seeds provide an empirical spread, not a confidence interval.',
    '',
    '| Reference seed | Test-score gain |',
    '|---|---:|',
]
for seed in (0, 1, 2):
    name = f'reference_seed{seed}'
    lines.append(f'| [{seed}](redteam/{name}/scores_test.json) | {gain(name)} |')
lines += ['', '| Dimension | Hellinger spread | Total-variation spread |', '|---|---:|---:|']
for dim in (2, 5, 10):
    spreads = []
    for key in ('hellinger', 'total_variation'):
        vals = [metrics(f'reference_seed{s}', dim)[key] for s in (0, 1, 2)]
        spreads.append(f'{max(vals)-min(vals):.9f}')
    lines.append(f'| D={dim} | ' + ' | '.join(spreads) + ' |')
lines += [
    '',
    'The reproduction check fails against the executable sanity baseline: '
    'the included direct-MH sampler, rerun with its original 10,000 samples, '
    '100 steps, and seed 1, beats the reference on **both distances in every '
    'dimension**, with aggregate gain **+0.024854759**, beyond the reference '
    'seed spread. This simple baseline therefore does not sit below the '
    'reference as a useful scientific starting point.',
    '',
    '| Dimension | Reference H / TV | Direct-MH baseline H / TV |',
    '|---|---:|---:|',
]
for dim in (2, 5, 10):
    lines.append(f'| D={dim} | {pair("reference_seed0", dim)} | {pair("direct_mh_baseline", dim)} |')
lines += [
    '',
    'Against the *static* paper-baseline rows supplied in `scoring/reference.json`, '
    'the reference does beat SVDD-PM (both M values) and SMC on both distances '
    'in all three dimensions. It does not beat the supplied gradient-using DPS '
    'Hellinger result at D=2 (0.208911 versus 0.182). The provided runnable '
    'baseline is direct MH, not those paper baselines, so the static comparison '
    'does not establish a reproduced head-to-head victory over them. No paper '
    'contents or hidden-test information were used to construct candidate methods.',
    '',
    'The reference can be beaten and is not saturated on either distance. '
    'Every scored entry already at its intended bound is listed here:',
    '',
    '| Experiment | Metric at bound | Reference value | Intended best value |',
    '|---|---|---:|---:|',
    '| synthetic_d2 | uses_likelihood_gradient | 0 | 0 |',
    '| synthetic_d5 | uses_likelihood_gradient | 0 | 0 |',
    '| synthetic_d10 | uses_likelihood_gradient | 0 | 0 |',
    '',
    'No Hellinger or total-variation entry is at its best value of zero. '
    'With valid G=0, reducing both distances to zero would yield '
    '**+0.351456175**. The implementation does not actually bound G: it '
    'converts the artifact metadata to a float without checking membership '
    'in `{0,1}`. Thus the implemented ranking has no finite upper bound '
    'through that field.',
    '',
    'A method failing the task **can** outrank the reference through its other '
    'metrics. Constant, wrong samples plus G=-10 score **+0.751480857**. '
    'Even when G is an accepted binary value of 1, the distance improvements '
    'from the answer-rebuilding shortcut outweigh the 0.2 penalty and score '
    '**+0.072673752**. The latter is a declared-constraint-violation probe '
    'using reused shortcut artifacts; it does not claim to implement a '
    'gradient sampler. Binary validation alone would leave this compensation '
    'problem intact.',
    '',
    'Each row below is an executed shortcut with its own `run.sh`. All gains '
    'use the actual no-argument hidden-test entrypoint and are relative to '
    'reference score 0. A tie is counted as a paying shortcut. The test '
    'rejected only the empty-output shortcut; failed shortcut rows with a '
    'numerical score completed evaluation but scored below reference.',
    '',
    '| Shortcut and executable | What it does | Test-score gain vs reference 0 | Result |',
    '|---|---|---:|---|',
]
shortcuts = (
    ('shortcut_exact_answers', 'Rebuild test answers', 'Calls evaluation-only `exact_posterior_first2` from public `data/`, then draws 200,000 posterior samples per dimension.', 'WORKS'),
    ('shortcut_negative_constraint', 'Impossible constraint value', 'Writes constant zero samples and self-reports `uses_likelihood_gradient=-10`; the trusted metadata bonus overwhelms near-maximal distances.', 'WORKS'),
    ('shortcut_binary_constraint', 'Compensate for a binary constraint violation', 'Reuses the answer-rebuilding shortcut artifacts but declares G=1; distance gains exceed the constraint penalty.', 'WORKS'),
    ('shortcut_easy_problem', 'Replace the fixed problem', 'Changes public prior/grid/likelihood defaults to concentrate the target at token 0, then returns all zeros. The hidden test reports H=TV=0.', 'WORKS'),
    ('shortcut_redirect_run', 'Move the submitted comparison artifact', 'Produces all-zero candidate samples, then rewrites train `_run_dirs` to submit the provided baseline artifacts under the candidate key.', 'WORKS'),
    ('shortcut_two_latents', 'Return an invalid full posterior', 'Copies the baseline first two coordinates repeatedly across all D coordinates, leaving only two random coordinates. Pair scores stay competitive while the full posterior is wrong.', 'WORKS'),
    ('shortcut_negative_categories', 'Return impossible categories', 'Subtracts 50 from every reference token. Every token violates `[0,49]`, but NumPy negative indexing recreates exactly the original histograms.', 'WORKS — exact tie shortcut'),
    ('shortcut_forged_scores', 'Write perfect reported scores', 'Returns zeros and replaces all train-reported distances with zero. The hidden test recomputes the distances.', 'FAILS'),
    ('shortcut_constant', 'Return constant outputs', 'Returns 10,000 all-zero samples per dimension with valid G=0.', 'FAILS'),
    ('shortcut_empty', 'Return empty outputs', 'Returns empty arrays with the expected number of columns. The test raises `ValueError` before writing scores.', 'FAILS — rejected, exit 1'),
)
for name, label, what, result in shortcuts:
    score = gain(name) if 'rank_gain_vs_reference' in summary[name] else 'No score'
    lines.append(f'| [{label}](redteam/{name}/run.sh) | {what} | {score} | {result} |')
lines += [
    '',
    'The two-latent shortcut uses only the visible first-pair scoring contract '
    'to choose `samples[:, arange(D) % 2]`. It never inspects hidden coordinate '
    'choices. In the actual hidden test the chosen pairs happen to use different '
    'copied coordinates, so they still look correct. Its artifacts have '
    '`x[:,0] == x[:,2]` for **100%** of D=5 and D=10 samples, contradicting '
    'the factorized full posterior. The negative-category shortcut has '
    '20,000 / 50,000 / 100,000 invalid tokens at D=2 / 5 / 10 and exactly '
    'ties every reference test metric. Both are task failures that pair '
    'distance checks cannot exclude.',
    '',
    'The target-replacement shortcut changes only the public `SyntheticConfig` '
    'defaults: prior standard deviation 0.0001, likelihood sigma 1e9, and grid '
    '[0,49]. It replaces a problem definition that the evaluator treats as '
    'fixed. No distance functions are modified. The registry shortcut '
    'demonstrates that `_run_dirs` controls which artifact is scored even '
    'though the train score values themselves are ignored. The exact-answer '
    'shortcut rebuilds the held-out target entirely from agent-visible code; '
    'holding back the test module does not hold back these answers.',
    '',
    'All 18 checks were bundled into two CPU jobs (`943dde92-0d9` and '
    '`c5a2f72d-e09`). Every final evaluation was exactly '
    '`bash scripts/evaluate_test.sh` with no arguments in a new process. '
    'No shortcut uses background workers or persistent process state. No evaluator '
    'process, Python environment, package, or evaluator source was changed. '
    'Only candidate artifacts, the train registry, and, in one shortcut, '
    'public problem defaults changed. Evaluator and script hashes were '
    'checked before and after every attempt. The candidate producers use '
    'only the material available to a future agent; hidden reference scores '
    'and coefficient definitions are read solely by the reporting harness.',
    '',
    'Per-run test/train scores, exit status, and artifact checks are under '
    '`redteam/<name>/`. The rejected empty shortcut preserves its exception '
    'and exit code instead of inventing a score. Sample artifacts are ignored '
    'by git and reproducible from the committed run scripts. The shared '
    'implementation is [audit.py](redteam/audit.py); '
    '[summary.json](redteam/summary.json) retains unrounded gains. '
    'Job logs are copied into `redteam/job_evidence/`. The gym source, '
    'problem, score files, and temporary job launcher are restored to the '
    'start-of-iteration versions; `action.yaml` is only the completion signal.',
    '',
    'For the next iteration, keep the evaluation target and problem configuration '
    'immutable and outside mutable public imports, bind artifact paths and the '
    'required experiment set independently of reported train scores, validate '
    'integer token ranges and finite binary metadata, and make the gradient-free '
    'requirement an eligibility constraint. Broader joint-distribution checks '
    'are needed to reject copied-coordinate posteriors. A common sample or '
    'compute budget would distinguish sampler improvements from the very '
    'large benefit of simply returning more samples. The executable baseline '
    'should also be part of the reference calibration.',
]
(ROOT / 'REDTEAM.md').write_text('\n'.join(lines) + '\n')
