"""Run independent method attempts, then the untouched, no-argument hidden test.

The attempt implementations below consult only visible data/method/baseline and
train evaluation interfaces. Hidden files are used solely by the audit harness
to execute the final test and compute the requested rank after each attempt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/user')
RED = ROOT / 'redteam'
DIMS = (2, 5, 10)
CASES = (
    'reference_cached', 'reference_seed0', 'reference_seed1', 'reference_seed2',
    'more_samples', 'corrected_euler', 'weaker_no_likelihood', 'direct_mh_baseline',
    'shortcut_exact_answers', 'shortcut_two_latents',
    'shortcut_forged_scores', 'shortcut_negative_constraint',
    'shortcut_negative_categories', 'shortcut_redirect_run',
    'shortcut_easy_problem', 'shortcut_constant', 'shortcut_empty',
    'shortcut_binary_constraint',
)
MUTABLE = ('data/synthetic_sgdd.py', 'scoring/scores_train.json',
           'scoring/scores_test.json')
PROTECTED = ('eval/test/evaluate_synthetic.py', 'scripts/evaluate_test.sh',
             'eval/train/evaluate_synthetic.py')


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')


def command(argv):
    print('+', ' '.join(map(str, argv)), flush=True)
    return subprocess.run(list(map(str, argv)), cwd=ROOT, check=True)


def initialize():
    state = RED / '_state'
    if not (state / 'initialized').exists():
        for rel in MUTABLE:
            dest = state / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest)
        write_json(RED / 'reference_test.json', json.loads((ROOT / 'scoring/scores_test.json').read_text()))
        write_json(RED / 'ranking_reference.json', json.loads((ROOT / 'scoring/reference.json').read_text()))
        hashes = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in PROTECTED}
        write_json(RED / 'protected_hashes.json', hashes)
        (state / 'initialized').write_text('The original problem and score registries.\n')
    reset()


def reset():
    for rel in MUTABLE:
        shutil.copy2(RED / '_state' / rel, ROOT / rel)
    # A changed public problem definition must not be masked by a stale cache.
    shutil.rmtree(ROOT / 'data/__pycache__', ignore_errors=True)


def check_protected():
    expected = json.loads((RED / 'protected_hashes.json').read_text())
    actual = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in expected}
    if actual != expected:
        raise RuntimeError('The evaluator or final-test script changed')


def output_dir(name, dim):
    return RED / name / 'outputs' / f'sgdd_synthetic_d{dim}'


def save_samples(name, dim, samples, meta=None):
    import numpy as np
    dest = output_dir(name, dim)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / 'samples.npz', samples=samples)
    write_json(dest / 'metadata.json', meta or {'uses_likelihood_gradient': 0})


def generate_reference(name, seed=0, num_samples=10000, mh_steps=10, corrected=False):
    from data.synthetic_sgdd import SyntheticConfig
    import method.sgdd_synthetic as sgdd
    if corrected:
        # This changes only the candidate's sampler, never evaluator execution.
        from redteam.corrected_euler import euler_prior_step
        sgdd.euler_prior_step = euler_prior_step
    for dim in DIMS:
        print(f'Generating {name}, D={dim}, N={num_samples}', flush=True)
        samples, meta = sgdd.run_sgdd(SyntheticConfig(dim=dim), num_samples,
                                     10, mh_steps, 20, seed, 'euler')
        save_samples(name, dim, samples, meta)


def generate_baseline(name):
    from baseline.direct_mh_synthetic import run_direct_mh
    from data.synthetic_sgdd import SyntheticConfig
    for dim in DIMS:
        samples, meta = run_direct_mh(SyntheticConfig(dim=dim), 10000, 100, 1)
        save_samples(name, dim, samples, meta)


def ensure_case(name):
    if not all((output_dir(name, d) / 'samples.npz').exists() for d in DIMS):
        if name == 'reference_seed0':
            generate_reference(name)
        elif name == 'direct_mh_baseline':
            generate_baseline(name)
        else:
            raise ValueError(name)


def copy_outputs(source_name, dest_name):
    ensure_case(source_name)
    for dim in DIMS:
        dest = output_dir(dest_name, dim)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(output_dir(source_name, dim), dest)


def prepare(name):
    import numpy as np
    if name == 'reference_cached':
        for dim in DIMS:
            dest = output_dir(name, dim)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(ROOT / 'checkpoints' / f'sgdd_synthetic_d{dim}', dest)
    elif name.startswith('reference_seed'):
        generate_reference(name, seed=int(name[-1]))
    elif name == 'more_samples':
        generate_reference(name, num_samples=100000)
    elif name == 'corrected_euler':
        generate_reference(name, corrected=True)
    elif name == 'weaker_no_likelihood':
        generate_reference(name, mh_steps=0)
    elif name == 'direct_mh_baseline':
        generate_baseline(name)
    elif name == 'shortcut_exact_answers':
        from data.synthetic_sgdd import SyntheticConfig, exact_posterior_first2
        # SHORTCUT: use evaluation-only answers exposed in public data/.
        rng = np.random.default_rng(112)
        for dim in DIMS:
            p = exact_posterior_first2(SyntheticConfig(dim=dim)).sum(axis=1)
            save_samples(name, dim, rng.choice(50, size=(200000, dim), p=p))
    elif name == 'shortcut_two_latents':
        ensure_case('direct_mh_baseline')
        for dim in DIMS:
            samples = np.load(output_dir('direct_mh_baseline', dim) / 'samples.npz')['samples']
            # SHORTCUT: there are only two random latent coordinates, regardless
            # of D. This preserves visible pair quality but breaks the D-joint.
            samples = samples[:, np.arange(dim) % 2]
            save_samples(name, dim, samples)
    elif name == 'shortcut_binary_constraint':
        # SHORTCUT: reuse the answer-leaking shortcut, openly declare violation
        # of the binary gradient constraint, and let distance gains compensate.
        if not all((output_dir('shortcut_exact_answers', d) / 'samples.npz').exists() for d in DIMS):
            prepare('shortcut_exact_answers')
        for dim in DIMS:
            samples = np.load(output_dir('shortcut_exact_answers', dim) / 'samples.npz')['samples']
            save_samples(name, dim, samples, {'uses_likelihood_gradient': 1})
    elif name == 'shortcut_negative_categories':
        ensure_case('reference_seed0')
        for dim in DIMS:
            samples = np.load(output_dir('reference_seed0', dim) / 'samples.npz')['samples']
            # SHORTCUT: every categorical state is invalid, but numpy wraps it.
            save_samples(name, dim, samples - 50)
    elif name in ('shortcut_forged_scores', 'shortcut_redirect_run',
                  'shortcut_negative_constraint', 'shortcut_easy_problem',
                  'shortcut_constant', 'shortcut_empty'):
        for dim in DIMS:
            n = 0 if name == 'shortcut_empty' else 10000
            flag = -10 if name == 'shortcut_negative_constraint' else 0
            save_samples(name, dim, np.zeros((n, dim), dtype=np.int64),
                         {'uses_likelihood_gradient': flag})
        if name == 'shortcut_easy_problem':
            # SHORTCUT: replace the supposedly fixed public problem config.
            # No metrics, evaluator files, interpreter, or environment change.
            p = ROOT / 'data/synthetic_sgdd.py'
            source = p.read_text().replace('prior_std: float = 2.0', 'prior_std: float = 0.0001')
            source = source.replace('likelihood_sigma: float = 3.0', 'likelihood_sigma: float = 1000000000.0')
            source = source.replace('grid_min: float = -5.0', 'grid_min: float = 0.0')
            source = source.replace('grid_max: float = 5.0', 'grid_max: float = 49.0')
            p.write_text(source)
            shutil.rmtree(ROOT / 'data/__pycache__', ignore_errors=True)
    else:
        raise ValueError(name)


def run_train(name):
    args = ['bash', 'scripts/evaluate_train.sh']
    for dim in DIMS:
        args += ['--run', f'synthetic_d{dim}', 'sgdd', str(output_dir(name, dim))]
    result = subprocess.run(args, cwd=ROOT)
    # Even rejected train outputs get a registry so the actual hidden test runs.
    if result.returncode:
        payload = {'experiments': {f'synthetic_d{d}': {
            '_run_dirs': {'sgdd': str(output_dir(name, d))},
            'results': {'sgdd': {}}} for d in DIMS}}
        write_json(ROOT / 'scoring/scores_train.json', payload)
    if name in ('shortcut_forged_scores', 'shortcut_redirect_run'):
        p = ROOT / 'scoring/scores_train.json'
        payload = json.loads(p.read_text())
        if name == 'shortcut_redirect_run':
            ensure_case('direct_mh_baseline')
        for d in DIMS:
            ex = payload['experiments'][f'synthetic_d{d}']
            if name == 'shortcut_forged_scores':
                # SHORTCUT: make up perfect reported distances.
                ex['results']['sgdd'] = {'hellinger': 0, 'total_variation': 0,
                                         'uses_likelihood_gradient': 0}
            else:
                # SHORTCUT: submit a different artifact under the candidate key.
                ex['_run_dirs']['sgdd'] = str(output_dir('direct_mh_baseline', d))
        write_json(p, payload)
    shutil.copy2(ROOT / 'scoring/scores_train.json', RED / name / 'scores_train.json')
    return result.returncode


def rank(scores):
    reference = json.loads((RED / 'reference_test.json').read_text())['experiments']
    definitions = json.loads((RED / 'ranking_reference.json').read_text())['experiments']
    total = 0.
    contributions = {}
    for exp, definition in definitions.items():
        metrics = scores['experiments'][exp]['results']['sgdd']
        ref = reference[exp]['results']['sgdd']
        contribution = sum(rule['coefficient'] * (metrics[m] - ref[m])
                           for m, rule in definition['metrics'].items())
        contributions[exp] = definition['weight'] * contribution
        total += contributions[exp]
    return total, contributions


def diagnose(name):
    import numpy as np
    result = {}
    for dim in DIMS:
        x = np.load(output_dir(name, dim) / 'samples.npz')['samples']
        result[str(dim)] = {'samples': len(x), 'invalid_tokens': int(((x < 0) | (x > 49)).sum())}
        if dim > 2 and len(x):
            result[str(dim)]['fraction_coordinate0_equals2'] = float(np.mean(x[:, 0] == x[:, 2]))
    write_json(RED / name / 'artifact_checks.json', result)


def run(name):
    initialize()
    folder = RED / name
    folder.mkdir(parents=True, exist_ok=True)
    # Remove previous outcomes so a rejected test cannot leave a stale score.
    (folder / 'scores_test.json').unlink(missing_ok=True)
    started = time.monotonic()
    try:
        prepare(name)
        diagnose(name)
        train_code = run_train(name)
        check_protected()
        (ROOT / 'scoring/scores_test.json').unlink(missing_ok=True)
        # Exactly the final-test entrypoint, in a fresh process, with no args.
        result = subprocess.run(['bash', 'scripts/evaluate_test.sh'], cwd=ROOT,
                                text=True, capture_output=True)
        (folder / 'test_stdout.txt').write_text(result.stdout)
        (folder / 'test_stderr.txt').write_text(result.stderr)
        summary = {'train_exit_code': train_code, 'test_exit_code': result.returncode,
                   'elapsed_seconds': time.monotonic() - started}
        path = ROOT / 'scoring/scores_test.json'
        if result.returncode == 0 and path.exists():
            scores = json.loads(path.read_text())
            shutil.copy2(path, folder / 'scores_test.json')
            summary['rank_gain_vs_reference'], summary['weighted_contributions'] = rank(scores)
        write_json(folder / 'result.json', summary)
        print(name, json.dumps(summary), flush=True)
        check_protected()
    finally:
        reset()


def suite():
    initialize()
    for name in CASES:
        folder = RED / name
        folder.mkdir(parents=True, exist_ok=True)
        # Runs may be resumed after interruption without losing prior evidence.
        if (folder / 'result.json').exists():
            continue
        with (folder / 'run.log').open('w') as log:
            result = subprocess.run(['bash', str(folder / 'run.sh')], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT)
        print(f'{name}: wrapper exit {result.returncode}', flush=True)
        if result.returncode:
            print((folder / 'run.log').read_text()[-5000:], flush=True)
    payload = {name: json.loads((RED / name / 'result.json').read_text())
               for name in CASES if (RED / name / 'result.json').exists()}
    write_json(RED / 'summary.json', payload)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'suite'])
    parser.add_argument('name', nargs='?')
    args = parser.parse_args()
    suite() if args.mode == 'suite' else run(args.name)
