"""Report audit evidence from evaluator-produced raw-image measurements only."""
import json
from pathlib import Path
import numpy as np
from eval.ranking import gains,BOUNDS

BASE=Path('redteam/final_hardening/results')


def read(name):return json.loads((BASE/name).read_text())


def main():
    ref=json.loads(Path('scoring/reference.json').read_text())
    canonical=read('canonical_test.json');audit=read('audit_test.json')
    combined={'experiments':{}}
    for exp in ref['experiments']:
        combined['experiments'][exp]={'results':{**canonical['experiments'][exp]['results'],**audit['experiments'][exp]['results']}}
    rank=gains(combined,ref)
    scales={(e,m):float(np.std([v[m] for v in combined['experiments'][e]['results'].values()])) for e,r in ref['experiments'].items() for m in r['metrics']}
    points=read('test_points.json')['experiments'];ap=read('audit_points.json')['experiments']
    failures={n:r for e in audit['experiments'].values() for n,r in e.get('_failures',{}).items()}
    # All metrics and both experiments use the same sampled image indices.
    # Resample indices jointly, preserving their correlation and pairing.
    rng=np.random.default_rng(3071);paired={}
    for name in ['longer','longer_accurate','seed1','weaker']:
        if name not in rank or name in failures:continue
        delta=None
        for exp,e in ref['experiments'].items():
            for m,d in e['metrics'].items():
                scale=scales[exp,m]
                if scale==0:continue
                difference=np.asarray(ap[exp][name][m])-np.asarray(points[exp]['sgdd'][m])
                term=e['weight']*abs(d['coefficient'])*(1 if d['higher_is_better'] else -1)*difference/scale
                delta=term if delta is None else delta+term
        bootstrap=delta[rng.integers(0,len(delta),(4000,len(delta)))].mean(1)
        paired[name]={'gain':float(delta.mean()),'ci95':np.quantile(bootstrap,[.025,.975]).tolist(),
                      'paired_psnr':{e:float(np.mean(np.asarray(ap[e][name]['psnr'])-np.asarray(points[e]['sgdd']['psnr']))) for e in ref['experiments']}}
    seed_noise=abs(rank.get('seed1',float('inf')))
    better=[n for n in ['longer','longer_accurate'] if n in paired and paired[n]['ci95'][0]>0 and rank[n]>seed_noise]
    baseline_ok=all(rank[n]<0 for n in ['direct_mh','svdd_mc','smc'])
    weaker_ok=rank['weaker']<0
    shortcut_names=[n for n in rank if n.startswith('shortcut_')]
    shortcut_ok=all(rank[n]<0 for n in shortcut_names)
    bounds=[]
    for exp,e in ref['experiments'].items():
        for m,d in e['metrics'].items():
            value=e['results']['sgdd'][m];bound=BOUNDS[m][1 if d['higher_is_better'] else 0]
            if np.isclose(value,bound):bounds.append({'experiment':exp,'metric':m,'value':value})
    result={'gains':rank,'paired':paired,'seed1_absolute_gain':seed_noise,'better_than_noise':better,
            'baselines_below_reference':baseline_ok,'weaker_below_reference':weaker_ok,'shortcuts_below_reference':shortcut_ok,
            'reference_at_best_bound':bounds,'failures':failures}
    (BASE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    out=['\n## Final bundled job measurements\n',
         'These gains use the final metric directions, coefficient magnitudes and population standard deviations across all attempts in this audit. Before/after scalar gains use different experiments and are not numerically comparable. Only the canonical four methods are retained in the official split score files.\n',
         '| Method | Gain over final reference | Outcome |','|---|---:|---|']
    for name in sorted(rank,key=rank.get,reverse=True):
        out.append(f'| {name} | {rank[name]:+.6f} | {"failed, all worst values" if name in failures or any(name in e.get("_failures",{}) for e in canonical["experiments"].values()) else "valid"} |')
    out.extend(['\nPaired uncertainty (4,000 bootstrap resamples of the same image indices across every metric and experiment):\n',
                '| Variant | Gain | Paired 95% interval | Mean PSNR changes (XOR; AND), dB |',
                '|---|---:|---|---|'])
    for name,v in paired.items():
        changes='; '.join(f'{e}: {d:+.4f}' for e,d in v['paired_psnr'].items())
        out.append(f'| {name} | {v["gain"]:+.6f} | [{v["ci95"][0]:+.6f}, {v["ci95"][1]:+.6f}] | {changes} |')
    out.append(f'\nThe independent sampling-seed change has absolute gain {seed_noise:.6f}. Variants with a positive paired interval and a gain exceeding that seed change: {", ".join(better) or "NONE; headroom has not been established"}.')
    out.append(f'\nSimple comparators below reference: {baseline_ok}. Fixed-coupling weaker variant below reference: {weaker_ok}. Replayed shortcuts below reference: {shortcut_ok}.')
    out.extend(['\nCanonical reference, before: gain 0 by definition on the archived synthetic task. After: gain 0 by definition on the final image task. Final raw quantities:\n',
                '| Experiment | PSNR, dB | Class consistency | Measurement error |','|---|---:|---:|---:|'])
    for exp,e in ref['experiments'].items():
        v=e['results']['sgdd'];out.append(f'| {exp} | {v["psnr"]:.6f} | {v["accuracy"]:.6f} | {v["measurement_error"]:.6f} |')
    out.append('\nReference metrics at their best bound: '+(json.dumps(bounds) if bounds else 'none')+'. The sparse AND experiment is the new headroom stress test; below-bound numbers alone are not counted as demonstrated headroom.')
    out.extend(['\nShortcut replay results (the old analytic producer now errors because its factorized/chain model does not describe the new prior; logical and template solvers separately probe replacements):\n',
                '| Replay | Final gain |','|---|---:|'])
    for n in shortcut_names:out.append(f'| {n} | {rank[n]:+.6f} |')
    train=read('canonical_train.json')
    out.extend(['\nPublic/hidden reference comparison, with the same metrics and gates:\n','| Experiment | Train PSNR | Test PSNR | Train accuracy | Test accuracy | Train measurement error | Test measurement error |','|---|---:|---:|---:|---:|---:|---:|'])
    for exp in ref['experiments']:
        a=train['experiments'][exp]['results']['sgdd'];b=canonical['experiments'][exp]['results']['sgdd']
        out.append(f'| {exp} | {a["psnr"]:.6f} | {b["psnr"]:.6f} | {a["accuracy"]:.6f} | {b["accuracy"]:.6f} | {a["measurement_error"]:.6f} | {b["measurement_error"]:.6f} |')
    if not better or not baseline_ok or not weaker_ok or not shortcut_ok:
        out.append('\n**Unresolved:** '+('; '.join(x for x,b in [('an honest improvement beyond noise is unproved',not better),('a simple comparator still pays',not baseline_ok),('the designated weaker variant still pays',not weaker_ok),('a replayed shortcut still pays',not shortcut_ok)] if b))+'. The gym must not be described as fully hardened until these findings are resolved.')
    out.append('\nDetailed measurements, failure explanations and paired image measurements are in `redteam/final_hardening/results/`. No timing measurements enter these scores. Training/test comparison is retained in `canonical_train.json` and `canonical_test.json`.')
    p=Path('HARDENING.md');text=p.read_text().split('\n## Final bundled job measurements')[0]
    p.write_text(text+'\n'+'\n'.join(out)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
