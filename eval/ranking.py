"""Documented ranking calculation; the only ranked inputs are experiment results.

All attempts share each metric's empirical population standard deviation. Units
and coefficient signs cannot alter priority. Invalid/missing entries get their
fixed worst value in all metrics of all experiments before standardization.
"""
import math
import numpy as np
BOUNDS={'psnr':(0.,10*math.log10(1024)),'accuracy':(0.,1.),'measurement_error':(0.,1.)}

def gains(scores,reference,reference_method='sgdd'):
    experiments=reference['experiments']
    names=sorted({n for e in scores['experiments'].values() for n in e['results']})
    values={n:{} for n in names}
    for n in names:
        failed=False
        for exp,e in experiments.items():
            row=scores['experiments'].get(exp,{}).get('results',{}).get(n,{})
            for m,definition in e['metrics'].items():
                lo,hi=BOUNDS[m];v=row.get(m)
                if not isinstance(v,(float,int)) or not np.isfinite(v) or not lo<=v<=hi:failed=True
                values[n][(exp,m)]=v
        if failed:
            for exp,e in experiments.items():
                for m,d in e['metrics'].items():values[n][exp,m]=BOUNDS[m][0 if d['higher_is_better'] else 1]
    result={n:0. for n in names}
    for exp,e in experiments.items():
        for m,d in e['metrics'].items():
            deviation=float(np.std([values[n][exp,m] for n in names]))
            if deviation==0:continue
            baseline=e['results'][reference_method][m]
            for n in names:
                improvement=(values[n][exp,m]-baseline)*(1 if d['higher_is_better'] else -1)
                result[n]+=e['weight']*abs(d['coefficient'])*improvement/deviation
    return result
