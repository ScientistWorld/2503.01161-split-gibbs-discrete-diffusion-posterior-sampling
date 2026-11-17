"""Adversarial producers for the final audit; no imports from eval/."""
import argparse
import gzip
from pathlib import Path
import numpy as np
from data.image_problem import load_problem,save_samples


def logical_rule(p):
    x=np.zeros((p.n,p.dim),dtype=np.int8)
    for row,y in enumerate(p.observations):
        if p.kind=='and':
            x[row,np.unique(p.pairs[y==1])]=1
            continue
        edges=[[] for _ in range(p.dim)]
        for (a,b),v in zip(p.pairs,y):edges[a].append((b,v));edges[b].append((a,v))
        known=np.zeros(p.dim,bool)
        for start in range(p.dim):
            if known[start]:continue
            component=[start];known[start]=True
            for a in component:
                for b,v in edges[a]:
                    if not known[b]:known[b]=True;x[row,b]=x[row,a]^v;component.append(b)
            if x[row,component].sum()>len(component)/2:x[row,component]^=1
    return x


def template_lookup(p):
    # Private objective search over every publicly visible original MNIST image.
    with gzip.open('data/downloads/mnist/train-images-idx3-ubyte.gz','rb') as f:raw=f.read()
    images=np.frombuffer(raw,dtype=np.uint8,offset=16).reshape(-1,28,28)
    binary=(np.pad(images,((0,0),(2,2),(2,2)))>=128).reshape(-1,p.dim).astype(np.int8)
    best=np.full(p.n,10**9);answer=np.zeros((p.n,p.dim),dtype=np.int8)
    for begin in range(0,len(binary),2000):
        block=binary[begin:begin+2000];forward=p.forward(block)
        for k in range(p.n):
            loss=(forward!=p.observations[k]).sum(-1);i=loss.argmin()
            if loss[i]<best[k]:best[k]=loss[i];answer[k]=block[i]
    return answer


def main():
    a=argparse.ArgumentParser();a.add_argument('kind');a.add_argument('--problem',required=True);a.add_argument('--output-dir',required=True)
    args=a.parse_args();p=load_problem(args.problem);kind=args.kind
    if kind=='rule':x=logical_rule(p)
    elif kind=='template':x=template_lookup(p)
    elif kind=='copy_reference':
        with np.load(Path('eval/test/results/runs/sgdd')/p.spec['experiment']/'output/samples.npz') as f:x=f['samples'].copy()
    elif kind in ['negative','two_latents','extra']:
        # Public reference reconstructions supply realistic source arrays while
        # exercising numeric validation independently of the read boundary.
        with np.load(Path('checkpoints/train/sgdd')/p.spec['experiment']/'output/samples.npz') as f:x=f['samples'].copy()
        if kind=='negative':x=x.astype(np.int64)-2
        if kind=='two_latents':x=x[:,np.arange(p.dim)%2]
        if kind=='extra':x=np.concatenate([x,x])
    elif kind=='empty':x=np.zeros((0,p.dim),dtype=np.int8)
    elif kind=='float':x=np.zeros((p.n,p.dim),dtype=float)
    elif kind=='nan':x=np.full((p.n,p.dim),np.nan)
    elif kind=='old_exact':
        # Replay the actual former exact-posterior producer. Its Gaussian/chain
        # model cannot parse a learned image prior; failure is scored explicitly.
        from data.synthetic_sgdd import load_problem as old_load
        from redteam.hardening.variants import exact_posterior_samples
        x=exact_posterior_samples(old_load(args.problem),np.random.default_rng(p.seed))
    else:x=np.zeros((p.n,p.dim),dtype=np.int8)
    metadata={'psnr':1e99,'accuracy':1,'uses_likelihood_gradient':-10,'_run_dirs':{'sgdd':'checkpoints/test/sgdd'}}
    save_samples(args.output_dir,x,metadata)
    if kind=='forged':
        # The evaluator must neither trust nor use this as an artifact registry.
        Path('scoring/scores_train.json').write_text('{"_run_dirs":{"candidate":"checkpoints/test/sgdd"},"experiments":{}}')
    if kind=='tamper_problem':
        # Altering the method's input copy does not change evaluator-owned truth.
        Path(args.problem).write_text('{"observations":[],"num_samples":0}')
if __name__=='__main__':main()
