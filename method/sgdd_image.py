"""Released SGDD on binary MNIST: reverse prior diffusion, then split-MH.

Algorithm: sampling/sgdd.py at the upstream commit in data/sedd/PROVENANCE.md.
Single-site likelihood MH is calculated from incident edges only; this gives
exactly the same log acceptance ratio as recomputing the full operator.
"""
import argparse
import numpy as np
from data.image_problem import load_problem,save_samples,FrozenPrior


def mh(x,problem,steps,rng,eta=None,anchor=None):
    z=x.copy(); n=len(z);rows=np.arange(n)
    pi,other,valid=problem.incidence()
    obs=np.pad(problem.observations,((0,0),(0,1)))
    if anchor is None:anchor=x
    beta=np.clip(.5*(1-np.exp(-eta)),1e-12,1-1e-12) if eta is not None else .5
    mismatch_weight=np.log(beta)-np.log1p(-beta)
    for _ in range(steps):
        col=rng.integers(0,problem.dim,n);value=rng.integers(0,2,n,dtype=np.int8)
        old=z[rows,col];neighbors=z[rows[:,None],other[col]]
        a=(old[:,None]^neighbors) if problem.kind=='xor' else (old[:,None]&neighbors)
        b=(value[:,None]^neighbors) if problem.kind=='xor' else (value[:,None]&neighbors)
        y=obs[rows[:,None],pi[col]]
        delta=(((b!=y).astype(int)-(a!=y).astype(int))*valid[col]).sum(1)
        dh=(value!=anchor[rows,col]).astype(int)-(old!=anchor[rows,col]).astype(int)
        loga=-problem.strength*delta+mismatch_weight*dh
        accept=np.log(rng.random(n)+1e-300)<np.minimum(loga,0)
        z[rows[accept],col[accept]]=value[accept]
    return z


def run(problem,kind='sgdd',iterations=None,steps=20,seed_offset=0):
    rng=np.random.default_rng(problem.seed+seed_offset)
    x=rng.integers(0,2,(problem.n,problem.dim),dtype=np.int8)
    K=iterations or (50 if problem.kind=='xor' else 100)
    T=2000 if problem.kind=='xor' else 5000
    if kind=='direct_mh':return mh(x,problem,K*T,rng)
    prior=FrozenPrior()
    if kind=='unconditional':return prior.reverse(x,1.,128,rng)
    for k,t in enumerate(np.linspace(1.,1e-5,K)):
        if kind=='fixed_eta':t=(np.log(.5)-np.log(1e-4))/np.log(20/1e-4)
        x0=prior.reverse(x,float(t),steps,rng)
        sigma=1e-4**(1-t)*20.0**t
        x=mh(x0,problem,T,rng,eta=sigma,anchor=x0)
        if k%10==0:print(f'iteration {k}/{K}',flush=True)
    return x


def main():
    p=argparse.ArgumentParser();p.add_argument('--problem',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--kind',default='sgdd',choices=['sgdd','fixed_eta','unconditional','direct_mh'])
    p.add_argument('--iterations',type=int);p.add_argument('--steps',type=int,default=20)
    p.add_argument('--seed-offset',type=int,default=0);a=p.parse_args();problem=load_problem(a.problem)
    samples=run(problem,a.kind,a.iterations,a.steps,a.seed_offset)
    save_samples(a.output_dir,samples,{'algorithm':a.kind,'iterations':a.iterations})
if __name__=='__main__':main()
