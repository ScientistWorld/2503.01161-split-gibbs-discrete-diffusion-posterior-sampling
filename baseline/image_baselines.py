"""Paper Appendix C.2 SVDD (Eq.55 MC soft value) and SMC on a frozen prior."""
import argparse
import numpy as np
from data.image_problem import load_problem,save_samples,FrozenPrior,ImageProblem


def subset(problem,indices):
    spec=dict(problem.spec);spec['num_observations']=len(indices)
    return ImageProblem(spec,problem.pairs,problem.observations[indices])


def run(problem,kind,particles=20,steps=128):
    rng=np.random.default_rng(problem.seed);prior=FrozenPrior()
    # Three MC reverse-denoiser draws, as in Appendix C.2. Both baselines use
    # identical frozen prior, observations, and M; no scores supplied by methods.
    x=rng.integers(0,2,(particles,problem.n,problem.dim),dtype=np.int8)
    def value(x,t):
        flat=x.reshape(-1,problem.dim)
        repeated=subset(problem,np.tile(np.arange(problem.n),particles))
        draws=[repeated.log_likelihood(prior.reverse(flat,t,1,rng)) for _ in range(3)]
        draws=np.stack(draws);peak=draws.max(0)
        return (peak+np.log(np.exp(draws-peak).mean(0))).reshape(particles,problem.n)
    old=value(x,1.) if kind=='smc' else None
    for k,(t,s) in enumerate(zip(np.linspace(1,1e-5,steps+1)[:-1],np.linspace(1,1e-5,steps+1)[1:])):
        st=1e-4**(1-t)*20**t;ss=1e-4**(1-s)*20**s
        x=prior.transition(x.reshape(-1,problem.dim),st,st-ss,rng).reshape(x.shape)
        v=value(x,float(s))
        if kind=='svdd_mc':
            pick=v.argmax(0)
            parent=x[pick,np.arange(problem.n)]
            x=np.repeat(parent[None],particles,axis=0)
        else:
            w=v-old;w=np.exp(w-w.max(0));cdf=np.cumsum(w/w.sum(0),axis=0)
            u=rng.random((particles,problem.n))
            pick=np.minimum((u[:,None,:]>cdf[None,:,:]).sum(1),particles-1)
            x=x[pick,np.arange(problem.n)[None,:]];old=v[pick,np.arange(problem.n)[None,:]]
        if k%32==0:print(f'{kind} step {k}/{steps}',flush=True)
    return prior.reverse(x[0],1e-5,1,rng)


def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['svdd_mc','smc']);p.add_argument('--problem',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--particles',type=int,default=20);p.add_argument('--steps',type=int,default=128);a=p.parse_args()
    save_samples(a.output_dir,run(load_problem(a.problem),a.kind,a.particles,a.steps))
if __name__=='__main__':main()
