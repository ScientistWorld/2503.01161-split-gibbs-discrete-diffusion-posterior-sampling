"""Public inputs and frozen SEDD interface. No evaluation images or targets."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from data.sedd.transformer import SEDD

MODEL_DIR = Path('data/downloads/mnist_uniform')

@dataclass
class ImageProblem:
    spec: dict
    pairs: np.ndarray
    observations: np.ndarray
    @property
    def dim(self): return int(self.spec['dim'])
    @property
    def n(self): return len(self.observations)
    @property
    def seed(self): return int(self.spec['seed'])
    @property
    def kind(self): return self.spec['likelihood']['type']
    @property
    def strength(self): return 1.0 / self.spec['likelihood']['sigma_noise']**2
    def forward(self, x):
        a,b=x[...,self.pairs[:,0]],x[...,self.pairs[:,1]]
        return a ^ b if self.kind == 'xor' else a & b
    def log_likelihood(self,x):
        return -self.strength*(self.forward(x)!=self.observations).sum(-1)
    def incidence(self):
        entries=[[] for _ in range(self.dim)]
        for k,(a,b) in enumerate(self.pairs):
            entries[a].append((k,b)); entries[b].append((k,a))
        width=max(map(len,entries)); pi=np.full((self.dim,width),len(self.pairs),dtype=int)
        other=np.zeros_like(pi);valid=np.zeros_like(pi,dtype=bool)
        for d,row in enumerate(entries):
            for k,(p,b) in enumerate(row): pi[d,k]=p; other[d,k]=b; valid[d,k]=True
        return pi,other,valid

def load_problem(path):
    spec=json.loads(Path(path).read_text());l=spec['likelihood']
    return ImageProblem(spec,np.asarray(l['pairs'],dtype=int),np.asarray(l['observations'],dtype=np.int8))

def save_samples(out,samples,metadata=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'samples.npz',samples=np.asarray(samples))
    if metadata: (out/'metadata.json').write_text(json.dumps(metadata,indent=2))

class FrozenPrior:
    """Authors' 90M SEDD architecture and EMA parameters, shared by every sampler."""
    def __init__(self):
        torch.set_num_threads(1)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg=OmegaConf.load(MODEL_DIR/'config.yaml')
        self.model=SEDD(cfg)
        ckpt=torch.load(MODEL_DIR/'checkpoints-meta/checkpoint.pth',map_location='cpu',weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        assert len(list(self.model.parameters()))==len(ckpt['ema']['shadow_params']), 'EMA architecture mismatch'
        for parameter,shadow in zip(self.model.parameters(),ckpt['ema']['shadow_params']):
            parameter.data.copy_(shadow)
        del ckpt
        self.model.to(self.device).eval().requires_grad_(False)
    @torch.inference_mode()
    def log_scores(self,x,sigma):
        out=[]
        for i in range(0,len(x),64):
            xt=torch.as_tensor(x[i:i+64],device=self.device,dtype=torch.long)
            noise=torch.full((len(xt),),float(sigma),device=self.device)
            out.append(self.model(xt,noise).float().cpu().numpy())
        return np.concatenate(out)
    def reverse(self,x,t_start,steps,rng):
        """Released SGDD analytic CTMC integrator + final denoiser (sampling_utils.py)."""
        times=np.linspace(t_start,1e-5,steps+1)
        for a,b in zip(times[:-1],times[1:]):
            sigma=1e-4**(1-a)*20.0**a
            next_sigma=1e-4**(1-b)*20.0**b
            x=self.transition(x,sigma,sigma-next_sigma,rng)
        sigma=1e-4**(1-times[-1])*20.0**times[-1]
        return self.transition(x,sigma,sigma,rng)
    def transition(self,x,sigma,dsigma,rng):
        score=np.exp(np.clip(self.log_scores(x,sigma),-60,60))
        epow=np.exp(-dsigma)
        staggered=(epow-1)/(2*epow)*score.sum(-1,keepdims=True)+score/epow
        probs=staggered*( (1-epow)/2 + epow*np.eye(2)[x])
        # As in upstream exponential-race categorical: negative weights cannot win.
        probs=np.maximum(probs,0)
        totals=probs.sum(-1)
        p1=np.divide(probs[...,1],totals,out=np.full_like(totals,.5),where=totals>0)
        return (rng.random(x.shape)<p1).astype(np.int8)
