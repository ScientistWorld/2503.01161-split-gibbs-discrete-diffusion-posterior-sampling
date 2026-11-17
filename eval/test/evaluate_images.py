"""Independent test evaluator for the binary-image inverse problem.

Only inputs are written to sampler directories. Ground truth is regenerated in
this process AFTER the sampler returns. No method-written metric is read.
"""
import argparse
import gzip
import hashlib
import json
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import torch
from torch import nn
from eval.test.restrict import restrict_test_reads

ROOT=Path(__file__).resolve().parents[2]
test_NAME='test'
N=128
# Generation seeds are private to this evaluator and never enter problem.json.
GENERATION_SEED=93841729
SAMPLER_SEED=591713
EXPERIMENTS={'mnist_xor':('xor',.5,.01),'mnist_and_sparse':('and',2.,.1)}
WORST={'psnr':0.,'accuracy':0.,'measurement_error':1.}
SOURCE=ROOT/'eval/test/assets/t10k-images-idx3-ubyte.gz'
LABELS=ROOT/'eval/test/assets/t10k-labels-idx1-ubyte.gz'

class InvalidOutput(Exception): pass

class Classifier(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(1,16,3,1,1);self.fc=nn.Linear(16*16*16,10)
    def forward(self,x):
        x=torch.nn.functional.max_pool2d(torch.relu(self.conv(x)),2)
        return self.fc(x.reshape(-1,16*16*16))

def idx(path,offset):
    with gzip.open(path,'rb') as f: b=f.read()
    return np.frombuffer(b,dtype=np.uint8,offset=offset)

def latent_batch():
    # Source images are real MNIST. Continuous elastic/affine deformation makes
    # public-source lookup an approximate image method rather than an answer key.
    rng=np.random.default_rng(GENERATION_SEED)
    images=idx(SOURCE,16).reshape(-1,28,28)
    labels=idx(LABELS,8)
    eligible=np.arange(len(images))
    selected=rng.choice(eligible,N,replace=False)
    out=[]
    yy,xx=np.mgrid[:32,:32].astype(float)
    for image in images[selected]:
        base=np.pad(image,2).astype(float)
        angle=rng.uniform(-.15,.15);scale=rng.uniform(.92,1.08)
        dx=rng.uniform(-1.5,1.5);dy=rng.uniform(-1.5,1.5)
        # Many continuous degrees of freedom; no finite template catalogue.
        wx=gaussian_filter(rng.normal(size=(32,32)),2.)*5.
        wy=gaussian_filter(rng.normal(size=(32,32)),2.)*5.
        xc=(xx-15.5-dx)/scale;yc=(yy-15.5-dy)/scale
        xp=np.cos(angle)*xc+np.sin(angle)*yc+15.5+wx
        yp=-np.sin(angle)*xc+np.cos(angle)*yc+15.5+wy
        warped=map_coordinates(base,[yp,xp],order=1,mode='constant',cval=0.)
        out.append((warped>=128).astype(np.int8).ravel())
    return np.stack(out),labels[selected].copy()

def make_spec(exp):
    kind,ratio,sigma=EXPERIMENTS[exp]
    rng=np.random.default_rng(GENERATION_SEED+sorted(EXPERIMENTS).index(exp)+991)
    # Self edges give no information and complicate incidence-only MH.
    a=rng.integers(0,1024,int(ratio*1024));b=rng.integers(0,1023,len(a));b+=b>=a
    pairs=np.stack([a,b],axis=1)
    x,_=latent_batch()
    y=(x[:,a]^x[:,b]) if kind=='xor' else (x[:,a]&x[:,b])
    # Generation seed, image indices, deformation parameters and labels are absent.
    return {'experiment':exp,'dim':1024,'vocab_size':2,'num_observations':N,
            'num_samples':1,'seed':SAMPLER_SEED,'prior':{'type':'frozen_sedd',
            'path':'data/downloads/mnist_uniform','architecture':'sedd_small_original'},
            'likelihood':{'type':kind,'sigma_noise':sigma,'pairs':pairs.tolist(),
                          'observations':y.tolist()}}

def load_valid_samples(output_dir):
    try:
        with np.load(output_dir/'samples.npz',allow_pickle=False) as f: x=f['samples']
    except Exception as exc:raise InvalidOutput('unreadable/missing samples: '+type(exc).__name__)
    if x.shape!=(N,1024):raise InvalidOutput(f'shape {x.shape}; expected {(N,1024)}')
    if x.dtype==np.bool_ or not np.issubdtype(x.dtype,np.integer):raise InvalidOutput('integer dtype required')
    if np.any(x<0) or np.any(x>1):raise InvalidOutput('tokens must be 0 or 1')
    return x.astype(np.int8)

def classifier():
    path=ROOT/'data/downloads/mnist_classifier_eval.pth'
    expected='2ee30fb2934a77286971b7d65bb52461c0cbb13f7cf55985715d5b7a8d6613ad'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('fixed classifier changed')
    model=Classifier();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    model.eval().requires_grad_(False);return model

def score(x,spec,model):
    # Answers exist only here, in evaluator memory, while no sampler is running.
    truth,labels=latent_batch()
    mse=(x!=truth).mean(1)
    # Finite PSNR convention: exact recovery gets one-pixel resolution, 30.10dB.
    psnr=-10*np.log10(np.maximum(mse,1./1024))
    with torch.inference_mode():
        prediction=model(torch.tensor(x.reshape(-1,1,32,32)*128,dtype=torch.float32)).argmax(-1).numpy()
        target_class=model(torch.tensor(truth.reshape(-1,1,32,32)*128,dtype=torch.float32)).argmax(-1).numpy()
    accuracy=(prediction==target_class).astype(float)
    p=np.asarray(spec['likelihood']['pairs']);a=x[:,p[:,0]];b=x[:,p[:,1]]
    forward=(a^b) if spec['likelihood']['type']=='xor' else (a&b)
    mismatch=(forward!=np.asarray(spec['likelihood']['observations'])).mean(1)
    values={'psnr':float(psnr.mean()),'accuracy':float(accuracy.mean()),'measurement_error':float(mismatch.mean())}
    if not all(np.isfinite(list(values.values()))):raise InvalidOutput('nonfinite metric')
    if any(v<0 for v in values.values()) or values['accuracy']>1 or values['measurement_error']>1:raise InvalidOutput('impossible metric')
    return values,{'psnr':psnr.tolist(),'accuracy':accuracy.tolist(),'measurement_error':mismatch.tolist()}

def discover(directory):
    return {p.stem:p for p in sorted(directory.glob('*.sh')) if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',p.stem)}

def evaluate(samplers,workdir,rescore=False):
    torch.set_num_threads(1)
    specs={e:make_spec(e) for e in EXPERIMENTS};model=classifier()
    scores={'experiments':{e:{'_slice':test_NAME,'results':{}} for e in EXPERIMENTS}}
    detail={'experiments':{e:{} for e in EXPERIMENTS}}
    for name,script in samplers.items():
        print(f'[{test_NAME}] {name}',flush=True);failure=None;values={};points={}
        for exp,spec in specs.items():
            folder=workdir/name/exp;output=folder/'output';folder.mkdir(parents=True,exist_ok=True)
            try:
                if not rescore:
                    shutil.rmtree(output,ignore_errors=True);output.mkdir()
                    path=folder/'problem.json';path.write_text(json.dumps(spec))
                    with (folder/'sampler.log').open('w') as log:
                        # No wall-clock deadline: serialized, fixed-step inference.
                        proc=subprocess.run(['bash',str(script.resolve()),str(path.resolve()),str(output.resolve())],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,preexec_fn=lambda:restrict_test_reads(folder))
                    if proc.returncode:raise InvalidOutput(f'sampler exit {proc.returncode}')
                x=load_valid_samples(output)
                values[exp],points[exp]=score(x,spec,model)
                print(exp,values[exp],flush=True)
                v=values[exp]
                if v['psnr']<6. or v['accuracy']<.20 or v['measurement_error']>.10:
                    raise InvalidOutput('task gate: PSNR>=6dB, class accuracy>=20%, mismatch<=10% required; '+str(v))
            except (InvalidOutput,OSError,ValueError,subprocess.SubprocessError) as exc:
                reason=f'{exp}: {exc}';failure=(failure+'; '+reason) if failure else reason
                print('FAILED',reason,flush=True)
        for exp in EXPERIMENTS:
            entry=scores['experiments'][exp]
            if failure:
                entry['results'][name]=dict(WORST);entry.setdefault('_failures',{})[name]=failure
            else:
                entry['results'][name]=values[exp]
            if exp in points:detail['experiments'][exp][name]=points[exp]
        # Save incrementally so completed expensive runs survive interruption.
        target=ROOT/'scoring'/f'scores_{test_NAME}.json';target.write_text(json.dumps(scores,indent=2)+'\n')
    return scores,detail

def main():
    p=argparse.ArgumentParser();p.add_argument('--samplers-dir',type=Path,default=ROOT/'scripts/samplers')
    p.add_argument('--workdir',type=Path,default=ROOT/'eval/test/results/runs')
    p.add_argument('--rescore',action='store_true');p.add_argument('--details',type=Path,default=ROOT/'eval/test/results/points.json')
    a=p.parse_args();samplers=discover(a.samplers_dir)
    if not samplers:raise SystemExit('no samplers on disk')
    scores,detail=evaluate(samplers,a.workdir,a.rescore)
    (ROOT/'scoring'/f'scores_{test_NAME}.json').write_text(json.dumps(scores,indent=2)+'\n')
    if a.details:a.details.parent.mkdir(parents=True,exist_ok=True);a.details.write_text(json.dumps(detail))
if __name__=='__main__':main()
