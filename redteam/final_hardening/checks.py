"""Light, meaningful correctness and failure-semantics checks run inside the job."""
import json
import tempfile
import subprocess
from pathlib import Path
import numpy as np
from data.image_problem import load_problem
from method.sgdd_image import mh
from eval.test.restrict import restrict_test_reads
from eval.ranking import gains
from eval.test.evaluate_images import load_valid_samples,InvalidOutput,WORST


def main():
    # Local incident-edge energies agree with the complete XOR/AND objective.
    rng=np.random.default_rng(31)
    for kind in ['xor','and']:
        pairs=np.array([[0,1],[0,2],[1,3],[2,3]])
        obs=rng.integers(0,2,(7,4)).tolist()
        spec={'dim':4,'seed':91,'likelihood':{'type':kind,'sigma_noise':.5,'pairs':pairs.tolist(),'observations':obs}}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'p.json';path.write_text(json.dumps(spec));p=load_problem(path)
            z=rng.integers(0,2,(7,4),dtype=np.int8);col=rng.integers(0,4,7);val=1-z[np.arange(7),col]
            proposed=z.copy();proposed[np.arange(7),col]=val
            pi,other,valid=p.incidence();y=np.pad(p.observations,((0,0),(0,1)))
            n=z[np.arange(7)[:,None],other[col]];old=z[np.arange(7),col]
            a=old[:,None]^n if kind=='xor' else old[:,None]&n
            b=val[:,None]^n if kind=='xor' else val[:,None]&n
            target=y[np.arange(7)[:,None],pi[col]]
            delta=-p.strength*(((b!=target).astype(int)-(a!=target).astype(int))*valid[col]).sum(1)
            assert np.allclose(delta,p.log_likelihood(proposed)-p.log_likelihood(z))
    # The actual container must enforce the test read boundary while letting
    # a sampler read its own inputs and all public model/source files.
    private=Path('eval/test/results');private.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=private) as d:
        own=Path(d).resolve();(own/'input.json').write_text('{}')
        code="from pathlib import Path; assert Path('data/image_problem.py').read_text(); assert Path("+repr(str(own/'input.json'))+").read_text()=='{}'\ntry:Path('eval/test/evaluate_images.py').read_text()\nexcept PermissionError:pass\nelse:raise AssertionError('test generator readable')"
        subprocess.run(['python','-c',code],check=True,preexec_fn=lambda:restrict_test_reads(own))
    # No dtype wrapping can turn impossible tokens into valid predictions.
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)
        for arr in [np.full((128,1024),-1),np.full((128,1024),2),np.zeros((0,1024)),np.full((128,1024),np.nan)]:
            np.savez(path/'samples.npz',samples=arr)
            try:load_valid_samples(path)
            except InvalidOutput:pass
            else:raise AssertionError('invalid tokens accepted')
    ref=json.loads(Path('scoring/reference.json').read_text())
    # Test direction and coefficient-sign invariance without retraining anything.
    exps={}
    for exp,e in ref['experiments'].items():
        e['results']['sgdd'].update(psnr=10.,accuracy=.5,measurement_error=.2)
        exps[exp]={'results':{'sgdd':{'psnr':10.,'accuracy':.5,'measurement_error':.2},'better':{'psnr':12.,'accuracy':.7,'measurement_error':.1},'bad':{},'worse':{'psnr':8.,'accuracy':.3,'measurement_error':.4}}}
    score={'experiments':exps};a=gains(score,ref)
    assert a['better']>a['sgdd']>a['worse']>a['bad']
    for e in ref['experiments'].values():
        for definition in e['metrics'].values():definition['coefficient']*=-1
    assert a==gains(score,ref)
    print('energy, raw-output, missing-value and ranking checks passed')
if __name__=='__main__':main()
