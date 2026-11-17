"""Predeclared honest alternatives, on the same hidden inputs as the reference."""
import argparse
from data.image_problem import load_problem,save_samples
from method.sgdd_image import run

p=argparse.ArgumentParser();p.add_argument('kind',choices=['longer','longer_accurate','seed1','weaker'])
p.add_argument('--problem',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
problem=load_problem(a.problem)
iterations=(100 if problem.kind=='xor' else 200) if a.kind in ['longer','longer_accurate'] else None
samples=run(problem,kind='fixed_eta' if a.kind=='weaker' else 'sgdd',iterations=iterations,
            steps=40 if a.kind=='longer_accurate' else 20,seed_offset=1 if a.kind=='seed1' else 0)
save_samples(a.output_dir,samples)
