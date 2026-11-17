#!/bin/bash
# Public original model (1.39 GB), evaluator classifier and MNIST (12 MB).
set -euo pipefail
cd /home/user
mkdir -p data/downloads/mnist eval/test/assets
fetch() { if [ ! -s "$2" ]; then curl -L --fail --retry 4 "$1" -o "$2.part"; mv "$2.part" "$2"; fi; }
base=https://github.com/chuwd19/Split-Gibbs-Discrete-Diffusion-Posterior-Sampling/releases/download/model-weights
fetch "$base/mnist_uniform.zip" data/downloads/mnist_uniform.zip
fetch "$base/mnist_classifier_eval.pth" data/downloads/mnist_classifier_eval.pth
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz; do
 fetch "https://ossci-datasets.s3.amazonaws.com/mnist/$f" "data/downloads/mnist/$f"
done
for f in t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
 fetch "https://ossci-datasets.s3.amazonaws.com/mnist/$f" "eval/test/assets/$f"
done
python3 - <<'PY'
import hashlib,zipfile
from pathlib import Path
artifacts={'data/downloads/mnist_uniform.zip':'d623a0511047901adf89cd57d3b8ad75dd8a084329add4795c761d3a6b1b7ceb','data/downloads/mnist_classifier_eval.pth':'2ee30fb2934a77286971b7d65bb52461c0cbb13f7cf55985715d5b7a8d6613ad'}
for path,digest in artifacts.items():
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 if h.hexdigest()!=digest: raise SystemExit('SHA256 mismatch: '+path)
if not Path('data/downloads/mnist_uniform/checkpoints-meta/checkpoint.pth').exists():
 with zipfile.ZipFile('data/downloads/mnist_uniform.zip') as z:z.extractall('data/downloads')
PY
