#!/bin/bash
set -euo pipefail
if ! command -v uv >/dev/null 2>&1; then python -m pip install uv; fi
uv pip install --system numpy==1.26.4 scipy==1.14.1 einops==0.8.0 omegaconf==2.3.0
uv pip install --system torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
