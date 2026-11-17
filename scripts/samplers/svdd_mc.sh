#!/bin/bash
set -euo pipefail
python -m baseline.image_baselines svdd_mc --problem "$1" --output-dir "$2" --steps 16
