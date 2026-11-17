#!/bin/bash
set -euo pipefail
python -m baseline.image_baselines smc --problem "$1" --output-dir "$2" --steps 16
