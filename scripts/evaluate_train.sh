#!/bin/bash
set -euo pipefail
cd /home/user
python -m eval.train.evaluate_images "$@"
