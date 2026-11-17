#!/bin/bash
set -euo pipefail
cd /home/user
python -m eval.test.evaluate_images "$@"
