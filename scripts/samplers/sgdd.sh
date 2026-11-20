#!/bin/bash
set -euo pipefail
python -m method.sgdd_image --problem "$1" --output-dir "$2"
