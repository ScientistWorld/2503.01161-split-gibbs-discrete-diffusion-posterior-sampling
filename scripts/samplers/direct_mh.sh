#!/bin/bash
set -euo pipefail
python -m method.sgdd_image --kind direct_mh --problem "$1" --output-dir "$2"
