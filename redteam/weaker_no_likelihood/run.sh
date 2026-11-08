#!/bin/bash
set -euo pipefail
cd /home/user
python redteam/audit.py run weaker_no_likelihood
