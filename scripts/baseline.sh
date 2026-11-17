#!/bin/bash
# Public sampler interface: BASELINE PROBLEM_JSON OUTPUT_DIR.
set -euo pipefail
cd /home/user
name="$1"
shift
case "$name" in
 direct_mh|svdd_mc|smc) bash "scripts/samplers/$name.sh" "$@" ;;
 *) echo 'baseline must be direct_mh, svdd_mc or smc' >&2; exit 2 ;;
esac
