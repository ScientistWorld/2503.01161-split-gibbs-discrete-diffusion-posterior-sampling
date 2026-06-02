#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# You do not need to run this script during reproduction because the
# synthetic benchmark is generated procedurally from tracked code.

set -e

cd /home/user

mkdir -p data/downloads
echo "Synthetic SGDD reproduction uses procedural data; no external downloads are required."
