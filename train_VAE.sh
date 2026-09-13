#!/usr/bin/env bash

# Trains the VAE on the floor-plan images. Archives artifacts from any
# previous run, then launches VAE/run.py with the given mode.
#
# Usage:
#   ./train_VAE.sh          # runs in debug mode
#   ./train_VAE.sh prod     # runs in production mode

set -e

# Resolve the repo root so the script works from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Activating Conda environment..."
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ./venv

MODE="${1:-debug}"

echo "Archiving previous VAE artifacts..."
mkdir -p VAE/old_telemetry_logs VAE/old_outputs
mv VAE/telemetry_logs/* VAE/old_telemetry_logs/ 2>/dev/null || true
mv VAE/outputs/* VAE/old_outputs/ 2>/dev/null || true

echo "Running VAE training in ${MODE} mode..."
python VAE/run.py --mode "${MODE}"