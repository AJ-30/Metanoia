#!/usr/bin/env bash

# Trains the mask VAE on the floor-plan segmentation masks. Archives artifacts
# from any previous run, then launches mask_VAE/run.py with the given mode.
#
# Usage:
#   ./train_mask_VAE.sh          # runs in debug mode
#   ./train_mask_VAE.sh prod     # runs in production mode

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

echo "Archiving previous mask VAE artifacts..."
mkdir -p mask_VAE/old_telemetry_logs mask_VAE/old_outputs
mv mask_VAE/telemetry_logs/* mask_VAE/old_telemetry_logs/ 2>/dev/null || true
mv mask_VAE/outputs/* mask_VAE/old_outputs/ 2>/dev/null || true

echo "Running mask VAE training in ${MODE} mode..."
python mask_VAE/run.py --mode "${MODE}"
