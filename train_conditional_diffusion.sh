#!/usr/bin/env bash

# Trains the floorplan-conditioned diffusion model on frozen mask VAE latents.
# Archives artifacts from any previous run, then launches
# conditional_diffusion/run.py with the given mode.
#
# Usage:
#   ./train_conditional_diffusion.sh       # runs in debug mode
#   ./train_conditional_diffusion.sh prod  # runs in production mode

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

echo "Archiving previous conditional diffusion artifacts..."
mkdir -p conditional_diffusion/old_telemetry_logs conditional_diffusion/old_outputs
mv conditional_diffusion/telemetry_logs/* conditional_diffusion/old_telemetry_logs/ 2>/dev/null || true
mv conditional_diffusion/outputs/* conditional_diffusion/old_outputs/ 2>/dev/null || true

echo "Running conditional diffusion training in ${MODE} mode..."
python conditional_diffusion/run.py --mode "${MODE}"
