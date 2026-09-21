#!/usr/bin/env bash

# Trains the latent diffusion model on the frozen VAE latents. Archives
# artifacts from any previous run, then launches Diffusion/run.py with the
# given mode.
#
# Usage:
#   ./train_diffusion.sh       # runs in debug mode
#   ./train_diffusion.sh prod  # runs in production mode

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

echo "Archiving previous diffusion artifacts..."
mkdir -p Diffusion/old_telemetry_logs Diffusion/old_outputs
mv Diffusion/telemetry_logs/* Diffusion/old_telemetry_logs/ 2>/dev/null || true
mv Diffusion/outputs/* Diffusion/old_outputs/ 2>/dev/null || true

echo "Running diffusion training in ${MODE} mode..."
python Diffusion/run.py --mode "${MODE}"