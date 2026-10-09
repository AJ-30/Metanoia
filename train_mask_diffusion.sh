#!/usr/bin/env bash

# Trains the missing-component (TV) diffusion model in frozen mask VAE latent
# spaces. Archives artifacts from any previous run, then launches
# mask_diffusion/run.py with the given mode.
#
# The three frozen mask VAE checkpoints it depends on are NOT touched here; they
# are referenced by path from mask_diffusion/config.toml.
#
# NOTE: the latent cache in mask_diffusion/latent_cache is deliberately NOT
# archived. It is invalidated automatically by a fingerprint over the frozen
# checkpoints, the split dirs and the relevant hyperparameters, so keeping it
# across runs saves the one-time encode of ~70k samples.
#
# Usage:
#   ./train_mask_diffusion.sh          # runs in debug mode
#   ./train_mask_diffusion.sh prod     # runs in production mode

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

echo "Archiving previous mask diffusion artifacts..."
mkdir -p mask_diffusion/old_telemetry_logs mask_diffusion/old_outputs
mv mask_diffusion/telemetry_logs/* mask_diffusion/old_telemetry_logs/ 2>/dev/null || true
mv mask_diffusion/outputs/* mask_diffusion/old_outputs/ 2>/dev/null || true

echo "Running mask diffusion training in ${MODE} mode..."
python mask_diffusion/run.py --mode "${MODE}"