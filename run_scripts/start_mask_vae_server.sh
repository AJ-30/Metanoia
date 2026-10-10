#!/usr/bin/env bash

# Starts the Flask telemetry dashboard server that visualises the mask VAE
# training metrics read from mask_VAE/telemetry_logs (telemetry CSVs) together
# with the colorized reconstruction and generation grids from mask_VAE/outputs.
#
# The grids this model saves are already colored through the dataset's
# *color_index.json maps, so the dashboard shows floor plans rather than raw
# class-id masks.
#
# Usage:
#   ./run_scripts/start_mask_vae_server.sh [port]

set -e

# Resolve the repo root so the script works from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "Activating Conda environment..."
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ./venv

PORT="${1:-8080}"

echo "Starting mask VAE telemetry dashboard..."
echo "Dashboard: http://127.0.0.1:${PORT}"
python helpers/server.py --telemetry-dir ./mask_VAE/telemetry_logs --recon-dir ./mask_VAE/outputs --host 127.0.0.1 --port "${PORT}" --train-cmd ./train_mask_VAE.sh
