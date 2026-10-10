#!/usr/bin/env bash

# Starts a NEW Flask telemetry dashboard server instance that visualises the
# MASK DIFFUSION training metrics read from mask_diffusion/telemetry_logs
# (telemetry CSVs) with mask_diffusion/outputs grids (ground truth / diffusion
# generations / TV-over-ground-truth). Fully isolated from the other dashboards:
# different telemetry dir, different image dir, different default port, so all
# dashboards can run side by side.
#
# Reuses the generic dashboard in helpers/server.py (no duplicated server code);
# this script is the mask-diffusion-specific server entry point.
#
# Usage:
#   ./run_scripts/start_mask_diffusion_server.sh [port]   # default port 8083

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

PORT="${1:-8083}"

echo "Starting mask diffusion telemetry dashboard..."
echo "Dashboard: http://127.0.0.1:${PORT}"
python helpers/server.py --telemetry-dir ./mask_diffusion/telemetry_logs --recon-dir ./mask_diffusion/outputs --host 127.0.0.1 --port "${PORT}" --train-cmd ./train_mask_diffusion.sh