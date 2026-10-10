#!/usr/bin/env bash

# Starts a NEW Flask telemetry dashboard server instance that visualises the
# CONDITIONAL DIFFUSION training metrics read from
# conditional_diffusion/telemetry_logs (telemetry CSVs) with
# conditional_diffusion/outputs grids (floorplan / arrangement / overlay).
# Fully isolated from the other dashboards: different telemetry dir, different
# image dir, different default port, so all dashboards can run side by side.
#
# Reuses the generic dashboard in helpers/server.py (no duplicated server
# code); this script is the conditional-diffusion-specific server entry point.
#
# Usage:
#   ./run_scripts/start_conditional_diffusion_server.sh [port]   # default port 8082

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

PORT="${1:-8082}"

echo "Starting conditional diffusion telemetry dashboard..."
echo "Dashboard: http://127.0.0.1:${PORT}"
python helpers/server.py --telemetry-dir ./conditional_diffusion/telemetry_logs --recon-dir ./conditional_diffusion/outputs --host 127.0.0.1 --port "${PORT}" --train-cmd ./train_conditional_diffusion.sh
