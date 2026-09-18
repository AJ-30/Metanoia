#!/usr/bin/env bash

# Starts a NEW Flask telemetry dashboard server instance that visualises the
# DIFFUSION training metrics read from Diffusion/telemetry_logs (telemetry
# CSVs) with Diffusion/outputs grids. Fully isolated from the VAE dashboard:
# different telemetry dir, different image dir, different default port, so
# both dashboards can run side by side.
#
# Reuses the generic dashboard in helpers/server.py (no duplicated server
# code); this script is the diffusion-specific server entry point.
#
# Usage:
#   ./run_scripts/start_diffusion_server.sh [port]   # default port 8081

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

PORT="${1:-8081}"

echo "Starting diffusion telemetry dashboard..."
echo "Dashboard: http://127.0.0.1:${PORT}"
python helpers/server.py --telemetry-dir ./Diffusion/telemetry_logs --recon-dir ./Diffusion/outputs --host 127.0.0.1 --port "${PORT}"
