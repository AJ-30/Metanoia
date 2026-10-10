#!/usr/bin/env bash

# Starts the Flask telemetry dashboard server that visualises the VAE
# training metrics read from VAE/telemetry_logs (telemetry CSVs).
#
# Usage:
#   ./run_scripts/start_server.sh [port]

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

echo "Starting telemetry dashboard..."
echo "Dashboard: http://127.0.0.1:${PORT}"
python helpers/server.py --telemetry-dir ./VAE/telemetry_logs --recon-dir ./VAE/outputs --host 127.0.0.1 --port "${PORT}" --train-cmd ./train_VAE.sh