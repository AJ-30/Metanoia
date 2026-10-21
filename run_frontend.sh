#!/usr/bin/env bash

# Starts the missing-component upload interface: a Flask front end where a
# component is selected (components.toml pairs its diffusion checkpoint with its
# missing_yolo detector), a floor-plan class-id mask (and optionally its
# furniture mask) is uploaded, the matching generator draws the component, and
# the matching detector's bounding boxes are drawn on the result.
#
# Nothing is trained; uploads land under frontend/uploads and results under
# frontend/outputs, both referenced from frontend/config.toml.
#
# Usage:
#   ./run_frontend.sh                 # default host 127.0.0.1, port 8087
#   ./run_frontend.sh prod 0.0.0.0 9000

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
HOST="${2:-127.0.0.1}"
PORT="${3:-8087}"

echo "Starting the missing-component upload interface in ${MODE} mode..."
echo "UI: http://${HOST}:${PORT}"
python frontend/server.py --mode "${MODE}" --host "${HOST}" --port "${PORT}"
