#!/usr/bin/env bash

# Starts the demo: a Flask front end with two steps.
#
#   /complete   Step 0: a plan with a missing wall, door or window is completed by the wall/door/window model
#               (Reconstruction package, runs in this process on the GPU).
#   /           Step 1: upload page. A floor-plan class-id mask (and optionally its furniture mask) is uploaded, the
#               matching mask-diffusion model draws a TV, showcase or dining set, and the matching detector's bounding
#               boxes are drawn on the result.
#   /inpaint    Step 1 (alternative): editor that fills all three pieces.
#
# The completion step hands its result to steps 1 as the two masks they read, so the pages can also be used on their own.
# Component selection is in components.toml (diffusion checkpoint paired with detector).
# Nothing is trained; uploads land under frontend/uploads and results under frontend/outputs (frontend/config.toml).
#
# Usage:
#   ./run_frontend.sh                 # debug mode, host 127.0.0.1, port 8087
#   ./run_frontend.sh prod 0.0.0.0 9000
# Uses the conda env ./venv when it exists (setup/create_venv.sh); otherwise the active environment, or $PYTHON.

set -e

# Resolve the repo root so the script works from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ./venv ]; then
  echo "Activating Conda environment..."
  CONDA_BASE=$(conda info --base)
  # shellcheck disable=SC1091
  . "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate ./venv
fi

MODE="${1:-debug}"
HOST="${2:-127.0.0.1}"
PORT="${3:-8087}"

echo "Starting the demo in ${MODE} mode..."
echo "UI: http://${HOST}:${PORT}   (plan completion step: http://${HOST}:${PORT}/complete)"
"${PYTHON:-python}" frontend/server.py --mode "${MODE}" --host "${HOST}" --port "${PORT}"
