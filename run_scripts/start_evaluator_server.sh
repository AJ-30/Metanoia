#!/usr/bin/env bash

# Starts a NEW Flask telemetry dashboard server instance that visualises the
# MISSING-COMPONENT EVALUATION figures read from evaluator/outputs, together with
# the evaluator's per-sample telemetry CSV from evaluator/telemetry_logs. Fully
# isolated from the other dashboards: different telemetry dir, different image
# dir, different default port, so all dashboards can run side by side.
#
# Reuses the generic dashboard in helpers/server.py (no duplicated server code);
# this script is the evaluator-specific server entry point. The "eval" panel it
# shows comes from the IMAGE_KIND_PREFIXES entry helpers/server.py grew together
# with this stage.
#
# Usage:
#   ./run_scripts/start_evaluator_server.sh [port]   # default port 8086

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

PORT="${1:-8086}"

echo "Starting missing-component evaluation dashboard..."
echo "Dashboard: http://127.0.0.1:${PORT}"
python helpers/server.py --telemetry-dir ./evaluator/telemetry_logs --recon-dir ./evaluator/outputs --host 127.0.0.1 --port "${PORT}" --train-cmd ./run_eval.sh
