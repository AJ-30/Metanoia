#!/usr/bin/env bash

# Starts the Flask telemetry dashboard server that visualises missing_yolo
# training metrics read from missing_yolo/telemetry_logs (telemetry CSVs) with
# GT-vs-pred overlays and training curves from missing_yolo/outputs.
#
# The CSV carries two kinds of row, distinguished by the row_kind column:
#   detection  - one epoch's Ultralytics mAP/precision/recall
#   generation - the component/labels.txt agreement of the pixels drawn for that
#                epoch (drawn from the row's draw_seed)
# Rows of one kind leave the other kind's columns empty, so each curve has gaps
# where it has no measurement rather than invented values.
#
# Telemetry rows are written while training runs, so curves grow live - start
# this first, then run ./train_missing_yolo.sh.
#
# Usage:
#   ./run_scripts/start_missing_yolo_server.sh [port]

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

PORT="${1:-8088}"

echo "Starting missing_yolo telemetry dashboard..."
echo "Dashboard: http://127.0.0.1:${PORT}"
python helpers/server.py --telemetry-dir ./missing_yolo/telemetry_logs --recon-dir ./missing_yolo/outputs --host 127.0.0.1 --port "${PORT}" --train-cmd "./train_missing_yolo.sh"