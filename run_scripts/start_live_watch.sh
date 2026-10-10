#!/usr/bin/env bash

# Publishes live YOLO training snapshots for the dashboard's Live panel.
# Read-only sidecar: it only READS the live dataset (overlays, labels,
# manifest) and the trainer's best.pt (via a temp copy), and only WRITES
# snapshot PNGs into YOLO/outputs/. Safe to start/stop anytime — the
# trainer in ./train_YOLO.sh is never disturbed.
#
# Start order while training runs:
#   ./run_scripts/start_yolo_server.sh   # dashboard (Live panel)
#   ./run_scripts/start_live_watch.sh    # snapshots (this script)
#
# Usage:
#   ./run_scripts/start_live_watch.sh [mode]   # mode: debug (default) or prod

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

MODE="${1:-debug}"

# Same MPS-fallback flag as train_YOLO.sh: sidecar inference runs NMS
# (torchvision::nms) with no MPS kernel; only that op falls back to CPU.
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "Publishing live snapshots (mode=${MODE}, Ctrl-C to stop)..."
python YOLO/live_watch.py --mode "${MODE}"
