#!/usr/bin/env bash

# Runs the offline inference viewer (YOLO/predict.py) over a trained run and
# writes GT-vs-prediction overlays plus a contact sheet into
# YOLO/outputs/predict/. Read-only with respect to training state: it never
# rewrites the dataset or telemetry and never loads the frozen diffusion stack.
#
# Usage:
#   ./run_scripts/predict_YOLO.sh                       # debug, newest best.pt, val
#   ./run_scripts/predict_YOLO.sh prod
#   ./run_scripts/predict_YOLO.sh debug --split test --limit 20
#   ./run_scripts/predict_YOLO.sh debug --image /path/to/floorplan.png
#
# The first argument selects the run mode (debug|prod); everything after it is
# forwarded verbatim to predict.py, so `./run_scripts/predict_YOLO.sh debug
# --limit 20` works as written.

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

# Ultralytics runs NMS (torchvision::nms) during prediction, which has no MPS
# kernel and raises NotImplementedError without this flag. Training stays on
# MPS; only that op falls back to CPU (negligible for a handful of images).
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Optional leading mode argument (debug|prod); anything else is left for argparse.
MODE="debug"
if [ "$1" = "debug" ] || [ "$1" = "prod" ]; then
  MODE="$1"
  shift
fi

echo "Running YOLO inference viewer (${MODE})..."
python YOLO/predict.py --mode "${MODE}" "$@"