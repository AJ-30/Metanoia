#!/usr/bin/env bash

# Benchmarks a trained YOLO detector against diffusion model outputs on the
# testing split (using the live redraw pipeline), alongside real ground-truth.
#
# Usage:
#   ./YOLO_validation_run.sh                     # debug mode (uses newest best.pt)
#   ./YOLO_validation_run.sh prod                # prod mode
#   ./YOLO_validation_run.sh debug --cap 50      # cap test pool to 50 samples
#   ./YOLO_validation_run.sh debug --weights YOLO/models/yolo_xxx_best.pt
#
# Arguments:
#   First optional argument: "debug" or "prod" (defaults to "debug").
#   All additional arguments are forwarded to YOLO/validate_diffusion.py.

set -e

# Resolve repository root so script can be invoked from any directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/YOLO/validate_diffusion.py" ]; then
    ROOT_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/validate_diffusion.py" ]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    ROOT_DIR="$SCRIPT_DIR"
fi
cd "$ROOT_DIR"

echo "Activating Conda environment..."
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    . "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate ./venv 2>/dev/null || conda activate base 2>/dev/null || true
fi

# Fallback to local virtualenv python if conda activation didn't set it
if [ -f "./venv/bin/python" ]; then
    PYTHON_BIN="./venv/bin/python"
else
    PYTHON_BIN="python"
fi

# Enable CPU fallback for MPS ops unsupported on macOS (e.g. torchvision::nms)
export PYTORCH_ENABLE_MPS_FALLBACK=1

MODE="debug"
if [ "$1" = "debug" ] || [ "$1" = "prod" ]; then
    MODE="$1"
    shift
fi

echo "Running YOLO diffusion test validation in ${MODE} mode..."
"$PYTHON_BIN" YOLO/validate_diffusion.py --mode "${MODE}" "$@"
