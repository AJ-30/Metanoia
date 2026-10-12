#!/usr/bin/env bash

# Finetunes YOLO26 on live conditional-diffusion overlay floor plans.
# Archives artifacts from any previous run, then launches YOLO/run.py.
# No prebuilt dataset is needed: the trainer indexes the live mask dataset,
# uses every available sample, redraws every train image from a fresh
# diffusion draw at the end of each epoch, derives each image's boxes from the
# arrangement that was actually drawn, and publishes GT-vs-prediction overlays
# per epoch (see YOLO/dataset.py and YOLO/README.md).
#
# Per-epoch wall clock on M1 Pro with the full pool (~69.8k train / ~19.9k val
# / ~10k test): ~15-20 min redraw + ~5-8 min train + ~15-25 min validation.
#
# Usage:
#   ./train_YOLO.sh          # debug mode
#   ./train_YOLO.sh prod     # prod mode

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

# Ultralytics runs NMS (torchvision::nms) during validation, which has no MPS
# kernel and crashes with NotImplementedError. This flag keeps training on MPS
# and falls back to CPU for that op only (negligible cost vs full-CPU training).
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "Archiving previous YOLO artifacts..."
mkdir -p YOLO/old_telemetry_logs YOLO/old_outputs
mv YOLO/telemetry_logs/* YOLO/old_telemetry_logs/ 2>/dev/null || true
mv YOLO/outputs/* YOLO/old_outputs/ 2>/dev/null || true

echo "Running YOLO training in ${MODE} mode..."
python YOLO/run.py --mode "${MODE}"
