#!/usr/bin/env bash

# Finetunes YOLO26 on the prebuilt frozen-VAE augmented floor-plan dataset.
# Archives artifacts from any previous run, then launches YOLO/run.py.
# This script never touches the dataset (build it first with
# ./run_augmentation.sh); the trainer fails fast if it is missing/incomplete.
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
