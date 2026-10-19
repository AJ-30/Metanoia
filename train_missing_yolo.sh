#!/usr/bin/env bash

# Trains the missing-component detector for ONE component, on images produced by
# the frozen mask_diffusion generator, with the dataset's own labels.txt boxes as
# the only detection target (see missing_yolo/README.md).
#
# One run = one component. The component comes from MISSING_COMPONENT in
# missing_yolo/hyperparameters.toml unless the second/third argument overrides
# it, so the three runs are launched as:
#
#   ./train_missing_yolo.sh debug                # MISSING_COMPONENT from TOML
#   ./train_missing_yolo.sh prod   showcase      # forces the showcase run
#   ./train_missing_yolo.sh prod   dining_set
#
# Each run writes its own dataset tree (<LIVE_DATASET_BASE_ROOT>/<component>), so
# the three runs never clobber each other. No prebuilt dataset is needed: the
# trainer indexes the mask dataset, encodes the generator conditions once,
# renders val/test once and redraws every train image from a fresh generator
# draw at the end of each epoch.
#
# Per-epoch wall clock on M1 Pro with the full pool (~69.8k train, plus ~9.9k
# val and ~9.9k test carved from the validate split): the mask-diffusion reverse
# chain is 1000 steps per sample, so expect roughly an hour of generation per
# epoch. Use the sample caps in missing_yolo/hyperparameters.toml to smoke-test
# the pipeline first.

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
COMPONENT="${2:-}"

# Ultralytics runs NMS (torchvision::nms) during validation, which has no MPS
# kernel and crashes with NotImplementedError. This flag keeps training on MPS
# and falls back to CPU for that op only (negligible cost vs full-CPU training).
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "Archiving previous missing_yolo artifacts..."
mkdir -p missing_yolo/old_telemetry_logs missing_yolo/old_outputs
mv missing_yolo/telemetry_logs/* missing_yolo/old_telemetry_logs/ 2>/dev/null || true
mv missing_yolo/outputs/* missing_yolo/old_outputs/ 2>/dev/null || true

echo "Running missing_yolo training in ${MODE} mode${COMPONENT:+ for component '${COMPONENT}'}..."
if [ -n "$COMPONENT" ]; then
  python missing_yolo/run.py --mode "${MODE}" --component "${COMPONENT}"
else
  python missing_yolo/run.py --mode "${MODE}"
fi