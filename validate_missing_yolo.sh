#!/usr/bin/env bash

# Standalone validation for the three missing-component detectors.
#
# One run = one report folder per component. Each component's val half is
# regenerated from its frozen mask_diffusion checkpoint, then the matching
# missing_yolo weights are run over every regenerated image. The output is:
#
#   missing_yolo/validation_outputs/<component>/
#       metrics.json / metrics.txt / per_image.csv / dataset / samples /
#
# Usage:
#   ./validate_missing_yolo.sh                         # debug, all components
#   ./validate_missing_yolo.sh debug tv                # one component
#   ./validate_missing_yolo.sh debug tv --cap 24       # smoke run
#   ./validate_missing_yolo.sh prod all                # full production run
#
# Any arguments after the mode/component pair are forwarded to
# missing_yolo/validate.py.

set -e

# Resolve the repo root so the script works from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use the project's virtualenv directly. This keeps the script runnable from a
# non-interactive shell (background jobs, CI) where `conda activate` is not on
# PATH; set PYTHON to override the interpreter explicitly.
if [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

MODE="${1:-debug}"
COMPONENT="${2:-all}"
shift 2 2>/dev/null || true

# Ultralytics NMS (torchvision::nms) has no MPS kernel and crashes with
# NotImplementedError. Keep the heavy work on MPS and fall back to CPU only for
# that op.
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "Archiving previous missing_yolo validation artifacts..."
mkdir -p missing_yolo/old_validation_outputs
if [ -d missing_yolo/validation_outputs ] && [ -n "$(ls -A missing_yolo/validation_outputs 2>/dev/null)" ]; then
  # One timestamped archive per run, so repeated runs never merge or overwrite
  # an older component report.
  ARCHIVE_DIR="missing_yolo/old_validation_outputs/run_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$ARCHIVE_DIR"
  mv missing_yolo/validation_outputs/* "$ARCHIVE_DIR"/
fi

echo "Running missing_yolo validation in ${MODE} mode for '${COMPONENT}'..."
"$PYTHON" missing_yolo/validate.py --mode "${MODE}" --component "${COMPONENT}" "$@"
