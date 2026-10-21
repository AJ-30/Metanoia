#!/usr/bin/env bash

# Evaluates YOLO on the missing component that mask_diffusion generates.
# Archives artifacts from any previous run, then launches evaluator/run.py with
# the given mode (and forwards any extra flags, e.g. --cap -1 for the full
# test split).
#
# Nothing is trained here: the run selects one missing component, which pairs a
# mask_diffusion checkpoint with the matching missing_yolo detector through
# components.toml, and writes only under evaluator/outputs and
# evaluator/telemetry_logs.
#
# Usage:
#   ./run_eval.sh                          # default component, smoke pass
#   ./run_eval.sh prod --component showcase
#   ./run_eval.sh debug --cap -1           # whole test split, default component
#   ./run_eval.sh debug --weights path/to/weights.pt

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
shift || true

echo "Archiving previous evaluator artifacts..."
mkdir -p evaluator/old_telemetry_logs evaluator/old_outputs
mv evaluator/telemetry_logs/* evaluator/old_telemetry_logs/ 2>/dev/null || true
mv evaluator/outputs/* evaluator/old_outputs/ 2>/dev/null || true

echo "Running missing-component evaluation in ${MODE} mode ${*}..."
python evaluator/run.py --mode "${MODE}" "$@"
