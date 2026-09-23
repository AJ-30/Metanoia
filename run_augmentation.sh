#!/usr/bin/env bash

# Builds the 15x frozen-VAE augmented YOLO dataset (no training involved).
# Output lands in the YOLO Augmented Dataset folder per YOLO/config.toml.
# Knobs live in YOLO/augment_hyperparameters.toml.
#
# NOTE: this script never archives/deletes the dataset (it is the product).
# A fresh build clears and regenerates the split dirs by design.
#
# Usage:
#   ./run_augmentation.sh         # debug mode
#   ./run_augmentation.sh prod    # prod mode

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

echo "Building augmented dataset in ${MODE} mode..."
python YOLO/run_augment.py --mode "${MODE}"
