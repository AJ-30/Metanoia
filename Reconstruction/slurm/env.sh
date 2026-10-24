#!/bin/bash
# Shared environment for the SLURM scripts in this folder. Edit here to run
# with a different conda installation / environment.
#
# Must be sourced from the repository root (sbatch from the repo root).
set -eo pipefail  # no -u: conda activation scripts read unset variables

CONDA_BASE="/home/micml/26d0361/miniconda3"
# "recon" has a PyTorch build for CUDA 12.8, matching the cluster driver (12.8);
# the cu130 build in "metanoia" silently falls back to CPU on these nodes.
CONDA_ENV="recon"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

if [ ! -d "Reconstruction" ]; then
    echo "Submit from the repository root (the folder containing Reconstruction/)." >&2
    exit 1
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "PyTorch cannot use the GPU (driver/CUDA build mismatch?). Refusing to run on CPU." >&2
    exit 1
fi
