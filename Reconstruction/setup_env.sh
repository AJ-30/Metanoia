#!/bin/bash
# Install the Reconstruction package dependencies into a conda environment.
#
# Usage:  bash Reconstruction/setup_env.sh [env_name]
#   env_name defaults to "recon". The environment is created (Python 3.10)
#   if it does not exist; existing packages are only added to, never removed.
set -eo pipefail  # no -u: conda activation scripts read unset variables

ENV_NAME="${1:-recon}"
# PyTorch build matching the cluster's NVIDIA driver (CUDA 12.8). Newer
# default builds (CUDA 13) silently fall back to CPU on these GPUs.
# Override for other machines, e.g. TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
# for Blackwell GPUs such as the GB10 (DGX Spark).
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make `conda activate` available in this non-interactive shell.
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Creating conda env ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"
python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
echo "Environment ${ENV_NAME} ready. Run commands from the repository root."
