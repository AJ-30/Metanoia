#!/bin/bash
# Train the diffusion model on a DGX node with 4 GPUs (DDP via torchrun).
# Submit from the repository root:
#     mkdir -p Reconstruction/logs/slurm && sbatch Reconstruction/slurm/train_dgx.sbatch <run_name>
# Resubmitting with the same run name resumes from the latest checkpoint.
# Change GPUS together with --gres (global batch = GPUS x BATCH_SIZE_PER_GPU).
#SBATCH --job-name=recon-train
#SBATCH --output=Reconstruction/logs/slurm/%x.%j.out
#SBATCH --error=Reconstruction/logs/slurm/%x.%j.err
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00

RUN="${1:?usage: sbatch train_dgx.sbatch <run_name>}"
GPUS=4
# Stop cleanly (with a checkpoint) before the 72 h job limit above.
MAX_HOURS=71.5

source Reconstruction/slurm/env.sh
torchrun --standalone --nproc_per_node="${GPUS}" -m Reconstruction.model.train \
    --run-name "${RUN}" --mode prod --max-hours "${MAX_HOURS}"
