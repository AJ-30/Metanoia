#!/bin/bash
# Train the diffusion model on 2 A40 GPUs (DDP via torchrun). The a40 QoS allows
# at most 2 GPUs per job (a 4-GPU request stays pending with QOSMaxGRESPerJob).
# Global batch here is 2 x BATCH_SIZE_PER_GPU = 32 (DGX script: 64); a run can move
# between the scripts, but its effective batch size changes with it.
# Submit from the repository root:
#     mkdir -p Reconstruction/logs/slurm && sbatch Reconstruction/slurm/train_a40.sbatch <run_name>
# Resubmitting with the same run name resumes from the latest checkpoint.
# Change GPUS together with --gres (global batch = GPUS x BATCH_SIZE_PER_GPU).
#SBATCH --job-name=recon-train
#SBATCH --output=Reconstruction/logs/slurm/%x.%j.out
#SBATCH --error=Reconstruction/logs/slurm/%x.%j.err
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00

RUN="${1:?usage: sbatch train_a40.sbatch <run_name>}"
GPUS=2
# Stop cleanly (with a checkpoint) before the 96 h job limit above (a40 maximum: 4 days).
MAX_HOURS=95.5

source Reconstruction/slurm/env.sh
torchrun --standalone --nproc_per_node="${GPUS}" -m Reconstruction.model.train \
    --run-name "${RUN}" --mode prod --max-hours "${MAX_HOURS}"
