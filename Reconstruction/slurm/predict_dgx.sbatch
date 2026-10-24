#!/bin/bash
# Predict the frozen cases of a split with a checkpoint of a trained run, then evaluate.
# Submit from the repository root:
#     sbatch Reconstruction/slurm/predict_dgx.sbatch <run_name> [checkpoint=latest] [dataset=resplan] [split=test] [samples=1]
# Restarting with the same arguments skips cases that already have prediction files.
#SBATCH --job-name=recon-predict
#SBATCH --output=Reconstruction/logs/slurm/%x.%j.out
#SBATCH --error=Reconstruction/logs/slurm/%x.%j.err
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00

GPUS=4
source Reconstruction/slurm/predict_eval.sh "${GPUS}" "$@"
