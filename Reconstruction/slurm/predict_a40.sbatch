#!/bin/bash
# Predict the frozen cases of a split with a checkpoint of a trained run, then
# evaluate, on 2 A40 GPUs (the a40 QoS maximum per job).
# Submit from the repository root:
#     sbatch Reconstruction/slurm/predict_a40.sbatch <run_name> [checkpoint=latest] [dataset=resplan] [split=test] [samples=1]
#     e.g. sbatch Reconstruction/slurm/predict_a40.sbatch run2-a40 60000
# Restarting with the same arguments skips cases that already have prediction files.
# Outputs: Reconstruction/runs/<run>/predictions/<dataset>_<split>/step_<step>_k<K>/ and
#          Reconstruction/outputs/evaluation/<dataset>/<split>/<run>_step<step>_k<K>/
#SBATCH --job-name=recon-predict
#SBATCH --output=Reconstruction/logs/slurm/%x.%j.out
#SBATCH --error=Reconstruction/logs/slurm/%x.%j.err
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

GPUS=2
source Reconstruction/slurm/predict_eval.sh "${GPUS}" "$@"
