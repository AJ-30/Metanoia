#!/bin/bash

#SBATCH --job-name=PGD_1000-OOD_scoring
#SBATCH --output=logs/slurm/log.%x.job_%j.out
#SBATCH --error=logs/slurm/log.%x.job_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=a40
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --qos=a40
#SBATCH --cpus-per-task=32


source /home/micml/30007046/miniconda3/etc/profile.d/conda.sh
conda activate seg

test_at_epoch="500" #This is for AE, not seg model
id_cat="0"
id_dataset="New-Retina-Config1"
pgd_steps=1000

# OOD scoring

python our-ood/scripts/test_our_dstransUnet_ood_scores.py \
    --id_csv baselines/Our_method/

