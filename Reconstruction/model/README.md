# model: conditional diffusion for missing-component reconstruction

Milestone 4. A conditional DDPM generates **only the missing part** of a
floor plan, given the incomplete plan. The final plan is the input plus the
generated part (composition rules in `evaluation/labels.py`), so the existing
structure is preserved by construction.

## Method

| Part | Choice |
|---|---|
| Input | noisy missing part x_t (13 ch) + incomplete plan (13 ch) + x/y coordinates (2 ch); maps scaled to [-1, 1] |
| Network | ADM-style U-Net, 256x256, base 128, mults (1, 2, 2, 4, 4), 2 res-blocks/level, attention at 32 and 16 px, timestep scale/shift in every res-block |
| Target | v-prediction, cosine schedule, T = 1000 |
| Loss | v-MSE weighted x9 within 3 px of the missing part (targets are mostly empty) + 0.1 x cross-entropy of an auxiliary head predicting the missing type |
| Training | AdamW 1e-4, 2k warmup then cosine to 10%, bf16, grad clip 1, EMA 0.9999, 150k steps, global batch 64 (4 GPUs x 16) |
| Data | ResPlan train, a new random missing component per plan per epoch, random rotations/flips |
| Sampling | deterministic DDIM, 50 steps, EMA weights |

All values are in `hyperparameters.toml`; `--mode debug` swaps in a tiny
model and 200 steps (`[overrides.debug]`).

## Running on the cluster (from the repository root)

```bash
bash Reconstruction/setup_env.sh recon        # once: env with the CUDA 12.8 PyTorch build
mkdir -p Reconstruction/logs/slurm            # once: SLURM log folder

sbatch Reconstruction/slurm/smoke_test.sbatch             # ~15 min end-to-end check on 1 A40
sbatch Reconstruction/slurm/train_dgx.sbatch v1           # full training, 4 DGX GPUs, 72 h
sbatch Reconstruction/slurm/train_dgx.sbatch v1           # resubmit = resume from latest checkpoint
sbatch Reconstruction/slurm/predict_dgx.sbatch v1 latest resplan test 1   # predict + evaluate (DGX)
sbatch Reconstruction/slurm/predict_a40.sbatch v1 60000                   # same on 2 A40s, checkpoint at step 60000
sbatch Reconstruction/slurm/predict_dgx.sbatch v1 latest cubicasa test 1  # cross-dataset test
```

The cluster driver supports CUDA 12.8; a PyTorch built for CUDA 13 cannot
use the GPUs. `slurm/env.sh` and `train.py` refuse to run on CPU.

## Watching a run: `Reconstruction/runs/<run>/`

| File | Content |
|---|---|
| `train_log.csv` | every 50 steps: loss, diffusion loss, type loss/accuracy, lr, grad norm, samples/s, GPU memory, hours |
| `val_log.csv` | every 5k steps, EMA model on 256 fixed val cases: F1@0.5, primary detection (overall and per type), IoU, type/connection accuracy, validity checks |
| `val_grids/val_step_*.png` | Input / Prediction / Composed / Ground truth for 8 fixed val cases |
| `checkpoints/` | last 3 step checkpoints + `latest.pt` |
| `run_info.json` | hyperparameters, git commit, world size, one entry per (re)start |
| `predictions/<dataset>_<split>/step_<step>_k<K>/` | `<case_id>.npz` from `predict.py` (thresholded, uint8 {0, 255}) |

Choose the checkpoint by `val_log.csv` (primary detection / F1), not by loss.

## Files

| File | Purpose |
|---|---|
| `unet.py` | Conditional U-Net + type head |
| `diffusion.py` | Cosine schedule, v-parameterization, DDIM sampler |
| `objective.py` | Foreground-weighted loss + type loss |
| `sampling.py` | Batch sampling of missing parts (validation and prediction) |
| `checkpoint.py` | EMA, checkpoint save / resume |
| `validation.py` | Fixed val subset scored with the `evaluation/` metrics |
| `train.py` | Training CLI (single GPU or DDP via torchrun) |
| `predict.py` | Writes predictions for `evaluation/evaluate.py` |
| `config.toml` / `hyperparameters.toml` | Paths / all hyperparameters |

Structural constraints (losses and sampling guidance) are the next milestone
and plug into `objective.py` and `sampling.py`.
