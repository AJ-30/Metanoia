# Conditional Diffusion — floorplan latent -> arrangement latent

Trains a **conditional** DDPM (Ho et al. 2020) that generates an arrangement
(furniture) mask from a floorplan mask, operating purely in the latent spaces
of two **frozen, fully trained mask_VAEs**:

| Role | Checkpoint | Channels | Latent |
|---|---|---|---|
| Condition (floorplan) | `mask_vae_ideal-wombat_epoch_30.pt` | 18 | 256 |
| Target (arrangement) | `mask_vae_ace-squid_epoch_30.pt` | 11 | 256 |

Each sample folder provides BOTH `floorplan_segmentation.png` and
`furniture_segmentation.png` of the same layout, so every training pair
`(floorplan, arrangement)` is naturally aligned. Both masks are encoded once
with their frozen encoders and cached in RAM as `(c, x_0)` latent pairs; the
conditional denoiser (time + floorplan-aware MLP) learns `p(x_0 | c)`.
Sampling encodes a floorplan mask to `c`, runs the reverse chain from noise
conditioned on `c`, and decodes with the frozen arrangement decoder.

Conditioning is concatenation (`[x_t, c]` projection + per-block condition
injection). No cross-attention, no classifier-free guidance in v1 — KISS.

Every `GRID_INTERVAL` epochs the runner saves a single unified grid with
`3 + NUM_GEN_DRAWS` rows per sample column (`NUM_GRID_SAMPLES` columns):

1. full rendered image (`FULL_IMAGE_FILE`, alpha flattened onto `BACKGROUND_COLOR`),
2. floorplan ground truth (rooms/walls, dataset colors),
3. arrangement ground truth (furniture, dataset colors),
4. one overlay row per fresh noise draw (`NUM_GEN_DRAWS` draws of the
   predicted arrangement blended over the floorplan base with
   `OVERLAY_ALPHA`; furniture background stays transparent).

The GT rows judge fidelity; the draw rows show the distribution
`p(arrangement|floorplan)` rather than a single sample. The grid is saved as
`reconstruction_epoch_<N>_<run>.png`, so the generic dashboard
(`helpers/server.py`) shows it in its "Reconstructions" panel with no server
change (the "Generations" panel stays empty by design — there is only one
grid now).

## Files

- `model.py` — `ConditionalLatentDenoiser` (concat-condition residual MLP),
  `ConditionalLatentDiffusion` (`q_sample`, conditional `reverse_step`,
  `sample_conditional`) and the linear beta schedule builder.
- `runner.py` — `ConditionalDiffusionRunner`: frozen-VAE loading (rebuilt from
  each checkpoint's own hyperparameters), `PairedMaskDataset` (folders
  providing both masks), paired latent caching + standardization, conditional
  MSE+L1 training loop, per-epoch test evaluation, unified grids,
  checkpoints, telemetry.
- `run.py` — Entry point. Just calls `ConditionalDiffusionRunner().train()`.
- `config.toml` — Non-model config (device, split paths, the two frozen
  `*_CHECKPOINT_PATH`s, mask filenames, color index files).
- `hyperparameters.toml` — Denoiser architecture + diffusion schedule knobs.
- `README.md` — This file.

## Running

From the repo root (make sure the conda env `./venv` is active):

```bash
python conditional_diffusion/run.py --mode debug      # debug mode
python conditional_diffusion/run.py --mode prod       # prod mode
```

Or with the launcher script (auto-activates the conda env, archives previous
artifacts, then trains):

```bash
./train_conditional_diffusion.sh         # debug mode
./train_conditional_diffusion.sh prod    # prod mode
```

### Visualising training metrics and conditional samples

```bash
./run_scripts/start_conditional_diffusion_server.sh              # http://127.0.0.1:8082
./run_scripts/start_conditional_diffusion_server.sh 9000         # custom port
```

The "Reconstructions" panel shows the unified grid (full render / floorplan /
arrangement GT / one overlay row per draw) with an epoch dropdown and
auto-refresh. Telemetry curves: `train_loss` / `l1_loss` / `test_loss` /
`test_mask_bce` (mask-level weighted BCE of sampled arrangements vs GT masks,
both VAEs frozen, refreshed every `MASK_BCE_INTERVAL` epochs and attached to
the next telemetry row).

## Configuration

### `config.toml`

| Key | Meaning |
| --- | ------- |
| `DEVICE` | `mps`, `cuda` or `cpu` (falls back to CPU) |
| `TRAIN_IMAGES_DIR` | Split folder of samples to train on |
| `TEST_IMAGES_DIR` | Split folder of samples used for evaluation |
| `COND_VAE_CHECKPOINT_PATH` | Frozen floorplan mask VAE (condition encoder) |
| `TARGET_VAE_CHECKPOINT_PATH` | Frozen arrangement mask VAE (diffusion space + decoder) |
| `COND_MASK_FILE` / `TARGET_MASK_FILE` | Mask filenames inside each sample folder |
| `FULL_IMAGE_FILE` | Full render shown as the top reconstruction row |
| `MASK_COLOR_INDEX_DIR` | Directory holding the `*_color_index.json` files |
| `COND_COLOR_INDEX_FILE` / `TARGET_COLOR_INDEX_FILE` | Color index JSONs for rendering |
| `LATENT_CACHE_DIR` | Cached `(c, x_0)` latent pairs; reused when the fingerprint matches |

### `hyperparameters.toml`

| Key | Meaning |
| --- | ------- |
| `LATENT_DIM` | Arrangement latent size (must match target checkpoint) |
| `COND_DIM` | Floorplan latent size (must match condition checkpoint) |
| `HIDDEN_DIM` | Denoiser MLP width |
| `NUM_LAYERS` | Denoiser residual blocks |
| `DROPOUT` | Dropout inside the denoiser (training only) |
| `TIME_EMB_DIM` | Timestep embedding size |
| `TIME_MAX_PERIOD` | Longest sinusoid period |
| `NUM_TIMESTEPS` | DDPM forward steps |
| `BETA_START/BETA_END` | Linear noise schedule endpoints |
| `LEARNING_RATE` | Adam learning rate |
| `BATCH_SIZE` | Latent pairs per batch |
| `IMAGE_ENCODE_BATCH_SIZE` | Mask pairs per batch for the one-time encoder pass |
| `NUM_EPOCHS` | Training epochs |
| `SEED` | RNG seed (Python + torch) |
| `TELEMETRY_INTERVAL` | Epochs between telemetry rows |
| `CHECKPOINT_INTERVAL` | Epochs between checkpoint saves |
| `NUM_WORKERS` | Mask DataLoader workers (keep 0 on macOS) |
| `MAX_SAMPLES` | Cap on train samples (negative = all) |
| `IMAGE_SIZE` | Mask side length (must equal both frozen VAEs') |
| `BACKGROUND_COLOR` | RGBA backdrop for flattening the top-row render |
| `L1_WEIGHT` | Weight on L1 next to MSE in the noise-prediction loss |
| `MASK_BCE_POS_WEIGHT` | Positive weight for the periodic mask-level BCE eval |
| `MASK_BCE_INTERVAL` | Epochs between mask-level BCE evals |
| `GRID_INTERVAL` | Epochs between unified grid saves |
| `NUM_GRID_SAMPLES` | Fixed test pairs (columns) per unified grid |
| `NUM_GEN_DRAWS` | Noise draws (overlay rows) per floorplan in the unified grid |
| `TEST_LOSS_SAMPLES` | Test pairs for the per-epoch `test_loss` |
| `USE_MU_ONLY` | `true` = encoder mean; `false` = sampled z augmentation |
| `NORMALIZE_LATENTS` / `NORMALIZE_COND` | Standardize target / condition latents |
| `LATENT_STATS_SAMPLES` | Pairs used to estimate latent mean/std |
| `LATENT_STD_MIN` | Floor for latent std |
| `OVERLAY_ALPHA` | Furniture blend weight in the overlay row |

## Outputs

- Checkpoints: `conditional_diffusion/models/conditional_diffusion_<run>_epoch_<N>.pt`
  (denoiser + optimizer, both latent stats, both VAE checkpoint paths)
- Telemetry: `conditional_diffusion/telemetry_logs/<run>.csv`
- Unified grids: `conditional_diffusion/outputs/reconstruction_epoch_<N>_<run>.png`
- Latent cache: `conditional_diffusion/latent_cache/train_pairs.pt` + `test_pairs.pt`
  (frozen-encoder outputs + stats + fingerprint; any config, dataset or
  checkpoint change invalidates it automatically)
- Debug logs: `debug_logs/conditional_diffusion.log`

## Smoke test

Set `MAX_SAMPLES = 64`, `NUM_EPOCHS = 12`, `NUM_GRID_SAMPLES = 4`,
`NUM_GEN_DRAWS = 2`, `GRID_INTERVAL = 10`, `LATENT_STATS_SAMPLES = 64` and run
`--mode debug`. Encoding 64 pairs once takes seconds; each diffusion epoch
is MLP-only; the unified grid saves at epoch 10 and at the final epoch 12.
