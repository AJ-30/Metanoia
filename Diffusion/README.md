# Diffusion — DDPM on Frozen VAE Latents

Trains a denoising diffusion probabilistic model (DDPM, Ho et al. 2020) on the
latent vectors of a **frozen, fully trained VAE**. The VAE encoder compresses
each floor-plan image once to a flat `LATENT_DIM` vector; the diffusion
denoiser (time-conditioned MLP) learns that latent distribution. Sampling runs
the reverse chain from pure noise and decodes with the frozen VAE decoder.

Training uses **all** samples in the `train/` split (encoded once and cached
in RAM as latents, ~15 MB for 15k x 256). After **every epoch** the runner:

- evaluates the composite noise-prediction loss (MSE + `L1_WEIGHT` * L1) on
  `TEST_LOSS_SAMPLES` test latents;
- samples `NUM_GEN_SAMPLES` novel latents from noise and decodes them to a
  generation grid.

No reconstruction grids are produced: the diffusion dashboard shows
generations only (its "Reconstructions" panel stays empty by design).

At the end of training the model is evaluated once on **all** test latents.

## Source dataset layout

The dataset is **folder-per-sample**: a split folder holds one sub-folder per
layout, and each sub-folder carries every render of that layout.

```
<split>/<hash>_original/            # or <hash>_augmented_<NNN>
    full.png                        # 854 x H, RGBA — all furniture + floorplan
    floorplan_only.png              # 854 x H, RGBA — rooms only, no furniture
    no_tv.png                       # all furniture except the tv
    ground_truth.svg                # reloadable layout with embedded metadata
    labels.txt                      # "<Class_Name> cx cy w h", normalized
```

The dataset stores every render side by side, so which one the diffusion model
learns is a single TOML switch in `Diffusion/config.toml`, not a code change:

```toml
TRAIN_IMAGE_VARIANT = "full_png"    # or "floorplan_only"

[default.IMAGE_VARIANTS]
full_png    = "full.png"
floorplan_only = "floorplan_only.png"
```

Diffusion is trained on the **full render** (`full.png`). `FloorPlanDataset` is
imported from `VAE.runner` — single source of truth for image preprocessing —
and only `image_size` differs: Diffusion takes it from the frozen VAE checkpoint
while the VAE takes it from its own TOML. `VAE/config.toml` and `YOLO/config.toml`
each carry a copy of the variant table; keep them in sync.

The model is unconditional: it reads no labels, masks or room types. The only
conditioning is the DDPM timestep. `labels.txt` is used solely by the YOLO
pipeline, and `no_tv.png` is never read (it omits the tv pixels while
`labels.txt` still lists the tv box).

## Files

- `model.py` — `LatentDenoiser` (time-conditioned residual MLP),
  `LatentDiffusion` (DDPM schedule buffers, `q_sample`, `reverse_step`,
  `sample`) and the linear beta schedule builder.
- `runner.py` — `DiffusionRunner`: frozen-VAE loading (rebuilt from the
  checkpoint's own hyperparameters, with `NUM_STAGES` inference and
  `LATENT_ADDITIONAL_NOISE` back-fill for older checkpoints), latent caching +
  standardization, composite MSE+L1 training loop, per-epoch test evaluation,
  checkpoints, telemetry, generation grids.
- `run.py` — Entry point. Just calls `DiffusionRunner().train()`.
- `config.toml` — Non-model config (device, split paths, frozen `VAE_CHECKPOINT_PATH`,
  `TRAIN_IMAGE_VARIANT` + `IMAGE_VARIANTS`).
- `hyperparameters.toml` — Denoiser architecture + diffusion schedule knobs.
- `README.md` — This file.

## Running

From the repo root (make sure the conda env `./venv` is active):

```bash
python Diffusion/run.py --mode debug      # debug mode
python Diffusion/run.py --mode prod       # prod mode
```

Or with the launcher script (auto-activates the conda env, archives previous
diffusion artifacts, then trains):

```bash
./train_Diffusion.sh         # debug mode
./train_Diffusion.sh prod    # prod mode
```

### Visualising training metrics and samples (new server + new telemetry)

Diffusion telemetry is fully isolated from the VAE run: CSVs go to
`Diffusion/telemetry_logs/<run_id>.csv` and grids to `Diffusion/outputs/`.
Start the **separate** diffusion dashboard (default port **8081**, so it can
run next to the VAE dashboard on 8080):

```bash
./run_scripts/start_diffusion_server.sh              # http://127.0.0.1:8081
./run_scripts/start_diffusion_server.sh 9000         # custom port
```

It reuses the generic Flask + Chart.js dashboard in `helpers/server.py` (no
duplicated server code): loss curves for `train_loss` / `l1_loss` /
`test_loss`, and a "Generations" panel (`generation_epoch_*.png` with an
epoch dropdown), auto-refreshing every 3 s while training runs.

## Configuration

### `config.toml`

| Key                        | Meaning                                          |
| -------------------------- | ------------------------------------------------ |
| `DEVICE`                   | `mps`, `cuda` or `cpu` (falls back to CPU)       |
| `TRAIN_IMAGES_DIR`         | Split folder of samples to train on             |
| `TEST_IMAGES_DIR`          | Split folder of samples used for evaluation     |
| `VAE_CHECKPOINT_PATH`      | Frozen VAE checkpoint supplying latents + decoder|
| `TRAIN_IMAGE_VARIANT`      | Which render to learn: `full_png` or `floorplan_only`|
| `IMAGE_VARIANTS`           | Variant → filename table (`[default.IMAGE_VARIANTS]`)|
| `DEBUG_LOG_FILE_PATH`      | Log file written by the shared logger            |
| `MODEL_SAVE_PATH`          | Where `.pt` checkpoints are saved                |
| `TELEMETRY_LOG_DIRECTORY`  | Where the run's `*.csv` telemetry goes           |
| `OUTPUT_SAMPLES_DIRECTORY` | Where per-epoch grid images are saved            |

`[debug]` / `[prod]` sections follow the same convention as `config.toml`.

> **Frozen VAE checkpoint.** `LATENT_DIM` in `Diffusion/hyperparameters.toml`
> must equal the latent dim stored inside `VAE_CHECKPOINT_PATH`. The runner
> fails fast on mismatch.
>
> The `rare-roughy` run (512-dim latents, 5-stage arch) is **not loadable by the
> current `VAE/model.py`** — it was trained with 4×4 stride-2 conv kernels and an
> extra `3x3` stage, while `model.py` now builds 3×3 kernels and 4 stages, so
> `load_state_dict` rejects it on both key names and shapes. The
> `LATENT_ADDITIONAL_NOISE` back-fill in `resolve_latent_additional_noise()`
> fixes the *missing key* half of that, but not the architecture half.
>
> `next-collie` (256-dim, 3×3, 4 stages) **is** loadable and is what the YOLO
> pipeline freezes, so to train diffusion today point `VAE_CHECKPOINT_PATH` at
> e.g. `vae_next-collie_epoch_500.pt` and set `LATENT_DIM = 256`.

### `hyperparameters.toml`

| Key                   | Meaning                                                             |
| --------------------- | ------------------------------------------------------------------- |
| `LATENT_DIM`          | VAE latent size (must match the frozen checkpoint)                  |
| `HIDDEN_DIM`          | Denoiser MLP width                                                  |
| `NUM_LAYERS`          | Denoiser residual blocks                                            |
| `TIME_EMB_DIM`        | Timestep embedding size                                             |
| `TIME_MAX_PERIOD`     | Longest sinusoid period of the time embedding                       |
| `NUM_TIMESTEPS`       | DDPM forward steps                                                  |
| `BETA_START/BETA_END` | Linear noise schedule endpoints                                     |
| `LEARNING_RATE`       | Adam learning rate                                                  |
| `BATCH_SIZE`          | Latents per batch                                                   |
| `NUM_EPOCHS`          | Training epochs                                                     |
| `SEED`                | RNG seed (Python + torch) for reproducibility                       |
| `TELEMETRY_INTERVAL`  | Epochs between telemetry CSV rows (1 = every epoch)                 |
| `CHECKPOINT_INTERVAL` | Epochs between checkpoint saves                                     |
| `NUM_WORKERS`         | Image DataLoader workers                                            |
| `MAX_SAMPLES`         | Cap on train samples used (negative = use all)                       |
| `PRELOAD_IMAGES`      | Cache decoded images in RAM for the one-time encoding pass          |
| `BACKGROUND_COLOR`    | RGBA backdrop used to flatten the alpha channel                     |
| `NUM_GEN_SAMPLES`     | Novel latents sampled per generation grid                           |
| `GRID_NROW`           | Images per row in the generation grid                               |
| `TEST_LOSS_SAMPLES`   | Test latents used for the per-epoch `test_loss`                     |
| `L1_WEIGHT`           | Weight on L1 next to MSE in the noise-prediction loss               |
| `USE_MU_ONLY`         | `true` = train on encoder mean; `false` = sampled z augmentation    |
| `NORMALIZE_LATENTS`   | Standardize latents with dataset mean/std (recommended)             |
| `LATENT_STATS_SAMPLES`| Images used to estimate the latent mean/std                         |
| `LATENT_STD_MIN`      | Floor for the latent std (avoids div-by-zero)                       |

## Outputs

- Model checkpoints: `Diffusion/models/diffusion_<run_id>_epoch_<N>.pt`
  (includes denoiser + optimizer state, latent mean/std, VAE checkpoint path)
- Telemetry CSV: `Diffusion/telemetry_logs/<run_id>.csv`
- Generation grids: `Diffusion/outputs/generation_epoch_<N>_<run_id>.png`
- Debug logs: `debug_logs/diffusion.log`

## New smoke-test / full run

For a fast sanity check set `MAX_SAMPLES = 64`, `NUM_EPOCHS = 1`,
`NUM_GEN_SAMPLES = 4`, `LATENT_STATS_SAMPLES = 64` and run with
`--mode debug`. Encoding 64 images once takes seconds; each diffusion epoch
is MLP-only (milliseconds on MPS).
