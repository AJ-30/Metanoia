# VAE — Variational Autoencoder on Floor-Plan Images

Trains a convolutional VAE (PyTorch) on the images in
`datasets/work/{train,test}`.

Training uses **all** samples in the `train/` folder. After **every epoch**
the runner:

- reconstructs `NUM_RECON_SAMPLES` fixed test samples and saves a grid image;
- generates `NUM_GEN_SAMPLES` novel floor plans from the latent space
  (random `N(0, I)` latents decoded straight through the decoder);
- evaluates the model on `TEST_LOSS_SAMPLES` test images and logs `test_loss`.

At the end of training the model is evaluated once on the **full test set**
and the final `test_loss` is logged.

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

The dataset stores every render side by side, so which one the VAE learns is a
single TOML switch, not a code change:

```toml
TRAIN_IMAGE_VARIANT = "full_png"    # or "layout_only"

[default.IMAGE_VARIANTS]
full_png    = "full.png"
layout_only = "floorplan_only.png"
```

`TRAIN_IMAGE_VARIANT` picks an entry of the `[default.IMAGE_VARIANTS]` table, so
no render filename is hardcoded in Python. A sample folder that does not
provide the selected render is skipped, and an empty split raises
`FileNotFoundError` rather than failing later inside the training loop.

The VAE is unsupervised, so `labels.txt` is never read here — it is only used by
the YOLO pipeline. `no_tv.png` is never read: it omits the tv pixels while
`labels.txt` still lists the tv box, so the two disagree.

## Files

- `model.py` — Convolutional VAE (Encoder / Decoder) and the VAE loss function.
- `runner.py` — `FloorPlanDataset` (shared with Diffusion), `VAERunner`:
  dataset building, training loop, per-epoch test evaluation, checkpoints,
  telemetry, reconstruction and generation grids.
- `run.py` — Entry point. Just calls `VAERunner().train()`.
- `config.toml` — Non-model configuration (device, paths, logging, telemetry).
- `hyperparameters.toml` — Model architecture and training hyperparameters.
- `README.md` — This file.

## Running

From the repo root (make sure the conda env `./venv` is active):

```bash
python VAE/run.py --mode debug      # debug mode
python VAE/run.py --mode prod       # prod mode
```

Or with the launcher script (auto-activates the conda env, archives previous
artifacts, then trains):

```bash
./train_VAE.sh         # debug mode
./train_VAE.sh prod    # prod mode
```

### Visualising training metrics and reconstructions

While training runs, telemetry rows (including the per-epoch `test_loss`) are
written to `VAE/telemetry_logs/<run_id>.csv` and reconstruction + generation
grids are saved to `VAE/outputs/` after every epoch. Start the dashboard server
to see the loss curves and the images:

```bash
./run_scripts/start_server.sh              # http://127.0.0.1:8080
./run_scripts/start_server.sh 9000         # custom port
```

The dashboard (Flask + Chart.js in `helpers/server.py`) lists every run from
`VAE/telemetry_logs`, plots epoch vs train/test/recon/KL loss, and shows a
"Reconstructions" panel (grid of the fixed test samples, top row originals /
bottom row reconstructions) plus a "Generations" panel (novel images sampled
from the latent space). Both have an epoch dropdown to flip through every
epoch's grid, and the dashboard auto-refreshes so the newest images and the
`test_loss` curve appear at the end of each epoch.

## Configuration

### `config.toml`

| Key                      | Meaning                                          |
| ------------------------ | ------------------------------------------------ |
| `DEVICE`                 | `mps`, `cuda` or `cpu` (falls back to CPU)       |
| `TRAIN_IMAGES_DIR`       | Split folder of samples to train on             |
| `TEST_IMAGES_DIR`        | Split folder of samples used for evaluation     |
| `DEBUG_LOG_FILE_PATH`    | Log file written by the shared logger            |
| `MODEL_SAVE_PATH`        | Where `.pt` checkpoints are saved                |
| `TELEMETRY_LOG_DIRECTORY`| Where the run's `*.csv` telemetry goes           |
| `OUTPUT_SAMPLES_DIRECTORY`| Where reconstruction grid images are saved      |

`[debug]` / `[prod]` sections follow the same convention as the root `config.toml`.

### `hyperparameters.toml`

| Key                  | Meaning                                              |
| -------------------- | ---------------------------------------------------- |
| `IMAGE_SIZE`         | Side length of the resized square images (must be ÷ 16) |
| `OUT_CHANNELS`       | 3 for RGB images                                     |
| `LATENT_DIM`         | Dimensionality of the latent space                   |
| `BASE_CHANNELS`      | Channel count of the deepest conv layer              |
| `DROPOUT`            | Dropout probability (encoder + decoder, training only)|
| `LEARNING_RATE`      | Adam learning rate                                   |
| `BATCH_SIZE`         | Images per batch                                     |
| `NUM_EPOCHS`         | Number of training epochs                            |
| `TELEMETRY_INTERVAL` | Epochs between telemetry CSV rows (1 = every epoch)  |
| `CHECKPOINT_INTERVAL`| Epochs between checkpoint saves                     |
| `NUM_WORKERS`        | DataLoader workers                                   |
| `MAX_SAMPLES`        | Cap on samples used (negative = use all)             |
| `PRELOAD_IMAGES`     | Cache decoded images in RAM (recommended: `true`)    |
| `BACKGROUND_COLOR`   | RGBA backdrop used to flatten the alpha channel      |
| `NUM_RECON_SAMPLES`  | Fixed test images shown per reconstruction grid      |
| `NUM_GEN_SAMPLES`    | Novel images generated per generation grid            |
| `TEST_LOSS_SAMPLES`  | Test images used for the per-epoch `test_loss`        |
| `KL_WEIGHT`          | Beta-VAE weight on the KL-divergence term             |
| `L1_WEIGHT`          | Weight on L1 next to MSE in the reconstruction term   |
| `TRAIN_IMAGE_VARIANT`| Which render to learn: `full_png` or `layout_only`    |
| `IMAGE_VARIANTS`     | Variant → filename table (`[default.IMAGE_VARIANTS]`) |

`Diffusion/hyperparameters.toml` carries a copy of `TRAIN_IMAGE_VARIANT` and
`IMAGE_VARIANTS`; keep them in sync.

## Outputs

- Model checkpoints: `VAE/models/vae_<run_id>_epoch_<N>.pt`
- Telemetry CSV: `VAE/telemetry_logs/<run_id>.csv`
- Reconstruction grids: `VAE/outputs/reconstruction_epoch_<N>_<run_id>.png`
- Generation grids: `VAE/outputs/generation_epoch_<N>_<run_id>.png`
- Debug logs: `debug_logs/vae.log`

## New smoke-test / full run

For a fast sanity check set `MAX_SAMPLES = 64`, `NUM_EPOCHS = 1`,
`PRELOAD_IMAGES = false`, `NUM_WORKERS = 0` (macOS pickling) and run with
`--mode debug`.

On first launch with `PRELOAD_IMAGES = true` every train sample is decoded into
RAM once. Each cached image costs `3 x IMAGE_SIZE x IMAGE_SIZE` bytes (uint8), so
`IMAGE_SIZE = 256` is ~0.2 MB per sample — about 2.9 GB for 15k samples, scaling
linearly with the split. Turn `PRELOAD_IMAGES` off when the split outgrows the
machine's RAM.