# VAE — Variational Autoencoder on Floor-Plan Images

Trains a convolutional VAE (PyTorch) on the images in
`FurnitureArrangerDataset-main/super_augmentation_2/dataset/{train,test}`.

Training uses **all** images in the `train/` folder. After **every epoch**
the runner:

- reconstructs `NUM_RECON_SAMPLES` fixed test samples and saves a grid image;
- evaluates the model on `TEST_LOSS_SAMPLES` test images and logs `test_loss`.

At the end of training the model is evaluated once on the **full test set**
and the final `test_loss` is logged.

## Files

- `model.py` — Convolutional VAE (Encoder / Decoder) and the VAE loss function.
- `runner.py` — `VAERunner` class: dataset building, training loop, per-epoch
  test evaluation, checkpoints, telemetry and reconstruction sample grids.
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
written to `VAE/telemetry_logs/<run_id>.csv` and a reconstruction grid is saved
to `VAE/outputs/` after every epoch. Start the dashboard server to see the loss
curves and the reconstructions:

```bash
./run_scripts/start_server.sh              # http://127.0.0.1:8080
./run_scripts/start_server.sh 9000         # custom port
```

The dashboard (Flask + Chart.js in `helpers/server.py`) lists every run from
`VAE/telemetry_logs`, plots epoch vs train/test/recon/KL loss, and shows a
"Reconstructions" panel with a dropdown to flip through every epoch's grid of
the fixed test samples (the dashboard auto-refreshes, so the newest grid and
test-loss curve appear at the end of each epoch).

## Configuration

### `config.toml`

| Key                      | Meaning                                          |
| ------------------------ | ------------------------------------------------ |
| `DEVICE`                 | `mps`, `cuda` or `cpu` (falls back to CPU)       |
| `TRAIN_IMAGES_DIR`       | Folder of images to train on                     |
| `TEST_IMAGES_DIR`        | Folder of images used for evaluation             |
| `DEBUG_LOG_FILE_PATH`    | Log file written by the shared logger            |
| `MODEL_SAVE_PATH`        | Where `.pt` checkpoints are saved                |
| `TELEMETRY_LOG_DIRECTORY`| Where the run's `*.csv` telemetry goes           |
| `OUTPUT_SAMPLES_DIRECTORY`| Where reconstruction grid images are saved      |
| `TELEMETRY_INTERVAL`     | Epochs between checkpoint saves                  |

`[debug]` / `[prod]` sections follow the same convention as the root `config.toml`.

### `hyperparameters.toml`

| Key                  | Meaning                                              |
| -------------------- | ---------------------------------------------------- |
| `IMAGE_SIZE`         | Side length of the resized square images (must be ÷ 16) |
| `OUT_CHANNELS`       | 3 for RGB images                                     |
| `LATENT_DIM`         | Dimensionality of the latent space                   |
| `BASE_CHANNELS`      | Channel count of the deepest conv layer              |
| `LEARNING_RATE`      | Adam learning rate                                   |
| `BATCH_SIZE`         | Images per batch                                     |
| `NUM_EPOCHS`         | Number of training epochs                            |
| `NUM_WORKERS`        | DataLoader workers                                   |
| `MAX_SAMPLES`        | Cap on images used (negative = use all)              |
| `PRELOAD_IMAGES`     | Cache decoded images in RAM (recommended: `true`)    |
| `BACKGROUND_COLOR`   | RGBA backdrop used to flatten the alpha channel      |
| `NUM_RECON_SAMPLES`  | Fixed test images shown per reconstruction grid      |
| `TEST_LOSS_SAMPLES`  | Test images used for the per-epoch `test_loss`       |
| `KL_WEIGHT`          | Beta-VAE weight on the KL-divergence term            |

## Outputs

- Model checkpoints: `VAE/models/vae_<run_id>_epoch_<N>.pt`
- Telemetry CSV: `VAE/telemetry_logs/<run_id>.csv`
- Reconstruction grids: `VAE/outputs/reconstruction_epoch_<N>_<run_id>.png`
- Debug logs: `debug_logs/vae.log`

## New smoke-test / full run

For a fast sanity check set `MAX_SAMPLES = 64`, `NUM_EPOCHS = 1`,
`PRELOAD_IMAGES = false`, `NUM_WORKERS = 0` (macOS pickling) and run with
`--mode debug`.

On first launch with `PRELOAD_IMAGES = true` the 15k+ train images are decoded
into RAM once (takes a few minutes and several GB of RAM); afterwards each
epoch is GPU-bound.