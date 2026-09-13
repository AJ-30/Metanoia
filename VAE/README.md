# VAE — Variational Autoencoder on Floor-Plan Images

Trains a convolutional VAE (PyTorch) on the images in
`FurnitureArrangerDataset-main/super_augmentation_1/images`.

## Files

- `model.py` — Convolutional VAE (Encoder / Decoder) and the VAE loss function.
- `runner.py` — `VAERunner` class: dataset building, training loop, checkpoints,
  telemetry and reconstruction sample grids.
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

### Visualising training metrics

While training runs, telemetry rows are written to `VAE/telemetry_logs/<run_id>.csv`.
Start the dashboard server to see the loss curves:

```bash
./run_scripts/start_server.sh              # http://127.0.0.1:8080
./run_scripts/start_server.sh 9000         # custom port
```

The dashboard (Flask + Chart.js in `helpers/server.py`) lists every run from
`VAE/telemetry_logs` and plots epoch vs train/KL/recon loss.

## Configuration

### `config.toml`

| Key                      | Meaning                                          |
| ------------------------ | ------------------------------------------------ |
| `DEVICE`                 | `mps`, `cuda` or `cpu` (falls back to CPU)       |
| `IMAGE_DIR`              | Folder of images to train on                     |
| `DEBUG_LOG_FILE_PATH`    | Log file written by the shared logger            |
| `MODEL_SAVE_PATH`        | Where `.pt` checkpoints are saved                |
| `TELEMETRY_LOG_DIRECTORY`| Where the run's `*.csv` telemetry goes           |
| `OUTPUT_SAMPLES_DIRECTORY`| Where reconstruction grid images are saved      |
| `SAVE_AFTER_INTERVAL`    | Save a reconstruction grid every N epochs        |

`[debug]` / `[prod]` sections follow the same convention as the root `config.toml`.

### `hyperparameters.toml`

| Key               | Meaning                                              |
| ----------------- | ---------------------------------------------------- |
| `IMAGE_SIZE`      | Side length of the resized square images (must be ÷ 16) |
| `OUT_CHANNELS`    | 3 for RGB images                                     |
| `LATENT_DIM`      | Dimensionality of the latent space                   |
| `BASE_CHANNELS`   | Channel count of the deepest conv layer              |
| `LEARNING_RATE`   | Adam learning rate                                   |
| `BATCH_SIZE`      | Images per batch                                     |
| `NUM_EPOCHS`      | Number of training epochs                            |
| `NUM_WORKERS`     | DataLoader workers                                   |
| `MAX_SAMPLES`     | Cap on images used (negative = use all)              |
| `BACKGROUND_COLOR`| RGBA backdrop used to flatten the alpha channel      |
| `NUM_RECON_SAMPLES`| Images shown per reconstruction grid                |
| `KL_WEIGHT`       | Beta-VAE weight on the KL-divergence term            |

## Outputs

- Model checkpoints: `VAE/models/vae_<run_id>_epoch_<N>.pt`
- Telemetry CSV: `VAE/telemetry_logs/<run_id>.csv`
- Reconstruction grids: `VAE/outputs/reconstruction_epoch_<N>_<run_id>.png`
- Debug logs: `debug_logs/vae.log`

## New smoke-test / full run

For a fast sanity check set `MAX_SAMPLES = 64`, `NUM_EPOCHS = 1`,
`NUM_WORKERS = 0` (macOS pickling) and run with `--mode debug`.