# mask_VAE — Variational Autoencoder over Floor-Plan Segmentation Masks

Trains a convolutional VAE (PyTorch) on the **integer segmentation masks** of
`datasets/work/{train,test}`, with a per-class **binary cross-entropy**
reconstruction loss and a KL term on the latent.

This is a sibling of `VAE/`, not a modification of it. `VAE/` learns rendered
floor-plan *images*; `mask_VAE/` learns the *semantic layout* directly. The
rendered images (`full.png`, `floorplan_only.png`, `no_tv.png`) and
`labels.txt` are never read here.

After **every epoch** the runner:

- evaluates the model on a fixed test subset and logs losses plus foreground
  IoU / Dice to `mask_VAE/telemetry_logs/<run_id>.csv`;
- writes one IoU and one Dice value **per class** to
  `mask_VAE/telemetry_logs/<run_id>_per_class.csv`;
- saves a **colorized** reconstruction grid — ground truth above reconstruction,
  one row pair per mask group;
- saves a **colorized** generation grid of layouts sampled from the latent prior.

At the end of training the model is evaluated once on the **full test set** and
the worst-reconstructed classes are named in the log.

## Why the saved grids are colored, not raw masks

The mask PNGs store plain integer class ids, not colors, so rendering them
naively produces a near-black picture where only the class *indices* differ. The
dataset ships the authoritative lookup for turning those ids into display colors:
`floorplan_color_index.json` and `furniture_color_index.json`, which map
`"<class id>"` to `{"name", "color": "#rrggbb"}`, with id `0` as background.

`mask_VAE/masking.py` reads those JSONs at startup and every grid saved to
`mask_VAE/outputs/` is passed through them: the model's per-class logits are
reduced with an `argmax` over the group's channels, and the winning class id is
looked up in the color table. **The dashboard therefore serves floor plans, not
grayscale class-id maps.** Changing a color in the dataset's JSON changes the
saved grids and the dashboard with no code change.

Verified against the current dataset: the six floorplan classes present in a
sample render to exactly the RGB triples the JSON specifies, e.g. `living` to
`(217, 217, 217)` and `wall` to `(255, 217, 47)`.

## Source dataset layout

The dataset is **folder-per-sample**: a split folder holds one sub-folder per
layout, and each sub-folder carries every render of that layout.

```
<dataset root>/
    floorplan_color_index.json      # "<class id>" -> {"name", "color"}
    furniture_color_index.json
<split>/<hash>_original/            # or <hash>_augmented_<NNN>
    floorplan_segmentation.png      # 854 x H, L — integer class ids
    furniture_segmentation.png      # 854 x H, L — integer class ids
    full.png                        # not read by this model
    floorplan_only.png              # not read by this model
    no_tv.png                       # not read by this model
    labels.txt                      # not read by this model
    ground_truth.svg
```

Masks are resized to a square with **nearest-neighbour** interpolation. That is
not a detail: an interpolating filter blends neighbouring class ids into values
that belong to no class, which silently invalidates the one-vs-rest target.

### Channel layout

The model input and output are one-vs-rest binary tensors with **an explicit
background channel**. With `MASK_GROUPS = ["floorplan", "furniture"]`:

| Global channel | Meaning |
| -------------- | ------- |
| `0` | Shared background: on wherever the pixel is `0` in *every* selected mask |
| `1 .. 17` | `floorplan` classes 1..17, in color-map key order |
| `18 .. 27` | `furniture` classes 1..10, in color-map key order |

The background channel is shared on purpose: both masks are rendered from the
same layout, so a pixel is background for the model exactly when it is
background in both. Total: `1 + 17 + 10 = 28` channels.

### Split sizes

The dataset is large and heavily augmented — **69 776 train / 10 043 test /
19 920 validate** samples. `MAX_SAMPLES = -1` therefore means a full epoch is
~4361 steps at `BATCH_SIZE = 16`, and `PRELOAD_MASKS` must stay `false` (see
below). Use `MAX_SAMPLES` to cap an epoch when iterating on the setup.

The number of classes lives in the dataset's JSON files, not in this repo.
`MASK_CHANNELS` in `hyperparameters.toml` states the expected total and the
runner **asserts it against the layout at startup**, so a dataset regenerated
with a different class table fails immediately with a message naming the value
to copy in, instead of training a model that cannot be decoded.

## Choosing which masks to learn

`MASK_GROUPS` in `config.toml` is the switch. It is also an architectural
switch: its order and length define the channel count, so changing it makes
previously saved checkpoints unloadable.

```toml
MASK_GROUPS = ["floorplan"]               # background + 17 = 18 channels
MASK_GROUPS = ["furniture"]               # background + 10 = 11 channels
MASK_GROUPS = ["floorplan", "furniture"]  # background + 17 + 10 = 28 channels
```

After changing it, update `MASK_CHANNELS` in `hyperparameters.toml` to the
count the startup assertion prints.

## Files

- `masking.py` — Reads the color index JSONs, owns the channel layout
  (`MaskLayout`), encodes integer masks into one-vs-rest channels, and renders
  any per-class tensor back to colored RGB.
- `dataset.py` — `MaskDataset`: enumerates sample folders that provide all
  selected masks, resizes nearest-neighbour, caches as `uint8`.
- `model.py` — `MaskDecoder`, `MaskVAEModel` and `mask_vae_loss` (BCE + KL).
  The encoder is **imported and reused from `VAE/model.py`**, which is
  activation-agnostic, so there is exactly one encoder definition in the repo.
- `metrics.py` — Per-class IoU and Dice, accumulated through a confusion matrix.
- `per_class_telemetry.py` — The per-class CSV writer.
- `runner.py` — `MaskVAERunner`: dataloaders, training loop, evaluation,
  colorized grids, checkpoints, telemetry.
- `run.py` — Entry point. Calls `MaskVAERunner().train()`.
- `config.toml` — Non-model configuration (device, paths, which masks, colors).
- `hyperparameters.toml` — Architecture and training hyperparameters.
- `README.md` — This file.

## Running

From the repo root (make sure the conda env `./venv` is active):

```bash
python mask_VAE/run.py --mode debug      # debug mode
python mask_VAE/run.py --mode prod       # prod mode
```

Or with the launcher script (auto-activates the conda env, archives previous
artifacts, then trains):

```bash
./train_mask_VAE.sh         # debug mode
./train_mask_VAE.sh prod    # prod mode
```

### Visualising training metrics, colored reconstructions and generations

Start the dashboard to see the loss curves, the foreground IoU / Dice curves and
the colorized grids:

```bash
./run_scripts/start_mask_vae_server.sh          # http://127.0.0.1:8080
./run_scripts/start_mask_vae_server.sh 9000     # custom port
```

The dashboard (Flask + Chart.js in `helpers/server.py`) plots every numeric
column of `<run_id>.csv`, including `test_fg_iou` and `test_fg_dice`, and shows
a "Reconstructions" panel (grid of fixed test samples, ground truth above
reconstruction) and a "Generations" panel. Both have an epoch dropdown, and the
dashboard auto-refreshes so the newest grids appear at the end of each epoch.

The per-class CSV is **not** charted by the dashboard — it is a wide CSV meant to
be read directly, which is where you go to find out *which* class is failing:

```bash
column -s, -t < mask_VAE/telemetry_logs/<run_id>_per_class.csv | less -S
```

## Configuration

### `config.toml`

| Key | Meaning |
| --- | ------- |
| `DEVICE` | `mps`, `cuda` or `cpu` (falls back to CPU) |
| `TRAIN_IMAGES_DIR` | Split folder of samples to train on |
| `TEST_IMAGES_DIR` | Split folder of samples used for evaluation |
| `MASK_GROUPS` | Ordered list of mask groups to learn; defines the channel count |
| `MASK_COLOR_INDEX_DIR` | Directory holding the `*_color_index.json` files |
| `DEBUG_LOG_FILE_PATH` | Log file written by the shared logger |
| `MODEL_SAVE_PATH` | Where `.pt` checkpoints are saved |
| `TELEMETRY_LOG_DIRECTORY` | Where the main `*.csv` telemetry goes |
| `PER_CLASS_TELEMETRY_LOG_DIRECTORY` | Where the per-class `*.csv` goes |
| `OUTPUT_SAMPLES_DIRECTORY` | Where the colorized grids are saved |
| `IMAGE_VARIANTS`-style tables | `MASK_FILES` and `MASK_COLOR_INDEX_FILES`: group → filename, so no filename is hardcoded in Python |

`MASK_COLOR_INDEX_DIR` falls back to the parent of `TRAIN_IMAGES_DIR` when a
JSON is not found there, which is what happens if only the split paths were
updated for a new machine.

> **TOML gotcha:** every non-table key must stay **above** the first
> `[default.<TABLE>]` header in the file. A bare key written after such a header
> belongs to that sub-table, not to `[default]`. Both `mask_VAE` and `VAE` check
> this explicitly and raise an error naming the offending key.

### `hyperparameters.toml`

| Key | Meaning |
| --- | ------- |
| `IMAGE_SIZE` | Square side length of the masks (must be divisible by 16) |
| `MASK_CHANNELS` | Class channels; asserted against the layout at startup |
| `LATENT_DIM` | Dimensionality of the flat latent vector |
| `BASE_CHANNELS` | Channel count of the deepest conv layer |
| `DROPOUT` | Dropout probability (encoder + decoder, training only) |
| `LATENT_ADDITIONAL_NOISE` | Std-dev of extra latent noise added during training |
| `KL_WEIGHT` | Beta-VAE weight on the KL term. `0.0` = no latent regularization |
| `BCE_POS_WEIGHT` | Positive-class weight of the BCE. `1.0` = plain BCE |
| `LEARNING_RATE` | Adam learning rate |
| `BATCH_SIZE` | Masks per batch |
| `NUM_EPOCHS` | Number of training epochs |
| `SEED` | RNG seed for Python and torch, applied before anything is built |
| `NUM_WORKERS` | DataLoader workers |
| `MAX_SAMPLES` | Cap on samples per split (negative = use all) |
| `PRELOAD_MASKS` | Cache decoded masks in RAM as `uint8` (recommended) |
| `TELEMETRY_INTERVAL` | Epochs between telemetry rows (1 = every epoch) |
| `CHECKPOINT_INTERVAL` | Epochs between checkpoint saves |
| `NUM_RECON_SAMPLES` | Fixed test masks shown per reconstruction grid |
| `NUM_GEN_SAMPLES` | Novel layouts sampled from the latent prior |
| `TEST_LOSS_SAMPLES` | Test masks used for the per-epoch metrics |

### Reading the metrics

| Column | Meaning |
| ------ | ------- |
| `train_loss` / `test_loss` | `bce_loss + KL_WEIGHT * kl_loss` |
| `bce_loss` / `test_bce` | Per-image BCE summed over all pixels × channels |
| `kl_loss` / `test_kl` | KL summed over latent dims, averaged over images |
| `kl_per_dim` | `kl_loss / LATENT_DIM`. The posterior-collapse diagnostic: when it approaches zero the encoder has stopped using the latent |
| `test_fg_iou` / `test_fg_dice` | Mean over **foreground** classes only, so the easy background class cannot hide a model that learned nothing |

The loss magnitudes are in the hundreds of thousands by construction: like
`VAE/model.py`, the BCE is a **per-image sum** over 28 × 256 × 256 elements
rather than a mean over them. This keeps the reduction convention consistent
with the image VAE so the two are directly comparable, and it makes the
`kl_per_dim` and `test_fg_iou` columns the ones to read for actual progress.

## Outputs

- Model checkpoints: `mask_VAE/models/mask_vae_<run_id>_epoch_<N>.pt`
- Telemetry CSV: `mask_VAE/telemetry_logs/<run_id>.csv`
- Per-class telemetry CSV: `mask_VAE/telemetry_logs/<run_id>_per_class.csv`
- Reconstruction grids: `mask_VAE/outputs/reconstruction_epoch_<N>_<run_id>.png`
- Generation grids: `mask_VAE/outputs/generation_epoch_<N>_<run_id>.png`
- Debug logs: `debug_logs/mask_vae.log`

Checkpoints store the hyperparameters, `MASK_GROUPS`, the per-group channel
indices and the class names alongside the weights, so a later stage can rebuild
this exact model without guessing which mask layout was in effect.

## Smoke test / full run

For a fast sanity check set `MAX_SAMPLES = 32`, `NUM_EPOCHS = 1`,
`PRELOAD_MASKS = true` and run with `--mode debug`. Preloading is safe at that
size (~60 MB) and makes the smoke test independent of disk speed.

### Caching, memory and throughput

`PRELOAD_MASKS = false` is the correct default **at the current dataset size**.
A cached sample costs `MASK_CHANNELS x IMAGE_SIZE x IMAGE_SIZE` bytes as
`uint8`, i.e. **~1.8 MB** at `28 x 256 x 256`, so caching all 69 776 train masks
would need **~126 GB**. Turn it on only together with a `MAX_SAMPLES` cap that
keeps the total within RAM — for example `MAX_SAMPLES = 4096` uses ~7.4 GB and
removes the PNG decode from the training loop.

`NUM_WORKERS = 0` is deliberate. Measured on this machine with the real masks:
**114 masks/s in-process vs 38 masks/s with 4 workers**, because decoding these
small PNGs is cheaper than the multiprocessing queue round-trip. Worker
processes also fail outright on macOS here, because the dataset holds a
per-sample state that does not survive the `spawn` start method. Keep it at 0.

Activations scale with `BATCH_SIZE x MASK_CHANNELS x IMAGE_SIZE**2`; at
`BATCH_SIZE = 16`, `IMAGE_SIZE = 256`, `BASE_CHANNELS = 128` that is roughly
3.5 GB on MPS. Halve `BATCH_SIZE` on a smaller machine.

Two costs scale with the split and are easy to be surprised by:

- One epoch is ~4361 steps at the full 69 776 samples, so wall-clock per epoch
  is minutes, not seconds.
- The **final** full-test evaluation ignores `TEST_LOSS_SAMPLES` and covers all
  10 043 test masks, which takes a few minutes. The per-epoch metrics are capped
  at `TEST_LOSS_SAMPLES` and stay fast.

## Known limitations

- **The latent is a flat vector.** It matches `VAE/` so these latents stay
  interchangeable with the ones the Diffusion and YOLO stages consume, but it
  means a missing region cannot be addressed at a chosen spatial location in
  latent space. Making the latent spatial would break that interchangeability.
- **`BCE_POS_WEIGHT = 1.0` is a plain BCE.** Mask pixels are dominated by
  background, so the all-background answer already scores well. If
  `test_fg_iou` plateaus early while `test_bce` keeps falling, raise it.
- **The dataset is small** (251 train samples in 31 families), and no geometric
  augmentation is applied here. Sample diversity comes entirely from the
  pre-generated `_augmented_<NNN>` folders.
- **Colors are looked up, not learned.** Two classes sharing a hex value (the
  furniture map gives both `couch` and `chair` `#f472b6`) are indistinguishable
  in the saved grids, though they remain separate channels with separate IoU.