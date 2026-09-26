# YOLO — Detection on Frozen-VAE Augmented Floor Plans

Finetunes the latest Mac-compatible Ultralytics YOLO (**YOLO26n**, Jan 2026,
Roboflow-supported, MPS/CPU friendly) to predict the dataset's ground-truth
furniture boxes on images decoded from **frozen VAE latents**.

Two independent runners (separate hyperparameter files, separate launch
scripts) — augmentation never runs as a side effect of training:

- **Augmentation:** `YOLO/augment_runner.py` + `YOLO/augment_hyperparameters.toml`,
  launched via `./run_augmentation.sh`.
- **Training:** `YOLO/runner.py` + `YOLO/hyperparameters.toml`, launched via
  `./train_YOLO.sh`. Fails fast if the dataset is missing/incomplete.

## Pipeline

**Step A — 5x augmentation (`augment.py`):** per source train image, 5 variants:
1. Load RGBA → white backdrop → resize 256×256.
2. Uniform-random frozen checkpoint from
   `vae_next-collie_epoch_{75,100,200,300,400,500}.pt`; encode to latent mean `mu`.
3. `z = mu + N(0, s²)` → decode, with `s` drawn uniformly per variant from
   `LATENT_NOISE_SCALES = [0.0, 0.01, 0.05, 0.075, 0.1]` (`0.0` = clean decode).
4. Gaussian blur on random patches (photometric only; boxes untouched).

No geometric augmentation is applied (no rotation, no flip). Labels stay in
normalized YOLO `cls cx cy w h` and are copied from the source (the VAE resize
is a full-image warp, so normalized coords are invariant). Every variant's
provenance (source, checkpoint, blur count, seed) is recorded in
`augment_manifest.csv`; `seed = BASE_SEED + global_index` drives both the
Python RNG and a per-variant `torch.Generator` for the latent noise, so reruns
are exactly reproducible. Train/val/test output dirs are cleared on regen, so
shrunk configs leave no orphaned files. Val/test mirrors are skipped when
already complete (stem-set comparison, no re-encode).

Throughput: the loop shows a `tqdm` bar plus `img/s` + ETA logs every
`PROGRESS_LOG_SOURCES` sources. Only PNG/label file writes are offloaded to
`SAVE_WORKERS` threads (the VAE forward stays on the main thread — MPS is not
thread-safe for concurrent forwards), so output bytes are identical to a
serial run; worker failures are re-raised on the main thread, never swallowed.

GPU saturation: variants are processed in chunks of `AUG_CHUNK_SOURCES`
sources, grouped by checkpoint epoch so weights load once per group, and
encode/decode runs in full `VAE_BATCH_SIZE` batches instead of one image at a
time (the full 5x run drops from ~151k batch-1 launches + 75k weight loads
to batched launches). Per-row noise keeps its own seeded
generator, so batched rows are bit-identical to single-image decodes.

I/O is vectorized throughout: `torchvision` C-speed tensor↔PIL conversion
(no per-pixel Python loops) and threaded saves at `PNG_COMPRESS_LEVEL = 1`
(~3x faster PNG encoding than the default 6, pixels identical). Measured
~111 outputs/s on MPS for the train phase (vs ~24/s before).

**Step B — finetune (`runner.py`):** `YOLO("yolo26n.pt")` → `.train()` on the
augmented set (`nc=10`: couch, chair, sofa, sofa_cum_bed, center_table,
dining_set, side_table, showcase, tv, vase_table). Device `mps` (fallback CPU),
`workers=0` for macOS, `imgsz=416` (train images are VAE-native 256px),
30 epochs with early-stopping patience 10. Ultralytics' internal
mosaic/mixup/flips are disabled (the VAE pipeline already applies VAE-latent
+ blur); hsv jitter kept. Val split (clean resize) drives
early stopping; the untouched test split is evaluated once at the end for the
unbiased report. Note the train/val domain gap by design: train is
VAE-decoded + blurred (the deployment domain for the diffusion stage), val is
clean — val scores measure generalization to real floor plans.

Class notes: id 3 (`sofa_cum_bed`) has zero boxes in the current splits but is
kept as `nc=10` per the ontology; `chair` is ~60:1 outnumbered by `couch`,
so expect weak minority-class AP without further sampling/focal tuning.

## Telemetry & graphs

- Per-epoch `TelemetryWriter` CSV (`YOLO/telemetry_logs/<run>.csv`):
  `epoch, train_box/cls/dfl, val_box/cls/dfl, precision, recall, mAP50,
  mAP50-95, lr` — mirrored from Ultralytics `results.csv`, plotted live by the
  existing dashboard.
- Val overlays `YOLO/outputs/val_pred_final_*_<run>.png`: green GT boxes +
  red predicted boxes with `class + confidence` annotations.
- Ultralytics curves copied to `YOLO/outputs/`: `results.png`, `PR_curve.png`,
  `F1_curve.png`, `confusion_matrix.png`, `labels.jpg`.
- Best weights: `YOLO/models/yolo_<run>_best.pt`.

## Running

```bash
./venv/bin/pip install "ultralytics>=8.3.0"  # one-time (pulls opencv, torch deps)
./venv/bin/pip install tqdm                  # one-time (augment progress bar)
./run_augmentation.sh       # build 5x dataset (once; reruns regenerate)
./train_YOLO.sh             # debug train (fails fast if dataset incomplete)
./train_YOLO.sh prod        # prod train
./run_scripts/start_yolo_server.sh   # dashboard: loss/mAP curves + overlays + figures
```

Smoke test: set `MAX_SAMPLES=64`, `AUG_FACTOR=2` in
`YOLO/augment_hyperparameters.toml`, then `./run_augmentation.sh`.

> **Mac note:** `train_YOLO.sh` exports `PYTORCH_ENABLE_MPS_FALLBACK=1`.
> Ultralytics runs NMS (`torchvision::nms`) during validation, which has no
> MPS kernel and raises `NotImplementedError` without this flag. Training
> stays on MPS; only that op falls back to CPU. If you launch `YOLO/run.py`
> manually (not via the script), export the flag first.

## Configuration

`YOLO/config.toml`: device, source dataset dirs, VAE dir/run id, output dirs.
`YOLO/augment_hyperparameters.toml`: augmentation knobs (seeds, noise std,
blur ranges, save workers, class names).
`YOLO/hyperparameters.toml`: training knobs (YOLO model/imgsz/batch/epochs/lr,
class names). `CLASS_NAMES` is duplicated in both hyperparameter files — keep
in sync.

## Outputs

- Augmented images/labels: `/Users/aatmaj/Downloads/FurnitureArrangerDataset-main/YOLO Augmented Dataset/{images,labels}/{train,val,test}/`
- Manifest + data config: `augment_manifest.csv`, `data.yaml` (train/val/test) in the same root
- Telemetry: `YOLO/telemetry_logs/<run>.csv`
- Overlays + curves: `YOLO/outputs/`
- Weights: `YOLO/models/yolo_<run>_best.pt`
