# YOLO — Detection on Live Conditional-Diffusion Overlays

Finetunes the latest Mac-compatible Ultralytics YOLO (**YOLO26n**, Jan 2026,
Roboflow-supported, MPS/CPU friendly) to predict the dataset's ground-truth
furniture boxes on images drawn **live from the frozen floorplan→arrangement
diffusion model** (`conditional_diffusion_strong-fawn_epoch_2480`, TV-inclusive).

There is NO fixed train dataset. Per run, `LIVE_TRAIN_SAMPLES` train stems are
selected once from the live mask dataset; before epoch 1 and at the end of
every epoch (Ultralytics `on_train_epoch_end` callback) every train PNG is
redrawn from a fresh diffusion draw conditioned on the GT floorplan mask:

1. GT `floorplan_segmentation.png` → frozen floorplan VAE encoder → condition
   `c` (encoded once per run and cached; `seed = BASE_SEED + epoch` only
   changes the diffusion noise, never the condition).
2. `c` + fresh noise → 1000-step reverse chain → arrangement latent.
3. Latent → frozen TV-inclusive arrangement decoder → furniture render blended
   over the GT floorplan base with `LIVE_OVERLAY_ALPHA` (the same
   `overlay_furniture_on_floorplan` helper the conditional-diffusion grids use).
4. GT `labels.txt` boxes are copied **verbatim** as the target, even though the
   sampled furniture will generally not coincide with them spatially. This is
   an intentional noisy-label regime: YOLO learns a floorplan-conditioned
   layout prior. Expect lower mAP than the reconstruction baseline.

Val/test are **deterministic GT overlays** (GT floorplan + GT arrangement, no
diffusion) in the same color domain, so early stopping and the final unbiased
report stay stable. The single Ultralytics `model.train()` call keeps its
scheduler, patience and `results.csv` untouched; `cache=False` is load-bearing
(a snapshot cache would freeze the first epoch's pixels forever). Per-epoch
provenance (`epoch, output, source, seed`) lands in `live_manifest.csv`.

Implementation: `YOLO/live_diffusion.py` (frozen stack + rendering, reusing the
conditional-diffusion/mask_VAE helpers) + `YOLO/runner.py` (indexing, epoch
callback, finetune, telemetry). Knobs: `LIVE_*` in `YOLO/config.toml` /
`YOLO/hyperparameters.toml`. The older frozen image-VAE 2x pipeline
(`YOLO/augment*.py`, `./run_augmentation.sh`) is kept standalone for ablations;
the trainer never reads it.

## Source dataset layout

The dataset is **folder-per-sample**: a split folder holds one sub-folder per
layout, and each sub-folder carries every render of that layout plus its
labels. There is no separate `images/` or `labels/` directory.

```
<split>/<hash>_original/            # or <hash>_augmented_<NNN>
    full.png                        # 854 x H, RGBA — the ONLY render YOLO reads
    floorplan_only.png              # ignored
    no_tv.png                       # ignored (see below)
    ground_truth.svg                # ignored
    labels.txt                      # "<Class_Name> cx cy w h", normalized to [0,1]
```

Split folders are configured in `YOLO/config.toml`
(`SOURCE_TRAIN_SPLIT_DIR`, `SOURCE_VAL_SPLIT_DIR`, `SOURCE_TEST_SPLIT_DIR`).
Note the source split is named `validate` while the generated output split is
named `val`.

Which render is read is also a `YOLO/config.toml` switch, never a hardcoded
filename: `SOURCE_IMAGE_VARIANT` selects an entry of the
`[default.IMAGE_VARIANTS]` table (the same table the VAE and Diffusion configs
carry), and `SOURCE_LABEL_FILE_NAME` names the annotation file that must sit
next to it.

```toml
SOURCE_IMAGE_VARIANT = "full_png"     # or "floorplan_only"
SOURCE_LABEL_FILE_NAME = "labels.txt"

[default.IMAGE_VARIANTS]
full_png    = "full.png"
floorplan_only = "floorplan_only.png"
```

Only the full render is used, because the boxes in `labels.txt` describe the
**full** furniture set: `no_tv.png` omits the tv pixels while `labels.txt` still
lists the tv box, so its boxes would not match its pixels.

`labels.txt` names its class rather than numbering it, so every line is resolved
to a numeric id through `CLASS_NAMES` before being written to the generated
`labels/` tree. An unrecognised class name aborts the run rather than silently
training on a shifted ontology.

## Pipeline

**Step A — 2x augmentation (`augment.py`):** per source train sample, 2 variants:
1. Load RGBA `full.png` → white backdrop → resize 256×256.
2. Uniform-random frozen checkpoint from
   `vae_next-collie_epoch_{75,100,200,300,400,500}.pt`; encode to latent mean `mu`.
3. `z = mu + N(0, s²)` → decode, with `s` drawn uniformly per variant from
   `LATENT_NOISE_SCALES = [0.0, 0.01, 0.05, 0.075, 0.1]` (`0.0` = clean decode).
4. Gaussian blur on random patches (photometric only; boxes untouched).

**No geometric augmentation anywhere in the YOLO pipeline** — no rotation and no
flip, neither in latent space nor in pixel space. Latent-space flips/rotations
are not even representable: the VAE latent is a flat `LATENT_DIM`-vector with no
spatial axes, so geometry would have to be applied in pixel space. It is not
applied there either. Ultralytics' own `FLIPLR`/`FLIPUD` are pinned to `0.0` in
`YOLO/hyperparameters.toml`. Labels therefore stay in normalized
`cls cx cy w h` and are copied from the source (the VAE resize is a full-image
warp, so normalized coords are invariant) — verifiable as: both variants of a
sample have byte-identical label files.

Every variant's provenance (source sample, checkpoint, blur count, seed) is
recorded in `augment_manifest.csv`; `seed = BASE_SEED + global_index` drives both
the Python RNG and a per-variant `torch.Generator` for the latent noise, so reruns
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
time. Per-row noise keeps its own seeded generator, so batched rows are
bit-identical to single-image decodes.

I/O is vectorized throughout: `torchvision` C-speed tensor↔PIL conversion
(no per-pixel Python loops) and threaded saves at `PNG_COMPRESS_LEVEL = 1`
(~3x faster PNG encoding than the default 6, pixels identical).

**Step B — finetune (`runner.py`):** `YOLO("yolo26n.pt")` → `.train()` on the
augmented set (`nc=12`: Center_table, Chair, Couch, Dining_table_big,
Dining_table_round, Showcase, Side_table, Sofa, Sofa_cum_bed, TV, Vase_table,
Dining_table_medium).
Device `mps` (fallback CPU), `workers=0` for macOS, `imgsz=416` (train images
are VAE-native 256px), 30 epochs with early-stopping patience 10. Ultralytics'
internal mosaic/mixup/flips are disabled (the VAE pipeline already applies
VAE-latent + blur); hsv jitter kept. Val split (clean resize) drives
early stopping; the untouched test split is evaluated once at the end for the
unbiased report. Note the train/val domain gap by design: train is
VAE-decoded + blurred (the deployment domain for the diffusion stage), val is
clean — val scores measure generalization to real floor plans.

Class notes: the class order in `CLASS_NAMES` **is** the id order, and it is
duplicated in `YOLO/hyperparameters.toml` — keep both copies in sync. Rare
classes (`Chair`, `Dining_table_big`, `Sofa_cum_bed`) have few boxes, so expect
weak minority-class AP without further sampling/focal tuning.

## Telemetry & graphs

- Per-epoch `TelemetryWriter` CSV (`YOLO/telemetry_logs/<run>.csv`):
  `epoch, train_box/cls/dfl, val_box/cls/dfl, precision, recall, mAP50,
  mAP50-95, lr` — mirrored incrementally from Ultralytics `results.csv` at
  every epoch end (plus a final sweep), so the dashboard plots live while
  training runs. (`train_dfl`/`val_dfl` are sourced from YOLO26's `l1_loss`
  column; see `RESULTS_CSV_MAP` in `YOLO/runner.py`.)
- Val overlays `YOLO/outputs/val_pred_final_*_<run>.png`: green GT boxes +
  red predicted boxes with `class + confidence` annotations.
- Live snapshots `YOLO/outputs/live_epoch_<E>_<stem>.png` (side-by-side:
  current-epoch overlay | overlay + GT/pred boxes), published while training
  runs by the read-only sidecar `YOLO/live_watch.py` (see below). Shown in the
  dashboard's Live panel.
- Ultralytics curves copied to `YOLO/outputs/`: `results.png`, `PR_curve.png`,
  `F1_curve.png`, `confusion_matrix.png`, `labels.jpg`.
- Best weights: `YOLO/models/yolo_<run>_best.pt`.

## Running

```bash
./venv/bin/pip install "ultralytics>=8.3.0"  # one-time (pulls opencv, torch deps)
./train_YOLO.sh             # debug train (tiny live subset; see LIVE_* knobs)
./train_YOLO.sh prod        # prod train
./run_scripts/start_yolo_server.sh   # dashboard: loss/mAP curves + overlays + figures
```

## Live snapshots while training runs

`YOLO/live_watch.py` (via `./run_scripts/start_live_watch.sh`) is a read-only
sidecar: it polls `live_manifest.csv` for new epochs, snapshots the fixed
`LIVE_WATCH_STEMS` stems (overlay | overlay+boxes, using a temp copy of the
latest `best.pt` so the trainer is never disturbed), and writes
`live_epoch_<E>_<stem>.png` for the dashboard's Live panel. Only the current
epoch is ever published — past epochs' pixels are already overwritten, so
backfilling would mislabel them. Boxes use `LIVE_WATCH_CONF` (far below the
trainer's `CONF_THRESHOLD` so emerging predictions stay visible; raise it as
mAP improves). Each tick also appends new Ultralytics rows to
`<run>_live.csv`, so curves plot live even though the trainer mirrors only at
the end. Only `LIVE_WATCH_*` knobs affect it; it is safe to start/stop anytime
mid-training.

Smoke test: set `LIVE_MAX_SOURCES=4` in `YOLO/hyperparameters.toml`, then
`./train_YOLO.sh` (2 epochs exercise the redraw callback; restore afterwards).
The frozen-VAE ablation path is unchanged: `./run_augmentation.sh` builds the
2x dataset (once; reruns regenerate); the live trainer never reads it.

> **Mac note:** `train_YOLO.sh` exports `PYTORCH_ENABLE_MPS_FALLBACK=1`.
> Ultralytics runs NMS (`torchvision::nms`) during validation, which has no
> MPS kernel and raises `NotImplementedError` without this flag. Training
> stays on MPS; only that op falls back to CPU. If you launch `YOLO/run.py`
> manually (not via the script), export the flag first.

## Configuration

`YOLO/config.toml`: device, live mask-dataset split dirs
(`LIVE_SOURCE_*_SPLIT_DIR`), frozen checkpoint paths
(`COND_DIFFUSION_CHECKPOINT_PATH`, `COND/TARGET_VAE_CHECKPOINT_PATH`), mask
filenames + color indexes, live dataset root, output dirs. The legacy
render-only `SOURCE_*_SPLIT_DIR` / `SOURCE_IMAGE_VARIANT` keys serve only the
standalone frozen-VAE augmentation below.
`YOLO/augment_hyperparameters.toml`: augmentation knobs (seeds, `AUG_FACTOR`,
noise std, blur ranges, save workers, class names).
`YOLO/hyperparameters.toml`: training knobs (YOLO model/imgsz/batch/epochs/lr,
Ultralytics augmentation overrides, class names). `CLASS_NAMES` is duplicated in
both hyperparameter files — keep in sync.

## Outputs

- Live images/labels: `/Users/aatmaj/Downloads/datasets/yolo_live/{images,labels}/{train,val,test}/`
  (train redrawn every epoch; val/test deterministic GT overlays)
- Manifest + data config: `live_manifest.csv`, `data.yaml` (train/val/test) in the same root
- Telemetry: `YOLO/telemetry_logs/<run>.csv`
- Overlays + curves: `YOLO/outputs/`
- Weights: `YOLO/models/yolo_<run>_best.pt`
