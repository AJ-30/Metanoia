# YOLO — Detection on Live Conditional-Diffusion Overlays

Finetunes the latest Mac-compatible Ultralytics YOLO (**YOLO26n**, Jan 2026,
Roboflow-supported, MPS/CPU friendly) to detect furniture on images drawn
**live from the frozen floorplan→arrangement diffusion model**
(`conditional_diffusion_strong-fawn_epoch_2480`, TV-inclusive).

There is NO fixed train dataset. Per run, `LIVE_TRAIN_SAMPLES` train stems are
selected once from the live mask dataset; before epoch 1 and at the end of
every epoch (Ultralytics `on_train_epoch_end` callback) every train image is
redrawn from a fresh diffusion draw conditioned on the GT floorplan mask:

1. GT `floorplan_segmentation.png` → frozen floorplan VAE encoder → condition
   `c` (encoded once per run and cached; `seed = BASE_SEED + epoch` only
   changes the diffusion noise, never the condition).
2. `c` + fresh noise → 1000-step reverse chain → arrangement latent.
3. Latent → frozen TV-inclusive arrangement decoder → furniture render blended
   over the GT floorplan base with `LIVE_OVERLAY_ALPHA` (the same
   `overlay_furniture_on_floorplan` helper the conditional-diffusion grids use).
4. **Boxes are derived from the arrangement that was just drawn** — connected
   components per furniture class (`connected_component_boxes`), mapped through
   `mask→YOLO id` and written as normalized `cls cx cy w h`. Image and label
   therefore always agree, which is what makes this an actual detection task
   rather than label noise. `LIVE_REGENERATE_LABELS = false` restores the old
   verbatim-GT-label behaviour for ablations; measured on this dataset only
   ~27% of those boxes overlap the drawn furniture at IoU > 0.5 (vs ~94% for
   boxes derived from the pixels they label).

Val/test are **deterministic GT overlays** (GT floorplan + GT arrangement, no
diffusion) with labels derived by the same routine, so all three splits share
one labelling convention and early stopping plus the final unbiased report stay
stable. Mirrors carry a `.mirror_signature` sidecar (ontology, aliases, alpha,
image size); changing any of those re-renders instead of silently reusing labels
written under the old class ids.

## Ontology: 10 classes, and why not 12

The arrangement mask vocabulary is 10 classes (`sofa`, `tv`, `dining_set`,
`side_table`, `showcase`, `couch`, `chair`, `center_table`, `vase_table`,
`sofa_cum_bed`). The source `labels.txt` names 12 furniture types, but the three
`Dining_table_*` sizes are indistinguishable in both the mask and the render, so
they are folded into a single `Dining_table` class via `CLASS_ALIASES`.
`build_mask_to_yolo_id_map` fails fast if the two ontologies cannot be matched
one-to-one, rather than silently training on a shifted class space. Note also
that `sofa`, `couch` and `chair` are three mask ids that render in the *same*
colour, so they are separable by shape/context but not by colour alone.

## Training budget (read this before changing `LIVE_TRAIN_SAMPLES`)

Ultralytics accumulates gradients over `round(nbs / batch)` batches and steps
the optimizer only once per accumulation group, so an epoch performs
`batches_per_epoch // accumulate` real weight updates. Two settings must stay
consistent or training silently does nothing:

- `NOMINAL_BATCH_SIZE` (= `nbs`) is pinned to `BATCH_SIZE` so accumulation is 1.
  The Ultralytics default `nbs=64` with `batch=16` gives `accumulate=4`.
- `LIVE_TRAIN_SAMPLES` must be a multiple of `BATCH_SIZE` and large enough for
  many batches per epoch. Every stem costs one 1000-step reverse chain per
  epoch (~0.3–0.6 s/image measured on MPS).
- `LIVE_VAL_SAMPLES` caps the deterministic GT validation overlays (default: 128)
  so Ultralytics' end-of-epoch validation does not evaluate the entire ~20k pool
  every epoch (~13.5 min/epoch on MPS vs ~3s for 128 samples).

`validate_training_budget` computes the resulting step count and **raises before
training** if an epoch would take no optimizer step, or if the whole run would
take fewer than two — the configuration that produced a 30-epoch / 3.3-hour run
with 10 optimizer steps and mAP50 ≈ 0.05. Smoke tests:

```bash
./venv/bin/python YOLO/test_yolo_pipeline.py   # pure-logic unit tests
# then temporarily lower LIVE_TRAIN_SAMPLES / NUM_EPOCHS and run ./train_YOLO.sh
```

Implementation: `YOLO/live_diffusion.py` (frozen stack, rendering, label
derivation, ontology bridge) + `YOLO/runner.py` (indexing, epoch callback,
budget guard, finetune, telemetry). Knobs: `LIVE_*` in
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
live set (`nc=10`: Center_table, Chair, Couch, Dining_table, Showcase,
Side_table, Sofa, Sofa_cum_bed, TV, Vase_table — see *Ontology* above).
Device `mps` (fallback CPU), `workers=0` for macOS, `imgsz=416` (train images
are 256px), 30 epochs with early-stopping patience 10. Ultralytics' internal
mosaic/mixup/flips are disabled (the live draws already supply the variation);
hsv jitter kept. `nbs = BATCH_SIZE` keeps accumulation at 1 so every batch
updates the weights. Val (deterministic GT overlays) drives early stopping; the
untouched test split is evaluated once at the end for the unbiased report. Note
the train/val domain gap by design: train pixels are fresh diffusion draws,
val is clean — val scores measure generalization to real floor plans.

Class notes: the class order in `CLASS_NAMES` **is** the id order, and it is
duplicated in `YOLO/augment_hyperparameters.toml` — keep both copies in sync.
`CLASS_ALIASES` (also duplicated) maps the dataset's source names onto that
vocabulary; an unmapped source name aborts the run rather than silently
training on a shifted ontology. Rare classes (`Chair`, `Sofa_cum_bed`) have
few boxes, so expect weak minority-class AP without further sampling/focal
tuning. `sofa`/`couch`/`chair` render in the same colour, so they are
distinguishable only by shape and context.

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
./run_scripts/predict_YOLO.sh   # offline inference viewer (see below)
./run_scripts/start_yolo_server.sh   # dashboard: loss/mAP curves + overlays + figures
```

## Looking at the output (`predict.py`)

`./run_scripts/predict_YOLO.sh` runs the saved best weights over the live
dataset and writes GT-vs-prediction overlays plus a single contact sheet to
`YOLO/outputs/predict/`. It is read-only with respect to training state: it
never rewrites the dataset, never touches telemetry, and never loads the frozen
diffusion stack. It reuses the trainer's own drawing and device helpers, so the
images match what training produced.

```bash
./run_scripts/predict_YOLO.sh                                  # newest best.pt, val split
./run_scripts/predict_YOLO.sh prod                             # prod-mode config
./run_scripts/predict_YOLO.sh debug --split test --limit 20    # different split/size
./run_scripts/predict_YOLO.sh debug --image /path/plan.png     # one explicit image
./run_scripts/predict_YOLO.sh debug --weights path/best.pt     # a specific run
./run_scripts/predict_YOLO.sh debug --conf 0.25                # evaluation threshold
```

Green boxes are ground truth, red boxes are predictions with `class + confidence`
(caps at `PREDICT_MAX_BOXES`, highest-confidence first).

**Reading the confidence.** `PREDICT_CONF` defaults to `0.001`, far below the
evaluation `CONF_THRESHOLD = 0.25`, because a lightly-trained model peaks near
`1e-3`. At the evaluation threshold the viewer would report zero detections and
look broken rather than merely untrained. The tool always logs the highest
confidence it actually saw, so "no boxes" is distinguishable from "model is
broken" — if it prints a peak, the model is running and just under-trained.

The wrapper exports `PYTORCH_ENABLE_MPS_FALLBACK=1`; Ultralytics runs
`torchvision::nms` during prediction, which has no MPS kernel.

## Live snapshots while training runs

`YOLO/live_watch.py` (via `./run_scripts/start_live_watch.sh`) is a read-only
sidecar: it polls `live_manifest.csv` for new epochs, snapshots the fixed
`LIVE_WATCH_STEMS` stems (overlay | overlay+boxes, using a temp copy of the
latest `best.pt` so the trainer is never disturbed), and writes
`live_epoch_<E>_<stem>.png` for the dashboard's Live panel. Only the current
epoch is ever published — past epochs' pixels are already overwritten, so
backfilling would mislabel them. Boxes use `LIVE_WATCH_CONF` (defaults to the
trainer's `CONF_THRESHOLD`, so the panel shows what evaluation counts) and are
capped at `LIVE_WATCH_MAX_BOXES` highest-confidence-first; a near-zero
confidence is deliberately avoided, because an undertrained model then emits its
full NMS output and the snapshot becomes unreadable red spaghetti. Each tick
also appends new Ultralytics rows to `<run>_live.csv`, so curves plot live even
though the trainer mirrors only at the end. Only `LIVE_WATCH_*` knobs affect it;
it is safe to start/stop anytime mid-training.

Smoke test: temporarily set `LIVE_TRAIN_SAMPLES` and `NUM_EPOCHS` low in
`YOLO/hyperparameters.toml`, then `./train_YOLO.sh` (2+ epochs exercise the
redraw callback; restore afterwards). `validate_training_budget` will reject a
combination too small to update the weights. The frozen-VAE ablation path is
unchanged: `./run_augmentation.sh` builds the 2x dataset (once; reruns
regenerate); the live trainer never reads it.

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
`NOMINAL_BATCH_SIZE`, Ultralytics augmentation overrides, class names +
`CLASS_ALIASES`, all `LIVE_*` knobs). `CLASS_NAMES` and `CLASS_ALIASES` are
duplicated in both hyperparameter files — keep in sync.

## Outputs

- Live images/labels: `/Users/aatmaj/Downloads/datasets/yolo_live/{images,labels}/{train,val,test}/`
  (train redrawn every epoch; val/test deterministic GT overlays). Boxes on all
  three splits are derived from the pixels they label.
- Manifest + data config: `live_manifest.csv` (`epoch, output, source, seed,
  boxes, label_source`), `data.yaml` (train/val/test) in the same root
- Mirror content sidecars: `.mirror_signature` in each mirrored image dir
- Telemetry: `YOLO/telemetry_logs/<run>.csv`
- Overlays + curves: `YOLO/outputs/`
- Weights: `YOLO/models/yolo_<run>_best.pt`
