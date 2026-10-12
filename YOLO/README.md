# YOLO — Furniture Detection on Live Diffusion-Drawn Floor Plans

Finetunes YOLO26n to detect furniture bounding boxes on floor-plan overlays
where **every train image is redrawn live from the frozen
floorplan→arrangement diffusion model at the end of every epoch**, using
**all available samples** of the dataset. At the end of every epoch the
trainer publishes GT (green) vs predicted (red) boxes drawn directly on the
floorplan + furniture images.

```
./train_YOLO.sh            # train (debug mode)
./train_YOLO.sh prod       # train (prod mode)
./venv/bin/python YOLO/test_yolo_pipeline.py   # pure-logic unit tests
./run_scripts/predict_YOLO.sh                  # offline GT-vs-pred viewer
./run_scripts/start_yolo_server.sh             # dashboard (telemetry + overlays)
```

## What the model sees, and what it must predict

Each image is an overlay of a floor plan and its furniture arrangement:

* the floor plan comes from the GT `floorplan_segmentation.png` (the
  diffusion **condition** — the model never sees the GT furniture of that
  sample when drawing train images);
* the furniture on train images is a **fresh diffusion draw** for that floor
  plan (TV-inclusive arrangement decoder), blended over the floor plan with
  `REDRAW_OVERLAY_ALPHA`;
* val/test images are **deterministic GT overlays** (GT floor plan + GT
  arrangement mask, no diffusion) rendered once per configuration, so
  per-epoch validation, early stopping and the final report stay stable.

Boxes are **derived from the arrangement that was actually drawn**
(connected components per furniture class, `REDRAW_MIN_INSTANCE_AREA` drops
decoder speckle), mapped through the mask→YOLO id table and written as
normalized `cls cx cy w h`. Image and target therefore always agree — this is
a real detection task, not a noisy-label regime. All three splits share this
one labelling convention. (Verified end to end: rendered pixels are exact
recomputations of the GT overlay, and every box contains its class's pixels.)

## Every epoch uses all available data

The whole folder-per-sample dataset is used, every epoch — no subsampling:

| split | source | count | pixels |
|-------|--------|-------|--------|
| train | `LIVE_SOURCE_TRAIN_SPLIT_DIR` | 69,776 | redrawn every epoch, seed `BASE_SEED + epoch` |
| val   | `LIVE_SOURCE_VAL_SPLIT_DIR`   | 19,920 | deterministic GT overlay, rendered once |
| test  | `LIVE_SOURCE_TEST_SPLIT_DIR`  | 10,043 | deterministic GT overlay, rendered once |

Per-epoch cost on an M1 Pro (MPS): ~15–20 min redraw (1000-step reverse
chain for all 70k train images on MPS, batched decode, threaded box/PG writes)
+ ~5–8 min training + ~15–25 min validation on the full val split.
`SOURCE_SAMPLE_CAP` / `VAL_SAMPLE_CAP` / `TEST_SAMPLE_CAP` (all `-1` by
default) exist only to smoke-test quickly; a real run with caps trains on a
subset. The frozen condition latents (one 128-float vector per train floor
plan) are encoded **once per run** and reused every epoch — only the diffusion
noise is fresh.

## End-of-epoch visualization

In the trainer's `on_fit_epoch_end` callback (after Ultralytics has written
that epoch's `results.csv` row):

1. the fresh results row is mirrored into the project telemetry CSV, so the
   dashboard plots while training runs;
2. `EPOCH_OVERLAY_STEMS` (default 32) **fixed** val stems — chosen once per
   run with `BASE_SEED`, listed in `YOLO/outputs/epoch_overlay_stems.txt` —
   are re-predicted with the epoch's own weights (the EMA model validation
   scored) and saved with green GT + red predicted boxes (class + confidence,
   capped at `EPOCH_OVERLAY_MAX_BOXES`, threshold `EPOCH_OVERLAY_CONF`) to
   `YOLO/outputs/epoch_XX/`, plus `YOLO/outputs/montage_epoch_XX.png`;
3. all 70k train images are redrawn from a fresh diffusion draw for the next
   epoch, and the freshly drawn boxes are pushed into the trainer's
   in-memory dataset labels (skipped on the final / early-stopped epoch).

Because the stems are fixed, comparing `montage_epoch_01.png` with
`montage_epoch_30.png` shows learning directly. The log line per epoch also
reports the peak confidence the model emitted, so "no boxes drawn" is always
distinguishable from "model not firing yet" (`EPOCH_OVERLAY_CONF` defaults to
0.25, the threshold a trained detector is judged at, so the panel shows what
evaluation would count).

## Keeping each epoch's boxes attached to its pixels

Ultralytics reads `labels/*.txt` **once**, when the train dataloader is
built, and keeps the parsed boxes in memory for the whole run; only images are
re-read from disk each batch (`cache=False`). Rewriting the label files per
epoch is therefore not enough — the model would train on the *first* draw's
boxes while looking at every later draw's pixels (the exact pixel/target
mismatch this pipeline is built to avoid). So after each redraw the runner
pushes the drawn boxes into `trainer.train_loader.dataset.labels`
(`apply_label_lines_to_dataset`, same dict format `cache_labels` produces).
Verified inside a real training loop: label mismatches are 0 at every epoch
start while the pixel fingerprint changes per epoch. This requires
`WORKERS = 0` (worker processes would hold forked copies of the label list);
the trainer fails fast with a clear error if `WORKERS != 0`.

## Ontology: 10 classes

`CLASS_NAMES` (index == class id) must match what the arrangement mask can
express. The mask vocabulary is 10 classes (`sofa`, `tv`, `dining_set`,
`side_table`, `showcase`, `couch`, `chair`, `center_table`, `vase_table`,
`sofa_cum_bed`); the source dataset's three `Dining_table_*` sizes are
indistinguishable in mask and render and fold onto one class via
`CLASS_ALIASES` (`dining_set` → `Dining_table`).
`build_mask_to_yolo_id_map` aborts on any ontology mismatch instead of
silently training on a shifted class space. Note `sofa`, `couch` and `chair`
render in the same colour: they are separable by shape/context only.

## Layout of the code

| file | role |
|------|------|
| `run.py` | CLI entry point (`--mode debug|prod`) |
| `runner.py` | dataset wiring, budget guard, Ultralytics training, per-epoch callback, final reports |
| `dataset.py` | frozen VAEs + diffusion stack, deterministic val/test renderer, live epoch redrawer, box/label derivation, signatures |
| `overlay.py` | GT/pred box drawing, per-epoch overlays, montage |
| `predict.py` | offline viewer over the rendered splits or one image |
| `test_yolo_pipeline.py` | pure-logic unit tests (no GPU/checkpoints) |

Reused, not duplicated: rendering helpers (`overlay_furniture_on_floorplan`,
`render_group_rgb`), mask encoding (`encode_masks_to_channels`), the frozen
VAE loader (`build_frozen_mask_vae`), the project logger/telemetry/config
helpers.

## Configuration

`YOLO/config.toml`: device, dataset split dirs, mask filenames, frozen
checkpoint paths, live dataset root, output dirs.
`YOLO/hyperparameters.toml`: every training/rendering number — `YOLO_MODEL`,
`IMG_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, `LR0`, `PATIENCE`, `NOMINAL_BATCH_SIZE`
(pinned to `BATCH_SIZE` so gradient accumulation is 1 — see the budget guard),
all `REDRAW_*` knobs (alpha, image size, chunk sizes, workers, min instance
area, PNG compression), `EPOCH_OVERLAY_*`, `PREDICT_*`, `CLASS_NAMES`,
`CLASS_ALIASES`, Ultralytics augmentation overrides.

`NOMINAL_BATCH_SIZE` matters: Ultralytics steps the optimizer once per
`round(nbs / batch)` batches, so a mismatch can silently starve training.
`validate_training_budget` fails fast before the first draw when the run would
take too few optimizer steps.

## Reproducibility

* Epoch E's pixels are a pure function of `BASE_SEED + E`: the chain reseeds
  `torch.manual_seed` before each redraw and runs on MPS with fixed chunk
  sizes (verified bit-identical across same-seed runs).
* Ultralytics itself runs with `deterministic=True` (its default) and
  `seed = BASE_SEED`.
* Val/test renders are deterministic and cached behind a content signature
  (stems + class names + aliases + alpha + image size + min area): an ontology
  or alpha change re-renders instead of reusing stale pixels/labels.
* `redraw_manifest.csv` records one provenance row per redraw (epoch, seed,
  images, boxes, seconds).

## Outputs

* Live dataset: `LIVE_DATASET_ROOT`
  (`/Users/aatmaj/Downloads/datasets/yolo_live_v2`) —
  `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`,
  `redraw_manifest.csv`, `.render_signature` sidecars.
* Per-epoch GT-vs-pred overlays + montages: `YOLO/outputs/epoch_XX/`,
  `YOLO/outputs/montage_epoch_XX.png`.
* Ultralytics curves copied to `YOLO/outputs/` (`results.png`, `PR_curve.png`,
  `F1_curve.png`, `confusion_matrix.png`, `labels.jpg`, `results.csv`).
* Telemetry CSV (dashboard): `YOLO/telemetry_logs/<run>.csv` with
  `epoch, train_box/cls/dfl, val_box/cls/dfl, precision, recall, mAP50,
  mAP50-95, lr`.
* Best weights: `YOLO/models/yolo_<run>_best.pt`.

## Running

```bash
./venv/bin/python YOLO/test_yolo_pipeline.py   # unit tests first
./train_YOLO.sh                                 # full run (all data)
./train_YOLO.sh prod
./run_scripts/predict_YOLO.sh debug --limit 20  # offline viewer
./run_scripts/start_yolo_server.sh              # dashboard on :8000
```

The trainer is cwd-independent: relative config paths (logs, telemetry,
outputs, checkpoints) are anchored to the repository root, so `python
YOLO/run.py` behaves the same from any directory.

To smoke-test the pipeline quickly, temporarily lower
`SOURCE_SAMPLE_CAP`/`VAL_SAMPLE_CAP`/`TEST_SAMPLE_CAP` and `NUM_EPOCHS` in
`YOLO/hyperparameters.toml` (e.g. 32/32/8 and 2), run `./train_YOLO.sh`, then
restore. The run writes to a fresh dataset root, so a capped run never
contaminates a full one's labels.

> **Mac note:** `train_YOLO.sh` and the viewers export
> `PYTORCH_ENABLE_MPS_FALLBACK=1` — Ultralytics runs NMS
> (`torchvision::nms`), which has no MPS kernel. Training stays on MPS; only
> that op falls back to CPU.

## Source dataset layout

Folder-per-sample, split by layout family (no family appears in two splits,
so there is no train/val leakage):

```
<split>/<hash>_original/            # or <hash>_augmented_<NNN>
    floorplan_segmentation.png      # condition mask (class ids, L mode)
    furniture_segmentation.png      # arrangement mask (class ids, L mode)
    ...                             # other renders ignored by this pipeline
```

Split folders and dataset-root color index JSONs are configured in
`YOLO/config.toml`.
