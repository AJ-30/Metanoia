# missing_yolo — a detector for the **missing component** of a generated layout

This stage trains Ultralytics YOLO26 to localise **exactly one** furniture
component in images produced by the frozen `mask_diffusion` generator. One run
targets one component; three runs give three detectors (`tv`, `showcase`,
`dining_set`), each with its own dataset tree, telemetry and weights.

```
 floorplan_segmentation.png ─┐
                             ├─► frozen mask_diffusion ─► latent ─► frozen decoder ─┐
 furniture_segmentation.png ─┘   (component erased by the                            │
                                  partial VAE layout)                                ▼
                                                                    floorplan render
 labels.txt ── component boxes ───────────────────────────►  image + detection target
      │                                                        (YOLO26 training pair)
      └── never a box derived from generated pixels
```

**The image** is the dataset's floorplan render with the **generated** (full,
component-included) arrangement blended over it — the same overlay style the
mask_diffusion and YOLO grids use.

**The target** is the dataset's own `labels.txt` box for that component, and
nothing else. No box is ever derived from the generated pixels: a target defined
by the same model whose output it would grade is not a ground truth.

---

## 1. Why the labels do not move (and why that is safe here)

`YOLO/` trains on boxes derived from connected components of each epoch's draw,
so its labels change every epoch. Ultralytics reads label files **once**, when
the train dataloader is built, and only re-reads images — which is why `YOLO/`
must push every epoch's boxes into `trainer.train_loader.dataset.labels`
in place, and why it pins `WORKERS = 0`.

This stage never faces that problem, and the reason is worth stating plainly:

* the target of a sample is a fixed function of its `labels.txt`, which is fixed;
* the only thing that changes per epoch is the **pixel**, from a fresh seed.

So the once-read labels are correct for every epoch by construction. There is no
in-memory patching, no second label source and no drift between the box the
model is scored against and the box the dataset states. `WORKERS = 0` is still
pinned here, but for the ordinary reason (worker processes would keep serving
forked file handles for the previous epoch's PNGs), not for label syncing.

### Label noise is kept, not hidden

The generator places the component where it thinks it belongs, which is not
always where `labels.txt` says it is. Per the project's decision, such samples
are **kept and labelled with the `labels.txt` box anyway** — no filtering, no
relabelling, no dropping.

That choice is only safe if its cost is visible, so every generation pass
measures it and reports it:

| metric | meaning |
|---|---|
| `agreement_mean` | mean box-IoU between the generated component mask and the `labels.txt` box, over positive samples |
| `agreement_frac_<t>` | fraction of those samples with IoU ≥ `t` (`AGREEMENT_BOX_IOU_THRESHOLDS`) |
| `agreement_scored` | how many samples could be scored (positives the generator drew something for) |

These land in the telemetry CSV (`row_kind = generation`) and in
`redraw_manifest.csv`. A run whose agreement collapses is a run injecting label
noise, and these columns are how you see it before reading the mAP.

**Measured** (agreement mean = box-IoU between the generated component and the
`labels.txt` box, over positive samples of a capped run on an M1 Pro; the full
1,000-step chain):

| component | generator | train agreement mean |
|---|---|---|
| `tv` | `mask_diffusion_amused-corgi_epoch__no_tv_700.pt` | **0.92** (2,048 samples) |
| `dining_set` | `mask_diffusion_brief-shark_epoch_710_no_dining.pt` | high (see a run's manifest) |
| `showcase` | `mask_diffusion_kind-worm_epoch_no_showcase_1010.pt` | **0.67** — the weakest generator, matching its own held-out mask IoU of ≈ 0.30 |

Expect `showcase` to be the run whose mAP is most suppressed by label noise.

---

## 2. Splits and leakage

| split | source | generator saw it? | notes |
|---|---|---|---|
| train | `work/train` (~69,776) | **yes** | the generator was trained here, so train metrics are generation metrics, which is expected |
| val | ~50 % of `work/validate` (~9.9k) | no | early stopping and every per-epoch overlay |
| test | the other ~50 % of `work/validate` (~9.9k) | no | final report only, never early-stopped on |
| — | `work/test` | **yes** | deliberately unused: the generator trained on it, so it could not give an honest number |

The carve is **by layout family** (`<hash>` with `_original` / `_augmented_NNN`
stripped) and seeded by `BASE_SEED`, so augmented siblings of one layout can
never straddle the val/test boundary and re-running carves the same two halves.
Both halves are logged with their sample counts at startup.

---

## 3. Configuration

### `config.toml` — paths and one row per component

```
[default.COMPONENT_VARIANTS.tv]
mask_diffusion_checkpoint = "./mask_diffusion/models/mask_diffusion_amused-corgi_epoch__no_tv_700.pt"
detector_class_name       = "TV"
label_names               = ["TV"]
```

`dining_set` folds the dataset's three sizes
(`Dining_table_big` / `_medium` / `_round`) into one class, because the mask
vocabulary has a single `dining_set` channel.

Deliberately **not** configured: the three mask VAE paths, the missing channel,
the architecture and the latent statistics. They are read from the generator
checkpoint, so retargeting a component can never drift onto the wrong VAEs.

### `hyperparameters.toml` — every number

`MISSING_COMPONENT = "tv"` selects the variant row. `--component` on the command
line overrides it, so the three runs need no file edits. Grouped sections:
reproducibility (`BASE_SEED`, `VAL_RENDER_SEED`, `TEST_RENDER_SEED`, `CHAIN_SEED_STRIDE`),
splits (`VAL_FAMILY_FRACTION`, the three sample caps), generation
(`RENDER_IMAGE_SIZE`, `RENDER_OVERLAY_ALPHA`, `RENDER_CHAIN_CHUNK`,
`RENDER_DECODE_CHUNK`, `RENDER_IO_WORKERS`, `RENDER_USE_MU_ONLY`),
agreement telemetry (`AGREEMENT_BOX_IOU_THRESHOLDS`), the YOLO26 finetune block,
epoch-end overlays, and the Ultralytics augmentation overrides.

---

## 4. Files

```
missing_yolo/
├── __init__.py              package overview
├── run.py                   CLI entry: python missing_yolo/run.py --mode debug|prod [--component X]
├── runner.py                MissingYoloRunner: wiring, validation, epoch callback, telemetry, reports
├── dataset.py               FrozenMaskDiffusionStack (frozen generator) + ArrangementGenerator (writes splits)
├── labels.py                labels.txt -> target lines; component/labels.txt agreement telemetry
├── test_missing_yolo.py     AAA unit tests, no checkpoints or GPU needed
├── config.toml              paths + [COMPONENT_VARIANTS] table
├── hyperparameters.toml     every tunable number
├── models/ outputs/ telemetry_logs/ old_outputs/ old_telemetry_logs/
```

Reused rather than duplicated: `conditional_diffusion.runner.build_frozen_mask_vae`
/ `overlay_furniture_on_floorplan` / `resolve_color_index_file`,
`mask_VAE.masking` (layouts, encoding, rendering),
`mask_diffusion.model.MaskComponentDiffusion`, `evaluator.dataset_gt` (`Box`,
`read_component_boxes`), `YOLO.dataset` (`SampleFolder`, `list_sample_folders`,
`load_mask_ids_batch`, `clear_generated_split`, `write_data_yaml`,
`split_is_current`, `write_render_signature`), `YOLO.overlay`
(`choose_fixed_stems`, `save_epoch_overlays`), and `YOLO.runner`
(`map_results_row`, `read_results_csv`, `resolve_overlay_model`,
`resolve_train_device`, `validate_training_budget`, `ULTRALYTICS_ARTIFACTS`).
No new third-party dependency: `ultralytics` is imported lazily inside `train()`.

`mask_diffusion/inference.py` is deliberately **not** imported — it is the shared
evaluation API, and no training runner may depend on it. The frozen stack is
built from the training package's model plus `build_frozen_mask_vae`, following
the `YOLO/` precedent.

---

## 5. Per-epoch behaviour

1. `results.csv` gains a row → a `row_kind = detection` telemetry row is written
   **before** the long redraw, so the dashboard curves grow during the run.
2. A fixed seeded subset of val images is re-predicted with that epoch's EMA
   weights: green GT + red predictions per image plus
   `outputs/montage_epoch_XX.png`.
3. Every train image is regenerated from a fresh draw (`seed = BASE_SEED + E`)
   and a `row_kind = generation` telemetry row records the draw's provenance and
   agreement. Skipped on the final/early-stopped epoch.

Val and test are drawn **once**, each with its own fixed seed, and skipped
entirely when their `.render_signature` sidecar still matches (same stems,
generator file size/mtime, component, overlay alpha, image size, seed). Val and
test pixels therefore never move mid-run, so early stopping and the final report
are stable.

---

## 6. Telemetry

One CSV per run (`missing_yolo/telemetry_logs/<run_id>.csv`) with two row kinds,
distinguished by `row_kind`, so each column has real gaps rather than invented
values (unmeasured cells are **empty**, never `0.0`):

* `detection` — `train_*`, `val_*`, `precision`, `recall`, `mAP50`, `mAP50_95`, `lr`;
* `generation` — `draw_index`, `draw_seed`, `draw_images`, `draw_seconds`,
  `agreement_scored`, `agreement_mean`, `agreement_frac_*`.

A generation row's `epoch` is the epoch its pixels will **train**: draw index `d`
→ epoch `d + 1`, so the rows read in chronological order on one axis.

Dashboard: `./run_scripts/start_missing_yolo_server.sh [port]` (default 8088) —
same server, same figure names, no server change needed.

---

## 7. Running

```bash
./train_missing_yolo.sh debug              # MISSING_COMPONENT from the TOML
./train_missing_yolo.sh prod   showcase    # force the showcase run
./train_missing_yolo.sh prod   dining_set
```

or directly:

```bash
python missing_yolo/run.py --mode prod --component tv
```

Start the dashboard first (`./run_scripts/start_missing_yolo_server.sh`) to watch
metrics as they are written. Every run archives the previous run's
`outputs/` and `telemetry_logs/` into `old_*` before starting.

Tests (no checkpoints, no dataset, no GPU):

```bash
./venv/bin/python missing_yolo/test_missing_yolo.py     # or: -m pytest
```

---

## 8. Measured costs and knobs

Full-pool numbers on an M1 Pro, from the generation pass benchmark
(2,048 samples: sample + decode + render + PNG + agreement):

| operation | measured | projected for 69,776 train samples |
|---|---|---|
| reverse chain (1,000 steps) | 5.4 ms/chain | ~6 min |
| full pass incl. decode/render/write | 24.9 ms/sample | **~29 min/epoch** |
| condition encoding (once per run) | ~1.1 ms/sample | ~80 s per split |

So a 30-epoch full run is roughly **19–20 h per component** (~60 h for all
three), dominated by generation. `SOURCE_SAMPLE_CAP` and `NUM_EPOCHS` are the
dials: a capped debug run (a few hundred samples, 2 epochs) exercises the whole
pipeline in minutes.

Memory bound: one decode chunk's logits are `RENDER_DECODE_CHUNK × 11 × 256 ×
256` float32 (~1.5 GB at 512), consumed immediately by the renderer. Lower
`RENDER_DECODE_CHUNK` first if MPS memory is tight; lower `RENDER_CHAIN_CHUNK`
if sampling is.

---

## 9. Failure-fast rules

The run aborts before the expensive work rather than training on a silently
wrong setup:

* unknown `--component` / `MISSING_COMPONENT` → `KeyError` listing the valid keys;
* incomplete variant row, missing checkpoint file, empty class name;
* generator checkpoint predating the current format (missing architecture keys,
  statistics or recorded VAE paths);
* the resolved component absent from the target layout, or its recorded channel
  disagreeing with the name lookup;
* `RENDER_IMAGE_SIZE` ≠ the frozen VAEs' `IMAGE_SIZE`, or latent widths
  disagreeing with the generator's `LATENT_DIM`/`COND_DIM`;
* a target VAE that **excludes** the component (it could never draw it), or a
  partial VAE that erases nothing / something other than the component (the task
  would silently become unconditional generation);
* any sample missing its `labels.txt` (it would silently become a background);
* val/test overlap after the carve; duplicate stems; a threshold outside 0..1;
* train set too small for the configured gradient accumulation (reused
  `validate_training_budget`).

## 10. Known limitations

* Agreement is measured per epoch but **not acted on**. A component whose
  generator disagrees with `labels.txt` badly (notably `showcase`) trains on a
  visibly misplaced target; the telemetry quantifies the cost, it does not remove it.
* Val/test are carved from one split, so both halves share the dataset's
  augmentation policy and its `work/validate` provenance.
* One component per run: a single detector covering all three is not this stage's
  job, and the class id is therefore always 0.
* `LIVE_DATASET_BASE_ROOT` must be on fast local storage; the dashboard and
  Ultralytics both warn about slow reads otherwise.