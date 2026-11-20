# evaluator — YOLO on the missing component the diffusion generates

Deliverable 4: **component-level evaluation**. The package answers one question
about a trained `mask_diffusion` run:

> If the generated component is accepted into the layout, can the detector that
> has to live with it see it, and where does it say it is?

It does that by rebuilding every test sample as **the real layout with the missing
component erased, plus the component the diffusion generated**, letting a trained
YOLO look at that image, and scoring **only the missing component**: the
detector's class-filtered metrics, the box loss against the dataset's own
ground-truth box, the overlap with the layout's other furniture, and the
generated channel's mask Dice/IoU.

The primary result is an **ensemble**: every sample is generated with
`DRAWS_PER_SAMPLE` independent chains, the chains' missing-component softmax
probabilities are averaged per pixel, and the average is thresholded once at
`ENSEMBLE_PROBABILITY_THRESHOLD`. The individual chains are still evaluated and
reported as diagnostics, so the spread and the ensemble can be compared in the
same run.

One **component selection** drives the whole run. `--component tv` (or
`showcase`, `dining_set`) resolves BOTH models from `components.toml` at the
repository root: the diffusion checkpoint trained for that component and the
`missing_yolo` detector trained on that same component. The per-component
detectors are single-class, so the missing component is class id `0`; the
multi-class `YOLO/` weights remain usable through `--weights`.

Nothing here is trained; only `evaluator/outputs`, `evaluator/telemetry_logs`
and `debug_logs/evaluator.log` are written to.

---

## 1. What it measures

| Layer | Question | Where it comes from |
|---|---|---|
| **Detection** | Can the detector see the generated component at all? | Ultralytics `model.val()` restricted to the component's class |
| **Localization (the box loss)** | If it sees it, where does it think it is? | The dataset's own `labels.txt` box vs the detector's box |
| **Overlap** | Did the generated component land on other furniture? | The detector's highest-confidence component box vs the other named `labels.txt` boxes |
| **Channel** | How much of the true component did the generation cover? | Dice/IoU of the generated channel vs the ground-truth mask channel |

### The box loss, precisely

For one (sample, draw):

* ground truth = **the dataset's `labels.txt` box**, never a mask region and
  never the diffusion output;
* prediction = the detector's box for that class on the composite image, at or
  above `YOLO_CONF`;
* pairing = greedy in confidence order, a prediction claims a ground-truth box
  once it clears `BOX_MATCH_IOU`, and no ground truth is claimed twice.

Reported per matched pair: **`box_iou`**, **`l1_norm_xywh`** (mean absolute
difference of `cx, cy, w, h`) and **`center_distance_px`**. Whatever is left
unpaired is counted as a **miss** (a ground-truth box nothing reached) or a
**false positive** (a box the detector invented where the dataset says there is
none) — which is what makes the *negative* samples worth evaluating.

### The overlap metric, precisely

For one (sample, draw), the **overlap** is the fraction of the detector's
highest-confidence component box that is covered by the union of every other
named furniture box in the same `labels.txt`:

```text
overlap = area(final component box ∩ union(other ground-truth boxes))
          ------------------------------------------------------------
                         area(final component box)
```

`0` is best (the component touches nothing else), `1` means it is completely
covered by existing furniture. When the detector returns no component box the
sample contributes no overlap value; when the layout has no other named
furniture the value is `0`. The union is computed exactly, so two other boxes
overlapping each other are not counted twice. The metric is reported per sample
in the telemetry CSV and summarized per draw, for the ensemble and in the
across-draw aggregate.

---

## 2. What the evaluated image is

For each sample, one 256x256 RGB image per result (each draw and the ensemble):

1. the floorplan, rendered from its class ids;
2. the **real furniture with the missing component remapped to background** —
   the same erase rule the partial VAE's layout applies, so no trace of the
   ground-truth component can leak into the image;
3. the **generated component** painted on top.

Both layers are blended at `COMPOSITE_OVERLAY_ALPHA` / `GENERATED_OVERLAY_ALPHA`
(default 0.6, the weight the detector was trained on), so an evaluated image
looks like the training domain instead of measuring the domain shift. The
generated component is layered last and wins every overlap, so a misplaced
component is never hidden behind the furniture it was placed against.

A per-draw component is the decoder's **argmax** equal to the component's
channel — the same rule the training grids use. The ensemble component is the
per-pixel mean of the draws' softmax probabilities for that channel, thresholded
at `ENSEMBLE_PROBABILITY_THRESHOLD` (default `0.5`). Both the per-draw and
ensemble composites are written, so a disagreement between the ensemble and a
single chain stays visible.

---

## 3. Running it

```bash
./run_eval.sh                 # smoke pass, SAMPLE_CAP (250) samples
./run_eval.sh prod            # same, prod logging level
python evaluator/run.py --cap -1 --draws 4   # the whole test split
```

Useful options (each overrides the matching TOML key):

| Flag | Overrides | Meaning |
|---|---|---|
| `--component NAME` | `MISSING_COMPONENT_CLASS_NAME` | Which component to judge (`tv`, `showcase`, `dining_set`, ...); selects its checkpoint **and** its `missing_yolo` detector from `components.toml` |
| `--checkpoint PATH` | selected checkpoint | Manual escape hatch: judge a diffusion run not listed in the registry |
| `--weights PATH` | selected detector | Manual escape hatch: judge a detector not listed in the registry |
| `--cap N` | `SAMPLE_CAP` | `-1` keeps every sample |
| `--draws N` | `DRAWS_PER_SAMPLE` | Independent diffusion chains per sample; also the ensemble size |
| `--seed N` | `SEED` | Base seed of the sampling and the visualization |
| `--skip-yolo` | – | Channel metrics and figures only, no detector call |
| `--output-dir PATH` | – | Where to put this run's artifacts |

`--component` is the normal way to switch: the selected row of `components.toml`
supplies both models, and the detector's own class names decide the class id
(per-component `missing_yolo` detectors have one class, id `0`). An explicit
`--checkpoint` or `--weights` still wins over the registry row, so an
unregistered run can be measured without editing a file.

### Batch comparison table

To evaluate every component over the full validation split and collect the
overlap metric into one table, run:

```bash
./run_scripts/run_validation_comparison.sh --skip-existing
```

The runner writes:

| Path | Content |
|---|---|
| `evaluator/outputs/validation_comparison/overlap_comparison.html` | Color-coded table for quick reading |
| `evaluator/outputs/validation_comparison/overlap_comparison.md` | The same tables in Markdown |
| `evaluator/outputs/validation_comparison/overlap_comparison.csv` | One machine-readable row per component |
| `evaluator/outputs/validation_comparison/overlap_by_draw.csv` | Per-draw overlap statistics |
| `evaluator/outputs/validation_comparison/runs.json` | Completed-run manifest used by `--skip-existing` |

The table is rebuilt after each component, so an interrupted batch still leaves
a usable partial table. Use `--table-only` to regenerate it without evaluating.

### Collision IoU post-processing

The evaluator summary already holds matched-box IoU. Collision IoU needs the raw
final target box, which telemetry does not store, so the post-processing script
re-runs each component's detector on the saved ensemble composites:

```bash
./venv/bin/python evaluator/analyze_collision_iou.py \
    --manifest evaluator/outputs/test_comparison/runs.json
```

It writes `collision_iou_pairs.csv` (one row per target-other furniture pair,
with the sample's `matched_boxes` and `best_iou` for filtering),
`collision_iou_samples.csv` (one row per sample), `collision_iou_summary.csv`,
`collision_iou_results.md`, and `collision_iou_table.tex` beside the manifest.

### Cost, measured on this machine (MPS)

A 1000-step reverse chain costs **~140 ms** at `GENERATION_CHUNK = 32`. The
default smoke pass (250 samples x 4 draws = 1000 chains plus 1250 detector
calls, including the ensemble) finishes in **~12 minutes**. The full test split
(10,043 x 4 = ~40k chains) is **~3.5 hours**, dominated by sampling and detector
validation.

`BLOCK_SIZE` (512) exists only to bound memory: one block's masks are in memory
at a time, so the full split runs with the same code path as the smoke pass.

---

## 4. Artifacts

Written under `evaluator/outputs/eval_<run_id>/`:

| Path | Content |
|---|---|
| `dataset_ensemble/{images,labels}/test` | The ensemble's dataset: averaged probability, thresholded once |
| `dataset_draw_<d>/{images,labels}/test` | One dataset per draw, so each draw's detector metrics are computed over that draw's own images only |
| `eval_composite_ensemble.png` | Contact sheet of the primary ensemble images |
| `eval_composite_draw_<d>.png` | Contact sheet of the evaluated per-draw images, green GT box, red prediction |
| `eval_best_ensemble.png` / `eval_worst_ensemble.png` | The ranked ends of the ensemble's box-IoU distribution |
| `eval_best_draw_<d>.png` / `eval_worst_draw_<d>.png` | The ranked ends of each draw's box-IoU distribution |
| `eval_compare.png` | The ensemble's five-panel strip of a fixed seeded subset: true layout, conditioned layout, true component, ensemble component, annotated ensemble composite |
| `evaluation_summary.json` | The whole report: the primary `ensemble` section, `per_draw` diagnostics and everything needed to reproduce the run |
| `evaluation_report.txt` | The same numbers as a human-readable table |

The same figures are also **published** to `evaluator/outputs/` as
`<figure_stem>_<run_id>.png`, because the dashboard serves one flat image
directory: that is how the `eval` panel finds them, and several runs' figures
coexist there the way the VAE stage's grids do.

Plus `evaluator/telemetry_logs/<run_id>.csv`, one row per (sample, result). A
result is a draw (`draw` = `0..N-1`) or the ensemble (`draw` = `-1`). Each row
also carries `generated_pixels` and `probability_mean`, so the effect of
averaging and thresholding is visible next to the per-draw numbers. The shared
dashboard discovers the file like any other run.

### Qualitative full-pipeline report

A separate, numbers-free view renders five shared validation samples for every
component in `components.toml`:

```bash
python evaluator/visualization_full_pipeline_report.py
# or choose a subset / another split:
python evaluator/visualization_full_pipeline_report.py \
    --components tv showcase --samples 5 --validation-dir /path/to/validate
```

Each component gets one four-column figure directly under
`evaluator/outputs/full_pipeline_report/`:

1. the conditioned floorplan (real furniture with the component erased);
2. the ensemble diffusion reconstruction (the averaged-probability component
   composited in);
3. the same reconstruction with YOLO boxes in red and `labels.txt` GT boxes in
   green;
4. the real layout, component included.

The five samples come from a seeded shuffle of the validation split and are
shared by every component, so rows stay comparable. `report.json` records the
chosen stems and per-row GT/prediction counts; the individual panels are kept
under `panels/<component>/<stem>/`.

To watch a run's figures next to the training dashboards:

```bash
./run_scripts/start_evaluator_server.sh      # http://127.0.0.1:8086
```

---

## 5. Configuration

`config.toml` holds paths, the default component and the device;
`hyperparameters.toml` holds every number, including `DRAWS_PER_SAMPLE`
(ensemble size) and `ENSEMBLE_PROBABILITY_THRESHOLD` (inclusive threshold on the
averaged missing-component probability). The **component -> models** pairing is
declared once, in `components.toml` at the repository root, so the evaluator and
the frontend can never disagree about which detector belongs to which generator:

```toml
[default.COMPONENT_VARIANTS.tv]
mask_diffusion_checkpoint = "./mask_diffusion/models/mask_diffusion_amused-corgi_epoch__no_tv_700.pt"
detector_weights = "./missing_yolo/models/tv_best.pt"
```

Three things are deliberately **not** declared by a stage:

* the **frozen VAE paths**, which are read from the selected diffusion checkpoint
  itself, so an evaluation cannot sample with a VAE the run never saw;
* the **detector's class names**, which are read from the selected weights: a
  `missing_yolo` detector has one class (id `0`) and the multi-class `YOLO/`
  weights have ten, and only the weights know which id means what;
* the **component aliases** (`dining_set` -> `Dining_table`), which still come
  from `YOLO/hyperparameters.toml`, because the mask and dataset vocabularies
  spell some components differently.

`COMPONENT_LABEL_NAMES` maps a mask component name to the names the dataset's own
`labels.txt` uses for it. This is required because the two vocabularies differ
(`dining_set` in the mask, `Dining_table_big/medium/round` in the labels), and a
name that resolves to nothing would silently score zero boxes instead of failing.
Every mask class is listed, so `--component <name>` needs no edit either: point
the config at a run trained for that component and the label names come along.

---

## 6. Package layout

| File | Responsibility |
|---|---|
| `run.py` | CLI, config/hyperparams loading, component selection, path anchoring, `PYTORCH_ENABLE_MPS_FALLBACK` |
| `variants.py` | Reads the shared `components.toml` and applies one component's models to a config |
| `settings.py` | Path anchoring + reading the detector aliases from the YOLO stage |
| `ontology.py` | One component resolved across mask / detector / label vocabularies |
| `dataset_gt.py` | The dataset's own `labels.txt` boxes |
| `compositor.py` | The erased ground truth plus the generated component composite |
| `yolo_eval.py` | Eval dataset, detector validation, per-image predictions |
| `box_metrics.py` | Box matching, IoU, L1, center distance, distributions |
| `channel_metrics.py` | Mask Dice/IoU of the generated channel |
| `results.py` | Per-draw accumulators, per-sample records, across-draw aggregation |
| `visualize.py` | Contact sheets, comparison strips, ranked panels |
| `telemetry.py` | Per-sample CSV and the summary JSON |
| `report.py` | The human-readable table, as a pure function of the summary |
| `pipeline.py` | The streaming loop that ties it all together |
| `visualization_full_pipeline_report.py` | Five shared validation samples x every component, four-column qualitative figure |
| `test_evaluator.py` | Unit tests of the pure logic (`./venv/bin/python evaluator/test_evaluator.py`) |

`mask_diffusion/inference.py` is the shared inference API: it loads a
self-describing checkpoint and exposes encode / normalize / generate. It has no
writes and imports nothing from the training runner's state, so using it cannot
perturb a training run.

---

## 7. Reproducibility

`SEED` seeds every sampler generator; draw `d` uses
`SEED + DRAW_SEED_STRIDE * d` and the chunk holding absolute chain `k` of the
split adds `CHUNK_SEED_STRIDE * (chain_offset + k // GENERATION_CHUNK)`, so
`--seed` fully determines the pixels and the ensemble is deterministic.
`GENERATION_CHUNK` and `BLOCK_SIZE` fix
how the RNG stream is cut up, so keep them fixed when comparing two runs (they
change the noise stream, not its distribution) — the same policy the YOLO stage's
live redraw uses.

## 8. Leakage guards

* The composite is built from the class ids with the component's channel folded
  into the background — the erase rule the conditioning encoder itself applies.
  A unit test asserts exactly one channel stays active per pixel after the erase.
* The ground-truth box is read from the dataset's label file, and the test
  asserts a line naming any other class is ignored.
* The evaluated dataset is written to `evaluator/outputs/...`, never into the live
  dataset the detector was trained on.
* The label files carry only the component's class, so every filtered metric is
  attributable to exactly one component.
* The detector's class vocabulary is read from the selected weights and the
  component is resolved in that vocabulary before anything is evaluated, instead
  of trusting a file name or another stage's class list.
* A selected component's checkpoint and weights are checked for existence before
  any sampling starts, so a wrong registry row cannot waste a diffusion pass.
