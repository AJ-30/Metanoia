# data_prep: floor-plan datasets and missing-component generation

Milestones 1 and 2: turns the ResPlan and CubiCasa5K releases into rasterized
plans with train/val/test splits, generates missing-component cases and
visualizes them. The original releases are only read, never modified or
extracted. Everything is written to `PROCESSED_ROOT/<dataset>/` (see
`config.toml`).

Both datasets share one taxonomy and one output format, so they can be used
separately (e.g. train on ResPlan, test on CubiCasa) or mixed in training.

## Files

| File | Purpose |
|---|---|
| `config.toml` | Run config: input paths, output root, debug/prod subset size, seed, workers |
| `params.toml` | Canvas, taxonomy + dataset class maps, colours, wall splitting, case generation, dedup |
| `schema.py` | Taxonomy object (classes, channels, component types, colours) |
| `geometry.py` | `PlanGeometry`: dataset-neutral vector plan (one floor) |
| `resplan_source.py` | Read-only ResPlan loading, official splits, geometry extraction |
| `cubicasa_source.py` | Read-only CubiCasa5K loading (straight from the zip), SVG parsing per floor |
| `sources.py` | Dataset adapters used by the build script (add new datasets here) |
| `rasterize.py` | Vector plan to two uint8 instance layers + instance metadata |
| `wall_split.py` | Splits merged walls into segments at skeleton junctions |
| `plan_store.py` | Writes/reads a split (memory-mapped arrays + JSONL records) |
| `dedup.py` | Near-duplicate search between train and val/test |
| `missing.py` | Case specs, removal rules, case construction, metadata |
| `case_source.py` | Framework-free access to the cases of one split |
| `dataset.py` | PyTorch `Dataset` wrapper (+ rotation/flip augmentation) |
| `render.py` | Colour rendering and comparison panels (reused later for predictions) |
| `build_dataset.py` | CLI: runs the whole build for one dataset |
| `visualize.py` | CLI: before/after comparison grids |

## Usage (from the repository root)

```bash
python -m Reconstruction.data_prep.build_dataset --dataset resplan  --mode prod   # ~6 min
python -m Reconstruction.data_prep.build_dataset --dataset cubicasa --mode prod   # ~3 min
# --mode debug: 200 items per split into datasets/processed_debug/
# --overwrite:  rebuild an existing output of this script

python -m Reconstruction.data_prep.visualize --dataset cubicasa --mode prod --split test
```

## Pipeline (`build_dataset.py`)

1. **Load** the release and its official splits (read-only).
   * ResPlan: `ResPlan.pkl` + `split.json`. The 683 `augmented` plans
     (rotated/flipped copies of other plans) are left out of train by
     default, because their originals may sit in val/test.
   * CubiCasa5K: `train/val/test.txt` (4200/400/400 houses) and each house's
     `model.svg`, read from `cubicasa5k.zip`. Splits are by house, so all
     floors of a house stay in one split.
2. **Convert** each item to `PlanGeometry` (one per plan / floor) and
   **rasterize** it onto a 256x256 canvas, fitted with 8 px padding, aspect
   preserved, north up. `metres_per_px` is stored per plan.
3. **Split walls** into segments: skeletonize, cut at T/X junctions, drop
   short spurs, assign each wall pixel to its nearest branch. Walls never
   include openings, so segments also end at doors/windows.
4. **Size bins**: per component type, area tertiles on **train only**
   (`size_bins.json`), reused for val/test.
5. **Near-duplicates**: val/test plans whose 64x64 label thumbnail agrees
   with a train plan on at least 95% of occupied pixels are excluded from the
   frozen cases (`near_duplicates.json`).
6. **Freeze val/test cases** (`cases_<split>.jsonl`): for every plan and
   every component type present, one instance, removed once in easy mode and
   once in hard mode at margin 0. For a seeded 25% of plans, the easy case is
   also stored at every margin (4, 8, 16 px) for the missing-region-size sweep.
   Each line holds the spec plus its metadata.
7. **Summary** (`summary.json`, also logged) and conversion problems
   (`problems.json`: errors, skipped floors, suspicious scales).

## Dataset specifics

| | ResPlan | CubiCasa5K |
|---|---|---|
| Unit of a plan | one plan | one floor of a house (1-3 floors per SVG) |
| Scale | coordinates normalized per plan; metres from `area` / footprint | 100 SVG units per metre (verified on metric labels: p5-p95 99.9-100.3) |
| Openings vs walls | walls already stop at openings | openings lie inside walls; cut out of the walls on load |
| Class mapping | `[taxonomy.resplan_class_map]` | `[taxonomy.cubicasa_class_map]` (elements) and `[taxonomy.cubicasa_space_map]` (room types, by first type token; unknown types -> `other_room`) |
| Skipped | nothing | floors with < 4 walls or no room (e.g. empty attics); furniture/labels ignored |

CubiCasa floors whose median door width falls outside 0.5-1.2 m are kept but
listed as "suspicious scale" in `problems.json` (about 1%, mostly garage
doors); exclude them from metric (metre) error statistics if needed.

## Representation

Each plan has two instance layers (uint8, 0 = empty):

* `structure`: wall segments, railings, columns, windows, doors (drawn on top).
* `room`: living, bedroom, bathroom, kitchen, storage, balcony, other_room, stair.

A case is expressed as multi-hot semantic maps with C = 13 channels,
`[wall, railing, column, window, door, living, bedroom, bathroom, kitchen,
storage, balcony, other_room, stair]`. Classes a dataset lacks stay empty
(ResPlan has no railings, columns or `other_room`).

| Array | Shape | Meaning |
|---|---|---|
| `full` (F) | (C, 256, 256) | complete plan |
| `incomplete` (I) | (C, 256, 256) | plan with the component removed: **the only model input at test time** |
| `target` (R) | (C, 256, 256) | what is missing, `F AND NOT I` |
| `mask` (M) | (256, 256) | erased pixels (for oracle baselines and metrics only) |
| `primary` | (256, 256) | pixels of the removed (primary) instance (metrics only) |

## Component types and removal modes

| Type | Classes | Easy | Hard |
|---|---|---|---|
| wall | wall segment | erase the segment | also erase doors/windows in it |
| door | door, front door | erase it (gap stays in the wall) | fill the gap with wall (plan looks complete) |
| window | window | erase it (gap stays) | fill the gap with wall |
| room | living ... other_room | erase the room label (walls stay) | also erase walls/openings touching no other room |
| other | stair, railing, column | erase it | (not used) |

A margin m > 0 additionally erases everything within m px of the removed
pixels.

In hard door/window cases the model must draw the opening over existing wall
pixels; when composing the final plan, openings take priority over walls.

Training cases are drawn on the fly: component type uniform over the types
present in the plan (balances rare types), instance uniform, hard mode with
probability 0.3, margin 0/4/8/16 with probabilities 0.7/0.1/0.1/0.1. All of
these are set in `params.toml`. Randomness is keyed on (seed, epoch, index),
so runs are reproducible for any number of data-loader workers.

## Case metadata (for robustness analysis)

`case_id`, `plan_id`, `comp_type`, `primary_cls`, `mode`, `margin_px`,
`removed` (all erased instances), `primary_area_px`, `target_area_px`,
`size_bin` (small/medium/large), `location` (exterior if within 6 px of the
outside of the building, else interior), `num_neighbours` (instances within
2 px), `centre_offset` (distance from plan centre / canvas size),
`metres_per_px`.

## Visualization

```bash
python -m Reconstruction.data_prep.visualize --dataset resplan --mode prod --split test
python -m Reconstruction.data_prep.visualize --dataset cubicasa --mode prod --split train --types door window --rows 8
```

Each row shows: **Before (complete) | After removal (model input) | Missing
component** (highlighted over a faded plan) **| Erased region**. The row label
gives plan id, class, mode, margin, size bin, location and number of removed
instances. Grids go to `Reconstruction/outputs/data_prep/<dataset>/<split>/`.

## Environment note

In the `metanoia` conda env, importing `cv2` *after* `torch` fails
(`GCC_12.0.0 not found`: torch loads an older system `libgcc_s`). The dataset
path (`dataset.py`, `case_source.py`, `plan_store.py`, `missing.py`) never
imports OpenCV. If a script needs both, import `cv2` before `torch`.

## Known limitations

* Scale varies per plan (about 0.03-0.11 m/px) because each plan is fitted to
  the canvas; metric errors must be converted with `metres_per_px`.
* L-shaped (and C-shaped) wall runs have no junction and stay one segment.
* Some wall segments are short stubs between an opening and a corner; they
  fall in the "small" size bin.
* Near-duplicate search compares plans in the same orientation only.
* ResPlan has few "other" components (stairs in about 4% of plans); CubiCasa
  adds railings, columns and more stairs.
* The CubiCasa room mapping (`cubicasa_space_map`) is a judgement call; e.g.
  halls/entries map to `living` because ResPlan has no corridor class.
