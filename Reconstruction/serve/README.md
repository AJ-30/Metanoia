# serve: web demo of the wall / door / window completion model

Upload a floor plan, and the trained diffusion model redraws its missing
component on this machine's GPU. Three modes:

| Mode | Model input | Scored against |
|---|---|---|
| Complete my plan | the plan as uploaded | nothing (no ground truth) |
| Remove and restore | the plan minus one component, removed with the dataset's rules (easy / hard, margin) | the removed component (`evaluation/` metrics) |
| Erase and redraw | the plan with a freehand eraser mask cleared in every layer | what was erased (component matches, pixel IoU, validity checks) |

In erase mode the additions are by default kept inside the erased area plus a
`[erase] KEEP_RIM_PX` rim, so an edit never changes the rest of the plan.

This is the standalone page of method 1. In the combined demo (`./run_frontend.sh`, see the repository README) the same
app is mounted inside the furniture demo and its result is handed on to the furniture models.

## Running (from the repository root)

```bash
pip install -r Reconstruction/requirements.txt      # once, in an environment with CUDA PyTorch
python -m Reconstruction.serve.app --mode prod      # http://127.0.0.1:8090
```

Requirements on disk: `Reconstruction/data_prep/`, the processed dataset
(`datasets/processed/resplan`, including `test/structure.npy` and
`test/room.npy` for the example plans) and the checkpoint named in
`config.toml` (`runs/<RUN_NAME>/checkpoints/`).

## Inputs

| Upload | Format |
|---|---|
| Plan JSON / GeoJSON | One object keyed by ResPlan categories (`wall`, `door`, `front_door`, `window`, `living`, `bedroom`, `bathroom`, `kitchen`, `storage`, `balcony`, `stair`), values as WKT, GeoJSON geometry/feature or coordinate rings. Optional `area` (m²) + `inner` or `metres_per_unit` for metric scale. Rasterized exactly like the dataset build. |
| Image (PNG/JPEG/WebP) | Plan drawn in the palette of `data_prep/params.toml [colors]` on white; padded to square and resized to 256² (nearest). |
| Array (`.npy`/`.npz`) | (13, 256, 256) multi-hot map in the taxonomy's channel order; loaded with `allow_pickle=False`. |

ResPlan itself is a pickle and is **not** accepted. Convert plans locally:

```bash
python -m Reconstruction.serve.convert --ids 1 59 --out Reconstruction/outputs/serve_uploads
```

writes `<id>.json`, `<id>.png` and `<id>.npy` for each plan (ready to upload).

## Model profiles: ResPlan and FurnitureArranger

The page has a **Model** selector when more than one profile is available. `resplan` (the 121M-parameter ResPlan
model, configured by `config.toml`) is the default and behaves exactly as before. `profiles.toml` adds:

| Profile | Trained on | Restores | Sees furniture |
|---|---|---|---|
| `furniture_shell` | FurnitureArranger living rooms (shell only, 13 channels) | walls, doors, windows | no, furniture is not drawn |
| `furniture_walls` | FurnitureArranger living rooms (22 channels) | walls, doors, windows | yes, as context |

Both are 33.5M-parameter models (step 18,000). Test-set detection (IoU >= 0.5) of the removed component:
`furniture_shell` wall 0.74, door 0.71, window 0.41; `furniture_walls` wall 0.60, door 0.73, window 0.53.

* **Uploads.** Besides images and arrays, these profiles take the dataset's own layout: an `.svg` from
  `FurnitureArranger/accepted/` (only its embedded `<metadata>` JSON is read, the SVG is never parsed as XML) or a JSON
  with `floorplan` (WKT per category) and `arrangement.furniture`. Plans are drawn at the dataset's fixed scale, one plan
  unit per pixel, so a plan must be at most about 240 units across; the metric scale (0.03 m per unit) is an assumption.
* **What the models restore.** "Remove and restore" refuses furniture (the models were never trained to redraw it) and the
  eraser never clears furniture. All three modes, K-sample voting and the clean-up pipelines work as for ResPlan.
* **Loading.** A profile's model loads on its first use; the ResPlan model loads at start-up. A profile whose files are
  missing is left out of the selector (a warning in the log), so a checkout without the weights still serves ResPlan.
* **API.** `GET /api/profiles` lists the profiles. `info`, `plan`, `example` and `predict` accept an optional `profile`
  (query, form field or JSON field) that defaults to `resplan`, so older clients keep working.
* **Weights and examples are in git** (unlike the ResPlan checkpoint): `Reconstruction/serve/weights/*.pt` are serve-only
  float16 EMA checkpoints (67 MB each) and `datasets/processed/furniture*/examples/` hold 40 test plans each, next to the
  committed `params.toml` / `size_bins.json`. `python -m Reconstruction.serve.export_assets` rebuilds them from a training
  run (see its docstring). To serve another model, add a table to `profiles.toml`.

## What a request does

Exactly the offline pipeline, for one plan: `data_prep` rasterization /
`MissingComponentGenerator.build_case` -> `model.sampling.sample_missing`
(EMA weights, DDIM) -> `postprocess` vote fusion of K samples + chosen
clean-up pipeline -> `evaluation.labels.compose` (input is never erased) ->
`evaluation.case_metrics` when there is a ground truth.

## Safety limits

| Where | Limit |
|---|---|
| app | eraser masks must be canvas-sized and clear at most `[erase] MAX_FRACTION` of it; no pickles; vertex, pixel and archive-size caps (`params.toml [upload]`); one GPU job at a time, 3 queued, then HTTP 503; checkpoint loaded with `weights_only=True` |
| data | uploads are parsed in memory and kept for at most an hour in a 64-plan cache; nothing is written to disk |

## Files

| File | Purpose |
|---|---|
| `engine.py` | Upload parsing, model loading, inference, rendering (one engine per profile) |
| `app.py` | FastAPI routes, profile registry and GPU queue |
| `profiles.toml` | The FurnitureArranger profiles (label, dataset, checkpoint, restorable types) |
| `export_assets.py` | Training run -> serve-only checkpoint, dataset parameters and example plans |
| `weights/` | Serve-only checkpoints of the profiles in `profiles.toml` |
| `static/index.html` | The interface |
| `convert.py` | ResPlan pickle -> JSON / PNG / NPY uploads |
| `config.toml` / `params.toml` | Paths and checkpoint / upload, inference and cache parameters |
