# evaluation: scoring missing-component reconstructions

Milestone 3. Scores any predictor on the frozen val/test cases of a processed
dataset, broken down the way the robustness study needs (component type,
mode, missing-region size, component size, location, hard-case clue).

## Usage (from the repository root)

```bash
# Built-in baselines
python -m Reconstruction.evaluation.evaluate --dataset resplan  --predictor rules  --mode prod
python -m Reconstruction.evaluation.evaluate --dataset cubicasa --predictor oracle --mode prod

# A trained model: one <case_id>.npz per case with `samples` (K, C, H, W), scores in [0, 1]
python -m Reconstruction.evaluation.evaluate --dataset resplan --predictions runs/v1/preds --name diffusion_v1 --mode prod
```

`--mode debug` evaluates the first 400 cases only. `--split val` for model
selection; keep `test` for final numbers.

Outputs in `Reconstruction/outputs/evaluation/<dataset>/<split>/<name>/`:
`cases.csv` (one row per case, all metrics), `summary.json` (all tables),
`summary.md` (headline columns; the `by_type_mode` table splits easy and
hard cases), `examples_easy.png` / `examples_hard.png` (Input | Prediction |
Composed plan | Ground truth, one grid per removal mode).

## How a prediction is scored

1. **Decode**: channels above 0.5 are "on". Structure pixels take the
   highest-priority class (door > window > column > railing > wall); room
   pixels take the highest-scoring room class.
2. **Compose** with the input: a predicted structure class replaces the input
   only if it has higher priority (so a door can be drawn over the wall of a
   hard case, but existing openings are never erased); predicted rooms only
   fill unlabelled pixels. The input is therefore always preserved.
3. **Added parts** = composed plan minus input. The ground truth is treated
   the same way (complete plan minus input), so both sides are comparable.
4. **Components** = connected pieces per class (>= 4 px), matched one-to-one
   within the same class by maximum IoU (Hungarian).

## Metrics

| Metric | Meaning |
|---|---|
| `precision/recall/f1@t` | Component detection at IoU >= t (0.5 headline; also 0.25, 0.75). Micro-averaged over cases. |
| `primary_detection` | Share of cases where the removed component itself is found at IoU >= 0.5 |
| `bestk_primary_detection` | Same, best of K samples (with K > 1 predictions) |
| `primary_iou` | Mean IoU of the removed component with its matched prediction |
| `type_accuracy` | Majority predicted class on the removed component's pixels has the right type |
| `connection_accuracy` | Doors/windows: the prediction links the same rooms (and exterior) as the truth, even if shifted along the wall |
| `loc_error_m_median` | Centroid distance (m) between the removed component and its prediction |
| `pixel_iou` | Class-aware IoU of all added pixels |
| `shell_exact` | Composed plan has exactly as many connected shell pieces (walls + openings + railings + columns) as the truth |
| `openings_on_wall` | Doors/windows touching a wall |
| `rooms_enclosed` | Rooms (except balconies) not open to the outside |
| `rooms_reachable` | Rooms in the largest group linked by doors or open boundaries |

The last three are ground-truth-free validity checks on the composed plan;
compare them with the oracle row (their value on the true plans).

**Hard-case clue**: for hard door/window cases, `clue` says whether the
removal left a room without any door (window). The table
`hard_openings_by_clue` reports these two groups separately.

## Baselines

| Name | What it does |
|---|---|
| `oracle` | Returns the ground truth; must score 1.0 on every detection metric (sanity check). |
| `empty` | Predicts nothing; the lower bound. |
| `rules` | Hand-written: closes wall-thick, partly unlabelled gaps in the shell (doors if <= 1.6 m and interior, windows if facing outside, walls if longer) and labels enclosed unlabelled areas as bedrooms. Cannot solve hard cases by design. |

## Files

| File | Purpose |
|---|---|
| `labels.py` | Label maps, decoding, composition, added parts |
| `components.py` | Component extraction and matching |
| `structure_checks.py` | Validity checks, clue and connection helpers |
| `case_metrics.py` | Per-case metrics (prediction-independent context computed once) |
| `baselines.py` | Built-in predictors and the saved-prediction loader |
| `report.py` | Aggregation into tables, Markdown/JSON output |
| `examples.py` | Qualitative prediction grids |
| `evaluate.py` | CLI |
| `config.toml` / `params.toml` | Paths and run size / metric and baseline parameters |

## Notes

* The rules baseline was tuned on 400 test cases while the metrics were
  being developed; use `--split val` for any further tuning.
* `metres_per_px` comes from the dataset; CubiCasa floors flagged with a
  suspicious scale (`problems.json`) can distort `loc_error_m`.
