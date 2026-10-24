# postprocess: K-sample fusion and structure-aware clean-up

Works on the predictions saved by `model/predict.py`; needs no GPU and no
model. Output is in the same format, so `evaluation.evaluate` scores it
unchanged and every pipeline becomes one row of the ablation table.

## Usage (from the repository root)

```bash
python -m Reconstruction.postprocess.run --dataset resplan --split val \
    --predictions Reconstruction/runs/<run>/predictions/resplan_val/step_<N>_k<K> --pipeline full --mode prod
python -m Reconstruction.evaluation.evaluate --dataset resplan --split val \
    --predictions Reconstruction/runs/<run>/predictions/resplan_val/step_<N>_k<K>+full --name <run>_full --mode prod
```

`--mode debug` processes only the first 400 cases. Choose parameters on
`val`; run `test` once at the end.

## What it does

1. **Fusion**: K samples are voted per pixel and channel (`[aggregate]
   VOTE_THRESHOLD`). With K = 1 this is the identity. Generate K > 1 with
   `model.predict --samples K` to use it.
2. **Steps** (in `steps.py`, order = pipeline order) on the *added* labels:

| Step | Rule enforced |
|---|---|
| `remove_small` | no tiny fragments |
| `openings_on_wall` | every door / window touches a wall |
| `close_wall_gaps` | predicted walls connect to nearby walls (axis-aligned gaps up to `MAX_GAP_PX`) |
| `align_walls` | near-rectangular predicted walls become straight, axis-aligned boxes |
| `snap_rooms` | a wall-enclosed free area touched by the prediction becomes one room of the majority class |

Pipelines (`params.toml [pipelines]`): `none` (fusion only), `clean`,
`rooms`, `walls`, `full`. Add your own as a list of step names. Every
parameter lives in `params.toml`.

The log line after a run reports how many predictions the steps changed.

## Tests

`python -m pytest Reconstruction/tests/test_postprocess.py -q`
