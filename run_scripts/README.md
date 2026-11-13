Run the `build.sh` file for production build and the `debug.sh` file for debugging. `server.sh` is meant to open telemetry server (raw CSV listing over http).

`start_server.sh` starts the Flask telemetry dashboard, which visualises the VAE training metrics (loss curves) from `VAE/telemetry_logs` with charts:

```bash
./run_scripts/start_server.sh [port]   # default port 8080
```

`start_diffusion_server.sh` starts a separate Flask dashboard instance for
the diffusion run, reading the isolated telemetry in
`Diffusion/telemetry_logs` and grids in `Diffusion/outputs` (default port
8081, so both dashboards can run side by side):

```bash
./run_scripts/start_diffusion_server.sh [port]   # default port 8081
```
`predict_YOLO.sh` is an offline inference viewer for a finished YOLO run. It runs
the saved best weights over images from the live dataset and writes GT-vs-prediction
overlays plus a contact sheet to `YOLO/outputs/predict/`. Read-only with respect
to training state — it never rewrites the dataset or telemetry. The first argument
selects the run mode (`debug`|`prod`); the rest is forwarded to `YOLO/predict.py`:

```bash
./run_scripts/predict_YOLO.sh                                  # newest best.pt, val split
./run_scripts/predict_YOLO.sh prod
./run_scripts/predict_YOLO.sh debug --split test --limit 20
./run_scripts/predict_YOLO.sh debug --image /path/to/plan.png
./run_scripts/predict_YOLO.sh debug --weights path/best.pt --conf 0.25
```

It exports `PYTORCH_ENABLE_MPS_FALLBACK=1` because Ultralytics runs
`torchvision::nms` during prediction, which has no MPS kernel. See
`YOLO/README.md` for how to read the confidence values.

`start_evaluator_server.sh` starts a separate Flask dashboard instance for the
missing-component evaluation, reading the evaluator's per-sample telemetry in
`evaluator/telemetry_logs` and its figures in `evaluator/outputs` (default port
8086, so all dashboards can run side by side). The panel it shows is served by
the `eval` entry in `helpers/server.py`'s `IMAGE_KIND_PREFIXES`; without that
entry the panel is simply empty and every other dashboard is unaffected.

```bash
./run_scripts/start_evaluator_server.sh [port]   # default port 8086
```

Run `./run_eval.sh` first (see `evaluator/README.md`) to produce the figures.

`run_validation_comparison.sh` runs the evaluator for every missing component
(`tv`, `showcase`, `dining_set`) over the full validation split and writes one
overlap comparison table. It rebuilds the table after each component and keeps
`runs.json`, so `--skip-existing` resumes an interrupted multi-hour batch.

```bash
./run_scripts/run_validation_comparison.sh --skip-existing
# table: evaluator/outputs/validation_comparison/overlap_comparison.html
```

The validation split has 19,920 samples and each component uses 4 draws, so the
full three-component batch is a multi-hour run. The script keeps the machine
awake with `caffeinate` on macOS and writes a partial table after every
component.

To rebuild the table from completed runs without evaluating again:

```bash
./run_scripts/run_validation_comparison.sh --table-only
```

`run_frontend.sh` (at the repository root) starts the upload interface: a page
where a floor-plan class-id mask is uploaded, the configured mask_diffusion run
generates the missing component and the trained detector's bounding boxes are
drawn on the result (default port 8087). See `frontend/README.md`.

```bash
./run_frontend.sh                 # http://127.0.0.1:8087
./run_frontend.sh prod 0.0.0.0 9000
```
