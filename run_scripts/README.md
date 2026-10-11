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
