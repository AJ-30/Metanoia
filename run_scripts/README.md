Run the `build.sh` file for production build and the `debug.sh` file for debugging. `server.sh` is meant to open telemetry server (raw CSV listing over http).

`start_server.sh` starts the Flask telemetry dashboard, which visualises the VAE training metrics (loss curves) from `VAE/telemetry_logs` with charts:

```bash
./run_scripts/start_server.sh [port]   # default port 8080
```