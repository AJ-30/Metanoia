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