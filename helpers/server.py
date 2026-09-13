"""
Flask dashboard server for telemetry visualisation.

Serves a single-page dashboard that lists every telemetry run (the CSV files
produced by ``helpers/telemetry_writer.TelemetryWriter``) and plots the logged
metrics (epoch vs train/recon/KL loss, etc.) using Chart.js.

The directory with the telemetry CSVs is configurable, so the same server can
visualise any run: the main project's ``telemetry_logs/`` or the VAE's
``VAE/telemetry_logs/``.

Usage:
    python helpers/server.py --telemetry-dir ./VAE/telemetry_logs --port 8080
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template_string

# Constant-ish tuning values: page title and Chart.js CDN source. Kept here
# (module constants) since they are presentation defaults, not experiment knobs.
PAGE_TITLE: str = "Telemetry Dashboard"

# Skip this file name: run_scripts/server.sh may create an index.csv symlink
# that just points at the latest run CSV and is not itself a telemetry run.
SKIP_NAMES: set[str] = {"index.csv"}

# Identifiers of common non-metric columns stored in telemetry CSVs. These are
# never plotted as training metrics but may be used as chart x-axes.
TIMESTAMP_COLUMN: str = "timestamp"
RUNTIME_COLUMN: str = "runtime"
X_AXIS_PREFERENCES: list[str] = ["epoch", RUNTIME_COLUMN, TIMESTAMP_COLUMN]

# Dark theme dashboard: single page, charts rendered client side.
INDEX_HTML: str = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1117; --panel: #171a23; --border: #262b3a;
    --text: #e6e9f0; --muted: #8b93a7; --accent: #6c8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    display: flex; min-height: 100vh;
  }
  aside {
    width: 260px; flex-shrink: 0; border-right: 1px solid var(--border);
    padding: 20px 14px; background: var(--panel); overflow-y: auto;
  }
  aside h1 { font-size: 16px; margin-bottom: 4px; }
  aside .subtitle { font-size: 12px; color: var(--muted); margin-bottom: 18px; }
  .run {
    display: block; width: 100%; text-align: left; background: none; border: 1px solid var(--border);
    color: var(--text); padding: 10px 12px; border-radius: 8px; cursor: pointer; margin-bottom: 8px;
    font-size: 13px; transition: border-color .15s, background .15s;
  }
  .run:hover { border-color: var(--accent); }
  .run.active { border-color: var(--accent); background: rgba(108,140,255,.08); }
  .run .rid { font-weight: 600; }
  .run .meta { color: var(--muted); font-size: 11px; margin-top: 3px; }
  main { flex: 1; padding: 24px 28px; overflow-y: auto; }
  .empty {
    color: var(--muted); margin-top: 40vh; text-align: center;
  }
  .stats { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 22px; }
  .stat {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 18px; min-width: 140px;
  }
  .stat .label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
  .stat .value { font-size: 18px; font-weight: 700; margin-top: 4px; }
  .chart-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; margin-bottom: 20px;
  }
  .chart-card h3 { font-size: 14px; margin-bottom: 8px; }
  .chart-wrap { position: relative; height: 280px; }
</style>
</head>
<body>
<aside>
  <h1>📈 {{ title }}</h1>
  <div class="subtitle">Pick a training run to view its metrics</div>
  <div id="run-list">Loading runs…</div>
</aside>
<main>
  <div id="dashboard" class="empty">Select a run on the left to see its training metrics.</div>
</main>
<script>
  const charts = {};
  const COLORS = ["#6c8cff", "#ffd166", "#06d6a0", "#ef476f", "#b388ff", "#4cc9f0"];

  async function loadRuns() {
    try {
      const res = await fetch("/api/runs");
      const runs = await res.json();
      renderRunList(runs);
      if (runs.length > 0) selectRun(runs[0].run_id);
    } catch (err) {
      document.getElementById("run-list").innerHTML = "<div class='empty'>Could not load runs.</div>";
    }
  }

  function renderRunList(runs) {
    const list = document.getElementById("run-list");
    list.innerHTML = "";
    for (const run of runs) {
      const btn = document.createElement("button");
      btn.className = "run";
      btn.dataset.runId = run.run_id;
      btn.innerHTML = `<div class="rid">${run.run_id}</div>
        <div class="meta">${run.num_rows} rows · updated ${run.updated_at}</div>`;
      btn.onclick = () => selectRun(run.run_id);
      list.appendChild(btn);
    }
  }

  function selectRun(runId) {
    document.querySelectorAll(".run").forEach(b =>
      b.classList.toggle("active", b.dataset.runId === runId));
    loadRun(runId);
  }

  async function loadRun(runId) {
    const res = await fetch(`/api/runs/${runId}`);
    const data = await res.json();
    renderDashboard(data);
  }

  function renderDashboard(data) {
    const container = document.getElementById("dashboard");
    container.className = "";
    container.innerHTML = "";

    const stats = document.createElement("div");
    stats.className = "stats";
    stats.innerHTML = `
      <div class="stat"><div class="label">Run</div><div class="value">${data.run_id}</div></div>
      <div class="stat"><div class="label">Rows</div><div class="value">${data.rows.length}</div></div>
      <div class="stat"><div class="label">Metrics</div><div class="value">${data.metric_columns.length}</div></div>`;
    container.appendChild(stats);

    const xCol = data.x_column;
    const labels = data.rows.map(r => (xCol ? r[xCol] : r.index));

    for (const [i, col] of data.metric_columns.entries()) {
      const card = document.createElement("div");
      card.className = "chart-card";
      card.innerHTML = `
        <h3>${col}</h3>
        <div class="chart-wrap"><canvas id="chart-${col}"></canvas></div>`;
      container.appendChild(card);

      const ctx = document.getElementById(`chart-${col}`).getContext("2d");
      const color = COLORS[i % COLORS.length];

      const values = data.rows.map(r => r[col]);
      const minV = Math.min(...values);
      const maxV = Math.max(...values);

      if (charts[col]) charts[col].destroy();
      charts[col] = new Chart(ctx, {
        type: "line",
        data: {
          labels: labels,
          datasets: [{
            label: col,
            data: values,
            borderColor: color,
            backgroundColor: color + "22",
            fill: true,
            tension: 0.25,
            pointRadius: 1.5,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { title: { display: true, text: xCol || "index" }, ticks: { maxTicksLimit: 12 }, grid: { color: "#262b3a" } },
            y: { title: { display: true, text: col }, grid: { color: "#262b3a" } },
          },
          plugins: { legend: { display: false } },
        },
      });
    }
  }

  loadRuns();
</script>
</body>
</html>
"""


def _safe_float(value: str) -> float | None:
    """Convert a raw CSV string cell to float when possible.

    Args:
        value (str): Raw cell value from the CSV.

    Returns:
        float | None: Parsed float, or None when the cell is empty/unparseable.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def discover_runs(telemetry_dir: str) -> list[dict[str, Any]]:
    """Scan a directory for telemetry CSV runs, newest first.

    Args:
        telemetry_dir (str): Directory holding the per-run CSV files.

    Returns:
        list[dict[str, Any]]: One dict per run with keys run_id, file, num_rows,
            metric_columns, updated_at.
    """
    telemetry_path: Path = Path(telemetry_dir)
    if not telemetry_path.is_dir():
        return []

    runs: list[dict[str, Any]] = []
    for file_path in sorted(telemetry_path.glob("*.csv")):
        # Skip the index.csv convenience symlink and unreadable files.
        if file_path.name in SKIP_NAMES or not file_path.is_file():
            continue

        try:
            with open(file_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    continue
                rows: list[dict[str, Any]] = [
                    {k: _safe_float(v) if k not in (TIMESTAMP_COLUMN,) else v for k, v in row.items()}
                    for row in reader
                ]
                if not rows:
                    continue
        except (csv.Error, OSError):
            continue

        # Columns that look like a metric, i.e. every non-meta column.
        meta: set[str] = {TIMESTAMP_COLUMN, RUNTIME_COLUMN}
        metric_columns: list[str] = [c for c in rows[0].keys() if c not in meta]

        runs.append({
            "run_id": file_path.stem,
            "file": file_path.name,
            "num_rows": len(rows),
            "metric_columns": metric_columns,
            "updated_at": file_path.stat().st_mtime,
        })

    # Newest run first so the most recent training appears on top.
    runs.sort(key=lambda r: r["updated_at"], reverse=True)
    return runs


def read_run(telemetry_dir: str, run_id: str) -> dict[str, Any] | None:
    """Read a single run's CSV rows together with its metadata.

    Args:
        telemetry_dir (str): Directory holding the per-run CSV files.
        run_id (str): CSV stem (without the .csv extension).

    Returns:
        dict[str, Any] | None: Dict with run_id, x_column, metric_columns and
            rows; None when the run file does not exist.
    """
    telemetry_path: Path = Path(telemetry_dir) / f"{run_id}.csv"
    if not telemetry_path.is_file():
        return None

    with open(telemetry_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        rows: list[dict[str, Any]] = [
            {k: _safe_float(v) if k != TIMESTAMP_COLUMN else v for k, v in row.items()}
            for row in reader
        ]

    # Pick the first available x-axis column among the preferred candidates.
    x_column: str | None = next(
        (c for c in X_AXIS_PREFERENCES if c in rows[0]), None
    ) if rows else None

    metric_columns: list[str] = [
        c for c in rows[0].keys()
        if c not in {TIMESTAMP_COLUMN, RUNTIME_COLUMN, x_column}
    ] if rows else []

    return {
        "run_id": run_id,
        "x_column": x_column,
        "metric_columns": metric_columns,
        "rows": rows,
    }


def create_app(telemetry_dir: str) -> Flask:
    """Create and configure the Flask application.

    Args:
        telemetry_dir (str): Directory read for telemetry CSV files.

    Returns:
        Flask: Configured application with the dashboard routes.
    """
    app: Flask = Flask(__name__)

    @app.route("/")
    def index() -> str:
        """Serve the single-page dashboard."""
        return render_template_string(INDEX_HTML, title=PAGE_TITLE)

    @app.route("/api/runs")
    def api_runs() -> Response:
        """Return the list of runs as JSON."""
        return jsonify(discover_runs(telemetry_dir))

    @app.route("/api/runs/<run_id>")
    def api_run(run_id: str) -> Response:
        """Return a single run's rows as JSON.

        Args:
            run_id (str): CSV stem identifying the run.
        """
        data: dict[str, Any] | None = read_run(telemetry_dir, run_id)
        if data is None:
            return jsonify({"error": "Run not found"}), 404
        return jsonify(data)

    return app


def main() -> None:
    """Parse CLI arguments and run the Flask development server."""
    parser = argparse.ArgumentParser(description="Serve the telemetry dashboard.")
    # Environment variables allow easy override without touching the script.
    default_dir: str = os.environ.get("TELEMETRY_DIR", "./telemetry_logs")
    default_port: int = int(os.environ.get("TELEMETRY_PORT", "8080"))

    parser.add_argument("--telemetry-dir", default=default_dir,
                        help="Directory with telemetry CSV files (default: ./telemetry_logs)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=default_port, help="Port to bind (default: 8080)")
    args: argparse.Namespace = parser.parse_args()

    # The telemetry directory may not exist yet when start_server.sh runs.
    os.makedirs(args.telemetry_dir, exist_ok=True)

    app: Flask = create_app(args.telemetry_dir)
    print(f"Telemetry dashboard for '{args.telemetry_dir}': http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()