"""
Flask dashboard server for telemetry visualisation.

Serves a single-page dashboard that lists every telemetry run (the CSV files
produced by ``helpers/telemetry_writer.TelemetryWriter``), plots the logged
metrics (epoch vs train/test/recon/KL loss, YOLO box/cls/dfl loss and mAP,
etc.) using Chart.js, and displays per-run figures: the VAE's per-epoch
reconstruction/generation grids plus the YOLO GT-vs-pred overlays and
training-curve figures (results/PR/F1/confusion matrix).

The directories with the telemetry CSVs and the images are configurable, so
the same server visualises VAE runs (--telemetry-dir ./VAE/telemetry_logs
--recon-dir ./VAE/outputs) and YOLO runs (--telemetry-dir
./YOLO/telemetry_logs --recon-dir ./YOLO/outputs); panels belonging to the
other pipeline simply show their empty state. For mask_VAE runs the dashboard
additionally renders a mask layout legend from the ``layout_<run_id>.json``
sidecars (class colors, with excluded classes marked as background), and the
per-class companion CSVs (``*_per_class.csv``) are not listed as runs.

Usage:
    python helpers/server.py --telemetry-dir ./VAE/telemetry_logs --recon-dir ./VAE/outputs --port 8080
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template_string, send_from_directory

# Constant-ish tuning values: page title and Chart.js CDN source. Kept here
# (module constants) since they are presentation defaults, not experiment knobs.
PAGE_TITLE: str = "Telemetry Dashboard"

# Skip this file name: run_scripts/server.sh may create an index.csv symlink
# that just points at the latest run CSV and is not itself a telemetry run.
SKIP_NAMES: set[str] = {"index.csv"}

# Suffix of the per-class companion CSVs (mask_VAE writes
# "<run_id>_per_class.csv" next to "<run_id>.csv"). They hold per-class IoU /
# Dice columns for the same run, not a run of their own, so the run list skips
# them instead of showing every training run twice.
PER_CLASS_SUFFIX: str = "_per_class.csv"

# Prefix of the mask layout sidecar JSONs the mask_VAE runner writes next to
# its grids ("layout_<run_id>.json"): mode, class colors and excluded classes.
LAYOUT_FILE_PREFIX: str = "layout_"
LAYOUT_FILE_SUFFIX: str = ".json"

# Prefix of saved grid filenames, used to recover the epoch number from a name
# such as "reconstruction_epoch_7_<run>.png" / "generation_epoch_7_<run>.png".
# YOLO runs instead save "val_pred_<tag>_<stem>_<run>.png" overlays (no epoch
# number) and "<run>_results.png / _PR_curve.png / _F1_curve.png /
# _confusion_matrix.png / labels" curve artifacts, matched by substring below.
# Plain prefix lists keep this server agnostic to run ids.
IMAGE_KIND_PREFIXES: dict[str, list[str]] = {
    "reconstruction": ["reconstruction_epoch_"],
    "generation": ["generation_epoch_"],
    "pred": ["val_pred_"],
    "curve": ["results", "PR_curve", "F1_curve", "confusion_matrix", "labels"],
}

# Image extensions served as dashboard figures.
IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")

# Fallback epoch pattern: matches "_epoch_<N>_" or "epoch_<N>" anywhere in a
# filename (VAE-style names); YOLO overlay/curve names carry no epoch number
# and yield None, in which case the dashboard labels options by filename.
EPOCH_PATTERN: str = r"epoch_(\d+)"


def _image_epoch(filename: str) -> int | None:
    """Recover the epoch number from an image filename when present.

    Args:
        filename (str): Image file name to inspect.

    Returns:
        int | None: Epoch integer, or None when the name carries no epoch.
    """
    match: re.Match[str] | None = re.search(EPOCH_PATTERN, filename)
    return int(match.group(1)) if match else None

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
  .recon-panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px; margin-bottom: 20px;
  }
  .recon-panel h3 { font-size: 14px; margin-bottom: 10px; }
  .recon-controls { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
  .recon-controls label { font-size: 12px; color: var(--muted); }
  .recon-controls select {
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 10px; font-size: 13px;
  }
  .recon-img-wrap {
    background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
    padding: 8px; text-align: center;
  }
  .recon-img-wrap img { max-width: 100%; max-height: 60vh; border-radius: 4px; }
  .legend-chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip {
    display: inline-flex; align-items: center; gap: 6px; font-size: 12px;
    background: var(--bg); border: 1px solid var(--border); border-radius: 999px;
    padding: 4px 10px 4px 6px;
  }
  .chip .swatch {
    width: 14px; height: 14px; border-radius: 50%; border: 1px solid var(--border);
    flex-shrink: 0;
  }
  .chip.excluded { opacity: .75; }
  .chip.excluded .name { text-decoration: line-through; }
  .chip .badge {
    font-size: 10px; color: var(--muted); border: 1px solid var(--border);
    border-radius: 999px; padding: 1px 6px;
  }
  .layout-meta { font-size: 12px; color: var(--muted); margin-bottom: 10px; }
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
  let currentRunId = null;   // run selected by the user (or newest, by default)
  let lastDashboardKey = null; // JSON fingerprint used to skip identical redraws
  let selectedRecon = { value: null };  // reconstruction grid chosen in the epoch dropdown
  let selectedGen = { value: null };    // generation grid chosen in the epoch dropdown
  let selectedPred = { value: null };   // YOLO overlay chosen in the file dropdown
  let selectedCurve = { value: null };  // YOLO curve figure chosen in the file dropdown

  async function jsonFetch(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url} -> ${res.status}`);
    return res.json();
  }

  // Poll the backend on a timer so the graphs update live while training runs.
  async function refresh() {
    try {
      const runs = await jsonFetch("/api/runs");
      if (runs.length === 0) {
        currentRunId = null;
        lastDashboardKey = null;
        document.getElementById("run-list").innerHTML =
          `<div class="empty">No telemetry runs yet.<br>Start training:<br><code>{{ train_cmd }}</code></div>`;
        document.getElementById("dashboard").className = "empty";
        document.getElementById("dashboard").innerHTML =
          `No telemetry found yet. Start training and this page refreshes automatically.`;
        return;
      }
      // Default to the newest run, but keep the user's selection if still valid.
      if (!currentRunId || !runs.some(r => r.run_id === currentRunId)) {
        currentRunId = runs[0].run_id;
      }
      renderRunList(runs);

      const data = await jsonFetch(`/api/runs/${currentRunId}`);
      const key = JSON.stringify(data);
      if (key !== lastDashboardKey) {
        lastDashboardKey = key;
        renderDashboard(data);
      }
    } catch (err) {
      document.getElementById("run-list").innerHTML =
        "<div class='empty'>Could not reach the server.</div>";
    }
  }

  function renderRunList(runs) {
    const list = document.getElementById("run-list");
    list.innerHTML = "";
    for (const run of runs) {
      const btn = document.createElement("button");
      btn.className = "run" + (run.run_id === currentRunId ? " active" : "");
      btn.innerHTML = `<div class="rid">${run.run_id}</div>
        <div class="meta">${run.num_rows} rows · ${new Date(run.updated_at * 1000).toLocaleTimeString()}</div>`;
      btn.onclick = () => { currentRunId = run.run_id; lastDashboardKey = null; refresh(); };
      list.appendChild(btn);
    }
  }

  // Load the available image grids for a panel (reconstructions or generated
  // samples) and populate its epoch dropdown. A new grid is written after every
  // epoch, so the newest one is highlighted.
  async function loadGrids(apiUrl, selectId, imgId, selectState, emptyText, altText) {
    try {
      const list = await jsonFetch(apiUrl);
      const select = document.getElementById(selectId);
      const img = document.getElementById(imgId);
      if (!select || !img) return;

      select.innerHTML = "";
      if (list.length === 0) {
        img.removeAttribute("src");
        img.alt = emptyText;
        return;
      }

      // Auto-advance to the newest grid (list is newest-first) unless the user
      // explicitly picked a specific epoch.
      if (selectState.value === null || !list.some(r => r.file === selectState.value)) {
        selectState.value = list[0].file;
      }
      for (const rec of list) {
        const opt = document.createElement("option");
        opt.value = rec.file;
        opt.textContent = rec.epoch !== null ? `Epoch ${rec.epoch}` : rec.file;
        opt.selected = rec.file === selectState.value;
        select.appendChild(opt);
      }
      select.onchange = () => { selectState.value = select.value; showGrid(list, imgId, selectState, altText); };
      showGrid(list, imgId, selectState, altText);
    } catch (err) {
      // No grids available yet is fine; keep the panel quiet.
    }
  }

  function showGrid(list, imgId, selectState, altText) {
    const img = document.getElementById(imgId);
    if (!img) return;
    const rec = list.find(r => r.file === selectState.value);
    if (rec) {
      img.src = rec.url;
      img.alt = altText;
    }
  }

  // Load the mask layout sidecars and render the legend for the selected run
  // (falling back to the newest sidecar when the run has none, e.g. a run
  // from before sidecars existed). Runs without any sidecar keep a quiet
  // one-line note instead of an empty panel.
  async function loadLayout(runId) {
    const meta = document.getElementById("layout-meta");
    const chips = document.getElementById("layout-chips");
    if (!meta || !chips) return;
    try {
      const list = await jsonFetch("/api/layouts");
      if (list.length === 0) {
        meta.textContent = "No layout sidecar yet — written at the start of each mask_VAE run.";
        return;
      }
      const entry = list.find(r => r.run_id === runId) || list[0];
      const excluded = new Set(entry.excluded_classes || []);
      meta.textContent =
        `Mode ${entry.mask_mode} · groups ${(entry.groups || []).join(", ") || "—"}` +
        (excluded.size ? ` · excluded (as background): ${[...excluded].join(", ")}` : " · no excluded classes");
      chips.innerHTML = "";
      for (const ch of (entry.channels || [])) {
        const chip = document.createElement("span");
        chip.className = "chip" + (ch.excluded ? " excluded" : "");
        chip.title = ch.excluded ? `${ch.label} is trained and rendered as background` : ch.label;
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = ch.color;
        chip.appendChild(swatch);
        const name = document.createElement("span");
        name.className = "name";
        name.textContent = ch.label;
        chip.appendChild(name);
        if (ch.excluded) {
          const badge = document.createElement("span");
          badge.className = "badge";
          badge.textContent = "as background";
          chip.appendChild(badge);
        }
        chips.appendChild(chip);
      }
    } catch (err) {
      meta.textContent = "Could not load the mask layout.";
    }
  }

  function renderDashboard(data) {
    const container = document.getElementById("dashboard");
    container.className = "";
    container.innerHTML = "";

    // A run whose CSV has only a header (training just started) has no rows yet.
    if (data.rows.length === 0) {
      container.className = "empty";
      container.innerHTML = `<div>
        Training started — waiting for the first telemetry row to be written.<br>
        <span style="opacity:.7">The dashboard updates automatically.</span>
      </div>`;
      return;
    }

    const stats = document.createElement("div");
    stats.className = "stats";
    stats.innerHTML = `
      <div class="stat"><div class="label">Run</div><div class="value">${data.run_id}</div></div>
      <div class="stat"><div class="label">Rows</div><div class="value">${data.rows.length}</div></div>
      <div class="stat"><div class="label">Metrics</div><div class="value">${data.metric_columns.length}</div></div>`;
    container.appendChild(stats);

    // Mask layout panel: class legend with the dataset colors, so the
    // colorized grids can be read without guessing which color is which
    // class. Classes the run excludes (e.g. tv in arrangement_filtered) are
    // struck through and badged "as background": they train and render as
    // background and never appear in the grids.
    const layoutPanel = document.createElement("div");
    layoutPanel.className = "recon-panel";
    layoutPanel.innerHTML = `
      <h3>Mask layout</h3>
      <div class="layout-meta" id="layout-meta">Loading layout…</div>
      <div class="legend-chips" id="layout-chips"></div>`;
    container.appendChild(layoutPanel);
    loadLayout(data.run_id);

    // Reconstruction panel: shows the latest per-epoch grid and lets the user
    // flip back through any previous epoch's grid.
    const reconPanel = document.createElement("div");
    reconPanel.className = "recon-panel";
    reconPanel.innerHTML = `
      <h3>Reconstructions (fixed test samples · original row / reconstruction row)</h3>
      <div class="recon-controls">
        <label for="recon-select">Epoch</label>
        <select id="recon-select"></select>
      </div>
      <div class="recon-img-wrap"><img id="recon-img" alt="No reconstructions yet — generated after each epoch."></div>`;
    container.appendChild(reconPanel);

    // Generation panel: novel images decoded from random latent vectors.
    const genPanel = document.createElement("div");
    genPanel.className = "recon-panel";
    genPanel.innerHTML = `
      <h3>Generations (novel samples from the latent space)</h3>
      <div class="recon-controls">
        <label for="gen-select">Epoch</label>
        <select id="gen-select"></select>
      </div>
      <div class="recon-img-wrap"><img id="gen-img" alt="No generations yet — generated after each epoch."></div>`;
    container.appendChild(genPanel);

    // Prediction panel: YOLO GT-vs-pred overlays (green GT + red predictions).
    // Empty for VAE runs, which save no val_pred_*.png files.
    const predPanel = document.createElement("div");
    predPanel.className = "recon-panel";
    predPanel.innerHTML = `
      <h3>Predictions (green ground truth / red predictions + confidence)</h3>
      <div class="recon-controls">
        <label for="pred-select">Image</label>
        <select id="pred-select"></select>
      </div>
      <div class="recon-img-wrap"><img id="pred-img" alt="No prediction overlays yet — saved during YOLO training."></div>`;
    container.appendChild(predPanel);

    // Curves panel: YOLO results/PR/F1/confusion-matrix figures copied to outputs.
    // Empty for VAE runs.
    const curvePanel = document.createElement("div");
    curvePanel.className = "recon-panel";
    curvePanel.innerHTML = `
      <h3>Training curves (results / PR / F1 / confusion matrix)</h3>
      <div class="recon-controls">
        <label for="curve-select">Figure</label>
        <select id="curve-select"></select>
      </div>
      <div class="recon-img-wrap"><img id="curve-img" alt="No curve figures yet — copied from the Ultralytics run dir."></div>`;
    container.appendChild(curvePanel);

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

    loadGrids("/api/recs", "recon-select", "recon-img", selectedRecon,
      "No reconstructions yet — generated after each epoch.",
      "Original (top) vs reconstructed (bottom) test samples");
    loadGrids("/api/gens", "gen-select", "gen-img", selectedGen,
      "No generations yet — generated after each epoch.",
      "Novel images decoded from random latent vectors");
    loadGrids("/api/preds", "pred-select", "pred-img", selectedPred,
      "No prediction overlays yet — saved during YOLO training.",
      "Ground truth (green) vs predicted (red) boxes");
    loadGrids("/api/curves", "curve-select", "curve-img", selectedCurve,
      "No curve figures yet — copied from the Ultralytics run dir.",
      "YOLO training curve figure");
  }

  refresh();
  setInterval(refresh, 3000); // live-updating dashboard while training runs
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
        # Skip the index.csv convenience symlink, the per-class companion
        # CSVs (same run, not a new run) and unreadable files.
        if (
            file_path.name in SKIP_NAMES
            or file_path.name.endswith(PER_CLASS_SUFFIX)
            or not file_path.is_file()
        ):
            continue

        try:
            with open(file_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames: list[str] | None = reader.fieldnames
                if fieldnames is None:
                    continue
                rows: list[dict[str, Any]] = [
                    {k: _safe_float(v) if k not in (TIMESTAMP_COLUMN,) else v for k, v in row.items()}
                    for row in reader
                ]
        except (csv.Error, OSError):
            continue

        # Columns that look like a metric, i.e. every non-meta column.
        meta: set[str] = {TIMESTAMP_COLUMN, RUNTIME_COLUMN}
        metric_columns: list[str] = [c for c in fieldnames if c not in meta]

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
        fieldnames: list[str] | None = reader.fieldnames
        if fieldnames is None:
            return None
        rows: list[dict[str, Any]] = [
            {k: _safe_float(v) if k != TIMESTAMP_COLUMN else v for k, v in row.items()}
            for row in reader
        ]

    # Pick the first available x-axis column among the preferred candidates.
    x_column: str | None = next(
        (c for c in X_AXIS_PREFERENCES if c in fieldnames), None
    )

    metric_columns: list[str] = [
        c for c in fieldnames if c not in {TIMESTAMP_COLUMN, RUNTIME_COLUMN, x_column}
    ]

    return {
        "run_id": run_id,
        "x_column": x_column,
        "metric_columns": metric_columns,
        "rows": rows,
    }


def discover_grids(grid_dir: str, kind: str) -> list[dict[str, Any]]:
    """Scan a directory for dashboard images of one kind, newest first.

    Matches files whose name starts with one of the kind's registered prefixes
    (see ``IMAGE_KIND_PREFIXES``) and ends with a served image extension.
    Keeps the historical name so the VAE ``/api/recs`` and ``/api/gens``
    routes behave exactly as before.

    Args:
        grid_dir (str): Directory holding the per-epoch grid images.
        kind (str): Image kind, one of "reconstruction", "generation", "pred"
            (YOLO GT-vs-pred overlays) or "curve" (YOLO training curves).

    Returns:
        list[dict[str, Any]]: One dict per image with keys file, epoch (None
            when the filename carries no epoch number) and url.
    """
    grid_path: Path = Path(grid_dir)
    if not grid_path.is_dir():
        return []

    prefixes: list[str] = IMAGE_KIND_PREFIXES.get(kind, [])
    grids: list[dict[str, Any]] = []
    for file_path in sorted(grid_path.iterdir()):
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Curve artifacts carry a run-name prefix ("train_results.png"), so
        # they match by substring; every other kind matches by filename prefix.
        if kind == "curve":
            if not any(s in file_path.name for s in prefixes):
                continue
        elif not any(file_path.name.startswith(p) for p in prefixes):
            continue
        grids.append({
            "file": file_path.name,
            "epoch": _image_epoch(file_path.name),
            "url": f"/grid/{kind}/{file_path.name}",
            "updated_at": file_path.stat().st_mtime,
        })

    # Newest image first (most recent write) so the dashboard can auto-select
    # the latest figure.
    grids.sort(key=lambda g: g["updated_at"], reverse=True)
    return grids


def discover_layouts(grid_dir: str) -> list[dict[str, Any]]:
    """Scan a directory for mask layout sidecar JSONs, newest first.

    The mask_VAE runner writes one ``layout_<run_id>.json`` per training run
    next to its colorized grids, naming the mask mode, the class colors and
    the classes excluded as background (e.g. ``tv`` in
    ``arrangement_filtered``). The dashboard renders this as a class legend so
    an excluded class is visibly marked as background instead of silently
    missing from the grids.

    Args:
        grid_dir (str): Directory holding the per-epoch grid images.

    Returns:
        list[dict[str, Any]]: One parsed sidecar payload per file with keys
            file, run_id, mask_mode, groups, excluded_classes, channels and
            updated_at. Unreadable files are skipped.
    """
    grid_path: Path = Path(grid_dir)
    if not grid_path.is_dir():
        return []

    layouts: list[dict[str, Any]] = []
    for file_path in sorted(grid_path.iterdir()):
        if (
            not file_path.name.startswith(LAYOUT_FILE_PREFIX)
            or not file_path.name.endswith(LAYOUT_FILE_SUFFIX)
            or not file_path.is_file()
        ):
            continue
        try:
            with open(file_path, encoding="utf-8") as f:
                payload: dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        layouts.append({
            "file": file_path.name,
            "run_id": payload.get("run_id", file_path.stem),
            "mask_mode": payload.get("mask_mode", "unknown"),
            "groups": payload.get("groups", []),
            "excluded_classes": payload.get("excluded_classes", []),
            "channels": payload.get("channels", []),
            "updated_at": file_path.stat().st_mtime,
        })

    # Newest sidecar first so the dashboard can fall back to the latest layout
    # when the selected run has none (e.g. a run from before sidecars existed).
    layouts.sort(key=lambda item: item["updated_at"], reverse=True)
    return layouts


def create_app(telemetry_dir: str, recon_dir: str, train_cmd: str = "./train_VAE.sh") -> Flask:
    """Create and configure the Flask application.

    Args:
        telemetry_dir (str): Directory read for telemetry CSV files.
        recon_dir (str): Directory read for reconstruction grid PNGs.
        train_cmd (str): Training command shown in the empty-state panel when no
            telemetry exists yet. Defaults to the VAE script so existing launchers
            that do not pass one keep working unchanged.

    Returns:
        Flask: Configured application with the dashboard routes.
    """
    app: Flask = Flask(__name__)

    # send_from_directory requires an absolute directory, and discover_* works
    # reliably with one too, so resolve both passed-in paths up front.
    telemetry_path: Path = Path(telemetry_dir).resolve()
    recon_path: Path = Path(recon_dir).resolve()

    @app.route("/")
    def index() -> str:
        """Serve the single-page dashboard."""
        return render_template_string(INDEX_HTML, title=PAGE_TITLE, train_cmd=train_cmd)

    @app.route("/api/runs")
    def api_runs() -> Response:
        """Return the list of runs as JSON."""
        return jsonify(discover_runs(str(telemetry_path)))

    @app.route("/api/runs/<run_id>")
    def api_run(run_id: str) -> Response:
        """Return a single run's rows as JSON.

        Args:
            run_id (str): CSV stem identifying the run.
        """
        data: dict[str, Any] | None = read_run(str(telemetry_path), run_id)
        if data is None:
            return jsonify({"error": "Run not found"}), 404
        return jsonify(data)

    @app.route("/api/recs")
    def api_recs() -> Response:
        """Return the list of reconstruction grids as JSON."""
        return jsonify(discover_grids(str(recon_path), "reconstruction"))

    @app.route("/api/gens")
    def api_gens() -> Response:
        """Return the list of generation grids as JSON."""
        return jsonify(discover_grids(str(recon_path), "generation"))

    @app.route("/api/preds")
    def api_preds() -> Response:
        """Return the list of YOLO GT-vs-pred overlay images as JSON."""
        return jsonify(discover_grids(str(recon_path), "pred"))

    @app.route("/api/curves")
    def api_curves() -> Response:
        """Return the list of YOLO training-curve figures as JSON."""
        return jsonify(discover_grids(str(recon_path), "curve"))

    @app.route("/api/layouts")
    def api_layouts() -> Response:
        """Return the mask layout sidecars (mode, colors, exclusions) as JSON."""
        return jsonify(discover_layouts(str(recon_path)))

    @app.route("/grid/<kind>/<path:filename>")
    def grid_images(kind: str, filename: str) -> Response:
        """Serve a dashboard image (reconstruction, generation, pred or curve).

        Args:
            kind (str): Image kind, one of "reconstruction", "generation",
                "pred" or "curve", used to keep the static route somewhat
                self-documenting.
            filename (str): Name of the image file.
        """
        if kind not in {"reconstruction", "generation", "pred", "curve"}:
            return jsonify({"error": "Unknown grid kind"}), 404
        return send_from_directory(str(recon_path), filename)

    return app


def main() -> None:
    """Parse CLI arguments and run the Flask development server."""
    parser = argparse.ArgumentParser(description="Serve the telemetry dashboard.")
    # Environment variables allow easy override without touching the script.
    default_dir: str = os.environ.get("TELEMETRY_DIR", "./telemetry_logs")
    default_recon_dir: str = os.environ.get("TELEMETRY_RECON_DIR", "./VAE/outputs")
    default_port: int = int(os.environ.get("TELEMETRY_PORT", "8080"))

    parser.add_argument("--telemetry-dir", default=default_dir,
                        help="Directory with telemetry CSV files (default: ./telemetry_logs)")
    parser.add_argument("--recon-dir", default=default_recon_dir,
                        help="Directory with reconstruction grid PNGs (default: ./VAE/outputs)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=default_port, help="Port to bind (default: 8080)")
    parser.add_argument("--train-cmd", default="./train_VAE.sh",
                        help="Training command shown when no telemetry exists yet (default: ./train_VAE.sh)")
    args: argparse.Namespace = parser.parse_args()

    # The telemetry/recon directories may not exist yet when start_server.sh runs.
    os.makedirs(args.telemetry_dir, exist_ok=True)
    os.makedirs(args.recon_dir, exist_ok=True)

    app: Flask = create_app(args.telemetry_dir, args.recon_dir, args.train_cmd)
    print(f"Telemetry dashboard for '{args.telemetry_dir}': http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()