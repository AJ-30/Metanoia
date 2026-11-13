"""Run the evaluator for every component on one split and build a comparison table.

The evaluator itself answers one component at a time. This module is a thin
batch driver around :func:`evaluator.pipeline.run_evaluation`:

1. load the same evaluator config/hyperparameters the normal CLI uses;
2. point the evaluator at the validation split (or the test split);
3. run every requested component with the full sample set;
4. read each run's summary JSON and per-sample telemetry CSV;
5. write a Markdown, CSV and HTML comparison table under
   ``evaluator/outputs/<split>_comparison/``.

The evaluator is not modified: this module only selects a split and calls its
public entry point. A small manifest (``runs.json``) records completed runs so
``--skip-existing`` can resume an interrupted multi-hour batch without redoing
finished components.

Typical use::

    caffeinate -dimsu ./venv/bin/python evaluator/run_validation_comparison.py \
        --skip-existing

The table is written to::

    evaluator/outputs/validation_comparison/overlap_comparison.html
"""

from __future__ import annotations

import os

# Ultralytics runs torchvision::nms during detector validation. MPS has no
# kernel for it, so the fallback must be set before torch is imported by any
# project module.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import html
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Repository root, resolved from this file so the command works from anywhere.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluator.pipeline import run_evaluation  # noqa: E402
from evaluator.report import format_report  # noqa: E402
from evaluator.settings import PATH_CONFIG_KEYS, anchor_path_keys  # noqa: E402
from evaluator.variants import component_names, select_component  # noqa: E402
from helpers.config import load_config  # noqa: E402
from helpers.hyperparams import load_hyperparams  # noqa: E402
from helpers.logger import setup_logger  # noqa: E402

# The evaluator always reads its sample split from TEST_IMAGES_DIR. The batch
# driver swaps that one key, which is why the normal evaluator needs no change.
SPLIT_CONFIG_KEYS: dict[str, str] = {
    "validation": "VALIDATION_IMAGES_DIR",
    "test": "TEST_IMAGES_DIR",
}

# Reporting thresholds. They are display choices, not model hyperparameters.
OVERLAP_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)
OVERLAP_QUANTILES: tuple[float, ...] = (0.90, 0.95, 0.99)

# Table and manifest names. The split-specific parent directory is chosen in
# main() so validation and test tables never overwrite each other.
TABLE_STEM: str = "overlap_comparison"
PER_DRAW_STEM: str = "overlap_by_draw"
MANIFEST_NAME: str = "runs.json"
TOP_WORST_CASES: int = 3

SCRIPT_TOML: str = "config.toml"
HYPERPARAMS_TOML: str = "hyperparameters.toml"


@dataclass(frozen=True)
class ComponentResult:
    """One completed component run, reduced to the numbers the table shows.

    Attributes:
        component (str): Component key, e.g. ``tv`` or ``showcase``.
        run_id (str): Evaluator run id, which names the telemetry CSV.
        split_dir (str): Split directory the evaluator actually read.
        samples (int): Number of samples evaluated.
        draws (int): Independent diffusion chains per sample.
        predicted (int): Ensemble samples where the detector returned a target
            box, so the overlap metric is defined.
        missing_prediction (int): Ensemble samples where no target box was
            predicted; these are reported as ``n/a`` rather than zero.
        mean_other_gt (float | None): Mean number of other ground-truth
            furniture boxes per ensemble sample.
        overlap_mean (float | None): Mean overlap over predicted ensemble
            samples.
        overlap_median (float | None): Median overlap.
        overlap_p90 (float | None): 90th percentile overlap.
        overlap_p95 (float | None): 95th percentile overlap.
        overlap_p99 (float | None): 99th percentile overlap.
        overlap_max (float | None): Largest observed overlap.
        positive_rate (float | None): Fraction of predicted samples with any
            overlap (``overlap > 0``).
        rate_gt_5pct (float | None): Fraction with overlap above 5% of the
            target box.
        rate_gt_10pct (float | None): Fraction with overlap above 10%.
        rate_gt_25pct (float | None): Fraction with overlap above 25%.
        rate_gt_50pct (float | None): Fraction with overlap above 50%.
        mean_positive (float | None): Mean overlap among positive-overlap
            samples, i.e. how severe a collision is when one occurs.
        worst_stem (str | None): Ensemble sample with the largest overlap.
        worst_overlap (float | None): Overlap of that worst sample.
        map50 (float | None): Ensemble detector mAP@50.
        map50_95 (float | None): Ensemble detector mAP@50-95.
        precision (float | None): Ensemble detector precision.
        recall (float | None): Ensemble detector recall.
        box_iou (float | None): Mean IoU of matched target boxes.
        channel_dice (float | None): Mean generated-channel Dice.
        elapsed_seconds (float): Wall-clock seconds of the evaluator run.
        summary_path (Path): The run's ``evaluation_summary.json``.
        telemetry_path (Path): The run's per-sample telemetry CSV.
        per_draw (dict[str, dict[str, float | None]]): Overlap statistics keyed
            by draw string (``-1`` is the ensemble, ``0``..``N-1`` are draws).
        worst_cases (list[dict[str, Any]]): The largest positive-overlap
            ensemble rows, for debugging.
    """

    component: str
    run_id: str
    split_dir: str
    samples: int
    draws: int
    predicted: int
    missing_prediction: int
    mean_other_gt: float | None
    overlap_mean: float | None
    overlap_median: float | None
    overlap_p90: float | None
    overlap_p95: float | None
    overlap_p99: float | None
    overlap_max: float | None
    positive_rate: float | None
    rate_gt_5pct: float | None
    rate_gt_10pct: float | None
    rate_gt_25pct: float | None
    rate_gt_50pct: float | None
    mean_positive: float | None
    worst_stem: str | None
    worst_overlap: float | None
    map50: float | None
    map50_95: float | None
    precision: float | None
    recall: float | None
    box_iou: float | None
    channel_dice: float | None
    elapsed_seconds: float
    summary_path: Path
    telemetry_path: Path
    per_draw: dict[str, dict[str, float | None]]
    worst_cases: list[dict[str, Any]]


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the batch runner's command line.

    Args:
        argv (list[str] | None): Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: Parsed options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run every missing component over one split and write an overlap "
            "comparison table."
        )
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help=(
            "Components to run, e.g. tv showcase dining_set. Defaults to every "
            "row of components.toml."
        ),
    )
    parser.add_argument(
        "--split",
        choices=tuple(SPLIT_CONFIG_KEYS),
        default="validation",
        help="Dataset split to evaluate (default: validation).",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=-1,
        help="Samples per component; negative keeps every sample (default: -1).",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=None,
        help="Diffusion chains per sample; defaults to DRAWS_PER_SAMPLE.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed; defaults to the evaluator's SEED.",
    )
    parser.add_argument(
        "--mode",
        choices=("debug", "prod"),
        default="debug",
        help="Evaluator config mode (default: debug).",
    )
    parser.add_argument(
        "--table-dir",
        type=str,
        default=None,
        help=(
            "Directory for the table and manifest. Defaults to "
            "evaluator/outputs/<split>_comparison/."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Reuse completed component runs already recorded in runs.json. "
            "Use this to resume an interrupted batch."
        ),
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Rebuild the table from runs.json without running the evaluator.",
    )
    return parser.parse_args(argv)


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Read the completed-run manifest, or return an empty one.

    Args:
        path (Path): ``runs.json`` path.

    Returns:
        dict[str, dict[str, Any]]: Component key -> manifest entry.

    Raises:
        ValueError: If the file exists but is not a JSON object.
    """
    if not path.is_file():
        return {}
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Comparison manifest {path} must hold a JSON object.")
    return {str(key): dict(value) for key, value in data.items()}


def _write_manifest(path: Path, manifest: dict[str, dict[str, Any]]) -> None:
    """Write the manifest atomically so an interruption cannot corrupt it.

    Args:
        path (Path): ``runs.json`` path.
        manifest (dict[str, dict[str, Any]]): Component key -> entry.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _entry_is_complete(entry: dict[str, Any]) -> bool:
    """Report whether a manifest entry has both of its output files.

    Args:
        entry (dict[str, Any]): One manifest entry.

    Returns:
        bool: ``True`` when the summary JSON and telemetry CSV both exist.
    """
    summary_path: Path = Path(str(entry.get("summary_path", "")))
    telemetry_path: Path = Path(str(entry.get("telemetry_path", "")))
    return summary_path.is_file() and telemetry_path.is_file()


def _entry_matches(entry: dict[str, Any], args: argparse.Namespace) -> bool:
    """Report whether a completed run used the requested batch settings.

    Reusing a run from a different split, cap, draw count or seed would mix
    incomparable rows into one table, so ``--skip-existing`` only reuses an
    entry when every requested setting matches.

    Args:
        entry (dict[str, Any]): One manifest entry.
        args (argparse.Namespace): Current batch-runner options.

    Returns:
        bool: ``True`` when the entry's split, cap, draws and seed match.
    """
    return (
        entry.get("split") == args.split
        and int(entry.get("cap", -1)) == args.cap
        and entry.get("requested_draws") == args.draws
        and entry.get("requested_seed") == args.seed
    )


def _read_telemetry(path: Path) -> list[dict[str, str]]:
    """Read one evaluator telemetry CSV.

    Args:
        path (Path): Telemetry CSV path.

    Returns:
        list[dict[str, str]]: One dictionary per row, in file order.

    Raises:
        FileNotFoundError: If the CSV is absent.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Telemetry CSV is missing: {path}.")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _overlap_value(row: dict[str, str]) -> float | None:
    """Read one row's overlap cell.

    Args:
        row (dict[str, str]): One telemetry row.

    Returns:
        float | None: The overlap fraction, or ``None`` for a blank cell (no
            target prediction).
    """
    raw: str = row.get("overlap", "").strip()
    return float(raw) if raw else None


def _quantile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank quantile for a non-empty list.

    Args:
        values (list[float]): Measurements.
        fraction (float): Quantile in ``0..1``, e.g. ``0.90``.

    Returns:
        float: The selected value.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        raise ValueError("Cannot take a quantile of an empty list.")
    ordered: list[float] = sorted(values)
    index: int = min(
        len(ordered) - 1,
        max(0, int(fraction * (len(ordered) - 1))),
    )
    return ordered[index]


def _overlap_stats(values: list[float]) -> dict[str, float | None]:
    """Summarize a list of overlap fractions.

    Args:
        values (list[float]): Overlap values for one draw or component.

    Returns:
        dict[str, float | None]: Distribution and threshold statistics. Empty
            when there are no predictions, so callers render ``n/a``.
    """
    if not values:
        return {}
    positive: list[float] = [value for value in values if value > 0.0]
    stats: dict[str, float | None] = {
        "count": float(len(values)),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _quantile(values, OVERLAP_QUANTILES[0]),
        "p95": _quantile(values, OVERLAP_QUANTILES[1]),
        "p99": _quantile(values, OVERLAP_QUANTILES[2]),
        "max": max(values),
        "positive_rate": len(positive) / len(values),
        "mean_positive": statistics.fmean(positive) if positive else None,
    }
    for threshold in OVERLAP_THRESHOLDS:
        key: str = f"rate_gt_{int(round(threshold * 100))}pct"
        stats[key] = sum(1 for value in values if value > threshold) / len(values)
    return stats


def _as_float(value: Any) -> float | None:
    """Convert an optional JSON number to a float.

    Args:
        value (Any): Number, string, or ``None``.

    Returns:
        float | None: The float value, or ``None``.
    """
    return None if value is None else float(value)


def _component_result(entry: dict[str, Any]) -> ComponentResult:
    """Reduce one completed run to the numbers the comparison table shows.

    Args:
        entry (dict[str, Any]): Manifest entry with summary and telemetry paths.

    Returns:
        ComponentResult: Parsed, table-ready result.

    Raises:
        FileNotFoundError: If the summary or telemetry file is missing.
        KeyError: If the summary does not have the evaluator's expected shape.
    """
    summary_path: Path = Path(str(entry["summary_path"]))
    telemetry_path: Path = Path(str(entry["telemetry_path"]))
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = _read_telemetry(telemetry_path)

    ensemble_rows: list[dict[str, str]] = [
        row for row in rows if row.get("draw") == "-1"
    ]
    ensemble_values: list[float] = [
        value
        for row in ensemble_rows
        for value in [_overlap_value(row)]
        if value is not None
    ]
    stats: dict[str, float | None] = _overlap_stats(ensemble_values)
    other_gt_values: list[int] = [
        int(row["other_gt_boxes"]) for row in ensemble_rows
    ]

    per_draw: dict[str, dict[str, float | None]] = {}
    draw_keys: list[str] = sorted(
        {row["draw"] for row in rows},
        key=int,
    )
    for draw_key in draw_keys:
        draw_values: list[float] = [
            value
            for row in rows
            if row["draw"] == draw_key
            for value in [_overlap_value(row)]
            if value is not None
        ]
        per_draw[draw_key] = _overlap_stats(draw_values)

    scored_rows: list[tuple[float, dict[str, str]]] = [
        (value, row)
        for row in ensemble_rows
        for value in [_overlap_value(row)]
        if value is not None and value > 0.0
    ]
    scored_rows.sort(key=lambda item: item[0], reverse=True)
    worst_cases: list[dict[str, Any]] = [
        {
            "stem": row["stem"],
            "overlap": value,
            "other_gt": int(row["other_gt_boxes"]),
            "predictions": int(row["pred_boxes"]),
            "matched": int(row["matched_boxes"]),
        }
        for value, row in scored_rows[:TOP_WORST_CASES]
    ]

    ensemble: dict[str, Any] = dict(summary.get("ensemble", {}))
    detector: dict[str, Any] = dict(ensemble.get("detector_metrics") or {})
    box_loss: dict[str, Any] = dict(ensemble.get("box_loss") or {})
    channel: dict[str, Any] = dict(ensemble.get("channel_scores") or {})
    predicted: int = int(float(stats.get("count", 0.0) or 0.0))

    return ComponentResult(
        component=str(summary["missing_component"]["name"]),
        run_id=str(summary["run_id"]),
        split_dir=str(summary["configuration"]["split_dir"]),
        samples=int(summary["configuration"]["samples_evaluated"]),
        draws=int(summary["configuration"]["draws_per_sample"]),
        predicted=predicted,
        missing_prediction=len(ensemble_rows) - predicted,
        mean_other_gt=(
            statistics.fmean(other_gt_values) if other_gt_values else None
        ),
        overlap_mean=_as_float(stats.get("mean")),
        overlap_median=_as_float(stats.get("median")),
        overlap_p90=_as_float(stats.get("p90")),
        overlap_p95=_as_float(stats.get("p95")),
        overlap_p99=_as_float(stats.get("p99")),
        overlap_max=_as_float(stats.get("max")),
        positive_rate=_as_float(stats.get("positive_rate")),
        rate_gt_5pct=_as_float(stats.get("rate_gt_5pct")),
        rate_gt_10pct=_as_float(stats.get("rate_gt_10pct")),
        rate_gt_25pct=_as_float(stats.get("rate_gt_25pct")),
        rate_gt_50pct=_as_float(stats.get("rate_gt_50pct")),
        mean_positive=_as_float(stats.get("mean_positive")),
        worst_stem=worst_cases[0]["stem"] if worst_cases else None,
        worst_overlap=worst_cases[0]["overlap"] if worst_cases else None,
        map50=_as_float(detector.get("mAP50")),
        map50_95=_as_float(detector.get("mAP50_95")),
        precision=_as_float(detector.get("precision")),
        recall=_as_float(detector.get("recall")),
        box_iou=_as_float(dict(box_loss.get("box_iou") or {}).get("mean")),
        channel_dice=_as_float(dict(channel.get("dice") or {}).get("mean")),
        elapsed_seconds=float(summary["elapsed_seconds"]),
        summary_path=summary_path,
        telemetry_path=telemetry_path,
        per_draw=per_draw,
        worst_cases=worst_cases,
    )


def _format_number(value: float | None, digits: int = 4) -> str:
    """Format an optional number for a table.

    Args:
        value (float | None): Number or missing value.
        digits (int): Decimal places.

    Returns:
        str: Formatted number, or ``n/a``.
    """
    return "n/a" if value is None else f"{value:.{digits}f}"


def _format_percent(value: float | None) -> str:
    """Format an optional fraction as a percentage.

    Args:
        value (float | None): Fraction in ``0..1``, or missing.

    Returns:
        str: Percentage with two decimals, or ``n/a``.
    """
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _format_elapsed(seconds: float) -> str:
    """Format a wall-clock duration.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Hours with two decimals.
    """
    return f"{seconds / 3600.0:.2f} h"


def _run_component(component: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run the evaluator for one component and return its manifest entry.

    Args:
        component (str): Component key from ``components.toml``.
        args (argparse.Namespace): Parsed batch-runner options.

    Returns:
        dict[str, Any]: Manifest entry pointing at the new summary and telemetry.
    """
    script_dir: Path = ROOT_DIR / "evaluator"
    config: SimpleNamespace = load_config(
        args.mode,
        str(script_dir / SCRIPT_TOML),
    )
    anchor_path_keys(config, PATH_CONFIG_KEYS)
    hyperparams: SimpleNamespace = load_hyperparams(
        str(script_dir / HYPERPARAMS_TOML)
    )
    hyperparams.SAMPLE_CAP = args.cap
    if args.draws is not None:
        hyperparams.DRAWS_PER_SAMPLE = args.draws
    if args.seed is not None:
        hyperparams.SEED = args.seed

    select_component(config, component)
    split_key: str = SPLIT_CONFIG_KEYS[args.split]
    config.TEST_IMAGES_DIR = getattr(config, split_key)
    log = setup_logger(config)
    log.info(
        f"Batch comparison run_id={config.runtime.RUN_ID}, component={component}, "
        f"split={config.TEST_IMAGES_DIR}, cap={hyperparams.SAMPLE_CAP}, "
        f"draws={hyperparams.DRAWS_PER_SAMPLE}."
    )

    # The normal evaluator CLI builds the same namespace. No output override is
    # used, so each component gets its own eval_<run_id> directory.
    run_args: SimpleNamespace = SimpleNamespace(skip_yolo=False, output_dir=None)
    summary: dict[str, Any] = run_evaluation(run_args, config, hyperparams, log)
    print("\n".join(format_report(summary)))

    output_root: Path = Path(str(summary["output_root"]))
    run_id: str = str(summary["run_id"])
    summary_path: Path = output_root / "evaluation_summary.json"
    telemetry_path: Path = (
        Path(str(config.TELEMETRY_LOG_DIRECTORY)) / f"{run_id}.csv"
    )
    return {
        "component": component,
        "run_id": run_id,
        "split": args.split,
        "split_dir": str(summary["configuration"]["split_dir"]),
        "cap": args.cap,
        "requested_draws": args.draws,
        "requested_seed": args.seed,
        "samples": int(summary["configuration"]["samples_evaluated"]),
        "draws": int(summary["configuration"]["draws_per_sample"]),
        "summary_path": str(summary_path),
        "telemetry_path": str(telemetry_path),
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Build one Markdown table.

    Args:
        headers (list[str]): Column headers.
        rows (list[list[str]]): Row cells, already formatted.

    Returns:
        list[str]: Markdown lines.
    """
    lines: list[str] = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _build_markdown(results: list[ComponentResult]) -> str:
    """Build the Markdown comparison report.

    Args:
        results (list[ComponentResult]): Completed component results.

    Returns:
        str: Markdown document.
    """
    lines: list[str] = [
        "# Missing-component overlap comparison",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
    ]
    if results:
        lines.extend(
            [
                f"Split: `{results[0].split_dir}`",
                f"Samples per component: {results[0].samples}",
                f"Draws per sample: {results[0].draws}",
            ]
        )
    else:
        lines.append("No completed component runs yet.")
    lines.extend(
        [
            "",
            "## How to read",
            "",
            "- **Overlap** is the fraction of the detector's highest-confidence "
            "target box covered by the union of the other named furniture boxes "
            "in the same `labels.txt`.",
            "- **0** is best (no collision); **1** means the target box is fully "
            "covered by existing furniture.",
            "- `n/a` means the detector returned no target box for that sample.",
            "- The headline table uses the **ensemble** result. Per-draw rows are "
            "diagnostics.",
            "- `>0` is any collision, `>5%` is material, `>25%` is large.",
            "",
            "## Primary ensemble results",
            "",
        ]
    )
    if results:
        rows: list[list[str]] = [
            [
                result.component,
                str(result.samples),
                str(result.draws),
                str(result.predicted),
                str(result.missing_prediction),
                _format_number(result.mean_other_gt, 2),
                _format_number(result.overlap_mean),
                _format_number(result.overlap_median),
                _format_number(result.overlap_p90),
                _format_number(result.overlap_p95),
                _format_number(result.overlap_p99),
                _format_number(result.overlap_max),
                _format_percent(result.positive_rate),
                _format_percent(result.rate_gt_5pct),
                _format_percent(result.rate_gt_10pct),
                _format_percent(result.rate_gt_25pct),
                _format_percent(result.rate_gt_50pct),
                _format_number(result.mean_positive),
            ]
            for result in results
        ]
        lines.extend(
            _markdown_table(
                [
                    "Component",
                    "Samples",
                    "Draws",
                    "Predicted",
                    "n/a",
                    "Other GT (mean)",
                    "Mean",
                    "Median",
                    "p90",
                    "p95",
                    "p99",
                    "Max",
                    ">0",
                    ">5%",
                    ">10%",
                    ">25%",
                    ">50%",
                    "Mean positive",
                ],
                rows,
            )
        )
    else:
        lines.append("_No completed runs._")

    lines.extend(["", "## Detector and generation metrics", ""])
    if results:
        rows = [
            [
                result.component,
                _format_number(result.map50),
                _format_number(result.map50_95),
                _format_number(result.precision),
                _format_number(result.recall),
                _format_number(result.box_iou),
                _format_number(result.channel_dice),
                _format_elapsed(result.elapsed_seconds),
            ]
            for result in results
        ]
        lines.extend(
            _markdown_table(
                [
                    "Component",
                    "mAP@50",
                    "mAP@50-95",
                    "Precision",
                    "Recall",
                    "Box IoU",
                    "Channel Dice",
                    "Elapsed",
                ],
                rows,
            )
        )
    else:
        lines.append("_No completed runs._")

    lines.extend(["", "## Per-draw overlap", ""])
    if results:
        rows = []
        for result in results:
            for draw_key in sorted(result.per_draw, key=int):
                stats: dict[str, float | None] = result.per_draw[draw_key]
                label: str = "ensemble" if draw_key == "-1" else f"draw {draw_key}"
                rows.append(
                    [
                        result.component,
                        label,
                        _format_number(_as_float(stats.get("count")), 0),
                        _format_number(_as_float(stats.get("mean"))),
                        _format_number(_as_float(stats.get("p90"))),
                        _format_number(_as_float(stats.get("max"))),
                        _format_percent(_as_float(stats.get("positive_rate"))),
                        _format_percent(_as_float(stats.get("rate_gt_5pct"))),
                        _format_percent(_as_float(stats.get("rate_gt_25pct"))),
                    ]
                )
        lines.extend(
            _markdown_table(
                [
                    "Component",
                    "Result",
                    "Predicted",
                    "Mean",
                    "p90",
                    "Max",
                    ">0",
                    ">5%",
                    ">25%",
                ],
                rows,
            )
        )
    else:
        lines.append("_No completed runs._")

    lines.extend(["", "## Worst ensemble cases", ""])
    if results:
        rows = []
        for result in results:
            for case in result.worst_cases:
                rows.append(
                    [
                        result.component,
                        str(case["stem"]),
                        _format_number(float(case["overlap"])),
                        str(case["other_gt"]),
                        str(case["predictions"]),
                        str(case["matched"]),
                    ]
                )
        if rows:
            lines.extend(
                _markdown_table(
                    [
                        "Component",
                        "Stem",
                        "Overlap",
                        "Other GT",
                        "Predictions",
                        "Matched",
                    ],
                    rows,
                )
            )
        else:
            lines.append("_No positive-overlap cases._")
    else:
        lines.append("_No completed runs._")

    lines.extend(["", "## Artifacts", ""])
    if results:
        rows = [
            [
                result.component,
                f"`{result.summary_path}`",
                f"`{result.telemetry_path}`",
            ]
            for result in results
        ]
        lines.extend(
            _markdown_table(
                ["Component", "Summary", "Telemetry"],
                rows,
            )
        )
    else:
        lines.append("_No completed runs._")

    lines.append("")
    return "\n".join(lines)


def _build_csv(results: list[ComponentResult], path: Path) -> None:
    """Write the headline results as a machine-readable CSV.

    Args:
        results (list[ComponentResult]): Completed component results.
        path (Path): Destination CSV path.
    """
    fieldnames: list[str] = [
        "component",
        "run_id",
        "split_dir",
        "samples",
        "draws",
        "predicted",
        "missing_prediction",
        "mean_other_gt",
        "overlap_mean",
        "overlap_median",
        "overlap_p90",
        "overlap_p95",
        "overlap_p99",
        "overlap_max",
        "positive_rate",
        "rate_gt_5pct",
        "rate_gt_10pct",
        "rate_gt_25pct",
        "rate_gt_50pct",
        "mean_positive",
        "worst_stem",
        "worst_overlap",
        "map50",
        "map50_95",
        "precision",
        "recall",
        "box_iou",
        "channel_dice",
        "elapsed_seconds",
        "summary_path",
        "telemetry_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "component": result.component,
                    "run_id": result.run_id,
                    "split_dir": result.split_dir,
                    "samples": result.samples,
                    "draws": result.draws,
                    "predicted": result.predicted,
                    "missing_prediction": result.missing_prediction,
                    "mean_other_gt": result.mean_other_gt,
                    "overlap_mean": result.overlap_mean,
                    "overlap_median": result.overlap_median,
                    "overlap_p90": result.overlap_p90,
                    "overlap_p95": result.overlap_p95,
                    "overlap_p99": result.overlap_p99,
                    "overlap_max": result.overlap_max,
                    "positive_rate": result.positive_rate,
                    "rate_gt_5pct": result.rate_gt_5pct,
                    "rate_gt_10pct": result.rate_gt_10pct,
                    "rate_gt_25pct": result.rate_gt_25pct,
                    "rate_gt_50pct": result.rate_gt_50pct,
                    "mean_positive": result.mean_positive,
                    "worst_stem": result.worst_stem,
                    "worst_overlap": result.worst_overlap,
                    "map50": result.map50,
                    "map50_95": result.map50_95,
                    "precision": result.precision,
                    "recall": result.recall,
                    "box_iou": result.box_iou,
                    "channel_dice": result.channel_dice,
                    "elapsed_seconds": result.elapsed_seconds,
                    "summary_path": str(result.summary_path),
                    "telemetry_path": str(result.telemetry_path),
                }
            )


def _build_per_draw_csv(results: list[ComponentResult], path: Path) -> None:
    """Write per-draw overlap statistics as a CSV.

    Args:
        results (list[ComponentResult]): Completed component results.
        path (Path): Destination CSV path.
    """
    fieldnames: list[str] = [
        "component",
        "draw",
        "predicted",
        "mean",
        "median",
        "p90",
        "p95",
        "p99",
        "max",
        "positive_rate",
        "rate_gt_5pct",
        "rate_gt_10pct",
        "rate_gt_25pct",
        "rate_gt_50pct",
        "mean_positive",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for draw_key in sorted(result.per_draw, key=int):
                stats = result.per_draw[draw_key]
                writer.writerow(
                    {
                        "component": result.component,
                        "draw": draw_key,
                        "predicted": int(float(stats.get("count", 0.0) or 0.0)),
                        "mean": stats.get("mean"),
                        "median": stats.get("median"),
                        "p90": stats.get("p90"),
                        "p95": stats.get("p95"),
                        "p99": stats.get("p99"),
                        "max": stats.get("max"),
                        "positive_rate": stats.get("positive_rate"),
                        "rate_gt_5pct": stats.get("rate_gt_5pct"),
                        "rate_gt_10pct": stats.get("rate_gt_10pct"),
                        "rate_gt_25pct": stats.get("rate_gt_25pct"),
                        "rate_gt_50pct": stats.get("rate_gt_50pct"),
                        "mean_positive": stats.get("mean_positive"),
                    }
                )


def _overlap_class(value: float | None) -> str:
    """Return a CSS class for one overlap value.

    Args:
        value (float | None): Overlap fraction.

    Returns:
        str: CSS class name; ``na`` for a missing value.
    """
    if value is None:
        return "na"
    if value < 0.01:
        return "good"
    if value < 0.05:
        return "ok"
    if value < 0.10:
        return "warn"
    return "bad"


def _html_table(
    headers: list[str],
    rows: list[list[tuple[str, str]]],
) -> str:
    """Build one HTML table.

    Args:
        headers (list[str]): Column headers.
        rows (list[list[tuple[str, str]]]): Cells as ``(text, css_class)``.

    Returns:
        str: HTML table.
    """
    header_html: str = "".join(
        f"<th>{html.escape(header)}</th>" for header in headers
    )
    body_html: str = ""
    for row in rows:
        cells: str = "".join(
            f'<td class="{html.escape(css_class)}">{html.escape(text)}</td>'
            for text, css_class in row
        )
        body_html += f"<tr>{cells}</tr>"
    return (
        "<table>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
    )


def _build_html(results: list[ComponentResult]) -> str:
    """Build a standalone, color-coded HTML report.

    Args:
        results (list[ComponentResult]): Completed component results.

    Returns:
        str: Complete HTML document.
    """
    sections: list[str] = []
    if results:
        headline_rows: list[list[tuple[str, str]]] = []
        for result in results:
            headline_rows.append(
                [
                    (result.component, ""),
                    (str(result.samples), ""),
                    (str(result.draws), ""),
                    (str(result.predicted), ""),
                    (str(result.missing_prediction), ""),
                    (_format_number(result.mean_other_gt, 2), ""),
                    (_format_number(result.overlap_mean), _overlap_class(result.overlap_mean)),
                    (_format_number(result.overlap_median), _overlap_class(result.overlap_median)),
                    (_format_number(result.overlap_p90), _overlap_class(result.overlap_p90)),
                    (_format_number(result.overlap_p95), _overlap_class(result.overlap_p95)),
                    (_format_number(result.overlap_p99), _overlap_class(result.overlap_p99)),
                    (_format_number(result.overlap_max), _overlap_class(result.overlap_max)),
                    (_format_percent(result.positive_rate), ""),
                    (_format_percent(result.rate_gt_5pct), ""),
                    (_format_percent(result.rate_gt_10pct), ""),
                    (_format_percent(result.rate_gt_25pct), ""),
                    (_format_percent(result.rate_gt_50pct), ""),
                    (_format_number(result.mean_positive), _overlap_class(result.mean_positive)),
                ]
            )
        sections.append("<h2>Primary ensemble results</h2>")
        sections.append(
            _html_table(
                [
                    "Component",
                    "Samples",
                    "Draws",
                    "Predicted",
                    "n/a",
                    "Other GT (mean)",
                    "Mean",
                    "Median",
                    "p90",
                    "p95",
                    "p99",
                    "Max",
                    ">0",
                    ">5%",
                    ">10%",
                    ">25%",
                    ">50%",
                    "Mean positive",
                ],
                headline_rows,
            )
        )

        detector_rows: list[list[tuple[str, str]]] = [
            [
                (result.component, ""),
                (_format_number(result.map50), ""),
                (_format_number(result.map50_95), ""),
                (_format_number(result.precision), ""),
                (_format_number(result.recall), ""),
                (_format_number(result.box_iou), ""),
                (_format_number(result.channel_dice), ""),
                (_format_elapsed(result.elapsed_seconds), ""),
            ]
            for result in results
        ]
        sections.append("<h2>Detector and generation metrics</h2>")
        sections.append(
            _html_table(
                [
                    "Component",
                    "mAP@50",
                    "mAP@50-95",
                    "Precision",
                    "Recall",
                    "Box IoU",
                    "Channel Dice",
                    "Elapsed",
                ],
                detector_rows,
            )
        )

        per_draw_rows: list[list[tuple[str, str]]] = []
        for result in results:
            for draw_key in sorted(result.per_draw, key=int):
                stats = result.per_draw[draw_key]
                label: str = "ensemble" if draw_key == "-1" else f"draw {draw_key}"
                per_draw_rows.append(
                    [
                        (result.component, ""),
                        (label, ""),
                        (_format_number(_as_float(stats.get("count")), 0), ""),
                        (_format_number(_as_float(stats.get("mean"))), _overlap_class(_as_float(stats.get("mean")))),
                        (_format_number(_as_float(stats.get("p90"))), _overlap_class(_as_float(stats.get("p90")))),
                        (_format_number(_as_float(stats.get("max"))), _overlap_class(_as_float(stats.get("max")))),
                        (_format_percent(_as_float(stats.get("positive_rate"))), ""),
                        (_format_percent(_as_float(stats.get("rate_gt_5pct"))), ""),
                        (_format_percent(_as_float(stats.get("rate_gt_25pct"))), ""),
                    ]
                )
        sections.append("<h2>Per-draw overlap</h2>")
        sections.append(
            _html_table(
                [
                    "Component",
                    "Result",
                    "Predicted",
                    "Mean",
                    "p90",
                    "Max",
                    ">0",
                    ">5%",
                    ">25%",
                ],
                per_draw_rows,
            )
        )

        worst_rows: list[list[tuple[str, str]]] = []
        for result in results:
            for case in result.worst_cases:
                worst_rows.append(
                    [
                        (result.component, ""),
                        (str(case["stem"]), ""),
                        (_format_number(float(case["overlap"])), _overlap_class(float(case["overlap"]))),
                        (str(case["other_gt"]), ""),
                        (str(case["predictions"]), ""),
                        (str(case["matched"]), ""),
                    ]
                )
        sections.append("<h2>Worst ensemble cases</h2>")
        if worst_rows:
            sections.append(
                _html_table(
                    [
                        "Component",
                        "Stem",
                        "Overlap",
                        "Other GT",
                        "Predictions",
                        "Matched",
                    ],
                    worst_rows,
                )
            )
        else:
            sections.append("<p>No positive-overlap cases.</p>")

        artifacts_rows: list[list[tuple[str, str]]] = [
            [
                (result.component, ""),
                (str(result.summary_path), ""),
                (str(result.telemetry_path), ""),
            ]
            for result in results
        ]
        sections.append("<h2>Artifacts</h2>")
        sections.append(
            _html_table(
                ["Component", "Summary", "Telemetry"],
                artifacts_rows,
            )
        )
    else:
        sections.append("<p>No completed component runs yet.</p>")

    generated: str = datetime.now().astimezone().isoformat(timespec="seconds")
    split: str = results[0].split_dir if results else "n/a"
    samples: str = str(results[0].samples) if results else "n/a"
    draws: str = str(results[0].draws) if results else "n/a"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Missing-component overlap comparison</title>
<style>
  :root {{ --bg:#f8fafc; --panel:#ffffff; --text:#0f172a; --muted:#475569; --border:#cbd5e1; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0f1117; --panel:#171a23; --text:#e6e9ef; --muted:#98a2b3; --border:#334155; }}
  }}
  body {{ margin:0; background:var(--bg); color:var(--text);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
  main {{ max-width:1600px; margin:0 auto; padding:28px 18px 64px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  h2 {{ font-size:16px; margin:28px 0 10px; color:#0d9488; }}
  p {{ color:var(--muted); max-width:1100px; }}
  table {{ border-collapse:collapse; width:100%; margin:8px 0 20px; background:var(--panel);
           border:1px solid var(--border); font-size:13px; }}
  th, td {{ border:1px solid var(--border); padding:6px 8px; text-align:right; white-space:nowrap; }}
  th:first-child, td:first-child {{ text-align:left; }}
  thead th {{ position:sticky; top:0; background:var(--panel); z-index:1; }}
  tbody tr:nth-child(even) {{ background:rgba(148, 163, 184, 0.10); }}
  td.good {{ background:#bbf7d0; color:#052e16; }}
  td.ok {{ background:#fef08a; color:#422006; }}
  td.warn {{ background:#fed7aa; color:#431407; }}
  td.bad {{ background:#fecaca; color:#450a0a; }}
  td.na {{ color:var(--muted); }}
  code {{ font-size:12px; }}
</style>
</head>
<body>
<main>
<h1>Missing-component overlap comparison</h1>
<p>
  <strong>Overlap</strong> is the fraction of the detector's highest-confidence
  target box covered by the union of the other named furniture boxes in the same
  <code>labels.txt</code>. <strong>0</strong> is best; <strong>1</strong> means
  the target box is fully covered. <code>n/a</code> means no target box was
  predicted. The headline table uses the <strong>ensemble</strong>; per-draw rows
  are diagnostics.
</p>
<p>
  Generated: {html.escape(generated)}<br>
  Split: <code>{html.escape(split)}</code><br>
  Samples per component: {html.escape(samples)} &middot;
  Draws per sample: {html.escape(draws)}
</p>
{''.join(sections)}
</main>
</body>
</html>
"""


def _write_tables(results: list[ComponentResult], table_dir: Path) -> dict[str, Path]:
    """Write the Markdown, CSV and HTML comparison artifacts.

    Args:
        results (list[ComponentResult]): Completed component results.
        table_dir (Path): Destination directory.

    Returns:
        dict[str, Path]: Written artifact paths keyed by kind.
    """
    table_dir.mkdir(parents=True, exist_ok=True)
    markdown_path: Path = table_dir / f"{TABLE_STEM}.md"
    csv_path: Path = table_dir / f"{TABLE_STEM}.csv"
    per_draw_path: Path = table_dir / f"{PER_DRAW_STEM}.csv"
    html_path: Path = table_dir / f"{TABLE_STEM}.html"

    markdown_path.write_text(_build_markdown(results), encoding="utf-8")
    _build_csv(results, csv_path)
    _build_per_draw_csv(results, per_draw_path)
    html_path.write_text(_build_html(results), encoding="utf-8")
    return {
        "markdown": markdown_path,
        "csv": csv_path,
        "per_draw_csv": per_draw_path,
        "html": html_path,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the requested components and keep the comparison table up to date.

    Args:
        argv (list[str] | None): Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        int: Process exit code, ``0`` on success.
    """
    args: argparse.Namespace = parse_arguments(argv)
    known_components: list[str] = component_names()
    components: list[str] = (
        list(args.components) if args.components is not None else known_components
    )
    unknown: list[str] = [name for name in components if name not in known_components]
    if unknown:
        raise KeyError(
            f"Unknown component(s) {unknown}; choose from {known_components}."
        )

    table_dir: Path = (
        Path(args.table_dir).expanduser().resolve()
        if args.table_dir is not None
        else ROOT_DIR / "evaluator" / "outputs" / f"{args.split}_comparison"
    )
    manifest_path: Path = table_dir / MANIFEST_NAME
    manifest: dict[str, dict[str, Any]] = _load_manifest(manifest_path)

    if args.table_only:
        results: list[ComponentResult] = [
            _component_result(manifest[component])
            for component in components
            if component in manifest and _entry_is_complete(manifest[component])
        ]
        paths: dict[str, Path] = _write_tables(results, table_dir)
        print(f"Table written to {paths['html']}")
        print(f"Markdown table: {paths['markdown']}")
        print(f"CSV: {paths['csv']}")
        print(f"Per-draw CSV: {paths['per_draw_csv']}")
        return 0

    for component in components:
        if (
            args.skip_existing
            and component in manifest
            and _entry_is_complete(manifest[component])
            and _entry_matches(manifest[component], args)
        ):
            print(f"Skipping {component}: completed run already recorded.")
        else:
            entry: dict[str, Any] = _run_component(component, args)
            manifest[component] = entry
            _write_manifest(manifest_path, manifest)

        # Rebuild after every component so a long batch always leaves a usable
        # partial table behind.
        results = [
            _component_result(manifest[name])
            for name in components
            if name in manifest and _entry_is_complete(manifest[name])
        ]
        paths = _write_tables(results, table_dir)
        print(f"Updated table after {component}: {paths['html']}")

    print(f"\nFinal table: {table_dir / f'{TABLE_STEM}.html'}")
    print(f"Markdown: {table_dir / f'{TABLE_STEM}.md'}")
    print(f"CSV: {table_dir / f'{TABLE_STEM}.csv'}")
    print(f"Per-draw CSV: {table_dir / f'{PER_DRAW_STEM}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
