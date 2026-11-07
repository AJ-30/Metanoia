"""Telemetry for the evaluator: one row per sample/result plus the final summary.

Two artifacts are written:

* ``evaluator/telemetry_logs/<run_id>.csv`` - one row per (sample, result),
  where a result is either a draw (``draw`` = 0..N-1) or the ensemble
  (``draw`` = -1), holding the ground-truth box count, the detector's
  prediction count, the matched / missed / false-positive counts, the
  localization losses of the best-matched box, the channel Dice/IoU and the
  probability/mask statistics. The dashboard discovers this file like any other
  run's telemetry, so the per-sample numbers are plottable without extra wiring.
* ``<output>/eval_<run_id>/evaluation_summary.json`` - the aggregate report: the
  primary ensemble's detector metrics for the missing component, the per-draw
  diagnostics, the box-loss distributions, the channel scores and everything
  needed to reproduce the run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from helpers.telemetry_writer import TelemetryWriter

# Column order of the per-sample CSV. Kept here so the writer and the summary's
# documentation cannot drift apart.
TELEMETRY_FIELDNAMES: tuple[str, ...] = (
    "stem",
    "draw",
    "gt_boxes",
    "pred_boxes",
    "matched_boxes",
    "missed_boxes",
    "false_positives",
    "best_iou",
    "best_l1_norm",
    "best_center_distance",
    "channel_dice",
    "channel_iou",
    "generated_pixels",
    "probability_mean",
)

# Suffix of the summary file. The dashboard reads the montages by their
# ``eval_`` prefix; this file carries the numbers themselves.
SUMMARY_FILENAME: str = "evaluation_summary.json"


class SampleTelemetry:
    """Appends one row per evaluated (sample, result) pair to the run CSV.

    Args:
        directory (str): Telemetry directory (``TELEMETRY_LOG_DIRECTORY``).
        config (SimpleNamespace): Config carrying ``runtime.RUN_ID`` and
            ``runtime.START_TIME``.
    """

    def __init__(self, directory: str, config: SimpleNamespace) -> None:
        self.writer: TelemetryWriter = TelemetryWriter(
            list(TELEMETRY_FIELDNAMES), directory=directory, config=config
        )

    def log(self, **metrics: Any) -> None:
        """Append one sample row and flush it.

        Args:
            **metrics (Any): Values for every field of
                :data:`TELEMETRY_FIELDNAMES`.
        """
        self.writer.log(**metrics)


def write_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    """Write the aggregate evaluation report as JSON.

    Args:
        summary_path (Path): Destination file.
        summary (dict[str, Any]): Everything the pipeline accumulated.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )


def write_report_table(path: Path, lines: list[str]) -> None:
    """Write the human-readable report next to the montages.

    Args:
        path (Path): Destination text file.
        lines (list[str]): Report lines, already formatted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
