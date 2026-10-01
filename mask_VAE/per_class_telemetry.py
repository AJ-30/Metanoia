"""
Per-class telemetry CSV for mask_VAE.

A single averaged loss tells the developer almost nothing about a 21-class mask
model: the loss can fall steadily while one rare class, say ``stair``, is never
predicted at all. This writer records one IoU and one Dice column per class, so
the class that is holding the model back is named rather than guessed.

The file is a plain CSV next to the main telemetry CSV, named
``<run_id>_per_class.csv``, with ``timestamp`` and ``runtime`` in front exactly
like :class:`helpers.telemetry_writer.TelemetryWriter` writes them, so both files
can be read by the same tools.
"""

from __future__ import annotations

import csv
import os
import time
from types import SimpleNamespace
from typing import Any

# Suffix appended to the run id to name the per-class file. Kept next to the
# other constants so the main and per-class files are obviously the same run.
PER_CLASS_FILE_SUFFIX: str = "_per_class"


class PerClassTelemetryWriter:
    """
    Append-only CSV writer with one column per class.

    Args:
        class_names (list[str]): Column names, one per class, in the order the
            per-class values are passed to :meth:`log`. Must be unique and
            non-empty, since they are written verbatim as a CSV header.
        directory (str): Directory the CSV is written to. Created if missing.
        config (SimpleNamespace): Active config, used only for the run id and
            the run start time.

    Raises:
        AssertionError: If the config has no run id yet.
        ValueError: If ``class_names`` is empty or contains duplicates.
    """

    def __init__(
        self,
        class_names: list[str],
        directory: str,
        config: SimpleNamespace,
    ) -> None:
        if not class_names:
            raise ValueError("Per-class telemetry needs at least one class column.")
        if len(set(class_names)) != len(class_names):
            duplicates: list[str] = sorted(
                {name for name in class_names if class_names.count(name) > 1}
            )
            raise ValueError(
                f"Per-class telemetry column names must be unique, got duplicates "
                f"{duplicates}. Mask class names are prefixed with their group "
                "for exactly this reason."
            )

        assert (
            config.runtime.RUN_ID is not None
        ), "RUN_ID must be set in runtime before setting up the per-class writer."

        self.config: SimpleNamespace = config
        self.class_names: list[str] = list(class_names)
        os.makedirs(directory, exist_ok=True)

        self.filepath: str = os.path.join(
            directory, f"{config.runtime.RUN_ID}{PER_CLASS_FILE_SUFFIX}.csv"
        )
        file_exists: bool = os.path.isfile(self.filepath)

        # Kept open for the lifetime of the run and flushed on every row, the
        # same deliberate trade-off as helpers/telemetry_writer.py: a crash
        # mid-training still leaves every epoch that finished on disk.
        self.file = open(self.filepath, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            # One "iou/<class>" and one "dice/<class>" column per class keeps the
            # two metrics from interleaving in a wide row.
            fieldnames=["timestamp", "runtime", "epoch"]
            + [f"iou/{name}" for name in self.class_names]
            + [f"dice/{name}" for name in self.class_names],
        )
        if not file_exists:
            self.writer.writeheader()
            self.file.flush()

    def log(self, epoch: int, iou: list[float], dice: list[float]) -> None:
        """
        Append one row of per-class metrics.

        Args:
            epoch (int): Epoch the metrics were measured at.
            iou (list[float]): Per-class IoU, ordered like ``class_names``.
            dice (list[float]): Per-class Dice, ordered like ``class_names``.

        Raises:
            AssertionError: If the writer was never constructed.
            ValueError: If either list is not one value per class.
        """
        if len(iou) != len(self.class_names) or len(dice) != len(self.class_names):
            raise ValueError(
                f"Expected {len(self.class_names)} per-class values, got "
                f"{len(iou)} IoU and {len(dice)} Dice values."
            )

        row: dict[str, Any] = {
            "timestamp": time.time(),
            "runtime": time.time() - self.config.runtime.START_TIME,
            "epoch": epoch,
        }
        for name, iou_value, dice_value in zip(self.class_names, iou, dice, strict=True):
            row[f"iou/{name}"] = iou_value
            row[f"dice/{name}"] = dice_value
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        """Flush and close the underlying file."""
        self.file.close()
