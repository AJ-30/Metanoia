"""Extra validation tables: confusion matrices, false-positive means, precision sweep.

The main validation report already prints the official Ultralytics confusion
matrix and the configured threshold table. This module adds the three tables a
reader reaches for next, and writes them as both CSV and a readable text block:

* a **box-level** and **image-level confusion matrix** at every configured
  confidence threshold;
* a **false-positive summary** whose counts are averaged over the run (per
  image, per positive image, per background image, and per image that actually
  produced a false positive);
* a **precision table** over a finer confidence sweep, so precision/recall can
  be read as a curve rather than at three points.

The arithmetic is the same greedy IoU matching the rest of the validation report
uses. Nothing here loads a model or rerenders an image, so the tables can be
regenerated from an existing report's saved labels and predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

# Project root so ``evaluator``/``helpers``/``missing_yolo`` import when run as a
# script from any directory.
_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_ROOT_DIR: Path = _SCRIPT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from evaluator.box_metrics import (  # noqa: E402
    MatchOutcome,
    PredictedBox,
    intersection_over_union,
    match_boxes,
)
from evaluator.dataset_gt import Box  # noqa: E402
from helpers.config import load_config  # noqa: E402
from helpers.hyperparams import load_hyperparams  # noqa: E402
from missing_yolo.validation_metrics import (  # noqa: E402
    read_ground_truth_labels,
    read_prediction_labels,
)

# CSV names written beside metrics.json.
CONFUSION_MATRIX_CSV: str = "confusion_matrix.csv"
FALSE_POSITIVE_CSV: str = "false_positive_summary.csv"
PRECISION_TABLE_CSV: str = "precision_table.csv"
TABLES_TXT: str = "tables.txt"

# Column order of the false-positive summary CSV.
_FALSE_POSITIVE_FIELDS: tuple[str, ...] = (
    "conf_threshold",
    "fp_boxes",
    "total_images",
    "positive_images",
    "background_images",
    "mean_fp_per_image",
    "mean_fp_per_positive_image",
    "mean_fp_per_background_image",
    "mean_fp_per_image_with_fp",
    "images_with_fp",
    "fraction_images_with_fp",
    "median_fp_per_image",
    "p90_fp_per_image",
    "max_fp_per_image",
    "mean_fp_confidence",
    "mean_fp_best_iou",
)

# Column order of the precision-sweep CSV.
_PRECISION_FIELDS: tuple[str, ...] = (
    "conf_threshold",
    "tp",
    "fp",
    "fn",
    "ground_truth_boxes",
    "predicted_boxes",
    "precision",
    "recall",
    "f1",
    "jaccard",
    "positive_images",
    "detected_positive_images",
    "missed_positive_images",
    "background_images",
    "false_alarm_images",
    "correct_empty_images",
    "image_accuracy",
    "positive_detection_rate",
    "false_alarm_rate",
    "specificity",
    "balanced_accuracy",
)


def _ratio(numerator: float, denominator: float) -> float | None:
    """Return ``numerator / denominator``, or None for a zero denominator.

    Args:
        numerator (float): Count or value on top.
        denominator (float): Count or value below.

    Returns:
        float | None: The ratio, or ``None`` when it is undefined.
    """
    return None if denominator == 0 else numerator / denominator


def _harmonic_mean(precision: float | None, recall: float | None) -> float | None:
    """Return the harmonic mean of precision and recall.

    Args:
        precision (float | None): Box precision, or None when undefined.
        recall (float | None): Box recall, or None when undefined.

    Returns:
        float | None: F1, or ``None`` when either input is undefined.
    """
    if precision is None or recall is None:
        return None
    denominator: float = precision + recall
    return None if denominator == 0.0 else 2.0 * precision * recall / denominator


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    """Return a nearest-rank percentile of an already sorted sequence.

    Args:
        ordered (Sequence[float]): Sorted values.
        fraction (float): Percentile as ``0..1``.

    Returns:
        float: The selected value.
    """
    return float(ordered[max(0, int(fraction * (len(ordered) - 1)))])


def _optional_median(values: Sequence[float]) -> float | None:
    """Return the median of a sequence, or None when it is empty.

    Args:
        values (Sequence[float]): Measurements.

    Returns:
        float | None: Median, or ``None`` for an empty sequence.
    """
    return statistics.median(values) if values else None


def _optional_percentile(values: Sequence[float], fraction: float) -> float | None:
    """Return a nearest-rank percentile, or None when there are no values.

    Args:
        values (Sequence[float]): Measurements.
        fraction (float): Percentile as ``0..1``.

    Returns:
        float | None: The selected value, or ``None`` for an empty sequence.
    """
    if not values:
        return None
    return _percentile(sorted(float(value) for value in values), fraction)


@dataclass
class _ThresholdAccumulator:
    """Running counts for one confidence threshold.

    Attributes:
        conf_threshold (float): Predictions below this confidence are ignored.
        iou_threshold (float): Minimum IoU for a prediction to claim a GT box.
        image_size (int): Render side length for pixel distances.
        tp (int): Matched ground-truth boxes.
        fp (int): Unmatched predictions.
        fn (int): Unmatched ground-truth boxes.
        positive_images (int): Images with at least one GT box.
        detected_positive_images (int): Positive images with at least one TP.
        background_images (int): Images with no GT box.
        false_alarm_images (int): Background images with at least one FP.
        correct_empty_images (int): Background images with no FP.
        fp_counts_all (list[int]): FP count of every image.
        fp_counts_positive (list[int]): FP count of every positive image.
        fp_counts_background (list[int]): FP count of every background image.
        fp_confidence_sum (float): Sum of every FP confidence.
        fp_best_iou_sum (float): Sum of every FP's best IoU against GT.
        fp_best_iou_count (int): FPs that had at least one GT box to compare to.
    """

    conf_threshold: float
    iou_threshold: float
    image_size: int
    tp: int = 0
    fp: int = 0
    fn: int = 0
    positive_images: int = 0
    detected_positive_images: int = 0
    background_images: int = 0
    false_alarm_images: int = 0
    correct_empty_images: int = 0
    fp_counts_all: list[int] = field(default_factory=list)
    fp_counts_positive: list[int] = field(default_factory=list)
    fp_counts_background: list[int] = field(default_factory=list)
    fp_confidence_sum: float = 0.0
    fp_best_iou_sum: float = 0.0
    fp_best_iou_count: int = 0

    def add(
        self,
        ground_truth: Sequence[Box],
        predictions: Sequence[PredictedBox],
    ) -> None:
        """Fold one image into the running totals.

        Args:
            ground_truth (Sequence[Box]): The image's GT boxes.
            predictions (Sequence[PredictedBox]): All predictions of the image;
                those below the threshold are dropped here.
        """
        kept: list[PredictedBox] = [
            prediction
            for prediction in predictions
            if prediction.confidence >= self.conf_threshold
        ]
        outcome: MatchOutcome = match_boxes(
            list(ground_truth), kept, self.iou_threshold, self.image_size
        )
        tp: int = sum(1 for match in outcome.matches if match.loss is not None)
        fp: int = len(outcome.false_positives)
        fn: int = len(ground_truth) - tp
        self.tp += tp
        self.fp += fp
        self.fn += fn

        is_positive: bool = bool(ground_truth)
        if is_positive:
            self.positive_images += 1
            if tp > 0:
                self.detected_positive_images += 1
            self.fp_counts_positive.append(fp)
        else:
            self.background_images += 1
            if fp > 0:
                self.false_alarm_images += 1
            else:
                self.correct_empty_images += 1
            self.fp_counts_background.append(fp)
        self.fp_counts_all.append(fp)

        for prediction in outcome.false_positives:
            self.fp_confidence_sum += prediction.confidence
            if ground_truth:
                best_iou: float = max(
                    intersection_over_union(gt_box, prediction.box)
                    for gt_box in ground_truth
                )
                self.fp_best_iou_sum += best_iou
                self.fp_best_iou_count += 1

    def box_confusion(self, class_name: str) -> dict[str, Any]:
        """Return the box-level 2x2 confusion matrix.

        Args:
            class_name (str): Component/class name used as the first label.

        Returns:
            dict[str, Any]: Labels, row-major matrix, and TP/FP/FN/TN counts.
            ``TN`` is zero because background-to-background box pairs are not
            measured for object detection.
        """
        return {
            "labels": [class_name, "background"],
            "matrix": [[self.tp, self.fn], [self.fp, 0]],
            "counts": {
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
                "tn": 0,
            },
        }

    def image_confusion(self) -> dict[str, Any]:
        """Return the image-level 2x2 confusion matrix.

        A positive image is TP when at least one GT box was matched and FN when
        nothing matched. A background image is FP when any prediction was made
        and TN when it stayed empty.

        Returns:
            dict[str, Any]: Labels, row-major matrix, and TP/FP/FN/TN counts.
        """
        missed: int = self.positive_images - self.detected_positive_images
        return {
            "labels": ["positive", "background"],
            "matrix": [
                [self.detected_positive_images, missed],
                [self.false_alarm_images, self.correct_empty_images],
            ],
            "counts": {
                "tp": self.detected_positive_images,
                "fp": self.false_alarm_images,
                "fn": missed,
                "tn": self.correct_empty_images,
            },
        }

    def false_positive_summary(self) -> dict[str, Any]:
        """Return aggregate FP statistics, averaged over the whole run.

        Returns:
            dict[str, Any]: Totals, per-image means, distribution of the
                per-image FP count, and mean FP confidence/best IoU.
        """
        total_images: int = self.positive_images + self.background_images
        images_with_fp: int = sum(1 for count in self.fp_counts_all if count > 0)
        return {
            "conf_threshold": float(self.conf_threshold),
            "fp_boxes": self.fp,
            "total_images": total_images,
            "positive_images": self.positive_images,
            "background_images": self.background_images,
            "mean_fp_per_image": _ratio(self.fp, total_images),
            "mean_fp_per_positive_image": _ratio(
                sum(self.fp_counts_positive), self.positive_images
            ),
            "mean_fp_per_background_image": _ratio(
                sum(self.fp_counts_background), self.background_images
            ),
            "mean_fp_per_image_with_fp": _ratio(self.fp, images_with_fp),
            "images_with_fp": images_with_fp,
            "fraction_images_with_fp": _ratio(images_with_fp, total_images),
            "median_fp_per_image": _optional_median(self.fp_counts_all),
            "p90_fp_per_image": _optional_percentile(self.fp_counts_all, 0.90),
            "max_fp_per_image": (
                max(self.fp_counts_all) if self.fp_counts_all else None
            ),
            "mean_fp_confidence": _ratio(
                self.fp_confidence_sum, self.fp
            ),
            "mean_fp_best_iou": _ratio(
                self.fp_best_iou_sum, self.fp_best_iou_count
            ),
        }

    def precision_row(self) -> dict[str, Any]:
        """Return one row of the precision sweep.

        Returns:
            dict[str, Any]: Counts and rates at this confidence threshold.
        """
        precision: float | None = _ratio(self.tp, self.tp + self.fp)
        recall: float | None = _ratio(self.tp, self.tp + self.fn)
        f1: float | None = _harmonic_mean(precision, recall)
        jaccard: float | None = _ratio(self.tp, self.tp + self.fp + self.fn)
        total_images: int = self.positive_images + self.background_images
        missed: int = self.positive_images - self.detected_positive_images
        image_accuracy: float | None = _ratio(
            self.detected_positive_images + self.correct_empty_images,
            total_images,
        )
        detection_rate: float | None = _ratio(
            self.detected_positive_images, self.positive_images
        )
        false_alarm_rate: float | None = _ratio(
            self.false_alarm_images, self.background_images
        )
        specificity: float | None = _ratio(
            self.correct_empty_images, self.background_images
        )
        balanced_accuracy: float | None = (
            None
            if detection_rate is None or specificity is None
            else (detection_rate + specificity) / 2.0
        )
        return {
            "conf_threshold": float(self.conf_threshold),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "ground_truth_boxes": self.tp + self.fn,
            "predicted_boxes": self.tp + self.fp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": jaccard,
            "positive_images": self.positive_images,
            "detected_positive_images": self.detected_positive_images,
            "missed_positive_images": missed,
            "background_images": self.background_images,
            "false_alarm_images": self.false_alarm_images,
            "correct_empty_images": self.correct_empty_images,
            "image_accuracy": image_accuracy,
            "positive_detection_rate": detection_rate,
            "false_alarm_rate": false_alarm_rate,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
        }


def _validate_thresholds(thresholds: Sequence[float], name: str) -> list[float]:
    """Validate and sort a confidence-threshold sequence.

    Args:
        thresholds (Sequence[float]): Confidence thresholds.
        name (str): Human-readable name for error messages.

    Returns:
        list[float]: Unique thresholds in ascending order.

    Raises:
        ValueError: If the list is empty or a value is outside ``0..1``.
    """
    values: list[float] = sorted({float(value) for value in thresholds})
    if not values:
        raise ValueError(f"{name} must contain at least one confidence threshold.")
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} contains {value}, outside the 0..1 confidence range."
            )
    return values


def build_validation_tables(
    stems: Sequence[str],
    ground_truth_dir: Path,
    prediction_dir: Path,
    report_thresholds: Sequence[float],
    precision_thresholds: Sequence[float],
    iou_threshold: float,
    image_size: int,
    class_id: int = 0,
    class_name: str = "object",
) -> dict[str, Any]:
    """Build confusion matrices, FP means and the precision table for one split.

    Args:
        stems (Sequence[str]): Sample stems to evaluate, in dataset order.
        ground_truth_dir (Path): Rendered ``labels/val`` directory.
        prediction_dir (Path): Ultralytics ``save_txt`` labels directory.
        report_thresholds (Sequence[float]): Thresholds for the confusion
            matrices and FP summary.
        precision_thresholds (Sequence[float]): Thresholds for the precision
            sweep. These may be a finer superset of ``report_thresholds``.
        iou_threshold (float): Minimum IoU for a prediction to claim a box.
        image_size (int): Render side length for pixel distances.
        class_id (int): Only this class id is scored.
        class_name (str): Component/class name for matrix labels.

    Returns:
        dict[str, Any]: ``confusion_matrix``, ``false_positive_summary`` and
        ``precision_table`` sections, all JSON-serializable.

    Raises:
        ValueError: If the split is empty, a threshold is outside ``0..1``, or
            the IoU threshold is outside ``0..1``.
    """
    if not stems:
        raise ValueError("Cannot build validation tables for an empty split.")
    report_values: list[float] = _validate_thresholds(
        report_thresholds, "report_thresholds"
    )
    precision_values: list[float] = _validate_thresholds(
        precision_thresholds, "precision_thresholds"
    )
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError(f"Match IoU threshold {iou_threshold} is outside 0..1.")
    if int(image_size) <= 0:
        raise ValueError(f"Image size must be positive, got {image_size}.")

    all_thresholds: list[float] = sorted(set(report_values) | set(precision_values))
    accumulators: dict[float, _ThresholdAccumulator] = {
        threshold: _ThresholdAccumulator(
            conf_threshold=threshold,
            iou_threshold=float(iou_threshold),
            image_size=int(image_size),
        )
        for threshold in all_thresholds
    }

    for stem in stems:
        ground_truth: list[Box] = read_ground_truth_labels(
            ground_truth_dir / f"{stem}.txt", class_id
        )
        predictions: list[PredictedBox] = read_prediction_labels(
            prediction_dir / f"{stem}.txt", class_id
        )
        for threshold in all_thresholds:
            accumulators[threshold].add(ground_truth, predictions)

    return {
        "confusion_matrix": {
            "box": {
                f"{threshold:g}": accumulators[threshold].box_confusion(class_name)
                for threshold in report_values
            },
            "image": {
                f"{threshold:g}": accumulators[threshold].image_confusion()
                for threshold in report_values
            },
        },
        "false_positive_summary": {
            f"{threshold:g}": accumulators[threshold].false_positive_summary()
            for threshold in report_values
        },
        "precision_table": [
            accumulators[threshold].precision_row()
            for threshold in precision_values
        ],
    }


def _format_optional(value: float | None, digits: int = 4) -> str:
    """Format an optional number for the text report.

    Args:
        value (float | None): Number or None.
        digits (int): Decimal places.

    Returns:
        str: The formatted number, or ``"n/a"``.
    """
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _matrix_lines(matrix: dict[str, Any], title: str) -> list[str]:
    """Format one 2x2 confusion matrix as text lines.

    Args:
        matrix (dict[str, Any]): Matrix with ``labels`` and ``matrix``.
        title (str): Section title.

    Returns:
        list[str]: Title, header and two matrix rows.
    """
    labels: list[str] = [str(label) for label in matrix["labels"]]
    width: int = max(len(label) for label in labels) + 2
    lines: list[str] = [title]
    lines.append(" " * width + "".join(f"{label:>14}" for label in labels))
    for label, row in zip(labels, matrix["matrix"], strict=True):
        lines.append(
            f"{label:<{width}}" + "".join(f"{int(value):>14}" for value in row)
        )
    counts: dict[str, Any] = matrix["counts"]
    lines.append(
        "  counts: "
        f"TP {counts['tp']} | FP {counts['fp']} | "
        f"FN {counts['fn']} | TN {counts['tn']}"
    )
    return lines


def format_tables_report(
    tables: dict[str, Any],
    class_name: str,
    iou_threshold: float,
    official_confusion: dict[str, Any] | None = None,
    official_conf_threshold: float | None = None,
) -> list[str]:
    """Format the extra tables as a readable text block.

    Args:
        tables (dict[str, Any]): Output of :func:`build_validation_tables`.
        class_name (str): Component/class name.
        iou_threshold (float): Match IoU threshold.
        official_confusion (dict[str, Any] | None): Ultralytics confusion matrix
            from ``metrics.json.ultralytics.confusion_matrix``.
        official_conf_threshold (float | None): Low-confidence threshold used
            for the official pass, when known.

    Returns:
        list[str]: Report lines, ready to join with newlines.
    """
    lines: list[str] = [
        "",
        "=" * 78,
        "        VALIDATION TABLES: CONFUSION, FALSE POSITIVES, PRECISION",
        "=" * 78,
        f"Detector class     : {class_name}",
        f"Match IoU          : {iou_threshold:.2f}",
        "-" * 78,
    ]
    if official_confusion and official_confusion.get("matrix"):
        threshold_text: str = (
            f" at conf {official_conf_threshold:g}"
            if official_conf_threshold is not None
            else ""
        )
        lines += _matrix_lines(
            official_confusion,
            f"Official Ultralytics confusion matrix{threshold_text} "
            "(rows=true, columns=predicted)",
        )
        lines.append("-" * 78)

    for threshold, matrix in tables["confusion_matrix"]["box"].items():
        lines += _matrix_lines(
            matrix,
            f"Box-level confusion matrix at conf {threshold} "
            "(rows=true, columns=predicted)",
        )
    lines.append("-" * 78)
    for threshold, matrix in tables["confusion_matrix"]["image"].items():
        lines += _matrix_lines(
            matrix,
            f"Image-level confusion matrix at conf {threshold} "
            "(rows=true, columns=predicted)",
        )

    lines += [
        "-" * 78,
        "False positives: means over the run",
        "-" * 78,
        "  conf   FP    mean/img mean/pos mean/bg mean/fpimg median p90  max  "
        "imgsWithFP frac  meanConf meanBestIoU",
    ]
    for key in sorted(
        tables["false_positive_summary"], key=lambda item: float(item)
    ):
        stats: dict[str, Any] = tables["false_positive_summary"][key]
        lines.append(
            f"  {key:<6} {stats['fp_boxes']:<5} "
            f"{_format_optional(stats['mean_fp_per_image']):<8} "
            f"{_format_optional(stats['mean_fp_per_positive_image']):<8} "
            f"{_format_optional(stats['mean_fp_per_background_image']):<7} "
            f"{_format_optional(stats['mean_fp_per_image_with_fp']):<9} "
            f"{_format_optional(stats['median_fp_per_image']):<6} "
            f"{_format_optional(stats['p90_fp_per_image']):<4} "
            f"{_format_optional(stats['max_fp_per_image']):<4} "
            f"{stats['images_with_fp']:<10} "
            f"{_format_optional(stats['fraction_images_with_fp']):<6} "
            f"{_format_optional(stats['mean_fp_confidence']):<8} "
            f"{_format_optional(stats['mean_fp_best_iou'])}"
        )
    lines.append(
        "  meanBestIoU is over FPs on positive images; a background FP has no "
        "GT box to overlap."
    )

    lines += [
        "-" * 78,
        "Precision table over the confidence sweep",
        "-" * 78,
        "  conf   TP    FP    FN    prec   rec    F1     boxJ   "
        "imgAcc detRate FA-rate spec   balAcc",
    ]
    for row in tables["precision_table"]:
        lines.append(
            f"  {row['conf_threshold']:<6g} {row['tp']:<5} {row['fp']:<5} "
            f"{row['fn']:<5} "
            f"{_format_optional(row['precision']):<6} "
            f"{_format_optional(row['recall']):<6} "
            f"{_format_optional(row['f1']):<6} "
            f"{_format_optional(row['jaccard']):<6} "
            f"{_format_optional(row['image_accuracy']):<6} "
            f"{_format_optional(row['positive_detection_rate']):<7} "
            f"{_format_optional(row['false_alarm_rate']):<7} "
            f"{_format_optional(row['specificity']):<6} "
            f"{_format_optional(row['balanced_accuracy'])}"
        )
    lines += [
        "-" * 78,
        "Artifacts",
        "-" * 78,
        f"  confusion matrix  : {CONFUSION_MATRIX_CSV}",
        f"  false positives   : {FALSE_POSITIVE_CSV}",
        f"  precision table   : {PRECISION_TABLE_CSV}",
        f"  this report       : {TABLES_TXT}",
        "=" * 78,
        "",
    ]
    return lines


def write_tables(
    output_dir: Path,
    tables: dict[str, Any],
    class_name: str,
    iou_threshold: float,
    official_confusion: dict[str, Any] | None = None,
    official_conf_threshold: float | None = None,
) -> dict[str, Path]:
    """Write the extra tables as CSV and text into one component folder.

    Args:
        output_dir (Path): Component report folder.
        tables (dict[str, Any]): Output of :func:`build_validation_tables`.
        class_name (str): Component/class name.
        iou_threshold (float): Match IoU threshold.
        official_confusion (dict[str, Any] | None): Ultralytics confusion matrix.
        official_conf_threshold (float | None): Confidence of the official pass.

    Returns:
        dict[str, Path]: Written paths keyed by ``confusion_matrix``,
        ``false_positive_summary``, ``precision_table`` and ``tables_txt``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # One row per matrix cell, so the CSV can be filtered by scope, level,
    # threshold, or true/predicted label.
    confusion_rows: list[dict[str, Any]] = []
    if official_confusion and official_confusion.get("matrix"):
        labels: list[str] = [str(label) for label in official_confusion["labels"]]
        for true_label, row in zip(
            labels, official_confusion["matrix"], strict=True
        ):
            for predicted_label, count in zip(labels, row, strict=True):
                confusion_rows.append(
                    {
                        "scope": "official",
                        "level": "box",
                        "conf_threshold": (
                            "" if official_conf_threshold is None
                            else f"{float(official_conf_threshold):g}"
                        ),
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(count),
                    }
                )
    for level, matrices in tables["confusion_matrix"].items():
        for threshold, matrix in matrices.items():
            labels = [str(label) for label in matrix["labels"]]
            for true_label, row in zip(labels, matrix["matrix"], strict=True):
                for predicted_label, count in zip(labels, row, strict=True):
                    confusion_rows.append(
                        {
                            "scope": "thresholded",
                            "level": level,
                            "conf_threshold": threshold,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "count": int(count),
                        }
                    )

    confusion_path: Path = output_dir / CONFUSION_MATRIX_CSV
    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scope",
                "level",
                "conf_threshold",
                "true_label",
                "predicted_label",
                "count",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(confusion_rows)

    fp_path: Path = output_dir / FALSE_POSITIVE_CSV
    with fp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_FALSE_POSITIVE_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for key in sorted(
            tables["false_positive_summary"], key=lambda item: float(item)
        ):
            writer.writerow(tables["false_positive_summary"][key])

    precision_path: Path = output_dir / PRECISION_TABLE_CSV
    with precision_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_PRECISION_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(tables["precision_table"])

    tables_path: Path = output_dir / TABLES_TXT
    tables_path.write_text(
        "\n".join(
            format_tables_report(
                tables,
                class_name=class_name,
                iou_threshold=iou_threshold,
                official_confusion=official_confusion,
                official_conf_threshold=official_conf_threshold,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "confusion_matrix": confusion_path,
        "false_positive_summary": fp_path,
        "precision_table": precision_path,
        "tables_txt": tables_path,
    }


def regenerate_component_tables(
    component_root: Path,
    hyperparams: SimpleNamespace,
) -> dict[str, Any]:
    """Rebuild every extra table for an existing report, without rerendering.

    Args:
        component_root (Path): Component folder holding ``metrics.json`` and the
            saved dataset/prediction labels.
        hyperparams (SimpleNamespace): Stage hyperparameters with the confidence
            lists, match IoU and render size.

    Returns:
        dict[str, Any]: The updated metrics dictionary that was written back.

    Raises:
        FileNotFoundError: If the report's metrics or labels are absent.
    """
    metrics_path: Path = component_root / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"No component report at {metrics_path}.")
    summary: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    ground_truth_dir: Path = component_root / "dataset" / "labels" / "val"
    if not ground_truth_dir.is_dir():
        raise FileNotFoundError(f"No ground-truth labels at {ground_truth_dir}.")
    save_dir: str | None = summary.get("ultralytics", {}).get("save_dir")
    prediction_dir: Path = (
        Path(save_dir) / "labels"
        if save_dir
        else component_root / "ultralytics" / "val" / "labels"
    )
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"No saved predictions at {prediction_dir}.")

    stems: list[str] = sorted(
        path.stem for path in ground_truth_dir.glob("*.txt")
    )
    tables: dict[str, Any] = build_validation_tables(
        stems=stems,
        ground_truth_dir=ground_truth_dir,
        prediction_dir=prediction_dir,
        report_thresholds=[
            float(value) for value in hyperparams.VALIDATION_CONF_THRESHOLDS
        ],
        precision_thresholds=[
            float(value) for value in hyperparams.VALIDATION_PRECISION_THRESHOLDS
        ],
        iou_threshold=float(hyperparams.VALIDATION_MATCH_IOU),
        image_size=int(hyperparams.RENDER_IMAGE_SIZE),
        class_id=0,
        class_name=str(summary["class_name"]),
    )
    official_confusion: dict[str, Any] | None = (
        summary.get("ultralytics", {}).get("confusion_matrix")
    )
    paths: dict[str, Path] = write_tables(
        output_dir=component_root,
        tables=tables,
        class_name=str(summary["class_name"]),
        iou_threshold=float(hyperparams.VALIDATION_MATCH_IOU),
        official_confusion=official_confusion,
        official_conf_threshold=float(hyperparams.VALIDATION_VAL_CONF),
    )
    summary["confusion_matrices"] = tables["confusion_matrix"]
    summary["false_positive_summary"] = tables["false_positive_summary"]
    summary["precision_table"] = tables["precision_table"]
    summary["tables"] = {name: str(path) for name, path in paths.items()}
    metrics_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    """Regenerate the extra tables for existing component reports.

    Returns:
        int: Process exit code, ``0`` on success.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Rebuild confusion-matrix, false-positive and precision tables for "
            "existing missing_yolo validation reports."
        )
    )
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    parser.add_argument(
        "--component",
        default="all",
        help="Component key, or 'all' (default).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Validation output root (default: VALIDATION_OUTPUT_DIRECTORY).",
    )
    args: argparse.Namespace = parser.parse_args()

    # Imported here so the table arithmetic (and its tests) does not pull in the
    # trainer/YOLO stack just to resolve config paths.
    from missing_yolo.runner import anchor_config_paths

    config: SimpleNamespace = load_config(
        args.mode, str(_SCRIPT_DIR / "config.toml")
    )
    anchor_config_paths(config)
    hyperparams: SimpleNamespace = load_hyperparams(
        str(_SCRIPT_DIR / "hyperparameters.toml")
    )
    output_root: Path = (
        Path(str(args.output)).expanduser().resolve()
        if args.output
        else Path(str(config.VALIDATION_OUTPUT_DIRECTORY))
    )
    components: list[str] = sorted(
        path.parent.name
        for path in output_root.glob("*/metrics.json")
        if path.parent.name != "old_validation_outputs"
    )
    if str(args.component) != "all":
        if str(args.component) not in components:
            raise SystemExit(
                f"No report for '{args.component}' under {output_root}; "
                f"found {components}."
            )
        components = [str(args.component)]
    if not components:
        raise SystemExit(f"No component reports found under {output_root}.")

    for component in components:
        summary = regenerate_component_tables(
            output_root / component, hyperparams
        )
        print(
            f"{component}: wrote confusion_matrix.csv, "
            "false_positive_summary.csv, precision_table.csv, tables.txt "
            f"({summary['dataset']['samples']} images)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
