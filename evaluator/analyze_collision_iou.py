"""Post-process completed evaluator runs into matched-box and collision-IoU tables.

The evaluator already reports the IoU of the matched target box through
``evaluation_summary.json``. It does not store the raw predicted box, so the
collision IoU cannot be recovered from telemetry alone. This script therefore
re-runs each component's detector over the saved **ensemble** composite images
in ``dataset_ensemble/images/test`` and measures every target-other-furniture
pair.

Definitions
-----------
* **Matched-box IoU** - IoU between a target ground-truth box and its matched
  target prediction, using the evaluator's normal ``BOX_MATCH_IOU`` assignment
  (0.5 by default). These values come from the evaluator summary.
* **Collision IoU** - IoU between the final target prediction (the
  highest-confidence target box, the same box the overlap metric uses) and one
  other ground-truth furniture box. A pair is counted when IoU is strictly
  greater than zero.

The script writes:

* ``collision_iou_pairs.csv`` - one row per target-other furniture pair,
  including the sample's ``matched_boxes`` and ``best_iou`` so callers can
  filter to samples with a matched target box.
* ``collision_iou_samples.csv`` - one row per ensemble sample.
* ``collision_iou_summary.csv`` - one row per component.
* ``collision_iou_results.md`` - human-readable tables and notes.
* ``collision_iou_table.tex`` - two LaTeX tables for the paper.

Usage::

    ./venv/bin/python evaluator/analyze_collision_iou.py \
        --manifest evaluator/outputs/test_comparison/runs.json
"""

from __future__ import annotations

import os

# The detector's NMS falls back to CPU on MPS. Set the fallback before torch is
# imported by Ultralytics or any evaluator module.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

# Repository root, resolved from this file so the command works from anywhere.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluator.box_metrics import (  # noqa: E402
    PredictedBox,
    intersection_over_union,
    pixel_boxes_to_predictions,
)
from evaluator.dataset_gt import NamedBox, parse_named_label_line  # noqa: E402
from helpers.hyperparams import load_hyperparams  # noqa: E402
from evaluator.overlap_metrics import (  # noqa: E402
    highest_confidence_prediction,
    overlap_fraction,
)
from evaluator.yolo_eval import predict_component_boxes  # noqa: E402

# The evaluator's sample folders use this label file. It is fixed by the
# dataset layout, not by a model hyperparameter.
LABEL_FILENAME: str = "labels.txt"

# A collision is any positive IoU. This is intentionally the same boundary as
# the overlap metric's "any collision" bucket.
COLLISION_IOU_THRESHOLD: float = 0.0

# Report quantiles, matching the evaluator's existing summary style.
REPORT_QUANTILES: tuple[float, ...] = (0.10, 0.90)

# Progress cadence. A full test split takes tens of minutes on CPU, so the user
# needs visible progress without a line per image.
PROGRESS_EVERY: int = 500

# Default report order. The manifest is written with sorted keys, but papers
# usually list the components in this order.
PREFERRED_COMPONENT_ORDER: tuple[str, ...] = ("tv", "showcase", "dining_set")


@dataclass(frozen=True)
class CollisionPair:
    """One target-prediction / other-furniture pair.

    Attributes:
        component (str): Component key, e.g. ``tv``.
        stem (str): Dataset sample stem.
        target_confidence (float): Confidence of the final target prediction.
        other_index (int): Zero-based index of the other box in the label file.
        other_name (str): Dataset name of the other furniture.
        collision_iou (float): IoU between the final target box and the other
            box, in normalized coordinates.
        overlap_fraction (float | None): The evaluator's union-coverage overlap
            for the same sample, copied for context.
        matched_boxes (int): Evaluator matched target-box count for the sample.
        best_iou (float | None): Evaluator best matched target IoU for the
            sample. ``0.0`` means the sample had no matched target box.
    """

    component: str
    stem: str
    target_confidence: float
    other_index: int
    other_name: str
    collision_iou: float
    overlap_fraction: float | None
    matched_boxes: int
    best_iou: float | None


@dataclass(frozen=True)
class CollisionSample:
    """One ensemble sample's collision summary.

    Attributes:
        component (str): Component key.
        stem (str): Dataset sample stem.
        predicted (bool): Whether the detector returned a target box.
        target_confidence (float | None): Confidence of the final target box.
        other_gt_boxes (int): Number of other furniture boxes in the layout.
        collision_boxes (int): Number of other boxes with positive IoU.
        max_collision_iou (float | None): Largest collision IoU for the sample.
        sum_collision_iou (float | None): Sum of the sample's collision IoUs.
        overlap_fraction (float | None): Evaluator union-coverage overlap.
        matched_boxes (int): Evaluator matched target-box count.
        best_iou (float | None): Evaluator best matched target IoU.
    """

    component: str
    stem: str
    predicted: bool
    target_confidence: float | None
    other_gt_boxes: int
    collision_boxes: int
    max_collision_iou: float | None
    sum_collision_iou: float | None
    overlap_fraction: float | None
    matched_boxes: int
    best_iou: float | None


@dataclass(frozen=True)
class ComponentSummary:
    """One component's matched-box and collision-IoU aggregates.

    Attributes:
        component (str): Component key.
        run_id (str): Evaluator run id.
        split_dir (str): Dataset split directory.
        samples (int): Number of ensemble samples.
        predicted (int): Samples with a target prediction.
        no_prediction (int): Samples without a target prediction.
        matched_boxes (int): Matched target boxes from the evaluator summary.
        matched_iou_mean (float | None): Mean matched-box IoU.
        matched_iou_median (float | None): Median matched-box IoU.
        matched_iou_p10 (float | None): 10th percentile matched-box IoU.
        matched_iou_p90 (float | None): 90th percentile matched-box IoU.
        collision_samples (int): Predicted samples with at least one positive
            collision IoU.
        collision_boxes (int): Total positive-IoU target-other pairs.
        collision_iou_mean (float | None): Mean collision IoU over pairs.
        collision_iou_median (float | None): Median collision IoU over pairs.
        collision_iou_p10 (float | None): 10th percentile collision IoU.
        collision_iou_p90 (float | None): 90th percentile collision IoU.
        max_collision_iou_mean (float | None): Mean of each collision sample's
            largest collision IoU.
        max_collision_iou_median (float | None): Median of those maxima.
        max_collision_iou_p10 (float | None): 10th percentile of those maxima.
        max_collision_iou_p90 (float | None): 90th percentile of those maxima.
        collision_sample_rate (float | None): Collision samples / predicted
            samples.
        detector_prediction_mismatches (int): Samples where the re-run detector
            prediction count differed from telemetry.
        overlap_mismatches (int): Samples where recomputed union overlap
            differed from telemetry by more than ``1e-6``.
        summary_path (Path): Evaluator summary JSON.
        telemetry_path (Path): Evaluator telemetry CSV.
    """

    component: str
    run_id: str
    split_dir: str
    samples: int
    predicted: int
    no_prediction: int
    matched_boxes: int
    matched_iou_mean: float | None
    matched_iou_median: float | None
    matched_iou_p10: float | None
    matched_iou_p90: float | None
    collision_samples: int
    collision_boxes: int
    collision_iou_mean: float | None
    collision_iou_median: float | None
    collision_iou_p10: float | None
    collision_iou_p90: float | None
    max_collision_iou_mean: float | None
    max_collision_iou_median: float | None
    max_collision_iou_p10: float | None
    max_collision_iou_p90: float | None
    collision_sample_rate: float | None
    detector_prediction_mismatches: int
    overlap_mismatches: int
    summary_path: Path
    telemetry_path: Path


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options.

    Args:
        argv (list[str] | None): Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: Parsed options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute matched-box IoU and target-vs-other-furniture collision IoU "
            "from completed evaluator runs."
        )
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(
            ROOT_DIR / "evaluator" / "outputs" / "test_comparison" / "runs.json"
        ),
        help="Comparison manifest written by run_validation_comparison.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory; defaults to the manifest's directory.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help="Subset of component keys; defaults to every manifest entry.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum ensemble samples per component; negative keeps all.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        default="cpu",
        help=(
            "Detector device. CPU matches the evaluator's per-image prediction "
            "path; MPS is not faster here because NMS falls back to CPU."
        ),
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    """Read one JSON object.

    Args:
        path (Path): JSON file path.

    Returns:
        dict[str, Any]: Parsed object.

    Raises:
        ValueError: If the file does not contain a JSON object.
    """
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def _read_telemetry(path: Path) -> dict[str, dict[str, str]]:
    """Read ensemble telemetry rows keyed by sample stem.

    Args:
        path (Path): Evaluator telemetry CSV.

    Returns:
        dict[str, dict[str, str]]: Ensemble rows (``draw == "-1"``) keyed by
            stem.

    Raises:
        FileNotFoundError: If the telemetry file is absent.
        ValueError: If a stem appears more than once in the ensemble rows.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Telemetry CSV is missing: {path}.")
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("draw") != "-1":
                continue
            stem: str = row["stem"]
            if stem in rows:
                raise ValueError(f"Duplicate ensemble telemetry row for {stem}.")
            rows[stem] = row
    return rows


def _named_other_boxes(
    sample_dir: Path,
    label_filename: str,
    target_names: tuple[str, ...],
) -> list[NamedBox]:
    """Read every named label line whose name is not a target label.

    Args:
        sample_dir (Path): Dataset sample folder.
        label_filename (str): Label filename inside the folder.
        target_names (tuple[str, ...]): Dataset names of the target component.

    Returns:
        list[NamedBox]: Other furniture boxes with their dataset names.

    Raises:
        FileNotFoundError: If the sample has no label file.
        ValueError: If a label line is malformed.
    """
    label_path: Path = sample_dir / label_filename
    if not label_path.is_file():
        raise FileNotFoundError(f"Sample {sample_dir} has no {label_filename}.")
    boxes: list[NamedBox] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        named: NamedBox | None = parse_named_label_line(raw)
        if named is not None and named.name not in target_names:
            boxes.append(named)
    return boxes


def _quantile(values: list[float], fraction: float) -> float | None:
    """Return a nearest-rank quantile, or ``None`` for an empty list.

    Args:
        values (list[float]): Measurements.
        fraction (float): Quantile in ``0..1``.

    Returns:
        float | None: Selected value, or ``None`` when there are no values.
    """
    if not values:
        return None
    ordered: list[float] = sorted(values)
    index: int = min(
        len(ordered) - 1,
        max(0, int(fraction * (len(ordered) - 1))),
    )
    return ordered[index]


def _stats(values: list[float]) -> dict[str, float | None]:
    """Summarize a list of IoU values.

    Args:
        values (list[float]): IoU measurements.

    Returns:
        dict[str, float | None]: Mean, median, p10, p90, min and max; all
            ``None`` when the list is empty.
    """
    if not values:
        return {
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": _quantile(values, REPORT_QUANTILES[0]),
        "p90": _quantile(values, REPORT_QUANTILES[1]),
        "min": min(values),
        "max": max(values),
    }


def _float_or_none(value: Any) -> float | None:
    """Convert an optional telemetry value to float.

    Args:
        value (Any): Raw value.

    Returns:
        float | None: Parsed float, or ``None`` for blank/None.
    """
    if value is None:
        return None
    text: str = str(value).strip()
    return float(text) if text else None


def _load_detector(weights_path: Path, device: str) -> Any:
    """Load one Ultralytics detector on the requested device.

    Args:
        weights_path (Path): Component detector weights.
        device (str): ``cpu`` or ``mps``.

    Returns:
        Any: Loaded Ultralytics model.

    Raises:
        FileNotFoundError: If the weights file is absent.
    """
    from ultralytics import YOLO  # lazy import keeps the module import cheap

    if not weights_path.is_file():
        raise FileNotFoundError(f"Detector weights are missing: {weights_path}.")
    model: Any = YOLO(str(weights_path))
    model.to(device)
    return model


def _ensemble_image_path(output_root: Path, stem: str) -> Path:
    """Return the saved ensemble image path for one sample.

    Args:
        output_root (Path): Evaluator output root.
        stem (str): Dataset sample stem.

    Returns:
        Path: Ensemble image path.
    """
    return (
        output_root
        / "dataset_ensemble"
        / "images"
        / "test"
        / f"{stem}__ensemble.png"
    )


def _final_prediction(
    model: Any,
    image_path: Path,
    confidence: float,
    inference_size: int,
    render_size: int,
    max_detections: int,
    class_id: int,
) -> tuple[list[PredictedBox], PredictedBox | None]:
    """Run the detector and return all target predictions plus the final box.

    The detector is run at the evaluator's ``YOLO_IMGSZ`` (416 by default),
    while its pixel boxes are normalized against the saved composite image's
    own side length (256 by default). Those are two different numbers and must
    not be conflated.

    Args:
        model (Any): Loaded detector.
        image_path (Path): Ensemble composite image.
        confidence (float): Detector confidence threshold.
        inference_size (int): Ultralytics inference size (``imgsz``).
        render_size (int): Saved image side length used to normalize boxes.
        max_detections (int): Maximum detections.
        class_id (int): Target detector class id.

    Returns:
        tuple[list[PredictedBox], PredictedBox | None]: All target predictions
            and the highest-confidence one.
    """
    pixel_boxes: list[tuple[float, float, float, float, int, float]] = (
        predict_component_boxes(
            model=model,
            image_path=image_path,
            confidence=confidence,
            image_size=inference_size,
            max_detections=max_detections,
            class_id=class_id,
        )
    )
    predictions: list[PredictedBox] = pixel_boxes_to_predictions(
        pixel_boxes,
        class_id,
        render_size,
    )
    return predictions, highest_confidence_prediction(predictions)


def _write_pair_row(writer: csv.DictWriter, pair: CollisionPair) -> None:
    """Write one collision-pair CSV row.

    Args:
        writer (csv.DictWriter): Open pair CSV writer.
        pair (CollisionPair): Pair to write.
    """
    writer.writerow(
        {
            "component": pair.component,
            "stem": pair.stem,
            "target_confidence": pair.target_confidence,
            "other_index": pair.other_index,
            "other_name": pair.other_name,
            "collision_iou": pair.collision_iou,
            "overlap_fraction": pair.overlap_fraction,
            "matched_boxes": pair.matched_boxes,
            "best_iou": pair.best_iou,
        }
    )


def _write_sample_row(writer: csv.DictWriter, sample: CollisionSample) -> None:
    """Write one sample-level CSV row.

    Args:
        writer (csv.DictWriter): Open sample CSV writer.
        sample (CollisionSample): Sample summary to write.
    """
    writer.writerow(
        {
            "component": sample.component,
            "stem": sample.stem,
            "predicted": int(sample.predicted),
            "target_confidence": sample.target_confidence,
            "other_gt_boxes": sample.other_gt_boxes,
            "collision_boxes": sample.collision_boxes,
            "max_collision_iou": sample.max_collision_iou,
            "sum_collision_iou": sample.sum_collision_iou,
            "overlap_fraction": sample.overlap_fraction,
            "matched_boxes": sample.matched_boxes,
            "best_iou": sample.best_iou,
        }
    )


def _summarize_component(
    entry: dict[str, Any],
    pair_csv: csv.DictWriter,
    sample_csv: csv.DictWriter,
    limit: int,
    device: str,
    hyperparams: SimpleNamespace,
) -> ComponentSummary:
    """Process one component and write its pair/sample rows.

    Args:
        entry (dict[str, Any]): Manifest entry for the component.
        pair_csv (csv.DictWriter): Open pair CSV writer.
        sample_csv (csv.DictWriter): Open sample CSV writer.
        limit (int): Maximum samples; negative keeps all.
        device (str): Detector device.
        hyperparams (SimpleNamespace): Evaluator hyperparameters, used for the
            detector's maximum box count.

    Returns:
        ComponentSummary: Aggregated component result.

    Raises:
        FileNotFoundError: If a required evaluator artifact is missing.
        ValueError: If the manifest or summary shape is inconsistent.
    """
    component: str = str(entry["component"])
    summary_path: Path = Path(str(entry["summary_path"]))
    telemetry_path: Path = Path(str(entry["telemetry_path"]))
    summary: dict[str, Any] = _load_json(summary_path)
    telemetry: dict[str, dict[str, str]] = _read_telemetry(telemetry_path)

    configuration: dict[str, Any] = dict(summary["configuration"])
    missing: dict[str, Any] = dict(summary["missing_component"])
    output_root: Path = Path(str(summary["output_root"]))
    split_dir: Path = Path(str(configuration["split_dir"]))
    target_names: tuple[str, ...] = tuple(str(name) for name in missing["dataset_label_names"])
    class_id: int = int(missing["detector_class_id"])
    confidence: float = float(configuration["yolo_conf"])
    inference_size: int = int(configuration["yolo_imgsz"])
    max_detections: int = int(hyperparams.YOLO_MAX_BOXES)

    stems: list[str] = sorted(telemetry)
    if limit >= 0:
        stems = stems[:limit]

    model: Any | None = None
    if any(int(telemetry[stem]["pred_boxes"]) > 0 for stem in stems):
        model = _load_detector(
            Path(str(configuration["yolo_weights"])),
            device=device,
        )

    pair_values: list[float] = []
    max_values: list[float] = []
    predicted: int = 0
    collision_samples: int = 0
    detector_prediction_mismatches: int = 0
    overlap_mismatches: int = 0

    for index, stem in enumerate(stems, start=1):
        row: dict[str, str] = telemetry[stem]
        other_boxes: list[NamedBox] = _named_other_boxes(
            split_dir / stem,
            LABEL_FILENAME,
            target_names,
        )
        telemetry_other_count: int = int(row["other_gt_boxes"])
        if telemetry_other_count != len(other_boxes):
            raise ValueError(
                f"{component}/{stem}: telemetry has {telemetry_other_count} other "
                f"boxes but labels.txt has {len(other_boxes)}."
            )

        telemetry_pred_count: int = int(row["pred_boxes"])
        overlap_value: float | None = _float_or_none(row.get("overlap"))
        if telemetry_pred_count <= 0:
            _write_sample_row(
                sample_csv,
                CollisionSample(
                    component=component,
                    stem=stem,
                    predicted=False,
                    target_confidence=None,
                    other_gt_boxes=len(other_boxes),
                    collision_boxes=0,
                    max_collision_iou=None,
                    sum_collision_iou=None,
                    overlap_fraction=overlap_value,
                    matched_boxes=int(row["matched_boxes"]),
                    best_iou=_float_or_none(row.get("best_iou")),
                ),
            )
            continue

        if model is None:  # pragma: no cover - guarded by the model-load check
            raise RuntimeError("Detector model was not loaded for a predicted sample.")
        image_path: Path = _ensemble_image_path(output_root, stem)
        if not image_path.is_file():
            raise FileNotFoundError(f"Ensemble image is missing: {image_path}.")
        with Image.open(image_path) as image:
            width, height = image.size
        if width != height:
            raise ValueError(
                f"{image_path} is not square: {width}x{height}; the evaluator "
                "expects a square render domain."
            )
        predictions, final_prediction = _final_prediction(
            model=model,
            image_path=image_path,
            confidence=confidence,
            inference_size=inference_size,
            render_size=width,
            max_detections=max_detections,
            class_id=class_id,
        )
        if len(predictions) != telemetry_pred_count:
            detector_prediction_mismatches += 1
            print(
                f"  warning: {component}/{stem}: re-run has {len(predictions)} "
                f"target predictions, telemetry has {telemetry_pred_count}",
                flush=True,
            )
        if final_prediction is None:
            _write_sample_row(
                sample_csv,
                CollisionSample(
                    component=component,
                    stem=stem,
                    predicted=False,
                    target_confidence=None,
                    other_gt_boxes=len(other_boxes),
                    collision_boxes=0,
                    max_collision_iou=None,
                    sum_collision_iou=None,
                    overlap_fraction=overlap_value,
                    matched_boxes=int(row["matched_boxes"]),
                    best_iou=_float_or_none(row.get("best_iou")),
                ),
            )
            continue

        predicted += 1
        recomputed_overlap: float = overlap_fraction(
            final_prediction.box,
            [named.box for named in other_boxes],
        )
        if overlap_value is None or abs(recomputed_overlap - overlap_value) > 1e-6:
            overlap_mismatches += 1
            if overlap_mismatches <= 5:
                print(
                    f"  warning: {component}/{stem}: recomputed overlap "
                    f"{recomputed_overlap:.8f} != telemetry {overlap_value}",
                    flush=True,
                )

        collision_ious: list[float] = []
        for other_index, named in enumerate(other_boxes):
            collision_iou: float = intersection_over_union(
                named.box,
                final_prediction.box,
            )
            if collision_iou > COLLISION_IOU_THRESHOLD:
                collision_ious.append(collision_iou)
                pair_values.append(collision_iou)
                _write_pair_row(
                    pair_csv,
                    CollisionPair(
                        component=component,
                        stem=stem,
                        target_confidence=final_prediction.confidence,
                        other_index=other_index,
                        other_name=named.name,
                        collision_iou=collision_iou,
                        overlap_fraction=overlap_value,
                        matched_boxes=int(row["matched_boxes"]),
                        best_iou=_float_or_none(row.get("best_iou")),
                    ),
                )

        sample_max: float | None = max(collision_ious) if collision_ious else None
        if sample_max is not None:
            collision_samples += 1
            max_values.append(sample_max)
        _write_sample_row(
            sample_csv,
            CollisionSample(
                component=component,
                stem=stem,
                predicted=True,
                target_confidence=final_prediction.confidence,
                other_gt_boxes=len(other_boxes),
                collision_boxes=len(collision_ious),
                max_collision_iou=sample_max,
                sum_collision_iou=sum(collision_ious) if collision_ious else None,
                overlap_fraction=overlap_value,
                matched_boxes=int(row["matched_boxes"]),
                best_iou=_float_or_none(row.get("best_iou")),
            ),
        )

        if index % PROGRESS_EVERY == 0:
            print(
                f"  {component}: {index}/{len(stems)} samples, "
                f"{len(pair_values)} collision pairs",
                flush=True,
            )

    ensemble_box_iou: dict[str, Any] = dict(
        dict(summary["ensemble"]["box_loss"])["box_iou"]
    )
    collision_stats: dict[str, float | None] = _stats(pair_values)
    max_stats: dict[str, float | None] = _stats(max_values)
    return ComponentSummary(
        component=component,
        run_id=str(summary["run_id"]),
        split_dir=str(split_dir),
        samples=len(stems),
        predicted=predicted,
        no_prediction=len(stems) - predicted,
        matched_boxes=int(summary["ensemble"]["outcomes"]["matched"]),
        matched_iou_mean=_float_or_none(ensemble_box_iou.get("mean")),
        matched_iou_median=_float_or_none(ensemble_box_iou.get("median")),
        matched_iou_p10=_float_or_none(ensemble_box_iou.get("p10")),
        matched_iou_p90=_float_or_none(ensemble_box_iou.get("p90")),
        collision_samples=collision_samples,
        collision_boxes=len(pair_values),
        collision_iou_mean=collision_stats["mean"],
        collision_iou_median=collision_stats["median"],
        collision_iou_p10=collision_stats["p10"],
        collision_iou_p90=collision_stats["p90"],
        max_collision_iou_mean=max_stats["mean"],
        max_collision_iou_median=max_stats["median"],
        max_collision_iou_p10=max_stats["p10"],
        max_collision_iou_p90=max_stats["p90"],
        collision_sample_rate=(
            collision_samples / predicted if predicted else None
        ),
        detector_prediction_mismatches=detector_prediction_mismatches,
        overlap_mismatches=overlap_mismatches,
        summary_path=summary_path,
        telemetry_path=telemetry_path,
    )


def _format_count(value: int) -> str:
    """Format an integer with thousands separators.

    Args:
        value (int): Count.

    Returns:
        str: Formatted count.
    """
    return f"{value:,}"


def _format_float(value: float | None, digits: int = 4) -> str:
    """Format an optional float for a table.

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


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Build one Markdown table.

    Args:
        headers (list[str]): Column headers.
        rows (list[list[str]]): Formatted row cells.

    Returns:
        list[str]: Markdown lines.
    """
    lines: list[str] = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _write_summary_csv(summaries: list[ComponentSummary], path: Path) -> None:
    """Write the per-component summary CSV.

    Args:
        summaries (list[ComponentSummary]): Component results.
        path (Path): Destination CSV path.
    """
    fieldnames: list[str] = [
        "component",
        "run_id",
        "split_dir",
        "samples",
        "predicted",
        "no_prediction",
        "matched_boxes",
        "matched_iou_mean",
        "matched_iou_median",
        "matched_iou_p10",
        "matched_iou_p90",
        "collision_samples",
        "collision_boxes",
        "collision_iou_mean",
        "collision_iou_median",
        "collision_iou_p10",
        "collision_iou_p90",
        "max_collision_iou_mean",
        "max_collision_iou_median",
        "max_collision_iou_p10",
        "max_collision_iou_p90",
        "collision_sample_rate",
        "detector_prediction_mismatches",
        "overlap_mismatches",
        "summary_path",
        "telemetry_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "component": summary.component,
                    "run_id": summary.run_id,
                    "split_dir": summary.split_dir,
                    "samples": summary.samples,
                    "predicted": summary.predicted,
                    "no_prediction": summary.no_prediction,
                    "matched_boxes": summary.matched_boxes,
                    "matched_iou_mean": summary.matched_iou_mean,
                    "matched_iou_median": summary.matched_iou_median,
                    "matched_iou_p10": summary.matched_iou_p10,
                    "matched_iou_p90": summary.matched_iou_p90,
                    "collision_samples": summary.collision_samples,
                    "collision_boxes": summary.collision_boxes,
                    "collision_iou_mean": summary.collision_iou_mean,
                    "collision_iou_median": summary.collision_iou_median,
                    "collision_iou_p10": summary.collision_iou_p10,
                    "collision_iou_p90": summary.collision_iou_p90,
                    "max_collision_iou_mean": summary.max_collision_iou_mean,
                    "max_collision_iou_median": summary.max_collision_iou_median,
                    "max_collision_iou_p10": summary.max_collision_iou_p10,
                    "max_collision_iou_p90": summary.max_collision_iou_p90,
                    "collision_sample_rate": summary.collision_sample_rate,
                    "detector_prediction_mismatches": (
                        summary.detector_prediction_mismatches
                    ),
                    "overlap_mismatches": summary.overlap_mismatches,
                    "summary_path": str(summary.summary_path),
                    "telemetry_path": str(summary.telemetry_path),
                }
            )


def _build_markdown(summaries: list[ComponentSummary]) -> str:
    """Build the human-readable results report.

    Args:
        summaries (list[ComponentSummary]): Component results.

    Returns:
        str: Markdown document.
    """
    split_dir: str = summaries[0].split_dir if summaries else "n/a"
    lines: list[str] = [
        "# Matched-box and collision IoU results",
        "",
        f"Split: `{split_dir}`",
        "",
        "## Definitions",
        "",
        "- **Matched-box IoU** is the IoU between a target ground-truth box and "
        "its matched target prediction. Values are from the evaluator summary; "
        "matching uses IoU ≥ 0.5.",
        "- **Collision IoU** is the IoU between the final target prediction "
        "(highest-confidence target box) and one other ground-truth furniture "
        "box. A pair is counted when IoU > 0.",
        "- **Collision samples** counts predicted samples with at least one "
        "positive collision IoU. **Collision boxes** counts target–other pairs.",
        "- The pair CSV carries `matched_boxes` and `best_iou`, so collision "
        "rows can be filtered to samples with a matched target box.",
        "",
        "## Matched target boxes",
        "",
    ]
    rows: list[list[str]] = [
        [
            summary.component,
            _format_count(summary.matched_boxes),
            _format_float(summary.matched_iou_mean),
            _format_float(summary.matched_iou_median),
            _format_float(summary.matched_iou_p10),
            _format_float(summary.matched_iou_p90),
        ]
        for summary in summaries
    ]
    lines.extend(
        _markdown_table(
            [
                "Component",
                "Matched boxes",
                "Mean IoU",
                "Median IoU",
                "p10",
                "p90",
            ],
            rows,
        )
    )
    lines.extend(["", "## Collision IoU (per target–other pair)", ""])
    rows = [
        [
            summary.component,
            _format_count(summary.collision_samples),
            _format_count(summary.collision_boxes),
            _format_float(summary.collision_iou_mean),
            _format_float(summary.collision_iou_median),
            _format_float(summary.collision_iou_p10),
            _format_float(summary.collision_iou_p90),
        ]
        for summary in summaries
    ]
    lines.extend(
        _markdown_table(
            [
                "Component",
                "Collision samples",
                "Collision boxes",
                "Mean IoU",
                "Median IoU",
                "p10",
                "p90",
            ],
            rows,
        )
    )
    lines.extend(["", "## Collision IoU (per-sample maximum)", ""])
    rows = [
        [
            summary.component,
            _format_count(summary.collision_samples),
            _format_float(summary.max_collision_iou_mean),
            _format_float(summary.max_collision_iou_median),
            _format_float(summary.max_collision_iou_p10),
            _format_float(summary.max_collision_iou_p90),
        ]
        for summary in summaries
    ]
    lines.extend(
        _markdown_table(
            [
                "Component",
                "Collision samples",
                "Mean max IoU",
                "Median max IoU",
                "p10",
                "p90",
            ],
            rows,
        )
    )
    lines.extend(["", "## Run checks", ""])
    rows = [
        [
            summary.component,
            _format_count(summary.samples),
            _format_count(summary.predicted),
            _format_count(summary.no_prediction),
            _format_percent(summary.collision_sample_rate),
            str(summary.detector_prediction_mismatches),
            str(summary.overlap_mismatches),
        ]
        for summary in summaries
    ]
    lines.extend(
        _markdown_table(
            [
                "Component",
                "Samples",
                "Predicted",
                "No prediction",
                "Collision sample rate",
                "Prediction-count mismatches",
                "Overlap mismatches",
            ],
            rows,
        )
    )
    lines.extend(["", "## Artifacts", ""])
    for summary in summaries:
        lines.append(
            f"- `{summary.component}`: `{summary.summary_path}`, "
            f"`{summary.telemetry_path}`"
        )
    lines.append("")
    return "\n".join(lines)


def _split_label(summaries: list[ComponentSummary]) -> str:
    """Return a human-readable split name for table captions.

    Args:
        summaries (list[ComponentSummary]): Component results.

    Returns:
        str: ``validation`` for a ``validate`` directory, otherwise the
            directory's own name, or ``evaluated`` when there are no rows.
    """
    if not summaries:
        return "evaluated"
    name: str = Path(summaries[0].split_dir).name
    return "validation" if name in {"validate", "validation"} else name


def _build_latex(summaries: list[ComponentSummary]) -> str:
    """Build the two LaTeX tables.

    Args:
        summaries (list[ComponentSummary]): Component results.

    Returns:
        str: LaTeX source with the matched-box and collision-IoU tables.
    """
    split_label: str = _split_label(summaries)
    matched_rows: list[str] = []
    collision_rows: list[str] = []
    for summary in summaries:
        component: str = summary.component.replace("_", r"\_")
        matched_rows.append(
            f"{component} & {_format_count(summary.matched_boxes)} & "
            f"{_format_float(summary.matched_iou_mean)} & "
            f"{_format_float(summary.matched_iou_median)} & "
            f"{_format_float(summary.matched_iou_p10)} & "
            f"{_format_float(summary.matched_iou_p90)} \\\\"
        )
        collision_rows.append(
            f"{component} & {_format_count(summary.collision_samples)} & "
            f"{_format_count(summary.collision_boxes)} & "
            f"{_format_float(summary.collision_iou_mean)} & "
            f"{_format_float(summary.collision_iou_median)} & "
            f"{_format_float(summary.collision_iou_p10)} & "
            f"{_format_float(summary.collision_iou_p90)} \\\\"
        )
    return "\n".join(
        [
            r"\begin{table}[ht]",
            r"    \centering",
            r"    \begin{tabular}{lrrrrr}",
            r"        \hline",
            r"        Component & Matched boxes & Mean IoU & Median IoU & p10 & p90 \\",
            r"        \hline",
            *matched_rows,
            r"        \hline",
            r"    \end{tabular}",
            f"    \\caption{{Intersection over Union (IoU) statistics for matched "
            f"target-component bounding boxes at IoU $\\ge 0.5$ on the {split_label} split.}}",
            r"    \label{tab:iou_statistics}",
            r"\end{table}",
            "",
            r"\begin{table}[ht]",
            r"    \centering",
            r"    \begin{tabular}{lrrrrrr}",
            r"        \hline",
            r"        Component & Collision samples & Collision boxes & Mean IoU & Median IoU & p10 & p90 \\",
            r"        \hline",
            *collision_rows,
            r"        \hline",
            r"    \end{tabular}",
            f"    \\caption{{Collision IoU between the final target-component "
            f"prediction and each other ground-truth furniture box with IoU $>0$ "
            f"on the {split_label} split. Collision samples counts predicted "
            f"samples with at least one positive pair; collision boxes counts "
            f"target--other pairs. Statistics are over those pairs.}}",
            r"    \label{tab:collision_iou}",
            r"\end{table}",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """Run the collision-IoU post-processing.

    Args:
        argv (list[str] | None): Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        int: Process exit code, ``0`` on success.
    """
    args: argparse.Namespace = parse_arguments(argv)
    manifest_path: Path = Path(args.manifest).expanduser().resolve()
    output_dir: Path = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else manifest_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = _load_json(manifest_path)
    components: list[str] = (
        list(args.components)
        if args.components is not None
        else sorted(
            manifest,
            key=lambda name: (
                PREFERRED_COMPONENT_ORDER.index(name)
                if name in PREFERRED_COMPONENT_ORDER
                else len(PREFERRED_COMPONENT_ORDER),
                name,
            ),
        )
    )
    missing: list[str] = [name for name in components if name not in manifest]
    if missing:
        raise KeyError(f"Components not present in {manifest_path}: {missing}.")

    hyperparams: SimpleNamespace = load_hyperparams(
        str(ROOT_DIR / "evaluator" / "hyperparameters.toml")
    )
    pair_path: Path = output_dir / "collision_iou_pairs.csv"
    sample_path: Path = output_dir / "collision_iou_samples.csv"
    summary_path: Path = output_dir / "collision_iou_summary.csv"
    markdown_path: Path = output_dir / "collision_iou_results.md"
    latex_path: Path = output_dir / "collision_iou_table.tex"

    summaries: list[ComponentSummary] = []
    with (
        pair_path.open("w", newline="", encoding="utf-8") as pair_handle,
        sample_path.open("w", newline="", encoding="utf-8") as sample_handle,
    ):
        pair_writer = csv.DictWriter(
            pair_handle,
            fieldnames=[
                "component",
                "stem",
                "target_confidence",
                "other_index",
                "other_name",
                "collision_iou",
                "overlap_fraction",
                "matched_boxes",
                "best_iou",
            ],
        )
        sample_writer = csv.DictWriter(
            sample_handle,
            fieldnames=[
                "component",
                "stem",
                "predicted",
                "target_confidence",
                "other_gt_boxes",
                "collision_boxes",
                "max_collision_iou",
                "sum_collision_iou",
                "overlap_fraction",
                "matched_boxes",
                "best_iou",
            ],
        )
        pair_writer.writeheader()
        sample_writer.writeheader()
        for component in components:
            print(f"Processing {component}...", flush=True)
            summaries.append(
                _summarize_component(
                    entry=dict(manifest[component]),
                    pair_csv=pair_writer,
                    sample_csv=sample_writer,
                    limit=args.limit,
                    device=args.device,
                    hyperparams=hyperparams,
                )
            )

    _write_summary_csv(summaries, summary_path)
    markdown_path.write_text(_build_markdown(summaries), encoding="utf-8")
    latex_path.write_text(_build_latex(summaries), encoding="utf-8")
    print(f"Pair CSV: {pair_path}")
    print(f"Sample CSV: {sample_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Markdown: {markdown_path}")
    print(f"LaTeX: {latex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
