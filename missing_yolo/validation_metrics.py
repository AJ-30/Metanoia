"""Detection metrics for the standalone ``missing_yolo`` validation run.

The trainer reports Ultralytics' own mAP/precision/recall at the end of a run,
but a report also needs thresholded, image-level and localisation numbers that
are easy to interpret and compare across the three components. This module owns
that arithmetic and nothing else:

* parse the rendered YOLO label files and the prediction files Ultralytics
  writes during ``model.val(save_txt=True)``;
* match predictions to ground truth greedily by confidence at a fixed IoU
  threshold (the same rule the evaluator uses);
* accumulate box-level TP/FP/FN, image-level detection/false-alarm counts,
  matched-box IoU and centre distance;
* choose the reproducible, balanced positive/negative sample subset used for the
  visual panels.

It is deliberately free of Ultralytics, torch, checkpoints and the filesystem
layout of a run, so it can be unit-tested on small files. The orchestration
lives in :mod:`missing_yolo.validate`.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from evaluator.box_metrics import (
    PredictedBox,
    box_loss,
    intersection_over_union,
    match_boxes,
)
from evaluator.dataset_gt import Box

# Fields in a rendered YOLO ground-truth file: ``cls cx cy w h``.
_GROUND_TRUTH_FIELDS: int = 5
# Fields in an Ultralytics ``save_txt=True, save_conf=True`` prediction file:
# ``cls cx cy w h confidence``.
_PREDICTION_FIELDS: int = 6

# Image-level outcome labels used in the per-image CSV and sample manifest.
STATUS_DETECTED_ALL: str = "detected_all"
STATUS_DETECTED_PARTIAL: str = "detected_partial"
STATUS_MISSED: str = "missed"
STATUS_FALSE_ALARM: str = "false_alarm"
STATUS_CORRECT_EMPTY: str = "correct_empty"


def _finite_float(raw: str, path: Path, line_number: int, field_name: str) -> float:
    """Parse one finite float from a label line.

    Args:
        raw (str): The text field to parse.
        path (Path): File being read, used in the error message.
        line_number (int): 1-based line number, used in the error message.
        field_name (str): Human-readable field name for the error message.

    Returns:
        float: The parsed finite value.

    Raises:
        ValueError: If the field is not numeric or is NaN/infinite, which would
            make every downstream IoU meaningless.
    """
    try:
        value: float = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{path}:{line_number}: {field_name} {raw!r} is not numeric."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"{path}:{line_number}: {field_name} {raw!r} is not finite."
        )
    return value


def _integer_field(raw: str, path: Path, line_number: int, field_name: str) -> int:
    """Parse one integer-valued label field without silently truncating it.

    Args:
        raw (str): The text field to parse.
        path (Path): File being read, used in the error message.
        line_number (int): 1-based line number, used in the error message.
        field_name (str): Human-readable field name for the error message.

    Returns:
        int: The parsed integer.

    Raises:
        ValueError: If the field is not numeric, not finite, or not a whole
            number. A class id of ``0.5`` must never be read as class ``0``.
    """
    value: float = _finite_float(raw, path, line_number, field_name)
    if not value.is_integer():
        raise ValueError(
            f"{path}:{line_number}: {field_name} {raw!r} is not an integer."
        )
    return int(value)


def read_ground_truth_labels(path: Path, class_id: int = 0) -> list[Box]:
    """Read one rendered YOLO ground-truth label file.

    The rendered dataset has one single-class label file per val sample: an
    empty file for a background sample, otherwise one ``<cls> cx cy w h`` line
    per box of the missing component.

    Args:
        path (Path): Label file path, normally
            ``<component>/dataset/labels/val/<stem>.txt``.
        class_id (int): The only class id this validation run scores.

    Returns:
        list[Box]: The sample's normalized ground-truth boxes, in file order.

    Raises:
        FileNotFoundError: If the label file is absent. A missing file would
            silently turn a positive sample into a background one.
        ValueError: If a line is malformed, names another class, or lies outside
            the normalized ``0..1`` range.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Ground-truth label file is missing: {path}. The rendered val "
            "split is incomplete; regenerate it instead of scoring a partial "
            "dataset."
        )
    boxes: list[Box] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts: list[str] = raw.split()
        if len(parts) != _GROUND_TRUTH_FIELDS:
            raise ValueError(
                f"{path}:{line_number}: expected {_GROUND_TRUTH_FIELDS} fields "
                f"(cls cx cy w h), got {len(parts)}: {raw!r}."
            )
        label_class: int = _integer_field(
            parts[0], path, line_number, "class id"
        )
        if label_class != int(class_id):
            raise ValueError(
                f"{path}:{line_number}: class id {label_class} is not the "
                f"validated class {class_id}; this stage is single-class."
            )
        values: list[float] = [
            _finite_float(part, path, line_number, name)
            for part, name in zip(parts[1:], ("cx", "cy", "w", "h"), strict=True)
        ]
        for value, name in zip(values, ("cx", "cy", "w", "h"), strict=True):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{path}:{line_number}: {name}={value} is outside the "
                    "normalized 0..1 range."
                )
        boxes.append(Box(cx=values[0], cy=values[1], w=values[2], h=values[3]))
    return boxes


def read_prediction_labels(path: Path, class_id: int = 0) -> list[PredictedBox]:
    """Read one Ultralytics prediction file written by ``save_txt``.

    Ultralytics writes one file per image that has at least one detection, in
    normalized ``cls cx cy w h confidence`` form (``save_conf=True``). Images
    with no detections have no file at all, which is a legitimate empty
    prediction rather than an error.

    Args:
        path (Path): Prediction file path, normally
            ``<ultralytics run>/labels/<stem>.txt``.
        class_id (int): The only class id this validation run scores; other
            classes are ignored.

    Returns:
        list[PredictedBox]: The component's predictions, in file order.

    Raises:
        ValueError: If a line has the wrong field count, a non-numeric or
            non-finite value, or a confidence outside ``0..1``.
    """
    if not path.is_file():
        return []
    predictions: list[PredictedBox] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts: list[str] = raw.split()
        if len(parts) != _PREDICTION_FIELDS:
            raise ValueError(
                f"{path}:{line_number}: expected {_PREDICTION_FIELDS} fields "
                f"(cls cx cy w h confidence), got {len(parts)}: {raw!r}."
            )
        predicted_class: int = _integer_field(
            parts[0], path, line_number, "class id"
        )
        values: list[float] = [
            _finite_float(part, path, line_number, name)
            for part, name in zip(
                parts[1:], ("cx", "cy", "w", "h", "confidence"), strict=True
            )
        ]
        if predicted_class != int(class_id):
            continue
        confidence: float = values[4]
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"{path}:{line_number}: confidence {confidence} is outside 0..1."
            )
        predictions.append(
            PredictedBox(
                box=Box(cx=values[0], cy=values[1], w=values[2], h=values[3]),
                confidence=confidence,
            )
        )
    return predictions


def summarize_values(values: Sequence[float]) -> dict[str, float] | None:
    """Summarize a distribution, or return None when it is empty.

    Args:
        values (Sequence[float]): Measurements such as matched-box IoUs.

    Returns:
        dict[str, float] | None: ``mean``, ``median``, ``p10``, ``p90``,
            ``min`` and ``max``; None when there is nothing to summarize, so a
            report can print ``n/a`` instead of a misleading zero.
    """
    if not values:
        return None
    ordered: list[float] = sorted(float(value) for value in values)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p10": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    """Return ``numerator / denominator``, or None for a zero denominator.

    Args:
        numerator (float): Count or value on top.
        denominator (float): Count or value below.

    Returns:
        float | None: The ratio, or None when it is undefined.
    """
    return None if denominator == 0 else numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    """Return the harmonic mean of precision and recall.

    Args:
        precision (float | None): Box precision, or None when undefined.
        recall (float | None): Box recall, or None when undefined.

    Returns:
        float | None: F1, or None when either input is undefined.
    """
    if precision is None or recall is None:
        return None
    denominator: float = precision + recall
    return None if denominator == 0.0 else 2.0 * precision * recall / denominator


def _best_localization(
    ground_truth: Sequence[Box],
    predictions: Sequence[PredictedBox],
    image_size: int,
) -> tuple[float | None, float | None]:
    """Return the best IoU and its centre distance over every GT/pred pair.

    This is a diagnostic independent of the matching threshold: it answers "how
    close was the closest prediction?" even when nothing was matched. The
    centre distance belongs to the same pair as the best IoU.

    Args:
        ground_truth (Sequence[Box]): Ground-truth boxes of the sample.
        predictions (Sequence[PredictedBox]): Predictions at the active
            confidence threshold.
        image_size (int): Render side length for the pixel centre distance.

    Returns:
        tuple[float | None, float | None]: Best IoU and its centre distance in
            pixels; ``(None, None)`` when either side is empty.
    """
    if not ground_truth or not predictions:
        return None, None
    best_iou: float = -1.0
    best_distance: float | None = None
    for gt_box in ground_truth:
        for prediction in predictions:
            iou: float = intersection_over_union(gt_box, prediction.box)
            if iou > best_iou:
                best_iou = iou
                best_distance = box_loss(gt_box, prediction.box, image_size).center_distance
    return best_iou, best_distance


@dataclass(frozen=True)
class SampleOutcome:
    """One image's outcome at one confidence threshold.

    Attributes:
        stem (str): Sample stem, also the image and label file stem.
        is_positive (bool): True when the dataset places the component in the
            image (at least one ground-truth box).
        gt_boxes (int): Number of ground-truth boxes.
        pred_boxes (int): Number of predictions kept at the threshold.
        tp (int): Ground-truth boxes matched by a prediction.
        fp (int): Predictions that matched no ground-truth box.
        fn (int): Ground-truth boxes nothing matched.
        best_iou (float | None): Best IoU over every GT/prediction pair.
        mean_matched_iou (float | None): Mean IoU of the matched boxes.
        best_center_distance_px (float | None): Centre distance of the
            best-IoU pair, in pixels.
        max_confidence (float | None): Highest prediction confidence, or None
            when the image had no predictions.
        status (str): Image-level outcome, one of the ``STATUS_*`` constants.
    """

    stem: str
    is_positive: bool
    gt_boxes: int
    pred_boxes: int
    tp: int
    fp: int
    fn: int
    best_iou: float | None
    mean_matched_iou: float | None
    best_center_distance_px: float | None
    max_confidence: float | None
    status: str

    def as_row(self, conf_threshold: float) -> dict[str, object]:
        """Convert the outcome to one CSV/JSON row.

        Args:
            conf_threshold (float): Confidence threshold the outcome was
                computed at.

        Returns:
            dict[str, object]: Flat row suitable for :mod:`csv` or JSON.
        """
        return {
            "stem": self.stem,
            "conf_threshold": float(conf_threshold),
            "is_positive": self.is_positive,
            "gt_boxes": self.gt_boxes,
            "pred_boxes": self.pred_boxes,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "best_iou": self.best_iou,
            "mean_matched_iou": self.mean_matched_iou,
            "best_center_distance_px": self.best_center_distance_px,
            "max_confidence": self.max_confidence,
            "status": self.status,
        }


@dataclass
class ThresholdAccumulator:
    """Running box- and image-level totals at one confidence threshold.

    Attributes:
        conf_threshold (float): Predictions below this confidence are ignored.
        iou_threshold (float): Minimum IoU for a prediction to claim a GT box.
        image_size (int): Render side length for pixel centre distances.
        tp (int): Matched ground-truth boxes.
        fp (int): Unmatched predictions.
        fn (int): Unmatched ground-truth boxes.
        positive_images (int): Images with at least one GT box.
        detected_positive_images (int): Positive images with at least one TP.
        background_images (int): Images with no GT box.
        false_alarm_images (int): Background images with at least one FP.
        matched_ious (list[float]): IoU of every matched box.
        center_distances (list[float]): Centre distance of every matched box.
        l1_norms (list[float]): Mean absolute box-parameter error of every match.
        tp_confidences (list[float]): Confidence of every matched prediction.
        fp_confidences (list[float]): Confidence of every false positive.
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
    matched_ious: list[float] = field(default_factory=list)
    center_distances: list[float] = field(default_factory=list)
    l1_norms: list[float] = field(default_factory=list)
    tp_confidences: list[float] = field(default_factory=list)
    fp_confidences: list[float] = field(default_factory=list)

    def add(
        self,
        stem: str,
        ground_truth: Sequence[Box],
        predictions: Sequence[PredictedBox],
    ) -> SampleOutcome:
        """Fold one image into the running totals.

        Args:
            stem (str): Sample stem.
            ground_truth (Sequence[Box]): The sample's GT boxes.
            predictions (Sequence[PredictedBox]): All predictions of the
                sample; those below ``conf_threshold`` are dropped here.

        Returns:
            SampleOutcome: The sample's outcome at this threshold.
        """
        kept: list[PredictedBox] = [
            prediction
            for prediction in predictions
            if prediction.confidence >= self.conf_threshold
        ]
        outcome = match_boxes(
            list(ground_truth), kept, self.iou_threshold, self.image_size
        )
        matched: list[float] = [
            float(match.loss.iou)
            for match in outcome.matches
            if match.loss is not None
        ]
        tp: int = len(matched)
        fn: int = len(ground_truth) - tp
        fp: int = len(outcome.false_positives)
        matched_predictions: list[PredictedBox] = [
            match.prediction
            for match in outcome.matches
            if match.prediction is not None
        ]

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.matched_ious.extend(matched)
        for match in outcome.matches:
            if match.loss is not None:
                self.center_distances.append(match.loss.center_distance)
                self.l1_norms.append(match.loss.l1_norm)
        for prediction in matched_predictions:
            self.tp_confidences.append(prediction.confidence)
        for prediction in outcome.false_positives:
            self.fp_confidences.append(prediction.confidence)

        is_positive: bool = bool(ground_truth)
        if is_positive:
            self.positive_images += 1
            if tp > 0:
                self.detected_positive_images += 1
        else:
            self.background_images += 1
            if fp > 0:
                self.false_alarm_images += 1

        best_iou, best_distance = _best_localization(
            ground_truth, kept, self.image_size
        )
        if not is_positive:
            status: str = STATUS_FALSE_ALARM if fp > 0 else STATUS_CORRECT_EMPTY
        elif tp == 0:
            status = STATUS_MISSED
        elif tp == len(ground_truth):
            status = STATUS_DETECTED_ALL
        else:
            status = STATUS_DETECTED_PARTIAL

        return SampleOutcome(
            stem=stem,
            is_positive=is_positive,
            gt_boxes=len(ground_truth),
            pred_boxes=len(kept),
            tp=tp,
            fp=fp,
            fn=fn,
            best_iou=best_iou,
            mean_matched_iou=statistics.fmean(matched) if matched else None,
            best_center_distance_px=best_distance,
            max_confidence=(
                max(prediction.confidence for prediction in kept) if kept else None
            ),
            status=status,
        )

    def summary(self) -> dict[str, object]:
        """Turn the running totals into a report dictionary.

        Returns:
            dict[str, object]: Box counts and rates, image-level counts and
                rates, localisation distributions and confidence means. Undefined
                ratios are None, never 0.0.
        """
        precision: float | None = _ratio(self.tp, self.tp + self.fp)
        recall: float | None = _ratio(self.tp, self.tp + self.fn)
        f1: float | None = _f1(precision, recall)
        jaccard: float | None = _ratio(self.tp, self.tp + self.fp + self.fn)
        total_images: int = self.positive_images + self.background_images
        correct_empty: int = self.background_images - self.false_alarm_images
        image_accuracy: float | None = _ratio(
            self.detected_positive_images + correct_empty, total_images
        )
        detection_rate: float | None = _ratio(
            self.detected_positive_images, self.positive_images
        )
        false_alarm_rate: float | None = _ratio(
            self.false_alarm_images, self.background_images
        )
        specificity: float | None = _ratio(correct_empty, self.background_images)
        balanced_accuracy: float | None = (
            None
            if detection_rate is None or specificity is None
            else (detection_rate + specificity) / 2.0
        )
        return {
            "conf_threshold": float(self.conf_threshold),
            "iou_threshold": float(self.iou_threshold),
            "box_counts": {
                "tp": self.tp,
                "fp": self.fp,
                "fn": self.fn,
                "ground_truth": self.tp + self.fn,
                "predictions": self.tp + self.fp,
            },
            "box_metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "jaccard": jaccard,
            },
            "image_counts": {
                "total": total_images,
                "positive": self.positive_images,
                "detected_positive": self.detected_positive_images,
                "missed_positive": self.positive_images
                - self.detected_positive_images,
                "background": self.background_images,
                "false_alarm": self.false_alarm_images,
                "correct_empty": correct_empty,
            },
            "image_metrics": {
                "accuracy": image_accuracy,
                "positive_detection_rate": detection_rate,
                "false_alarm_rate": false_alarm_rate,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
            },
            "localization": {
                "matched_iou": summarize_values(self.matched_ious),
                "center_distance_px": summarize_values(self.center_distances),
                "l1_norm_xywh": summarize_values(self.l1_norms),
            },
            "confidence": {
                "mean_true_positive": (
                    statistics.fmean(self.tp_confidences)
                    if self.tp_confidences
                    else None
                ),
                "mean_false_positive": (
                    statistics.fmean(self.fp_confidences)
                    if self.fp_confidences
                    else None
                ),
            },
        }


def threshold_key(conf_threshold: float) -> str:
    """Format a confidence threshold for use as a JSON/report key.

    Args:
        conf_threshold (float): Confidence threshold in ``0..1``.

    Returns:
        str: Compact representation, e.g. ``"0.25"``.
    """
    return f"{float(conf_threshold):g}"


def evaluate_split(
    stems: Sequence[str],
    ground_truth_dir: Path,
    prediction_dir: Path,
    conf_thresholds: Sequence[float],
    iou_threshold: float,
    image_size: int,
    class_id: int = 0,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    """Evaluate every image of one split at every requested confidence.

    Args:
        stems (Sequence[str]): Sample stems to evaluate, in dataset order.
        ground_truth_dir (Path): Rendered labels directory.
        prediction_dir (Path): Ultralytics ``save_txt`` labels directory.
        conf_thresholds (Sequence[float]): Confidence thresholds to report.
        iou_threshold (float): Minimum IoU for a positive match.
        image_size (int): Render side length for pixel distances.
        class_id (int): Only class id scored.

    Returns:
        tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
            ``(threshold_summaries, per_image_rows)``. Summaries are keyed by
            :func:`threshold_key`; rows are in long format (one row per
            sample and threshold).

    Raises:
        ValueError: If there are no stems, a threshold is outside ``0..1``, or
            the IoU threshold is outside ``0..1``.
    """
    if not stems:
        raise ValueError("Cannot evaluate an empty validation split.")
    thresholds: list[float] = sorted({float(value) for value in conf_thresholds})
    if not thresholds:
        raise ValueError("At least one confidence threshold is required.")
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Confidence threshold {threshold} is outside 0..1."
            )
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError(f"Match IoU threshold {iou_threshold} is outside 0..1.")

    accumulators: dict[float, ThresholdAccumulator] = {
        threshold: ThresholdAccumulator(
            conf_threshold=threshold,
            iou_threshold=float(iou_threshold),
            image_size=int(image_size),
        )
        for threshold in thresholds
    }
    rows: list[dict[str, object]] = []
    for stem in stems:
        ground_truth: list[Box] = read_ground_truth_labels(
            ground_truth_dir / f"{stem}.txt", class_id
        )
        predictions: list[PredictedBox] = read_prediction_labels(
            prediction_dir / f"{stem}.txt", class_id
        )
        for threshold in thresholds:
            outcome: SampleOutcome = accumulators[threshold].add(
                stem, ground_truth, predictions
            )
            rows.append(outcome.as_row(threshold))

    return {
        threshold_key(threshold): accumulator.summary()
        for threshold, accumulator in accumulators.items()
    }, rows


def choose_balanced_stems(
    positive_stems: Sequence[str],
    negative_stems: Sequence[str],
    positive_count: int,
    negative_count: int,
    seed: int,
) -> list[tuple[str, bool]]:
    """Choose a reproducible, balanced subset of positive and negative images.

    If one class has fewer stems than requested, the remaining slots are filled
    from the other class so a small smoke run still produces a complete panel.
    Positives are returned first, then negatives, then any fill stems.

    Args:
        positive_stems (Sequence[str]): Stems with at least one GT box.
        negative_stems (Sequence[str]): Stems with no GT box.
        positive_count (int): Desired positive images.
        negative_count (int): Desired negative images.
        seed (int): Seed for the reproducible shuffle.

    Returns:
        list[tuple[str, bool]]: ``(stem, is_positive)`` pairs in display order.

    Raises:
        ValueError: If a requested count is negative or a stem appears in both
            classes, which would make the selection ambiguous.
    """
    if positive_count < 0 or negative_count < 0:
        raise ValueError(
            f"Sample counts must be non-negative, got "
            f"{positive_count} positives and {negative_count} negatives."
        )
    positive_set: set[str] = set(positive_stems)
    negative_set: set[str] = set(negative_stems)
    overlap: set[str] = positive_set & negative_set
    if overlap:
        raise ValueError(
            f"{len(overlap)} stems are both positive and negative, e.g. "
            f"{sorted(overlap)[:3]}."
        )
    all_stems: list[str] = sorted(positive_set | negative_set)
    target: int = min(positive_count + negative_count, len(all_stems))
    if target == 0:
        return []

    rng: random.Random = random.Random(int(seed))
    take_positive: int = min(positive_count, len(positive_set), target)
    chosen_positive: list[str] = sorted(
        rng.sample(sorted(positive_set), take_positive)
    )
    remaining: int = target - take_positive
    take_negative: int = min(negative_count, len(negative_set), remaining)
    chosen_negative: list[str] = sorted(
        rng.sample(sorted(negative_set), take_negative)
    )

    used: set[str] = set(chosen_positive) | set(chosen_negative)
    unused: list[str] = [stem for stem in all_stems if stem not in used]
    fill_count: int = target - len(used)
    fill: list[str] = sorted(rng.sample(unused, fill_count)) if fill_count else []

    ordered: list[tuple[str, bool]] = [
        *((stem, True) for stem in chosen_positive),
        *((stem, False) for stem in chosen_negative),
        *((stem, stem in positive_set) for stem in fill),
    ]
    return ordered
