"""Ground-truth versus prediction scoring for one frontend upload.

The upload page can optionally take a YOLO ``labels.txt`` for the layout the
component was generated into. This module owns the two things that upload then
needs and nothing else:

* :func:`read_ground_truth_labels` - validate a user-supplied label file and
  turn its normalized ``cx cy w h`` lines into :class:`evaluator.dataset_gt.Box`
  values. The file may use the full YOLO form ``cls cx cy w h``, the dataset's
  own named form ``Name cx cy w h`` (e.g. ``TV cx cy w h``), or the shorter
  ``cx cy w h``; a blank file is a legitimate negative sample.
* :func:`compare_boxes` - pair ground-truth boxes with the detector's boxes
  using the evaluator's greedy IoU matching, then turn the pairing into the
  counts, rates and localization distributions the result page shows.

The arithmetic is imported from :mod:`evaluator.box_metrics` so the frontend
scores an upload exactly the way the evaluator scores a dataset split. Nothing
here loads a model, touches torch or writes a file, which keeps the comparison
unit-testable without a GPU.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from evaluator.box_metrics import (
    PredictedBox,
    intersection_over_union,
    match_boxes,
)
from evaluator.dataset_gt import Box

# Accepted label-line widths: with a leading class field (a numeric YOLO class
# id or the dataset's component name) or without one (a single-class detector
# makes the class field redundant).
_LABEL_FIELDS_WITH_CLASS: int = 5
_LABEL_FIELDS_WITHOUT_CLASS: int = 4

# Image-level outcome labels shown on the result page.
STATUS_DETECTED_ALL: str = "detected_all"
STATUS_DETECTED_PARTIAL: str = "detected_partial"
STATUS_MISSED: str = "missed"
STATUS_FALSE_ALARM: str = "false_alarm"
STATUS_CORRECT_EMPTY: str = "correct_empty"

# Human-readable status captions, kept beside the constants so the template
# never has to branch on magic strings.
STATUS_CAPTIONS: dict[str, str] = {
    STATUS_DETECTED_ALL: "every ground-truth box matched",
    STATUS_DETECTED_PARTIAL: "some ground-truth boxes matched",
    STATUS_MISSED: "no ground-truth box matched",
    STATUS_FALSE_ALARM: "no ground truth, but predictions were made",
    STATUS_CORRECT_EMPTY: "no ground truth and no predictions",
}


@dataclass(frozen=True)
class MetricSummary:
    """Distribution of one measurement over the matched boxes.

    Attributes:
        count (int): Number of measurements summarized.
        mean (float): Arithmetic mean.
        median (float): Median.
        p10 (float): Nearest-rank 10th percentile.
        p90 (float): Nearest-rank 90th percentile.
        minimum (float): Smallest value.
        maximum (float): Largest value.
    """

    count: int
    mean: float
    median: float
    p10: float
    p90: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class BoxMatch:
    """One ground-truth box and the prediction that claimed it, if any.

    Attributes:
        gt_index (int): Zero-based index of the box in the uploaded file.
        ground_truth (Box): The uploaded ground-truth box, normalized.
        prediction (PredictedBox | None): The matched prediction, or ``None``
            when no prediction reached the IoU threshold (a miss).
        iou (float | None): IoU of the pair, ``None`` for a miss.
        center_distance_px (float | None): Centre distance in render pixels,
            ``None`` for a miss.
        l1_norm (float | None): Mean absolute ``(cx, cy, w, h)`` error,
            ``None`` for a miss.
    """

    gt_index: int
    ground_truth: Box
    prediction: PredictedBox | None
    iou: float | None
    center_distance_px: float | None
    l1_norm: float | None


@dataclass(frozen=True)
class FalsePositive:
    """One prediction that matched no ground-truth box.

    Attributes:
        prediction (PredictedBox): The unmatched prediction.
        best_iou (float | None): Its highest IoU against any ground-truth box,
            ``None`` when the upload had no ground truth at all.
    """

    prediction: PredictedBox
    best_iou: float | None


@dataclass(frozen=True)
class DetectionComparison:
    """Everything the result page reports about one draw's boxes.

    Attributes:
        iou_threshold (float): Minimum IoU that counts as a match.
        ground_truth_boxes (int): Boxes read from the uploaded labels.
        predicted_boxes (int): Boxes the detector returned above its threshold.
        matched_boxes (int): Ground-truth boxes claimed by a prediction (TP).
        false_positive_boxes (int): Predictions that claimed nothing (FP).
        missed_boxes (int): Ground-truth boxes nothing claimed (FN).
        precision (float | None): ``TP / (TP + FP)``; ``None`` when undefined.
        recall (float | None): ``TP / (TP + FN)``; ``None`` when undefined.
        f1 (float | None): Harmonic mean of precision and recall.
        jaccard (float | None): ``TP / (TP + FP + FN)``, box-level agreement.
        mean_iou_over_union (float | None): Matched IoU divided by the union of
            GT and predicted boxes; unmatched boxes contribute zero.
        best_iou (float | None): Highest IoU over every GT/prediction pair,
            including pairs below the match threshold.
        matched_iou (MetricSummary | None): IoU distribution of matched pairs.
        center_distance_px (MetricSummary | None): Matched centre distances in
            render pixels.
        center_distance_normalized (MetricSummary | None): The same distances
            divided by the render side length.
        l1_norm (MetricSummary | None): Matched mean absolute box error.
        mean_tp_confidence (float | None): Mean confidence of matched boxes.
        mean_fp_confidence (float | None): Mean confidence of false positives.
        status (str): Image-level outcome, one of the ``STATUS_*`` constants.
        pairs (tuple[BoxMatch, ...]): One entry per ground-truth box, file order.
        false_positives (tuple[FalsePositive, ...]): Unmatched predictions,
            confidence order.
    """

    iou_threshold: float
    ground_truth_boxes: int
    predicted_boxes: int
    matched_boxes: int
    false_positive_boxes: int
    missed_boxes: int
    precision: float | None
    recall: float | None
    f1: float | None
    jaccard: float | None
    mean_iou_over_union: float | None
    best_iou: float | None
    matched_iou: MetricSummary | None
    center_distance_px: MetricSummary | None
    center_distance_normalized: MetricSummary | None
    l1_norm: MetricSummary | None
    mean_tp_confidence: float | None
    mean_fp_confidence: float | None
    status: str
    pairs: tuple[BoxMatch, ...]
    false_positives: tuple[FalsePositive, ...]

    @property
    def status_caption(self) -> str:
        """Return the readable explanation of :attr:`status`.

        Returns:
            str: A short sentence suitable for the result page.
        """
        return STATUS_CAPTIONS[self.status]


def _parse_number(raw: str, line_number: int, field_name: str) -> float:
    """Parse one finite numeric label field.

    Args:
        raw (str): Text field from the uploaded file.
        line_number (int): 1-based line number, for the error message.
        field_name (str): Human-readable field name, for the error message.

    Returns:
        float: The parsed finite value.

    Raises:
        ValueError: If the field is not numeric or is NaN/infinite.
    """
    try:
        value: float = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Ground-truth labels, line {line_number}: {field_name} {raw!r} "
            "is not numeric."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Ground-truth labels, line {line_number}: {field_name} {raw!r} "
            "is not finite."
        )
    return value


def _parse_class_id(raw: str, line_number: int) -> int:
    """Parse one integer class id without silently truncating it.

    Args:
        raw (str): Text field from the uploaded file.
        line_number (int): 1-based line number, for the error message.

    Returns:
        int: The parsed class id.

    Raises:
        ValueError: If the field is not a whole, finite number. A class id of
            ``0.5`` must never be read as class ``0``.
    """
    value: float = _parse_number(raw, line_number, "class id")
    if not value.is_integer():
        raise ValueError(
            f"Ground-truth labels, line {line_number}: class id {raw!r} is "
            "not an integer."
        )
    return int(value)


def _looks_numeric(raw: str) -> bool:
    """Report whether a label field starts with a number-like value.

    Args:
        raw (str): The first field of one label line.

    Returns:
        bool: ``True`` when :func:`float` would accept the field (including
            ``nan``/``inf``, so those reach the strict numeric parser and fail
            there with a precise message); ``False`` for a class name such as
            ``TV``.
    """
    try:
        float(raw)
    except ValueError:
        return False
    return True


def _class_field_belongs_to_component(
    raw: str,
    line_number: int,
    class_id: int,
    label_names: Sequence[str],
) -> bool:
    """Decide whether a five-field line describes the scored component.

    The first field is either a numeric YOLO class id or one of the dataset's
    own label names. A numeric id must be exactly ``class_id``; a name must be
    one of ``label_names``. A non-numeric name that is not this component is a
    line for another piece of furniture in the same dataset ``labels.txt``, so
    it is skipped the way :func:`evaluator.dataset_gt.parse_label_line` skips
    it, rather than failing the whole upload.

    Args:
        raw (str): First field of the label line.
        line_number (int): 1-based line number, for the error message.
        class_id (int): The selected detector's class id.
        label_names (Sequence[str]): Dataset label names of the selected
            component, e.g. ``("TV",)``. Empty when the caller only supports
            numeric YOLO files.

    Returns:
        bool: ``True`` when the line belongs to the selected component,
            ``False`` when it names another component and should be skipped.

    Raises:
        ValueError: If a numeric field is not a valid integer class id, or is a
            different class id than the selected component.
    """
    if label_names and raw in label_names:
        return True
    if _looks_numeric(raw):
        label_class: int = _parse_class_id(raw, line_number)
        if label_class != int(class_id):
            raise ValueError(
                f"Ground-truth labels, line {line_number}: class id "
                f"{label_class} is not the selected component's class "
                f"{class_id} ({', '.join(label_names) or 'unnamed'}). Upload "
                "labels for this component, name the component, or omit the "
                "class field (cx cy w h)."
            )
        return True
    if label_names:
        # A named line for another furniture class: the dataset's labels.txt
        # lists every object in the layout, not only the missing component.
        return False
    # No names were supplied, so a non-numeric field can only be a malformed
    # class id; report it through the numeric parser's clear message.
    _parse_number(raw, line_number, "class id")
    return True  # pragma: no cover - _parse_number always raises here


def _parse_box(values: Sequence[float], line_number: int) -> Box:
    """Validate four normalized box values and build the box.

    Args:
        values (Sequence[float]): ``(cx, cy, w, h)`` as written in the file.
        line_number (int): 1-based line number, for the error message.

    Returns:
        Box: The normalized ground-truth box.

    Raises:
        ValueError: If a value is outside ``0..1``, or if width or height is
            zero. A zero-area box would make every IoU against it meaningless.
    """
    for value, name in zip(values, ("cx", "cy", "w", "h"), strict=True):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Ground-truth labels, line {line_number}: {name}={value} is "
                "outside the normalized 0..1 range."
            )
    if values[2] <= 0.0 or values[3] <= 0.0:
        raise ValueError(
            f"Ground-truth labels, line {line_number}: width and height must "
            "be positive."
        )
    return Box(cx=values[0], cy=values[1], w=values[2], h=values[3])


def read_ground_truth_labels(
    path: Path,
    class_id: int,
    max_boxes: int,
    label_names: Sequence[str] = (),
) -> list[Box]:
    """Read and validate one uploaded YOLO or dataset label file.

    Accepted lines are ``cls cx cy w h`` (the normal YOLO form, with a numeric
    class id), ``Name cx cy w h`` (the dataset's own ``labels.txt`` form, e.g.
    ``TV cx cy w h``) or ``cx cy w h`` (class field implied). Blank lines and a
    completely empty file are valid: they describe a layout where the component
    is absent, which is a negative sample rather than an error.

    When ``label_names`` is supplied, named lines for another furniture class
    are skipped exactly as the evaluator skips them, so a raw dataset
    ``labels.txt`` listing every object can be uploaded unchanged.

    Args:
        path (Path): Uploaded label file, UTF-8 text.
        class_id (int): The only numeric class id this upload scores. A numeric
            line with another class id is refused instead of being silently
            dropped, because that almost always means the wrong component's
            labels were uploaded.
        max_boxes (int): Upper bound on accepted boxes, so one upload cannot
            turn the page into an unbounded table.
        label_names (Sequence[str]): Dataset label names of the selected
            component, e.g. ``("TV",)``. A line whose first field is one of
            these is a box of this component; a line naming any other component
            is skipped. Empty means only numeric class ids are understood.

    Returns:
        list[Box]: The file's boxes in file order.

    Raises:
        FileNotFoundError: If ``path`` is not a regular file.
        ValueError: If the file is not valid UTF-8, has a malformed line, names
            another numeric class, holds a value outside ``0..1``, has a
            zero-area box, or exceeds ``max_boxes``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth label file is missing: {path}.")
    if max_boxes < 0:
        raise ValueError(f"max_boxes must be non-negative, got {max_boxes}.")
    try:
        text: str = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "The ground-truth labels file is not valid UTF-8 text. Upload the "
            "plain-text labels.txt, not an image or a binary file."
        ) from exc

    boxes: list[Box] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        parts: list[str] = raw.split()
        if len(parts) == _LABEL_FIELDS_WITH_CLASS:
            belongs: bool = _class_field_belongs_to_component(
                parts[0], line_number, class_id, label_names
            )
            if not belongs:
                # Another object in the dataset's labels.txt, not the component.
                continue
            values: list[float] = [
                _parse_number(part, line_number, name)
                for part, name in zip(parts[1:], ("cx", "cy", "w", "h"), strict=True)
            ]
        elif len(parts) == _LABEL_FIELDS_WITHOUT_CLASS:
            values = [
                _parse_number(part, line_number, name)
                for part, name in zip(parts, ("cx", "cy", "w", "h"), strict=True)
            ]
        else:
            raise ValueError(
                f"Ground-truth labels, line {line_number}: expected "
                f"{_LABEL_FIELDS_WITH_CLASS} fields (cls/name cx cy w h) or "
                f"{_LABEL_FIELDS_WITHOUT_CLASS} fields (cx cy w h), got "
                f"{len(parts)}: {raw!r}."
            )
        boxes.append(_parse_box(values, line_number))
        if len(boxes) > max_boxes:
            raise ValueError(
                f"Ground-truth labels hold more than {max_boxes} boxes, the "
                "upload limit. Split the labels or raise "
                "MAX_GROUND_TRUTH_BOXES in frontend/hyperparameters.toml."
            )
    return boxes


def _summarize(values: Sequence[float]) -> MetricSummary | None:
    """Summarize a distribution, or return None when it is empty.

    Args:
        values (Sequence[float]): Measurements such as matched-box IoUs.

    Returns:
        MetricSummary | None: Mean, median, percentiles and extremes; ``None``
            when there is nothing to summarize, so the page can print ``n/a``
            instead of a misleading zero.
    """
    if not values:
        return None
    ordered: list[float] = sorted(float(value) for value in values)
    return MetricSummary(
        count=len(ordered),
        mean=statistics.fmean(ordered),
        median=statistics.median(ordered),
        p10=ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        p90=ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        minimum=ordered[0],
        maximum=ordered[-1],
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
        float | None: F1, or ``None`` when either input is undefined or both
            are zero.
    """
    if precision is None or recall is None:
        return None
    denominator: float = precision + recall
    return None if denominator == 0.0 else 2.0 * precision * recall / denominator


def _image_status(
    ground_truth_boxes: int,
    matched_boxes: int,
    false_positive_boxes: int,
) -> str:
    """Classify one image's box outcome.

    Args:
        ground_truth_boxes (int): Number of uploaded ground-truth boxes.
        matched_boxes (int): Number of those boxes a prediction claimed.
        false_positive_boxes (int): Predictions that claimed nothing.

    Returns:
        str: One of the ``STATUS_*`` constants.
    """
    if ground_truth_boxes == 0:
        return STATUS_FALSE_ALARM if false_positive_boxes > 0 else STATUS_CORRECT_EMPTY
    if matched_boxes == 0:
        return STATUS_MISSED
    if matched_boxes == ground_truth_boxes:
        return STATUS_DETECTED_ALL
    return STATUS_DETECTED_PARTIAL


def compare_boxes(
    ground_truth: Sequence[Box],
    predictions: Sequence[PredictedBox],
    iou_threshold: float,
    image_size: int,
) -> DetectionComparison:
    """Pair uploaded ground truth with one draw's predictions and score them.

    Predictions are expected to be the detector's boxes already filtered to the
    selected component and to the page's confidence threshold. Matching is the
    evaluator's greedy, confidence-ordered IoU assignment, so the page's
    precision/recall mean the same thing as the offline report's.

    Args:
        ground_truth (Sequence[Box]): Uploaded ground-truth boxes, normalized.
            Empty is valid and describes a negative layout.
        predictions (Sequence[PredictedBox]): Detector boxes, normalized.
        iou_threshold (float): Minimum IoU for a prediction to claim a box.
        image_size (int): Side length of the square render the boxes live in,
            used to report centre distances in pixels.

    Returns:
        DetectionComparison: Counts, rates, distributions and the per-box
        pairing.

    Raises:
        ValueError: If the IoU threshold is outside ``0..1`` or ``image_size``
            is not positive.
    """
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError(f"Match IoU threshold {iou_threshold} is outside 0..1.")
    if int(image_size) <= 0:
        raise ValueError(f"Image size must be positive, got {image_size}.")

    ground_truth_list: list[Box] = list(ground_truth)
    prediction_list: list[PredictedBox] = list(predictions)
    outcome = match_boxes(
        ground_truth_list,
        prediction_list,
        float(iou_threshold),
        int(image_size),
    )

    pairs: list[BoxMatch] = []
    matched_ious: list[float] = []
    center_distances_px: list[float] = []
    l1_norms: list[float] = []
    tp_confidences: list[float] = []
    for match in outcome.matches:
        if match.prediction is None or match.loss is None:
            pairs.append(
                BoxMatch(
                    gt_index=match.gt_index,
                    ground_truth=ground_truth_list[match.gt_index],
                    prediction=None,
                    iou=None,
                    center_distance_px=None,
                    l1_norm=None,
                )
            )
            continue
        pairs.append(
            BoxMatch(
                gt_index=match.gt_index,
                ground_truth=ground_truth_list[match.gt_index],
                prediction=match.prediction,
                iou=match.loss.iou,
                center_distance_px=match.loss.center_distance,
                l1_norm=match.loss.l1_norm,
            )
        )
        matched_ious.append(match.loss.iou)
        center_distances_px.append(match.loss.center_distance)
        l1_norms.append(match.loss.l1_norm)
        tp_confidences.append(match.prediction.confidence)

    false_positives: list[FalsePositive] = [
        FalsePositive(
            prediction=prediction,
            best_iou=(
                max(
                    (
                        intersection_over_union(gt_box, prediction.box)
                        for gt_box in ground_truth_list
                    ),
                    default=None,
                )
                if ground_truth_list
                else None
            ),
        )
        for prediction in outcome.false_positives
    ]

    best_iou: float | None = max(
        (
            intersection_over_union(gt_box, prediction.box)
            for gt_box in ground_truth_list
            for prediction in prediction_list
        ),
        default=None,
    )
    matched_boxes: int = len(matched_ious)
    false_positive_boxes: int = len(false_positives)
    ground_truth_boxes: int = len(ground_truth_list)
    missed_boxes: int = ground_truth_boxes - matched_boxes
    precision: float | None = _ratio(
        matched_boxes, matched_boxes + false_positive_boxes
    )
    recall: float | None = _ratio(matched_boxes, ground_truth_boxes)
    union_boxes: int = matched_boxes + false_positive_boxes + missed_boxes

    return DetectionComparison(
        iou_threshold=float(iou_threshold),
        ground_truth_boxes=ground_truth_boxes,
        predicted_boxes=len(prediction_list),
        matched_boxes=matched_boxes,
        false_positive_boxes=false_positive_boxes,
        missed_boxes=missed_boxes,
        precision=precision,
        recall=recall,
        f1=_harmonic_mean(precision, recall),
        jaccard=_ratio(matched_boxes, union_boxes),
        mean_iou_over_union=_ratio(sum(matched_ious), union_boxes),
        best_iou=best_iou,
        matched_iou=_summarize(matched_ious),
        center_distance_px=_summarize(center_distances_px),
        center_distance_normalized=_summarize(
            [distance / float(image_size) for distance in center_distances_px]
        ),
        l1_norm=_summarize(l1_norms),
        mean_tp_confidence=(
            statistics.fmean(tp_confidences) if tp_confidences else None
        ),
        mean_fp_confidence=(
            statistics.fmean(
                [item.prediction.confidence for item in false_positives]
            )
            if false_positives
            else None
        ),
        status=_image_status(
            ground_truth_boxes, matched_boxes, false_positive_boxes
        ),
        pairs=tuple(pairs),
        false_positives=tuple(false_positives),
    )
