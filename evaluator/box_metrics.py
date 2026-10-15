"""The bounding-box loss: dataset ground truth versus YOLO's prediction.

The evaluator's headline number is *where the generated component ends up as a
detector sees it*. Two boxes are involved, both for the missing component only:

* the **ground truth** - the dataset's own ``labels.txt`` box, which is never
  derived from a segmentation mask and never from the diffusion output;
* the **prediction** - the box the trained detector returns for that class on the
  composite image.

Pairing them needs a defensible rule, so a standard greedy-IoU assignment is
used: predictions are taken in confidence order and each takes the best
still-free ground-truth box once it clears ``BOX_MATCH_IOU``. Whatever is left
unpaired is a miss (ground truth the detector did not see) or a false positive
(box the detector invented where the dataset says there is none).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from evaluator.dataset_gt import Box


@dataclass(frozen=True)
class PredictedBox:
    """One predicted box for the missing component.

    Attributes:
        box (Box): The predicted box in normalized ``0..1`` coordinates.
        confidence (float): Detector confidence in ``0..1``.
    """

    box: Box
    confidence: float


@dataclass(frozen=True)
class BoxLoss:
    """Localization error of one ground-truth box against one prediction.

    Attributes:
        iou (float): Intersection over union in normalized coordinates.
        l1_norm (float): Mean absolute difference of ``(cx, cy, w, h)``; 0 is
            an exact match and 1 is the worst possible disagreement.
        center_distance (float): Euclidean distance between box centres in
            pixels of the render domain.
    """

    iou: float
    l1_norm: float
    center_distance: float


@dataclass(frozen=True)
class Match:
    """The outcome of pairing one ground-truth box with the predictions.

    Attributes:
        gt_index (int): Index of the ground-truth box in the sample's box list.
        prediction (PredictedBox | None): Matched prediction, or ``None`` when
            no prediction reached the IoU threshold (a miss).
        loss (BoxLoss | None): Localization error, ``None`` for a miss.
    """

    gt_index: int
    prediction: PredictedBox | None
    loss: BoxLoss | None


@dataclass(frozen=True)
class MatchOutcome:
    """Everything a single sample/draw pairing produced.

    Attributes:
        matches (list[Match]): One entry per ground-truth box, in order.
        false_positives (list[PredictedBox]): Predictions that matched nothing.
    """

    matches: list[Match] = field(default_factory=list)
    false_positives: list[PredictedBox] = field(default_factory=list)


def intersection_over_union(gt: Box, prediction: Box) -> float:
    """Compute the IoU of two boxes in normalized coordinates.

    Args:
        gt (Box): Ground-truth box.
        prediction (Box): Predicted box.

    Returns:
        float: IoU in ``0..1``; ``0.0`` when the boxes do not overlap (or when
            either box is empty, which cannot happen for a parsed label).
    """
    gt_x1, gt_y1, gt_x2, gt_y2 = gt.to_pixels(1)
    pd_x1, pd_y1, pd_x2, pd_y2 = prediction.to_pixels(1)
    inter_x1: float = max(gt_x1, pd_x1)
    inter_y1: float = max(gt_y1, pd_y1)
    inter_x2: float = min(gt_x2, pd_x2)
    inter_y2: float = min(gt_y2, pd_y2)
    intersection: float = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    gt_area: float = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    pd_area: float = (pd_x2 - pd_x1) * (pd_y2 - pd_y1)
    union: float = gt_area + pd_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def box_loss(gt: Box, prediction: Box, image_size: int) -> BoxLoss:
    """Score one ground-truth box against one prediction.

    Args:
        gt (Box): Ground-truth box.
        prediction (Box): Matched prediction.
        image_size (int): Render domain side length, used for the pixel distance.

    Returns:
        BoxLoss: The box IoU, the mean absolute difference of the four box
            parameters and the centre distance in pixels.
    """
    iou: float = intersection_over_union(gt, prediction)
    l1: float = (
        abs(gt.cx - prediction.cx)
        + abs(gt.cy - prediction.cy)
        + abs(gt.w - prediction.w)
        + abs(gt.h - prediction.h)
    ) / 4.0
    gt_x1, gt_y1, gt_x2, gt_y2 = gt.to_pixels(image_size)
    pd_x1, pd_y1, pd_x2, pd_y2 = prediction.to_pixels(image_size)
    gt_cx: float = (gt_x1 + gt_x2) / 2.0
    gt_cy: float = (gt_y1 + gt_y2) / 2.0
    pd_cx: float = (pd_x1 + pd_x2) / 2.0
    pd_cy: float = (pd_y1 + pd_y2) / 2.0
    center_distance: float = ((gt_cx - pd_cx) ** 2 + (gt_cy - pd_cy) ** 2) ** 0.5
    return BoxLoss(iou=iou, l1_norm=l1, center_distance=center_distance)


def match_boxes(
    ground_truth: list[Box],
    predictions: list[PredictedBox],
    iou_threshold: float,
    image_size: int,
) -> MatchOutcome:
    """Pair predictions with ground-truth boxes greedily, by confidence.

    Args:
        ground_truth (list[Box]): Ground-truth boxes of the sample.
        predictions (list[PredictedBox]): Predicted boxes for the component.
        iou_threshold (float): Minimum IoU for a prediction to claim a box.
        image_size (int): Render domain side length for the pixel distance.

    Returns:
        MatchOutcome: One :class:`Match` per ground-truth box (``None``
            prediction means a miss) plus every prediction that matched nothing.
    """
    ordered: list[PredictedBox] = sorted(
        predictions, key=lambda item: item.confidence, reverse=True
    )
    matched: list[int | None] = [None] * len(ground_truth)
    false_positives: list[PredictedBox] = []
    # Confidence order, not box order: a greedy assignment in confidence order
    # is the pairing a precision/recall curve would make at this threshold.
    for prediction_index, prediction in enumerate(ordered):
        best_index: int | None = None
        best_iou: float = iou_threshold
        for gt_index, gt_box in enumerate(ground_truth):
            if matched[gt_index] is not None:
                continue  # one prediction per ground-truth box
            iou: float = intersection_over_union(gt_box, prediction.box)
            if iou >= best_iou:
                best_iou = iou
                best_index = gt_index
        if best_index is None:
            false_positives.append(prediction)
        else:
            matched[best_index] = prediction_index
    matches: list[Match] = []
    for gt_index, prediction_index in enumerate(matched):
        if prediction_index is None:
            matches.append(Match(gt_index=gt_index, prediction=None, loss=None))
            continue
        prediction: PredictedBox = ordered[prediction_index]
        matches.append(
            Match(
                gt_index=gt_index,
                prediction=prediction,
                loss=box_loss(ground_truth[gt_index], prediction.box, image_size),
            )
        )
    return MatchOutcome(matches=matches, false_positives=false_positives)


def pixel_boxes_to_predictions(
    pixel_boxes: list[tuple[float, float, float, float, int, float]],
    class_id: int,
    image_size: int,
) -> list[PredictedBox]:
    """Keep only the component's detections and convert them to normalized boxes.

    Args:
        pixel_boxes (list[tuple[float, float, float, float, int, float]]):
            ``(x1, y1, x2, y2, class_id, confidence)`` detections in absolute
            pixels, as the detector returns them.
        class_id (int): Class index of the missing component; every other
            detection is dropped, which is what "evaluate only that component"
            means at the box level.
        image_size (int): Render domain side length the boxes are expressed in.

    Returns:
        list[PredictedBox]: The component's detections as normalized boxes.
    """
    predictions: list[PredictedBox] = []
    for x1, y1, x2, y2, det_class, confidence in pixel_boxes:
        if int(det_class) != int(class_id):
            continue
        cx: float = (x1 + x2) / 2.0 / image_size
        cy: float = (y1 + y2) / 2.0 / image_size
        width: float = (x2 - x1) / image_size
        height: float = (y2 - y1) / image_size
        predictions.append(
            PredictedBox(
                box=Box(cx=cx, cy=cy, w=width, h=height),
                confidence=float(confidence),
            )
        )
    return predictions


def summarize(values: list[float]) -> dict[str, float]:
    """Summarize a distribution for the report.

    Args:
        values (list[float]): Measurements, e.g. one IoU per matched box.

    Returns:
        dict[str, float]: ``mean``, ``median``, ``p10``, ``p90``, ``min`` and
            ``max``; all zero when there are no values, so a run with no matches
            reports zeros instead of crashing the summary.
    """
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    ordered: list[float] = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p10": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "min": ordered[0],
        "max": ordered[-1],
    }
