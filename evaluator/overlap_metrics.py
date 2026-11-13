"""Collision metric: how much of a predicted component box covers other furniture.

The detector's placement quality has two parts. The existing box loss asks
*whether the prediction is where the component's own ground truth is*. This
module answers the complementary question: *does the predicted component sit on
top of furniture the dataset says is already there?*

The metric is a normalized fraction, not an area, so a large TV and a small
showcase are comparable across samples:

    overlap = area(final target box ∩ union(other ground-truth boxes))
              ---------------------------------------------------------
                         area(final target box)

``0.0`` means the final predicted target box touches no other furniture and
``1.0`` means it is completely covered by other furniture. Lower is better.

Only the dataset's named label lines are used. The evaluator's labels carry one
name per object (``TV cx cy w h``), so "other furniture" is every named box whose
name is not one of the target component's label names. The final target box is
the detector's highest-confidence prediction for the target component, which is
the same "final box" a person reads from the result page.
"""

from __future__ import annotations

from typing import Sequence

from evaluator.box_metrics import PredictedBox
from evaluator.dataset_gt import Box

# Axis-aligned rectangle bounds in normalized ``0..1`` image coordinates.
Bounds = tuple[float, float, float, float]


def _bounds(box: Box) -> Bounds:
    """Convert a normalized centre/size box to ``x1, y1, x2, y2`` bounds.

    Args:
        box (Box): Box whose ``cx``, ``cy``, ``w`` and ``h`` are normalized to
            the render domain.

    Returns:
        Bounds: Left, top, right and bottom edges in the same normalized space.
    """
    half_width: float = box.w / 2.0
    half_height: float = box.h / 2.0
    return (
        box.cx - half_width,
        box.cy - half_height,
        box.cx + half_width,
        box.cy + half_height,
    )


def _intersection_bounds(first: Bounds, second: Bounds) -> Bounds | None:
    """Return the overlap of two rectangles, or ``None`` when they miss.

    Args:
        first (Bounds): First rectangle as ``x1, y1, x2, y2``.
        second (Bounds): Second rectangle in the same form.

    Returns:
        Bounds | None: The overlapping rectangle, or ``None`` when either side
            has zero or negative area.
    """
    x1: float = max(first[0], second[0])
    y1: float = max(first[1], second[1])
    x2: float = min(first[2], second[2])
    y2: float = min(first[3], second[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def intersection_area(first: Box, second: Box) -> float:
    """Return the area shared by two normalized boxes.

    Args:
        first (Box): First box.
        second (Box): Second box.

    Returns:
        float: Intersection area in normalized squared units; ``0.0`` when the
            boxes do not overlap.
    """
    overlap: Bounds | None = _intersection_bounds(_bounds(first), _bounds(second))
    if overlap is None:
        return 0.0
    return (overlap[2] - overlap[0]) * (overlap[3] - overlap[1])


def _merged_y_length(intervals: Sequence[tuple[float, float]]) -> float:
    """Return the total length covered by a set of y intervals.

    Args:
        intervals (Sequence[tuple[float, float]]): ``(y1, y2)`` pairs, possibly
            overlapping.

    Returns:
        float: Union length of the intervals.
    """
    if not intervals:
        return 0.0
    ordered: list[tuple[float, float]] = sorted(intervals)
    total: float = 0.0
    current_start: float
    current_end: float
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    total += current_end - current_start
    return total


def _clipped_rectangles(target: Box, others: Sequence[Box]) -> list[Bounds]:
    """Clip every other box to the target box and drop empty intersections.

    Args:
        target (Box): Box the overlap is measured against.
        others (Sequence[Box]): Other furniture boxes.

    Returns:
        list[Bounds]: Non-empty intersection rectangles inside ``target``.
    """
    target_bounds: Bounds = _bounds(target)
    rectangles: list[Bounds] = []
    for other in others:
        clipped: Bounds | None = _intersection_bounds(target_bounds, _bounds(other))
        if clipped is not None:
            rectangles.append(clipped)
    return rectangles


def covered_area(target: Box, others: Sequence[Box]) -> float:
    """Return the area of ``target`` covered by the union of ``others``.

    The union is computed exactly with an x-sweep. Summing per-box intersection
    areas would double-count the region where two other furniture boxes overlap,
    which would make the metric exceed the target's own area.

    Args:
        target (Box): Box the overlap is measured against.
        others (Sequence[Box]): Other furniture boxes.

    Returns:
        float: Covered area in normalized squared units, never larger than the
            target's area.
    """
    rectangles: list[Bounds] = _clipped_rectangles(target, others)
    if not rectangles:
        return 0.0
    x_edges: list[float] = sorted(
        {rectangle[0] for rectangle in rectangles}
        | {rectangle[2] for rectangle in rectangles}
    )
    total: float = 0.0
    for left, right in zip(x_edges, x_edges[1:]):
        width: float = right - left
        if width <= 0.0:
            continue
        # A rectangle covers this x-slab only when its own left edge is at or
        # before the slab's left edge and its right edge is at or after the
        # slab's right edge. The slab boundaries are rectangle edges, so this
        # comparison is exact.
        intervals: list[tuple[float, float]] = [
            (rectangle[1], rectangle[3])
            for rectangle in rectangles
            if rectangle[0] <= left and rectangle[2] >= right
        ]
        total += width * _merged_y_length(intervals)
    return min(total, target.w * target.h)


def overlap_fraction(target: Box, others: Sequence[Box]) -> float:
    """Return the fraction of ``target`` covered by other furniture.

    Args:
        target (Box): The final predicted component box.
        others (Sequence[Box]): Ground-truth boxes of the other furniture in the
            same layout.

    Returns:
        float: Covered fraction in ``0..1``; ``0.0`` when ``others`` is empty.

    Raises:
        ValueError: If ``target`` has zero or negative area, which would make
            the fraction undefined.
    """
    target_area: float = target.w * target.h
    if target_area <= 0.0:
        raise ValueError(
            "Cannot measure overlap for a target box with zero or negative area."
        )
    if not others:
        return 0.0
    return min(1.0, covered_area(target, others) / target_area)


def highest_confidence_prediction(
    predictions: Sequence[PredictedBox],
) -> PredictedBox | None:
    """Return the detector's final target box.

    Args:
        predictions (Sequence[PredictedBox]): Target-component predictions,
            already filtered to the page's confidence threshold.

    Returns:
        PredictedBox | None: The prediction with the highest confidence, or
            ``None`` when the detector returned no target box. ``max`` keeps the
            first prediction on a confidence tie, so the choice is deterministic.
    """
    if not predictions:
        return None
    return max(predictions, key=lambda prediction: prediction.confidence)


def overlap_for_final_prediction(
    predictions: Sequence[PredictedBox],
    other_ground_truth: Sequence[Box],
) -> float | None:
    """Measure the final target prediction against the other furniture boxes.

    Args:
        predictions (Sequence[PredictedBox]): Target-component predictions.
        other_ground_truth (Sequence[Box]): Ground-truth boxes of the other
            furniture in the same layout.

    Returns:
        float | None: The overlap fraction of the highest-confidence prediction,
            or ``None`` when the detector predicted no target box. A prediction
            with no other furniture to collide with scores ``0.0``.
    """
    final_prediction: PredictedBox | None = highest_confidence_prediction(predictions)
    if final_prediction is None:
        return None
    return overlap_fraction(final_prediction.box, other_ground_truth)
