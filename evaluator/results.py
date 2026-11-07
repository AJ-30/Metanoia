"""Accumulators for the evaluator: per-draw aggregates and per-sample records.

The pipeline streams sample blocks, so the running totals cannot be derived from
a stored list of rows at the end: they are accumulated as each (sample, result)
is measured. Two shapes are kept:

* :class:`DrawAccumulator` - one per result (each draw and the ensemble); it
  collects the loss distributions, the outcome counts and the channel scores,
  and turns them into the summary dictionary the report is built from.
* :class:`SampleRecord` - one per sample; it remembers where the evaluated
  images were written and what the detector predicted, which is everything the
  visualization needs after the loop has moved on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluator.box_metrics import BoxLoss, MatchOutcome, PredictedBox, summarize
from evaluator.dataset_gt import Box
from evaluator.yolo_eval import PixelBox

# Result key of the ensemble. Per-draw results use their draw index (0..N-1), so
# -1 cannot collide with one and the existing dict/list shapes can hold both the
# diagnostics and the primary ensemble result.
ENSEMBLE_RESULT: int = -1


def result_slug(result_key: int) -> str:
    """Readable filename suffix of one evaluated result.

    Args:
        result_key (int): Draw index, or :data:`ENSEMBLE_RESULT`.

    Returns:
        str: ``"draw_<d>"`` for a draw, ``"ensemble"`` for the ensemble.
    """
    return "ensemble" if result_key == ENSEMBLE_RESULT else f"draw_{result_key}"


@dataclass
class DrawAccumulator:
    """Running totals of one evaluated result (a draw or the ensemble).

    Attributes:
        box_ious (list[float]): IoU of every matched box.
        l1_norms (list[float]): Mean absolute box-parameter error of every
            matched box.
        center_distances (list[float]): Centre distance in pixels of every
            matched box.
        channel_dice (list[float]): Mask Dice of every sample.
        channel_iou (list[float]): Mask IoU of every sample.
        ground_truth_boxes (int): Ground-truth boxes seen.
        matched (int): Ground-truth boxes a prediction reached.
        missed (int): Ground-truth boxes nothing reached.
        false_positives (int): Predictions that matched no ground truth.
    """

    box_ious: list[float] = field(default_factory=list)
    l1_norms: list[float] = field(default_factory=list)
    center_distances: list[float] = field(default_factory=list)
    channel_dice: list[float] = field(default_factory=list)
    channel_iou: list[float] = field(default_factory=list)
    ground_truth_boxes: int = 0
    matched: int = 0
    missed: int = 0
    false_positives: int = 0

    def add(
        self, ground_truth: list[Box], outcome: MatchOutcome, channel_scores: tuple[float, float]
    ) -> None:
        """Fold one sample's measurement into the running totals.

        Args:
            ground_truth (list[Box]): Ground-truth boxes of the sample.
            outcome (MatchOutcome): The sample's matched/missed/false-positive
                outcome for this draw.
            channel_scores (tuple[float, float]): ``(dice, iou)`` of the
                generated channel for this sample and draw.
        """
        self.ground_truth_boxes += len(ground_truth)
        self.missed += sum(1 for match in outcome.matches if match.loss is None)
        for match in outcome.matches:
            if match.loss is None:
                continue
            self.matched += 1
            self.box_ious.append(match.loss.iou)
            self.l1_norms.append(match.loss.l1_norm)
            self.center_distances.append(match.loss.center_distance)
        self.false_positives += len(outcome.false_positives)
        self.channel_dice.append(channel_scores[0])
        self.channel_iou.append(channel_scores[1])

    def summary(self, detection: dict[str, float] | None) -> dict[str, Any]:
        """Turn the running totals into the report dictionary.

        Args:
            detection (dict[str, float] | None): Detector metrics for this
                draw, or ``None`` when detector validation did not run.

        Returns:
            dict[str, Any]: Detector metrics, box-loss distributions, channel
                score distributions and the outcome counts.
        """
        return {
            "detector_metrics": detection,
            "box_loss": {
                "box_iou": summarize(self.box_ious),
                "l1_norm_xywh": summarize(self.l1_norms),
                "center_distance_px": summarize(self.center_distances),
            },
            "channel_scores": {
                "dice": summarize(self.channel_dice),
                "iou": summarize(self.channel_iou),
            },
            "outcomes": {
                "ground_truth_boxes": self.ground_truth_boxes,
                "matched": self.matched,
                "missed": self.missed,
                "false_positives": self.false_positives,
                "prediction_free": self.matched + self.false_positives,
            },
        }


def average_statistics(
    per_draw: list[dict[str, Any]], section: str
) -> dict[str, dict[str, float]]:
    """Average one section's statistic dictionaries across the draws.

    Args:
        per_draw (list[dict[str, Any]]): The per-draw summaries.
        section (str): Section key to average, e.g. ``box_loss``.

    Returns:
        dict[str, dict[str, float]]: ``{distribution: {statistic: mean}}``.
    """
    totals: dict[str, dict[str, float]] = {}
    for draw in per_draw:
        for name, statistics in draw[section].items():
            bucket: dict[str, float] = totals.setdefault(name, {})
            for statistic, value in statistics.items():
                bucket[statistic] = bucket.get(statistic, 0.0) + float(value)
    count: int = len(per_draw)
    return {
        name: {statistic: value / count for statistic, value in bucket.items()}
        for name, bucket in totals.items()
    }


def aggregate_per_draw(per_draw: list[dict[str, Any]]) -> dict[str, Any]:
    """Average the per-draw distributions into one across-draw report.

    The counts are summed (they are totals, not distributions) while every
    distribution is averaged over the draws that produced it, so a four-draw run
    reads as one report with the spread preserved per draw.

    Args:
        per_draw (list[dict[str, Any]]): The per-draw summaries.

    Returns:
        dict[str, Any]: Across-draw means and summed counts; empty when nothing
            was evaluated.
    """
    if not per_draw:
        return {}
    totals: dict[str, int] = {
        key: sum(int(draw["outcomes"][key]) for draw in per_draw)
        for key in per_draw[0]["outcomes"]
    }
    scored: list[dict[str, dict[str, float]]] = [
        draw["detector_metrics"] for draw in per_draw if draw["detector_metrics"] is not None
    ]
    detector_means: dict[str, float] = (
        {
            key: sum(float(metrics[key]) for metrics in scored) / len(scored)
            for key in scored[0]
        }
        if scored
        else {}
    )
    return {
        "box_loss": average_statistics(per_draw, "box_loss"),
        "channel_scores": average_statistics(per_draw, "channel_scores"),
        "detector_metrics": detector_means or None,
        "outcomes": totals,
    }


@dataclass
class SampleRecord:
    """Everything remembered about one evaluated sample.

    Attributes:
        stem (str): Sample folder name, also the image file stem.
        ground_truth (list[Box]): Dataset ground-truth boxes of the component.
        image_paths (dict[int, Path]): Evaluated composite per result key; the
            ensemble uses :data:`ENSEMBLE_RESULT` (``-1``).
        predictions (dict[int, list[PredictedBox]]): Detector predictions for
            the component, per result key.
        losses (dict[int, list[BoxLoss]]): Localization losses of matched
            boxes, per result key.
    """

    stem: str
    ground_truth: list[Box] = field(default_factory=list)
    image_paths: dict[int, Path] = field(default_factory=dict)
    predictions: dict[int, list[PixelBox]] = field(default_factory=dict)
    losses: dict[int, list[BoxLoss]] = field(default_factory=dict)
