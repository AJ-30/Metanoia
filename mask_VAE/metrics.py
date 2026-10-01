"""
Per-class overlap metrics for mask_VAE telemetry.

A binary cross-entropy value is a poor progress signal for a multi-class mask
model: it keeps falling while the model simply predicts background everywhere,
because background is most of the image. These metrics say what actually
matters, namely how well each individual class is reconstructed, and they are
what the per-class telemetry CSV is built from.

Intersection over union and the Dice coefficient are both derived from a class
confusion matrix, which is accumulated with a single ``bincount`` over the
paired predicted and true class ids. That avoids materializing a one-hot tensor
of the same size as the input, which at 21 channels would be the largest
allocation in the evaluation loop. (At 28 channels for the combined
groups the one-hot fallback is about 3.3× larger, so the bincount approach
still matters.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mask_VAE.masking import MaskLayout

# Value written for a class that is absent from both the prediction and the
# target of an evaluation batch, and therefore has no overlap to measure. It is
# a real number so the CSV stays parseable, and it is skipped by the foreground
# averages so a class that simply does not occur cannot inflate the score.
ABSENT_CLASS_IOU: float = 0.0
ABSENT_CLASS_DICE: float = 0.0


@dataclass(frozen=True)
class ClassOverlap:
    """
    Per-class overlap of one evaluated batch.

    Both arrays are indexed by global channel, i.e. the same indexing as
    :attr:`MaskLayout.class_names`, and both hold :data:`ABSENT_CLASS_IOU` /
    :data:`ABSENT_CLASS_DICE` where the class is absent from the batch.

    Attributes:
        iou (torch.Tensor): ``(num_channels,)`` intersection over union, ``0..1``.
        dice (torch.Tensor): ``(num_channels,)`` Dice coefficient, ``0..1``.
    """

    iou: torch.Tensor
    dice: torch.Tensor


def confusion_matrix(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_channels: int,
) -> torch.Tensor:
    """
    Count predicted-versus-true class pairs, for later overlap measurement.

    Classes are decided by ``argmax`` over channels. For the targets, which are
    exact one-vs-rest binaries, that is the true class; for the logits it is the
    model's decision, which is exactly what gets rendered on the dashboard, so
    the metric and the picture can never disagree.

    Counting is done with a single ``bincount`` over the paired class ids, which
    avoids materializing a one-hot tensor of the same size as the input; at 21
    channels that would be the largest allocation of the evaluation loop.

    Args:
        logits (torch.Tensor): ``(B, num_channels, H, W)`` decoder logits.
        targets (torch.Tensor): ``(B, num_channels, H, W)`` binary one-vs-rest
            target masks, valued ``0.0`` or ``1.0``.
        num_channels (int): Number of class channels, background included.

    Returns:
        torch.Tensor: ``(num_channels, num_channels)`` matrix whose entry
            ``[p, t]`` counts the pixels predicted ``p`` whose truth is ``t``.
            Summing this over batches and measuring the result once gives the
            overlap of the whole evaluation set, rather than an average of
            per-batch overlaps.

    Raises:
        ValueError: If the two tensors disagree on shape, or on channel count.
    """
    if logits.shape != targets.shape:
        raise ValueError(
            f"Logits shape {tuple(logits.shape)} and target shape "
            f"{tuple(targets.shape)} must match."
        )
    if logits.shape[1] != num_channels:
        raise ValueError(
            f"Expected {num_channels} class channels, got {logits.shape[1]}."
        )

    predicted: torch.Tensor = logits.argmax(dim=1).reshape(-1)
    true: torch.Tensor = targets.argmax(dim=1).reshape(-1)
    pair_index: torch.Tensor = predicted * num_channels + true
    return torch.bincount(
        pair_index, minlength=num_channels * num_channels
    ).reshape(num_channels, num_channels)


def overlap_from_confusion(confusion: torch.Tensor) -> ClassOverlap:
    """
    Turn a class confusion matrix into per-class IoU and Dice.

    Args:
        confusion (torch.Tensor): ``(num_channels, num_channels)`` matrix as
            returned by :func:`confusion_matrix`, possibly summed over batches.

    Returns:
        ClassOverlap: Per-class IoU and Dice, indexed by global channel, with
            :data:`ABSENT_CLASS_IOU` / :data:`ABSENT_CLASS_DICE` for classes that
            do not occur.

    Raises:
        ValueError: If the matrix is not square.
    """
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError(
            f"The confusion matrix must be square, got shape {tuple(confusion.shape)}."
        )

    intersection: torch.Tensor = confusion.diagonal()
    predicted_counts: torch.Tensor = confusion.sum(dim=1)
    true_counts: torch.Tensor = confusion.sum(dim=0)
    union: torch.Tensor = predicted_counts + true_counts - intersection
    dice_denominator: torch.Tensor = predicted_counts + true_counts

    return ClassOverlap(
        iou=_safe_ratio(intersection, union, ABSENT_CLASS_IOU),
        dice=_safe_ratio(2.0 * intersection, dice_denominator, ABSENT_CLASS_DICE),
    )


def class_overlap(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_channels: int,
) -> ClassOverlap:
    """
    Measure per-class overlap between predicted and true one-vs-rest masks.

    A convenience wrapper around :func:`confusion_matrix` and
    :func:`overlap_from_confusion` for callers that only need one batch.

    Args:
        logits (torch.Tensor): ``(B, num_channels, H, W)`` decoder logits.
        targets (torch.Tensor): ``(B, num_channels, H, W)`` binary one-vs-rest
            target masks, valued ``0.0`` or ``1.0``.
        num_channels (int): Number of class channels, background included.

    Returns:
        ClassOverlap: Per-class IoU and Dice, indexed by global channel.
    """
    return overlap_from_confusion(confusion_matrix(logits, targets, num_channels))


def _safe_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    absent_value: float,
) -> torch.Tensor:
    """
    Divide two per-class tensors, substituting a fixed value where the
    denominator is zero.

    Args:
        numerator (torch.Tensor): Per-class numerator.
        denominator (torch.Tensor): Per-class denominator.
        absent_value (float): Value used where the denominator is zero.

    Returns:
        torch.Tensor: Per-class ratio, ``0`` wherever the denominator was zero.
    """
    is_absent: torch.Tensor = denominator <= 0
    safe_denominator: torch.Tensor = torch.where(
        is_absent, torch.ones_like(denominator), denominator
    )
    ratio: torch.Tensor = numerator / safe_denominator
    return torch.where(is_absent, torch.full_like(ratio, absent_value), ratio)


def foreground_mean(values: torch.Tensor, layout: MaskLayout) -> float:
    """
    Average a per-class metric over the foreground classes only.

    The background channel is excluded: it is the easiest class to get right and
    it dominates the pixel count, so including it would hide whether the floor
    plan and furniture classes are actually being learned.

    Args:
        values (torch.Tensor): ``(num_channels,)`` per-class metric.
        layout (MaskLayout): Layout that defines which channels are foreground.

    Returns:
        float: Mean over the foreground channels, or ``0.0`` when the layout has
            no foreground channel at all.

    Raises:
        ValueError: If the metric length does not match the layout.
    """
    if values.shape[0] != layout.num_channels:
        raise ValueError(
            f"Expected {layout.num_channels} per-class values, got {values.shape[0]}."
        )
    if not layout.foreground_channels:
        return 0.0
    foreground: torch.Tensor = values[list(layout.foreground_channels)]
    return float(foreground.mean().item())
