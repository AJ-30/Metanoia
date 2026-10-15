"""Channel-level Dice/IoU of the generated component, in mask space.

The detector loss answers "where does the component end up as a detector sees
it"; this module answers the cheaper, model-side question: "how much of the
true component did the generation cover". Both are reported because a component
can be pixel-plausible and still land where the detector never looks, and the
reverse.

Scores are computed on the decoded argmax channel against the ground-truth mask
channel of the same normalized ``256 x 256`` domain, so the two masks are
compared in exactly the space the decoder produced.
"""

from __future__ import annotations

import numpy as np
import torch


def binary_masks(
    generated: torch.Tensor, ground_truth_channel: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten both masks to comparable boolean arrays.

    Args:
        generated (torch.Tensor): Generated component mask, any shape.
        ground_truth_channel (torch.Tensor): Ground-truth component mask of the
            same shape.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(generated, ground_truth)`` boolean
            arrays flattened to one dimension.
    """
    return (
        np.asarray(generated.detach().cpu().numpy()).astype(bool).reshape(-1),
        np.asarray(ground_truth_channel.detach().cpu().numpy())
        .astype(bool)
        .reshape(-1),
    )


def dice(generated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Dice coefficient of two boolean masks.

    Args:
        generated (np.ndarray): Boolean generated mask.
        ground_truth (np.ndarray): Boolean ground-truth mask of the same shape.

    Returns:
        float: ``2|A n B| / (|A| + |B|)``; ``1.0`` when both masks are empty
            (a sample where neither mask claims any pixel agrees perfectly), and
            ``0.0`` when they disagree about everything.
    """
    if not generated.any() and not ground_truth.any():
        return 1.0
    intersection: int = int(np.logical_and(generated, ground_truth).sum())
    denominator: int = int(generated.sum()) + int(ground_truth.sum())
    if denominator == 0:
        return 0.0
    return 2.0 * intersection / denominator


def iou(generated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Intersection over union of two boolean masks.

    Args:
        generated (np.ndarray): Boolean generated mask.
        ground_truth (np.ndarray): Boolean ground-truth mask of the same shape.

    Returns:
        float: ``|A n B| / |A u B|``; ``1.0`` when both masks are empty and
            ``0.0`` when they share no pixel at all.
    """
    if not generated.any() and not ground_truth.any():
        return 1.0
    intersection: int = int(np.logical_and(generated, ground_truth).sum())
    union: int = int(np.logical_or(generated, ground_truth).sum())
    if union == 0:
        return 0.0
    return intersection / union


def channel_scores(
    generated: torch.Tensor, ground_truth_channel: torch.Tensor
) -> tuple[float, float]:
    """Dice and IoU of one generated channel against the ground-truth channel.

    Args:
        generated (torch.Tensor): Generated component mask for one sample.
        ground_truth_channel (torch.Tensor): Ground-truth component mask of the
            same spatial shape.

    Returns:
        tuple[float, float]: ``(dice, iou)``.
    """
    generated_mask, truth_mask = binary_masks(generated, ground_truth_channel)
    return dice(generated_mask, truth_mask), iou(generated_mask, truth_mask)
