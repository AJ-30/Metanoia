"""Per-case metrics of a predicted missing part.

For one case, everything independent of the prediction is computed once
(``prepare_case``); each predicted sample is then scored with
``score_prediction``. With K samples per case, ``score_samples`` reports the
first sample (what a deployed system outputs) and the best of K against the
ground truth (how often *some* sample is right; relevant because several
answers can be valid).

Metric definitions (per case):

* detection: predicted vs ground-truth *added* components, matched one-to-one
  within the same class; ``tp@t`` = matches with IoU >= t.
* primary: the ground-truth component of the removed instance.
  ``primary_iou`` = IoU of its matched prediction (0 if unmatched),
  ``primary_detected`` = primary_iou >= first IoU threshold.
* ``loc_error_m``: centroid distance (metres) between the primary component
  and its matched prediction, or the nearest prediction of the same class.
* ``type_correct``: on the pixels of the primary component, the majority of
  predicted (added) labels in its layer have the primary's type. ("Largest
  predicted component" would be wrong for margin cases, where neighbouring
  rooms are also missing and usually bigger than the removed door.)
* ``connection_correct`` (doors/windows only): the predicted opening links
  the same rooms (and exterior) as the true one, even if shifted along the wall.
* ``pixel_iou``: class-aware IoU of all added pixels.
* validity (composed plan): shell pieces minus ground truth, and the fractions
  of openings on walls, enclosed rooms and reachable rooms.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from Reconstruction.data_prep.missing import Case
from Reconstruction.data_prep.schema import STRUCTURE_LAYER, Schema
from Reconstruction.evaluation.components import Component, extract_components, iou_matrix, match
from Reconstruction.evaluation.labels import EMPTY, LabelMaps, added, compose, semantic_to_labels
from Reconstruction.evaluation.structure_checks import PlanStructure

_OPENINGS: tuple[str, str] = ("door", "window")
# Case metadata copied into every result row (report tables group by these).
RECORD_FIELDS: tuple[str, ...] = ("case_id", "plan_id", "comp_type", "primary_cls", "mode", "margin_px",
                                  "size_bin", "location", "num_neighbours", "metres_per_px")
# Fields compared across samples for the best-of-K report.
BEST_OF_K_FIELDS: tuple[str, ...] = ("primary_iou", "primary_detected", "loc_error_m", "connection_correct")


@dataclass
class CaseContext:
    """Prediction-independent data of one case.

    Attributes:
        base (LabelMaps): Incomplete input plan.
        truth (list[Component]): Ground-truth added components.
        truth_added (LabelMaps): Ground-truth added labels.
        primary (int): Index of the primary component in ``truth`` (-1 if none).
        reference (PlanStructure): Structure of the complete plan.
        reference_shell (int): Shell pieces of the complete plan.
        truth_connection (frozenset[int] | None): Rooms/exterior linked by the
            removed opening (None unless the primary is a door/window).
        metres_per_px (float): Scale (NaN if unknown).
        fields (dict[str, Any]): Per-case descriptive fields (clue, references).
    """

    base: LabelMaps
    truth: list[Component]
    truth_added: LabelMaps
    primary: int
    reference: PlanStructure
    reference_shell: int
    truth_connection: frozenset[int] | None
    metres_per_px: float
    fields: dict[str, Any]


def _ratio(pair: tuple[int, int]) -> float:
    """Fraction ok / total, NaN when there is nothing to check.

    Args:
        pair (tuple[int, int]): (ok, total).

    Returns:
        float: Ratio.
    """
    return pair[0] / pair[1] if pair[1] else math.nan


def prepare_case(case: Case, schema: Schema, params: SimpleNamespace) -> CaseContext:
    """Compute everything about a case that does not depend on a prediction.

    Args:
        case (Case): Generated case.
        schema (Schema): Taxonomy.
        params (SimpleNamespace): Evaluation parameters.

    Returns:
        CaseContext: Case context.
    """
    base: LabelMaps = semantic_to_labels(case.incomplete, schema, 0.5)
    full: LabelMaps = semantic_to_labels(case.full, schema, 0.5)
    truth_added: LabelMaps = added(base, full)
    truth: list[Component] = extract_components(truth_added, schema, params.components.MIN_AREA_PX)
    primary_pixels: np.ndarray = np.flatnonzero(case.primary.ravel())
    overlaps: list[int] = [np.intersect1d(c.pixels, primary_pixels, assume_unique=True).size for c in truth]
    primary: int = int(np.argmax(overlaps)) if overlaps and max(overlaps) > 0 else -1
    reference = PlanStructure(full, schema, params)
    primary_cls: str | None = case.info.get("primary_cls")
    connection: frozenset[int] | None = None
    fields: dict[str, Any] = {"n_truth_components": len(truth)}
    if primary >= 0 and primary_cls in _OPENINGS:
        connection = reference.connection(truth[primary].mask(base.structure.shape))
        # Clue: the input has more rooms lacking this opening type than the complete plan.
        fields["clue"] = PlanStructure(base, schema, params).rooms_without(primary_cls) > reference.rooms_without(primary_cls)
    scale: float | None = case.info.get("metres_per_px")
    return CaseContext(base, truth, truth_added, primary, reference, reference.shell_pieces(), connection,
                       math.nan if scale is None else float(scale), fields)


def _primary_prediction(ctx: CaseContext, predicted: list[Component], pairs: list[tuple[int, int, float]]) -> tuple[int, float]:
    """Prediction associated with the primary component and its IoU.

    Matched prediction if any, else the nearest same-class prediction (IoU 0).

    Args:
        ctx (CaseContext): Case context.
        predicted (list[Component]): Predicted components.
        pairs (list[tuple[int, int, float]]): Matching result.

    Returns:
        tuple[int, float]: (prediction index or -1, IoU).
    """
    for pred_index, truth_index, iou in pairs:
        if truth_index == ctx.primary:
            return pred_index, iou
    target: Component = ctx.truth[ctx.primary]
    same = [(np.hypot(p.centroid[0] - target.centroid[0], p.centroid[1] - target.centroid[1]), i)
            for i, p in enumerate(predicted) if p.cls == target.cls]
    return (min(same)[1], 0.0) if same else (-1, 0.0)


def _type_at_primary(ctx: CaseContext, pred_added: LabelMaps, schema: Schema) -> str:
    """Component type most often predicted on the primary component's pixels.

    Args:
        ctx (CaseContext): Case context (``primary`` must be >= 0).
        pred_added (LabelMaps): Predicted added labels.
        schema (Schema): Taxonomy.

    Returns:
        str: Component type, or "none" if nothing was predicted there.
    """
    target: Component = ctx.truth[ctx.primary]
    in_structure: bool = schema.layer_of(target.cls) == STRUCTURE_LAYER
    labels: np.ndarray = (pred_added.structure if in_structure else pred_added.room).ravel()[target.pixels]
    labels = labels[labels != EMPTY]
    if labels.size == 0:
        return "none"
    classes: tuple[str, ...] = schema.structure_classes if in_structure else schema.room_classes
    return schema.component_type[classes[int(np.bincount(labels).argmax())]]


def _pixel_iou(truth: LabelMaps, pred: LabelMaps) -> float:
    """Class-aware IoU of added pixels over both layers.

    Args:
        truth (LabelMaps): Ground-truth added labels.
        pred (LabelMaps): Predicted added labels.

    Returns:
        float: IoU (NaN if both are empty).
    """
    inter: int = 0
    union: int = 0
    for t, p in ((truth.structure, pred.structure), (truth.room, pred.room)):
        inter += int(((t == p) & (t != EMPTY)).sum())
        union += int(((t != EMPTY) | (p != EMPTY)).sum())
    return inter / union if union else math.nan


def score_prediction(ctx: CaseContext, prediction: np.ndarray, schema: Schema, params: SimpleNamespace) -> dict[str, Any]:
    """All metrics of one predicted sample.

    Args:
        ctx (CaseContext): Case context.
        prediction (np.ndarray): (C, H, W) predicted missing part, scores in [0, 1].
        schema (Schema): Taxonomy.
        params (SimpleNamespace): Evaluation parameters.

    Returns:
        dict[str, Any]: Metric name -> value.
    """
    composed: LabelMaps = compose(ctx.base, semantic_to_labels(prediction, schema, params.decode.THRESHOLD))
    pred_added: LabelMaps = added(ctx.base, composed)
    predicted: list[Component] = extract_components(pred_added, schema, params.components.MIN_AREA_PX)
    pairs = match(iou_matrix(predicted, ctx.truth))
    thresholds: list[float] = params.components.IOU_THRESHOLDS
    out: dict[str, Any] = {"n_pred": len(predicted), "n_truth": len(ctx.truth)}
    out.update({f"tp@{t}": sum(iou >= t for _, _, iou in pairs) for t in thresholds})

    if ctx.primary >= 0:
        pred_index, iou = _primary_prediction(ctx, predicted, pairs)
        target: Component = ctx.truth[ctx.primary]
        out["primary_iou"] = iou
        out["primary_detected"] = iou >= thresholds[0]
        out["predicted_type"] = _type_at_primary(ctx, pred_added, schema)
        out["type_correct"] = out["predicted_type"] == target.comp_type
        if pred_index >= 0:
            p: Component = predicted[pred_index]
            out["loc_error_m"] = float(np.hypot(p.centroid[0] - target.centroid[0], p.centroid[1] - target.centroid[1])) * ctx.metres_per_px
        else:
            out["loc_error_m"] = math.nan
        if ctx.truth_connection is not None:
            out["connection_correct"] = pred_index >= 0 and \
                ctx.reference.connection(predicted[pred_index].mask(composed.structure.shape)) == ctx.truth_connection

    out["pixel_iou"] = _pixel_iou(ctx.truth_added, pred_added)
    structure = PlanStructure(composed, schema, params)
    out["shell_pieces_diff"] = structure.shell_pieces() - ctx.reference_shell
    out["openings_on_wall"] = _ratio(structure.openings_on_wall())
    out["rooms_enclosed"] = _ratio(structure.rooms_enclosed())
    out["rooms_reachable"] = _ratio(structure.rooms_reachable())
    return out


def score_samples(ctx: CaseContext, samples: np.ndarray, schema: Schema, params: SimpleNamespace) -> dict[str, Any]:
    """Score K samples: all metrics of sample 0 plus best-of-K primary metrics.

    Args:
        ctx (CaseContext): Case context.
        samples (np.ndarray): (K, C, H, W) predictions.
        schema (Schema): Taxonomy.
        params (SimpleNamespace): Evaluation parameters.

    Returns:
        dict[str, Any]: Metrics of sample 0, ``bestk_*`` fields and ``k``.
    """
    scores: list[dict[str, Any]] = [score_prediction(ctx, sample, schema, params) for sample in samples]
    result: dict[str, Any] = {**ctx.fields, **scores[0], "k": len(scores)}
    if ctx.primary >= 0:
        # Best sample = highest primary IoU, ties broken by more true positives.
        key = f"tp@{params.components.IOU_THRESHOLDS[0]}"
        best: dict[str, Any] = max(scores, key=lambda s: (s["primary_iou"], s[key]))
        result.update({f"bestk_{name}": best[name] for name in BEST_OF_K_FIELDS if name in best})
    return result
