"""Predictors evaluated by ``evaluate.py``.

A predictor maps a case to K predicted missing parts, shape (K, C, H, W),
scores in [0, 1]. Non-oracle predictors only look at the model input
(``case.incomplete``) and the plan scale, exactly like a trained model.

* ``empty``  - predicts nothing (lower bound).
* ``oracle`` - returns the ground truth (sanity check: must score perfectly).
* ``rules``  - hand-written baseline:
    1. close gaps in the building shell with horizontal/vertical morphological
       closing; keep gaps about as thick as a wall that are at least partly
       unlabelled (corridors between walls are fully room-labelled);
    2. gaps up to ``DOOR_MAX_M`` long become windows if they face the
       exterior, otherwise doors; longer gaps become walls;
    3. unlabelled areas enclosed by walls become ``FILL_ROOM_CLASS``.
  It cannot handle hard cases (nothing looks missing), by design.
* ``FilePredictor`` - predictions saved by a model, one
  ``<case_id>.npz`` per case holding ``samples`` (K, C, H, W): float scores
  in [0, 1], or uint8 scores * 255 (as written by ``model/predict.py``).
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from Reconstruction.data_prep.missing import Case
from Reconstruction.data_prep.schema import Schema
from Reconstruction.evaluation.labels import EMPTY, LabelMaps, semantic_to_labels
from Reconstruction.evaluation.structure_checks import PlanStructure

# (case, schema, eval params) -> (K, C, H, W) predicted missing part.
PredictorFn = Callable[[Case, Schema, SimpleNamespace], np.ndarray]


def predict_empty(case: Case, schema: Schema, params: SimpleNamespace) -> np.ndarray:
    """Predict that nothing is missing.

    Args:
        case (Case): Case (only its shape is used).
        schema (Schema): Taxonomy (unused; uniform signature).
        params (SimpleNamespace): Evaluation parameters (unused).

    Returns:
        np.ndarray: float32 (1, C, H, W) zeros.
    """
    return np.zeros((1,) + case.incomplete.shape, np.float32)


def predict_oracle(case: Case, schema: Schema, params: SimpleNamespace) -> np.ndarray:
    """Return the ground-truth missing part (metric sanity check).

    Args:
        case (Case): Case.
        schema (Schema): Taxonomy (unused).
        params (SimpleNamespace): Evaluation parameters (unused).

    Returns:
        np.ndarray: float32 (1, C, H, W) target.
    """
    return case.target[None].astype(np.float32)


def _wall_thickness_px(labels: LabelMaps, schema: Schema) -> float:
    """Median wall thickness: twice the distance-to-edge along wall centre lines.

    Args:
        labels (LabelMaps): Plan labels.
        schema (Schema): Taxonomy.

    Returns:
        float: Thickness in pixels (at least 1).
    """
    wall: np.ndarray = labels.structure == schema.structure_classes.index("wall")
    if not wall.any():
        return 1.0
    depth: np.ndarray = ndimage.distance_transform_edt(wall)
    centre: np.ndarray = skeletonize(wall)
    return max(1.0, 2.0 * float(np.median(depth[centre])) if centre.any() else 1.0)


def _shell_gaps(shell: np.ndarray, free: np.ndarray, length_px: int, max_thickness_px: float,
                min_free_fraction: float) -> list[tuple[np.ndarray, int]]:
    """Wall-thick gaps in the shell closable by horizontal or vertical closing.

    Args:
        shell (np.ndarray): bool (H, W) existing structure.
        free (np.ndarray): bool (H, W) unlabelled pixels. Corridors between
            two walls are fully room-labelled; real wall gaps are partly free.
        length_px (int): Longest gap to close.
        max_thickness_px (float): Maximum gap thickness across the wall.
        min_free_fraction (float): Minimum share of free pixels in a gap.

    Returns:
        list[tuple[np.ndarray, int]]: (gap mask, gap length in px).
    """
    gaps: list[tuple[np.ndarray, int]] = []
    taken: np.ndarray = np.zeros_like(shell)
    # Growing a seed by about half a wall thickness recovers the room-labelled
    # rim of a real gap without swallowing the rooms closing also filled.
    grow: np.ndarray = np.ones((int(max_thickness_px) // 2 * 2 + 1,) * 2, bool)
    for axis, kernel in ((1, np.ones((1, length_px), bool)), (0, np.ones((length_px, 1), bool))):
        candidates: np.ndarray = ndimage.binary_closing(shell, structure=kernel) & ~shell & ~taken
        seeds, count = ndimage.label(candidates & free)
        for seed_slice, seed_id in zip(ndimage.find_objects(seeds), range(1, count + 1)):
            if seed_slice[1 - axis].stop - seed_slice[1 - axis].start > max_thickness_px:
                continue  # thicker than a wall: a filled room or corridor, not a gap
            seed: np.ndarray = seeds == seed_id
            mask: np.ndarray = ndimage.binary_dilation(seed, structure=grow) & candidates
            if free[mask].mean() < min_free_fraction:
                continue
            rows, cols = np.nonzero(mask)
            gaps.append((mask, int(np.ptp(cols if axis == 1 else rows)) + 1))
            taken |= mask
    return gaps


def predict_rules(case: Case, schema: Schema, params: SimpleNamespace) -> np.ndarray:
    """Hand-written baseline (see module docstring).

    Args:
        case (Case): Case; only ``incomplete`` and the plan scale are used.
        schema (Schema): Taxonomy.
        params (SimpleNamespace): Evaluation parameters (``rules`` table).

    Returns:
        np.ndarray: float32 (1, C, H, W) prediction.
    """
    cfg: SimpleNamespace = params.rules
    scale = case.info.get("metres_per_px") or cfg.DEFAULT_METRES_PER_PX
    labels: LabelMaps = semantic_to_labels(case.incomplete, schema, 0.5)
    shell: np.ndarray = labels.structure != EMPTY
    thickness: float = _wall_thickness_px(labels, schema)
    gaps = _shell_gaps(shell, labels.room == EMPTY, max(3, int(round(cfg.GAP_MAX_M / scale))),
                       cfg.THICKNESS_FACTOR * thickness, cfg.MIN_FREE_FRACTION)

    # Exterior is computed with all gaps closed, so it does not leak inside.
    closed_structure: np.ndarray = labels.structure.copy()
    wall_index: int = schema.structure_classes.index("wall")
    for mask, _ in gaps:
        closed_structure[mask] = wall_index
    exterior: np.ndarray = PlanStructure(LabelMaps(closed_structure, labels.room), schema, params).exterior

    prediction: np.ndarray = np.zeros((1,) + case.incomplete.shape, np.float32)
    reach: np.ndarray = np.ones((2 * int(thickness) + 3,) * 2, bool)
    for mask, length in gaps:
        if length * scale > cfg.DOOR_MAX_M:
            cls = "wall"
        else:
            cls = "window" if (ndimage.binary_dilation(mask, structure=reach) & exterior).any() else "door"
        prediction[0, schema.channel_index(cls)][mask] = 1.0

    enclosed: np.ndarray = ~exterior & (closed_structure == EMPTY) & (labels.room == EMPTY)
    pieces, count = ndimage.label(enclosed)
    sizes: np.ndarray = np.bincount(pieces.ravel(), minlength=count + 1)
    room_channel: int = schema.channel_index(cfg.FILL_ROOM_CLASS)
    for piece in range(1, count + 1):
        if sizes[piece] >= cfg.MIN_ROOM_AREA_PX:
            prediction[0, room_channel][pieces == piece] = 1.0
    return prediction


class FilePredictor:
    """Loads saved model predictions: ``<directory>/<case_id>.npz`` with ``samples``."""

    def __init__(self, directory: Path) -> None:
        """Remember the prediction directory.

        Args:
            directory (Path): Directory of ``.npz`` files.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        if not directory.is_dir():
            raise FileNotFoundError(f"Prediction directory not found: {directory}")
        self.directory: Path = directory

    def __call__(self, case: Case, schema: Schema, params: SimpleNamespace) -> np.ndarray:
        """Load the samples of one case.

        Args:
            case (Case): Case (its ``case_id`` names the file).
            schema (Schema): Taxonomy (used for a shape check).
            params (SimpleNamespace): Evaluation parameters (unused).

        Returns:
            np.ndarray: float32 (K, C, H, W).

        Raises:
            FileNotFoundError: If the case has no prediction file.
            ValueError: If the array has the wrong shape.
        """
        path: Path = self.directory / f"{case.info['case_id']}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"No prediction for case {case.info['case_id']} in {self.directory}")
        with np.load(path) as data:
            raw: np.ndarray = data["samples"]
        # uint8 files (written by model/predict.py) store scores * 255.
        samples: np.ndarray = raw.astype(np.float32) / 255.0 if raw.dtype == np.uint8 else raw.astype(np.float32)
        if samples.ndim != 4 or samples.shape[1:] != case.incomplete.shape:
            raise ValueError(f"{path}: expected (K, {schema.num_channels}, H, W), got {samples.shape}")
        return samples


BUILTIN_PREDICTORS: dict[str, PredictorFn] = {
    "empty": predict_empty,
    "oracle": predict_oracle,
    "rules": predict_rules,
}
