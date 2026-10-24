"""Component extraction from label maps and prediction-to-ground-truth matching.

A *component* is an 8-connected piece of one class in one layer (e.g. one
door, one wall segment, one bedroom). Predicted and ground-truth components
are extracted the same way from the "added" label maps (see ``labels.py``)
and matched one-to-one, same class only, maximizing total IoU (Hungarian).
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from Reconstruction.data_prep.schema import ROOM_LAYER, STRUCTURE_LAYER, Schema
from Reconstruction.evaluation.labels import LabelMaps

_EIGHT_CONNECTED: np.ndarray = np.ones((3, 3), dtype=bool)


@dataclass
class Component:
    """One connected piece of one class.

    Attributes:
        cls (str): Class name.
        comp_type (str): Component type of the class ("wall", "door", ...).
        pixels (np.ndarray): Sorted flat pixel indices (int64).
        centroid (tuple[float, float]): (row, col) mean position.
    """

    cls: str
    comp_type: str
    pixels: np.ndarray
    centroid: tuple[float, float]

    @property
    def area(self) -> int:
        """Pixel count."""
        return int(self.pixels.size)

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Boolean mask of the component.

        Args:
            shape (tuple[int, int]): Image shape (H, W).

        Returns:
            np.ndarray: bool (H, W).
        """
        out: np.ndarray = np.zeros(shape[0] * shape[1], bool)
        out[self.pixels] = True
        return out.reshape(shape)


def extract_components(labels: LabelMaps, schema: Schema, min_area: int) -> list[Component]:
    """All components of a label map (both layers), ignoring tiny pieces.

    Args:
        labels (LabelMaps): Label maps (typically "added" labels).
        schema (Schema): Taxonomy.
        min_area (int): Minimum component area in pixels.

    Returns:
        list[Component]: Components, structure layer first.
    """
    components: list[Component] = []
    layers = ((STRUCTURE_LAYER, labels.structure, schema.structure_classes), (ROOM_LAYER, labels.room, schema.room_classes))
    width: int = labels.structure.shape[1]
    for _, label_map, classes in layers:
        for index in np.unique(label_map[label_map >= 0]):
            pieces, count = ndimage.label(label_map == index, structure=_EIGHT_CONNECTED)
            flat: np.ndarray = pieces.ravel()
            order: np.ndarray = np.argsort(flat, kind="stable")
            # Split sorted pixel indices by piece id: O(N log N), no per-piece scan.
            bounds: np.ndarray = np.searchsorted(flat[order], np.arange(1, count + 2))
            cls: str = classes[int(index)]
            for piece in range(count):
                pixels: np.ndarray = np.sort(order[bounds[piece]:bounds[piece + 1]])
                if pixels.size < min_area:
                    continue
                rows, cols = np.divmod(pixels, width)
                components.append(Component(cls, schema.component_type[cls], pixels, (float(rows.mean()), float(cols.mean()))))
    return components


def iou_matrix(predicted: list[Component], truth: list[Component]) -> np.ndarray:
    """Pairwise IoU; pairs of different classes get 0.

    Args:
        predicted (list[Component]): Predicted components.
        truth (list[Component]): Ground-truth components.

    Returns:
        np.ndarray: float (len(predicted), len(truth)).
    """
    matrix: np.ndarray = np.zeros((len(predicted), len(truth)))
    for i, pred in enumerate(predicted):
        for j, true in enumerate(truth):
            if pred.cls != true.cls:
                continue
            inter: int = np.intersect1d(pred.pixels, true.pixels, assume_unique=True).size
            if inter:
                matrix[i, j] = inter / (pred.area + true.area - inter)
    return matrix


def match(ious: np.ndarray) -> list[tuple[int, int, float]]:
    """One-to-one matching maximizing total IoU; zero-IoU pairs are dropped.

    Args:
        ious (np.ndarray): Output of ``iou_matrix``.

    Returns:
        list[tuple[int, int, float]]: (pred index, truth index, IoU) pairs.
    """
    if ious.size == 0:
        return []
    rows, cols = linear_sum_assignment(-ious)
    return [(int(r), int(c), float(ious[r, c])) for r, c in zip(rows, cols) if ious[r, c] > 0]
