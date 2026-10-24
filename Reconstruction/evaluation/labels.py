"""Label-map view of plans, prediction decoding and plan composition.

Evaluation works on two exclusive *label maps* per plan instead of multi-hot
channels:

* ``structure``: int8 (H, W), index into ``schema.structure_classes`` or -1.
* ``room``:      int8 (H, W), index into ``schema.room_classes`` or -1.

Composition rule (how a prediction is added to the incomplete input):

* structure: a predicted class replaces the input class only if it has a
  higher priority. Priority follows the taxonomy order (later = higher), so
  openings override walls (the hard door/window cases) but a prediction can
  never erase an existing door with a wall.
* room: the input label is kept; predicted room labels only fill pixels
  that have no room label.

"Added" pixels are those whose label in the composed plan differs from the
input. The ground truth is treated exactly the same way (full plan vs input),
so prediction and target are compared on equal terms.
"""

from dataclasses import dataclass

import numpy as np

from Reconstruction.data_prep.schema import Schema

EMPTY: int = -1


@dataclass
class LabelMaps:
    """Exclusive structure and room labels of one plan.

    Attributes:
        structure (np.ndarray): int8 (H, W) structure class index or -1.
        room (np.ndarray): int8 (H, W) room class index or -1.
    """

    structure: np.ndarray
    room: np.ndarray

    def occupied(self) -> np.ndarray:
        """Pixels carrying any label.

        Returns:
            np.ndarray: bool (H, W).
        """
        return (self.structure != EMPTY) | (self.room != EMPTY)


def semantic_to_labels(semantic: np.ndarray, schema: Schema, threshold: float) -> LabelMaps:
    """Decode a (C, H, W) map (binary or scores in [0, 1]) into label maps.

    Structure: highest-priority channel above ``threshold``. Room: the
    highest-scoring room channel, if it is above ``threshold``.

    Args:
        semantic (np.ndarray): (C, H, W) multi-hot map or per-channel scores.
        schema (Schema): Taxonomy.
        threshold (float): Decision threshold.

    Returns:
        LabelMaps: Decoded labels.
    """
    assert semantic.shape[0] == schema.num_channels, "channel count does not match the taxonomy"
    n_struct: int = len(schema.structure_classes)
    structure: np.ndarray = np.full(semantic.shape[1:], EMPTY, np.int8)
    for index in range(n_struct):  # later classes overwrite earlier ones = priority
        structure[semantic[index] > threshold] = index
    rooms: np.ndarray = semantic[n_struct:]
    best: np.ndarray = rooms.argmax(axis=0).astype(np.int8)
    room: np.ndarray = np.where(rooms.max(axis=0) > threshold, best, EMPTY).astype(np.int8)
    return LabelMaps(structure, room)


def labels_to_semantic(labels: LabelMaps, schema: Schema) -> np.ndarray:
    """One-hot (C, H, W) uint8 map from label maps (for rendering).

    Args:
        labels (LabelMaps): Label maps.
        schema (Schema): Taxonomy.

    Returns:
        np.ndarray: uint8 (C, H, W).
    """
    n_struct: int = len(schema.structure_classes)
    semantic: np.ndarray = np.zeros((schema.num_channels,) + labels.structure.shape, np.uint8)
    for index in range(n_struct):
        semantic[index] = labels.structure == index
    for index in range(len(schema.room_classes)):
        semantic[n_struct + index] = labels.room == index
    return semantic


def compose(base: LabelMaps, prediction: LabelMaps) -> LabelMaps:
    """Add a prediction to the incomplete plan (see module docstring).

    Args:
        base (LabelMaps): Incomplete input plan.
        prediction (LabelMaps): Decoded prediction of the missing part.

    Returns:
        LabelMaps: Composed plan.
    """
    structure: np.ndarray = np.where(prediction.structure > base.structure, prediction.structure, base.structure)
    room: np.ndarray = np.where(base.room != EMPTY, base.room, prediction.room)
    return LabelMaps(structure.astype(np.int8), room.astype(np.int8))


def added(base: LabelMaps, result: LabelMaps) -> LabelMaps:
    """Labels of ``result`` at pixels where it differs from ``base`` (-1 elsewhere).

    Args:
        base (LabelMaps): Incomplete input plan.
        result (LabelMaps): Composed or complete plan.

    Returns:
        LabelMaps: Added labels.
    """
    structure: np.ndarray = np.where((result.structure != base.structure) & (result.structure != EMPTY), result.structure, EMPTY)
    room: np.ndarray = np.where((result.room != base.room) & (result.room != EMPTY), result.room, EMPTY)
    return LabelMaps(structure.astype(np.int8), room.astype(np.int8))
