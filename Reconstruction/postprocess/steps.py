"""Structure-aware post-processing steps for a predicted missing part.

Every step works on the *added* label maps (what the prediction adds to the
incomplete plan, see ``evaluation/labels.py``) and receives the incomplete
plan as context. Steps are independent and order-aware; a pipeline is an
ordered list of step names from ``params.toml`` (``[pipelines]``), so every
ablation is a config change.

Step signature: ``(added, base, schema, cfg) -> added`` where ``cfg`` is the
step's own ``[steps.<name>]`` table. Steps never modify their inputs.

Architectural rules enforced here:

* ``remove_small``       no tiny fragments (noise)
* ``openings_on_wall``   every door / window sits on a wall
* ``close_wall_gaps``    predicted walls connect to the walls next to them
* ``align_walls``        predicted walls are straight and axis-aligned
* ``snap_rooms``         a room fills the wall-enclosed area it belongs to
"""

from types import SimpleNamespace
from typing import Callable

import numpy as np
from scipy import ndimage
from skimage.morphology import disk

from Reconstruction.data_prep.schema import Schema
from Reconstruction.evaluation.labels import EMPTY, LabelMaps, compose

StepFn = Callable[[LabelMaps, LabelMaps, Schema, SimpleNamespace], LabelMaps]

_EIGHT_CONNECTED: np.ndarray = np.ones((3, 3), dtype=bool)


def _copy(labels: LabelMaps) -> LabelMaps:
    """Independent copy of label maps.

    Args:
        labels (LabelMaps): Label maps.

    Returns:
        LabelMaps: Deep copy.
    """
    return LabelMaps(labels.structure.copy(), labels.room.copy())


def _drop_small(label_map: np.ndarray, min_area: int) -> np.ndarray:
    """Erase 8-connected same-class pieces smaller than ``min_area``.

    Args:
        label_map (np.ndarray): int8 (H, W) class indices or -1.
        min_area (int): Minimum piece area in pixels.

    Returns:
        np.ndarray: Cleaned copy.
    """
    out: np.ndarray = label_map.copy()
    for index in np.unique(label_map[label_map >= 0]):
        pieces, count = ndimage.label(label_map == index, structure=_EIGHT_CONNECTED)
        sizes: np.ndarray = np.bincount(pieces.ravel(), minlength=count + 1)
        small: np.ndarray = np.flatnonzero(sizes < min_area)
        out[np.isin(pieces, small[small > 0])] = EMPTY  # id 0 is the background
    return out


def remove_small(added: LabelMaps, base: LabelMaps, schema: Schema, cfg: SimpleNamespace) -> LabelMaps:
    """Drop added pieces below a minimum area (structure and room separately).

    Args:
        added (LabelMaps): Predicted added labels.
        base (LabelMaps): Incomplete plan (unused).
        schema (Schema): Taxonomy (unused).
        cfg (SimpleNamespace): ``STRUCTURE_MIN_AREA_PX``, ``ROOM_MIN_AREA_PX``.

    Returns:
        LabelMaps: Cleaned added labels.
    """
    return LabelMaps(_drop_small(added.structure, cfg.STRUCTURE_MIN_AREA_PX), _drop_small(added.room, cfg.ROOM_MIN_AREA_PX))


def openings_on_wall(added: LabelMaps, base: LabelMaps, schema: Schema, cfg: SimpleNamespace) -> LabelMaps:
    """Remove predicted doors / windows that touch no wall.

    Args:
        added (LabelMaps): Predicted added labels.
        base (LabelMaps): Incomplete plan.
        schema (Schema): Taxonomy.
        cfg (SimpleNamespace): ``TOUCH_RADIUS_PX``.

    Returns:
        LabelMaps: Added labels without floating openings.
    """
    out: LabelMaps = _copy(added)
    wall: int = schema.structure_classes.index("wall")
    # Walls of the input and of the prediction; a hard-case door replaces input wall pixels, so both count.
    near_wall: np.ndarray = ndimage.binary_dilation((base.structure == wall) | (added.structure == wall), structure=disk(cfg.TOUCH_RADIUS_PX))
    for name in ("door", "window"):
        index: int = schema.structure_classes.index(name)
        pieces, count = ndimage.label(added.structure == index, structure=_EIGHT_CONNECTED)
        for piece in range(1, count + 1):
            if not (near_wall & (pieces == piece)).any():
                out.structure[pieces == piece] = EMPTY
    return out


def close_wall_gaps(added: LabelMaps, base: LabelMaps, schema: Schema, cfg: SimpleNamespace) -> LabelMaps:
    """Bridge short axis-aligned gaps between walls next to a predicted wall.

    A horizontal / vertical morphological closing of all walls (input +
    prediction) finds the gaps; only free pixels within ``NEAR_PREDICTION_PX``
    of a predicted wall are filled, so gaps the model did not touch are kept.

    Args:
        added (LabelMaps): Predicted added labels.
        base (LabelMaps): Incomplete plan.
        schema (Schema): Taxonomy.
        cfg (SimpleNamespace): ``MAX_GAP_PX``, ``NEAR_PREDICTION_PX``.

    Returns:
        LabelMaps: Added labels with bridged walls.
    """
    wall: int = schema.structure_classes.index("wall")
    predicted: np.ndarray = added.structure == wall
    if not predicted.any():
        return added
    walls: np.ndarray = predicted | (base.structure == wall)
    free: np.ndarray = compose(base, added).structure == EMPTY
    length: int = cfg.MAX_GAP_PX + 1  # closing with a line of length L fills gaps shorter than L
    # Zero padding keeps the closing from eroding walls that touch the canvas border.
    padded: np.ndarray = np.pad(walls, length)
    closed: np.ndarray = np.zeros_like(padded)
    for line in (np.ones((1, length), bool), np.ones((length, 1), bool)):
        closed |= ndimage.binary_closing(padded, structure=line)
    bridge: np.ndarray = closed[length:-length, length:-length] & free
    bridge &= ndimage.binary_dilation(predicted, structure=disk(cfg.NEAR_PREDICTION_PX))
    out: LabelMaps = _copy(added)
    out.structure[bridge] = wall
    return out


def align_walls(added: LabelMaps, base: LabelMaps, schema: Schema, cfg: SimpleNamespace) -> LabelMaps:
    """Straighten predicted wall pieces that are nearly rectangles.

    A piece filling at least ``MIN_FILL`` of its bounding box (and thin
    enough) is replaced by the full box. Only pixels that are empty in the
    input and not claimed by another added class are touched.

    Args:
        added (LabelMaps): Predicted added labels.
        base (LabelMaps): Incomplete plan.
        schema (Schema): Taxonomy.
        cfg (SimpleNamespace): ``MIN_FILL``, ``MAX_THICKNESS_PX``.

    Returns:
        LabelMaps: Added labels with aligned walls.
    """
    wall: int = schema.structure_classes.index("wall")
    pieces, _ = ndimage.label(added.structure == wall, structure=_EIGHT_CONNECTED)
    out: LabelMaps = _copy(added)
    for piece, box in enumerate(ndimage.find_objects(pieces), start=1):
        height, width = box[0].stop - box[0].start, box[1].stop - box[1].start
        area: int = int((pieces[box] == piece).sum())
        if area / (height * width) < cfg.MIN_FILL or min(height, width) > cfg.MAX_THICKNESS_PX:
            continue
        target: np.ndarray = (base.structure[box] == EMPTY) & ((out.structure[box] == EMPTY) | (out.structure[box] == wall))
        out.structure[box][target] = wall  # box is a basic slice: a view, so this writes into out
    return out


def _snap_region(out: LabelMaps, region: np.ndarray, composed: LabelMaps, cfg: SimpleNamespace) -> None:
    """Make one enclosed free region a single room, or clear it (in place).

    Args:
        out (LabelMaps): Added labels being edited.
        region (np.ndarray): bool (H, W) free region.
        composed (LabelMaps): Input + prediction before snapping.
        cfg (SimpleNamespace): ``MIN_COVERAGE``, ``UNDER_WALL_PX``.
    """
    predicted: np.ndarray = out.room[region]
    labelled: np.ndarray = predicted[predicted >= 0]
    if labelled.size / predicted.size < cfg.MIN_COVERAGE:
        out.room[region] = EMPTY
        return
    room_class: int = int(np.bincount(labelled).argmax())
    out.room[region] = room_class
    # Rooms reach under the surrounding walls in the data; only claim pixels no room has.
    under: np.ndarray = ndimage.binary_dilation(region, structure=disk(cfg.UNDER_WALL_PX)) & ~region
    under &= (composed.structure != EMPTY) & (composed.room == EMPTY) & (out.room == EMPTY)
    out.room[under] = room_class


def snap_rooms(added: LabelMaps, base: LabelMaps, schema: Schema, cfg: SimpleNamespace) -> LabelMaps:
    """Make predicted rooms follow the walls that enclose them.

    Free space (no structure) is split into regions by the composed
    structure. Regions touching the canvas border (exterior), larger than
    ``MAX_REGION_FRACTION`` of the canvas, already holding an input room
    label (a partly removed room cannot be decided here) or without any
    predicted room are left untouched. Any other region becomes one room of
    the majority predicted class when the prediction covers enough of it,
    else it is cleared.

    Args:
        added (LabelMaps): Predicted added labels.
        base (LabelMaps): Incomplete plan.
        schema (Schema): Taxonomy (unused).
        cfg (SimpleNamespace): ``MIN_COVERAGE``, ``MAX_REGION_FRACTION``, ``UNDER_WALL_PX``.

    Returns:
        LabelMaps: Added labels with snapped rooms.
    """
    composed: LabelMaps = compose(base, added)
    regions, count = ndimage.label(composed.structure == EMPTY)
    height, width = regions.shape
    exterior: np.ndarray = np.unique(np.concatenate([regions[0], regions[-1], regions[:, 0], regions[:, -1]]))
    out: LabelMaps = _copy(added)
    for index in range(1, count + 1):
        if index in exterior:
            continue
        region: np.ndarray = regions == index
        if region.sum() > cfg.MAX_REGION_FRACTION * height * width or (base.room[region] != EMPTY).any() or (added.room[region] == EMPTY).all():
            continue
        _snap_region(out, region, composed, cfg)
    return out


STEPS: dict[str, StepFn] = {
    "remove_small": remove_small,
    "openings_on_wall": openings_on_wall,
    "close_wall_gaps": close_wall_gaps,
    "align_walls": align_walls,
    "snap_rooms": snap_rooms,
}
