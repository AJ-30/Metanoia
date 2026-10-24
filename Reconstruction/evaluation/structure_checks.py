"""Architectural validity checks on a single plan (no ground truth needed).

Computed on the composed plan (input + prediction) and, for reference, on the
complete ground-truth plan:

* ``shell_pieces``: connected pieces of the building shell (walls, railings,
  columns, openings). A missing wall splits the shell; a good reconstruction
  restores the ground-truth count.
* ``openings_on_wall``: doors/windows touching a wall.
* ``rooms_enclosed``: rooms (except open ones such as balconies) that do not
  touch the outside of the building directly.
* ``rooms_reachable``: rooms in the largest group connected through doors or
  open (wall-less) boundaries; 1.0 means every room can be walked to.
* ``rooms_without(cls)``: rooms (balconies included) touching no door (or no window). Comparing
  the input with the complete plan tells whether a removed opening left a
  visible clue (a room that lost its last door/window).
"""

from types import SimpleNamespace

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.morphology import disk

from Reconstruction.data_prep.schema import Schema
from Reconstruction.evaluation.labels import EMPTY, LabelMaps

_EIGHT_CONNECTED: np.ndarray = np.ones((3, 3), dtype=bool)
_OPENINGS: tuple[str, str] = ("door", "window")


def _touching_values(mask: np.ndarray, image: np.ndarray, radius: int) -> set[int]:
    """Values of ``image`` within ``radius`` px of ``mask`` (crop for speed).

    Args:
        mask (np.ndarray): bool (H, W) query region.
        image (np.ndarray): int (H, W) values to collect.
        radius (int): Dilation radius.

    Returns:
        set[int]: Distinct values found (including 0 / -1 if present).
    """
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return set()
    h, w = mask.shape
    crop = (slice(max(rows.min() - radius, 0), min(rows.max() + radius + 1, h)),
            slice(max(cols.min() - radius, 0), min(cols.max() + radius + 1, w)))
    grown: np.ndarray = ndimage.binary_dilation(mask[crop], structure=disk(radius))
    return {int(v) for v in np.unique(image[crop][grown])}


class PlanStructure:
    """Pre-computed pieces of one plan used by all validity checks."""

    def __init__(self, labels: LabelMaps, schema: Schema, params: SimpleNamespace) -> None:
        """Analyse a plan once.

        Args:
            labels (LabelMaps): Plan labels.
            schema (Schema): Taxonomy.
            params (SimpleNamespace): Evaluation parameters.
        """
        self._labels: LabelMaps = labels
        self._schema: Schema = schema
        self._radius: int = params.structure.TOUCH_RADIUS_PX
        self._min_area: int = params.components.MIN_AREA_PX
        open_ids: set[int] = {schema.room_classes.index(c) for c in params.structure.OPEN_ROOM_CLASSES if c in schema.room_classes}
        self._shell_ids: list[int] = [schema.structure_classes.index(c) for c in params.structure.SHELL_CLASSES]
        self.exterior: np.ndarray = self._exterior()
        self.room_pieces, self._room_class = self._room_components()
        self._closed_rooms: list[int] = [p for p, cls in self._room_class.items() if cls not in open_ids]
        self._openings: dict[str, list[np.ndarray]] = {c: self._pieces(labels.structure == schema.structure_classes.index(c))
                                                       for c in _OPENINGS}

    def _exterior(self) -> np.ndarray:
        """Unlabelled pixels connected to the canvas border.

        Returns:
            np.ndarray: bool (H, W).
        """
        pieces, _ = ndimage.label(~self._labels.occupied())
        border: np.ndarray = np.unique(np.concatenate([pieces[0], pieces[-1], pieces[:, 0], pieces[:, -1]]))
        return np.isin(pieces, border[border > 0])

    def _pieces(self, mask: np.ndarray) -> list[np.ndarray]:
        """Connected pieces of a mask, ignoring tiny ones.

        Args:
            mask (np.ndarray): bool (H, W).

        Returns:
            list[np.ndarray]: One bool mask per piece.
        """
        pieces, count = ndimage.label(mask, structure=_EIGHT_CONNECTED)
        sizes: np.ndarray = np.bincount(pieces.ravel(), minlength=count + 1)
        return [pieces == p for p in range(1, count + 1) if sizes[p] >= self._min_area]

    def _room_components(self) -> tuple[np.ndarray, dict[int, int]]:
        """Label image of room pieces (0 = none) and piece -> room class index.

        Returns:
            tuple[np.ndarray, dict[int, int]]: int32 (H, W) image and class map.
        """
        image: np.ndarray = np.zeros(self._labels.room.shape, np.int32)
        classes: dict[int, int] = {}
        for index in np.unique(self._labels.room[self._labels.room != EMPTY]):
            for piece in self._pieces(self._labels.room == index):
                piece_id: int = len(classes) + 1
                image[piece] = piece_id
                classes[piece_id] = int(index)
        return image, classes

    def shell_pieces(self) -> int:
        """Number of connected pieces of the building shell.

        Returns:
            int: Piece count.
        """
        return len(self._pieces(np.isin(self._labels.structure, self._shell_ids)))

    def openings_on_wall(self) -> tuple[int, int]:
        """Doors/windows touching a wall.

        Returns:
            tuple[int, int]: (openings touching a wall, all openings).
        """
        wall: int = self._schema.structure_classes.index("wall")
        pieces: list[np.ndarray] = self._openings["door"] + self._openings["window"]
        ok: int = sum(wall in _touching_values(p, self._labels.structure, self._radius) for p in pieces)
        return ok, len(pieces)

    def rooms_enclosed(self) -> tuple[int, int]:
        """Closed rooms not touching the building exterior directly.

        Returns:
            tuple[int, int]: (enclosed rooms, closed rooms).
        """
        exterior: np.ndarray = self.exterior.astype(np.int8)
        ok: int = sum(1 not in _touching_values(self.room_pieces == p, exterior, 1) for p in self._closed_rooms)
        return ok, len(self._closed_rooms)

    def rooms_reachable(self) -> tuple[int, int]:
        """Rooms in the largest group linked by doors or open boundaries.

        Returns:
            tuple[int, int]: (rooms in largest group, all rooms).
        """
        n: int = len(self._room_class)
        if n == 0:
            return 0, 0
        edges: set[tuple[int, int]] = set()
        for door in self._openings["door"]:
            rooms = sorted(v for v in _touching_values(door, self.room_pieces, self._radius) if v > 0)
            edges.update((a, b) for i, a in enumerate(rooms) for b in rooms[i + 1:])
        # Open boundaries: two different room pieces directly adjacent (no wall between).
        img: np.ndarray = self.room_pieces
        for a, b in ((img[:, :-1], img[:, 1:]), (img[:-1, :], img[1:, :])):
            pairs: np.ndarray = np.stack([a.ravel(), b.ravel()], axis=1)
            pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0) & (pairs[:, 0] != pairs[:, 1])]
            edges.update(map(tuple, np.unique(np.sort(pairs, axis=1), axis=0).tolist()))
        if not edges:
            return 1, n
        src, dst = zip(*edges)
        graph = coo_matrix((np.ones(len(src)), (np.array(src) - 1, np.array(dst) - 1)), shape=(n, n))
        _, groups = connected_components(graph, directed=False)
        return int(np.bincount(groups).max()), n

    def rooms_without(self, opening_cls: str) -> int:
        """Rooms (open ones such as balconies included) touching no piece of an opening class.

        Balconies count: one that loses its only door is a visible clue too.

        Args:
            opening_cls (str): "door" or "window".

        Returns:
            int: Number of such rooms.
        """
        touched: set[int] = set()
        for piece in self._openings[opening_cls]:
            touched |= _touching_values(piece, self.room_pieces, self._radius)
        return sum(p not in touched for p in self._room_class)

    def connection(self, mask: np.ndarray) -> frozenset[int]:
        """Which rooms (piece ids of THIS plan) and whether the exterior a region touches.

        Args:
            mask (np.ndarray): bool (H, W) region, e.g. one door.

        Returns:
            frozenset[int]: Room piece ids, plus -1 if it touches the exterior.
        """
        rooms: set[int] = {v for v in _touching_values(mask, self.room_pieces, self._radius) if v > 0}
        if 1 in _touching_values(mask, self.exterior.astype(np.int8), self._radius):
            rooms.add(-1)
        return frozenset(rooms)
