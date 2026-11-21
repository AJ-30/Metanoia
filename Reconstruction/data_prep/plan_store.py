"""On-disk storage of rasterized plans for one split.

Layout of ``<PROCESSED_ROOT>/<dataset>/<split>/``:

* ``room.npy``       uint8 (N, H, W) room instance layers (memory-mappable).
* ``structure.npy``  uint8 (N, H, W) structure instance layers.
* ``plans.jsonl``    one JSON object per plan (same order as the arrays):
  ``plan_id``, ``metres_per_px`` and the ``instances`` list.

Arrays are opened with ``mmap_mode="r"`` so data loaders touch only the plans
they read and cannot modify the store.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from Reconstruction.data_prep.schema import ROOM_LAYER, STRUCTURE_LAYER

if TYPE_CHECKING:
    # Type-only import: readers of a store (e.g. the PyTorch dataset) must not
    # pull in OpenCV. With this conda env, importing cv2 after torch fails
    # (torch loads an older system libgcc_s than OpenCV needs).
    from Reconstruction.data_prep.rasterize import RasterPlan

ROOM_FILE: str = "room.npy"
STRUCTURE_FILE: str = "structure.npy"
PLANS_FILE: str = "plans.jsonl"


def write_split(split_dir: Path, plans: list[RasterPlan]) -> None:
    """Write rasterized plans of one split.

    Args:
        split_dir (Path): Output directory (created; must not already hold a store).
        plans (list[RasterPlan]): Plans in the order they should be indexed.

    Raises:
        FileExistsError: If a store already exists in ``split_dir``.
        ValueError: If ``plans`` is empty.
    """
    if not plans:
        raise ValueError(f"No plans to write to {split_dir}")
    if (split_dir / PLANS_FILE).exists():
        raise FileExistsError(f"Plan store already exists: {split_dir}")
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / ROOM_FILE, np.stack([plan.room for plan in plans]))
    np.save(split_dir / STRUCTURE_FILE, np.stack([plan.structure for plan in plans]))
    with open(split_dir / PLANS_FILE, "w", encoding="utf-8") as handle:
        for plan in plans:
            # NaN is not valid JSON; unknown scale is stored as null.
            scale: float | None = None if math.isnan(plan.metres_per_px) else plan.metres_per_px
            record: dict[str, Any] = {"plan_id": plan.plan_id, "metres_per_px": scale, "instances": plan.instances}
            handle.write(json.dumps(record) + "\n")


class PlanStore:
    """Read-only access to one split's rasterized plans.

    Attributes:
        split (str): Split name (directory name).
    """

    def __init__(self, split_dir: Path) -> None:
        """Open the store lazily (arrays are memory-mapped).

        Args:
            split_dir (Path): Directory written by ``write_split``.

        Raises:
            FileNotFoundError: If the store files are missing.
        """
        for name in (ROOM_FILE, STRUCTURE_FILE, PLANS_FILE):
            if not (split_dir / name).is_file():
                raise FileNotFoundError(f"Missing {name} in {split_dir}")
        self.split: str = split_dir.name
        self._room: np.ndarray = np.load(split_dir / ROOM_FILE, mmap_mode="r")
        self._structure: np.ndarray = np.load(split_dir / STRUCTURE_FILE, mmap_mode="r")
        with open(split_dir / PLANS_FILE, "r", encoding="utf-8") as handle:
            self._records: list[dict[str, Any]] = [json.loads(line) for line in handle]
        assert len(self._records) == self._room.shape[0] == self._structure.shape[0], "store arrays and records disagree"

    def __len__(self) -> int:
        """Number of plans in the split."""
        return len(self._records)

    def layers(self, index: int) -> dict[str, np.ndarray]:
        """Writable copies of both instance layers of one plan.

        Args:
            index (int): Plan index within the split.

        Returns:
            dict[str, np.ndarray]: {"structure": uint8 (H, W), "room": uint8 (H, W)}.
        """
        return {STRUCTURE_LAYER: np.array(self._structure[index]), ROOM_LAYER: np.array(self._room[index])}

    def record(self, index: int) -> dict[str, Any]:
        """Metadata of one plan (a deep copy, safe to modify).

        Args:
            index (int): Plan index within the split.

        Returns:
            dict[str, Any]: ``plan_id``, ``metres_per_px`` and ``instances``.
        """
        return json.loads(json.dumps(self._records[index]))

    def instance_areas(self) -> list[tuple[str, int]]:
        """(class, pixel area) of every instance in the split.

        Returns:
            list[tuple[str, int]]: One entry per instance.
        """
        return [(inst["cls"], inst["area_px"]) for rec in self._records for inst in rec["instances"]]

    def label_thumbnails(self, size: int, class_index: dict[str, int]) -> np.ndarray:
        """Low-resolution semantic label images, used for near-duplicate search.

        Each thumbnail pixel holds a class index + 1 (0 = empty), structure
        classes taking precedence over rooms, sampled with nearest neighbour.

        Args:
            size (int): Thumbnail side in pixels (must divide the canvas size).
            class_index (dict[str, int]): Class -> semantic channel index.

        Returns:
            np.ndarray: uint8 (N, size, size) label thumbnails.
        """
        canvas: int = self._room.shape[1]
        assert canvas % size == 0, "thumbnail size must divide the canvas size"
        step: int = canvas // size
        out: np.ndarray = np.zeros((len(self), size, size), np.uint8)
        for i, rec in enumerate(self._records):
            luts: dict[str, np.ndarray] = {layer: np.zeros(256, np.uint8) for layer in (STRUCTURE_LAYER, ROOM_LAYER)}
            for inst in rec["instances"]:
                luts[inst["layer"]][inst["id"]] = class_index[inst["cls"]] + 1
            room: np.ndarray = luts[ROOM_LAYER][self._room[i, step // 2::step, step // 2::step]]
            structure: np.ndarray = luts[STRUCTURE_LAYER][self._structure[i, step // 2::step, step // 2::step]]
            out[i] = np.where(structure > 0, structure, room)
        return out
