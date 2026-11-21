"""Hand-off from the wall/door/window model to the furniture models.

The furniture models read ``floorplan_segmentation.png`` (rooms, walls, doors, windows ...) and
``furniture_segmentation.png`` (sofa, tv, dining set ...): single-channel PNGs whose pixel values are class ids, cropped
tightly to the plan and squeezed to a square. A completed plan of ``Reconstruction`` is two label maps (structure and
room class indices) on a fixed-scale padded canvas, so this module maps class names to the furniture models' ids, crops
both masks with one box and resizes. Class ids and the output size are in ``handoff.toml``.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

try:
    import tomllib
except ModuleNotFoundError:     # Python 3.10
    import tomli as tomllib

from Reconstruction.data_prep.schema import Schema
from Reconstruction.evaluation.labels import LabelMaps

CONFIG_FILE: Path = Path(__file__).resolve().with_name("handoff.toml")


def load_settings() -> SimpleNamespace:
    """Read ``handoff.toml``.

    Returns:
        SimpleNamespace: ``export`` (SimpleNamespace); ``mask_ids`` and ``furniture_ids`` (dict[str, int]).
    """
    data: dict = tomllib.loads(CONFIG_FILE.read_text())
    return SimpleNamespace(export=SimpleNamespace(**data["export"]), mask_ids=dict(data["mask_ids"]),
                           furniture_ids=dict(data["furniture_ids"]))


def draw_ids(names: tuple[str, ...], table: dict[str, int], layer: np.ndarray, ids: np.ndarray) -> None:
    """Write the ids of every class of one label layer into ``ids`` (later classes overwrite earlier ones).

    Args:
        names (tuple[str, ...]): Class names in the layer's index order.
        table (dict[str, int]): id per class name; names absent from it are skipped.
        layer (np.ndarray): int8 (H, W) class indices (-1 = empty).
        ids (np.ndarray): uint8 (H, W) output, modified in place.
    """
    for index, name in enumerate(names):
        if name in table:
            ids[layer == index] = table[name]


def floorplan_ids(labels: LabelMaps, schema: Schema, mask_ids: dict[str, int]) -> np.ndarray:
    """Floorplan class ids on Reconstruction's canvas, structure drawn over rooms.

    Args:
        labels (LabelMaps): Structure and room class indices (-1 = empty) of the plan.
        schema (Schema): Reconstruction taxonomy.
        mask_ids (dict[str, int]): the furniture models floorplan id of every non-furniture class, plus ``background``.

    Returns:
        np.ndarray: uint8 (H, W) class ids.

    Raises:
        KeyError: If a non-furniture class of the schema has no id in ``mask_ids``.
    """
    missing: list[str] = [n for n in schema.channels if n not in mask_ids and schema.component_type[n] != "furniture"]
    if missing:
        raise KeyError(f"handoff.toml [mask_ids] has no id for: {', '.join(missing)}")
    ids: np.ndarray = np.full(labels.room.shape, mask_ids["background"], dtype=np.uint8)
    draw_ids(schema.room_classes, mask_ids, labels.room, ids)
    draw_ids(schema.structure_classes, mask_ids, labels.structure, ids)
    return ids


def furniture_ids(labels: LabelMaps, schema: Schema, table: dict[str, int]) -> np.ndarray | None:
    """Furniture class ids on Reconstruction's canvas.

    Args:
        labels (LabelMaps): Structure and room class indices of the plan.
        schema (Schema): Reconstruction taxonomy.
        table (dict[str, int]): the furniture models furniture id per Reconstruction furniture class.

    Returns:
        np.ndarray | None: uint8 (H, W) ids, or ``None`` if the taxonomy has no furniture classes (shell-only models).
    """
    names: list[str] = [n for n in schema.structure_classes if schema.component_type[n] == "furniture"]
    if not names:
        return None
    ids: np.ndarray = np.zeros(labels.room.shape, dtype=np.uint8)
    draw_ids(schema.structure_classes, {n: table[n] for n in names}, labels.structure, ids)
    return ids


def plan_box(ids: np.ndarray, background: int) -> tuple[int, int, int, int]:
    """Bounding box of everything that is not background.

    Args:
        ids (np.ndarray): uint8 (H, W) class ids.
        background (int): Background id.

    Returns:
        tuple[int, int, int, int]: ``(row0, row1, col0, col1)`` with exclusive ends; the whole canvas if empty.
    """
    rows, cols = np.nonzero(ids != background)
    if rows.size == 0:
        return 0, ids.shape[0], 0, ids.shape[1]
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def square(ids: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize to ``size`` x ``size`` (aspect ratio not kept, as the furniture models' loader does).

    Args:
        ids (np.ndarray): uint8 (h, w) class ids.
        size (int): Output side.

    Returns:
        np.ndarray: uint8 (size, size).
    """
    return np.asarray(Image.fromarray(ids).resize((size, size), Image.NEAREST))


def to_furniture_masks(labels: LabelMaps, schema: Schema, settings: SimpleNamespace) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert a plan into the furniture models' floorplan and (when the plan has furniture) furniture masks.

    Args:
        labels (LabelMaps): The completed plan.
        schema (Schema): Reconstruction taxonomy.
        settings (SimpleNamespace): ``handoff.toml`` as read by ``load_settings`` (``export``, ``mask_ids``,
            ``furniture_ids``).

    Returns:
        tuple[np.ndarray, np.ndarray | None]: uint8 (SIZE, SIZE) floorplan ids and furniture ids (``None`` without
        furniture), both cropped with the box of the floorplan.
    """
    floor: np.ndarray = floorplan_ids(labels, schema, settings.mask_ids)
    items: np.ndarray | None = furniture_ids(labels, schema, settings.furniture_ids)
    r0, r1, c0, c1 = plan_box(floor, settings.mask_ids["background"])
    floor = floor[r0:r1, c0:c1]
    items = None if items is None else items[r0:r1, c0:c1]
    size: int = int(settings.export.SIZE)
    return square(floor, size), None if items is None else square(items, size)
