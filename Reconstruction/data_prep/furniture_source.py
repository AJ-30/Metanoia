"""Read-only loading of the FurnitureArranger release (living-room layouts).

Release facts this module relies on (checked on all 7,512 files):

* One SVG per accepted layout, in ``accepted/<annotator>/<hash>_accepted.svg``.
* Each SVG embeds a JSON ``<metadata>`` block with ``floorplan`` (WKT geometries
  with ResPlan's keys: wall, door, front_door, window, living, ...) and
  ``arrangement.furniture`` (type, centre x/y, angle, bbox_size_x/y).
* Furniture is an oriented box: size ``bbox_size_x`` x ``bbox_size_y`` centred at
  (x, y), rotated counter-clockwise by ``angle`` degrees (y up). Verified against
  the polygons drawn in the SVGs (also for the non-90-degree angles).
* Wall MULTIPOLYGONs can be topologically invalid (overlapping pieces), so every
  geometry is repaired with ``make_valid`` before use.
* The floorplan ``id`` is not unique (7,359 files share one value), so plans are
  identified by their position in the sorted file list.
* There is no official split. Layouts of the same floor plan (identical geometry)
  stay together; groups are assigned to train/val/test by a salted hash.
"""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from shapely import make_valid, wkt
from shapely.affinity import rotate
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from Reconstruction.data_prep.geometry import SPLIT_NAMES, PlanGeometry

_METADATA = re.compile(r"<metadata[^>]*>(.*?)</metadata>", re.S)
READ_THREADS: int = 32


def list_svgs(root: Path) -> list[Path]:
    """All layout SVGs below ``root``, sorted (this order defines plan ids).

    Args:
        root (Path): The ``accepted`` directory.

    Returns:
        list[Path]: Sorted file paths.

    Raises:
        FileNotFoundError: If the directory has no SVGs.
    """
    files: list[Path] = sorted(root.glob("*/*.svg"))
    if not files:
        raise FileNotFoundError(f"No SVG files found under {root}")
    return files


def parse_layout(text: str) -> dict[str, Any]:
    """Parse the embedded JSON of one layout SVG (or a bare layout JSON document).

    Args:
        text (str): SVG source with a ``<metadata>`` JSON block, or the JSON itself
            (an object with ``floorplan`` and ``arrangement.furniture``).

    Returns:
        dict[str, Any]: ``floorplan`` (WKT strings) and ``furniture`` (list of dicts).

    Raises:
        ValueError: If the text has no usable layout.
    """
    match = _METADATA.search(text)
    try:
        data: dict[str, Any] = json.loads(match.group(1) if match else text)
        return {"floorplan": data["floorplan"], "furniture": data.get("arrangement", {}).get("furniture", [])}
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        raise ValueError("no layout found: expected a FurnitureArranger SVG (with its <metadata> block) "
                         "or JSON with 'floorplan' and 'arrangement.furniture'") from None


def read_layout(path: Path) -> dict[str, Any]:
    """Parse the embedded JSON of one SVG.

    Args:
        path (Path): SVG file.

    Returns:
        dict[str, Any]: ``floorplan`` (WKT strings) and ``furniture`` (list of dicts).

    Raises:
        ValueError: If the file has no usable metadata.
    """
    try:
        return parse_layout(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(f"{path.name}: {error}") from None


def group_key(floorplan: dict[str, Any], salt: str) -> str:
    """Hash identifying a floor plan (geometry only, not furniture or id).

    Args:
        floorplan (dict[str, Any]): The ``floorplan`` JSON object.
        salt (str): Split salt (changes the group -> split assignment).

    Returns:
        str: Hex digest.
    """
    geometry = {k: v for k, v in floorplan.items() if k != "id"}
    return hashlib.sha1((salt + json.dumps(geometry, sort_keys=True)).encode()).hexdigest()


def assign_splits(files: list[Path], layouts: list[dict[str, Any]], cfg: SimpleNamespace) -> dict[str, list[int]]:
    """Group-aware, deterministic train/val/test assignment.

    Args:
        files (list[Path]): Sorted files (plan id = index + 1).
        layouts (list[dict[str, Any]]): Parsed layouts, same order.
        cfg (SimpleNamespace): ``params.furniture`` (fractions and salt).

    Returns:
        dict[str, list[int]]: Split -> plan ids (ascending).
    """
    splits: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    for index, layout in enumerate(layouts):
        # Uniform in [0, 1) from the group hash.
        u: float = int(group_key(layout["floorplan"], cfg.SPLIT_SALT)[:12], 16) / float(16 ** 12)
        if u < cfg.TEST_FRACTION:
            splits["test"].append(index + 1)
        elif u < cfg.TEST_FRACTION + cfg.VAL_FRACTION:
            splits["val"].append(index + 1)
        else:
            splits["train"].append(index + 1)
    return splits


def load_layouts(files: list[Path]) -> list[dict[str, Any]]:
    """Read every SVG (threaded: the file system is high-latency).

    Args:
        files (list[Path]): SVG files.

    Returns:
        list[dict[str, Any]]: Parsed layouts, same order.
    """
    with ThreadPoolExecutor(READ_THREADS) as pool:
        return list(pool.map(read_layout, files))


def furniture_polygon(item: dict[str, Any]) -> Polygon:
    """Oriented bounding box of one furniture item.

    Args:
        item (dict[str, Any]): Furniture JSON object.

    Returns:
        Polygon: Box centred at (x, y), rotated counter-clockwise by ``angle``.
    """
    x, y = float(item["x"]), float(item["y"])
    half_x, half_y = float(item["bbox_size_x"]) / 2, float(item["bbox_size_y"]) / 2
    return rotate(box(x - half_x, y - half_y, x + half_x, y + half_y), float(item["angle"]), origin=(x, y))


def _polygon_parts(geom: BaseGeometry) -> list[Polygon]:
    """Non-empty polygons of a geometry, descending into nested collections.

    ``make_valid`` can return a GeometryCollection holding MultiPolygons (and
    stray lines/points, which are dropped).

    Args:
        geom (BaseGeometry): Any geometry.

    Returns:
        list[Polygon]: Polygon parts.
    """
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return [part for sub in getattr(geom, "geoms", []) for part in _polygon_parts(sub)]


def _geometry(value: str) -> BaseGeometry:
    """Parse and repair a WKT string.

    Args:
        value (str): WKT.

    Returns:
        BaseGeometry: Valid geometry (possibly empty).
    """
    return make_valid(wkt.loads(value))


def layout_to_geometry(plan_id: int, layout: dict[str, Any], params: SimpleNamespace) -> PlanGeometry:
    """Convert one parsed layout into a ``PlanGeometry`` in our taxonomy.

    Args:
        plan_id (int): Plan id (sorted-file index + 1).
        layout (dict[str, Any]): Output of ``read_layout``.
        params (SimpleNamespace): Loaded ``params_furniture.toml``.

    Returns:
        PlanGeometry: Walls as one union; one instance per door / window / room /
        furniture piece.

    Raises:
        ValueError: If the plan has no walls or a furniture type is unknown.
    """
    class_map: dict[str, str] = dict(vars(params.taxonomy.furniture_class_map))
    known_furniture: set[str] = {c for c, t in vars(params.taxonomy.component_type).items() if t == "furniture"}
    wall_parts: list[Polygon] = []
    instances: list[tuple[str, Polygon | MultiPolygon]] = []
    for key, class_name in class_map.items():
        if key not in layout["floorplan"]:
            continue
        parts: list[Polygon] = _polygon_parts(_geometry(layout["floorplan"][key]))
        if class_name == "wall":
            wall_parts.extend(parts)
        else:
            instances.extend((class_name, part) for part in parts)
    if not wall_parts:
        raise ValueError(f"plan {plan_id} has no walls")
    # Shell-only taxonomy (no furniture classes): the furniture is deliberately not drawn.
    for item in layout["furniture"] if known_furniture else []:
        if item["type"] not in known_furniture:
            raise ValueError(f"plan {plan_id}: unknown furniture type '{item['type']}'")
        instances.append((item["type"], furniture_polygon(item)))
    return PlanGeometry(plan_id=plan_id, metres_per_unit=float(params.furniture.METRES_PER_UNIT),
                        wall=unary_union(wall_parts), instances=instances)
