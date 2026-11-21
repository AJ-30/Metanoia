"""Read-only access to the CubiCasa5K release (``cubicasa5k.zip``).

The zip is never extracted or modified: split lists and ``model.svg``
annotations are read straight from it.

CubiCasa5K facts this module relies on (checked on the release):

* ``train.txt`` / ``val.txt`` / ``test.txt`` list house folders
  (e.g. ``/high_quality/1245/``); 4200 / 400 / 400 houses.
* A house SVG holds one ``<g class="Floorplan ...">`` group per floor (1-3).
  Each floor becomes one plan with id ``house_id * FLOOR_ID_FACTOR + floor``.
* Elements are ``<g>`` groups whose first ``class`` token names the type:
  ``Wall`` (polygon; its ``Door``/``Window`` children are the openings),
  ``Railing``, ``Column`` (polygon or circle), ``Stairs`` (nested flight
  polygons) and ``Space <Type> ...`` (room polygon).
* Door/window polygons lie inside their wall polygon, so openings are cut out
  of the walls here to match ResPlan (where walls stop at openings).
* Coordinates are 100 SVG units per metre with y pointing down; y is flipped
  so plans are drawn north-up like ResPlan.
"""

import math
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from Reconstruction.data_prep.geometry import SPLIT_NAMES, PlanGeometry

SVG_NS: str = "{http://www.w3.org/2000/svg}"
ZIP_ROOT: str = "cubicasa5k"
FLOOR_TOKEN: str = "Floorplan"
SPACE_TOKEN: str = "Space"
OPENING_CLASSES: tuple[str, str] = ("door", "window")
_NUMBER: re.Pattern = re.compile(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?")
_TRANSFORM: re.Pattern = re.compile(r"(matrix|translate|scale|rotate)\s*\(([^)]*)\)")


# ----------------------------------------------------------------------
# Split index and raw reading
# ----------------------------------------------------------------------

def _house_id(folder: str) -> int:
    """Numeric house id from a split-list entry like ``/high_quality/1245/``.

    Args:
        folder (str): Entry of a split list.

    Returns:
        int: House id.
    """
    return int(folder.strip("/").split("/")[-1])


def load_cubicasa_splits(zip_path: Path) -> dict[str, list[tuple[int, str]]]:
    """Official house splits as (house id, member path of model.svg).

    Args:
        zip_path (Path): Path to ``cubicasa5k.zip``.

    Returns:
        dict[str, list[tuple[int, str]]]: Split -> sorted (house_id, svg member).

    Raises:
        FileNotFoundError: If the zip is missing.
        ValueError: If a house id repeats or appears in two splits.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"CubiCasa5K zip not found: {zip_path}")
    splits: dict[str, list[tuple[int, str]]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        members: set[str] = set(archive.namelist())
        for split in SPLIT_NAMES:
            folders: list[str] = archive.read(f"{ZIP_ROOT}/{split}.txt").decode().split()
            entries = [(_house_id(f), f"{ZIP_ROOT}/{f.strip('/')}/model.svg") for f in folders]
            missing: list[str] = [m for _, m in entries if m not in members]
            if missing:
                raise ValueError(f"{split}: {len(missing)} listed SVGs missing from the zip, e.g. {missing[0]}")
            splits[split] = sorted(entries)
    all_ids: list[int] = [hid for entries in splits.values() for hid, _ in entries]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("A CubiCasa house id repeats within or across splits")
    return splits


def read_svg_bytes(zip_path: Path, members: list[str]) -> list[bytes]:
    """Read SVG files from the zip (read-only), in the given order.

    Args:
        zip_path (Path): Path to ``cubicasa5k.zip``.
        members (list[str]): Member paths.

    Returns:
        list[bytes]: File contents.
    """
    with zipfile.ZipFile(zip_path) as archive:
        return [archive.read(member) for member in members]


# ----------------------------------------------------------------------
# SVG geometry helpers
# ----------------------------------------------------------------------

def _parse_transform(text: str | None) -> np.ndarray:
    """3x3 affine matrix of an SVG ``transform`` attribute.

    Args:
        text (str | None): Attribute value (None = identity).

    Returns:
        np.ndarray: 3x3 float matrix.

    Raises:
        ValueError: On transform functions this parser does not support.
    """
    matrix: np.ndarray = np.eye(3)
    if not text:
        return matrix
    for name, args_text in _TRANSFORM.findall(text):
        args: list[float] = [float(v) for v in _NUMBER.findall(args_text)]
        step: np.ndarray = np.eye(3)
        if name == "matrix":
            step[:2, :] = np.array(args, dtype=float).reshape(3, 2).T
        elif name == "translate":
            step[:2, 2] = [args[0], args[1] if len(args) > 1 else 0.0]
        elif name == "scale":
            step[0, 0], step[1, 1] = args[0], args[1] if len(args) > 1 else args[0]
        else:  # rotate(angle [cx cy])
            angle: float = math.radians(args[0])
            cx, cy = (args[1], args[2]) if len(args) == 3 else (0.0, 0.0)
            rot = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
            to_origin, back = np.eye(3), np.eye(3)
            to_origin[:2, 2], back[:2, 2] = (-cx, -cy), (cx, cy)
            step = back @ rot @ to_origin
        matrix = matrix @ step
    if _TRANSFORM.sub("", text).strip(" ,"):
        raise ValueError(f"Unsupported SVG transform: {text!r}")
    return matrix


def _apply(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply an affine matrix to (N, 2) points and flip y (SVG y-down -> y-up).

    Args:
        matrix (np.ndarray): 3x3 affine matrix.
        xy (np.ndarray): (N, 2) SVG coordinates.

    Returns:
        np.ndarray: (N, 2) plan coordinates with y up.
    """
    out: np.ndarray = xy @ matrix[:2, :2].T + matrix[:2, 2]
    out[:, 1] *= -1.0
    return out


def _shape_geometry(element: ET.Element, matrix: np.ndarray) -> Polygon | None:
    """Geometry of a ``<polygon>`` or ``<circle>`` element in plan coordinates.

    Args:
        element (ET.Element): Shape element.
        matrix (np.ndarray): Cumulative transform of its parent group.

    Returns:
        Polygon | None: Valid polygon, or None if degenerate/unsupported.
    """
    matrix = matrix @ _parse_transform(element.get("transform"))
    if element.tag == SVG_NS + "polygon":
        values: list[float] = [float(v) for v in _NUMBER.findall(element.get("points", ""))]
        if len(values) < 6:
            return None
        shape = Polygon(_apply(matrix, np.array(values[: len(values) // 2 * 2]).reshape(-1, 2)))
    elif element.tag == SVG_NS + "circle":
        centre: np.ndarray = _apply(matrix, np.array([[float(element.get("cx", 0)), float(element.get("cy", 0))]]))[0]
        # Uniform scale factor of the transform applied to the radius.
        radius: float = float(element.get("r", 0)) * math.sqrt(abs(np.linalg.det(matrix[:2, :2])))
        shape = Point(centre).buffer(radius)
    else:
        return None
    shape = shape.buffer(0)  # repairs self-intersections from hand-drawn annotations
    return shape if not shape.is_empty and shape.area > 0 else None


def _first_token(element: ET.Element) -> str:
    """First token of an element's ``class`` attribute ("" if none).

    Args:
        element (ET.Element): XML element.

    Returns:
        str: First class token.
    """
    tokens: list[str] = (element.get("class") or "").split()
    return tokens[0] if tokens else ""


def _walk_groups(element: ET.Element, matrix: np.ndarray) -> Iterator[tuple[ET.Element, np.ndarray]]:
    """Yield every ``<g>`` below ``element`` with its cumulative transform.

    Args:
        element (ET.Element): Root of the walk (not yielded itself).
        matrix (np.ndarray): Cumulative transform of ``element``.

    Yields:
        tuple[ET.Element, np.ndarray]: (group, transform including the group's own).
    """
    for child in element:
        if child.tag == SVG_NS + "g":
            child_matrix: np.ndarray = matrix @ _parse_transform(child.get("transform"))
            yield child, child_matrix
            yield from _walk_groups(child, child_matrix)


def _direct_shape(group: ET.Element, matrix: np.ndarray) -> Polygon | None:
    """Geometry of the first polygon/circle that is a direct child of a group.

    Args:
        group (ET.Element): Element group.
        matrix (np.ndarray): Cumulative transform of the group.

    Returns:
        Polygon | None: Geometry, or None if absent/degenerate.
    """
    for child in group:
        if child.tag in (SVG_NS + "polygon", SVG_NS + "circle"):
            return _shape_geometry(child, matrix)
    return None


def _all_shapes(group: ET.Element, matrix: np.ndarray) -> BaseGeometry | None:
    """Union of every polygon/circle anywhere inside a group (used for stairs).

    Args:
        group (ET.Element): Element group.
        matrix (np.ndarray): Cumulative transform of the group.

    Returns:
        BaseGeometry | None: Union, or None if empty.
    """
    shapes: list[Polygon] = [s for s in (_direct_shape(group, matrix),) if s is not None]
    for sub, sub_matrix in _walk_groups(group, matrix):
        shape: Polygon | None = _direct_shape(sub, sub_matrix)
        if shape is not None:
            shapes.append(shape)
    union: BaseGeometry = unary_union(shapes)
    return None if union.is_empty else union


# ----------------------------------------------------------------------
# Floor extraction
# ----------------------------------------------------------------------

def _room_class(group: ET.Element, cfg: SimpleNamespace, space_map: dict[str, str]) -> str:
    """Our room class for a ``Space <Type> ...`` group.

    Args:
        group (ET.Element): Space group.
        cfg (SimpleNamespace): ``params.cubicasa``.
        space_map (dict[str, str]): First type token -> room class.

    Returns:
        str: Room class.
    """
    tokens: list[str] = (group.get("class") or "").split()
    return space_map.get(tokens[1] if len(tokens) > 1 else "", cfg.UNKNOWN_SPACE_CLASS)


def _floor_elements(floor: ET.Element, matrix: np.ndarray, class_map: dict[str, str], cfg: SimpleNamespace,
                    space_map: dict[str, str]) -> tuple[list[BaseGeometry], list[tuple[str, BaseGeometry]]]:
    """Collect wall shapes and instances of one floor.

    Args:
        floor (ET.Element): Floorplan group.
        matrix (np.ndarray): Cumulative transform of the floor group.
        class_map (dict[str, str]): Element token -> class (non-room elements).
        cfg (SimpleNamespace): ``params.cubicasa``.
        space_map (dict[str, str]): Space type token -> room class.

    Returns:
        tuple[list[BaseGeometry], list[tuple[str, BaseGeometry]]]:
        wall shapes and (class, geometry) instances.
    """
    walls: list[BaseGeometry] = []
    instances: list[tuple[str, BaseGeometry]] = []
    for group, group_matrix in _walk_groups(floor, matrix):
        token: str = _first_token(group)
        if token == SPACE_TOKEN:
            cls: str | None = _room_class(group, cfg, space_map)
        else:
            cls = class_map.get(token)
        if cls is None:
            continue
        geom = _all_shapes(group, group_matrix) if cls == "stair" else _direct_shape(group, group_matrix)
        if geom is None:
            continue
        if cls == "wall":
            walls.append(geom)
        else:
            instances.append((cls, geom))
    return walls, instances


def _median_door_width_m(instances: list[tuple[str, BaseGeometry]], metres_per_unit: float) -> float:
    """Median door width (long side of its min. rotated rectangle) in metres.

    Args:
        instances (list[tuple[str, BaseGeometry]]): Floor instances.
        metres_per_unit (float): Scale.

    Returns:
        float: Median width, NaN if the floor has no measurable door.
    """
    widths: list[float] = []
    for cls, geom in instances:
        rect: BaseGeometry = geom.minimum_rotated_rectangle
        if cls == "door" and isinstance(rect, Polygon) and not rect.is_empty:
            corners: np.ndarray = np.asarray(rect.exterior.coords)[:3]
            widths.append(float(np.hypot(*np.diff(corners, axis=0).T).max()) * metres_per_unit)
    return float(np.median(widths)) if widths else math.nan


def parse_house(svg: bytes, house_id: int, params: SimpleNamespace) -> tuple[list[PlanGeometry], list[dict[str, Any]]]:
    """Convert one CubiCasa house SVG into one ``PlanGeometry`` per usable floor.

    Args:
        svg (bytes): Content of ``model.svg``.
        house_id (int): House id (folder name).
        params (SimpleNamespace): Loaded ``params.toml``.

    Returns:
        tuple[list[PlanGeometry], list[dict[str, Any]]]: Floors, and notes
        (skipped floors / suspicious scales) with ``plan_id`` and ``note``.

    Raises:
        ValueError: On malformed SVG content (caught per house by the caller).
    """
    cfg: SimpleNamespace = params.cubicasa
    class_map: dict[str, str] = dict(vars(params.taxonomy.cubicasa_class_map))
    space_map: dict[str, str] = dict(vars(params.taxonomy.cubicasa_space_map))
    room_classes: set[str] = set(space_map.values()) | {cfg.UNKNOWN_SPACE_CLASS}
    metres_per_unit: float = 1.0 / cfg.SVG_UNITS_PER_METRE
    root: ET.Element = ET.fromstring(svg)
    floors = [(g, m) for g, m in _walk_groups(root, np.eye(3)) if FLOOR_TOKEN in (g.get("class") or "").split()]
    if len(floors) >= cfg.FLOOR_ID_FACTOR:
        raise ValueError(f"house {house_id} has {len(floors)} floors (>= FLOOR_ID_FACTOR)")
    plans: list[PlanGeometry] = []
    notes: list[dict[str, Any]] = []
    for index, (floor, matrix) in enumerate(floors):
        plan_id: int = house_id * cfg.FLOOR_ID_FACTOR + index
        walls, instances = _floor_elements(floor, matrix, class_map, cfg, space_map)
        rooms: int = sum(1 for cls, _ in instances if cls in room_classes)
        if len(walls) < cfg.MIN_WALLS_PER_FLOOR or rooms < cfg.MIN_ROOMS_PER_FLOOR:
            notes.append({"plan_id": plan_id, "note": f"skipped: {len(walls)} walls, {rooms} rooms"})
            continue
        door_width: float = _median_door_width_m(instances, metres_per_unit)
        low, high = cfg.DOOR_WIDTH_RANGE_M
        if not math.isnan(door_width) and not low <= door_width <= high:
            notes.append({"plan_id": plan_id, "note": f"suspicious scale: median door width {door_width:.2f} m"})
        openings: BaseGeometry = unary_union([g for cls, g in instances if cls in OPENING_CLASSES])
        wall: BaseGeometry = unary_union(walls).difference(openings)
        plans.append(PlanGeometry(plan_id, metres_per_unit, wall, instances))
    return plans, notes


def validate_class_maps(params: SimpleNamespace, channels: tuple[str, ...]) -> None:
    """Fail fast if a CubiCasa mapping targets a class missing from the taxonomy.

    Args:
        params (SimpleNamespace): Loaded ``params.toml``.
        channels (tuple[str, ...]): Taxonomy classes.

    Raises:
        ValueError: On unknown target classes.
    """
    targets: set[str] = set(vars(params.taxonomy.cubicasa_class_map).values())
    targets |= set(vars(params.taxonomy.cubicasa_space_map).values()) | {params.cubicasa.UNKNOWN_SPACE_CLASS}
    unknown: set[str] = targets - set(channels)
    if unknown:
        raise ValueError(f"CubiCasa maps to classes not in the taxonomy: {sorted(unknown)}")
