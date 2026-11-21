"""Rasterize a ``PlanGeometry`` into the two instance layers.

Output per plan (see ``schema.py`` for the layer convention):

* ``room`` layer, uint8 (H, W): room instance ids.
* ``structure`` layer, uint8 (H, W): wall-segment / railing / column /
  window / door ids.
* instance metadata: layer, id, class, pixel area, bounding box and (for
  non-wall instances) the source polygon in pixel coordinates.

Each plan is fitted into the square canvas with a fixed padding, keeping its
aspect ratio, with north up (row index grows as y decreases). The resulting
metres-per-pixel is stored per plan, since plans differ in size.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from Reconstruction.data_prep.geometry import PlanGeometry
from Reconstruction.data_prep.schema import ROOM_LAYER, STRUCTURE_LAYER, Schema
from Reconstruction.data_prep.wall_split import split_walls

# Layer images are uint8, so a layer can hold at most 255 instances.
MAX_INSTANCES_PER_LAYER: int = 255


@dataclass(frozen=True)
class CanvasTransform:
    """Affine map from plan units to canvas pixel coordinates (col, row).

    Attributes:
        scale (float): Pixels per plan unit.
        min_x (float): Plan x mapped to column ``offset_x``.
        max_y (float): Plan y mapped to row ``offset_y`` (north up).
        offset_x (float): Column of ``min_x``.
        offset_y (float): Row of ``max_y``.
    """

    scale: float
    min_x: float
    max_y: float
    offset_x: float
    offset_y: float

    @classmethod
    def fit(cls, bounds: tuple[float, float, float, float], size: int, padding: int,
            fixed_scale: float = 0.0) -> "CanvasTransform":
        """Fit plan bounds into a square canvas, centred, aspect preserved.

        Args:
            bounds (tuple[float, float, float, float]): (min_x, min_y, max_x, max_y).
            size (int): Canvas side in pixels.
            padding (int): Empty border in pixels.
            fixed_scale (float): If > 0, draw every plan at this many pixels per plan
                unit instead of fitting each one (FurnitureArranger: all plans share one
                scale). Default 0 = fit each plan, as for ResPlan / CubiCasa5K.

        Returns:
            CanvasTransform: The fitted transform.

        Raises:
            ValueError: If the bounds are degenerate, or the plan does not fit at ``fixed_scale``.
        """
        min_x, min_y, max_x, max_y = bounds
        extent: float = max(max_x - min_x, max_y - min_y)
        if extent <= 0:
            raise ValueError(f"Degenerate plan bounds {bounds}")
        scale: float = (size - 2 * padding) / extent
        if fixed_scale > 0:
            if extent * fixed_scale > size - 2 * padding + 1e-6:
                raise ValueError(f"Plan extent {extent:.1f} units does not fit the canvas at {fixed_scale} px/unit")
            scale = fixed_scale
        # Centre the shorter side inside the padded area.
        offset_x: float = padding + (size - 2 * padding - (max_x - min_x) * scale) / 2
        offset_y: float = padding + (size - 2 * padding - (max_y - min_y) * scale) / 2
        return cls(scale, min_x, max_y, offset_x, offset_y)

    def upscaled(self, factor: int) -> "CanvasTransform":
        """Same framing on a canvas ``factor`` times larger.

        Args:
            factor (int): Integer upscale factor.

        Returns:
            CanvasTransform: Scaled transform.
        """
        return CanvasTransform(self.scale * factor, self.min_x, self.max_y, self.offset_x * factor, self.offset_y * factor)

    def to_pixels(self, coords: np.ndarray) -> np.ndarray:
        """Map (N, 2) plan coordinates (x, y) to (N, 2) pixel coords (col, row).

        Args:
            coords (np.ndarray): Plan coordinates.

        Returns:
            np.ndarray: float64 pixel coordinates.
        """
        cols: np.ndarray = (coords[:, 0] - self.min_x) * self.scale + self.offset_x
        rows: np.ndarray = (self.max_y - coords[:, 1]) * self.scale + self.offset_y
        return np.stack([cols, rows], axis=1)


@dataclass
class RasterPlan:
    """Rasterized plan: two instance layers plus per-instance metadata.

    Attributes:
        plan_id (int): ResPlan plan id.
        room (np.ndarray): uint8 (H, W) room instance ids.
        structure (np.ndarray): uint8 (H, W) structure instance ids.
        instances (list[dict[str, Any]]): One dict per instance with keys
            layer, id, cls, area_px, bbox [r0, c0, r1, c1) and polygon
            (pixel coords, or None for wall segments).
        metres_per_px (float): Metric size of one pixel (NaN if unknown).
    """

    plan_id: int
    room: np.ndarray
    structure: np.ndarray
    instances: list[dict[str, Any]]
    metres_per_px: float


def _polygon_parts(geom: BaseGeometry) -> list[Polygon]:
    """Non-empty polygon parts of a geometry (other geometry types are ignored).

    Args:
        geom (BaseGeometry): Polygon, MultiPolygon or collection.

    Returns:
        list[Polygon]: Polygon parts.
    """
    parts = geom.geoms if hasattr(geom, "geoms") else [geom]
    return [part for part in parts if isinstance(part, Polygon) and not part.is_empty]


def _fill_geometry(canvas: np.ndarray, geom: BaseGeometry, transform: CanvasTransform, value: int, shift_bits: int) -> None:
    """Paint every polygon part (holes cleared back to 0) into ``canvas`` in place.

    Args:
        canvas (np.ndarray): int32 image being painted.
        geom (BaseGeometry): (Multi)polygon in plan units.
        transform (CanvasTransform): Plan -> pixel transform.
        value (int): Value written inside the polygons.
        shift_bits (int): Fixed-point bits for sub-pixel accuracy.
    """
    factor: int = 1 << shift_bits

    def to_int(ring) -> np.ndarray:
        return np.round(transform.to_pixels(np.asarray(ring.coords)) * factor).astype(np.int32)

    for polygon in _polygon_parts(geom):
        cv2.fillPoly(canvas, [to_int(polygon.exterior)], int(value), lineType=cv2.LINE_8, shift=shift_bits)
        holes: list[np.ndarray] = [to_int(ring) for ring in polygon.interiors]
        if holes:
            cv2.fillPoly(canvas, holes, 0, lineType=cv2.LINE_8, shift=shift_bits)


def _rasterize_walls(geometry: PlanGeometry, transform: CanvasTransform, size: int, params: SimpleNamespace) -> np.ndarray:
    """Rasterize the wall union and split it into segment instances.

    Args:
        geometry (PlanGeometry): Plan geometry.
        transform (CanvasTransform): Plan -> canvas transform.
        size (int): Canvas size.
        params (SimpleNamespace): Loaded ``params.toml``.

    Returns:
        np.ndarray: int32 (size, size) wall segment labels (0 = no wall).
    """
    upscale: int = params.wall_split.UPSCALE
    shift: int = params.canvas.FILL_SHIFT_BITS
    mask: np.ndarray = np.zeros((size, size), np.int32)
    mask_up: np.ndarray = np.zeros((size * upscale, size * upscale), np.int32)
    _fill_geometry(mask, geometry.wall, transform, 1, shift)
    _fill_geometry(mask_up, geometry.wall, transform.upscaled(upscale), 1, shift)
    return split_walls(mask_up > 0, mask > 0, upscale,
                       params.wall_split.JUNCTION_CLEAR_RADIUS_PX, params.wall_split.MIN_SEGMENT_LENGTH_PX)


def _finalize_layer(provisional: np.ndarray, classes: dict[int, str], polygons: dict[int, list[list[float]] | None],
                    layer: str, min_area: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Drop tiny instances, relabel ids to 1..K and build metadata.

    Args:
        provisional (np.ndarray): int32 layer with provisional ids.
        classes (dict[int, str]): Provisional id -> class.
        polygons (dict[int, list[list[float]] | None]): Provisional id -> pixel polygon.
        layer (str): Layer name.
        min_area (int): Minimum pixel area to keep an instance.

    Returns:
        tuple[np.ndarray, list[dict[str, Any]]]: uint8 layer and instance metadata.

    Raises:
        ValueError: If more than ``MAX_INSTANCES_PER_LAYER`` instances survive.
    """
    areas: np.ndarray = np.bincount(provisional.ravel(), minlength=max(classes, default=0) + 1)
    kept: list[int] = [pid for pid in sorted(classes) if areas[pid] >= min_area]
    if len(kept) > MAX_INSTANCES_PER_LAYER:
        raise ValueError(f"{layer} layer has {len(kept)} instances (> {MAX_INSTANCES_PER_LAYER})")
    lut: np.ndarray = np.zeros(areas.size, dtype=np.uint8)
    metadata: list[dict[str, Any]] = []
    for new_id, pid in enumerate(kept, start=1):
        lut[pid] = new_id
        rows, cols = np.nonzero(provisional == pid)
        metadata.append({
            "layer": layer, "id": new_id, "cls": classes[pid], "area_px": int(areas[pid]),
            "bbox": [int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1],
            "polygon": polygons.get(pid),
        })
    return lut[provisional], metadata


def _pixel_polygon(geom: BaseGeometry, transform: CanvasTransform) -> list[list[float]] | None:
    """Exterior ring of the largest polygon part in pixel coordinates (for overlays).

    Args:
        geom (BaseGeometry): (Multi)polygon in plan units.
        transform (CanvasTransform): Plan -> pixel transform.

    Returns:
        list[list[float]] | None: [[col, row], ...] rounded to 0.01 px, or
        None if the geometry has no polygon part.
    """
    parts: list[Polygon] = _polygon_parts(geom)
    if not parts:
        return None
    largest: Polygon = max(parts, key=lambda part: part.area)
    return np.round(transform.to_pixels(np.asarray(largest.exterior.coords)), 2).tolist()


def rasterize_plan(geometry: PlanGeometry, schema: Schema, params: SimpleNamespace) -> RasterPlan:
    """Rasterize one plan into room and structure instance layers.

    Drawing order: rooms largest-first (smaller rooms win overlaps), then wall
    segments, then the remaining structure classes in taxonomy order, so
    openings overwrite walls where they touch.

    Args:
        geometry (PlanGeometry): Plan in our taxonomy.
        schema (Schema): Class taxonomy.
        params (SimpleNamespace): Loaded ``params.toml``.

    Returns:
        RasterPlan: Rasterized plan.
    """
    size: int = params.canvas.SIZE
    shift: int = params.canvas.FILL_SHIFT_BITS
    transform = CanvasTransform.fit(geometry.all_geometry().bounds, size, params.canvas.PADDING_PX,
                                    getattr(params.canvas, "FIXED_SCALE_PX_PER_UNIT", 0.0))

    room: np.ndarray = np.zeros((size, size), np.int32)
    room_classes: dict[int, str] = {}
    room_polys: dict[int, list[list[float]] | None] = {}
    room_items = [(c, p) for c, p in geometry.instances if schema.layer_of(c) == ROOM_LAYER]
    for pid, (cls, geom) in enumerate(sorted(room_items, key=lambda item: -item[1].area), start=1):
        _fill_geometry(room, geom, transform, pid, shift)
        room_classes[pid], room_polys[pid] = cls, _pixel_polygon(geom, transform)

    structure: np.ndarray = _rasterize_walls(geometry, transform, size, params)
    struct_classes: dict[int, str] = {int(i): "wall" for i in np.unique(structure) if i > 0}
    struct_polys: dict[int, list[list[float]] | None] = {pid: None for pid in struct_classes}
    next_id: int = max(struct_classes, default=0) + 1
    for cls in (c for c in schema.structure_classes if c != "wall"):
        for geom in (g for c, g in geometry.instances if c == cls):
            _fill_geometry(structure, geom, transform, next_id, shift)
            struct_classes[next_id], struct_polys[next_id] = cls, _pixel_polygon(geom, transform)
            next_id += 1

    min_area: int = params.canvas.MIN_INSTANCE_AREA_PX
    room_layer, room_meta = _finalize_layer(room, room_classes, room_polys, ROOM_LAYER, min_area)
    struct_layer, struct_meta = _finalize_layer(structure, struct_classes, struct_polys, STRUCTURE_LAYER, min_area)
    return RasterPlan(geometry.plan_id, room_layer, struct_layer, struct_meta + room_meta,
                      geometry.metres_per_unit / transform.scale)
