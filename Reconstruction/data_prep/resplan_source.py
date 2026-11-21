"""Read-only access to the original ResPlan release.

The pickle and ``split.json`` are only ever opened for reading; nothing here
writes next to the original data.

ResPlan facts this module relies on (checked on the release file):

* Each plan is a dict of Shapely geometries keyed by category, plus ``id``,
  ``area`` (floor area in m^2) and ``inner`` (the plan footprint).
* Coordinates are *normalized per plan* to roughly a 256-unit canvas (not
  metres), so metric scale is recovered from ``area`` / ``inner.area``.
* Walls are a union of pieces that stop at door/window openings; openings
  touch walls but do not overlap them.
"""

import json
import math
import pickle
from pathlib import Path
from typing import Any

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from Reconstruction.data_prep.geometry import SPLIT_NAMES, PlanGeometry

# Key of the augmented-copies list in split.json.
AUGMENTED_KEY: str = "augmented"
# ResPlan's known key typo, normalized on load.
_TYPO_KEYS: dict[str, str] = {"balacony": "balcony"}


def load_resplan_plans(pickle_path: Path) -> list[dict[str, Any]]:
    """Load the ResPlan pickle (read-only).

    Args:
        pickle_path (Path): Path to ``ResPlan.pkl``.

    Returns:
        list[dict[str, Any]]: Raw plan dicts as stored in the release.

    Raises:
        FileNotFoundError: If the pickle is missing.
    """
    if not pickle_path.is_file():
        raise FileNotFoundError(f"ResPlan pickle not found: {pickle_path}")
    with open(pickle_path, "rb") as handle:
        plans: list[dict[str, Any]] = pickle.load(handle)
    return plans


def load_split_ids(split_path: Path, include_augmented_in_train: bool) -> dict[str, list[int]]:
    """Load the official train/val/test plan ids.

    Augmented plans are rotations/flips/scalings of other plans. Keeping them
    out of training by default avoids leaking copies of val/test originals.

    Args:
        split_path (Path): Path to ``split.json``.
        include_augmented_in_train (bool): Append ``augmented`` ids to train.

    Returns:
        dict[str, list[int]]: Split name -> sorted plan ids.

    Raises:
        ValueError: If the splits overlap.
    """
    with open(split_path, "r", encoding="utf-8") as handle:
        raw: dict[str, list[int]] = json.load(handle)
    splits: dict[str, list[int]] = {name: sorted(raw[name]) for name in SPLIT_NAMES}
    if include_augmented_in_train:
        splits["train"] = sorted(set(splits["train"]) | set(raw[AUGMENTED_KEY]))
    for i, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[i + 1:]:
            shared: set[int] = set(splits[first]) & set(splits[second])
            if shared:
                raise ValueError(f"Splits {first}/{second} share {len(shared)} plan ids")
    return splits


def _polygon_parts(geom: BaseGeometry | None) -> list[Polygon]:
    """Split a (Multi)Polygon into non-empty polygons; other types are ignored.

    Args:
        geom (BaseGeometry | None): Geometry from a plan dict.

    Returns:
        list[Polygon]: Individual non-empty polygons.
    """
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [part for part in geom.geoms if not part.is_empty]
    # GeometryCollections can hold stray points/lines; keep polygonal parts only.
    return [part for part in getattr(geom, "geoms", []) if isinstance(part, Polygon) and not part.is_empty]


def _lookup_key(plan: dict[str, Any], key: str) -> BaseGeometry | None:
    """Fetch a category from a plan, accepting ResPlan's known key typos.

    Args:
        plan (dict[str, Any]): Raw ResPlan plan.
        key (str): Canonical category key (e.g. "balcony").

    Returns:
        BaseGeometry | None: The geometry, or None if the plan lacks the key.
    """
    if key in plan:
        return plan[key]
    for typo, canonical in _TYPO_KEYS.items():
        if canonical == key and typo in plan:
            return plan[typo]
    return None


def extract_plan_geometry(plan: dict[str, Any], class_map: dict[str, str]) -> PlanGeometry:
    """Convert a raw ResPlan dict into a ``PlanGeometry`` in our taxonomy.

    The input dict is not modified.

    Args:
        plan (dict[str, Any]): Raw ResPlan plan.
        class_map (dict[str, str]): ResPlan key -> our class name.

    Returns:
        PlanGeometry: Walls as one union plus one instance per polygon part.
    """
    wall_parts: list[Polygon] = []
    instances: list[tuple[str, Polygon]] = []
    for source_key, class_name in class_map.items():
        parts: list[Polygon] = _polygon_parts(_lookup_key(plan, source_key))
        if class_name == "wall":
            wall_parts.extend(parts)
        else:
            instances.extend((class_name, part) for part in parts)
    return PlanGeometry(
        plan_id=int(plan["id"]),
        metres_per_unit=_metres_per_unit(plan),
        wall=unary_union(wall_parts),
        instances=instances,
    )


def _metres_per_unit(plan: dict[str, Any]) -> float:
    """Metric scale of a plan's (per-plan normalized) coordinates.

    Args:
        plan (dict[str, Any]): Raw ResPlan plan with ``area`` (m^2) and ``inner``.

    Returns:
        float: sqrt(area_m2 / footprint area in units^2), or NaN if unavailable.
    """
    area_m2: float = float(plan.get("area") or 0.0)
    inner: BaseGeometry | None = plan.get("inner")
    footprint: float = float(inner.area) if inner is not None else 0.0
    if area_m2 <= 0 or footprint <= 0:
        return math.nan
    return math.sqrt(area_m2 / footprint)
