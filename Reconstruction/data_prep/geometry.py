"""Dataset-neutral vector representation of one floor plan.

Every dataset source (ResPlan, CubiCasa5K) converts its raw format into
``PlanGeometry``; rasterization and everything downstream only see this type.
Coordinates are in the source's own units with y pointing up.
"""

import math
from dataclasses import dataclass, field

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# Split names shared by all datasets.
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


@dataclass
class PlanGeometry:
    """Vector content of one plan (one floor) in our taxonomy.

    Attributes:
        plan_id (int): Unique plan id within its dataset.
        metres_per_unit (float): Metric size of one coordinate unit (NaN if unknown).
        wall (BaseGeometry): Union of all walls, with door/window openings
            already cut out (openings are separate instances).
        instances (list[tuple[str, Polygon | MultiPolygon]]): Non-wall
            instances as (class name, geometry), one per room / door / window /
            railing / column / stair.
    """

    plan_id: int
    metres_per_unit: float
    wall: BaseGeometry
    instances: list[tuple[str, Polygon | MultiPolygon]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Fail fast on a non-positive scale (NaN is allowed: unknown).

        Raises:
            ValueError: If ``metres_per_unit`` is zero or negative.
        """
        if not math.isnan(self.metres_per_unit) and self.metres_per_unit <= 0:
            raise ValueError(f"Plan {self.plan_id}: metres_per_unit must be positive, got {self.metres_per_unit}")

    def all_geometry(self) -> BaseGeometry:
        """Union of every rasterized geometry, used to fit the canvas.

        Returns:
            BaseGeometry: Union of walls and all instances.
        """
        return unary_union([self.wall] + [geom for _, geom in self.instances])
