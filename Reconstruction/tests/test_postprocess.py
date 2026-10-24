"""Unit tests for post-processing steps and sample aggregation.

Run from the repository root:
    python -m pytest Reconstruction/tests/test_postprocess.py -q
"""

from pathlib import Path

import numpy as np
import pytest

from Reconstruction.common.config_loader import load_params
from Reconstruction.data_prep.schema import Schema
from Reconstruction.evaluation.labels import EMPTY, LabelMaps
from Reconstruction.postprocess.run import aggregate_samples
from Reconstruction.postprocess.steps import align_walls, close_wall_gaps, openings_on_wall, remove_small, snap_rooms

ROOT: Path = Path(__file__).resolve().parents[1]
SIZE: int = 64


@pytest.fixture(scope="module")
def schema() -> Schema:
    """Project taxonomy."""
    return Schema.from_params(load_params(ROOT / "data_prep" / "params.toml"))


@pytest.fixture(scope="module")
def cfg():
    """Post-processing step parameters."""
    return load_params(ROOT / "postprocess" / "params.toml").steps


def empty_labels() -> LabelMaps:
    """All-empty label maps."""
    return LabelMaps(np.full((SIZE, SIZE), EMPTY, np.int8), np.full((SIZE, SIZE), EMPTY, np.int8))


def test_aggregate_votes() -> None:
    """A channel is on where at least half of the samples agree."""
    samples = np.zeros((3, 1, 2, 2), np.uint8)
    samples[:2, 0, 0, 0] = 255   # 2 of 3 -> on
    samples[:1, 0, 1, 1] = 255   # 1 of 3 -> off
    fused = aggregate_samples(samples, 0.5)
    assert fused[0, 0, 0] == 1 and fused[0, 1, 1] == 0


def test_remove_small_keeps_big(schema, cfg) -> None:
    """Fragments below the minimum area vanish, larger pieces stay."""
    added = empty_labels()
    added.structure[2:4, 2:4] = 0        # 4 px wall
    added.structure[20:23, 10:30] = 0    # 60 px wall
    out = remove_small(added, empty_labels(), schema, cfg.remove_small)
    assert (out.structure[2:4, 2:4] == EMPTY).all() and (out.structure[20:23, 10:30] == 0).all()


def test_floating_door_removed_but_door_on_wall_kept(schema, cfg) -> None:
    """Only the door touching a wall survives."""
    wall, door = schema.structure_classes.index("wall"), schema.structure_classes.index("door")
    base = empty_labels()
    base.structure[30:34, 5:25] = wall
    added = empty_labels()
    added.structure[28:36, 25:28] = door    # touches wall end
    added.structure[50:58, 50:53] = door    # floating
    out = openings_on_wall(added, base, schema, cfg.openings_on_wall)
    assert (out.structure[28:36, 25:28] == door).all() and (out.structure[50:58, 50:53] == EMPTY).all()


def test_wall_gap_bridged_only_next_to_prediction(schema, cfg) -> None:
    """A gap beside a predicted wall closes; an untouched input gap does not."""
    wall = schema.structure_classes.index("wall")
    base = empty_labels()
    base.structure[10:14, 5:20] = wall
    base.structure[10:14, 24:40] = wall     # gap of 4 px at cols 20..23 (closed via prediction below)
    base.structure[40:44, 5:20] = wall
    base.structure[40:44, 24:40] = wall     # untouched gap
    added = empty_labels()
    added.structure[10:14, 20:22] = wall    # prediction narrows the first gap
    out = close_wall_gaps(added, base, schema, cfg.close_wall_gaps)
    assert (out.structure[10:14, 20:24] == wall).all()
    assert (out.structure[40:44, 20:24] == EMPTY).all()


def test_align_walls_fills_bounding_box(schema, cfg) -> None:
    """A wall piece with a notch becomes a clean rectangle."""
    wall = schema.structure_classes.index("wall")
    added = empty_labels()
    added.structure[20:24, 10:40] = wall
    added.structure[21, 25] = EMPTY
    out = align_walls(added, empty_labels(), schema, cfg.align_walls)
    assert (out.structure[20:24, 10:40] == wall).all()


def test_snap_rooms_fills_enclosed_region(schema, cfg) -> None:
    """A roughly predicted room fills the whole walled-in area."""
    wall, bedroom = schema.structure_classes.index("wall"), 1
    base = empty_labels()
    base.structure[10:12, 10:40] = wall
    base.structure[38:40, 10:40] = wall
    base.structure[10:40, 10:12] = wall
    base.structure[10:40, 38:40] = wall
    added = empty_labels()
    added.room[15:30, 15:30] = bedroom      # covers ~half of the interior
    out = snap_rooms(added, base, schema, cfg.snap_rooms)
    assert (out.room[12:38, 12:38] == bedroom).all()


def test_snap_rooms_ignores_exterior(schema, cfg) -> None:
    """Rooms predicted outside any enclosure (border-connected space) are untouched."""
    added = empty_labels()
    added.room[2:8, 2:8] = 1
    out = snap_rooms(added, empty_labels(), schema, cfg.snap_rooms)
    assert np.array_equal(out.room, added.room)
