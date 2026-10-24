"""Unit tests for data preparation (rasterization, wall splitting, case generation).

Uses a tiny synthetic plan so the tests run without the ResPlan release:

    +-----------win----------+
    |  bedroom  |   living   |
    |          door          |
    |           |            |
    +------------------------+

Run from the repository root:
    python -m pytest Reconstruction/tests -q
"""

import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

from Reconstruction.common.config_loader import load_params
from Reconstruction.data_prep.dedup import find_near_duplicates
from Reconstruction.data_prep.missing import EASY, HARD, CaseSpec, MissingComponentGenerator
from Reconstruction.data_prep.plan_store import PlanStore, write_split
from Reconstruction.data_prep.rasterize import rasterize_plan
from Reconstruction.data_prep.cubicasa_source import parse_house
from Reconstruction.data_prep.geometry import PlanGeometry
from Reconstruction.data_prep.resplan_source import load_split_ids
from Reconstruction.data_prep.schema import Schema
from Reconstruction.data_prep.wall_split import split_walls

PARAMS_PATH: Path = Path(__file__).resolve().parents[1] / "data_prep" / "params.toml"


@pytest.fixture(scope="module")
def params():
    """Project parameters."""
    return load_params(PARAMS_PATH)


@pytest.fixture(scope="module")
def schema(params) -> Schema:
    """Project taxonomy."""
    return Schema.from_params(params)


@pytest.fixture(scope="module")
def synthetic_geometry() -> PlanGeometry:
    """Two rooms, a shared wall with a door and a window in the top wall."""
    door = box(48, 25, 52, 35)
    window = box(20, 56, 35, 60)
    walls = box(0, 0, 100, 60).difference(box(4, 4, 96, 56)).union(box(48, 4, 52, 56))
    walls = walls.difference(door).difference(window)
    instances = [("bedroom", box(4, 4, 48, 56)), ("living", box(52, 4, 96, 56)), ("door", door), ("window", window)]
    return PlanGeometry(plan_id=7, metres_per_unit=0.1, wall=walls, instances=instances)


@pytest.fixture(scope="module")
def store(tmp_path_factory, synthetic_geometry, schema, params) -> PlanStore:
    """A one-plan store built from the synthetic geometry."""
    split_dir: Path = tmp_path_factory.mktemp("processed") / "test"
    write_split(split_dir, [rasterize_plan(synthetic_geometry, schema, params)])
    return PlanStore(split_dir)


@pytest.fixture(scope="module")
def generator(schema, params) -> MissingComponentGenerator:
    """Generator without size bins."""
    return MissingComponentGenerator(schema, params, size_bins=None)


def _spec(store: PlanStore, cls: str, mode: str, margin: int = 0) -> CaseSpec:
    """Spec removing the first instance of ``cls`` from the synthetic plan."""
    inst = next(i for i in store.record(0)["instances"] if i["cls"] == cls)
    ctype = {"bedroom": "room", "living": "room"}.get(cls, cls)
    return CaseSpec("test", 0, 7, ctype, inst["layer"], inst["id"], mode, margin)


def test_split_walls_cuts_t_junction_into_three_segments():
    # Arrange: a T made of 4-px-thick bars on a 64 px canvas.
    mask = np.zeros((64, 64), bool)
    mask[10:14, 4:60] = True
    mask[10:60, 30:34] = True
    # Act
    labels = split_walls(mask.repeat(2, 0).repeat(2, 1), mask, upscale=2, junction_radius_px=3, min_segment_length_px=6)
    # Assert
    assert len(np.unique(labels[mask])) == 3
    assert (labels[~mask] == 0).all()


def test_rasterize_keeps_every_instance_and_metric_scale(store):
    record = store.record(0)
    counts = {c: sum(i["cls"] == c for i in record["instances"]) for c in ("bedroom", "living", "door", "window", "wall")}
    assert counts["bedroom"] == counts["living"] == counts["door"] == counts["window"] == 1
    assert counts["wall"] >= 2
    assert record["metres_per_px"] > 0


@pytest.mark.parametrize("cls,mode", [("wall", EASY), ("wall", HARD), ("door", EASY), ("door", HARD),
                                      ("window", EASY), ("bedroom", EASY), ("bedroom", HARD)])
def test_case_invariants(store, generator, schema, cls, mode):
    # Act
    case = generator.build_case(store.layers(0), store.record(0), _spec(store, cls, mode))
    # Assert: target is exactly what is missing, and it is not empty.
    assert np.array_equal(case.target, case.full & (1 - case.incomplete))
    assert case.target.any()
    # Assert: nothing changes outside the erased region (structure preserved).
    outside = case.mask == 0
    assert np.array_equal(case.full[:, outside], case.incomplete[:, outside])
    # Assert: removal only deletes, except the wall that closes a hard-mode opening.
    added = case.incomplete & (1 - case.full)
    if not (mode == HARD and cls == "door"):
        assert not added.any()
    else:
        assert np.array_equal(added.any(0), added[schema.channel_index("wall")] > 0)


def test_hard_door_is_closed_with_wall(store, generator, schema):
    case = generator.build_case(store.layers(0), store.record(0), _spec(store, "door", HARD))
    door_pixels = case.full[schema.channel_index("door")] > 0
    assert (case.incomplete[schema.channel_index("wall")][door_pixels] == 1).all()
    assert case.info["removed"][0]["cls"] == "door"


def test_hard_wall_also_removes_its_openings(store, generator):
    # The interior wall segments touch the door; removing one in hard mode takes the door too.
    record = store.record(0)
    layers = store.layers(0)
    door_id = next(i["id"] for i in record["instances"] if i["cls"] == "door")
    specs = [CaseSpec("test", 0, 7, "wall", i["layer"], i["id"], HARD, 0) for i in record["instances"] if i["cls"] == "wall"]
    removed_sets = [generator.build_case(layers, record, s).info["removed"] for s in specs]
    assert any(any(r["cls"] == "door" and r["id"] == door_id for r in removed) for removed in removed_sets)


def test_margin_grows_erased_region(store, generator):
    small = generator.build_case(store.layers(0), store.record(0), _spec(store, "door", EASY, 0))
    large = generator.build_case(store.layers(0), store.record(0), _spec(store, "door", EASY, 8))
    assert large.mask.sum() > small.mask.sum()
    assert (large.mask >= small.mask).all()


def test_eval_specs_are_deterministic_and_cover_types(store, generator):
    first = generator.enumerate_eval_specs("test", 0, store.record(0), seed=3)
    second = generator.enumerate_eval_specs("test", 0, store.record(0), seed=3)
    assert first == second
    assert {s.comp_type for s in first} >= {"wall", "door", "window", "room"}


def test_case_spec_round_trip():
    spec = CaseSpec("val", 3, 11, "door", "structure", 5, HARD, 4)
    assert CaseSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_overlapping_splits_are_rejected(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": [1, 2], "val": [2], "test": [3], "augmented": []}))
    with pytest.raises(ValueError):
        load_split_ids(path, include_augmented_in_train=False)


def test_dedup_flags_identical_plan_only():
    rng = np.random.default_rng(0)
    train = rng.integers(0, 5, size=(3, 16, 16), dtype=np.uint8)
    evals = np.stack([train[1], rng.integers(0, 5, size=(16, 16), dtype=np.uint8)])
    found = find_near_duplicates(train, evals, threshold=0.95)
    assert [(d["eval_index"], d["train_index"]) for d in found] == [(0, 1)]


# ----------------------------------------------------------------------
# CubiCasa SVG parsing
# ----------------------------------------------------------------------

# Two floors: floor 0 has a room, walls with a door cut into the right wall
# and a column; floor 1 is empty and must be skipped.
_CUBICASA_SVG: bytes = b"""<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg"><g id="Model" class="Model v1-1">
 <g class="Floor"><g id="Floor-1" transform="matrix(1,0,0,1,0,0)" class="Floorplan Floor-1">
  <g class="Space Bedroom"><polygon points="10,10 390,10 390,290 10,290"/>
    <g class="Dimension"><polygon points="-5,5,0,0,5,5"/></g></g>
  <g class="Wall"><polygon points="0,0 400,0 400,10 0,10"/></g>
  <g class="Wall"><polygon points="0,290 400,290 400,300 0,300"/></g>
  <g class="Wall"><polygon points="0,0 10,0 10,300 0,300"/></g>
  <g class="Wall"><polygon points="390,0 400,0 400,300 390,300"/>
    <g class="Door Swing"><polygon points="390,100 400,100 400,180 390,180"/><g class="Panel"><polygon points="1,1 2,2 3,1"/></g></g></g>
  <g class="Column" transform="translate(200,150)"><circle cx="0" cy="0" r="8"/></g>
  <g class="FixedFurniture Bed" transform="matrix(1,0,0,1,50,50)"><polygon points="0,0 90,0 90,200 0,200"/></g>
 </g>
 <g id="Floor-2" class="Floorplan Floor-2"></g></g>
</g></svg>"""


def test_cubicasa_parse_floor_classes_scale_and_openings(params):
    # Act
    plans, notes = parse_house(_CUBICASA_SVG, house_id=42, params=params)
    # Assert: one usable floor, the empty one is reported as skipped.
    assert [p.plan_id for p in plans] == [420]
    assert len(notes) == 1 and notes[0]["plan_id"] == 421 and notes[0]["note"].startswith("skipped")
    plan = plans[0]
    classes = sorted(cls for cls, _ in plan.instances)
    assert classes == ["bedroom", "column", "door"]  # furniture and dimension marks ignored
    assert plan.metres_per_unit == pytest.approx(0.01)
    # Door opening is cut out of the wall; walls keep the rest (4 walls, 10 units thick).
    door = next(g for cls, g in plan.instances if cls == "door")
    assert plan.wall.intersection(door).area == pytest.approx(0.0, abs=1e-6)
    assert plan.wall.area == pytest.approx(400 * 10 * 2 + 280 * 10 * 2 - 80 * 10)


def test_cubicasa_y_axis_is_flipped_to_north_up(params):
    plans, _ = parse_house(_CUBICASA_SVG, house_id=42, params=params)
    column = next(g for cls, g in plans[0].instances if cls == "column")
    # SVG (200, 150) -> plan (200, -150): translate() applied, y flipped.
    assert column.centroid.x == pytest.approx(200, abs=0.5)
    assert column.centroid.y == pytest.approx(-150, abs=0.5)


def test_cubicasa_floor_rasterizes(params, schema):
    plans, _ = parse_house(_CUBICASA_SVG, house_id=42, params=params)
    raster = rasterize_plan(plans[0], schema, params)
    counts = {c: sum(i["cls"] == c for i in raster.instances) for c in ("bedroom", "door", "column", "wall")}
    assert counts["bedroom"] == counts["door"] == counts["column"] == 1
    # The wall ring opened by the door has corners but no T-junctions, so it
    # stays one C-shaped segment (corners never split walls).
    assert counts["wall"] == 1
