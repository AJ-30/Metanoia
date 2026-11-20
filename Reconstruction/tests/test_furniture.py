"""Tests for the FurnitureArranger pipeline (synthetic layouts, CPU, no dataset needed).

Run from the repository root:
    python -m pytest Reconstruction/tests/test_furniture.py -q
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from Reconstruction.common.config_loader import load_params
from Reconstruction.data_prep.furniture_source import (assign_splits, furniture_polygon, group_key, layout_to_geometry,
                                                       read_layout)
from Reconstruction.data_prep.missing import EASY, HARD, MissingComponentGenerator
from Reconstruction.data_prep.plan_store import PlanStore, write_split
from Reconstruction.data_prep.rasterize import rasterize_plan
from Reconstruction.data_prep.schema import Schema
from Reconstruction.furniture_model.objective import compute_loss
from Reconstruction.furniture_model.train import check_inputs
from Reconstruction.model.diffusion import GaussianDiffusion
from Reconstruction.model.unet import ConditionalUNet

ROOT: Path = Path(__file__).resolve().parents[1]
PARAMS_PATH: Path = ROOT / "data_prep" / "params_furniture.toml"
HYPER_PATH: Path = ROOT / "furniture_model" / "hyperparameters.toml"

# A 100 x 80 room: walls with a door gap (bottom) and a window gap (top), two furniture items.
# The wall MULTIPOLYGON deliberately contains overlapping pieces (invalid geometry, as in the real data).
FLOORPLAN = {
    "wall": "MULTIPOLYGON (((0 0, 0 80, 5 80, 5 0, 0 0)), ((0 80, 0 85, 100 85, 100 80, 0 80)), ((95 0, 95 80, 100 80, 100 0, 95 0)), "
            "((0 -5, 0 0, 40 0, 40 -5, 0 -5)), ((60 -5, 60 0, 100 0, 100 -5, 60 -5)), ((0 -5, 0 3, 3 3, 3 -5, 0 -5)))",
    "door": "MULTIPOLYGON (((40 -5, 40 0, 60 0, 60 -5, 40 -5)))",
    "front_door": "MULTIPOLYGON EMPTY",
    "window": "MULTIPOLYGON (((30 80, 30 85, 70 85, 70 80, 30 80)))",
    "living": "MULTIPOLYGON (((5 0, 5 80, 95 80, 95 0, 5 0)))",
    "balcony": "MULTIPOLYGON EMPTY", "bathroom": "MULTIPOLYGON EMPTY", "bedroom": "MULTIPOLYGON EMPTY", "kitchen": "MULTIPOLYGON EMPTY",
    "id": 14433, "wall_depth": 5,
}
FURNITURE = [
    {"type": "tv", "x": 50, "y": 70, "angle": 0, "bbox_size_x": 40, "bbox_size_y": 20},
    {"type": "sofa", "x": 15, "y": 40, "angle": 270, "bbox_size_x": 80, "bbox_size_y": 20},
    {"type": "chair", "x": 70, "y": 30, "angle": 45, "bbox_size_x": 20, "bbox_size_y": 20},
]


@pytest.fixture(scope="module")
def params():
    return load_params(PARAMS_PATH)


@pytest.fixture(scope="module")
def schema(params) -> Schema:
    return Schema.from_params(params)


@pytest.fixture(scope="module")
def geometry(params):
    return layout_to_geometry(1, {"floorplan": FLOORPLAN, "furniture": FURNITURE}, params)


@pytest.fixture(scope="module")
def raster(geometry, schema, params):
    return rasterize_plan(geometry, schema, params)


def test_taxonomy_has_furniture_classes(schema, params):
    assert schema.num_channels == 22
    furniture = [c for c in schema.channels if schema.component_type[c] == "furniture"]
    assert len(furniture) == 9
    # furniture must be the lowest-priority structure classes (composition: later = higher)
    assert schema.structure_classes[:9] == tuple(furniture)
    assert "furniture" in params.missing.COMPONENT_TYPES


def test_furniture_polygon_size_and_rotation():
    box0 = furniture_polygon({"x": 50, "y": 50, "angle": 0, "bbox_size_x": 40, "bbox_size_y": 20})
    assert box0.bounds == (30, 40, 70, 60)
    box90 = furniture_polygon({"x": 50, "y": 50, "angle": 90, "bbox_size_x": 40, "bbox_size_y": 20})
    assert tuple(round(v, 6) for v in box90.bounds) == (40, 30, 60, 70)
    assert math.isclose(furniture_polygon({"x": 0, "y": 0, "angle": 45, "bbox_size_x": 20, "bbox_size_y": 20}).area, 400)


def test_invalid_walls_are_repaired(geometry):
    assert geometry.wall.is_valid and not geometry.wall.is_empty
    classes = sorted(c for c, _ in geometry.instances)
    assert classes == ["chair", "door", "living", "sofa", "tv", "window"]


def test_unknown_furniture_type_fails(params):
    bad = [{"type": "bathtub", "x": 1, "y": 1, "angle": 0, "bbox_size_x": 1, "bbox_size_y": 1}]
    with pytest.raises(ValueError, match="bathtub"):
        layout_to_geometry(1, {"floorplan": FLOORPLAN, "furniture": bad}, params)


def test_fixed_scale_rasterization(raster, schema, params):
    assert raster.metres_per_px == pytest.approx(params.furniture.METRES_PER_UNIT)  # 1 px per unit
    by_cls = {i["cls"]: i for i in raster.instances}
    # fillPoly includes boundary pixels, so areas are within one pixel of the exact unit area
    for cls, (w, h) in (("tv", (40, 20)), ("sofa", (80, 20)), ("door", (20, 5)), ("window", (40, 5))):
        assert w * h <= by_cls[cls]["area_px"] <= (w + 1) * (h + 1), cls
    # furniture lives in the structure layer, rooms in the room layer
    assert by_cls["tv"]["layer"] == "structure" and by_cls["living"]["layer"] == "room"


def test_plan_too_large_for_fixed_scale(schema, params):
    huge = dict(FLOORPLAN, wall="MULTIPOLYGON (((0 0, 0 5, 300 5, 300 0, 0 0)))")
    geometry = layout_to_geometry(2, {"floorplan": huge, "furniture": []}, params)
    with pytest.raises(ValueError, match="does not fit"):
        rasterize_plan(geometry, schema, params)


@pytest.fixture(scope="module")
def store(tmp_path_factory, raster):
    split_dir = tmp_path_factory.mktemp("processed") / "train"
    write_split(split_dir, [raster])
    return PlanStore(split_dir)


@pytest.fixture(scope="module")
def generator(schema, params):
    return MissingComponentGenerator(schema, params, size_bins=None)


def test_eval_cases_cover_types_and_furniture_instances(generator, store):
    specs = generator.enumerate_eval_specs("train", 0, store.record(0), seed=1)
    types = {s.comp_type for s in specs}
    assert types == {"wall", "door", "window", "room", "furniture"}
    furniture_instances = {(s.layer, s.instance_id) for s in specs if s.comp_type == "furniture"}
    assert len(furniture_instances) == 3  # instances_per_type.furniture = 3 (all three exist here)
    assert not any(s.mode == HARD for s in specs if s.comp_type in ("room", "furniture"))
    assert any(s.mode == HARD for s in specs if s.comp_type == "wall")


def test_furniture_case_removes_exactly_that_piece(generator, store, schema):
    record = store.record(0)
    spec = next(s for s in generator.enumerate_eval_specs("train", 0, record, seed=1) if s.comp_type == "furniture" and s.margin_px == 0)
    case = generator.build_case(store.layers(0), record, spec)
    removed = {schema.channels[c] for c in np.nonzero(case.target.any(axis=(1, 2)))[0]}
    assert len(removed) == 1 and schema.component_type[removed.pop()] == "furniture"
    assert case.info["comp_type"] == "furniture" and case.info["mode"] == EASY
    assert case.target.sum() == case.primary.sum() > 0
    assert (case.full == (case.incomplete | case.target)).all()


def test_train_type_weights_are_used(generator, store):
    rng = np.random.default_rng(0)
    counts = {}
    for _ in range(3000):
        spec = generator.sample_train_spec("train", 0, store.record(0), rng)
        counts[spec.comp_type] = counts.get(spec.comp_type, 0) + 1
    shares = {k: v / 3000 for k, v in counts.items()}
    assert shares["furniture"] == pytest.approx(0.45, abs=0.04)
    assert shares["room"] == pytest.approx(0.05, abs=0.02)
    assert not any(s == "other" for s in shares)


def test_group_split_is_deterministic_and_keeps_groups_together(params):
    layouts = [{"floorplan": dict(FLOORPLAN, id=i), "furniture": []} for i in range(3)]  # same geometry, different id
    layouts += [{"floorplan": dict(FLOORPLAN, wall="MULTIPOLYGON (((0 0, 0 9, 9 9, 9 0, 0 0)))", id=9), "furniture": []}]
    assert group_key(layouts[0]["floorplan"], "s") == group_key(layouts[2]["floorplan"], "s")
    splits = assign_splits([Path(f"{i}.svg") for i in range(4)], layouts, params.furniture)
    where = {pid: name for name, ids in splits.items() for pid in ids}
    assert where[1] == where[2] == where[3] and set(where) == {1, 2, 3, 4}
    assert splits == assign_splits([Path(f"{i}.svg") for i in range(4)], layouts, params.furniture)


def test_split_fractions_on_many_groups(params):
    layouts = [{"floorplan": dict(FLOORPLAN, wall=f"MULTIPOLYGON (((0 0, 0 {i + 5}, 9 {i + 5}, 9 0, 0 0)))"), "furniture": []} for i in range(3000)]
    splits = assign_splits([Path("x")] * 3000, layouts, params.furniture)
    assert len(splits["val"]) / 3000 == pytest.approx(params.furniture.VAL_FRACTION, abs=0.02)
    assert len(splits["test"]) / 3000 == pytest.approx(params.furniture.TEST_FRACTION, abs=0.02)


def test_read_layout_parses_svg(tmp_path):
    payload = {"arrangement": {"furniture": FURNITURE}, "floorplan": FLOORPLAN}
    svg = tmp_path / "a_accepted.svg"
    svg.write_text(f'<svg><metadata id="x" type="application/json">{json.dumps(payload)}</metadata></svg>')
    layout = read_layout(svg)
    assert layout["furniture"] == FURNITURE and layout["floorplan"]["id"] == 14433
    (tmp_path / "bad.svg").write_text("<svg></svg>")
    with pytest.raises(ValueError):
        read_layout(tmp_path / "bad.svg")


# ----------------------------------------------------------------------------- model side

def test_hyperparameters_cover_dataset_types(params):
    for mode in (None, "debug"):
        hyper = load_params(HYPER_PATH, mode)
        assert set(params.missing.COMPONENT_TYPES) | {"none"} <= set(hyper.model.TYPE_CLASSES)


def test_check_inputs_reports_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_dataset"):
        check_inputs(tmp_path, load_params(HYPER_PATH, "debug"))


def test_loss_and_backward_with_tiny_model(schema):
    hyper = load_params(HYPER_PATH, "debug")
    torch.manual_seed(0)
    size, channels = 32, schema.num_channels
    model = ConditionalUNet(channels, size, hyper.model)
    target = (torch.rand(2, channels, size, size) > 0.97).float()
    terms = compute_loss(model, GaussianDiffusion(1000, 0.999), torch.zeros_like(target), target, torch.tensor([4, 0]), hyper.loss)
    terms.total.backward()
    assert torch.isfinite(terms.total) and 0.0 <= float(terms.type_accuracy) <= 1.0
    assert all(p.grad is not None for p in model.parameters())
