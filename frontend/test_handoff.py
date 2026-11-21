"""The hand-off from the wall/door/window model to the furniture models: completed plan -> floorplan and furniture class-id masks.

The unit tests need nothing else. The calibration test compares the converter with the masks stored in ``Val-Samples/``
for the FurnitureArranger samples; it is skipped when that folder is absent.

Run from the repository root:
    python -m pytest frontend/test_handoff.py -q
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from Reconstruction.common.config_loader import load_params
from Reconstruction.data_prep.furniture_source import layout_to_geometry, parse_layout
from Reconstruction.data_prep.rasterize import rasterize_plan
from Reconstruction.data_prep.schema import Schema
from Reconstruction.evaluation.labels import EMPTY, LabelMaps, semantic_to_labels
from frontend.handoff import floorplan_ids, furniture_ids, load_settings, plan_box, square, to_furniture_masks

ROOT: Path = Path(__file__).resolve().parents[1]
SETTINGS = load_settings()
MASK_IDS: dict[str, int] = SETTINGS.mask_ids
SHELL: Schema = Schema.from_params(load_params(ROOT / "datasets" / "processed" / "furniture_shell" / "params.toml"))
FURN_PARAMS = load_params(ROOT / "datasets" / "processed" / "furniture" / "params.toml")
FURN: Schema = Schema.from_params(FURN_PARAMS)
VAL_SAMPLES: Path = ROOT / "Val-Samples"


def room(schema: Schema, size: int = 40) -> LabelMaps:
    """A living room (rows/cols 4..35) with a wall frame, a door in the top wall and a window in the left wall."""
    structure = np.full((size, size), EMPTY, np.int8)
    rooms = np.full((size, size), EMPTY, np.int8)
    rooms[4:36, 4:36] = schema.room_classes.index("living")
    wall = schema.structure_classes.index("wall")
    structure[4, 4:36] = structure[35, 4:36] = structure[4:36, 4] = structure[4:36, 35] = wall
    structure[4, 8:12] = schema.structure_classes.index("door")
    structure[35, 18:22] = schema.structure_classes.index("door")
    structure[10:14, 4] = schema.structure_classes.index("window")
    return LabelMaps(structure, rooms)


def test_every_class_has_a_furniture_id() -> None:
    assert {n for n in FURN.channels if FURN.component_type[n] != "furniture"} <= set(MASK_IDS)
    assert {n for n in FURN.channels if FURN.component_type[n] == "furniture"} <= set(SETTINGS.furniture_ids)


def test_classes_map_to_furniture_ids_with_structure_over_rooms() -> None:
    ids = floorplan_ids(room(SHELL), SHELL, MASK_IDS)
    assert ids.dtype == np.uint8 and ids.shape == (40, 40)
    assert ids[0, 0] == MASK_IDS["background"]
    assert ids[20, 20] == MASK_IDS["living"]
    assert ids[4, 30] == MASK_IDS["wall"]
    assert ids[4, 9] == MASK_IDS["door"]
    assert ids[11, 4] == MASK_IDS["window"]


def test_missing_class_id_is_reported() -> None:
    with pytest.raises(KeyError, match="no id for"):
        floorplan_ids(room(SHELL), SHELL, {"background": 0})


def test_plan_box_and_square_resize() -> None:
    ids = floorplan_ids(room(SHELL), SHELL, MASK_IDS)
    assert plan_box(ids, 0) == (4, 36, 4, 36)
    assert plan_box(np.zeros((8, 8), np.uint8), 0) == (0, 8, 0, 8)
    assert square(ids[4:36, 4:20], 64).shape == (64, 64)


def test_shell_plans_have_no_furniture_mask_and_furniture_plans_do() -> None:
    assert furniture_ids(room(SHELL), SHELL, SETTINGS.furniture_ids) is None
    plan = room(FURN)
    plan.structure[20:24, 20:30] = FURN.structure_classes.index("tv")
    ids = furniture_ids(plan, FURN, SETTINGS.furniture_ids)
    assert ids[21, 25] == SETTINGS.furniture_ids["tv"] and ids[0, 0] == 0


def test_masks_share_one_crop_and_are_square() -> None:
    plan = room(FURN)
    plan.structure[20:24, 20:30] = FURN.structure_classes.index("tv")
    floor, items = to_furniture_masks(plan, FURN, SETTINGS)
    size = SETTINGS.export.SIZE
    assert floor.shape == items.shape == (size, size)
    assert (floor != MASK_IDS["background"]).all()                       # the plan fills the image, as the furniture models' do
    assert (items == SETTINGS.furniture_ids["tv"]).sum() > 0


def test_round_trip_back_to_the_plan_grid() -> None:
    plan = room(FURN)
    floor, _ = to_furniture_masks(plan, FURN, SETTINGS)
    r0, r1, c0, c1 = plan_box(floorplan_ids(plan, FURN, MASK_IDS), 0)
    back = np.array(Image.fromarray(floor).resize((c1 - c0, r1 - r0), Image.NEAREST))
    assert (back == floorplan_ids(plan, FURN, MASK_IDS)[r0:r1, c0:c1]).mean() > 0.99


def labels_of(svg: Path, schema: Schema, params) -> LabelMaps:
    """Rasterize a layout SVG the way the demo does and return its label maps."""
    raster = rasterize_plan(layout_to_geometry(0, parse_layout(svg.read_text()), params), schema, params)
    semantic = np.zeros((schema.num_channels, 256, 256), np.uint8)
    layers = {"structure": raster.structure, "room": raster.room}
    for inst in raster.instances:
        semantic[schema.channel_index(inst["cls"])][layers[inst["layer"]] == inst["id"]] = 1
    return semantic_to_labels(semantic, schema, 0.5)


@pytest.mark.skipif(not VAL_SAMPLES.is_dir(), reason="Val-Samples folder not present")
def test_conversion_matches_the_masks_metanoia_stored() -> None:
    """Rasterize each sample's SVG, convert it, and compare with the stored masks (squeezed to 256)."""
    floor_acc: list[float] = []
    furn_acc: list[float] = []
    for folder in sorted(VAL_SAMPLES.iterdir()):
        if not (folder / "ground_truth.svg").exists():
            continue
        floor, items = to_furniture_masks(labels_of(folder / "ground_truth.svg", FURN, FURN_PARAMS), FURN, SETTINGS)
        stored_floor = np.asarray(Image.open(folder / "floorplan_segmentation.png").resize((256, 256), Image.NEAREST))
        stored_furn = np.asarray(Image.open(folder / "furniture_segmentation.png").resize((256, 256), Image.NEAREST))
        floor_acc.append(float((floor == stored_floor).mean()))
        furn_acc.append(float((items == stored_furn).mean()))
    assert floor_acc and min(floor_acc) > 0.95 and min(furn_acc) > 0.95
