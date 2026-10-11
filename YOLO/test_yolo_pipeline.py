"""Unit tests for the YOLO live-diffusion label/box/ontology logic.

Arrange-Act-Assert throughout, with a small deterministic ``FakeLog`` in place
of the project logger so the tests need no config or checkpoints. Covers the
three failure modes that silently produced an untrained model:

1. Gradient-accumulation starvation (``optimizer_steps_per_epoch`` /
   ``validate_training_budget``).
2. Train labels that contradict the drawn pixels
   (``connected_component_boxes`` / ``format_yolo_label_lines``).
3. An ontology the arrangement mask cannot express
   (``build_mask_to_yolo_id_map``).

Run with: ``./venv/bin/python -m pytest YOLO/test_yolo_pipeline.py``
(or ``./venv/bin/python YOLO/test_yolo_pipeline.py`` for a plain run).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

# Project root so ``YOLO``/``helpers`` import when run as a script.
_ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from mask_VAE.masking import MaskGroup, MaskLayout
from YOLO.augment import resolve_class_name_to_id
from YOLO.live_diffusion import (
    build_mask_to_yolo_id_map,
    connected_component_boxes,
    format_yolo_label_lines,
    mirror_signature,
    normalize_class_name,
)
from YOLO.predict import (
    build_contact_sheet,
    labels_dir_for,
    read_gt_lines,
    resolve_weights_path,
)
from YOLO.runner import (
    draw_boxes_on_image,
    optimizer_steps_per_epoch,
    validate_training_budget,
)


class FakeLog:
    """Minimal stand-in for the project logger (records instead of writing)."""

    def __init__(self) -> None:
        """Start with empty message lists."""
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        """Record an informational message."""
        self.infos.append(message)

    def warning(self, message: str) -> None:
        """Record a warning message."""
        self.warnings.append(message)


def make_layout(class_names: tuple[str, ...]) -> tuple[MaskLayout, MaskGroup]:
    """Build a single-group layout mirroring the furniture arrangement masks.

    Args:
        class_names (tuple[str, ...]): Local class names, index 0 = background.

    Returns:
        tuple[MaskLayout, MaskLayout]: The layout and its single group.
    """
    num_classes: int = len(class_names)
    group = MaskGroup(
        name="furniture",
        mask_file_name="furniture_segmentation.png",
        class_names=class_names,
        colors=tuple((0, 0, 0) for _ in range(num_classes)),
        channel_indices=tuple(range(num_classes)),
    )
    layout = MaskLayout(
        groups=(group,),
        class_names=tuple(f"furniture/{name}" for name in class_names),
        num_channels=num_classes,
        foreground_channels=tuple(range(1, num_classes)),
    )
    return layout, group


REAL_MASK_CLASSES: tuple[str, ...] = (
    "background",
    "sofa",
    "tv",
    "dining_set",
    "side_table",
    "showcase",
    "couch",
    "chair",
    "center_table",
    "vase_table",
    "sofa_cum_bed",
)
TEN_CLASS_ONTOLOGY: list[str] = [
    "Center_table",
    "Chair",
    "Couch",
    "Dining_table",
    "Showcase",
    "Side_table",
    "Sofa",
    "Sofa_cum_bed",
    "TV",
    "Vase_table",
]
DINING_ALIASES: dict[str, str] = {
    "Dining_table_big": "Dining_table",
    "Dining_table_round": "Dining_table",
    "Dining_table_medium": "Dining_table",
    "dining_set": "Dining_table",
}


# --------------------------------------------------------------------------
# 1. Optimizer-step budget (the starvation bug)
# --------------------------------------------------------------------------

def test_optimizer_steps_starved_by_accumulation() -> None:
    """The failing 30-epoch run: 8 images, batch 16, nbs 64 -> ~0 steps/epoch."""
    # Arrange / Act
    steps = optimizer_steps_per_epoch(8, 16, 64)
    # Assert: 1 batch per epoch, accumulate 4 -> never a full step.
    assert steps == 0


def test_optimizer_steps_when_accumulation_disabled() -> None:
    """Pinning nbs = batch removes accumulation, so batches are steps."""
    # Arrange / Act
    steps = optimizer_steps_per_epoch(256, 16, 16)
    # Assert: 16 batches, accumulate 1 -> 16 steps.
    assert steps == 16


def test_validate_budget_raises_when_starved() -> None:
    """A run that cannot update its weights must fail before wasting hours."""
    # Arrange
    log = FakeLog()
    # Act / Assert
    try:
        validate_training_budget(8, 16, 64, 30, log)
    except ValueError as exc:
        assert "optimizer steps" in str(exc)
    else:
        raise AssertionError("expected ValueError for a starved training budget")


def test_validate_budget_raises_when_whole_run_too_short() -> None:
    """One usable step in total is still not a training run."""
    # Arrange
    log = FakeLog()
    # Act / Assert: 16 images, batch 16, nbs 16 -> 1 step/epoch, 1 epoch total.
    try:
        validate_training_budget(16, 16, 16, 1, log)
    except ValueError as exc:
        assert "optimizer steps in total" in str(exc)
    else:
        raise AssertionError("expected ValueError for a too-short run")


def test_validate_budget_passes_and_logs() -> None:
    """A healthy budget returns its step count and logs a summary."""
    # Arrange
    log = FakeLog()
    # Act
    steps = validate_training_budget(256, 16, 16, 30, log)
    # Assert
    assert steps == 16
    assert any("Optimizer budget" in message for message in log.infos)


# --------------------------------------------------------------------------
# 2. Labels derived from the drawn pixels (the label/pixel mismatch)
# --------------------------------------------------------------------------

def test_connected_components_finds_each_instance() -> None:
    """One box per 4-connected region of a class, background ignored."""
    # Arrange
    id_map = np.zeros((64, 64), dtype=np.int32)
    id_map[4:20, 4:20] = 3
    id_map[40:56, 8:24] = 3
    id_map[30:40, 40:52] = 7
    # Act
    boxes = connected_component_boxes(id_map, min_area_px=10)
    # Assert: two id-3 blobs and one id-7 blob, no background box.
    assert sorted(boxes) == [
        (3, 4, 4, 20, 20),
        (3, 8, 40, 24, 56),
        (7, 40, 30, 52, 40),
    ]


def test_connected_components_drops_speckle() -> None:
    """Regions below the area floor are decoder noise, not furniture."""
    # Arrange
    id_map = np.zeros((64, 64), dtype=np.int32)
    id_map[4:24, 4:24] = 3       # 400 px real instance
    id_map[50:53, 50:53] = 3     # 9 px speckle
    # Act
    kept = connected_component_boxes(id_map, min_area_px=200)
    # Assert
    assert kept == [(3, 4, 4, 24, 24)]


def test_format_yolo_label_lines_normalizes_box() -> None:
    """Pixel boxes become normalized ``cls cx cy w h`` lines."""
    # Arrange
    boxes = [(3, 20, 10, 60, 50)]  # 40x40 box at (20,10) in a 100x100 image
    mask_to_yolo = {3: 3}
    # Act
    lines = format_yolo_label_lines(boxes, mask_to_yolo, 100, 100)
    # Assert: center (40,30), size 40x40, all normalized.
    assert lines == ["3 0.400000 0.300000 0.400000 0.400000"]


def test_format_yolo_label_lines_rejects_unmapped_class() -> None:
    """An unmapped class must raise rather than emit a wrong id."""
    # Arrange
    boxes = [(9, 0, 0, 10, 10)]
    # Act / Assert
    try:
        format_yolo_label_lines(boxes, {3: 3}, 64, 64)
    except ValueError as exc:
        assert "not in the mask->YOLO map" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unmapped class")


def test_derived_boxes_reproduce_known_layout() -> None:
    """A hand-built arrangement round-trips to the expected label lines."""
    # Arrange
    id_map = np.zeros((256, 256), dtype=np.int32)
    id_map[20:80, 20:80] = 8     # center_table
    id_map[100:160, 30:90] = 4   # side_table
    # Act
    boxes = connected_component_boxes(id_map, min_area_px=200)
    lines = format_yolo_label_lines(boxes, {8: 0, 4: 6}, 256, 256)
    # Assert
    assert sorted(lines) == sorted([
        "0 0.195312 0.195312 0.234375 0.234375",
        "6 0.234375 0.507812 0.234375 0.234375",
    ])


# --------------------------------------------------------------------------
# 3. Ontology the mask can actually express
# --------------------------------------------------------------------------

def test_normalize_class_name_bridges_conventions() -> None:
    """CamelCase, snake_case and spaced names normalize to the same key."""
    # Arrange / Act / Assert
    assert normalize_class_name("Center_table") == normalize_class_name("center table")
    assert normalize_class_name("Dining_table_big") == "diningtablebig"


def test_mask_to_yolo_map_resolves_real_ontology() -> None:
    """All 10 mask classes map onto the 10-class ontology via the alias."""
    # Arrange
    layout, group = make_layout(REAL_MASK_CLASSES)
    # Act
    resolved = build_mask_to_yolo_id_map(layout, group, TEN_CLASS_ONTOLOGY, DINING_ALIASES)
    # Assert: 10 foreground ids, no background entry, ids unique.
    assert len(resolved) == 10
    assert 0 not in resolved
    assert len(set(resolved.values())) == 10


def test_mask_to_yolo_map_folds_dining_set() -> None:
    """``dining_set`` resolves to the single Dining_table id (not three)."""
    # Arrange
    layout, group = make_layout(REAL_MASK_CLASSES)
    # Act
    resolved = build_mask_to_yolo_id_map(layout, group, TEN_CLASS_ONTOLOGY, DINING_ALIASES)
    # Assert: dining_set is mask id 3 -> Dining_table id 3.
    assert resolved[3] == TEN_CLASS_ONTOLOGY.index("Dining_table")


def test_mask_to_yolo_map_rejects_unmappable_mask_class() -> None:
    """A mask class with no counterpart must fail, not guess."""
    # Arrange
    layout, group = make_layout(("background", "sofa", "wardrobe"))
    # Act / Assert
    try:
        build_mask_to_yolo_id_map(layout, group, TEN_CLASS_ONTOLOGY, DINING_ALIASES)
    except ValueError as exc:
        assert "wardrobe" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unmappable mask class")


def test_mask_to_yolo_map_rejects_unreachable_yolo_class() -> None:
    """A YOLO class no mask class can produce must fail."""
    # Arrange: ontology keeps a class the mask vocabulary lacks.
    ontology = TEN_CLASS_ONTOLOGY + ["Fireplace"]
    layout, group = make_layout(REAL_MASK_CLASSES)
    # Act / Assert
    try:
        build_mask_to_yolo_id_map(layout, group, ontology, DINING_ALIASES)
    except ValueError as exc:
        assert "Fireplace" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unreachable YOLO class")


def test_mask_to_yolo_map_rejects_alias_to_unknown_class() -> None:
    """An alias pointing outside CLASS_NAMES must fail."""
    # Arrange
    bad_aliases = dict(DINING_ALIASES)
    bad_aliases["dining_set"] = "Not_A_Class"
    layout, group = make_layout(REAL_MASK_CLASSES)
    # Act / Assert
    try:
        build_mask_to_yolo_id_map(layout, group, TEN_CLASS_ONTOLOGY, bad_aliases)
    except ValueError as exc:
        assert "Not_A_Class" in str(exc)
    else:
        raise AssertionError("expected ValueError for a bad alias target")


# --------------------------------------------------------------------------
# 4. Mirror staleness (stale val labels reused under an old ontology)
# --------------------------------------------------------------------------

def test_mirror_signature_is_stable_for_same_inputs() -> None:
    """Identical inputs must produce the same signature (cache hits)."""
    # Arrange / Act
    first = mirror_signature([], TEN_CLASS_ONTOLOGY, DINING_ALIASES, 0.6, 256)
    second = mirror_signature([], TEN_CLASS_ONTOLOGY, DINING_ALIASES, 0.6, 256)
    # Assert
    assert first == second


def test_mirror_signature_changes_with_ontology() -> None:
    """A class-list change must invalidate the mirror (stale label guard)."""
    # Arrange
    renamed = list(TEN_CLASS_ONTOLOGY)
    renamed[3] = "Dining_table_round"
    # Act
    signature = mirror_signature([], TEN_CLASS_ONTOLOGY, DINING_ALIASES, 0.6, 256)
    changed = mirror_signature([], renamed, DINING_ALIASES, 0.6, 256)
    # Assert
    assert signature != changed


def test_mirror_signature_changes_with_render_settings() -> None:
    """Alpha/size/stem changes must also invalidate the mirror."""
    # Arrange / Act / Assert
    base = mirror_signature([], TEN_CLASS_ONTOLOGY, DINING_ALIASES, 0.6, 256)
    assert base != mirror_signature([], TEN_CLASS_ONTOLOGY, DINING_ALIASES, 0.7, 256)
    assert base != mirror_signature([], TEN_CLASS_ONTOLOGY, DINING_ALIASES, 0.6, 512)


# --------------------------------------------------------------------------
# 5. Offline inference viewer helpers
# --------------------------------------------------------------------------

def test_labels_dir_for_maps_split_images() -> None:
    """A split image resolves to its mirrored label directory."""
    # Arrange
    live_root = Path("/data/yolo_live")
    image = live_root / "images" / "val" / "abc.png"
    # Act / Assert
    assert labels_dir_for(image, live_root) == live_root / "labels" / "val"


def test_labels_dir_for_ignores_external_image() -> None:
    """An image outside the live root has no GT directory."""
    # Arrange / Act / Assert
    assert labels_dir_for(Path("/tmp/custom.png"), Path("/data/yolo_live")) == Path("")


def test_read_gt_lines_skips_blank_lines() -> None:
    """Blank lines in a label file are dropped, not parsed as boxes."""
    # Arrange
    label_file = Path(tempfile.mkdtemp()) / "a.txt"
    label_file.write_text("3 0.5 0.5 0.2 0.2\n\n4 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    # Act / Assert
    assert read_gt_lines(label_file) == ["3 0.5 0.5 0.2 0.2", "4 0.1 0.1 0.1 0.1"]


def test_read_gt_lines_missing_file_is_empty() -> None:
    """A missing label file yields no GT (e.g. --image without ground truth)."""
    # Arrange / Act / Assert
    assert read_gt_lines(Path(tempfile.mkdtemp()) / "missing.txt") == []


def test_build_contact_sheet_raises_on_no_tiles() -> None:
    """An empty contact sheet is a programming error, not a blank image."""
    # Arrange / Act / Assert
    try:
        build_contact_sheet([], 4)
    except ValueError as exc:
        assert "no tiles" in str(exc)
    else:
        raise AssertionError("expected ValueError for an empty contact sheet")


def test_build_contact_sheet_layout_is_grid_shaped() -> None:
    """N tiles in C columns produce ceil(N/C) rows of the right size."""
    # Arrange
    from PIL import Image as PILImage

    tiles = [(f"caption {i}", PILImage.new("RGB", (64, 64))) for i in range(5)]
    # Act
    sheet = build_contact_sheet(tiles, columns=3)
    # Assert: 3 columns, 2 rows, each row adds a caption strip.
    assert sheet.size == (64 * 3, (64 + 18) * 2)


# --------------------------------------------------------------------------
# 6. Label-name aliasing (source labels.txt -> class ids)
# --------------------------------------------------------------------------

def test_resolve_class_name_to_id_applies_aliases() -> None:
    """Source dining names all resolve to the merged Dining_table id."""
    # Arrange / Act
    mapping = resolve_class_name_to_id(TEN_CLASS_ONTOLOGY, DINING_ALIASES)
    # Assert
    dining_id = TEN_CLASS_ONTOLOGY.index("Dining_table")
    assert mapping["Dining_table_big"] == dining_id
    assert mapping["Dining_table_round"] == dining_id
    assert mapping["Dining_table_medium"] == dining_id
    assert mapping["Chair"] == TEN_CLASS_ONTOLOGY.index("Chair")


def test_resolve_class_name_to_id_rejects_bad_alias() -> None:
    """An alias target outside the vocabulary must fail."""
    # Arrange
    bad = {"Dining_table_big": "Nope"}
    # Act / Assert
    try:
        resolve_class_name_to_id(TEN_CLASS_ONTOLOGY, bad)
    except ValueError as exc:
        assert "Nope" in str(exc)
    else:
        raise AssertionError("expected ValueError for a bad alias target")


def test_resolve_class_name_to_id_rejects_duplicate_class() -> None:
    """A duplicated vocabulary entry must fail rather than shadow an id."""
    # Arrange
    duplicated = ["Chair", "Chair"]
    # Act / Assert
    try:
        resolve_class_name_to_id(duplicated, None)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("expected ValueError for a duplicate class name")


# --------------------------------------------------------------------------
# 7. Safe prediction visualization & path resolution
# --------------------------------------------------------------------------

def test_draw_boxes_on_image_handles_out_of_range_classes() -> None:
    """Out-of-range class ids fall back to 'cls_<id>' rather than raising IndexError."""
    # Arrange
    from PIL import Image as PILImage
    from types import SimpleNamespace

    img = PILImage.new("RGB", (100, 100), (255, 255, 255))
    gt_lines = ["15 0.5 0.5 0.2 0.2"]  # class id 15 is out of bounds
    pred_boxes = [(10.0, 10.0, 30.0, 30.0, 20, 0.95)]  # class id 20 is out of bounds
    class_names = ["Center_table", "Chair"]

    # Act
    annotated = draw_boxes_on_image(img, gt_lines, pred_boxes, class_names)

    # Assert
    assert annotated.size == (100, 100)


def test_draw_boxes_on_image_supports_pred_class_names() -> None:
    """Prediction boxes can use separate model class names from GT."""
    # Arrange
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (100, 100), (255, 255, 255))
    gt_lines = ["0 0.5 0.5 0.2 0.2"]
    pred_boxes = [(10.0, 10.0, 30.0, 30.0, 1, 0.88)]
    gt_classes = ["Center_table"]
    pred_classes = ["Center_table", "Sofa_12_class_special"]

    # Act
    annotated = draw_boxes_on_image(
        img, gt_lines, pred_boxes, gt_classes, pred_class_names=pred_classes
    )

    # Assert
    assert annotated.size == (100, 100)


def test_resolve_weights_path_prefers_override_and_falls_back() -> None:
    """resolve_weights_path prefers explicit path and handles directory fallbacks."""
    # Arrange
    from types import SimpleNamespace
    temp_dir = Path(tempfile.mkdtemp())
    dummy_weight = temp_dir / "yolo_test_best.pt"
    dummy_weight.write_text("weights", encoding="utf-8")
    cfg = SimpleNamespace(MODEL_SAVE_PATH=str(temp_dir))
    hyp = SimpleNamespace()

    # Act / Assert
    resolved = resolve_weights_path(cfg, hyp, override=str(dummy_weight))
    assert resolved == dummy_weight

    # Default fallback to newest checkpoint in directory
    resolved_default = resolve_weights_path(cfg, hyp, override=None)
    assert resolved_default == dummy_weight


def main() -> int:
    """Run every test in this module and report a pass/fail summary.

    Returns:
        int: Process exit code (0 when all tests pass).
    """
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures: list[tuple[str, BaseException]] = []
    for test in tests:
        try:
            test()
        except BaseException as exc:  # noqa: BLE001 - report any test failure
            failures.append((test.__name__, exc))
        else:
            print(f"PASS {test.__name__}")
    for name, exc in failures:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
