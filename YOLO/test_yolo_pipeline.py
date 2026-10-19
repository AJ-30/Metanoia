"""Unit tests for the YOLO live-diffusion dataset/visualization logic.

Arrange-Act-Assert throughout, with a small deterministic ``FakeLog`` in place
of the project logger so the tests need no config, checkpoints or GPU. Covers
the failure modes that matter for this pipeline:

1. Gradient-accumulation starvation (``optimizer_steps_per_epoch`` /
   ``validate_training_budget``).
2. Labels that contradict the pixels they label
   (``connected_component_boxes`` / ``boxes_to_yolo_lines``).
3. An ontology the arrangement mask cannot express
   (``build_mask_to_yolo_id_map``).
4. Stale deterministic renders silently reused
   (``render_signature`` / ``split_is_current``).
5. Telemetry mapping of Ultralytics results.csv rows.

Run with: ``./venv/bin/python -m pytest YOLO/test_yolo_pipeline.py``
(or ``./venv/bin/python YOLO/test_yolo_pipeline.py`` for a plain run).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace

# Project root so ``YOLO``/``helpers`` import when run as a script.
_ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from mask_VAE.masking import MaskGroup, MaskLayout
from YOLO.dataset import (
    apply_label_lines_to_dataset,
    boxes_to_yolo_lines,
    build_mask_to_yolo_id_map,
    clear_generated_split,
    connected_component_boxes,
    list_sample_folders,
    normalize_class_name,
    parse_label_lines_to_arrays,
    render_signature,
    resolve_class_name_to_id,
    split_is_current,
    write_data_yaml,
    write_label_file,
)
from YOLO.overlay import (
    choose_fixed_stems,
    draw_boxes_on_image,
    make_montage,
)
from YOLO.predict import labels_dir_for, read_gt_lines, resolve_weights_path
from YOLO.runner import (
    map_results_row,
    anchor_config_paths,
    optimizer_steps_per_epoch,
    read_results_csv,
    resolve_train_device,
    validate_training_budget,
)

CLASS_NAMES: list[str] = [
    "Center_table", "Chair", "Couch", "Dining_table", "Showcase",
    "Side_table", "Sofa", "Sofa_cum_bed", "TV", "Vase_table",
]

# Furniture arrangement mask vocabulary: id 0 is background, 1..10 the
# furniture classes (dining_set cannot be split by size, hence the alias).
FURNITURE_MASK_CLASSES: tuple[str, ...] = (
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
        tuple[MaskLayout, MaskGroup]: The layout and its single group.
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


def make_fake_sources(stems: list[str]) -> list:
    """Build fake SampleFolder-like sources for signature tests.

    Args:
        stems (list[str]): Stems the fake sources should carry.

    Returns:
        list: Lightweight objects with a ``.stem`` attribute.
    """
    from types import SimpleNamespace

    return [SimpleNamespace(stem=stem) for stem in stems]


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------


def test_normalize_class_name_collapses_conventions() -> None:
    """CamelCase, snake_case and spaced names must normalize to one key."""
    assert normalize_class_name("Center_table") == "centertable"
    assert normalize_class_name("center_table") == "centertable"
    assert normalize_class_name("Center Table") == "centertable"


def test_build_mask_to_yolo_id_map_maps_every_furniture_class() -> None:
    """Every mask class maps to a YOLO id; dining_set folds via alias."""
    layout, group = make_layout(FURNITURE_MASK_CLASSES)
    id_map = build_mask_to_yolo_id_map(
        layout, group, CLASS_NAMES, {"dining_set": "Dining_table"}
    )
    assert id_map[1] == CLASS_NAMES.index("Sofa")
    assert id_map[2] == CLASS_NAMES.index("TV")
    assert id_map[3] == CLASS_NAMES.index("Dining_table")
    assert id_map[8] == CLASS_NAMES.index("Center_table")
    assert len(id_map) == len(FURNITURE_MASK_CLASSES) - 1
    assert 0 not in id_map  # background is never a detection target


def test_build_mask_to_yolo_id_map_rejects_unmapped_mask_class() -> None:
    """A mask class with no YOLO counterpart must abort, not guess."""
    class_names: tuple[str, ...] = ("background", "sofa", "mystery_furniture")
    layout, group = make_layout(class_names)
    try:
        build_mask_to_yolo_id_map(layout, group, CLASS_NAMES, {})
    except ValueError as exc:
        assert "mystery_furniture" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unmapped mask class")


def test_build_mask_to_yolo_id_map_rejects_unreachable_yolo_class() -> None:
    """A YOLO class no mask class can produce must abort the run."""
    layout, group = make_layout(("background", "sofa"))
    try:
        build_mask_to_yolo_id_map(layout, group, CLASS_NAMES, {})
    except ValueError as exc:
        assert "no furniture mask counterpart" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unreachable YOLO class")


def test_resolve_class_name_to_id_rejects_bad_alias() -> None:
    """An alias pointing outside CLASS_NAMES must abort."""
    try:
        resolve_class_name_to_id(CLASS_NAMES, {"dining_set": "Does_not_exist"})
    except ValueError as exc:
        assert "Does_not_exist" in str(exc)
    else:
        raise AssertionError("Expected ValueError for alias outside CLASS_NAMES")


# ---------------------------------------------------------------------------
# Boxes and labels
# ---------------------------------------------------------------------------


def test_connected_component_boxes_finds_each_component() -> None:
    """Two disjoint regions of one class become two boxes; background skipped."""
    id_map = np.zeros((10, 10), dtype=np.int32)
    id_map[1:3, 1:3] = 1  # 4-pixel component at top-left
    id_map[6:9, 6:9] = 1  # 9-pixel component at bottom-right
    boxes = connected_component_boxes(id_map, min_area_px=1)
    assert len(boxes) == 2
    assert boxes[0] == (1, 1, 1, 3, 3)
    assert boxes[1] == (1, 6, 6, 9, 9)


def test_connected_component_boxes_drops_speckle() -> None:
    """Regions below the min area are decoder speckle, not furniture."""
    id_map = np.zeros((10, 10), dtype=np.int32)
    id_map[1:3, 1:3] = 2  # 4 pixels
    id_map[5, 5] = 2      # 1 pixel
    boxes = connected_component_boxes(id_map, min_area_px=3)
    assert len(boxes) == 1
    assert boxes[0][0] == 2


def test_boxes_to_yolo_lines_normalizes_corners() -> None:
    """A box covering the whole image normalizes to cx=cy=0.5, w=h=1."""
    boxes = [(1, 0, 0, 8, 8)]
    id_map = {1: 6}
    lines = boxes_to_yolo_lines(boxes, id_map, height=8, width=8)
    assert lines == ["6 0.500000 0.500000 1.000000 1.000000"]


def test_boxes_to_yolo_lines_rejects_unknown_class() -> None:
    """A box class missing from the map must abort rather than mislabel."""
    try:
        boxes_to_yolo_lines([(7, 0, 0, 4, 4)], {1: 0}, height=8, width=8)
    except ValueError as exc:
        assert "7" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown box class")


def test_write_label_file_empty_layout() -> None:
    """A layout with no furniture writes an empty (not missing) label file."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.txt"
        write_label_file(path, [])
        assert path.is_file()
        assert path.read_text(encoding="utf-8") == ""


class FakeUltralyticsDataset:
    """Minimal stand-in for Ultralytics' in-memory train dataset.

    Attributes mirror the two the label sync touches: ``im_files`` (stem
    lookup) and ``labels`` (per-image dicts with ``cls``/``bboxes`` arrays in
    the exact shape ``cache_labels`` produces).
    """

    def __init__(self, stems: list[str]) -> None:
        self.im_files: list[str] = [f"/data/images/train/{stem}.png" for stem in stems]
        self.labels: list[dict[str, object]] = [
            {
                "im_file": image_file,
                "cls": np.zeros((0, 1), dtype=np.float32),
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "normalized": True,
                "bbox_format": "xywh",
            }
            for image_file in self.im_files
        ]


def test_parse_label_lines_to_arrays_matches_ultralytics_format() -> None:
    """Parsed lines become (n,1) cls and (n,4) normalized xywh arrays."""
    cls, bboxes = parse_label_lines_to_arrays(["3 0.5 0.5 0.25 0.125"])
    assert cls.shape == (1, 1)
    assert bboxes.shape == (1, 4)
    assert cls[0, 0] == 3.0
    assert list(bboxes[0]) == [0.5, 0.5, 0.25, 0.125]


def test_parse_label_lines_to_arrays_empty_layout() -> None:
    """An empty label file maps to (0,1)/(0,4) arrays, not scalars."""
    cls, bboxes = parse_label_lines_to_arrays([])
    assert cls.shape == (0, 1)
    assert bboxes.shape == (0, 4)


def test_parse_label_lines_to_arrays_malformed_line_raises() -> None:
    """A line without exactly five numbers aborts rather than mislabels."""
    for bad in ("3 0.5 0.5", "a b c d e"):
        try:
            parse_label_lines_to_arrays([bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {bad!r}")


def test_apply_label_lines_updates_matching_stems_only() -> None:
    """Only stems present in the mapping are updated; others stay empty."""
    dataset = FakeUltralyticsDataset(["a", "b", "c"])
    updated = apply_label_lines_to_dataset(
        dataset,
        {
            "a": ["3 0.5 0.5 0.25 0.125", "6 0.1 0.1 0.2 0.2"],
            "c": [],
        },
    )
    assert updated == 2
    assert dataset.labels[0]["cls"].shape == (2, 1)  # type: ignore[index]
    assert dataset.labels[0]["bboxes"].shape == (2, 4)  # type: ignore[index]
    # "b" is absent from the mapping and "c" is an empty layout.
    assert dataset.labels[1]["cls"].shape == (0, 1)  # type: ignore[index]
    assert dataset.labels[2]["cls"].shape == (0, 1)  # type: ignore[index]
    assert dataset.labels[2]["bboxes"].shape == (0, 4)  # type: ignore[index]


def test_apply_label_lines_is_idempotent() -> None:
    """Re-applying the same lines yields the same arrays (no duplication)."""
    lines = ["3 0.5 0.5 0.25 0.125"]
    dataset = FakeUltralyticsDataset(["a"])
    apply_label_lines_to_dataset(dataset, {"a": lines})
    first_cls = dataset.labels[0]["cls"].copy()  # type: ignore[union-attr]
    apply_label_lines_to_dataset(dataset, {"a": lines})
    assert dataset.labels[0]["cls"].shape == first_cls.shape  # type: ignore[index]


# ---------------------------------------------------------------------------
# Render signature / staleness
# ---------------------------------------------------------------------------


def test_render_signature_changes_with_content() -> None:
    """Anything that changes rendered pixels or label ids changes the digest."""
    sources = make_fake_sources(["a", "b"])
    base = render_signature(sources, CLASS_NAMES, {}, 0.6, 256, 200)
    same = render_signature(sources, CLASS_NAMES, {}, 0.6, 256, 200)
    changed_alpha = render_signature(sources, CLASS_NAMES, {}, 0.7, 256, 200)
    changed_stems = render_signature(
        make_fake_sources(["a", "c"]), CLASS_NAMES, {}, 0.6, 256, 200
    )
    changed_area = render_signature(sources, CLASS_NAMES, {}, 0.6, 256, 100)
    assert base == same
    assert base != changed_alpha
    assert base != changed_stems
    assert base != changed_area


def test_split_is_current_requires_signature_and_stems() -> None:
    """A stem-only match with a stale signature is NOT current."""
    with tempfile.TemporaryDirectory() as tmp:
        img_dir = Path(tmp) / "images" / "val"
        img_dir.mkdir(parents=True)
        sources = make_fake_sources(["a", "b"])
        for stem in ("a", "b"):
            (img_dir / f"{stem}.png").write_bytes(b"")
        signature_path = img_dir / ".render_signature"
        signature = render_signature(sources, CLASS_NAMES, {}, 0.6, 256, 200)
        assert split_is_current(img_dir, sources, signature_path, signature) is False
        signature_path.write_text(f"{signature}\n")
        assert split_is_current(img_dir, sources, signature_path, signature) is True
        # An extra stale PNG (from another run's stem set) is not current.
        (img_dir / "stale.png").write_bytes(b"")
        assert split_is_current(img_dir, sources, signature_path, signature) is False


def test_clear_generated_split_keeps_foreign_files() -> None:
    """Only generated PNG/TXT files are removed; signatures survive."""
    with tempfile.TemporaryDirectory() as tmp:
        img_dir = Path(tmp) / "images"
        lbl_dir = Path(tmp) / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()
        (img_dir / "keep.png").write_bytes(b"")
        (img_dir / ".render_signature").write_text("sig")
        (lbl_dir / "keep.txt").write_text("")
        (lbl_dir / "notes.md").write_text("keep me")
        clear_generated_split(img_dir, lbl_dir)
        assert not (img_dir / "keep.png").exists()
        assert not (lbl_dir / "keep.txt").exists()
        assert (img_dir / ".render_signature").is_file()
        assert (lbl_dir / "notes.md").is_file()


# ---------------------------------------------------------------------------
# Source indexing
# ---------------------------------------------------------------------------


def test_list_sample_folders_requires_both_masks() -> None:
    """Only folders carrying BOTH masks are indexed; order is sorted."""
    with tempfile.TemporaryDirectory() as tmp:
        split = Path(tmp) / "train"
        (split / "b_sample").mkdir(parents=True)
        (split / "a_sample").mkdir(parents=True)
        (split / "c_partial").mkdir(parents=True)
        for name in ("a_sample", "b_sample"):
            (split / name / "floorplan_segmentation.png").write_bytes(b"")
            (split / name / "furniture_segmentation.png").write_bytes(b"")
        (split / "c_partial" / "floorplan_segmentation.png").write_bytes(b"")
        sources = list_sample_folders(
            str(split), "floorplan_segmentation.png", "furniture_segmentation.png", -1
        )
        assert [source.stem for source in sources] == ["a_sample", "b_sample"]


def test_list_sample_folders_cap_limits_pool() -> None:
    """A non-negative cap keeps only the first N sorted folders."""
    with tempfile.TemporaryDirectory() as tmp:
        split = Path(tmp) / "train"
        for name in ("a", "b", "c"):
            (split / name).mkdir(parents=True)
            (split / name / "floorplan_segmentation.png").write_bytes(b"")
            (split / name / "furniture_segmentation.png").write_bytes(b"")
        sources = list_sample_folders(
            str(split), "floorplan_segmentation.png", "furniture_segmentation.png", 2
        )
        assert [source.stem for source in sources] == ["a", "b"]


# ---------------------------------------------------------------------------
# Training budget
# ---------------------------------------------------------------------------


def test_optimizer_steps_per_epoch_default_nbs_starves() -> None:
    """The Ultralytics default nbs=64 with batch=16 needs >= 4 batches/step."""
    assert optimizer_steps_per_epoch(32, 16, 64) == 0
    assert optimizer_steps_per_epoch(64, 16, 64) == 1


def test_optimizer_steps_per_epoch_pinned_nbs_updates_every_batch() -> None:
    """nbs = batch makes every batch an optimizer step."""
    assert optimizer_steps_per_epoch(32, 16, 16) == 2
    assert optimizer_steps_per_epoch(69776, 16, 16) == 4361


def test_validate_training_budget_starvation_raises() -> None:
    """An epoch with zero optimizer steps must abort before training."""
    log = FakeLog()
    try:
        validate_training_budget(8, 16, 64, 30, log)
    except ValueError as exc:
        assert "optimizer steps" in str(exc)
    else:
        raise AssertionError("Expected ValueError for starved budget")


def test_validate_training_budget_tiny_run_raises() -> None:
    """A whole run with fewer than two optimizer steps must abort."""
    log = FakeLog()
    try:
        validate_training_budget(16, 16, 16, 1, log)
    except ValueError as exc:
        assert "optimizer steps in total" in str(exc)
    else:
        raise AssertionError("Expected ValueError for <2 total steps")


def test_validate_training_budget_healthy_run_logs() -> None:
    """A full-pool run reports its per-epoch step count and passes."""
    log = FakeLog()
    steps = validate_training_budget(69776, 16, 16, 30, log)
    assert steps == 4361
    assert log.infos[-1].startswith("Optimizer budget")


# ---------------------------------------------------------------------------
# Device resolution and telemetry mapping
# ---------------------------------------------------------------------------


def test_resolve_train_device_falls_back_to_cpu() -> None:
    """An unavailable requested device warns and returns CPU."""
    log = FakeLog()
    expected = "mps" if torch.backends.mps.is_available() else "cpu"
    assert resolve_train_device(expected, log) == expected
    assert resolve_train_device("tpu", log) == "cpu"
    assert log.warnings, "a fallback must be logged"


def test_map_results_row_maps_all_fields() -> None:
    """A results.csv row maps onto every telemetry field, with fallbacks."""
    log = FakeLog()
    row = {
        "epoch": "3",
        "train/box_loss": "1.5",
        "train/cls_loss": "2.5",
        # YOLO26 writes the dfl term as l1_loss; train_dfl must pick it up.
        "train/l1_loss": "3.5",
        "val/box_loss": "1.4",
        "val/cls_loss": "2.4",
        "val/dfl_loss": "3.4",
        "metrics/precision(B)": "0.5",
        "metrics/recall(B)": "0.6",
        "metrics/mAP50(B)": "0.7",
        "metrics/mAP50-95(B)": "0.4",
        "lr/pg0": "0.01",
    }
    metrics = map_results_row(row, log)
    assert metrics is not None
    assert metrics["epoch"] == 3.0
    assert metrics["train_dfl"] == 3.5
    assert metrics["val_dfl"] == 3.4
    assert metrics["mAP50"] == 0.7


def test_map_results_row_malformed_returns_none() -> None:
    """A row missing metric columns is skipped, not half-logged."""
    log = FakeLog()
    assert map_results_row({"epoch": "1"}, log) is None
    assert log.warnings, "a skipped row must warn"


def test_read_results_csv_missing_file() -> None:
    """A missing results.csv reads as no rows (training crashed early)."""
    assert read_results_csv("/definitely/not/here/results.csv") == []


def test_read_results_csv_reads_rows() -> None:
    """A written results.csv round-trips into row dicts."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.csv"
        path.write_text("epoch,lr/pg0\n1,0.01\n", encoding="utf-8")
        rows = read_results_csv(str(path))
        assert rows == [{"epoch": "1", "lr/pg0": "0.01"}]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def test_draw_boxes_on_image_draws_gt_and_pred() -> None:
    """Green GT pixels and red prediction pixels both land on the image.

    The boxes are offset so neither overwrites the other, and the check is
    channel-dominance (PIL's builtin font antialiases glyph edges, so exact
    color equality would be flaky).
    """
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    gt_lines = ["0 0.25 0.25 0.25 0.25"]
    pred_boxes = [(24.0, 24.0, 56.0, 56.0, 1, 0.9)]
    annotated = draw_boxes_on_image(img, gt_lines, pred_boxes, CLASS_NAMES)
    pixels = list(annotated.getdata())
    assert any(red == 0 and green > 128 for red, green, _ in pixels)
    assert any(green == 0 and red > 128 for red, green, _ in pixels)


def test_choose_fixed_stems_is_stable_and_sorted() -> None:
    """The overlay subset is seeded, sorted and capped."""
    stems = [f"stem_{i:03d}" for i in range(100)]
    first = choose_fixed_stems(stems, 32, 1234)
    second = choose_fixed_stems(stems, 32, 1234)
    third = choose_fixed_stems(stems, 32, 9999)
    assert first == second
    assert first == sorted(first)
    assert len(first) == 32
    assert first != third
    assert choose_fixed_stems(stems, -1, 1234) == stems


def test_make_montage_tiles_all_images() -> None:
    """The montage has cols columns and one row for every image."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for index in range(5):
            path = Path(tmp) / f"img_{index}.png"
            Image.new("RGB", (32, 16), (index * 10, 0, 0)).save(str(path))
            paths.append(path)
        out_path = Path(tmp) / "montage.png"
        make_montage(paths, out_path, cols=4)
        sheet = Image.open(out_path)
        assert sheet.size == (4 * 32, 2 * 16)


# ---------------------------------------------------------------------------
# Offline viewer helpers
# ---------------------------------------------------------------------------


def test_labels_dir_for_maps_images_to_labels() -> None:
    """images/<split> maps onto labels/<split>."""
    assert labels_dir_for(Path("/root/images/val")) == Path("/root/labels/val")


def test_read_gt_lines_missing_and_blank() -> None:
    """A missing file reads as no lines; blank lines are dropped."""
    with tempfile.TemporaryDirectory() as tmp:
        assert read_gt_lines(Path(tmp) / "missing.txt") == []
        path = Path(tmp) / "labels.txt"
        path.write_text("0 0.1 0.1 0.2 0.2\n\n", encoding="utf-8")
        assert read_gt_lines(path) == ["0 0.1 0.1 0.2 0.2"]


def test_resolve_weights_path_explicit_and_newest() -> None:
    """Explicit weights win; otherwise the newest best.pt is picked."""
    with tempfile.TemporaryDirectory() as tmp:
        models = Path(tmp)
        explicit = models / "explicit.pt"
        explicit.write_bytes(b"")
        assert resolve_weights_path(models, str(explicit)) == explicit
        older = models / "yolo_old_best.pt"
        older.write_bytes(b"")
        newer = models / "yolo_new_best.pt"
        newer.write_bytes(b"")
        # Make the older file strictly older than the newer one.
        import os

        os.utime(older, (1000, 1000))
        os.utime(newer, (2000, 2000))
        assert resolve_weights_path(models, None) == newer


def test_resolve_weights_path_missing_raises() -> None:
    """No weights and no explicit path is a hard error."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            resolve_weights_path(Path(tmp), None)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Expected FileNotFoundError without weights")


def test_write_data_yaml_points_at_the_live_root() -> None:
    """data.yaml declares the three splits and the ordered class names."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "yolo_live_v2"
        yaml_path = root / "data.yaml"
        written = write_data_yaml(str(root), str(yaml_path), CLASS_NAMES)
        text = Path(written).read_text(encoding="utf-8")
        # write_data_yaml resolves the root (macOS /var -> /private/var), so
        # the expectation must be resolved too.
        assert f'path: "{root.resolve()}"' in text
        assert "train: images/train" in text
        assert "val: images/val" in text
        assert "test: images/test" in text
        assert f"nc: {len(CLASS_NAMES)}" in text
        assert "  0: Center_table" in text
        assert "  9: Vase_table" in text


# ---------------------------------------------------------------------------
# Config path anchoring
# ---------------------------------------------------------------------------


def test_anchor_config_paths_roots_relative_keys() -> None:
    """Relative config paths become absolute under the repo root."""
    config = SimpleNamespace(
        DEBUG_LOG_FILE_PATH="./debug_logs/yolo.log",
        MODEL_SAVE_PATH="./YOLO/models/",
        LIVE_DATASET_ROOT="/Users/aatmaj/Downloads/datasets/yolo_live_v2",
    )
    anchor_config_paths(config)
    repo_root: Path = Path(__file__).resolve().parent.parent
    assert config.DEBUG_LOG_FILE_PATH == str(repo_root / "debug_logs/yolo.log")
    assert config.MODEL_SAVE_PATH == str(repo_root / "YOLO/models")
    # Absolute paths (the dataset may live on another disk) pass through.
    assert config.LIVE_DATASET_ROOT == "/Users/aatmaj/Downloads/datasets/yolo_live_v2"


def test_anchor_config_paths_creates_log_directory() -> None:
    """The logger's directory is created even on a fresh clone."""
    with tempfile.TemporaryDirectory() as tmp:
        config = SimpleNamespace(
            DEBUG_LOG_FILE_PATH=str(Path(tmp) / "logs" / "yolo.log")
        )
        anchor_config_paths(config)
        assert (Path(tmp) / "logs").is_dir()


def _run_all_tests() -> int:
    """Run every ``test_*`` function in this module as a plain script.

    Returns:
        int: Process exit code (0 when every test passed, 1 otherwise).
    """
    tests = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]
    failures: list[str] = []
    for name, function in tests:
        try:
            function()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append(f"{name}: {exc}")
            print(f"FAIL {name}: {exc}")
        else:
            print(f"pass {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} tests passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all_tests())
