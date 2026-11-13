"""Unit tests for the frontend's pure logic.

Arrange-Act-Assert throughout, with no context, detector or GPU: only the parts of
the stage that can be judged without a model. Covers the failure modes that matter
for an interface, because a wrong message or a crashed upload page is what a user
sees instead of the result:

1. An upload that is not a class-id mask (RGB photo, palette photo, garbage
   bytes) must produce a message about the upload, never a server path or a 500.
2. An id outside the layout's range must be refused rather than clamped onto a
   neighbouring class.
3. A missing furniture upload must become a blank layout, not a crash.
4. The draw count must be clamped, so one request cannot launch a 1000-step chain
   a thousand times.
5. Old request folders must be pruned so the outputs directory cannot grow
   without bound.
6. The staged masks must be re-saved under the exact filenames the frozen
   encoders read.
7. Uploaded labels must be parsed strictly (numeric YOLO lines, named dataset
   lines and class-less lines, empty file as a negative, another numeric
   class/out-of-range/zero-area rejected), and the comparison must count
   TP/FP/FN, IoU, confidence and overlap with other furniture correctly.
8. Inpainting must only erase the three component classes, must merge generated
   pixels deterministically, must resolve overlaps by ensemble probability, and
   must render both the editor and result pages.

Run with: ``./venv/bin/python frontend/test_frontend.py``
(or ``pytest frontend/test_frontend.py`` when pytest is installed).
"""

import base64
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Repo root so ``frontend``/``mask_diffusion`` import when run as a script.
_ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from evaluator.box_metrics import PredictedBox
from evaluator.dataset_gt import Box
from flask import render_template
from frontend.comparison import (
    STATUS_CORRECT_EMPTY,
    STATUS_DETECTED_ALL,
    STATUS_DETECTED_PARTIAL,
    STATUS_FALSE_ALARM,
    STATUS_MISSED,
    compare_boxes,
    read_ground_truth_labels,
    read_ground_truth_layout,
)
from frontend.inpainting import (  # noqa: E402
    erasable_channels_from_palette,
    load_furniture_palette,
    merge_component_mask,
    merge_inpainted_components,
    validate_inpaint_edit,
)
from frontend.pipeline import (  # noqa: E402
    _clamp_draws,
    build_blank_mask,
    save_upload_as,
    validate_and_load_mask,
)
from frontend.server import (  # noqa: E402
    GROUND_TRUTH_FIELD,
    GROUND_TRUTH_LABELS_FILENAME,
    INPAINT_EDITED_FIELD,
    UPLOAD_MASKS,
    _decode_edited_ids,
    _prune_old_runs,
    create_app,
    format_metric,
)

# Class counts of the two layouts the encoders were trained with: 18 floorplan
# classes and 11 furniture classes, background included.
FLOORPLAN_CLASSES: int = 18
FURNITURE_CLASSES: int = 11
IMAGE_SIZE: int = 32


def _write_png(path: Path, array: np.ndarray, mode: str) -> None:
    """Write one test image.

    Args:
        path (Path): Destination PNG.
        array (np.ndarray): Pixel data.
        mode (str): PIL image mode to save in.
    """
    Image.fromarray(array, mode=mode).save(str(path))


def test_validate_and_load_mask_resizes_and_keeps_ids() -> None:
    """Any input size becomes the encoder's square, ids untouched."""
    mask = np.zeros((16, 48), dtype=np.uint8)
    mask[4:8, 4:8] = 3
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mask.png"
        _write_png(path, mask, "L")
        ids = validate_and_load_mask(path, FLOORPLAN_CLASSES, IMAGE_SIZE, "floorplan")
    assert ids.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert ids.dtype == torch.int64
    # The inner square survives the nearest-neighbour resize.
    assert int(ids.max()) == 3


def test_validate_and_load_mask_accepts_palette_mode() -> None:
    """A palette-encoded class-id mask is as valid as a mode-L one."""
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:4, 0:4] = 5
    # The identity grey ramp keeps palette index == class id, which is what a
    # P-mode class-id mask means: converting to L gives the ids back unchanged.
    palette = [channel for index in range(256) for channel in (index, index, index)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mask.png"
        image = Image.new("P", mask.shape[::-1])
        image.putpalette(palette)
        image.putdata([int(value) for value in mask.reshape(-1)])
        image.save(str(path))
        ids = validate_and_load_mask(path, FLOORPLAN_CLASSES, IMAGE_SIZE, "floorplan")
    assert int(ids.max()) == 5


def test_validate_and_load_mask_rejects_rgb_photo() -> None:
    """A colour photo must be refused with a message about the mask, not a path."""
    photo = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype("uint8")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "photo.png"
        _write_png(path, photo, "RGB")
        try:
            validate_and_load_mask(path, FLOORPLAN_CLASSES, IMAGE_SIZE, "floorplan")
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("an RGB upload was accepted")
    assert "class-id mask" in message
    assert "floorplan" in message
    assert str(path) not in message  # never quote a server-side path


def test_validate_and_load_mask_rejects_unreadable_file() -> None:
    """Garbage bytes must become a message about the upload."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "garbage.png"
        path.write_bytes(b"definitely not a png")
        try:
            validate_and_load_mask(path, FLOORPLAN_CLASSES, IMAGE_SIZE, "furniture")
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("garbage bytes were accepted")
    assert "not a readable PNG" in message
    assert "furniture" in message


def test_validate_and_load_mask_rejects_out_of_range_id() -> None:
    """An id above the layout's range must be refused, never clamped."""
    mask = np.full((16, 16), 250, dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mask.png"
        _write_png(path, mask, "L")
        try:
            validate_and_load_mask(path, FLOORPLAN_CLASSES, IMAGE_SIZE, "floorplan")
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("an out-of-range class id was accepted")
    assert "250" in message and "18" in message


def test_build_blank_mask_is_all_background() -> None:
    """The empty layout the encoders read is every pixel background."""
    blank = build_blank_mask((IMAGE_SIZE, IMAGE_SIZE))
    assert blank.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert blank.dtype == torch.int64
    assert int(blank.max()) == 0
    assert int(blank.sum()) == 0


def test_save_upload_as_writes_the_expected_filename() -> None:
    """The staged mask is written where the encoders expect to read it."""
    ids = build_blank_mask((8, 8))
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp) / "sample" / "floorplan_segmentation.png"
        save_upload_as(ids, destination)
        assert destination.is_file()
        with Image.open(destination) as image:
            assert image.mode == "L"
            assert np.array(image).max() == 0


def test_clamp_draws_limits_one_request_cost() -> None:
    """The draw count is clamped into the allowed band on both ends."""
    assert _clamp_draws(0, 4) == 1
    assert _clamp_draws(-7, 4) == 1
    assert _clamp_draws(2, 4) == 2
    assert _clamp_draws(99, 4) == 4
    assert _clamp_draws(4, 1) == 1


def test_prune_old_runs_keeps_the_newest_folders() -> None:
    """Old request folders are removed so the outputs directory cannot grow."""
    import os
    import time as time_module

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "outputs"
        for index in range(4):
            folder = root / f"upload_{index}"
            folder.mkdir(parents=True)
            # Age the folders in creation order, oldest first.
            stamp: float = time_module.time() - (4 - index) * 60
            os.utime(folder, (stamp, stamp))
        _prune_old_runs(root, 2)
        remaining = sorted(path.name for path in root.iterdir())
        assert remaining == ["upload_2", "upload_3"]
        # Negative keeps every folder, and a missing directory is not an error.
        _prune_old_runs(root, -1)
        assert len(list(root.iterdir())) == 2
        _prune_old_runs(root / "absent", 2)


def test_upload_mask_roles_name_each_field_once() -> None:
    """Each form field maps to exactly one staged filename."""
    assert set(UPLOAD_MASKS) == {"floorplan", "furniture"}
    assert len(set(UPLOAD_MASKS.values())) == len(UPLOAD_MASKS)


def _write_text(path: Path, text: str) -> None:
    """Write one UTF-8 text fixture.

    Args:
        path (Path): Destination file.
        text (str): File contents.
    """
    path.write_text(text, encoding="utf-8")


def test_read_ground_truth_labels_accepts_both_line_forms() -> None:
    """Full YOLO lines and class-less lines both parse, in file order."""
    text = "0 0.5 0.5 0.2 0.2\n\n0.25 0.25 0.1 0.1\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.txt"
        _write_text(path, text)
        boxes = read_ground_truth_labels(path, class_id=0, max_boxes=10)
    assert len(boxes) == 2
    assert boxes[0] == Box(cx=0.5, cy=0.5, w=0.2, h=0.2)
    assert boxes[1] == Box(cx=0.25, cy=0.25, w=0.1, h=0.1)


def test_read_ground_truth_labels_accepts_dataset_component_names() -> None:
    """The dataset's ``Name cx cy w h`` lines score the named component only."""
    text = (
        "Sofa 0.1 0.1 0.1 0.1\n"
        "TV 0.5 0.5 0.2 0.2\n"
        "Chair 0.8 0.8 0.1 0.1\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.txt"
        _write_text(path, text)
        boxes = read_ground_truth_labels(
            path, class_id=0, max_boxes=10, label_names=("TV",)
        )
    assert boxes == [Box(cx=0.5, cy=0.5, w=0.2, h=0.2)]


def test_read_ground_truth_layout_separates_target_and_other_furniture() -> None:
    """A raw dataset labels file yields both the target and the rest of the layout."""
    text = (
        "Sofa 0.1 0.1 0.1 0.1\n"
        "TV 0.5 0.5 0.2 0.2\n"
        "Chair 0.8 0.8 0.1 0.1\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.txt"
        _write_text(path, text)
        layout = read_ground_truth_layout(
            path, class_id=0, max_boxes=10, label_names=("TV",)
        )
    assert layout.target_boxes == (Box(cx=0.5, cy=0.5, w=0.2, h=0.2),)
    assert layout.other_boxes == (
        Box(cx=0.1, cy=0.1, w=0.1, h=0.1),
        Box(cx=0.8, cy=0.8, w=0.1, h=0.1),
    )


def test_read_ground_truth_labels_named_form_keeps_numeric_class_strict() -> None:
    """A numeric line for another class is still refused when names are known."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.txt"
        _write_text(path, "1 0.5 0.5 0.2 0.2\n")
        try:
            read_ground_truth_labels(
                path, class_id=0, max_boxes=10, label_names=("TV",)
            )
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("labels for another numeric class were accepted")
    assert "class id 1" in message
    assert "TV" in message


def test_read_ground_truth_labels_empty_file_is_a_negative_sample() -> None:
    """A blank labels.txt is a legitimate negative, not a parse error."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.txt"
        _write_text(path, "\n   \n")
        boxes = read_ground_truth_labels(path, class_id=0, max_boxes=10)
    assert boxes == []


def test_read_ground_truth_labels_rejects_another_class() -> None:
    """Labels for another component must be refused, not silently dropped."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.txt"
        _write_text(path, "1 0.5 0.5 0.2 0.2\n")
        try:
            read_ground_truth_labels(path, class_id=0, max_boxes=10)
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("labels for another class were accepted")
    assert "class id 1" in message
    assert "line 1" in message


def test_read_ground_truth_labels_rejects_wrong_field_count() -> None:
    """A malformed line names the expected YOLO forms."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.txt"
        _write_text(path, "0 0.5 0.5\n")
        try:
            read_ground_truth_labels(path, class_id=0, max_boxes=10)
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("a short label line was accepted")
    assert "5 fields" in message and "4 fields" in message


def test_read_ground_truth_labels_rejects_out_of_range_values() -> None:
    """Normalized coordinates outside 0..1 are a format error."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.txt"
        _write_text(path, "0 1.5 0.5 0.2 0.2\n")
        try:
            read_ground_truth_labels(path, class_id=0, max_boxes=10)
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("an out-of-range box was accepted")
    assert "cx=1.5" in message and "0..1" in message


def test_read_ground_truth_labels_rejects_zero_area_and_nonfinite() -> None:
    """Zero-area and NaN boxes cannot produce a meaningful IoU."""
    cases = {
        "zero": "0 0.5 0.5 0.0 0.2\n",
        "nan": "0 0.5 nan 0.2 0.2\n",
    }
    for name, text in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.txt"
            _write_text(path, text)
            try:
                read_ground_truth_labels(path, class_id=0, max_boxes=10)
            except ValueError:
                pass
            else:  # pragma: no cover - failure path
                raise AssertionError(f"the {name} label line was accepted")


def test_read_ground_truth_labels_enforces_the_box_cap() -> None:
    """One upload cannot turn the page into an unbounded table."""
    text = "0 0.5 0.5 0.2 0.2\n" * 3
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.txt"
        _write_text(path, text)
        try:
            read_ground_truth_labels(path, class_id=0, max_boxes=2)
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("more boxes than the cap were accepted")
    assert "more than 2 boxes" in message


def test_read_ground_truth_labels_rejects_binary_content() -> None:
    """A binary file uploaded into the labels field gets a text message."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.txt"
        path.write_bytes(b"\xff\xfe\x00\x00")
        try:
            read_ground_truth_labels(path, class_id=0, max_boxes=10)
        except ValueError as exc:
            message = str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("binary content was accepted as labels")
    assert "UTF-8" in message


def test_compare_boxes_perfect_match_reports_full_agreement() -> None:
    """An identical box has IoU 1, no FP/FN and every rate at its best value."""
    gt = [Box(cx=0.5, cy=0.5, w=0.2, h=0.2)]
    prediction = [PredictedBox(box=Box(cx=0.5, cy=0.5, w=0.2, h=0.2), confidence=0.9)]
    comparison = compare_boxes(gt, prediction, iou_threshold=0.5, image_size=256)
    assert comparison.status == STATUS_DETECTED_ALL
    assert comparison.matched_boxes == 1
    assert comparison.false_positive_boxes == 0
    assert comparison.missed_boxes == 0
    assert comparison.precision == 1.0
    assert comparison.recall == 1.0
    assert comparison.f1 == 1.0
    assert comparison.jaccard == 1.0
    assert comparison.best_iou == 1.0
    assert comparison.matched_iou is not None
    assert comparison.matched_iou.mean == 1.0
    assert comparison.center_distance_px is not None
    assert comparison.center_distance_px.mean == 0.0
    assert comparison.mean_tp_confidence == 0.9
    assert comparison.other_ground_truth_boxes == 0
    assert comparison.overlap == 0.0
    assert comparison.pairs[0].prediction is not None


def test_compare_boxes_counts_partial_match_and_false_positive() -> None:
    """A matched box, a missed box and an invented box are counted separately."""
    gt = [
        Box(cx=0.3, cy=0.3, w=0.2, h=0.2),
        Box(cx=0.7, cy=0.7, w=0.2, h=0.2),
    ]
    predictions = [
        PredictedBox(box=Box(cx=0.3, cy=0.3, w=0.2, h=0.2), confidence=0.8),
        PredictedBox(box=Box(cx=0.05, cy=0.05, w=0.05, h=0.05), confidence=0.4),
    ]
    comparison = compare_boxes(gt, predictions, iou_threshold=0.5, image_size=256)
    assert comparison.status == STATUS_DETECTED_PARTIAL
    assert comparison.matched_boxes == 1
    assert comparison.false_positive_boxes == 1
    assert comparison.missed_boxes == 1
    assert comparison.precision == 0.5
    assert comparison.recall == 0.5
    assert comparison.f1 == 0.5
    assert comparison.jaccard == 1.0 / 3.0
    assert comparison.pairs[1].prediction is None
    assert comparison.false_positives[0].best_iou is not None
    assert comparison.false_positives[0].best_iou < 0.5


def test_compare_boxes_reports_overlap_with_other_furniture() -> None:
    """The final target box is measured against the rest of the uploaded layout."""
    gt = [Box(cx=0.1, cy=0.1, w=0.1, h=0.1)]
    predictions = [
        PredictedBox(box=Box(cx=0.5, cy=0.5, w=0.2, h=0.2), confidence=0.9)
    ]
    other = [Box(cx=0.5, cy=0.5, w=0.1, h=0.1)]
    comparison = compare_boxes(
        gt,
        predictions,
        iou_threshold=0.5,
        image_size=256,
        other_ground_truth=other,
    )
    assert comparison.other_ground_truth_boxes == 1
    assert comparison.overlap is not None
    assert abs(comparison.overlap - 0.25) < 1e-9


def test_compare_boxes_overlap_is_none_without_a_target_prediction() -> None:
    """A miss has no final box, so overlap is undefined rather than a good zero."""
    comparison = compare_boxes(
        [Box(cx=0.5, cy=0.5, w=0.2, h=0.2)],
        [],
        iou_threshold=0.5,
        image_size=256,
        other_ground_truth=[Box(cx=0.5, cy=0.5, w=0.1, h=0.1)],
    )
    assert comparison.overlap is None


def test_compare_boxes_all_missed_is_not_a_match() -> None:
    """A prediction below the IoU threshold is both a miss and a false positive."""
    gt = [Box(cx=0.5, cy=0.5, w=0.2, h=0.2)]
    predictions = [
        PredictedBox(box=Box(cx=0.05, cy=0.05, w=0.1, h=0.1), confidence=0.7)
    ]
    comparison = compare_boxes(gt, predictions, iou_threshold=0.5, image_size=256)
    assert comparison.status == STATUS_MISSED
    assert comparison.matched_boxes == 0
    assert comparison.missed_boxes == 1
    assert comparison.false_positive_boxes == 1
    assert comparison.matched_iou is None
    assert comparison.best_iou == 0.0


def test_compare_boxes_distinguishes_false_alarm_from_correct_empty() -> None:
    """With no ground truth, any prediction is a false alarm and none is correct."""
    prediction = [PredictedBox(box=Box(cx=0.5, cy=0.5, w=0.2, h=0.2), confidence=0.6)]
    false_alarm = compare_boxes([], prediction, iou_threshold=0.5, image_size=256)
    correct_empty = compare_boxes([], [], iou_threshold=0.5, image_size=256)
    assert false_alarm.status == STATUS_FALSE_ALARM
    assert false_alarm.precision == 0.0
    assert false_alarm.recall is None
    assert false_alarm.overlap == 0.0
    assert correct_empty.status == STATUS_CORRECT_EMPTY
    assert correct_empty.precision is None
    assert correct_empty.f1 is None
    assert correct_empty.overlap is None


def test_compare_boxes_rejects_invalid_parameters() -> None:
    """Bad thresholds and image sizes fail instead of producing silent numbers."""
    try:
        compare_boxes([], [], iou_threshold=1.5, image_size=256)
    except ValueError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("an out-of-range IoU threshold was accepted")
    try:
        compare_boxes([], [], iou_threshold=0.5, image_size=0)
    except ValueError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("a non-positive image size was accepted")


def test_format_metric_shows_undefined_values_as_na() -> None:
    """None is displayed as n/a, never as a misleading zero."""
    assert format_metric(None) == "n/a"
    assert format_metric(0.123456, 3) == "0.123"
    assert format_metric(0, 2) == "0.00"


def test_result_template_renders_both_comparison_branches() -> None:
    """The result page renders metrics with labels and a clear note without them."""
    app = create_app("debug")
    comparison = compare_boxes(
        [Box(cx=0.5, cy=0.5, w=0.2, h=0.2)],
        [PredictedBox(box=Box(cx=0.5, cy=0.5, w=0.2, h=0.2), confidence=0.9)],
        iou_threshold=0.5,
        image_size=256,
    )
    common = {
        "request_id": "upload_test",
        "panels": {
            "floorplan": "floorplan.png",
            "conditioned": "conditioned.png",
            "component_ensemble": "component_ensemble.png",
            "composite_ensemble": "composite_ensemble.png",
            "ground_truth_ensemble": "ground_truth_ensemble.png",
            "boxes_ensemble": "boxes_ensemble.png",
            "component_draw_0": "component.png",
            "composite_draw_0": "composite.png",
            "ground_truth_draw_0": "ground_truth.png",
            "boxes_draw_0": "boxes.png",
        },
        "component": "tv",
        "component_label": "TV",
        "draws": 4,
        "ensemble_threshold": 0.5,
        "generated_pixels": 10,
        "elapsed_seconds": 1.0,
        "seed": 1234,
        "conf": 0.25,
        "match_iou": 0.5,
        "image_size": 256,
        "checkpoint": "checkpoint.pt",
        "weights": "weights.pt",
    }
    with app.app_context():
        with_labels = render_template(
            "result.html",
            detections=[
                {
                    "draw": None,
                    "is_ensemble": True,
                    "panel_suffix": "ensemble",
                    "members": 4,
                    "found": [],
                    "comparison": comparison,
                },
                {
                    "draw": 0,
                    "is_ensemble": False,
                    "panel_suffix": "draw_0",
                    "members": None,
                    "found": [],
                    "comparison": comparison,
                },
            ],
            ground_truth_uploaded=True,
            ground_truth_boxes=1,
            **common,
        )
        without_labels = render_template(
            "result.html",
            detections=[
                {
                    "draw": None,
                    "is_ensemble": True,
                    "panel_suffix": "ensemble",
                    "members": 4,
                    "found": [],
                    "comparison": None,
                },
                {
                    "draw": 0,
                    "is_ensemble": False,
                    "panel_suffix": "draw_0",
                    "members": None,
                    "found": [],
                    "comparison": None,
                },
            ],
            ground_truth_uploaded=False,
            ground_truth_boxes=0,
            **common,
        )
    assert "Detection metrics" in with_labels
    assert "Ensemble (4 chains)" in with_labels
    assert "Draw 0 (diagnostic)" in with_labels
    assert "ground truth only (green overlay)" in with_labels
    assert "Mean matched IoU" in with_labels
    assert "Overlap with other GT furniture" in with_labels
    assert "nothing to score" in without_labels


def test_validate_inpaint_edit_accepts_erasing_component_pixels() -> None:
    """Erasing a TV/showcase/dining-set pixel to background is a legal edit."""
    original = torch.tensor(
        [
            [2, 5, 3, 1],
            [0, 2, 5, 3],
            [1, 0, 2, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.int64,
    )
    edited = original.clone()
    edited[0, 0] = 0  # erase TV
    edited[0, 1] = 0  # erase showcase
    edited[0, 2] = 0  # erase dining set
    validate_inpaint_edit(original, edited, {2, 3, 5})
    # A no-op edit is also legal.
    validate_inpaint_edit(original, original.clone(), {2, 3, 5})


def test_validate_inpaint_edit_rejects_protected_changes() -> None:
    """A sofa cannot be erased and a component cannot be relabelled."""
    original = torch.tensor([[1, 2, 5]], dtype=torch.int64)
    erased_sofa = torch.tensor([[0, 2, 5]], dtype=torch.int64)
    relabelled_tv = torch.tensor([[1, 1, 5]], dtype=torch.int64)
    for edited in (erased_sofa, relabelled_tv):
        try:
            validate_inpaint_edit(original, edited, {2, 3, 5})
        except ValueError as exc:
            assert "may only erase" in str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("a protected pixel change was accepted")


def test_merge_component_mask_protects_original_furniture() -> None:
    """Generated pixels fill background but never overwrite another class."""
    base = torch.tensor(
        [
            [2, 0, 1],
            [0, 0, 0],
        ],
        dtype=torch.int64,
    )
    generated = torch.tensor(
        [
            [True, True, True],
            [False, False, True],
        ]
    )
    protected = merge_component_mask(base, 2, generated, protect_original=True)
    assert protected.tolist() == [
        [True, True, False],
        [False, False, True],
    ]
    literal = merge_component_mask(base, 2, generated, protect_original=False)
    assert literal.tolist() == [
        [True, True, True],
        [False, False, True],
    ]


def test_merge_inpainted_components_uses_probability_on_overlap() -> None:
    """A background pixel goes to the generated component with the higher vote."""
    base = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=torch.int64,
    )
    tv = torch.tensor(
        [
            [True, True, False],
            [False, False, True],
        ]
    )
    showcase = torch.tensor(
        [
            [True, False, False],
            [True, False, True],
        ]
    )
    tv_probability = torch.tensor(
        [
            [0.9, 0.9, 0.0],
            [0.0, 0.0, 0.2],
        ]
    )
    showcase_probability = torch.tensor(
        [
            [0.1, 0.0, 0.0],
            [0.8, 0.0, 0.3],
        ]
    )
    merged = merge_inpainted_components(
        base,
        {"tv": 2, "showcase": 5},
        {2: tv, 5: showcase},
        {2: tv_probability, 5: showcase_probability},
        protect_original=True,
    )
    # (0,0) is an overlap: TV's 0.9 wins. (0,1) is TV only. (0,2) and (1,0)
    # are showcase only. (1,1) is an original sofa, so it survives even though
    # showcase generated there.
    assert merged.tolist() == [
        [2, 2, 0],
        [5, 1, 5],
    ]


def test_merge_inpainted_components_literal_union_can_overwrite() -> None:
    """Without protection, a generated component may claim another class."""
    base = torch.tensor([[1, 0]], dtype=torch.int64)
    tv = torch.tensor([[True, False]])
    showcase = torch.tensor([[True, False]])
    merged = merge_inpainted_components(
        base,
        {"tv": 2, "showcase": 5},
        {2: tv, 5: showcase},
        {
            2: torch.tensor([[0.9, 0.0]]),
            5: torch.tensor([[0.8, 0.0]]),
        },
        protect_original=False,
    )
    # The sofa at (0, 0) is not TV's own channel, so the literal union lets the
    # higher-probability TV reconstruction replace it.
    assert merged.tolist() == [[2, 0]]


def test_load_furniture_palette_and_erasable_channels() -> None:
    """The palette is sorted by id and maps registry names to class ids."""
    text = (
        '{"0": {"name": "background", "color": "#000000"}, '
        '"5": {"name": "showcase", "color": "#34d399"}, '
        '"2": {"name": "tv", "color": "#a78bfa"}, '
        '"3": {"name": "dining_set", "color": "#fb923c"}}'
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "furniture_color_index.json"
        path.write_text(text, encoding="utf-8")
        palette = load_furniture_palette(path)
    assert [entry["id"] for entry in palette] == [0, 2, 3, 5]
    assert palette[1]["rgb"] == (167, 139, 250)
    assert erasable_channels_from_palette(
        palette, ["tv", "showcase", "dining_set"]
    ) == {2, 3, 5}


def test_decode_edited_ids_roundtrip_and_rejects_bad_payloads() -> None:
    """The browser's raw bytes become the exact class-id tensor."""
    size = 4
    ids = torch.arange(size * size, dtype=torch.uint8).reshape(size, size)
    encoded = base64.b64encode(ids.numpy().tobytes()).decode("ascii")
    decoded = _decode_edited_ids(encoded, size)
    assert torch.equal(decoded, ids.to(torch.int64))
    for bad in ("", "not base64", base64.b64encode(b"\x00\x01").decode("ascii")):
        try:
            _decode_edited_ids(bad, size)
        except ValueError:
            pass
        else:  # pragma: no cover - failure path
            raise AssertionError("a malformed edited mask was accepted")


def test_inpaint_templates_render() -> None:
    """Both inpainting pages render with the payload the routes provide."""
    app = create_app("debug")
    palette = [
        {"id": 0, "name": "background", "rgb": [0, 0, 0], "hex": "#000000"},
        {"id": 2, "name": "tv", "rgb": [167, 139, 250], "hex": "#a78bfa"},
        {"id": 3, "name": "dining_set", "rgb": [251, 146, 60], "hex": "#fb923c"},
        {"id": 5, "name": "showcase", "rgb": [52, 211, 153], "hex": "#34d399"},
    ]
    with app.app_context():
        editor = render_template(
            "inpaint.html",
            components=["tv", "showcase", "dining_set"],
            canvas_size=256,
            palette=palette,
            erasable_channels=[2, 3, 5],
            brush_min=2,
            brush_max=48,
            brush_default=12,
            erase_overlay_alpha=0.55,
            undo_limit=30,
            max_draws=4,
            default_draws=4,
            ensemble_threshold=0.5,
            device="cpu",
            error=None,
        )
        result = render_template(
            "inpaint_result.html",
            request_id="inpaint_test",
            panels={
                "floorplan": "floorplan.png",
                "original_furniture": "original.png",
                "edited_furniture": "edited.png",
                "erase_overlay": "erase.png",
                "merged_furniture": "merged.png",
                "merged_composite": "merged_composite.png",
                "merged_boxes": "merged_boxes.png",
            },
            components=[
                {
                    "name": "tv",
                    "label": "TV",
                    "channel": 2,
                    "panels": {
                        "generated": "tv_generated.png",
                        "final": "tv_final.png",
                        "composite": "tv_composite.png",
                        "boxes": "tv_boxes.png",
                    },
                    "generated_pixels": 10,
                    "preserved_pixels": 2,
                    "final_pixels": 12,
                    "draws": [
                        {
                            "index": 0,
                            "panel": "tv_draw_0.png",
                            "generated_pixels": 9,
                        }
                    ],
                    "found": [],
                }
            ],
            merged_found=[],
            draws=4,
            ensemble_threshold=0.5,
            elapsed_seconds=1.0,
            image_size=256,
            protect_original=True,
            seed=1234,
            conf=0.25,
        )
    assert "Reconstruct and merge" in editor
    assert "const ERASABLE" in editor
    assert "Merged layout" in result
    assert "Merge rule" in result
    assert "Per-draw diagnostics" in result
    assert "Original furniture is protected" in result


def test_inpaint_edited_field_is_not_a_mask_field() -> None:
    """The base64 edit field cannot collide with an uploaded mask field."""
    assert INPAINT_EDITED_FIELD not in UPLOAD_MASKS
    assert INPAINT_EDITED_FIELD not in set(UPLOAD_MASKS.values())


def test_ground_truth_upload_constants_are_distinct() -> None:
    """The label field cannot collide with a mask field or its filename."""
    assert GROUND_TRUTH_FIELD not in UPLOAD_MASKS
    assert GROUND_TRUTH_LABELS_FILENAME not in set(UPLOAD_MASKS.values())


def _run_all_tests() -> int:
    """Run every ``test_*`` function in this module, reporting failures.

    Returns:
        int: Number of failed tests, ``0`` when all pass.
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
    return len(failures)


if __name__ == "__main__":
    raise SystemExit(1 if _run_all_tests() else 0)
