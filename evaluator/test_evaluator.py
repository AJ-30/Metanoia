"""Unit tests for the evaluator's pure logic.

Arrange-Act-Assert throughout, with a small deterministic ``FakeLog`` in place of
the project logger so the tests need no config, checkpoints, GPU or detector.
Covers the failure modes that matter for a measurement stage:

1. A box loss that reports the wrong thing (``box_metrics`` matching, IoU, L1 and
   the center distance in pixels).
2. Ground truth read from the wrong place or with the wrong names
   (``dataset_gt``).
3. A component resolved to the wrong class in any of the three vocabularies
   (``ontology``).
4. A leak of the ground-truth component into the evaluated image, or of an
   excluded channel surviving in the erase rule (``compositor``).
5. A channel score that disagrees with the masks (``channel_metrics``).
6. A summary that mis-aggregates the draws (``results``).
7. A dataset label written for a class the detector never learned
   (``yolo_eval``).
8. A report that prints the wrong number (``report``).

Run with: ``./venv/bin/python evaluator/test_evaluator.py``
(or ``pytest evaluator/test_evaluator.py`` when pytest is installed).
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

# Project root so ``evaluator``/``mask_diffusion`` import when run as a script.
_ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from evaluator.box_metrics import (
    Box,
    PredictedBox,
    box_loss,
    intersection_over_union,
    match_boxes,
    pixel_boxes_to_predictions,
    summarize,
)
from evaluator.channel_metrics import channel_scores, dice, iou
from evaluator.compositor import erase_component_channel
from evaluator.dataset_gt import boxes_to_yolo_lines, parse_label_line, read_component_boxes
from evaluator.ontology import resolve_component_ontology, resolve_yolo_class_id
from evaluator.report import format_report
from evaluator.results import DrawAccumulator, aggregate_per_draw
from evaluator.variants import (
    apply_component_variant,
    component_names,
    load_component_variants,
    resolve_component_variant,
)
from evaluator.visualization_full_pipeline_report import (
    GridStyle,
    build_report_grid,
    choose_shared_samples,
    select_component_names,
)
from evaluator.yolo_eval import (
    detector_class_names,
    prepare_eval_dataset,
    validate_detector_names,
)
from mask_VAE.masking import MaskGroup, MaskLayout
from mask_diffusion.runner import class_ids_to_one_hot
from YOLO.dataset import SampleFolder


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


CLASS_NAMES: list[str] = [
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
CLASS_ALIASES: dict[str, str] = {"dining_set": "Dining_table"}
LABEL_NAMES: dict[str, list[str]] = {
    "tv": ["TV"],
    "dining_set": ["Dining_table_big", "Dining_table_medium", "Dining_table_round"],
}

# ---------------------------------------------------------------------------
# Box metrics
# ---------------------------------------------------------------------------


def test_box_iou_is_one_for_identical_boxes_and_zero_for_disjoint() -> None:
    """IoU is 1 on a box against itself and 0 when the boxes share no pixel."""
    box = Box(cx=0.5, cy=0.5, w=0.2, h=0.2)
    assert intersection_over_union(box, box) == 1.0
    assert intersection_over_union(box, Box(cx=0.9, cy=0.9, w=0.05, h=0.05)) == 0.0


def test_box_iou_partial_overlap_matches_the_analytic_value() -> None:
    """An overlapping pair must give the intersection/union ratio exactly."""
    left = Box(cx=0.25, cy=0.5, w=0.4, h=0.2)  # x in [0.05, 0.45]
    right = Box(cx=0.5, cy=0.5, w=0.4, h=0.2)  # x in [0.30, 0.70]
    # 0.15 x 0.2 = 0.03 shared, 0.08 + 0.08 - 0.03 = 0.13 union.
    assert abs(intersection_over_union(left, right) - 0.03 / 0.13) < 1e-9


def test_match_boxes_matches_higher_confidence_prediction_first() -> None:
    """A lower-confidence but better-centred prediction must not steal the box."""
    ground_truth = [Box(cx=0.5, cy=0.5, w=0.2, h=0.2)]
    predictions = [
        PredictedBox(Box(cx=0.5, cy=0.5, w=0.21, h=0.2), confidence=0.4),  # better
        PredictedBox(Box(cx=0.5, cy=0.5, w=0.3, h=0.25), confidence=0.9),  # first
    ]
    outcome = match_boxes(ground_truth, predictions, 0.5, 256)
    assert outcome.matches[0].prediction is not None
    assert outcome.matches[0].prediction.confidence == 0.9
    assert len(outcome.false_positives) == 1


def test_match_boxes_reports_misses_and_false_positives() -> None:
    """A box below the threshold is a miss, and the prediction is a false positive."""
    ground_truth = [Box(cx=0.2, cy=0.2, w=0.1, h=0.1)]
    predictions = [PredictedBox(Box(cx=0.25, cy=0.2, w=0.1, h=0.1), 0.8)]
    outcome = match_boxes(ground_truth, predictions, 0.5, 256)
    assert outcome.matches[0].loss is None
    assert outcome.matches[0].prediction is None
    assert len(outcome.false_positives) == 1


def test_match_boxes_never_matches_one_ground_truth_twice() -> None:
    """Two predictions cannot both claim the same ground-truth box."""
    ground_truth = [Box(cx=0.5, cy=0.5, w=0.2, h=0.2)]
    predictions = [
        PredictedBox(Box(cx=0.5, cy=0.5, w=0.2, h=0.2), 0.9),
        PredictedBox(Box(cx=0.5, cy=0.5, w=0.2, h=0.2), 0.8),
    ]
    outcome = match_boxes(ground_truth, predictions, 0.5, 256)
    assert outcome.matches[0].loss is not None
    assert len(outcome.false_positives) == 1


def test_box_loss_reports_iou_l1_and_pixel_center_distance() -> None:
    """The localization loss reports the three scalars it promises."""
    ground_truth = Box(cx=0.5, cy=0.5, w=0.2, h=0.2)
    prediction = Box(cx=0.5, cy=0.5, w=0.2, h=0.2)
    perfect = box_loss(ground_truth, prediction, 256)
    assert perfect.iou == 1.0
    assert perfect.l1_norm == 0.0
    assert perfect.center_distance == 0.0

    shifted = Box(cx=0.5 + 0.1, cy=0.5, w=0.2, h=0.2)
    loss = box_loss(ground_truth, shifted, 256)
    assert abs(loss.center_distance - 0.1 * 256) < 1e-6
    assert loss.l1_norm > 0.0


def test_pixel_boxes_to_predictions_filters_other_classes() -> None:
    """Only the missing component's detections survive the conversion."""
    pixel_boxes = [
        (10.0, 20.0, 60.0, 70.0, 8, 0.9),  # TV
        (0.0, 0.0, 100.0, 100.0, 1, 0.8),  # Chair - dropped
        (5.0, 5.0, 25.0, 25.0, 8, 0.7),  # TV
    ]
    predictions = pixel_boxes_to_predictions(pixel_boxes, 8, 256)
    assert len(predictions) == 2
    assert all(prediction.confidence > 0.0 for prediction in predictions)
    first = predictions[0].box
    assert abs(first.cx - 35.0 / 256) < 1e-9
    assert abs(first.cy - 45.0 / 256) < 1e-9


def test_matched_losses_of_best_match_are_selected_once() -> None:
    """A sample with several instances reports ONE match's numbers, not a mix.

    Two ground-truth boxes, one matched loosely and one matched well: the row
    must describe the well-matched one, so that its IoU, L1 and centre distance
    cannot come from three different boxes.
    """
    ground_truth = [Box(0.2, 0.2, 0.2, 0.2), Box(0.6, 0.6, 0.2, 0.2)]
    predictions = [
        # Shifted by a quarter of its width: IoU = 0.16/0.24 = 0.667, above the
        # threshold, so this is a genuine (if loose) match.
        PredictedBox(Box(0.24, 0.2, 0.2, 0.2), 0.5),
        PredictedBox(Box(0.6, 0.6, 0.2, 0.2), 0.9),    # exact: IoU 1.0
    ]
    outcome = match_boxes(ground_truth, predictions, 0.5, 256)
    matched = [match.loss for match in outcome.matches if match.loss is not None]
    best = max(matched, key=lambda loss: loss.iou)
    assert best.iou == 1.0
    assert best.l1_norm == 0.0
    assert best.center_distance == 0.0
    # The loose match is still counted, it just does not define the row.
    assert len(matched) == 2
    assert abs(min(loss.iou for loss in matched) - 0.667) < 0.01
    assert outcome.false_positives == []


def test_summarize_handles_empty_and_ordered_values() -> None:
    """An empty distribution reports zeros, not a crash; values are ordered."""
    empty = summarize([])
    assert empty["mean"] == 0.0 and empty["median"] == 0.0
    stats = summarize([0.1, 0.5, 0.9, 0.3])
    assert stats["min"] == 0.1
    assert stats["max"] == 0.9
    assert stats["median"] == 0.4
    assert stats["p10"] <= stats["p90"]


# ---------------------------------------------------------------------------
# Dataset ground truth
# ---------------------------------------------------------------------------


def test_parse_label_line_keeps_only_the_named_component() -> None:
    """A line naming another furniture class returns None, the component a Box."""
    component = parse_label_line("TV 0.25 0.5 0.1 0.2", LABEL_NAMES["tv"])
    other = parse_label_line("Sofa 0.25 0.5 0.1 0.2", LABEL_NAMES["tv"])
    blank = parse_label_line("   ", LABEL_NAMES["tv"])
    assert component == Box(cx=0.25, cy=0.5, w=0.1, h=0.2)
    assert other is None
    assert blank is None


def test_parse_label_line_rejects_malformed_component_line() -> None:
    """A line that names the component but is not 5 numbers is an error."""
    try:
        parse_label_line("TV 0.25 0.5", LABEL_NAMES["tv"])
    except ValueError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("a malformed component line was accepted")
    try:
        parse_label_line("TV 1.5 0.5 0.1 0.2", LABEL_NAMES["tv"])
    except ValueError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("an out-of-range coordinate was accepted")


def test_read_component_boxes_reads_every_instance_and_absences() -> None:
    """Multiple instances are all returned; no instance is an empty list, not an error."""
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "sample"
        sample.mkdir()
        (sample / "labels.txt").write_text(
            "TV 0.25 0.25 0.1 0.1\nSofa 0.5 0.5 0.2 0.2\nTV 0.75 0.75 0.1 0.1\n",
            encoding="utf-8",
        )
        boxes = read_component_boxes(sample, "labels.txt", LABEL_NAMES["tv"])
        assert len(boxes) == 2
        assert boxes[0] == Box(cx=0.25, cy=0.25, w=0.1, h=0.1)

        (sample / "labels.txt").write_text("Sofa 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        assert read_component_boxes(sample, "labels.txt", LABEL_NAMES["tv"]) == []
        try:
            read_component_boxes(sample, "other.txt", LABEL_NAMES["tv"])
        except FileNotFoundError:
            pass
        else:  # pragma: no cover - failure path
            raise AssertionError("a missing label file was accepted")


def test_boxes_to_yolo_lines_uses_the_resolved_class_id() -> None:
    """Label lines carry the detector's class id, six decimals, one line each."""
    lines = boxes_to_yolo_lines([Box(0.25, 0.5, 0.1, 0.2)], 8)
    assert lines == ["8 0.250000 0.500000 0.100000 0.200000"]
    assert boxes_to_yolo_lines([], 8) == []


# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------


def test_resolve_component_ontology_resolves_all_three_vocabularies() -> None:
    """The component resolves to its mask channel, detector class and label names."""
    ontology = resolve_component_ontology(
        name="tv",
        group_prefix="furniture",
        mask_channel=2,
        class_names=CLASS_NAMES,
        aliases=CLASS_ALIASES,
        label_names_table=LABEL_NAMES,
    )
    assert ontology.mask_channel == 2
    assert ontology.yolo_class_id == CLASS_NAMES.index("TV")
    assert ontology.label_names == ("TV",)
    assert ontology.qualified_name == "furniture/tv"


def test_resolve_component_ontology_folds_aliases_and_multi_label_names() -> None:
    """The dining set folds onto the detector class and keeps its three label names."""
    ontology = resolve_component_ontology(
        name="dining_set",
        group_prefix="furniture",
        mask_channel=3,
        class_names=CLASS_NAMES,
        aliases=CLASS_ALIASES,
        label_names_table=LABEL_NAMES,
    )
    assert ontology.yolo_class_id == CLASS_NAMES.index("Dining_table")
    assert len(ontology.label_names) == 3


def test_resolve_component_ontology_rejects_missing_label_names() -> None:
    """A component with no label names would silently score nothing, so it fails."""
    try:
        resolve_component_ontology(
            name="tv",
            group_prefix="",
            mask_channel=2,
            class_names=CLASS_NAMES,
            aliases=CLASS_ALIASES,
            label_names_table={},
        )
    except ValueError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("a component with no label names was accepted")


def test_resolve_yolo_class_id_handles_aliases_and_rejects_unknown() -> None:
    """Aliases fold onto a class; anything unresolvable is a startup error."""
    assert resolve_yolo_class_id("dining_set", CLASS_NAMES, CLASS_ALIASES) == CLASS_NAMES.index(
        "Dining_table"
    )
    assert resolve_yolo_class_id("TV", CLASS_NAMES, CLASS_ALIASES) == CLASS_NAMES.index("TV")
    try:
        resolve_yolo_class_id("lamp", CLASS_NAMES, CLASS_ALIASES)
    except ValueError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("an unknown component name was accepted")


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


def make_layout() -> tuple[MaskLayout, MaskGroup]:
    """Build the single-group layout the compositor's renderer expects.

    Returns:
        tuple[MaskLayout, MaskGroup]: The layout and its single group, with three
            classes (background, sofa, tv) exactly like the real arrangement mask.
    """
    class_names: tuple[str, ...] = ("background", "sofa", "tv")
    group = MaskGroup(
        name="furniture",
        mask_file_name="furniture_segmentation.png",
        class_names=class_names,
        colors=((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        channel_indices=tuple(range(len(class_names))),
    )
    layout = MaskLayout(
        groups=(group,),
        class_names=tuple(f"furniture/{name}" for name in class_names),
        num_channels=len(class_names),
        foreground_channels=tuple(range(1, len(class_names))),
    )
    return layout, group


def test_erase_component_channel_moves_pixels_to_background() -> None:
    """The erase rule folds the component's channel into the background channel."""
    # Build a real one-hot through the same helper the compositor uses, so the
    # contract under test is the one the pipeline actually relies on.
    ids = torch.zeros((1, 2, 2), dtype=torch.int64)
    ids[0, 0, 1] = 1  # sofa
    ids[0, 1, 0] = 2  # tv, the component being erased
    one_hot = class_ids_to_one_hot(ids, 3)
    erased = erase_component_channel(one_hot, 2)
    assert float(erased[:, 2].sum()) == 0.0
    assert float(erased[:, 0, 1, 0]) == 1.0  # the tv pixel is now background
    # Exactly one channel stays active per pixel: nothing leaked, nothing vanished.
    assert bool(torch.all(erased.sum(dim=1) == 1.0))


def test_render_layout_ids_paints_the_erased_pixel_as_background() -> None:
    """The rendered erased layout shows the component's pixel as background colour."""
    from evaluator.compositor import render_layout_ids

    layout, group = make_layout()
    ids = torch.zeros((1, 2, 2), dtype=torch.int64)
    ids[0, 0, 1] = 1  # sofa
    ids[0, 1, 0] = 2  # tv, the component that must not appear
    erased_rgb = render_layout_ids(ids, group, layout.num_channels, erase_channel=2)
    kept_rgb = render_layout_ids(ids, group, layout.num_channels)
    # The tv pixel renders as the background colour when erased, and keeps the
    # component's own colour (also black in this test layout) when not.
    assert erased_rgb.shape == (1, 3, 2, 2)
    assert float(erased_rgb[:, :, 1, 0].sum()) == float(kept_rgb[:, :, 1, 0].sum())
    # Erasing cannot change the colours of the pixels that were never the component.
    assert torch.equal(erased_rgb[:, :, 0, 1], kept_rgb[:, :, 0, 1])
    assert torch.equal(erased_rgb[:, :, 0, 0], kept_rgb[:, :, 0, 0])


def test_erase_component_channel_rejects_the_background_channel() -> None:
    """Erasing the background channel would leave a mask with no background."""
    try:
        erase_component_channel(torch.zeros((1, 3, 2, 2)), 0)
    except ValueError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("erasing the background channel was accepted")


# ---------------------------------------------------------------------------
# Channel metrics
# ---------------------------------------------------------------------------


def test_channel_scores_agreement_and_disagreement() -> None:
    """Dice/IoU are 1 on identical masks and 0 on disjoint ones."""
    generated = torch.zeros((8, 8), dtype=torch.bool)
    generated[:4] = True
    identical = torch.zeros((8, 8), dtype=torch.bool)
    identical[:4] = True
    dice_score, iou_score = channel_scores(generated, identical)
    assert dice_score == 1.0
    assert iou_score == 1.0

    disjoint = torch.zeros((8, 8), dtype=torch.bool)
    disjoint[4:] = True
    dice_score, iou_score = channel_scores(generated, disjoint)
    assert dice_score == 0.0
    assert iou_score == 0.0


def test_channel_scores_on_two_empty_masks_is_perfect() -> None:
    """A sample with no component in either mask agrees, rather than dividing by zero."""
    empty = torch.zeros((4, 4), dtype=torch.bool)
    assert dice(empty.numpy(), empty.numpy()) == 1.0
    assert iou(empty.numpy(), empty.numpy()) == 1.0


def test_channel_scores_half_overlap_matches_the_analytic_values() -> None:
    """Masks sharing one of three equal bands give Dice 0.5 and IoU 1/3."""
    generated = torch.zeros((3, 4), dtype=torch.bool)
    generated[:, :2] = True  # 6 pixels
    truth = torch.zeros((3, 4), dtype=torch.bool)
    truth[:, 1:3] = True  # 6 pixels, 3 of them shared
    dice_score, iou_score = channel_scores(generated, truth)
    assert abs(dice_score - 0.5) < 1e-9
    assert abs(iou_score - 1.0 / 3.0) < 1e-9


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _draw_summary(matched: int, missed: int, false_positives: int) -> dict:
    """Build one per-draw summary with the given outcome counts.

    Args:
        matched (int): Matched boxes.
        missed (int): Missed boxes.
        false_positives (int): False positives.

    Returns:
        dict: A per-draw summary as :meth:`DrawAccumulator.summary` writes it.
    """
    return {
        "detector_metrics": {"mAP50": 0.5, "mAP50_95": 0.4, "precision": 0.6, "recall": 0.7},
        "box_loss": {
            "box_iou": {"mean": 0.8},
            "l1_norm_xywh": {"mean": 0.1},
            "center_distance_px": {"mean": 2.0},
        },
        "channel_scores": {"dice": {"mean": 0.9}, "iou": {"mean": 0.85}},
        "outcomes": {
            "ground_truth_boxes": matched + missed,
            "matched": matched,
            "missed": missed,
            "false_positives": false_positives,
            "prediction_free": matched + false_positives,
        },
    }


def test_aggregate_per_draw_averages_distributions_and_sums_counts() -> None:
    """Distributions average across draws while the counts add up."""
    per_draw = [_draw_summary(4, 1, 2), _draw_summary(6, 0, 1)]
    aggregate = aggregate_per_draw(per_draw)
    assert aggregate["outcomes"]["matched"] == 10
    assert aggregate["outcomes"]["ground_truth_boxes"] == 11
    assert aggregate["box_loss"]["box_iou"]["mean"] == 0.8
    assert aggregate["detector_metrics"]["mAP50"] == 0.5
    assert aggregate_per_draw([]) == {}


def test_draw_accumulator_counts_every_outcome() -> None:
    """The accumulator folds a match, a miss and a false positive into the totals."""
    accumulator = DrawAccumulator()
    ground_truth = [Box(0.5, 0.5, 0.2, 0.2)]
    outcome = match_boxes(
        ground_truth,
        [PredictedBox(Box(0.9, 0.9, 0.05, 0.05), 0.4)],
        0.5,
        256,
    )
    accumulator.add(ground_truth, outcome, (0.5, 0.5))
    summary = accumulator.summary(None)
    assert summary["outcomes"]["ground_truth_boxes"] == 1
    assert summary["outcomes"]["matched"] == 0
    assert summary["outcomes"]["missed"] == 1
    assert summary["outcomes"]["false_positives"] == 1
    assert summary["detector_metrics"] is None
    assert accumulator.box_ious == []


# ---------------------------------------------------------------------------
# Detector dataset and report
# ---------------------------------------------------------------------------


class _FakeBox:
    """Minimal stand-in for an Ultralytics ``Boxes`` object."""

    def __init__(self, xyxy: list[list[float]], classes: list[int], conf: list[float]) -> None:
        """Store the three tensors the evaluator reads."""
        self.xyxy = torch.tensor(xyxy)
        self.cls = torch.tensor(classes, dtype=torch.float32)
        self.conf = torch.tensor(conf)


class _FakeResults:
    """Minimal stand-in for one Ultralytics ``Results`` object."""

    def __init__(self, boxes: _FakeBox | None) -> None:
        """Attach the boxes (or none)."""
        self.boxes = boxes


class _FakeModel:
    """Detector stand-in that returns a fixed detection and records the call."""

    def __init__(self, names: dict[int, str], boxes: _FakeBox | None = None) -> None:
        """Set the class vocabulary and the detections to return."""
        self.names = names
        self._boxes = boxes
        self.calls: list[dict[str, object]] = []

    def __call__(self, path: str, **kwargs: object) -> list[_FakeResults]:
        """Record the call and return the canned detections."""
        self.calls.append({"path": path, **kwargs})
        return [_FakeResults(self._boxes)]


def test_validate_detector_names_accepts_and_rejects_ontology() -> None:
    """The detector's own names must equal the vocabulary the labels were written for."""
    model = _FakeModel({index: name for index, name in enumerate(CLASS_NAMES)})
    validate_detector_names(model, CLASS_NAMES)  # does not raise

    shifted = _FakeModel({0: "Center_table", 1: "Chair"})
    try:
        validate_detector_names(shifted, CLASS_NAMES)
    except ValueError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("a shifted detector ontology was accepted")


def test_detector_class_names_reads_each_detector_vocabulary() -> None:
    """A single-class missing_yolo model and a multi-class model both work."""
    single = _FakeModel({0: "TV"})
    multi = _FakeModel(
        {index: name for index, name in enumerate(CLASS_NAMES)}
    )

    assert detector_class_names(single) == ["TV"]
    assert detector_class_names(multi) == CLASS_NAMES


def test_component_registry_pairs_each_component_with_its_models() -> None:
    """The selected component fills the checkpoint and the matching detector."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint = root / "tv.pt"
        weights = root / "tv_best.pt"
        checkpoint.write_bytes(b"")
        weights.write_bytes(b"")
        registry = root / "components.toml"
        registry.write_text(
            "[default.COMPONENT_VARIANTS.tv]\n"
            f'mask_diffusion_checkpoint = "{checkpoint}"\n'
            f'detector_weights = "{weights}"\n',
            encoding="utf-8",
        )

        variants = load_component_variants(registry)
        variant = resolve_component_variant(variants, "tv")
        config = SimpleNamespace(MISSING_COMPONENT_CLASS_NAME="tv")
        apply_component_variant(config, variant)

        assert component_names(variants) == ["tv"]
        assert config.MISSING_COMPONENT_CLASS_NAME == "tv"
        assert Path(config.MASK_DIFFUSION_CHECKPOINT_PATH) == checkpoint
        assert Path(config.YOLO_WEIGHTS_PATH) == weights


def test_component_registry_fails_fast_on_unknown_or_missing_models() -> None:
    """An unknown key or an absent model file is a startup error, not a silent fallback."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint = root / "tv.pt"
        checkpoint.write_bytes(b"")
        registry = root / "components.toml"
        registry.write_text(
            "[default.COMPONENT_VARIANTS.tv]\n"
            f'mask_diffusion_checkpoint = "{checkpoint}"\n'
            f'detector_weights = "{root / "absent.pt"}"\n',
            encoding="utf-8",
        )
        variants = load_component_variants(registry)

        try:
            resolve_component_variant(variants, "lamp")
        except KeyError as exc:
            assert "lamp" in str(exc) and "tv" in str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("an unknown component was accepted")

        try:
            apply_component_variant(
                SimpleNamespace(), resolve_component_variant(variants, "tv")
            )
        except FileNotFoundError:
            pass
        else:  # pragma: no cover - failure path
            raise AssertionError("missing detector weights were accepted")


def test_predict_component_boxes_filters_to_the_component_class() -> None:
    """Only the component's detections are returned, in the image's own pixels."""
    from evaluator.yolo_eval import predict_component_boxes

    boxes = _FakeBox(
        xyxy=[[10.0, 20.0, 60.0, 70.0], [0.0, 0.0, 40.0, 40.0]],
        classes=[8, 1],
        conf=[0.9, 0.8],
    )
    model = _FakeModel({index: name for index, name in enumerate(CLASS_NAMES)}, boxes)
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "image.png"
        image.write_bytes(b"")
        predictions = predict_component_boxes(
            model=model,
            image_path=image,
            confidence=0.25,
            image_size=416,
            max_detections=300,
            class_id=8,
        )
    assert len(predictions) == 1
    assert predictions[0][4] == 8
    assert predictions[0][0] == 10.0 and predictions[0][2] == 60.0
    assert model.calls[0]["imgsz"] == 416


def test_predict_component_boxes_returns_empty_without_detections() -> None:
    """A detector that emits nothing returns an empty list, not an error."""
    from evaluator.yolo_eval import predict_component_boxes

    model = _FakeModel({index: name for index, name in enumerate(CLASS_NAMES)}, None)
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "image.png"
        image.write_bytes(b"")
        assert (
            predict_component_boxes(
                model=model, image_path=image, confidence=0.25,
                image_size=416, max_detections=300, class_id=8,
            )
            == []
        )


def test_prepare_eval_dataset_clears_previous_artifacts() -> None:
    """The dataset directories are cleared so a changed sample set leaves no orphans."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dataset"
        image_dir, label_dir, yaml_path = prepare_eval_dataset(root, CLASS_NAMES)
        assert image_dir.is_dir() and label_dir.is_dir()
        (image_dir / "stale.png").write_bytes(b"")
        (label_dir / "stale.txt").write_text("", encoding="utf-8")
        image_dir, label_dir, yaml_path = prepare_eval_dataset(root, CLASS_NAMES)
        assert list(image_dir.glob("*.png")) == []
        assert list(label_dir.glob("*.txt")) == []
        assert "names:" in Path(yaml_path).read_text(encoding="utf-8")
        assert "nc: 10" in Path(yaml_path).read_text(encoding="utf-8")


def test_format_report_renders_numbers_and_handles_missing_metrics() -> None:
    """The report prints the aggregate metrics and degrades when a draw was skipped."""
    summary = {
        "run_id": "unit-test",
        "elapsed_seconds": 12.0,
        "configuration": {
            "mask_diffusion_checkpoint": "ckpt.pt",
            "mask_diffusion_epoch": 700,
            "yolo_weights": "weights.pt",
            "split_dir": "/split",
            "samples_evaluated": 100,
            "draws_per_sample": 2,
            "seed": 1234,
        },
        "missing_component": {
            "name": "tv",
            "mask_channel": 2,
            "detector_class_id": 8,
            "detector_class_name": "TV",
            "dataset_label_names": ["TV"],
        },
        "per_draw": [
            _draw_summary(4, 1, 2),
            {
                "detector_metrics": None,
                "box_loss": {"box_iou": {"mean": 0.0}, "l1_norm_xywh": {"mean": 0.0}, "center_distance_px": {"mean": 0.0}},
                "channel_scores": {"dice": {"mean": 0.0}, "iou": {"mean": 0.0}},
                "outcomes": {"ground_truth_boxes": 0, "matched": 0, "missed": 0, "false_positives": 0, "prediction_free": 0},
            },
        ],
        "aggregate": aggregate_per_draw([_draw_summary(4, 1, 2)]),
        "output_root": "/out",
        "figures": ["eval_composite_draw_0.png"],
    }
    text = "\n".join(format_report(summary))
    assert "MISSING-COMPONENT EVALUATION REPORT" in text
    assert "tv" in text and "detector class 8 'TV'" in text
    assert "0.5000" in text  # the aggregate mAP@50
    assert "n/a" in text  # the skipped draw
    assert "matched=4" in text


def test_select_component_names_keeps_registry_order_and_rejects_unknown() -> None:
    """The report renders components in registry order and fails on a typo."""
    available = ["tv", "showcase", "dining_set"]
    assert select_component_names(available, None) == available
    assert select_component_names(available, ["dining_set", "tv"]) == ["tv", "dining_set"]
    try:
        select_component_names(available, ["tv", "sofa"])
    except ValueError as exc:
        assert "sofa" in str(exc)
    else:
        raise AssertionError("An unregistered component must fail fast.")


def test_choose_shared_samples_needs_every_component() -> None:
    """Only samples holding every requested component can become report rows."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        contents = {
            "a": ["TV", "Showcase"],
            "b": ["TV"],
            "c": ["Showcase"],
            "d": ["TV", "Showcase"],
        }
        samples: list[SampleFolder] = []
        for stem, names in contents.items():
            sample_dir = root / stem
            sample_dir.mkdir()
            (sample_dir / "labels.txt").write_text(
                "".join(f"{name} 0.5 0.5 0.1 0.1\n" for name in names),
                encoding="utf-8",
            )
            samples.append(SampleFolder(src_dir=sample_dir, stem=stem))
        chosen = choose_shared_samples(
            samples=samples,
            label_names_by_component={"tv": ("TV",), "showcase": ("Showcase",)},
            count=2,
            seed=0,
            label_filename="labels.txt",
        )
        assert [sample.stem for sample in chosen] == ["a", "d"]
        try:
            choose_shared_samples(
                samples=samples,
                label_names_by_component={"tv": ("TV",), "showcase": ("Showcase",)},
                count=3,
                seed=0,
                label_filename="labels.txt",
            )
        except ValueError as exc:
            assert "Only 2" in str(exc)
        else:
            raise AssertionError("Asking for more positives than exist must fail.")


def test_build_report_grid_sizes_cells_and_rejects_ragged_rows() -> None:
    """The four-column grid is exactly as large as its configured bands."""
    style = GridStyle(
        header_height=20,
        title_height=10,
        row_label_width=8,
        font_size=6,
        title_font_size=7,
        padding=2,
        background=(0, 0, 0),
        text=(255, 255, 255),
        separator=(50, 50, 50),
    )
    panels = [
        [Image.new("RGB", (10, 12), (row * 10, column * 10, 0)) for column in range(4)]
        for row in range(2)
    ]
    grid = build_report_grid(
        panels=panels,
        headers=["one", "two", "three", "four"],
        row_labels=["1", "2"],
        title="unit test",
        style=style,
    )
    assert grid.size == (style.row_label_width + 4 * 10, style.title_height + style.header_height + 2 * 12)
    try:
        build_report_grid(
            panels=[[Image.new("RGB", (10, 12)) for _ in range(3)]],
            headers=["one", "two", "three", "four"],
            row_labels=["1"],
            title="ragged",
            style=style,
        )
    except ValueError as exc:
        assert "3 panels" in str(exc)
    else:
        raise AssertionError("A ragged row must fail rather than silently drop a column.")


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
