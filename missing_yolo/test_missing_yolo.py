"""Unit tests for the missing_yolo generation, labelling and split logic.

Arrange-Act-Assert throughout, with a small deterministic ``FakeLog`` in place
of the project logger, so the tests need no config, no generator checkpoint, no
dataset and no GPU. Every heavy dependency (the frozen mask-diffusion stack, the
three mask VAEs, Ultralytics) is mocked out; what is exercised is the logic this
stage owns and can get wrong:

1. Target labelling - ``labels.txt`` parsing, the empty/background case, and
   malformed lines that must abort instead of silently training on nothing.
2. Target/generator agreement telemetry - the box-IoU that measures how much
   label noise a kept draw injects.
3. Split carving - layout families must not leak across val/test, and the carve
   must be reproducible from its seed.
4. Component resolution - an unknown or incomplete variant row must fail before
   a multi-hour run starts, and the checkpoint path must be anchored.
5. Render signatures - a changed generator, component, alpha or seed must
   invalidate an existing one-shot render.
6. Variant row -> telemetry column mapping, including the rule that an
   unmeasured cell is empty rather than 0.0.
7. Generation bookkeeping - stem/condition misalignment and out-of-index
   thresholds must fail fast instead of writing a half-labelled dataset.

Run with: ``./venv/bin/python -m pytest missing_yolo/test_missing_yolo.py``
(or ``./venv/bin/python missing_yolo/test_missing_yolo.py`` for a plain run).
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Project root so ``missing_yolo``/``helpers`` import when run as a script.
_ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from evaluator.dataset_gt import Box
from missing_yolo.dataset import (
    ComponentVariant,
    _chunked,
    _format_optional,
    carve_family_splits,
    layout_family,
    render_signature,
    require_label_files,
    resolve_component_variant,
)
from missing_yolo.labels import (
    build_label_index,
    component_agreement_iou,
    read_sample_labels,
)
from missing_yolo.runner import generation_row, telemetry_fields
from YOLO.dataset import SampleFolder, split_is_current

TV_LABEL_NAMES: tuple[str, ...] = ("TV",)
DINING_LABEL_NAMES: tuple[str, ...] = (
    "Dining_table_big",
    "Dining_table_medium",
    "Dining_table_round",
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


def make_sources(stems: list[str], root: Path | None = None) -> list[SampleFolder]:
    """Build source folders for tests that only care about stems.

    Args:
        stems (list[str]): Sample stems to create.
        root (Path | None): Sample parent directory; unused paths are fine.

    Returns:
        list[SampleFolder]: One folder per stem, in the given order.
    """
    base: Path = root if root is not None else Path("/nonexistent")
    return [SampleFolder(src_dir=base / stem, stem=stem) for stem in stems]


def make_variant(checkpoint: Path) -> ComponentVariant:
    """Build a component variant row for signature tests.

    Args:
        checkpoint (Path): Generator checkpoint path the row points at.

    Returns:
        ComponentVariant: A TV variant with one label name.
    """
    return ComponentVariant(
        component="tv",
        mask_diffusion_checkpoint=checkpoint,
        detector_class_name="TV",
        label_names=TV_LABEL_NAMES,
    )


def write_labels(sample_dir: Path, lines: list[str]) -> Path:
    """Write a sample's ``labels.txt`` file.

    Args:
        sample_dir (Path): Sample folder to create and write into.
        lines (list[str]): Raw label lines to write.

    Returns:
        Path: The written label file.
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    label_path: Path = sample_dir / "labels.txt"
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return label_path


# ---------------------------------------------------------------------------
# labels.py - targets from the dataset's own labels.txt
# ---------------------------------------------------------------------------


def test_read_sample_labels_keeps_only_component_lines() -> None:
    """Only lines naming the component become targets."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        sample_dir: Path = Path(tmp) / "sample"
        write_labels(
            sample_dir,
            ["TV 0.500000 0.500000 0.200000 0.100000", "Chair 0.100000 0.100000 0.050000 0.050000"],
        )
        # Act
        labels = read_sample_labels(sample_dir, "labels.txt", TV_LABEL_NAMES, 0)
        # Assert
        assert labels.positive
        assert labels.lines == ["0 0.500000 0.500000 0.200000 0.100000"]
        assert len(labels.boxes) == 1


def test_read_sample_labels_folds_several_dataset_names() -> None:
    """All three Dining_table sizes are targets of one run."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        sample_dir: Path = Path(tmp) / "sample"
        write_labels(
            sample_dir,
            [
                "Dining_table_big 0.100000 0.100000 0.300000 0.200000",
                "Dining_table_medium 0.600000 0.100000 0.200000 0.150000",
                "Dining_table_round 0.100000 0.600000 0.150000 0.150000",
                "TV 0.500000 0.500000 0.100000 0.100000",
            ],
        )
        # Act
        labels = read_sample_labels(sample_dir, "labels.txt", DINING_LABEL_NAMES, 0)
        # Assert
        assert len(labels.boxes) == 3
        assert all(line.startswith("0 ") for line in labels.lines)


def test_read_sample_labels_background_sample_is_not_an_error() -> None:
    """A sample without the component is a legitimate negative."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        sample_dir: Path = Path(tmp) / "sample"
        write_labels(sample_dir, ["Chair 0.100000 0.100000 0.050000 0.050000"])
        # Act
        labels = read_sample_labels(sample_dir, "labels.txt", TV_LABEL_NAMES, 0)
        # Assert
        assert not labels.positive
        assert labels.lines == []


def test_read_sample_labels_missing_file_raises() -> None:
    """A sample folder without labels.txt must abort the run."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        sample_dir: Path = Path(tmp) / "sample"
        sample_dir.mkdir(parents=True)
        # Act / Assert
        try:
            read_sample_labels(sample_dir, "labels.txt", TV_LABEL_NAMES, 0)
        except FileNotFoundError as exc:
            assert "labels.txt" in str(exc)
        else:
            raise AssertionError("Expected FileNotFoundError for a missing labels.txt.")


def test_read_sample_labels_malformed_component_line_raises() -> None:
    """A component line that is not five numbers must abort the run."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        sample_dir: Path = Path(tmp) / "sample"
        write_labels(sample_dir, ["TV 0.5 0.5"])
        # Act / Assert
        try:
            read_sample_labels(sample_dir, "labels.txt", TV_LABEL_NAMES, 0)
        except ValueError as exc:
            assert "5 numbers" in str(exc)
        else:
            raise AssertionError("Expected ValueError for a truncated label line.")


def test_read_sample_labels_out_of_range_box_raises() -> None:
    """A component box outside 0..1 must abort the run."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        sample_dir: Path = Path(tmp) / "sample"
        write_labels(sample_dir, ["TV 1.400000 0.500000 0.200000 0.100000"])
        # Act / Assert
        try:
            read_sample_labels(sample_dir, "labels.txt", TV_LABEL_NAMES, 0)
        except ValueError as exc:
            assert "outside the normalized range" in str(exc)
        else:
            raise AssertionError("Expected ValueError for an out-of-range box.")


def test_read_sample_labels_empty_label_names_raises() -> None:
    """Empty label names would turn every sample into a negative."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        sample_dir: Path = Path(tmp) / "sample"
        write_labels(sample_dir, ["TV 0.500000 0.500000 0.100000 0.100000"])
        # Act / Assert
        try:
            read_sample_labels(sample_dir, "labels.txt", (), 0)
        except ValueError as exc:
            assert "background" in str(exc)
        else:
            raise AssertionError("Expected ValueError for empty label names.")


def test_build_label_index_counts_positives_and_backgrounds() -> None:
    """The index reports how many samples carry the component and how many boxes."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        root: Path = Path(tmp)
        write_labels(root / "a", ["TV 0.500000 0.500000 0.100000 0.100000"])
        write_labels(root / "b", ["Chair 0.100000 0.100000 0.050000 0.050000"])
        write_labels(root / "c", ["TV 0.200000 0.200000 0.100000 0.100000",
                                  "TV 0.700000 0.700000 0.100000 0.100000"])
        sources = make_sources(["a", "b", "c"], root)
        # Act
        index = build_label_index(sources, "labels.txt", TV_LABEL_NAMES, 0)
        # Assert
        assert index.positive_samples == 2
        assert index.background_samples == 1
        assert index.total_boxes == 3
        assert index.lines_for("b") == []
        assert len(index.lines_for("c")) == 2


def test_build_label_index_duplicate_stems_raise() -> None:
    """Stem-keyed output would collide with duplicate stems."""
    # Arrange
    sources = make_sources(["a", "a"])
    # Act / Assert
    try:
        build_label_index(sources, "labels.txt", TV_LABEL_NAMES, 0)
    except ValueError as exc:
        assert "not unique" in str(exc)
    else:
        raise AssertionError("Expected ValueError for duplicate stems.")


def test_build_label_index_empty_sources_raise() -> None:
    """An empty split cannot be indexed."""
    # Act / Assert
    try:
        build_label_index([], "labels.txt", TV_LABEL_NAMES, 0)
    except ValueError as exc:
        assert "no samples" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an empty source list.")


def test_build_label_index_lines_for_unknown_stem_raise() -> None:
    """Asking for a stem outside the index is a bug, not a background sample."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        root: Path = Path(tmp)
        write_labels(root / "a", ["TV 0.500000 0.500000 0.100000 0.100000"])
        index = build_label_index(make_sources(["a"], root), "labels.txt", TV_LABEL_NAMES, 0)
        # Act / Assert
        try:
            index.lines_for("b")
        except KeyError as exc:
            assert "not in this label index" in str(exc)
        else:
            raise AssertionError("Expected KeyError for an unknown stem.")


# ---------------------------------------------------------------------------
# labels.py - agreement telemetry
# ---------------------------------------------------------------------------


def test_component_agreement_iou_perfect_overlap() -> None:
    """A generated component exactly on the target box scores 1.0."""
    # Arrange
    mask: np.ndarray = np.zeros((256, 256), dtype=bool)
    mask[96:160, 96:160] = True
    boxes: tuple[Box, ...] = (Box(cx=0.5, cy=0.5, w=0.25, h=0.25),)
    # Act
    iou = component_agreement_iou(mask, boxes, 256)
    # Assert
    assert iou == 1.0


def test_component_agreement_iou_disjoint_scores_zero() -> None:
    """A generated component nowhere near the target box scores 0.0."""
    # Arrange
    mask: np.ndarray = np.zeros((256, 256), dtype=bool)
    mask[0:16, 0:16] = True
    boxes: tuple[Box, ...] = (Box(cx=0.9, cy=0.9, w=0.1, h=0.1),)
    # Act
    iou = component_agreement_iou(mask, boxes, 256)
    # Assert
    assert iou == 0.0


def test_component_agreement_iou_takes_best_of_several_boxes() -> None:
    """Several target boxes (e.g. two TVs) are credited by their best match."""
    # Arrange
    mask: np.ndarray = np.zeros((256, 256), dtype=bool)
    mask[200:240, 10:50] = True
    boxes: tuple[Box, ...] = (
        Box(cx=0.1171875, cy=0.859375, w=0.15625, h=0.15625),
        Box(cx=0.5, cy=0.5, w=0.1, h=0.1),
    )
    # Act
    iou = component_agreement_iou(mask, boxes, 256)
    # Assert
    assert iou is not None and iou > 0.9


def test_component_agreement_iou_background_sample_is_unmeasurable() -> None:
    """No target box means nothing to agree with, so the value is None."""
    # Arrange
    mask: np.ndarray = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True
    # Act
    iou = component_agreement_iou(mask, (), 64)
    # Assert
    assert iou is None


def test_component_agreement_iou_empty_draw_scores_zero_not_none() -> None:
    """A positive sample the generator drew nothing for scores 0.0."""
    # Arrange
    mask: np.ndarray = np.zeros((64, 64), dtype=bool)
    boxes: tuple[Box, ...] = (Box(cx=0.5, cy=0.5, w=0.2, h=0.2),)
    # Act
    iou = component_agreement_iou(mask, boxes, 64)
    # Assert
    assert iou == 0.0


def test_component_agreement_iou_wrong_mask_shape_raises() -> None:
    """A mask from another render domain is a bug, not a zero."""
    # Arrange
    mask: np.ndarray = np.zeros((32, 32), dtype=bool)
    boxes: tuple[Box, ...] = (Box(cx=0.5, cy=0.5, w=0.2, h=0.2),)
    # Act / Assert
    try:
        component_agreement_iou(mask, boxes, 256)
    except ValueError as exc:
        assert "expected" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a mismatched mask shape.")


# ---------------------------------------------------------------------------
# dataset.py - split carving and label-file guards
# ---------------------------------------------------------------------------


def test_layout_family_strips_augmentation_suffixes() -> None:
    """Augmented siblings of one layout must map to one family key."""
    # Arrange / Act / Assert
    assert layout_family("abc_augmented_001") == "abc"
    assert layout_family("abc_augmented_012") == "abc"
    assert layout_family("abc_original") == "abc"
    assert layout_family("abc") == "abc"


def test_carve_family_splits_is_family_disjoint() -> None:
    """No family may appear on both sides of the val/test carve."""
    # Arrange
    sources: list[SampleFolder] = make_sources(
        [f"fam{index % 7}_augmented_{index:03d}" for index in range(40)]
    )
    # Act
    val_sources, test_sources = carve_family_splits(sources, 0.5, 1234)
    # Assert
    val_families: set[str] = {layout_family(source.stem) for source in val_sources}
    test_families: set[str] = {layout_family(source.stem) for source in test_sources}
    assert val_families and test_families
    assert not (val_families & test_families)
    assert len(val_sources) + len(test_sources) == len(sources)


def test_carve_family_splits_is_reproducible_from_seed() -> None:
    """The same seed always carves the same halves."""
    # Arrange
    sources: list[SampleFolder] = make_sources(
        [f"fam{index % 5}_augmented_{index:03d}" for index in range(25)]
    )
    # Act
    first = carve_family_splits(sources, 0.5, 99)
    second = carve_family_splits(sources, 0.5, 99)
    # Assert
    assert [source.stem for source in first[0]] == [source.stem for source in second[0]]
    assert [source.stem for source in first[1]] == [source.stem for source in second[1]]


def test_carve_family_splits_different_seed_can_move_a_family() -> None:
    """A different seed must be able to produce a different carve."""
    # Arrange
    sources: list[SampleFolder] = make_sources(
        [f"fam{index % 20}_augmented_{index:03d}" for index in range(40)]
    )
    # Act
    first = carve_family_splits(sources, 0.5, 1)
    second = carve_family_splits(sources, 0.5, 2)
    # Assert
    assert [source.stem for source in first[0]] != [source.stem for source in second[0]]


def test_carve_family_splits_invalid_fraction_raises() -> None:
    """A fraction outside (0, 1) would silently duplicate one half."""
    # Arrange
    sources: list[SampleFolder] = make_sources(["a_original", "b_original"])
    # Act / Assert
    for fraction in (0.0, 1.0, -0.5):
        try:
            carve_family_splits(sources, fraction, 1)
        except ValueError as exc:
            assert "VAL_FAMILY_FRACTION" in str(exc)
        else:
            raise AssertionError(f"Expected ValueError for fraction {fraction}.")


def test_require_label_files_passes_when_every_sample_is_labelled() -> None:
    """A fully labelled split passes the pre-flight check."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        root: Path = Path(tmp)
        write_labels(root / "a", ["TV 0.500000 0.500000 0.100000 0.100000"])
        write_labels(root / "b", ["Chair 0.100000 0.100000 0.050000 0.050000"])
        sources = make_sources(["a", "b"], root)
        # Act / Assert
        require_label_files(sources, "labels.txt")


def test_require_label_files_names_the_offenders() -> None:
    """A sample without labels.txt must abort before any generation happens."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        root: Path = Path(tmp)
        write_labels(root / "a", ["TV 0.500000 0.500000 0.100000 0.100000"])
        (root / "b").mkdir(parents=True)
        sources = make_sources(["a", "b"], root)
        # Act / Assert
        try:
            require_label_files(sources, "labels.txt")
        except FileNotFoundError as exc:
            assert "b" in str(exc)
        else:
            raise AssertionError("Expected FileNotFoundError for an unlabelled sample.")


# ---------------------------------------------------------------------------
# dataset.py - component resolution
# ---------------------------------------------------------------------------


def make_variant_config(checkpoint: Path) -> SimpleNamespace:
    """Build a config namespace holding one component variant row.

    Args:
        checkpoint (Path): Generator checkpoint the row points at.

    Returns:
        SimpleNamespace: Config with ``COMPONENT_VARIANTS``.
    """
    return SimpleNamespace(
        COMPONENT_VARIANTS={
            "tv": {
                "mask_diffusion_checkpoint": str(checkpoint),
                "detector_class_name": "TV",
                "label_names": ["TV"],
            }
        }
    )


def test_resolve_component_variant_anchors_relative_checkpoint() -> None:
    """A relative generator path is anchored at the repository root."""
    # Arrange
    config = make_variant_config(_ROOT_DIR / "mask_diffusion" / "models" / "gen.pt")
    (config.COMPONENT_VARIANTS["tv"]["mask_diffusion_checkpoint"]) = "mask_diffusion/models/gen.pt"
    (Path(config.COMPONENT_VARIANTS["tv"]["mask_diffusion_checkpoint"])).parent.mkdir(
        parents=True, exist_ok=True
    )
    checkpoint = _ROOT_DIR / "mask_diffusion" / "models" / "gen.pt"
    checkpoint.write_bytes(b"stub")
    # Act
    variant = resolve_component_variant(config, "tv")
    # Assert
    assert variant.mask_diffusion_checkpoint.is_absolute()
    assert variant.detector_class_name == "TV"
    assert variant.label_names == ("TV",)
    checkpoint.unlink()


def test_resolve_component_variant_unknown_component_lists_valid_keys() -> None:
    """A typo must fail with the list of components that do exist."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint: Path = Path(tmp) / "gen.pt"
        checkpoint.write_bytes(b"stub")
        config = make_variant_config(checkpoint)
        # Act / Assert
        try:
            resolve_component_variant(config, "dining")
        except KeyError as exc:
            assert "tv" in str(exc)
        else:
            raise AssertionError("Expected KeyError for an unknown component.")


def test_resolve_component_variant_missing_checkpoint_raises() -> None:
    """A run id that does not match the file on disk must fail fast."""
    # Arrange
    config = make_variant_config(Path("/nonexistent/gen.pt"))
    # Act / Assert
    try:
        resolve_component_variant(config, "tv")
    except FileNotFoundError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for a missing checkpoint.")


def test_resolve_component_variant_incomplete_row_raises() -> None:
    """A row missing a key must not fall back to a default."""
    # Arrange
    config = make_variant_config(Path("/nonexistent/gen.pt"))
    del config.COMPONENT_VARIANTS["tv"]["label_names"]
    # Act / Assert
    try:
        resolve_component_variant(config, "tv")
    except ValueError as exc:
        assert "label_names" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an incomplete variant row.")


def test_resolve_component_variant_empty_class_name_raises() -> None:
    """A run trains exactly one named class, so the name cannot be blank."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint: Path = Path(tmp) / "gen.pt"
        checkpoint.write_bytes(b"stub")
        config = make_variant_config(checkpoint)
        config.COMPONENT_VARIANTS["tv"]["detector_class_name"] = "  "
        # Act / Assert
        try:
            resolve_component_variant(config, "tv")
        except ValueError as exc:
            assert "detector_class_name" in str(exc)
        else:
            raise AssertionError("Expected ValueError for an empty class name.")


# ---------------------------------------------------------------------------
# dataset.py - render signature
# ---------------------------------------------------------------------------


def test_render_signature_is_stable_for_identical_inputs() -> None:
    """The same sources, variant, alpha, size and seed must fingerprint equally."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint: Path = Path(tmp) / "gen.pt"
        checkpoint.write_bytes(b"stub")
        variant = make_variant(checkpoint)
        sources = make_sources(["a", "b"])
        # Act
        first = render_signature(sources, variant, 0.6, 256, 7)
        second = render_signature(sources, variant, 0.6, 256, 7)
        # Assert
        assert first == second


def test_render_signature_changes_with_seed() -> None:
    """A different draw seed is different pixels and must re-render."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint: Path = Path(tmp) / "gen.pt"
        checkpoint.write_bytes(b"stub")
        variant = make_variant(checkpoint)
        sources = make_sources(["a", "b"])
        # Act / Assert
        assert render_signature(sources, variant, 0.6, 256, 7) != render_signature(
            sources, variant, 0.6, 256, 8
        )


def test_render_signature_changes_with_generator_file() -> None:
    """Retraining the generator must invalidate an existing render."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint: Path = Path(tmp) / "gen.pt"
        checkpoint.write_bytes(b"stub")
        variant = make_variant(checkpoint)
        sources = make_sources(["a"])
        before = render_signature(sources, variant, 0.6, 256, 7)
        checkpoint.write_bytes(b"stub-with-more-bytes")
        # Act
        after = render_signature(sources, variant, 0.6, 256, 7)
        # Assert
        assert before != after


def test_render_signature_changes_with_component_and_alpha() -> None:
    """A different component or overlay weight changes every pixel."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint: Path = Path(tmp) / "gen.pt"
        checkpoint.write_bytes(b"stub")
        variant = make_variant(checkpoint)
        sources = make_sources(["a"])
        base = render_signature(sources, variant, 0.6, 256, 7)
        other_component = ComponentVariant(
            component="showcase",
            mask_diffusion_checkpoint=checkpoint,
            detector_class_name="Showcase",
            label_names=("Showcase",),
        )
        # Act / Assert
        assert base != render_signature(sources, other_component, 0.6, 256, 7)
        assert base != render_signature(sources, variant, 0.5, 256, 7)
        assert base != render_signature(sources, variant, 0.6, 320, 7)
        assert base != render_signature(make_sources(["a", "b"]), variant, 0.6, 256, 7)


def test_split_is_current_rejects_mismatched_signature() -> None:
    """A stale sidecar must force a re-render even when the stems match."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        image_dir: Path = Path(tmp) / "images"
        label_dir: Path = Path(tmp) / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for stem in ("a", "b"):
            (image_dir / f"{stem}.png").write_bytes(b"stub")
        signature_path: Path = image_dir / ".render_signature"
        signature_path.write_text("old-digest\n", encoding="utf-8")
        sources = make_sources(["a", "b"])
        # Act / Assert
        assert not split_is_current(image_dir, sources, signature_path, "new-digest")
        signature_path.write_text("new-digest\n", encoding="utf-8")
        assert split_is_current(image_dir, sources, signature_path, "new-digest")


# ---------------------------------------------------------------------------
# dataset.py - chunking and CSV formatting helpers
# ---------------------------------------------------------------------------


def test_chunked_preserves_order_and_count() -> None:
    """Chunking must never drop or reorder a sample."""
    # Arrange
    sources: list[SampleFolder] = make_sources([f"s{index}" for index in range(7)])
    # Act
    chunks = _chunked(sources, 3)
    # Assert
    assert [source.stem for chunk in chunks for source in chunk] == [
        source.stem for source in sources
    ]
    assert all(len(chunk) <= 3 for chunk in chunks)


def test_chunked_rejects_zero_size() -> None:
    """A zero chunk size would loop forever."""
    # Act / Assert
    try:
        _chunked(make_sources(["a"]), 0)
    except ValueError as exc:
        assert ">= 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a zero chunk size.")


def test_format_optional_uses_empty_cell_for_unmeasured() -> None:
    """An unmeasured value must not read as a measured zero in the CSV."""
    # Assert
    assert _format_optional(None) == ""
    assert _format_optional(0.0) == "0.000000"


# ---------------------------------------------------------------------------
# runner.py - telemetry rows
# ---------------------------------------------------------------------------


def make_hyperparams(thresholds: list[float] | None = None) -> SimpleNamespace:
    """Build the minimal hyperparameters the telemetry helpers read.

    Args:
        thresholds (list[float] | None): Agreement thresholds to declare.

    Returns:
        SimpleNamespace: Hyperparameters namespace.
    """
    return SimpleNamespace(
        AGREEMENT_BOX_IOU_THRESHOLDS=(
            thresholds if thresholds is not None else [0.1, 0.5]
        )
    )


def make_generation_stats(
    agreement_mean: float | None, fractions: dict[str, float]
) -> SimpleNamespace:
    """Build a :class:`GenerationStats`-shaped record for row-mapping tests.

    Args:
        agreement_mean (float | None): Mean agreement IoU, or None when the
            pass could not score any sample.
        fractions (dict[str, float]): Per-threshold agreement fractions.

    Returns:
        SimpleNamespace: Stats namespace with the fields the row mapper reads.
    """
    return SimpleNamespace(
        split="train",
        epoch=3,
        seed=4321,
        images=10,
        boxes=6,
        seconds=12.5,
        agreement_scored=6,
        agreement_mean=agreement_mean,
        agreement_fractions=fractions,
    )


def test_telemetry_fields_include_generation_and_agreement_columns() -> None:
    """Every reported metric needs a column of its own."""
    # Arrange
    hyperparams = make_hyperparams([0.1, 0.5, 0.9])
    # Act
    fields = telemetry_fields(hyperparams)
    # Assert
    assert fields[:3] == ["row_kind", "epoch", "train_box"]
    assert "agreement_mean" in fields
    assert "agreement_frac_0.1" in fields
    assert "agreement_frac_0.9" in fields
    assert len(fields) == len(set(fields))


def test_generation_row_marks_kind_and_leaves_detection_cells_empty() -> None:
    """A generation row must not claim detection metrics it does not have."""
    # Arrange
    stats = make_generation_stats(0.5, {"0.1": 0.9, "0.5": 0.4})
    # Act
    row = generation_row(stats, 4)
    # Assert
    assert row["row_kind"] == "generation"
    assert row["epoch"] == 4
    assert row["draw_index"] == 3
    assert row["draw_seed"] == 4321
    assert row["train_box"] == ""
    assert row["agreement_frac_0.1"] == "0.900000"


def test_generation_row_leaves_unmeasured_agreement_empty() -> None:
    """No scored sample means an empty cell, never 0.0."""
    # Arrange
    stats = make_generation_stats(None, {"0.1": float("nan")})
    # Act
    row = generation_row(stats, 1)
    # Assert
    assert row["agreement_mean"] == ""
    assert row["agreement_frac_0.1"] == ""


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