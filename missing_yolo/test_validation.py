"""Unit tests for the standalone missing_yolo validation metrics.

Arrange-Act-Assert throughout, no checkpoints or Ultralytics. The tests cover
the arithmetic the report depends on:

1. Parsing the rendered YOLO ground-truth files and Ultralytics prediction files,
   including missing files, wrong classes and out-of-range values.
2. Greedy IoU matching and the resulting TP/FP/FN, image-level outcomes and
   threshold filtering.
3. The balanced, reproducible sample selection used by the visual panel.
4. Optional-distribution summaries that must be ``None`` rather than a
   misleading zero when nothing matched.

Run with: ``./venv/bin/python missing_yolo/test_validation.py``.
"""

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Project root so ``missing_yolo``/``evaluator`` import when run as a script.
_ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from evaluator.dataset_gt import Box
from missing_yolo.validation_metrics import (
    STATUS_CORRECT_EMPTY,
    STATUS_DETECTED_ALL,
    STATUS_FALSE_ALARM,
    STATUS_MISSED,
    choose_balanced_stems,
    evaluate_split,
    read_ground_truth_labels,
    read_prediction_labels,
    summarize_values,
)


class _CLOSE:
    """Tiny approximate-equality helper, so the tests need no pytest install."""

    def __init__(self, expected: float, tolerance: float = 1e-6) -> None:
        """Store the expected value and tolerance.

        Args:
            expected (float): Expected number.
            tolerance (float): Absolute tolerance.
        """
        self.expected: float = float(expected)
        self.tolerance: float = float(tolerance)

    def __eq__(self, other: object) -> bool:
        """Compare a number to the expected value within the tolerance.

        Args:
            other (object): Value being compared.

        Returns:
            bool: True when the values are close enough.
        """
        return abs(float(other) - self.expected) <= self.tolerance

    def __repr__(self) -> str:
        """Return a readable representation for assertion failures.

        Returns:
            str: Expected value and tolerance.
        """
        return f"_CLOSE({self.expected}, tol={self.tolerance})"


@contextmanager
def _expect_raises(exception_type: type[BaseException]) -> Iterator[None]:
    """Assert that a block raises a particular exception type.

    Args:
        exception_type (type[BaseException]): Expected exception class.

    Yields:
        None: Control to the ``with`` block.

    Raises:
        AssertionError: If the block does not raise, or raises another type.
    """
    try:
        yield
    except exception_type:
        return
    else:
        raise AssertionError(f"Expected {exception_type.__name__} to be raised.")


def write_text(path: Path, lines: list[str]) -> Path:
    """Write one text file, creating its parent directory.

    Args:
        path (Path): Destination path.
        lines (list[str]): Lines to write.

    Returns:
        Path: The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def test_read_ground_truth_labels_parses_a_box() -> None:
    """A rendered label line is parsed into a normalized box."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        path = write_text(
            Path(tmp) / "sample.txt", ["0 0.500000 0.500000 0.200000 0.100000"]
        )
        # Act
        boxes = read_ground_truth_labels(path)
        # Assert
        assert boxes == [Box(cx=0.5, cy=0.5, w=0.2, h=0.1)]


def test_read_ground_truth_labels_missing_file_raises() -> None:
    """A missing rendered label file must not silently become a background."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "missing.txt"
        # Act / Assert
        with _expect_raises(FileNotFoundError):
            read_ground_truth_labels(path)


def test_read_ground_truth_labels_rejects_wrong_class() -> None:
    """A multi-class label in a single-class validation run is an error."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        path = write_text(
            Path(tmp) / "sample.txt", ["1 0.500000 0.500000 0.200000 0.100000"]
        )
        # Act / Assert
        with _expect_raises(ValueError):
            read_ground_truth_labels(path)


def test_read_ground_truth_labels_rejects_non_integer_class() -> None:
    """A fractional class id must not be truncated into a valid class 0."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        path = write_text(
            Path(tmp) / "sample.txt", ["0.5 0.500000 0.500000 0.200000 0.100000"]
        )
        # Act / Assert
        with _expect_raises(ValueError):
            read_ground_truth_labels(path)


def test_read_ground_truth_labels_rejects_out_of_range_box() -> None:
    """A box outside the normalized domain is a malformed label file."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        path = write_text(
            Path(tmp) / "sample.txt", ["0 1.500000 0.500000 0.200000 0.100000"]
        )
        # Act / Assert
        with _expect_raises(ValueError):
            read_ground_truth_labels(path)


def test_read_prediction_labels_missing_file_is_empty() -> None:
    """Ultralytics writes no file when an image has no detections."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        # Act
        predictions = read_prediction_labels(Path(tmp) / "none.txt")
        # Assert
        assert predictions == []


def test_read_prediction_labels_filters_other_classes() -> None:
    """Only the validated class's predictions are kept."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        path = write_text(
            Path(tmp) / "sample.txt",
            [
                "0 0.500000 0.500000 0.200000 0.100000 0.90000",
                "1 0.100000 0.100000 0.100000 0.100000 0.80000",
            ],
        )
        # Act
        predictions = read_prediction_labels(path)
        # Assert
        assert len(predictions) == 1
        assert predictions[0].confidence == _CLOSE(0.9)
        assert predictions[0].box == Box(cx=0.5, cy=0.5, w=0.2, h=0.1)


def test_read_prediction_labels_rejects_bad_confidence() -> None:
    """A confidence outside 0..1 means the prediction format is not understood."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        path = write_text(
            Path(tmp) / "sample.txt",
            ["0 0.500000 0.500000 0.200000 0.100000 1.20000"],
        )
        # Act / Assert
        with _expect_raises(ValueError):
            read_prediction_labels(path)


def test_evaluate_split_counts_tp_fp_and_image_outcomes() -> None:
    """Matching, thresholding and image-level outcomes agree on a tiny split."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gt_dir = root / "gt"
        pred_dir = root / "pred"
        write_text(gt_dir / "positive.txt", ["0 0.5 0.5 0.2 0.2"])
        write_text(gt_dir / "background.txt", [])
        write_text(
            pred_dir / "positive.txt",
            [
                "0 0.5 0.5 0.2 0.2 0.90",
                "0 0.1 0.1 0.1 0.1 0.80",
            ],
        )
        write_text(
            pred_dir / "background.txt",
            ["0 0.2 0.2 0.1 0.1 0.90"],
        )
        # Act
        summaries, rows = evaluate_split(
            stems=["positive", "background"],
            ground_truth_dir=gt_dir,
            prediction_dir=pred_dir,
            conf_thresholds=[0.85, 0.5],
            iou_threshold=0.5,
            image_size=256,
        )
        # Assert
        assert list(summaries) == ["0.5", "0.85"]
        at_085 = summaries["0.85"]
        assert at_085["box_counts"] == {
            "tp": 1,
            "fp": 1,
            "fn": 0,
            "ground_truth": 1,
            "predictions": 2,
        }
        assert at_085["image_counts"] == {
            "total": 2,
            "positive": 1,
            "detected_positive": 1,
            "missed_positive": 0,
            "background": 1,
            "false_alarm": 1,
            "correct_empty": 0,
        }
        assert at_085["image_metrics"]["accuracy"] == _CLOSE(0.5)
        assert at_085["box_metrics"]["precision"] == _CLOSE(0.5)
        assert at_085["box_metrics"]["recall"] == _CLOSE(1.0)
        assert at_085["localization"]["matched_iou"]["mean"] == _CLOSE(1.0)

        at_05 = summaries["0.5"]
        assert at_05["box_counts"]["fp"] == 2
        assert at_05["box_metrics"]["precision"] == _CLOSE(1.0 / 3.0)

        assert len(rows) == 4
        statuses = {
            (row["stem"], row["conf_threshold"]): row["status"] for row in rows
        }
        assert statuses[("positive", 0.85)] == STATUS_DETECTED_ALL
        assert statuses[("background", 0.85)] == STATUS_FALSE_ALARM


def test_evaluate_split_counts_a_miss_and_a_correct_empty() -> None:
    """No prediction is a miss on a positive image and correct on a background."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gt_dir = root / "gt"
        pred_dir = root / "pred"
        write_text(gt_dir / "positive.txt", ["0 0.5 0.5 0.2 0.2"])
        write_text(gt_dir / "background.txt", [])
        pred_dir.mkdir(parents=True)
        # Act
        summaries, rows = evaluate_split(
            stems=["positive", "background"],
            ground_truth_dir=gt_dir,
            prediction_dir=pred_dir,
            conf_thresholds=[0.25],
            iou_threshold=0.5,
            image_size=256,
        )
        # Assert
        stats = summaries["0.25"]
        assert stats["box_counts"]["tp"] == 0
        assert stats["box_counts"]["fn"] == 1
        assert stats["image_counts"]["missed_positive"] == 1
        assert stats["image_counts"]["correct_empty"] == 1
        assert stats["image_metrics"]["accuracy"] == _CLOSE(0.5)
        assert stats["image_metrics"]["positive_detection_rate"] == _CLOSE(0.0)
        assert stats["image_metrics"]["specificity"] == _CLOSE(1.0)
        statuses = {row["stem"]: row["status"] for row in rows}
        assert statuses["positive"] == STATUS_MISSED
        assert statuses["background"] == STATUS_CORRECT_EMPTY


def test_evaluate_split_low_iou_is_a_false_positive_and_a_miss() -> None:
    """A prediction that does not overlap the GT box matches nothing."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gt_dir = root / "gt"
        pred_dir = root / "pred"
        write_text(gt_dir / "positive.txt", ["0 0.5 0.5 0.2 0.2"])
        write_text(
            pred_dir / "positive.txt",
            ["0 0.1 0.1 0.1 0.1 0.9"],
        )
        # Act
        summaries, _ = evaluate_split(
            stems=["positive"],
            ground_truth_dir=gt_dir,
            prediction_dir=pred_dir,
            conf_thresholds=[0.25],
            iou_threshold=0.5,
            image_size=256,
        )
        # Assert
        stats = summaries["0.25"]
        assert stats["box_counts"]["tp"] == 0
        assert stats["box_counts"]["fp"] == 1
        assert stats["box_counts"]["fn"] == 1
        assert stats["localization"]["matched_iou"] is None


def test_evaluate_split_rejects_empty_stem_list() -> None:
    """An empty split is a bug, not a report with zero metrics."""
    # Arrange
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Act / Assert
        with _expect_raises(ValueError):
            evaluate_split(
                stems=[],
                ground_truth_dir=root,
                prediction_dir=root,
                conf_thresholds=[0.25],
                iou_threshold=0.5,
                image_size=256,
            )


def test_choose_balanced_stems_is_reproducible_and_balanced() -> None:
    """The same seed chooses the same balanced subset."""
    # Arrange
    positives = ["p1", "p2", "p3"]
    negatives = ["n1", "n2", "n3", "n4"]
    # Act
    first = choose_balanced_stems(positives, negatives, 2, 2, seed=7)
    second = choose_balanced_stems(positives, negatives, 2, 2, seed=7)
    # Assert
    assert first == second
    assert len(first) == 4
    assert sum(1 for _, is_positive in first if is_positive) == 2
    assert sum(1 for _, is_positive in first if not is_positive) == 2


def test_choose_balanced_stems_fills_a_short_class() -> None:
    """When one class is short, the panel still reaches the requested size."""
    # Arrange
    positives = ["p1", "p2"]
    negatives = ["n1", "n2", "n3"]
    # Act
    selected = choose_balanced_stems(positives, negatives, 4, 4, seed=1)
    # Assert
    assert len(selected) == 5
    assert {stem for stem, _ in selected} == set(positives + negatives)
    assert sum(1 for _, is_positive in selected if is_positive) == 2


def test_choose_balanced_stems_rejects_overlap() -> None:
    """A stem cannot be both a positive and a negative sample."""
    # Arrange / Act / Assert
    with _expect_raises(ValueError):
        choose_balanced_stems(["same"], ["same"], 1, 1, seed=1)


def test_summarize_values_returns_none_for_empty() -> None:
    """An empty distribution prints n/a instead of a misleading zero."""
    # Arrange / Act
    result = summarize_values([])
    # Assert
    assert result is None


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
