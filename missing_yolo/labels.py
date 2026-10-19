"""Detection targets read from the dataset's own ``labels.txt``.

This module is the only place in ``missing_yolo`` that decides what a training
target is, and it answers one question: *where does the dataset say the missing
component is?* The answer comes from the ``labels.txt`` file the dataset ships
in every sample folder - the same source the evaluator scores against - and
never from the generated pixels, so a target can never be defined by the very
model whose output it would then be used to grade.

Three objects make up the index:

* :class:`SampleLabels` - one sample's target lines plus the boxes they came
  from, so agreement against the generated mask can be measured later without
  re-parsing the file.
* :class:`LabelIndex` - those per-sample targets for a whole split, with the
  positive/background counts the runner reports.
* :func:`component_agreement_iou` - the box-IoU between the component the
  generator actually drew and the box ``labels.txt`` claims. It is telemetry
  only: samples are never dropped or relabelled on its account, but its
  magnitude is what tells a developer how much label noise a run carries.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Project root so ``evaluator.*`` and ``YOLO.*`` import from any cwd
# (``python missing_yolo/run.py`` puts missing_yolo/ on sys.path).
_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_ROOT_DIR: Path = _SCRIPT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from evaluator.dataset_gt import Box, read_component_boxes


@dataclass(frozen=True)
class SampleLabels:
    """Detection targets of one sample, as the dataset itself states them.

    Attributes:
        lines (list[str]): Ultralytics label lines ``"<cls> <cx> <cy> <w> <h>"``
            for every box of the component, all carrying the same class id.
            Empty for a background sample (one holding no box of the component).
        boxes (tuple[Box, ...]): The normalized boxes ``lines`` were formatted
            from, kept so agreement against a generated mask can be measured
            without re-reading the file.
    """

    lines: list[str]
    boxes: tuple[Box, ...]

    @property
    def positive(self) -> bool:
        """True when the dataset places the component in this sample.

        Returns:
            bool: True when at least one box of the component exists.
        """
        return bool(self.boxes)


@dataclass(frozen=True)
class LabelIndex:
    """``labels.txt`` targets for every sample of one split.

    Attributes:
        by_stem (dict[str, SampleLabels]): Targets keyed by sample stem (the
            sample folder name), which is also the generated image's file stem.
        class_id (int): Detector class id every box of this run carries. A run
            trains a single class, so this is always ``0``; it stays explicit so
            the written label lines never depend on that coincidence.
    """

    by_stem: dict[str, SampleLabels]
    class_id: int

    @property
    def positive_samples(self) -> int:
        """Number of samples the dataset places the component in.

        Returns:
            int: Count of stems whose labels hold at least one box.
        """
        return sum(1 for labels in self.by_stem.values() if labels.positive)

    @property
    def background_samples(self) -> int:
        """Number of kept samples with no box of the component.

        Returns:
            int: Count of stems whose labels hold no box. These are kept on
                purpose: they are the negatives that keep a single-class
                detector from firing everywhere.
        """
        return len(self.by_stem) - self.positive_samples

    @property
    def total_boxes(self) -> int:
        """Total number of target boxes over the split.

        Returns:
            int: Sum of boxes across every sample.
        """
        return sum(len(labels.boxes) for labels in self.by_stem.values())

    def lines_for(self, stem: str) -> list[str]:
        """Return the label lines of one stem.

        Args:
            stem (str): Sample stem, i.e. the sample folder name.

        Returns:
            list[str]: Label lines for the stem; an empty list for a background
                sample or an unknown stem.

        Raises:
            KeyError: If the stem is absent from the index, which means the
                index and the source list disagree - a bug, not a background
                sample (those are present with an empty line list).
        """
        if stem not in self.by_stem:
            raise KeyError(
                f"Sample '{stem}' is not in this label index "
                f"({len(self.by_stem)} stems indexed). The index and the "
                "source split must be built from the same sample list."
            )
        return list(self.by_stem[stem].lines)


def read_sample_labels(
    sample_dir: Path,
    label_filename: str,
    label_names: tuple[str, ...],
    class_id: int,
) -> SampleLabels:
    """Read one sample's component boxes and format them as label lines.

    Args:
        sample_dir (Path): Sample folder holding the dataset's label file.
        label_filename (str): Label filename inside the folder (``labels.txt``).
        label_names (tuple[str, ...]): Names ``labels.txt`` uses for this run's
            component, e.g. ``("Dining_table_big", "Dining_table_medium",
            "Dining_table_round")``. Lines naming anything else are ignored.
        class_id (int): Detector class id for every formatted line.

    Returns:
        SampleLabels: The sample's lines and their source boxes. A sample with
            no box of the component yields empty ones, which is a legitimate
            negative rather than an error.

    Raises:
        FileNotFoundError: If the sample folder has no label file.
        ValueError: If a line naming the component is malformed, or if
            ``label_names`` is empty (which would silently turn every sample
            into a negative).
    """
    if not label_names:
        raise ValueError(
            "label_names is empty: no labels.txt line could ever match, so "
            "every sample would be trained as a background image."
        )
    boxes: list[Box] = read_component_boxes(sample_dir, label_filename, label_names)
    lines: list[str] = [box.yolo_line(class_id) for box in boxes]
    return SampleLabels(lines=lines, boxes=tuple(boxes))


def build_label_index(
    sources: list[object],
    label_filename: str,
    label_names: tuple[str, ...],
    class_id: int,
    pool: ThreadPoolExecutor | None = None,
) -> LabelIndex:
    """Index the ``labels.txt`` targets of a whole split, in parallel.

    The file read is tiny but there are ~70k of them, so the reads go through a
    thread pool (``labels.txt`` reads release the GIL) and the result is keyed
    by stem, which is the same key the generated image is written under.

    Args:
        sources (list[object]): Sample folders of the split, in the order their
            images are written; only ``src_dir``/``stem`` are read (duck-typed
            so this stays independent of the dataset module).
        label_filename (str): Label filename inside each sample folder.
        label_names (tuple[str, ...]): Names ``labels.txt`` uses for this run's
            component.
        class_id (int): Detector class id for every formatted line.
        pool (ThreadPoolExecutor | None): Worker pool doing the reads; a private
            pool is created when None.

    Returns:
        LabelIndex: Targets per stem, plus the class id they were written with.

    Raises:
        ValueError: If ``sources`` is empty or two samples share a stem, either
            of which would make stem-keyed writing ambiguous.
    """
    if not sources:
        raise ValueError("Cannot build a label index with no samples.")
    stems: list[str] = [str(getattr(source, "stem")) for source in sources]
    if len(set(stems)) != len(stems):
        raise ValueError(
            f"Sample stems are not unique ({len(stems)} samples, "
            f"{len(set(stems))} distinct stems); stem-keyed output would collide."
        )
    executor: ThreadPoolExecutor = (
        pool if pool is not None else ThreadPoolExecutor(max_workers=8)
    )
    owned: bool = pool is None
    try:
        targets: list[SampleLabels] = list(
            executor.map(
                lambda source: read_sample_labels(
                    Path(str(getattr(source, "src_dir"))),
                    label_filename,
                    label_names,
                    class_id,
                ),
                sources,
            )
        )
    finally:
        if owned:
            executor.shutdown(wait=True)
    return LabelIndex(
        by_stem={stem: labels for stem, labels in zip(stems, targets, strict=True)},
        class_id=class_id,
    )


def component_agreement_iou(
    component_mask: np.ndarray,
    boxes: tuple[Box, ...],
    image_size: int,
) -> float | None:
    """Box-IoU between the generated component and one sample's target boxes.

    Telemetry only - no sample is ever dropped or relabelled on the strength of
    this number. It answers "did the generator put the component where the
    dataset says it is?", which is the honest measure of how much label noise a
    run injects by keeping every draw.

    The intersection is measured against the component's actual pixels (not its
    bounding rectangle), so a generated component scattered over two regions is
    not credited with filling a box it misses.

    Args:
        component_mask (np.ndarray): Bool ``(S, S)`` array, true where the
            decoded arrangement put the missing component.
        boxes (tuple[Box, ...]): The sample's ``labels.txt`` boxes, normalized.
            Empty for a background sample, which has nothing to agree with.
        image_size (int): Side length of the square render domain the boxes are
            normalized against.

    Returns:
        float | None: The highest IoU over the sample's boxes, in ``0..1``;
            None when the sample has no box or the generator drew nothing.

    Raises:
        ValueError: If the mask shape does not match the render domain.
    """
    if component_mask.shape != (image_size, image_size):
        raise ValueError(
            f"Component mask is {component_mask.shape}, expected "
            f"({image_size}, {image_size})."
        )
    if not boxes:
        return None
    mask_area: int = int(np.count_nonzero(component_mask))
    if mask_area == 0:
        return 0.0
    best: float = 0.0
    for box in boxes:
        x1, y1, x2, y2 = box.to_pixels(image_size)
        x_lo: int = max(0, int(np.floor(x1)))
        y_lo: int = max(0, int(np.floor(y1)))
        x_hi: int = min(image_size, int(np.ceil(x2)))
        y_hi: int = min(image_size, int(np.ceil(y2)))
        box_width: int = max(0, x_hi - x_lo)
        box_height: int = max(0, y_hi - y_lo)
        box_area: int = box_width * box_height
        if box_area == 0:
            continue
        intersection: int = int(
            np.count_nonzero(component_mask[y_lo:y_hi, x_lo:x_hi])
        )
        union: int = box_area + mask_area - intersection
        best = max(best, intersection / union)
    return best