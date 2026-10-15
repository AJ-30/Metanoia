"""Ground-truth boxes for the missing component, read from the dataset itself.

The dataset ships one ``labels.txt`` per sample folder, holding the layout's own
bounding boxes in normalized ``cx cy w h`` form. These are the boxes the
evaluator scores against: they come from the source layout rather than from any
region-growing over a segmentation mask, so a generated component is judged
against what the dataset says the component is, not against a thresholded mask
that shares the model's own failure modes.

The coordinates are normalized against the square ``IMAGE_SIZE`` render domain
(the same affine map the masks are resized with), which is what the rest of the
pipeline renders and what the detector sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    """One normalized bounding box.

    Attributes:
        cx (float): Box centre x, normalized to ``0..1``.
        cy (float): Box centre y, normalized to ``0..1``.
        w (float): Box width, normalized to ``0..1``.
        h (float): Box height, normalized to ``0..1``.
    """

    cx: float
    cy: float
    w: float
    h: float

    def to_pixels(self, image_size: int) -> tuple[float, float, float, float]:
        """Convert to absolute pixel bounds of a square image.

        Args:
            image_size (int): Side length of the square image the box lives in.

        Returns:
            tuple[float, float, float, float]: ``x1, y1, x2, y2`` in pixels,
                unclipped so the loss can see how far an outlier sits outside.
        """
        half_w: float = self.w * image_size / 2.0
        half_h: float = self.h * image_size / 2.0
        return (
            self.cx * image_size - half_w,
            self.cy * image_size - half_h,
            self.cx * image_size + half_w,
            self.cy * image_size + half_h,
        )

    def yolo_line(self, class_id: int) -> str:
        """Format as one YOLO label line for a detector dataset.

        Args:
            class_id (int): Detector class index this box belongs to.

        Returns:
            str: ``"<cls> <cx> <cy> <w>"`` at six decimals.
        """
        return f"{class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def parse_label_line(raw: str, label_names: tuple[str, ...]) -> Box | None:
    """Parse one dataset label line, keeping only the named component.

    Args:
        raw (str): One line of ``labels.txt``:
            ``"<Name> <cx> <cy> <w> <h>"``.
        label_names (tuple[str, ...]): Dataset label names of the missing
            component; lines naming any other furniture are ignored.

    Returns:
        Box | None: The parsed box, or ``None`` for a line that belongs to a
            different component or is blank.

    Raises:
        ValueError: If the line names the component but is not five numbers in
            ``0..1``, which would mean the label format changed under us.
    """
    parts: list[str] = raw.split()
    if not parts:
        return None
    if parts[0] not in label_names:
        return None
    if len(parts) != 5:
        raise ValueError(
            f"Label line {raw!r} names the missing component but does not hold "
            "exactly 5 numbers (name cx cy w h)."
        )
    try:
        cx, cy, width, height = (float(value) for value in parts[1:5])
    except ValueError as exc:
        raise ValueError(f"Label line {raw!r} is not numeric: {exc}") from exc
    for value, label in ((cx, "cx"), (cy, "cy"), (width, "w"), (height, "h")):
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Label line {raw!r} has {label}={value} outside the normalized "
                "range 0..1; the dataset's label format is not understood."
            )
    return Box(cx=cx, cy=cy, w=width, h=height)


def read_component_boxes(
    sample_dir: Path, label_filename: str, label_names: tuple[str, ...]
) -> list[Box]:
    """Read the missing component's ground-truth boxes of one sample.

    Args:
        sample_dir (Path): Sample folder holding the dataset's label file.
        label_filename (str): Label filename inside the folder, e.g.
            ``labels.txt``.
        label_names (tuple[str, ...]): Dataset label names of the component.

    Returns:
        list[Box]: Every box the dataset attributes to the missing component,
            in file order. Empty when the sample contains none, which is a
            legitimate negative case, not an error.

    Raises:
        FileNotFoundError: If the sample folder has no label file at all.
        ValueError: If a line naming the component is malformed.
    """
    label_path: Path = sample_dir / label_filename
    if not label_path.is_file():
        raise FileNotFoundError(f"Sample {sample_dir} has no {label_filename}.")
    boxes: list[Box] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        box: Box | None = parse_label_line(raw, label_names)
        if box is not None:
            boxes.append(box)
    return boxes


def boxes_to_yolo_lines(boxes: list[Box], class_id: int) -> list[str]:
    """Format boxes as YOLO label lines, one per box.

    Args:
        boxes (list[Box]): Boxes to write.
        class_id (int): Detector class index for all of them.

    Returns:
        list[str]: One label line per box; empty for an empty list.
    """
    return [box.yolo_line(class_id) for box in boxes]
