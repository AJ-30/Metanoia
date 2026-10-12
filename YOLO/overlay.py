"""End-of-epoch and offline visualization: GT vs predicted boxes.

Draws ground-truth boxes (green) and predicted boxes (red, with class name and
confidence) directly on the floorplan+furniture overlay images the model sees,
so every visual check is done on the real input rather than a schematic.

Two entry points:

* :func:`save_epoch_overlays` - called by the trainer at the end of EVERY
  epoch on a fixed, seeded subset of the val split (the same stems each
  epoch, so changes in pixels and improving boxes stay comparable). Uses the
  epoch's own weights (the EMA model Ultralytics validated with, when
  available) and writes one PNG per stem plus a montage into
  ``YOLO/outputs/epoch_XX/``.
* :func:`build_contact_sheet` - used by the offline viewer
  (``YOLO/predict.py``) to tile finished overlays into one contact sheet.

All boxes are drawn with fine 1px lines: overlays are 256px, so wider lines
would merge adjacent furniture boxes into unreadable bars.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

# Green (ground truth) and red (prediction) box colors. Kept here so both the
# epoch overlays and the offline viewer use exactly the same convention.
_GT_COLOR: tuple[int, int, int] = (0, 255, 0)
_PRED_COLOR: tuple[int, int, int] = (255, 0, 0)


def draw_boxes_on_image(
    img: Image.Image,
    gt_lines: list[str],
    pred_boxes: list[tuple[float, float, float, float, int, float]],
    class_names: list[str],
) -> Image.Image:
    """Draw ground-truth (green) and predicted (red) boxes with annotations.

    Args:
        img (Image.Image): RGB image to annotate (copied before drawing).
        gt_lines (list[str]): Ground-truth YOLO label lines (normalized).
        pred_boxes (list[tuple]): Predicted
            ``(x1, y1, x2, y2, cls, conf)`` tuples in absolute pixels.
        class_names (list[str]): Class index -> display name mapping for both
            GT and predictions (train and val share one ontology).

    Returns:
        Image.Image: Annotated copy of the input image.
    """
    img = img.copy()
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(img)
    width_px: int
    height_px: int
    width_px, height_px = img.size
    _draw_gt_boxes(draw, gt_lines, class_names, width_px, height_px)
    _draw_pred_boxes(draw, pred_boxes, class_names)
    return img


def _draw_gt_boxes(
    draw: ImageDraw.ImageDraw,
    gt_lines: list[str],
    class_names: list[str],
    width_px: int,
    height_px: int,
) -> None:
    """Draw normalized ground-truth boxes in green on an image.

    Args:
        draw (ImageDraw.ImageDraw): Drawing context of the target image.
        gt_lines (list[str]): Normalized ``cls cx cy w h`` label lines.
        class_names (list[str]): Class index -> display name mapping.
        width_px (int): Image width in pixels.
        height_px (int): Image height in pixels.
    """
    for raw in gt_lines:
        if not raw.strip():
            continue
        parts: list[str] = raw.split()
        cls_id: int = int(parts[0])
        cx, cy, box_w, box_h = (float(v) for v in parts[1:5])
        x1: float = (cx - box_w / 2.0) * width_px
        y1: float = (cy - box_h / 2.0) * height_px
        x2: float = (cx + box_w / 2.0) * width_px
        y2: float = (cy + box_h / 2.0) * height_px
        draw.rectangle([x1, y1, x2, y2], outline=_GT_COLOR, width=1)
        name: str = (
            class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
        )
        draw.text((x1 + 2, y1 + 2), f"GT:{name}", fill=_GT_COLOR)


def _draw_pred_boxes(
    draw: ImageDraw.ImageDraw,
    pred_boxes: list[tuple[float, float, float, float, int, float]],
    class_names: list[str],
) -> None:
    """Draw pixel-space predicted boxes in red on an image.

    Args:
        draw (ImageDraw.ImageDraw): Drawing context of the target image.
        pred_boxes (list[tuple]): ``(x1, y1, x2, y2, cls, conf)`` in pixels.
        class_names (list[str]): Class index -> display name mapping.
    """
    for x1, y1, x2, y2, cls_id, conf in pred_boxes:
        draw.rectangle([x1, y1, x2, y2], outline=_PRED_COLOR, width=1)
        name: str = (
            class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
        )
        draw.text((x1 + 2, y2 + 2), f"{name} {conf:.2f}", fill=_PRED_COLOR)


def choose_fixed_stems(
    sources: list[str], count: int, seed: int
) -> list[str]:
    """Choose the fixed overlay subset of the val split, once per run.

    The same stems are published every epoch so the panel stays comparable:
    pixel changes come from the redraw and box quality changes come from
    training, never from a changing sample set.

    Args:
        sources (list[str]): All candidate stems, in sorted order.
        count (int): Subset size; negative or >= len keeps every stem.
        seed (int): Run seed (``BASE_SEED``); the subset is reproducible.

    Returns:
        list[str]: The chosen stems, in sorted order.
    """
    if not sources:
        return []
    if count < 0 or count >= len(sources):
        return list(sources)
    rng: random.Random = random.Random(seed)
    return sorted(rng.sample(sources, count))


def predict_boxes(
    model: object,
    image_path: Path,
    conf: float,
    max_boxes: int,
    imgsz: int,
) -> tuple[list[tuple[float, float, float, float, int, float]], float]:
    """Run inference on one image and cap the returned boxes.

    Boxes are returned in absolute pixels of the ORIGINAL image (Ultralytics
    undoes its letterboxing), highest-confidence first, capped at
    ``max_boxes`` so an undertrained model cannot bury the visualization.

    Args:
        model (object): Ultralytics YOLO model (``predict`` API).
        image_path (Path): Image to run inference on.
        conf (float): Confidence threshold for returned boxes.
        max_boxes (int): Hard cap on returned boxes (highest-confidence first).
        imgsz (int): Inference image size (``IMG_SIZE``).

    Returns:
        tuple[list[tuple[float, float, float, float, int, float]], float]:
            The capped ``(x1, y1, x2, y2, cls, conf)`` boxes and the highest
            confidence the model emitted at all (before any cap), so "no
            boxes drawn" is always distinguishable from "model is broken".
    """
    results = model.predict(  # type: ignore[attr-defined]
        source=str(image_path), conf=conf, imgsz=imgsz, verbose=False
    )
    boxes: list[tuple[float, float, float, float, int, float]] = []
    peak_conf: float = 0.0
    if len(results) > 0 and results[0].boxes is not None:
        xyxy = results[0].boxes.xyxy.tolist()
        cls_list = results[0].boxes.cls.tolist()
        conf_list = results[0].boxes.conf.tolist()
        for (x1, y1, x2, y2), cls_id, box_conf in zip(xyxy, cls_list, conf_list):
            peak_conf = max(peak_conf, float(box_conf))
            boxes.append((x1, y1, x2, y2, int(cls_id), float(box_conf)))
    boxes.sort(key=lambda box: box[5], reverse=True)
    return boxes[:max_boxes], peak_conf


def save_epoch_overlays(
    model: object,
    stems: list[str],
    val_img_dir: Path,
    val_lbl_dir: Path,
    out_dir: Path,
    epoch: int,
    class_names: list[str],
    conf: float,
    max_boxes: int,
    montage_cols: int,
    imgsz: int,
) -> dict[str, float]:
    """Save GT-vs-prediction overlays for one epoch on the fixed val subset.

    Writes one annotated PNG per stem plus a montage grid, all named with the
    1-based epoch so they never collide across epochs.

    Args:
        model (object): Ultralytics YOLO model (usually the trainer's EMA
            model, i.e. the weights validation scored).
        stems (list[str]): Fixed val stems to visualize.
        val_img_dir (Path): Rendered val images directory.
        val_lbl_dir (Path): Rendered val labels directory.
        out_dir (Path): Parent outputs directory; overlays land in
            ``out_dir/epoch_XX/``.
        epoch (int): 1-based epoch number.
        class_names (list[str]): Class index -> display name mapping.
        conf (float): Confidence threshold for predictions.
        max_boxes (int): Hard cap on boxes drawn per image.
        montage_cols (int): Columns of the montage grid.
        imgsz (int): Inference image size (``IMG_SIZE``).

    Returns:
        dict[str, float]: Summary counters (``images``, ``gt_boxes``,
            ``pred_boxes``, ``peak_conf``) for logging.
    """
    epoch_dir: Path = out_dir / f"epoch_{epoch:02d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    gt_boxes: int = 0
    pred_boxes: int = 0
    peak_conf: float = 0.0
    rendered_paths: list[Path] = []
    for stem in stems:
        img_path: Path = val_img_dir / f"{stem}.png"
        if not img_path.is_file():
            continue
        lbl_path: Path = val_lbl_dir / f"{stem}.txt"
        gt_lines: list[str] = (
            lbl_path.read_text(encoding="utf-8").splitlines()
            if lbl_path.is_file()
            else []
        )
        gt_boxes += sum(1 for line in gt_lines if line.strip())
        img: Image.Image = Image.open(img_path).convert("RGB")
        boxes, peak = predict_boxes(model, img_path, conf, max_boxes, imgsz)
        peak_conf = max(peak_conf, peak)
        pred_boxes += len(boxes)
        annotated: Image.Image = draw_boxes_on_image(img, gt_lines, boxes, class_names)
        out_path: Path = epoch_dir / f"{stem}.png"
        annotated.save(str(out_path))
        rendered_paths.append(out_path)
    if rendered_paths:
        make_montage(
            rendered_paths, out_dir / f"montage_epoch_{epoch:02d}.png", montage_cols
        )
    return {
        "images": float(len(rendered_paths)),
        "gt_boxes": float(gt_boxes),
        "pred_boxes": float(pred_boxes),
        "peak_conf": peak_conf,
    }


def make_montage(paths: list[Path], out_path: Path, cols: int) -> None:
    """Tile images into one contact-sheet PNG.

    Args:
        paths (list[Path]): Images to tile, row-major order.
        out_path (Path): Destination PNG (``cols`` columns, rows as needed).
        cols (int): Number of columns; must be >= 1.
    """
    if not paths:
        return
    if cols < 1:
        raise ValueError(f"Montage columns must be >= 1, got {cols}.")
    images: list[Image.Image] = [Image.open(path).convert("RGB") for path in paths]
    cell_width, cell_height = images[0].size
    rows: int = (len(images) + cols - 1) // cols
    sheet: Image.Image = Image.new(
        "RGB", (cell_width * cols, cell_height * rows), (16, 16, 16)
    )
    for index, image in enumerate(images):
        row: int = index // cols
        col: int = index % cols
        sheet.paste(image, (col * cell_width, row * cell_height))
    sheet.save(str(out_path))
