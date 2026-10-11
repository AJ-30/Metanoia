"""Offline inference viewer for a trained live-diffusion YOLO run.

Runs the saved best weights over images from the live dataset and writes
GT-vs-prediction overlays, so a finished run can be inspected without
re-training. This is deliberately read-only with respect to training state: it
never rewrites the dataset, never touches telemetry, and never loads the
frozen diffusion stack (the pixels already exist on disk).

It reuses the same primitives the trainer uses to draw its overlays
(:func:`YOLO.runner.draw_boxes_on_image`) and the same device resolution
(:func:`YOLO.runner.resolve_train_device`), so what you see here matches what
``runner.save_prediction_overlays`` produced during training.

Usage:
    python YOLO/predict.py --mode debug
    python YOLO/predict.py --mode prod --weights YOLO/models/yolo_<run>_best.pt
    python YOLO/predict.py --mode debug --split test --limit 20 --conf 0.001
    python YOLO/predict.py --mode debug --image /path/to/floorplan.png

Notes:
  * ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is required on MPS: Ultralytics runs
    ``torchvision::nms`` during prediction, which has no MPS kernel. The
    wrapper script ``./run_scripts/predict_YOLO.sh`` exports it for you.
  * An undertrained model legitimately produces zero detections at a high
    confidence. The tool reports the peak confidence it saw so "no detections"
    is distinguishable from "model is broken".
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure PyTorch falls back to CPU for ops unsupported on MPS (e.g. torchvision::nms)
# so standalone inference runs on Apple Silicon without manual env exports.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from PIL import Image, ImageDraw, ImageFont

# Project root so ``helpers/`` and ``YOLO.*`` import from any working directory.
SCRIPT_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from helpers.config import load_config
from helpers.hyperparams import load_hyperparams
from helpers.logger import setup_logger
from YOLO.runner import draw_boxes_on_image, resolve_train_device

# Height of the caption strip above each contact-sheet tile.
HEADER_HEIGHT_PX: int = 18

# Caption font size in pixels; also sets the minimum useful strip height.
FONT_SIZE_PX: int = 12


def resolve_weights_path(
    config: SimpleNamespace, hyperparams: SimpleNamespace, override: str | None
) -> Path:
    """Pick the checkpoint to visualize, preferring an explicit ``--weights``.

    Falls back to the newest ``yolo_<run>_best.pt`` in ``MODEL_SAVE_PATH`` so
    ``--mode prod`` alone is enough to inspect the latest run. Works regardless
    of whether launched from repository root or the ``YOLO/`` subdirectory.

    Args:
        config (SimpleNamespace): Runtime config (``MODEL_SAVE_PATH``).
        hyperparams (SimpleNamespace): Unused here; kept for a uniform
            ``config, hyperparams, log`` call signature across the module.
        override (str | None): Explicit weights path from the CLI.

    Returns:
        Path: Resolved checkpoint path.

    Raises:
        FileNotFoundError: If no checkpoint can be found.
    """
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            root_candidate = ROOT_DIR / str(override).lstrip("./")
            if root_candidate.is_file():
                return root_candidate
            raise FileNotFoundError(f"--weights path does not exist: {path}")
        return path
    model_dir = Path(str(config.MODEL_SAVE_PATH))
    if not model_dir.is_dir():
        # Fall back to root-relative path if launched from a subfolder
        candidate_dir = ROOT_DIR / str(config.MODEL_SAVE_PATH).lstrip("./")
        if candidate_dir.is_dir():
            model_dir = candidate_dir
        else:
            local_models = SCRIPT_DIR / "models"
            if local_models.is_dir():
                model_dir = local_models
    candidates = sorted(model_dir.glob("yolo_*_best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No trained weights found in {model_dir}. Run ./train_YOLO.sh first, "
            "or pass --weights <path/to/best.pt>."
        )
    return candidates[-1]


def collect_images(
    live_root: Path, split: str, limit: int, single_image: str | None
) -> list[Path]:
    """Gather the images to run inference on.

    Args:
        live_root (Path): Live dataset root holding ``images/<split>``.
        split (str): Split name ("train", "val" or "test").
        limit (int): Maximum images to return; <= 0 means all.
        single_image (str | None): Explicit image path, overriding the split.

    Returns:
        list[Path]: Sorted image paths.

    Raises:
        FileNotFoundError: If the split directory or the explicit image is
            missing, so a typo fails loudly instead of rendering nothing.
    """
    if single_image:
        path = Path(single_image).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--image path does not exist: {path}")
        return [path]
    image_dir = live_root / "images" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"No images for split '{split}' at {image_dir}. Run ./train_YOLO.sh "
            "first to populate the live dataset."
        )
    images = sorted(image_dir.glob("*.png"))
    if limit > 0:
        images = images[:limit]
    return images


def labels_dir_for(image_path: Path, live_root: Path) -> Path:
    """Return the label directory that mirrors an image's split.

    Args:
        image_path (Path): Image being annotated.
        live_root (Path): Live dataset root.

    Returns:
        Path: ``<live_root>/labels/<split>`` for split images, or an empty
            path for images supplied via ``--image`` (no GT available).
    """
    try:
        relative = image_path.relative_to(live_root / "images")
    except ValueError:
        return Path("")
    return live_root / "labels" / relative.parent.name


def read_gt_lines(label_path: Path) -> list[str]:
    """Read GT YOLO label lines, tolerating a missing/partial file.

    Args:
        label_path (Path): Path of the ``.txt`` label file.

    Returns:
        list[str]: Raw label lines; empty when the file is absent (an image
            supplied with ``--image`` has no ground truth).
    """
    if not label_path.is_file():
        return []
    try:
        return [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return []


def run_inference(
    model: object, images: list[Path], conf: float, device: str
) -> tuple[list[list[tuple[float, float, float, float, int, float]]], float]:
    """Run the model over all images and cap boxes per image.

    Args:
        model (object): Loaded Ultralytics YOLO model.
        images (list[Path]): Images to run on.
        conf (float): Confidence threshold.
        device (str): Ultralytics device string ("mps", "cuda" or "cpu").

    Returns:
        tuple[list[list[tuple[float, float, float, float, int, float]]], float]:
            Per-image boxes ``(x1, y1, x2, y2, cls, conf)`` and the highest
            confidence seen across the whole batch (so a run with no
            detections can still report how close the model came).
    """
    results = model.predict(  # type: ignore[attr-defined]
        source=[str(path) for path in images],
        conf=conf,
        device=device,
        verbose=False,
    )
    per_image: list[list[tuple[float, float, float, float, int, float]]] = []
    peak_confidence: float = 0.0
    for result in results:
        boxes: list[tuple[float, float, float, float, int, float]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            for (x1, y1, x2, y2), class_id, box_conf in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist(),
            ):
                boxes.append(
                    (float(x1), float(y1), float(x2), float(y2), int(class_id), float(box_conf))
                )
                peak_confidence = max(peak_confidence, float(box_conf))
            # Highest confidence first, then cap, so an early model cannot
            # bury the image under hundreds of overlapping boxes.
            boxes.sort(key=lambda box: box[5], reverse=True)
        per_image.append(boxes)
    return per_image, peak_confidence


def annotate_image(
    image_path: Path,
    gt_lines: list[str],
    boxes: list[tuple[float, float, float, float, int, float]],
    class_names: list[str],
    max_boxes: int,
    pred_class_names: list[str] | None = None,
) -> Image.Image:
    """Draw GT (green) and predicted (red) boxes on one image.

    Args:
        image_path (Path): Image to annotate.
        gt_lines (list[str]): Ground-truth label lines for the image.
        boxes (list[tuple[float, float, float, float, int, float]]): Predictions.
        class_names (list[str]): Class id -> display name mapping for GT.
        max_boxes (int): Cap on drawn predictions, highest confidence first.
        pred_class_names (list[str] | None): Class names for predictions.

    Returns:
        Image.Image: Annotated copy of the input image.
    """
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
    return draw_boxes_on_image(
        rgb_image,
        gt_lines,
        boxes[:max_boxes],
        class_names,
        pred_class_names=pred_class_names,
    )


def load_caption_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load the largest bundled TrueType font available for captions.

    PIL's default bitmap font is unreadable at contact-sheet scale, so a
    TrueType face is preferred and the default is kept as a fallback for
    environments where no usable font file ships.

    Returns:
        ImageFont.FreeTypeFont | ImageFont.ImageFont: Font object usable with
            ``ImageDraw.text``.
    """
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, FONT_SIZE_PX)
            except OSError:
                continue
    return ImageFont.load_default()


def build_contact_sheet(
    tiles: list[tuple[str, Image.Image]], columns: int
) -> Image.Image:
    """Compose annotated tiles into one grid, each with a caption strip.

    Args:
        tiles (list[tuple[str, Image.Image]]): Caption and image per tile.
        columns (int): Number of tiles per row.

    Returns:
        Image.Image: Single RGB contact sheet.
    """
    if not tiles:
        raise ValueError("Cannot build a contact sheet with no tiles.")
    tile_width, tile_height = tiles[0][1].size
    rows: int = (len(tiles) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (tile_width * columns, (tile_height + HEADER_HEIGHT_PX) * rows),
        (15, 17, 23),
    )
    font = load_caption_font()
    draw = ImageDraw.Draw(sheet)
    for index, (caption, tile) in enumerate(tiles):
        row, column = divmod(index, columns)
        top = row * (tile_height + HEADER_HEIGHT_PX)
        left = column * tile_width
        sheet.paste(tile, (left, top + HEADER_HEIGHT_PX))
        draw.text((left + 4, top + 3), caption, fill=(230, 233, 240), font=font)
    return sheet


def main() -> None:
    """Parse CLI args and write GT-vs-pred overlays for a trained run."""
    parser = argparse.ArgumentParser(
        description="Visualize predictions from a trained live-diffusion YOLO run."
    )
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    parser.add_argument("--weights", default=None, help="Checkpoint path (default: newest best).")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=None, help="Max images (default: PREDICT_LIMIT).")
    parser.add_argument("--conf", type=float, default=None, help="Confidence (default: PREDICT_CONF).")
    parser.add_argument("--columns", type=int, default=4, help="Contact-sheet columns.")
    parser.add_argument("--image", default=None, help="Single image path (overrides --split).")
    args = parser.parse_args()

    config: SimpleNamespace = load_config(args.mode, str(SCRIPT_DIR / "config.toml"))
    hyperparams: SimpleNamespace = load_hyperparams(str(SCRIPT_DIR / "hyperparameters.toml"))
    log = setup_logger(config)

    # Lazily imported so a missing ultralytics reports one clear message.
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Run `./venv/bin/pip install ultralytics`."
        ) from exc

    weights_path: Path = resolve_weights_path(config, hyperparams, args.weights)
    live_root: Path = Path(str(config.LIVE_DATASET_ROOT))
    limit: int = args.limit if args.limit is not None else int(hyperparams.PREDICT_LIMIT)
    conf: float = args.conf if args.conf is not None else float(hyperparams.PREDICT_CONF)
    max_boxes: int = int(hyperparams.PREDICT_MAX_BOXES)
    class_names: list[str] = list(hyperparams.CLASS_NAMES)
    device_str: str = resolve_train_device(str(config.DEVICE), log)

    images: list[Path] = collect_images(live_root, args.split, limit, args.image)
    if not images:
        raise FileNotFoundError(f"No images found for split '{args.split}' in {live_root}.")

    log.info(
        f"Predicting on {len(images)} image(s) from split '{args.split}' with "
        f"{weights_path.name} (conf={conf}, max_boxes={max_boxes}, device={device_str})."
    )
    model = YOLO(str(weights_path))
    # Extract prediction class names directly from the model checkpoint if present
    # so inspecting checkpoints trained on older or different class vocabularies
    # never throws IndexError.
    pred_class_names: list[str] = class_names
    if hasattr(model, "names") and isinstance(model.names, dict):
        pred_class_names = [model.names[i] for i in sorted(model.names.keys())]

    per_image_boxes, peak_confidence = run_inference(model, images, conf, device_str)

    out_base: Path = Path(str(config.OUTPUT_SAMPLES_DIRECTORY))
    if not out_base.is_dir() and (ROOT_DIR / str(config.OUTPUT_SAMPLES_DIRECTORY).lstrip("./")).is_dir():
        out_base = ROOT_DIR / str(config.OUTPUT_SAMPLES_DIRECTORY).lstrip("./")
    out_dir: Path = out_base / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles: list[tuple[str, Image.Image]] = []
    for image_path, boxes in zip(images, per_image_boxes):
        label_path = labels_dir_for(image_path, live_root) / f"{image_path.stem}.txt"
        gt_lines = read_gt_lines(label_path)
        annotated = annotate_image(
            image_path, gt_lines, boxes, class_names, max_boxes, pred_class_names=pred_class_names
        )
        gt_count: int = len(gt_lines)
        caption: str = f"{image_path.stem[-18:]}  gt={gt_count} pred={len(boxes)}"
        tiles.append((caption, annotated))
        annotated.save(str(out_dir / f"{image_path.stem}.png"))

    sheet = build_contact_sheet(tiles, max(1, args.columns))
    sheet_path: Path = out_dir / f"contact_sheet_{args.split}.png"
    sheet.save(str(sheet_path))

    log.info(f"Wrote {len(tiles)} annotated image(s) to {out_dir}.")
    log.info(f"Contact sheet: {sheet_path}")
    if peak_confidence <= 0.0:
        log.warning(
            "No detections at all. The model may be untrained, or --conf is too "
            "high for its current confidence range; try a lower --conf."
        )
    else:
        log.info(f"Highest detection confidence seen: {peak_confidence:.4f}")
        if peak_confidence < conf:
            log.warning(
                f"Detections were capped by --conf={conf}; the model peaked at "
                f"{peak_confidence:.4f}. Lower --conf to see more."
            )
    print(f"\nSaved to: {out_dir}")
    print(f"Contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()