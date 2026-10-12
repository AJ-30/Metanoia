"""Offline inference viewer for a trained YOLO floor-plan detector.

Runs the saved best weights over the deterministic val/test overlays of the
live dataset (or over one explicit image) and writes GT-vs-prediction
overlays plus a single contact sheet to ``YOLO/outputs/predict/``. Green boxes
are ground truth, red boxes are predictions with ``class + confidence``
(capped at ``PREDICT_MAX_BOXES``, highest-confidence first).

It is read-only with respect to training state: it never rewrites the dataset,
never touches telemetry, and never loads the frozen diffusion stack. It reuses
the trainer's own drawing helpers, so the images match what training showed.

Usage:

    python YOLO/predict.py --mode debug                     # newest best.pt, val
    python YOLO/predict.py --mode debug --split test
    python YOLO/predict.py --mode debug --limit 20
    python YOLO/predict.py --mode debug --image /path/to/plan.png
    python YOLO/predict.py --mode debug --weights path/best.pt
    python YOLO/predict.py --mode debug --conf 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

# Project root so ``YOLO.*`` and ``helpers`` import from any cwd.
_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_ROOT_DIR: Path = _SCRIPT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from helpers.config import load_config
from helpers.hyperparams import load_hyperparams
from helpers.logger import setup_logger
from YOLO.overlay import draw_boxes_on_image, make_montage, predict_boxes
from YOLO.runner import anchor_config_paths

# Glob picking up the persisted best weights of previous runs.
_BEST_WEIGHTS_GLOB: str = "yolo_*_best.pt"


def labels_dir_for(image_dir: Path) -> Path:
    """Return the labels directory sibling of an images directory.

    Args:
        image_dir (Path): ``.../images/<split>`` directory.

    Returns:
        Path: The matching ``.../labels/<split>`` directory.
    """
    return image_dir.parent.parent / "labels" / image_dir.name


def read_gt_lines(lbl_path: Path) -> list[str]:
    """Read one label file into non-empty YOLO lines.

    Args:
        lbl_path (Path): Label ``.txt`` path (may not exist).

    Returns:
        list[str]: Non-empty label lines, empty when the file is missing.
    """
    if not lbl_path.is_file():
        return []
    return [line for line in lbl_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_weights_path(models_dir: Path, explicit: str | None) -> Path:
    """Pick the weights file: explicit flag, else the newest persisted best.

    Args:
        models_dir (Path): ``MODEL_SAVE_PATH`` holding ``yolo_*_best.pt``.
        explicit (str | None): Explicit weights path from ``--weights``.

    Returns:
        Path: Weights file to load.

    Raises:
        FileNotFoundError: If the explicit path is missing, or no persisted
            best weights exist yet.
    """
    if explicit:
        weights_path: Path = Path(explicit)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        return weights_path
    candidates: list[Path] = sorted(
        models_dir.glob(_BEST_WEIGHTS_GLOB), key=lambda path: path.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(
            f"No {(_BEST_WEIGHTS_GLOB)!r} weights in {models_dir}; train a run "
            "first (./train_YOLO.sh) or pass --weights."
        )
    return candidates[-1]


def build_contact_sheet(
    overlay_paths: list[Path], out_path: Path, cols: int
) -> None:
    """Tile finished overlays into one contact-sheet PNG.

    Args:
        overlay_paths (list[Path]): Overlays to tile, row-major order.
        out_path (Path): Destination PNG path.
        cols (int): Number of columns; must be >= 1.
    """
    make_montage(overlay_paths, out_path, cols)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the viewer CLI arguments.

    Args:
        argv (list[str] | None): Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="View GT vs predicted boxes of a trained YOLO run."
    )
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--conf", type=float, default=None)
    return parser.parse_args(argv)


def _collect_image_targets(
    config: SimpleNamespace,
    hyperparams: SimpleNamespace,
    args: argparse.Namespace,
) -> list[Path]:
    """Resolve which images to run over.

    Args:
        config (SimpleNamespace): Runtime config with dataset paths.
        hyperparams (SimpleNamespace): Training/viewer knobs.
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        list[Path]: Images to visualize (explicit image, or the split's PNGs
            capped by ``--limit`` / ``PREDICT_LIMIT``).
    """
    if args.image:
        image_path: Path = Path(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return [image_path]
    image_dir: Path = Path(config.LIVE_DATASET_ROOT) / "images" / args.split
    images: list[Path] = sorted(image_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No rendered images in {image_dir}; train first.")
    limit: int = (
        args.limit if args.limit is not None else int(hyperparams.PREDICT_LIMIT)
    )
    return images if limit <= 0 else images[:limit]


def run_viewer(argv: list[str] | None = None) -> int:
    """Run the offline viewer end to end.

    Args:
        argv (list[str] | None): CLI arguments (defaults to ``sys.argv[1:]``).

    Returns:
        int: Number of overlay images written.
    """
    args: argparse.Namespace = _parse_args(argv)
    config: SimpleNamespace = load_config(args.mode, str(_SCRIPT_DIR / "config.toml"))
    # Resolve relative config paths against the repo root, so the viewer finds
    # the dataset/weights/outputs no matter which directory it runs from.
    anchor_config_paths(config)
    log: SimpleNamespace = setup_logger(config)
    hyperparams: SimpleNamespace = load_hyperparams(
        str(_SCRIPT_DIR / "hyperparameters.toml")
    )
    class_names: list[str] = list(hyperparams.CLASS_NAMES)
    conf: float = (
        args.conf if args.conf is not None else float(hyperparams.PREDICT_CONF)
    )
    max_boxes: int = int(hyperparams.PREDICT_MAX_BOXES)
    imgsz: int = int(hyperparams.IMG_SIZE)

    from ultralytics import YOLO  # lazy: keep config-only runs dependency-free

    weights_path: Path = resolve_weights_path(
        Path(config.MODEL_SAVE_PATH), args.weights
    )
    log.info(f"Loading weights: {weights_path}")
    model = YOLO(str(weights_path))

    out_dir: Path = Path(config.OUTPUT_SAMPLES_DIRECTORY) / "predict"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets: list[Path] = _collect_image_targets(config, hyperparams, args)
    written: list[Path] = []
    peak_conf: float = 0.0
    for image_path in targets:
        lbl_path: Path = (
            labels_dir_for(image_path.parent) / f"{image_path.stem}.txt"
        )
        gt_lines: list[str] = read_gt_lines(lbl_path)
        boxes, peak = predict_boxes(model, image_path, conf, max_boxes, imgsz)
        peak_conf = max(peak_conf, peak)
        img: Image.Image = Image.open(image_path).convert("RGB")
        annotated = draw_boxes_on_image(img, gt_lines, boxes, class_names)
        out_path: Path = out_dir / f"{image_path.stem}.png"
        annotated.save(str(out_path))
        written.append(out_path)
    if written:
        build_contact_sheet(written, out_dir / "contact_sheet.png", 4)
    log.info(
        f"Wrote {len(written)} overlays to {out_dir} "
        f"(peak confidence seen: {peak_conf:.4f}, conf threshold {conf})."
    )
    return len(written)


if __name__ == "__main__":
    run_viewer()
