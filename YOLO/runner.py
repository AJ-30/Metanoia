"""YOLO26 finetune runner on the frozen-VAE augmented floor-plan dataset.

Flow:
  1. Fail fast unless a complete augmented dataset already exists on disk
     (build it first with ./run_augmentation.sh; this runner never writes data).
  2. Finetune ``YOLO(<YOLO_MODEL>)`` (default ``yolo26n.pt``) with Ultralytics
     on MPS (fallback CPU), ``workers=0`` for macOS.
  3. Mirror Ultralytics ``results.csv`` rows into the shared
     ``TelemetryWriter`` CSV so the existing Flask dashboard plots loss, mAP,
     precision and recall live.
  4. Save validation overlays (ground-truth green + predicted red boxes with
     class names and confidence) plus Ultralytics curves (PR, F1, confusion
     matrix) into the outputs directory for qualitative inspection.

The ``ultralytics`` import is lazy (inside the training method) so that the
augmentation step and unit tests run without the YOLO dependency installed.
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw

# Project root so ``helpers/`` and ``YOLO.augment`` import from any cwd.
SCRIPT_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from helpers.config import load_config
from helpers.hyperparams import load_hyperparams
from helpers.logger import setup_logger
from helpers.telemetry_writer import TelemetryWriter
from YOLO.augment import list_sample_dirs

# Telemetry columns mirror Ultralytics results.csv (B = box branch).
TELEMETRY_FIELDS: list[str] = [
    "epoch",
    "train_box", "train_cls", "train_dfl",
    "val_box", "val_cls", "val_dfl",
    "precision", "recall", "mAP50", "mAP50_95", "lr",
]

# Ultralytics results.csv column names mapped onto TELEMETRY_FIELDS order.
RESULTS_CSV_MAP: list[tuple[str, str]] = [
    ("epoch", "epoch"),
    ("train_box", "train/box_loss"),
    ("train_cls", "train/cls_loss"),
    ("train_dfl", "train/dfl_loss"),
    ("val_box", "val/box_loss"),
    ("val_cls", "val/cls_loss"),
    ("val_dfl", "val/dfl_loss"),
    ("precision", "metrics/precision(B)"),
    ("recall", "metrics/recall(B)"),
    ("mAP50", "metrics/mAP50(B)"),
    ("mAP50_95", "metrics/mAP50-95(B)"),
    ("lr", "lr/pg0"),
]


def resolve_train_device(requested: str, log: SimpleNamespace) -> str:
    """Resolve the requested device to an Ultralytics-compatible device string.

    Args:
        requested (str): One of "mps", "cuda" or "cpu".
        log (SimpleNamespace): Configured logger for fallback warnings.

    Returns:
        str: Device string to pass to ``YOLO.train`` ("mps", "cuda" or "cpu").
    """
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested != "cpu":
        log.warning(f"Device '{requested}' unavailable; falling back to CPU.")
    return "cpu"


def read_results_csv(results_path: str) -> list[dict[str, str]]:
    """Read an Ultralytics ``results.csv`` into a list of row dicts.

    Args:
        results_path (str): Path to the results.csv file.

    Returns:
        list[dict[str, str]]: One dict per epoch with raw string values.
            Empty list when the file does not exist (training crashed early).
    """
    if not os.path.isfile(results_path):
        return []
    with open(results_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
        pred_boxes (list[tuple]): Predicted (x1, y1, x2, y2, cls, conf) in
            absolute pixels.
        class_names (list[str]): Class index -> display name mapping.

    Returns:
        Image.Image: Annotated copy of the input image.
    """
    img = img.copy()
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(img)
    w_px: int
    h_px: int
    w_px, h_px = img.size
    # Ground truth in green: normalized cxcywh -> pixel corners.
    for raw in gt_lines:
        if not raw.strip():
            continue
        parts: list[str] = raw.split()
        cls_id: int = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1: float = (cx - bw / 2.0) * w_px
        y1: float = (cy - bh / 2.0) * h_px
        x2: float = (cx + bw / 2.0) * w_px
        y2: float = (cy + bh / 2.0) * h_px
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1 + 2, y1 + 2), f"GT:{class_names[cls_id]}", fill=(0, 255, 0))
    # Predictions in red with confidence scores.
    for x1, y1, x2, y2, cls_id, conf in pred_boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1 + 2, y2 + 2), f"{class_names[cls_id]} {conf:.2f}", fill=(255, 0, 0))
    return img


class YOLORunner:
    """Runner wiring dataset preparation, YOLO26 finetune and telemetry."""

    def __init__(self, mode: str) -> None:
        """Load config/hyperparams and set up logging + telemetry.

        Args:
            mode (str): Run mode ("debug" or "prod") selecting config overrides.
        """
        self.config: SimpleNamespace = load_config(mode, str(SCRIPT_DIR / "config.toml"))
        self.hyperparams: SimpleNamespace = load_hyperparams(
            str(SCRIPT_DIR / "hyperparameters.toml"))
        self.log = setup_logger(self.config)
        self.telemetry_writer: TelemetryWriter = TelemetryWriter(
            fieldnames=TELEMETRY_FIELDS,
            directory=self.config.TELEMETRY_LOG_DIRECTORY,
            config=self.config,
        )
        self.device_str: str = resolve_train_device(self.config.DEVICE, self.log)
        self.log.info(
            f"Starting YOLO training run_id: {self.config.runtime.RUN_ID} "
            f"on device: {self.device_str}")

    def check_dataset(self) -> str:
        """Fail fast unless a complete augmented dataset is already on disk.

        The trainer never builds data (build it first with
        ``./run_augmentation.sh``) so a finished dataset is never touched.
        Completeness = manifest exists, train PNG count equals manifest rows,
        and data.yaml plus non-empty val/test mirrors exist.

        Returns:
            str: Absolute path of the existing ``data.yaml``.

        Raises:
            FileNotFoundError: If any required artifact is missing or the
                train set is incomplete, naming the recovery command.
        """
        manifest: Path = Path(self.config.AUGMENT_MANIFEST_PATH)
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Augment manifest missing: {manifest}. "
                "Build the dataset first: ./run_augmentation.sh")
        with open(manifest, encoding="utf-8") as f:
            manifest_rows: int = sum(1 for _ in f) - 1  # minus the header row
        # A partial/interrupted augmentation is self-consistent (images match
        # its own manifest) but still unusable, so verify the manifest itself
        # covers every labeled source (read-only use of the augment knobs).
        aug_hyperparams: SimpleNamespace = load_hyperparams(
            str(SCRIPT_DIR / "augment_hyperparameters.toml"))
        # list_sample_dirs only returns folders that carry BOTH the render and
        # the label file, so its length is already the labeled-source count.
        src_samples: list[Path] = list_sample_dirs(
            str(self.config.SOURCE_TRAIN_SPLIT_DIR),
            str(aug_hyperparams.SOURCE_IMAGE_FILE_NAME),
            str(aug_hyperparams.SOURCE_LABEL_FILE_NAME),
        )
        max_samples: int = int(aug_hyperparams.MAX_SAMPLES)
        if max_samples >= 0:
            src_samples = src_samples[:max_samples]
        aug_factor: int = int(aug_hyperparams.AUG_FACTOR)
        expected_rows: int = len(src_samples) * aug_factor
        if manifest_rows != expected_rows:
            raise FileNotFoundError(
                f"Incomplete augmentation: manifest has {manifest_rows} rows, "
                f"expected {expected_rows} ({len(src_samples)} labeled sources x "
                f"{aug_factor}). Finish it: ./run_augmentation.sh")
        train_imgs: list[Path] = list(
            (Path(self.config.AUGMENTED_DATASET_ROOT) / "images" / "train").glob("*.png"))
        if len(train_imgs) != manifest_rows:
            raise FileNotFoundError(
                f"Incomplete augmented train set: {len(train_imgs)} images vs "
                f"{manifest_rows} manifest rows. Rebuild: ./run_augmentation.sh")
        data_yaml: Path = Path(self.config.DATA_YAML_PATH)
        if not data_yaml.is_file():
            raise FileNotFoundError(
                f"data.yaml missing: {data_yaml}. Rebuild: ./run_augmentation.sh")
        for split in ("val", "test"):
            split_dir: Path = Path(self.config.AUGMENTED_DATASET_ROOT) / "images" / split
            if not split_dir.is_dir() or len(list(split_dir.glob("*.png"))) == 0:
                raise FileNotFoundError(
                    f"Augmented {split} split empty: {split_dir}. "
                    "Rebuild: ./run_augmentation.sh")
        self.log.info(
            f"Dataset OK: {len(train_imgs)} train images, manifest {manifest_rows} rows.")
        return str(data_yaml)

    def mirror_results_to_telemetry(self, results_path: str) -> int:
        """Copy Ultralytics per-epoch rows into the shared telemetry CSV.

        Args:
            results_path (str): Path to Ultralytics ``results.csv``.

        Returns:
            int: Number of epochs mirrored.
        """
        rows: list[dict[str, str]] = read_results_csv(results_path)
        mirrored: int = 0
        for row in rows:
            metrics: dict[str, float] = {}
            try:
                for ours, theirs in RESULTS_CSV_MAP:
                    metrics[ours] = float(row[theirs])
            except (KeyError, ValueError) as exc:
                self.log.warning(f"Skipping malformed results.csv row {row}: {exc}")
                continue
            # Write every TELEMETRY_INTERVAL-th epoch plus the final row.
            epoch_num: int = int(metrics["epoch"])
            is_last: bool = row is rows[-1]
            if epoch_num % int(self.hyperparams.TELEMETRY_INTERVAL) == 0 or is_last:
                self.telemetry_writer.log(**metrics)
                mirrored += 1
        self.log.info(f"Mirrored {mirrored} epochs to telemetry.")
        return mirrored

    def save_prediction_overlays(self, model: object, epoch_tag: str) -> int:
        """Run inference on val samples and save GT-vs-pred overlay grids.

        Args:
            model (object): Trained Ultralytics YOLO model (``predict`` API).
            epoch_tag (str): Tag included in output filenames (e.g. "final").

        Returns:
            int: Number of overlay images saved.
        """
        val_img_dir: Path = Path(self.config.AUGMENTED_DATASET_ROOT) / "images" / "val"
        val_lbl_dir: Path = Path(self.config.AUGMENTED_DATASET_ROOT) / "labels" / "val"
        out_dir: Path = Path(self.config.OUTPUT_SAMPLES_DIRECTORY)
        out_dir.mkdir(parents=True, exist_ok=True)
        class_names: list[str] = list(self.hyperparams.CLASS_NAMES)
        conf: float = float(self.hyperparams.CONF_THRESHOLD)
        val_images: list[Path] = sorted(val_img_dir.glob("*.png"))
        num: int = min(int(self.hyperparams.NUM_OVERLAY_SAMPLES), len(val_images))
        saved: int = 0
        for img_path in val_images[:num]:
            img: Image.Image = Image.open(img_path).convert("RGB")
            lbl_path: Path = val_lbl_dir / f"{img_path.stem}.txt"
            gt_lines: list[str] = []
            if lbl_path.is_file():
                gt_lines = lbl_path.read_text(encoding="utf-8").splitlines()
            # Ultralytics predict returns pixel-space xyxy boxes.
            results = model.predict(source=str(img_path), conf=conf, verbose=False)  # type: ignore[attr-defined]
            pred_boxes: list[tuple[float, float, float, float, int, float]] = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.tolist()
                cls_list = boxes.cls.tolist()
                conf_list = boxes.conf.tolist()
                for (x1, y1, x2, y2), c, cf in zip(xyxy, cls_list, conf_list):
                    pred_boxes.append((x1, y1, x2, y2, int(c), float(cf)))
            annotated: Image.Image = draw_boxes_on_image(img, gt_lines, pred_boxes, class_names)
            out_path: Path = out_dir / (
                f"val_pred_{epoch_tag}_{img_path.stem}_{self.config.runtime.RUN_ID}.png")
            annotated.save(str(out_path))
            saved += 1
        self.log.info(f"Saved {saved} val prediction overlays (tag={epoch_tag}).")
        return saved

    def copy_training_artifacts(self, save_dir: str) -> None:
        """Copy Ultralytics plots into the outputs dir for dashboard inspection.

        Args:
            save_dir (str): Ultralytics run directory (holds results.png,
                PR_curve.png, F1_curve.png, confusion_matrix.png, labels.jpg).
        """
        out_dir: Path = Path(self.config.OUTPUT_SAMPLES_DIRECTORY)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("results.png", "PR_curve.png", "F1_curve.png",
                     "confusion_matrix.png", "labels.jpg", "results.csv"):
            src: Path = Path(save_dir) / name
            if src.is_file():
                shutil.copy(str(src), str(out_dir / f"{Path(save_dir).name}_{name}"))
                self.log.info(f"Copied artifact {name} to outputs.")

    def train(self) -> None:
        """Finetune YOLO26 on the augmented dataset with full telemetry."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run "
                "`./venv/bin/pip install ultralytics` (see YOLO/README.md)."
            ) from exc

        data_yaml: str = self.check_dataset()
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)

        model = YOLO(str(self.hyperparams.YOLO_MODEL))
        self.log.info(f"Loaded pretrained weights: {self.hyperparams.YOLO_MODEL}")
        save_dir: str = str(Path(self.config.OUTPUT_SAMPLES_DIRECTORY).resolve()
                            / f"ultralytics_{self.config.runtime.RUN_ID}")

        model.train(
            data=data_yaml,
            epochs=int(self.hyperparams.NUM_EPOCHS),
            imgsz=int(self.hyperparams.IMG_SIZE),
            batch=int(self.hyperparams.BATCH_SIZE),
            lr0=float(self.hyperparams.LR0),
            optimizer=str(self.hyperparams.OPTIMIZER),
            patience=int(self.hyperparams.PATIENCE),
            device=self.device_str,
            workers=int(self.hyperparams.WORKERS),
            seed=int(self.hyperparams.BASE_SEED),
            # Internal augmentation is pinned off for geometry (our VAE
            # pipeline already applies VAE-latent + blur); hsv kept.
            mosaic=float(self.hyperparams.MOSAIC),
            mixup=float(self.hyperparams.MIXUP),
            copy_paste=float(self.hyperparams.COPY_PASTE),
            fliplr=float(self.hyperparams.FLIPLR),
            flipud=float(self.hyperparams.FLIPUD),
            hsv_h=float(self.hyperparams.HSV_H),
            hsv_s=float(self.hyperparams.HSV_S),
            hsv_v=float(self.hyperparams.HSV_V),
            project=save_dir,
            name="train",
            verbose=True,
        )
        run_dir: str = os.path.join(save_dir, "train")
        self.mirror_results_to_telemetry(os.path.join(run_dir, "results.csv"))
        self.copy_training_artifacts(run_dir)
        self.save_prediction_overlays(model, epoch_tag="final")

        # Persist the finetuned weights next to the run's other artifacts.
        best_src: str = os.path.join(run_dir, "weights", "best.pt")
        if os.path.isfile(best_src):
            best_dst: str = os.path.join(
                self.config.MODEL_SAVE_PATH, f"yolo_{self.config.runtime.RUN_ID}_best.pt")
            shutil.copy(best_src, best_dst)
            self.log.info(f"Saved best weights to {best_dst}")

        # Final validation summary on the mirrored val split, plus an unbiased
        # report on the held-out test split (never early-stopped on).
        metrics = model.val(data=data_yaml, device=self.device_str, verbose=False)
        try:
            self.log.info(
                f"FINAL VAL: mAP50={metrics.box.map50:.4f} "
                f"mAP50-95={metrics.box.map:.4f} "
                f"precision={metrics.box.mp:.4f} recall={metrics.box.mr:.4f}")
        except AttributeError:
            self.log.info(f"FINAL VAL metrics: {metrics}")
        test_img_dir: Path = Path(self.config.AUGMENTED_DATASET_ROOT) / "images" / "test"
        if test_img_dir.is_dir() and len(list(test_img_dir.glob("*.png"))) > 0:
            test_metrics = model.val(
                data=data_yaml, split="test", device=self.device_str, verbose=False)
            try:
                self.log.info(
                    f"FINAL TEST: mAP50={test_metrics.box.map50:.4f} "
                    f"mAP50-95={test_metrics.box.map:.4f} "
                    f"precision={test_metrics.box.mp:.4f} recall={test_metrics.box.mr:.4f}")
            except AttributeError:
                self.log.info(f"FINAL TEST metrics: {test_metrics}")
        else:
            self.log.warning("No mirrored test split; skipping final test report.")


def main() -> None:
    """Parse CLI args and launch the YOLO finetune run."""
    parser = argparse.ArgumentParser(description="Finetune YOLO26 on VAE-augmented data.")
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    args: argparse.Namespace = parser.parse_args()
    YOLORunner(mode=args.mode).train()


if __name__ == "__main__":
    main()
