"""YOLO26 finetune runner on live conditional-diffusion overlay floor plans.

Flow (no fixed dataset is ever built):
  1. Index the live mask dataset (GT floorplan + GT arrangement + GT labels),
     pick the run's train stems once, and encode their GT floorplans once
     with the frozen floorplan VAE (see ``YOLO/live_diffusion.py``).
  2. Redraw every train PNG from a FRESH diffusion draw before epoch 1 and at
     the end of every epoch (Ultralytics ``on_train_epoch_end`` callback);
     GT labels stay the target despite the layout mismatch (intentional
     noisy-label regime — YOLO learns a floorplan-conditioned layout prior).
  3. Finetune ``YOLO(<YOLO_MODEL>)`` (default ``yolo26n.pt``) with Ultralytics
     on MPS (fallback CPU), ``workers=0`` for macOS, ``cache=False`` so the
     per-epoch pixel rewrites are actually seen (a cache would freeze the
     first epoch's pixels forever).
  4. Mirror Ultralytics ``results.csv`` rows into the shared
     ``TelemetryWriter`` CSV incrementally at every epoch end (plus a final
     sweep), so the existing Flask dashboard plots loss, mAP, precision and
     recall live while training runs.
  5. Save validation overlays (ground-truth green + predicted red boxes with
     class names and confidence) plus Ultralytics curves (PR, F1, confusion
     matrix) into the outputs directory for qualitative inspection.

Val/test are deterministic GT overlays (no diffusion), so early stopping and
the final unbiased report stay stable while train pixels vary every epoch.

The ``ultralytics`` import is lazy (inside the training method) so that unit
tests run without the YOLO dependency installed. Torch/diffusion imports stay
inside ``YOLO.live_diffusion`` for the same reason.
"""

import argparse
import csv
import math
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
from YOLO.augment import clear_generated_split, resolve_class_name_to_id
from YOLO.live_diffusion import (
    LiveDiffusionPipeline,
    LiveSource,
    list_live_sources,
    select_train_subset,
    write_live_data_yaml,
)

# Telemetry columns mirror Ultralytics results.csv (B = box branch).
TELEMETRY_FIELDS: list[str] = [
    "epoch",
    "train_box", "train_cls", "train_dfl",
    "val_box", "val_cls", "val_dfl",
    "precision", "recall", "mAP50", "mAP50_95", "lr",
]

# Ultralytics results.csv column names mapped onto TELEMETRY_FIELDS order.
# Each entry lists fallback columns: YOLO26 writes the distribution-focal term
# as ``*/l1_loss`` while older generations wrote ``*/dfl_loss``. The telemetry
# field names stay ``*_dfl`` so the existing dashboard keeps plotting them.
RESULTS_CSV_MAP: list[tuple[str, list[str]]] = [
    ("epoch", ["epoch"]),
    ("train_box", ["train/box_loss"]),
    ("train_cls", ["train/cls_loss"]),
    ("train_dfl", ["train/dfl_loss", "train/l1_loss"]),
    ("val_box", ["val/box_loss"]),
    ("val_cls", ["val/cls_loss"]),
    ("val_dfl", ["val/dfl_loss", "val/l1_loss"]),
    ("precision", ["metrics/precision(B)"]),
    ("recall", ["metrics/recall(B)"]),
    ("mAP50", ["metrics/mAP50(B)"]),
    ("mAP50_95", ["metrics/mAP50-95(B)"]),
    ("lr", ["lr/pg0"]),
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


def optimizer_steps_per_epoch(
    num_train_images: int, batch_size: int, nominal_batch_size: int
) -> int:
    """Return how many optimizer steps one epoch actually performs.

    Ultralytics accumulates gradients over ``accumulate = round(nbs / batch)``
    batches (see ``BaseTrainer._do_train``) and only calls ``optimizer_step()``
    once that many batches have been seen. An epoch therefore performs
    ``batches_per_epoch // accumulate`` real weight updates, NOT
    ``batches_per_epoch``: with fewer batches than the accumulation factor the
    epoch loop runs to completion without ever updating the weights, and the
    validation metrics stay bit-identical epoch after epoch.

    Args:
        num_train_images (int): Number of train images Ultralytics will see.
        batch_size (int): Per-step batch size (``BATCH_SIZE``).
        nominal_batch_size (int): Ultralytics ``nbs`` (``NOMINAL_BATCH_SIZE``),
            the nominal batch size that sets the accumulation factor.

    Returns:
        int: Optimizer steps performed in one epoch (0 when starved).
    """
    batches_per_epoch: int = max(1, math.ceil(num_train_images / max(1, batch_size)))
    accumulate: int = max(1, round(nominal_batch_size / max(1, batch_size)))
    return batches_per_epoch // accumulate


def validate_training_budget(
    num_train_images: int,
    batch_size: int,
    nominal_batch_size: int,
    num_epochs: int,
    log: SimpleNamespace,
) -> int:
    """Fail fast when the run is too small to actually train, else warn.

    Guards the silent failure mode where a tiny train set combined with
    gradient accumulation yields a full-looking epoch loop in which the weights
    are barely updated (observed: 8 images, 30 epochs, 10 optimizer steps,
    mAP50 ~0.05). This raises instead, so the mistake cannot reach a 3-hour run.

    Args:
        num_train_images (int): Number of train images for the run.
        batch_size (int): Per-step batch size (``BATCH_SIZE``).
        nominal_batch_size (int): Ultralytics ``nbs`` (``NOMINAL_BATCH_SIZE``).
        num_epochs (int): Configured epoch count (``NUM_EPOCHS``).
        log (SimpleNamespace): Configured logger for the budget summary.

    Returns:
        int: Optimizer steps performed per epoch (always >= 1 on return).

    Raises:
        ValueError: If an epoch would perform no optimizer step at all, or if
            the whole run would perform fewer than two.
    """
    steps_per_epoch: int = optimizer_steps_per_epoch(
        num_train_images, batch_size, nominal_batch_size
    )
    if steps_per_epoch < 1:
        raise ValueError(
            f"Training budget too small: {num_train_images} images at "
            f"BATCH_SIZE={batch_size} with NOMINAL_BATCH_SIZE="
            f"{nominal_batch_size} yields {steps_per_epoch} optimizer steps per "
            "epoch, so the weights would never be updated. Set "
            "NOMINAL_BATCH_SIZE = BATCH_SIZE (accumulation 1) in "
            "YOLO/hyperparameters.toml, and/or raise LIVE_TRAIN_SAMPLES."
        )
    if steps_per_epoch * num_epochs < 2:
        raise ValueError(
            f"Training budget too small: {num_train_images} images over "
            f"{num_epochs} epochs yields only {steps_per_epoch * num_epochs} "
            "optimizer steps in total. Raise LIVE_TRAIN_SAMPLES or NUM_EPOCHS."
        )
    log.info(
        f"Optimizer budget: {num_train_images} images, batch {batch_size}, "
        f"nbs {nominal_batch_size} -> {steps_per_epoch} optimizer step(s) per "
        f"epoch, {steps_per_epoch * num_epochs} over {num_epochs} epochs."
    )
    if steps_per_epoch < 4:
        log.warning(
            f"Only {steps_per_epoch} optimizer step(s) per epoch; convergence "
            "will be slow and noisy. Raise LIVE_TRAIN_SAMPLES for real runs."
        )
    return steps_per_epoch


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
    # Fine 1px lines: overlays are small (256px), so width=1 keeps adjacent
    # furniture boxes distinguishable instead of merging into thick bars.
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
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=1)
        draw.text((x1 + 2, y1 + 2), f"GT:{class_names[cls_id]}", fill=(0, 255, 0))
    # Predictions in red with confidence scores.
    for x1, y1, x2, y2, cls_id, conf in pred_boxes:
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=1)
        draw.text((x1 + 2, y2 + 2), f"{class_names[cls_id]} {conf:.2f}", fill=(255, 0, 0))
    return img


class YOLORunner:
    """Runner wiring live-diffusion data, YOLO26 finetune and telemetry."""

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

    def clear_stale_ultralytics_cache(self, live_root: Path) -> int:
        """Delete stale Ultralytics ``*.cache`` files under the live root.

        The label cache records the train stem set; the subset is reselected
        every run, so a cache from a previous run (different stems) must not
        survive. Train pixels change every epoch but stems/labels/shapes do
        not, so no per-epoch clearing is needed.

        Args:
            live_root (Path): Live dataset root to clean.

        Returns:
            int: Number of cache files removed.
        """
        removed: int = 0
        if live_root.is_dir():
            for cache_file in live_root.rglob("*.cache"):
                cache_file.unlink()
                removed += 1
        if removed:
            self.log.info(f"Removed {removed} stale Ultralytics cache files.")
        return removed

    def index_live_split(
        self,
        split_dir: str,
        class_name_to_id: dict[str, int],
        max_sources: int,
    ) -> list[LiveSource]:
        """Index one live split (fail fast when it has no usable samples).

        Args:
            split_dir (str): Split folder holding one sub-folder per sample.
            class_name_to_id (dict[str, int]): Class name -> id map used to
                validate the GT labels once at indexing time.
            max_sources (int): Pool cap; negative keeps every sample.

        Returns:
            list[LiveSource]: Sorted live sources with validated GT labels.

        Raises:
            FileNotFoundError: If the split directory is missing.
            ValueError: If no sample provides both masks plus GT labels.
        """
        sources: list[LiveSource] = list_live_sources(
            split_dir,
            str(self.config.COND_MASK_FILE),
            str(self.config.TARGET_MASK_FILE),
            str(self.config.SOURCE_LABEL_FILE_NAME),
            class_name_to_id,
            max_sources,
        )
        if not sources:
            raise ValueError(
                f"Live split has no usable samples: {split_dir}. Every sample "
                f"folder must carry '{self.config.COND_MASK_FILE}', "
                f"'{self.config.TARGET_MASK_FILE}' and "
                f"'{self.config.SOURCE_LABEL_FILE_NAME}'."
            )
        self.log.info(f"Indexed {len(sources)} live sources from {split_dir}.")
        return sources

    def prepare_live_dataset(
        self,
        pipeline: LiveDiffusionPipeline,
        class_name_to_id: dict[str, int],
    ) -> tuple[str, list[LiveSource], torch.Tensor]:
        """Prepare dirs, deterministic val/test mirrors, data.yaml + train set.

        Train stems are selected once and their GT floorplans encoded once;
        the initial pixel draw (epoch seed 0) lands before Ultralytics ever
        indexes the files. Val/test mirrors are deterministic GT overlays,
        skipped when already complete.

        Args:
            pipeline (LiveDiffusionPipeline): Frozen diffusion + render stack.
            class_name_to_id (dict[str, int]): Class name -> id map for GT
                label validation.

        Returns:
            tuple[str, list[LiveSource], torch.Tensor]: (data.yaml path, the
                run's train sources in stem order, their cached normalized
                condition latents on CPU).

        Raises:
            FileNotFoundError: If a live source split directory is missing.
            ValueError: If any split has no usable samples.
        """
        max_sources: int = int(self.hyperparams.LIVE_MAX_SOURCES)
        pool: list[LiveSource] = self.index_live_split(
            str(self.config.LIVE_SOURCE_TRAIN_SPLIT_DIR),
            class_name_to_id,
            max_sources,
        )
        train_sources: list[LiveSource] = select_train_subset(
            pool, int(self.hyperparams.LIVE_TRAIN_SAMPLES), int(self.hyperparams.BASE_SEED)
        )
        # Fail fast BEFORE the expensive diffusion setup: a too-small train set
        # would otherwise produce an epoch loop that never updates the weights.
        validate_training_budget(
            len(train_sources),
            int(self.hyperparams.BATCH_SIZE),
            int(self.hyperparams.NOMINAL_BATCH_SIZE),
            int(self.hyperparams.NUM_EPOCHS),
            self.log,
        )
        self.log.info(
            f"Live train set: {len(train_sources)} stems "
            f"(pool {len(pool)}, redrawn from fresh diffusion draws every epoch)."
        )

        live_root: Path = Path(self.config.LIVE_DATASET_ROOT)
        train_img_dir: Path = live_root / "images" / "train"
        train_lbl_dir: Path = live_root / "labels" / "train"
        val_img_dir: Path = live_root / "images" / "val"
        val_lbl_dir: Path = live_root / "labels" / "val"
        test_img_dir: Path = live_root / "images" / "test"
        test_lbl_dir: Path = live_root / "labels" / "test"
        for directory in (
            train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir,
            test_img_dir, test_lbl_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        # Fresh subset every run: drop last run's pixels and its cache.
        clear_generated_split(train_img_dir, train_lbl_dir)
        self.clear_stale_ultralytics_cache(live_root)
        old_manifest: Path = Path(self.config.LIVE_MANIFEST_PATH)
        if old_manifest.is_file():
            old_manifest.unlink()

        val_sources: list[LiveSource] = self.index_live_split(
            str(self.config.LIVE_SOURCE_VAL_SPLIT_DIR),
            class_name_to_id,
            max_sources,
        )
        pipeline.mirror_deterministic_split(val_sources, val_img_dir, val_lbl_dir)
        test_split_src: str = str(getattr(self.config, "LIVE_SOURCE_TEST_SPLIT_DIR", ""))
        if test_split_src and Path(test_split_src).is_dir():
            test_sources: list[LiveSource] = self.index_live_split(
                test_split_src, class_name_to_id, max_sources
            )
            pipeline.mirror_deterministic_split(test_sources, test_img_dir, test_lbl_dir)
        else:
            self.log.warning("Live test source dir missing; skipping test mirroring.")

        data_yaml: str = write_live_data_yaml(
            str(live_root),
            str(self.config.LIVE_DATA_YAML_PATH),
            list(self.hyperparams.CLASS_NAMES),
        )
        train_cond: torch.Tensor = pipeline.encode_source_conditions(train_sources)
        # Epoch-0 draw: epoch 1 trains on fresh pixels, and the end-of-epoch
        # callback redraws for every later epoch (seed = BASE_SEED + epoch).
        pipeline.regenerate_train_epoch(
            train_sources,
            train_cond,
            0,
            int(self.hyperparams.BASE_SEED),
            train_img_dir,
            train_lbl_dir,
            old_manifest,
        )
        return data_yaml, train_sources, train_cond

    def mirror_results_to_telemetry(
        self, results_path: str, after_epoch: int = 0
    ) -> int:
        """Copy Ultralytics per-epoch rows into the shared telemetry CSV.

        Only rows with ``epoch > after_epoch`` are mirrored, so the epoch-end
        callback can mirror incrementally while training runs (the dashboard
        polls the CSV live) and the final sweep picks up exactly the rest —
        no epoch is ever mirrored twice.

        Note on cadence: a live callback always sees its just-finished epoch
        as the file's last row, so every epoch lands in telemetry while
        training runs; ``TELEMETRY_INTERVAL`` thins only rows mirrored by a
        non-live (single-shot) call.

        Args:
            results_path (str): Path to Ultralytics ``results.csv``.
            after_epoch (int): Skip rows at or below this epoch number
                (already mirrored earlier).

        Returns:
            int: Number of epochs mirrored by this call.
        """
        rows: list[dict[str, str]] = read_results_csv(results_path)
        mirrored: int = 0
        for row in rows:
            metrics: dict[str, float] = {}
            try:
                for ours, candidates in RESULTS_CSV_MAP:
                    for candidate in candidates:
                        if candidate in row and row[candidate] not in ("", None):
                            metrics[ours] = float(row[candidate])
                            break
                    else:
                        raise KeyError(
                            f"none of {candidates} present in results.csv row")
            except (KeyError, ValueError) as exc:
                self.log.warning(f"Skipping malformed results.csv row {row}: {exc}")
                continue
            if int(metrics["epoch"]) <= after_epoch:
                continue
            # Write every TELEMETRY_INTERVAL-th epoch plus the final row.
            epoch_num: int = int(metrics["epoch"])
            is_last: bool = row is rows[-1]
            if epoch_num % int(self.hyperparams.TELEMETRY_INTERVAL) == 0 or is_last:
                self.telemetry_writer.log(**metrics)
                mirrored += 1
        if mirrored:
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
        val_img_dir: Path = Path(self.config.LIVE_DATASET_ROOT) / "images" / "val"
        val_lbl_dir: Path = Path(self.config.LIVE_DATASET_ROOT) / "labels" / "val"
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
        """Finetune YOLO26 on live diffusion overlays with full telemetry."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run "
                "`./venv/bin/pip install ultralytics` (see YOLO/README.md)."
            ) from exc

        pipeline: LiveDiffusionPipeline = LiveDiffusionPipeline(
            self.config, self.hyperparams, self.log
        )
        class_name_to_id: dict[str, int] = resolve_class_name_to_id(
            list(self.hyperparams.CLASS_NAMES),
            dict(getattr(self.hyperparams, "CLASS_ALIASES", {}) or {}),
        )
        data_yaml, train_sources, train_cond = self.prepare_live_dataset(
            pipeline, class_name_to_id
        )
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)

        model = YOLO(str(self.hyperparams.YOLO_MODEL))
        self.log.info(f"Loaded pretrained weights: {self.hyperparams.YOLO_MODEL}")
        save_dir: str = str(Path(self.config.OUTPUT_SAMPLES_DIRECTORY).resolve()
                            / f"ultralytics_{self.config.runtime.RUN_ID}")

        # Per-epoch pixel refresh INSIDE the single Ultralytics run: the
        # dataloader (workers=0, cache=False) re-reads PNGs from disk every
        # batch, and stems never change, so rewriting pixels here is safe.
        # Internal scheduler, patience and results.csv keep working untouched.
        live_root: Path = Path(self.config.LIVE_DATASET_ROOT)
        train_img_dir: Path = live_root / "images" / "train"
        train_lbl_dir: Path = live_root / "labels" / "train"
        manifest_path: Path = Path(self.config.LIVE_MANIFEST_PATH)
        base_seed: int = int(self.hyperparams.BASE_SEED)
        completed_epochs: list[int] = [0]
        # Ultralytics appends results.csv once per epoch under its run dir;
        # mirroring it here keeps the dashboard live while training runs.
        ultralytics_results: str = os.path.join(save_dir, "train", "results.csv")

        def _redraw_for_next_epoch(_trainer: object) -> None:
            """Ultralytics epoch-end callback: redraw pixels + mirror telemetry."""
            completed_epochs[0] += 1
            self.log.info(
                f"Epoch {completed_epochs[0]} done; redrawing live train pixels...")
            pipeline.regenerate_train_epoch(
                train_sources,
                train_cond,
                completed_epochs[0],
                base_seed,
                train_img_dir,
                train_lbl_dir,
                manifest_path,
            )
            self.mirror_results_to_telemetry(
                ultralytics_results, after_epoch=completed_epochs[0] - 1
            )

        model.add_callback("on_train_epoch_end", _redraw_for_next_epoch)

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
            # Nominal batch size drives gradient accumulation
            # (accumulate = round(nbs / batch)). Pinning it equal to the batch
            # size disables accumulation, so every batch updates the weights;
            # the Ultralytics default nbs=64 would otherwise step only once
            # every `accumulate` batches and can starve a small live train set
            # (see validate_training_budget).
            nbs=int(self.hyperparams.NOMINAL_BATCH_SIZE),
            # cache=False is load-bearing for live training: the train PNGs
            # are rewritten every epoch and any snapshot cache would freeze
            # the first epoch's pixels forever.
            cache=False,
            # Internal augmentation is pinned off for geometry (the live
            # diffusion draws already supply the variation); hsv kept.
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
        # Final sweep: mirrors whatever the epoch-end callbacks have not yet
        # covered (e.g. rows flushed after the last callback, or everything
        # when callbacks never fired).
        self.mirror_results_to_telemetry(
            os.path.join(run_dir, "results.csv"),
            after_epoch=completed_epochs[0],
        )
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
        test_img_dir: Path = Path(self.config.LIVE_DATASET_ROOT) / "images" / "test"
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
    """Parse CLI args and launch the live-diffusion YOLO finetune run."""
    parser = argparse.ArgumentParser(description="Finetune YOLO26 on live diffusion overlays.")
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    args: argparse.Namespace = parser.parse_args()
    YOLORunner(mode=args.mode).train()


if __name__ == "__main__":
    main()
