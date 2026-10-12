"""YOLO26 finetune runner on live diffusion-drawn floor plans.

Flow (no fixed dataset is ever built; every available sample is used):

1. Index the whole folder-per-sample mask dataset (train/val/test splits,
   ``SOURCE_SAMPLE_CAP``/``VAL_SAMPLE_CAP``/``TEST_SAMPLE_CAP`` keep every
   sample by default) and load the frozen rendering stack
   (:mod:`YOLO.dataset`).
2. Render the deterministic val/test GT overlays once (skipped when a content
   signature says they are already current), then encode the train floorplan
   conditions once with the frozen floorplan VAE.
3. Draw the epoch-0 train pixels, then launch Ultralytics. At the end of EVERY
   epoch (``on_fit_epoch_end`` callback, which fires after validation and
   metrics are on disk) the runner:
     a. mirrors the fresh ``results.csv`` row into the project telemetry CSV
        so the dashboard plots live,
     b. re-predicts a FIXED subset of val images with the epoch's own (EMA)
        weights and saves green-GT + red-prediction overlays plus a montage
        under ``YOLO/outputs/epoch_XX/``,
     c. redraws every train image from a FRESH diffusion draw for the next
        epoch and pushes the freshly drawn boxes into the trainer's in-memory
        dataset labels (Ultralytics snapshots labels once at dataloader
        construction and only re-reads images, so the push is what keeps each
        epoch's targets attached to that epoch's pixels; requires
        ``WORKERS = 0``). Skipped on the final/early-stopped epoch.
4. After training: mirror any remaining telemetry rows, copy Ultralytics
   curves, persist the best weights next to the run's artifacts, and report
   the final val plus the unbiased held-out test metrics.

Budget guard: Ultralytics only steps the optimizer once per
gradient-accumulation group, so a train set too small for the configured
``nbs`` would silently do nothing. :func:`validate_training_budget` fails fast
before the (expensive) first draw in that case.

The ``ultralytics`` import is lazy (inside :meth:`YOLORunner.train`) so unit
tests run without the YOLO dependency installed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Ensure PyTorch falls back to CPU for ops unsupported on MPS (e.g.
# torchvision::nms) so validation and prediction do not fail on macOS Apple
# Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Project root so ``helpers/`` and ``YOLO.*`` import from any cwd.
SCRIPT_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from helpers.config import load_config
from helpers.hyperparams import load_hyperparams
from helpers.logger import setup_logger
from helpers.telemetry_writer import TelemetryWriter
from YOLO.dataset import (
    DeterministicSplitRenderer,
    FrozenRenderStack,
    LiveEpochRedrawer,
    SampleFolder,
    apply_label_lines_to_dataset,
    clear_generated_split,
    list_sample_folders,
    write_data_yaml,
)
from YOLO.overlay import choose_fixed_stems, save_epoch_overlays

# Telemetry columns mirror Ultralytics results.csv (B = box branch).
TELEMETRY_FIELDS: list[str] = [
    "epoch",
    "train_box", "train_cls", "train_dfl",
    "val_box", "val_cls", "val_dfl",
    "precision", "recall", "mAP50", "mAP50_95", "lr",
]

# Config keys that are filesystem paths. Relative ones are anchored to the
# repository root (not the current working directory) so the trainer behaves
# identically no matter where it is launched from: ``train_YOLO.sh`` cds to
# the root, but a bare ``python YOLO/run.py`` from elsewhere must not scatter
# logs, telemetry, checkpoints and dataset manifests across the filesystem.
# Absolute paths (e.g. the live dataset root on another disk) pass through.
_PATH_CONFIG_KEYS: tuple[str, ...] = (
    "DEBUG_LOG_FILE_PATH",
    "MODEL_SAVE_PATH",
    "TELEMETRY_LOG_DIRECTORY",
    "OUTPUT_SAMPLES_DIRECTORY",
    "COND_DIFFUSION_CHECKPOINT_PATH",
    "COND_VAE_CHECKPOINT_PATH",
    "TARGET_VAE_CHECKPOINT_PATH",
    "LIVE_DATASET_ROOT",
    "LIVE_DATA_YAML_PATH",
    "LIVE_MANIFEST_PATH",
)


def anchor_config_paths(config: SimpleNamespace) -> None:
    """Resolve relative config paths against the repository root in place.

    Args:
        config (SimpleNamespace): Parsed config whose path keys are rewritten
            to absolute paths anchored at the repo root. Missing keys are
            left untouched (the run fails later with a clear KeyError naming
            the key).
    """
    for key in _PATH_CONFIG_KEYS:
        if not hasattr(config, key):
            continue
        path: Path = Path(str(getattr(config, key)))
        setattr(config, key, str(path if path.is_absolute() else ROOT_DIR / path))
    # The project logger writes to this file on setup, so its directory must
    # exist even on a fresh clone.
    Path(str(config.DEBUG_LOG_FILE_PATH)).parent.mkdir(parents=True, exist_ok=True)

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

# Ultralytics artifacts copied into the outputs directory for the dashboard.
_ULTRALYTICS_ARTIFACTS: tuple[str, ...] = (
    "results.png",
    "PR_curve.png",
    "F1_curve.png",
    "confusion_matrix.png",
    "labels.jpg",
    "results.csv",
)


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
    mAP50 ~0.05). This raises instead, so the mistake cannot reach a multi-hour
    run.

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
            "YOLO/hyperparameters.toml, and/or raise the sample caps."
        )
    if steps_per_epoch * num_epochs < 2:
        raise ValueError(
            f"Training budget too small: {num_train_images} images over "
            f"{num_epochs} epochs yields only {steps_per_epoch * num_epochs} "
            "optimizer steps in total. Raise the sample caps or NUM_EPOCHS."
        )
    log.info(
        f"Optimizer budget: {num_train_images} images, batch {batch_size}, "
        f"nbs {nominal_batch_size} -> {steps_per_epoch} optimizer step(s) per "
        f"epoch, {steps_per_epoch * num_epochs} over {num_epochs} epochs."
    )
    if steps_per_epoch < 4:
        log.warning(
            f"Only {steps_per_epoch} optimizer step(s) per epoch; convergence "
            "will be slow and noisy. Raise the sample caps for real runs."
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


def _map_results_row(
    row: dict[str, str], log: SimpleNamespace
) -> dict[str, float] | None:
    """Map one results.csv row onto the telemetry field names.

    Args:
        row (dict[str, str]): Raw results.csv row.
        log (SimpleNamespace): Configured logger for malformed-row warnings.

    Returns:
        dict[str, float] | None: Mapped metrics, or None when the row is
            malformed (missing columns or unparseable numbers).
    """
    metrics: dict[str, float] = {}
    try:
        for ours, candidates in RESULTS_CSV_MAP:
            for candidate in candidates:
                if candidate in row and row[candidate] not in ("", None):
                    metrics[ours] = float(row[candidate])
                    break
            else:
                raise KeyError(f"none of {candidates} present in results.csv row")
    except (KeyError, ValueError) as exc:
        log.warning(f"Skipping malformed results.csv row {row}: {exc}")
        return None
    return metrics


def _resolve_overlay_model(
    trainer: object,
    live_model: object,
    overlay_model: object,
    overlay_ready: list[bool],
) -> object:
    """Pick the weights the epoch-end overlays should be drawn with.

    Ultralytics validates (and saves as ``best.pt``) the EMA weights, so the
    overlays should show those: the trainer's EMA module is a raw ``nn.Module``
    without a ``predict`` API, so its state dict is loaded into the dedicated
    ``overlay_model`` wrapper. The architecture swap is lazy because the
    trainer rebuilds the model for ``data.yaml``'s class count only when
    ``train()`` starts (a fresh ``yolo26n.pt`` wrapper still has its 80 COCO
    classes, which would not accept the EMA state dict). Without an EMA the
    live training model itself is used.

    Args:
        trainer (object): Ultralytics trainer (for its ``ema`` attribute).
        live_model (object): The training YOLO wrapper (fallback weights).
        overlay_model (object): Dedicated YOLO wrapper receiving EMA weights.
        overlay_ready (list[bool]): One-element mutable flag tracking whether
            the architecture swap already happened.

    Returns:
        object: The YOLO wrapper to predict with.
    """
    ema_module: object | None = getattr(getattr(trainer, "ema", None), "ema", None)
    if ema_module is None:
        return live_model
    if not overlay_ready[0]:
        overlay_model.model = copy.deepcopy(trainer.model)  # type: ignore[attr-defined]
        overlay_ready[0] = True
    overlay_model.model.load_state_dict(ema_module.state_dict())  # type: ignore[attr-defined]
    return overlay_model


class YOLORunner:
    """Runner wiring live-diffusion data, YOLO26 finetune and telemetry."""

    def __init__(self, mode: str) -> None:
        """Load config/hyperparams and set up logging + telemetry.

        Args:
            mode (str): Run mode ("debug" or "prod") selecting config overrides.
        """
        self.config: SimpleNamespace = load_config(mode, str(SCRIPT_DIR / "config.toml"))
        anchor_config_paths(self.config)
        self.hyperparams: SimpleNamespace = load_hyperparams(
            str(SCRIPT_DIR / "hyperparameters.toml")
        )
        self.log: SimpleNamespace = setup_logger(self.config)
        self.telemetry_writer: TelemetryWriter = TelemetryWriter(
            fieldnames=TELEMETRY_FIELDS,
            directory=self.config.TELEMETRY_LOG_DIRECTORY,
            config=self.config,
        )
        self.device_str: str = resolve_train_device(self.config.DEVICE, self.log)
        self.class_names: list[str] = list(self.hyperparams.CLASS_NAMES)
        self.class_aliases: dict[str, str] = dict(
            getattr(self.hyperparams, "CLASS_ALIASES", {}) or {}
        )
        self.log.info(
            f"Starting YOLO training run_id: {self.config.runtime.RUN_ID} "
            f"on device: {self.device_str}"
        )

    # -- dataset preparation -------------------------------------------------

    def _index_split(
        self, split_dir: str, cap: int, stack: FrozenRenderStack, role: str
    ) -> list[SampleFolder]:
        """Index one source split, tolerating only a missing test split.

        Args:
            split_dir (str): Split folder holding one sub-folder per sample.
            cap (int): Pool cap over samples; negative keeps every sample.
            stack (FrozenRenderStack): Provides the required mask filenames.
            role (str): Split role name for log messages ("train"/"val"/"test").

        Returns:
            list[SampleFolder]: Indexed sample folders of the split.
        """
        sources: list[SampleFolder] = list_sample_folders(
            split_dir,
            stack.floorplan_mask_file,
            stack.arrangement_mask_file,
            cap,
        )
        self.log.info(f"Indexed {len(sources)} {role} samples from {split_dir}.")
        return sources

    def _prepare_live_dataset(
        self, stack: FrozenRenderStack, redrawer: LiveEpochRedrawer
    ) -> tuple[str, list[SampleFolder], torch.Tensor, list[str]]:
        """Build the live dataset: deterministic val/test + epoch-0 train draw.

        Args:
            stack (FrozenRenderStack): Frozen VAEs + diffusion denoiser.
            redrawer (LiveEpochRedrawer): Per-epoch train redrawer.

        Returns:
            tuple[str, list[SampleFolder], torch.Tensor, list[str]]:
                (data.yaml path, train sources, cached condition latents,
                fixed overlay stems).

        Raises:
            FileNotFoundError: If train/val source split dirs are missing.
            ValueError: If a split has no usable samples.
        """
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

        test_split_src: str = str(self.config.LIVE_SOURCE_TEST_SPLIT_DIR)
        test_sources: list[SampleFolder] = (
            self._index_split(
                test_split_src, int(self.hyperparams.TEST_SAMPLE_CAP), stack, "test"
            )
            if Path(test_split_src).is_dir()
            else []
        )
        if not test_sources:
            self.log.warning("Live test source dir missing; skipping test split.")
        val_sources: list[SampleFolder] = self._index_split(
            str(self.config.LIVE_SOURCE_VAL_SPLIT_DIR),
            int(self.hyperparams.VAL_SAMPLE_CAP),
            stack,
            "val",
        )
        train_sources: list[SampleFolder] = self._index_split(
            str(self.config.LIVE_SOURCE_TRAIN_SPLIT_DIR),
            int(self.hyperparams.SOURCE_SAMPLE_CAP),
            stack,
            "train",
        )

        # Fresh stale-artifact sweep: a train PNG or label left over from a
        # previous run (different stems) would otherwise leak into
        # Ultralytics' file list, and a label cache from another stem set
        # would mislabel this one.
        clear_generated_split(train_img_dir, train_lbl_dir)
        self._clear_ultralytics_cache(live_root)
        manifest_path: Path = Path(self.config.LIVE_MANIFEST_PATH)
        if manifest_path.is_file():
            manifest_path.unlink()

        # Deterministic val/test GT overlays, rendered once per configuration.
        deterministic: DeterministicSplitRenderer = DeterministicSplitRenderer(
            stack, self.hyperparams, self.log
        )
        deterministic.render_split(val_sources, val_img_dir, val_lbl_dir)
        if test_sources:
            deterministic.render_split(test_sources, test_img_dir, test_lbl_dir)

        data_yaml: str = write_data_yaml(
            str(live_root), str(self.config.LIVE_DATA_YAML_PATH), self.class_names
        )

        # Conditions are cached once: GT floorplans never change, so only the
        # diffusion noise varies per epoch.
        cond_norm: torch.Tensor = stack.encode_source_conditions(train_sources)
        validate_training_budget(
            len(train_sources),
            int(self.hyperparams.BATCH_SIZE),
            int(self.hyperparams.NOMINAL_BATCH_SIZE),
            int(self.hyperparams.NUM_EPOCHS),
            self.log,
        )

        # Epoch-0 draw: epoch 1 trains on fresh pixels; the end-of-epoch
        # callback then redraws for every later epoch (seed = BASE_SEED + E).
        redrawer.redraw_epoch(
            train_sources, cond_norm, 0, train_img_dir, train_lbl_dir, manifest_path
        )

        overlay_stems: list[str] = choose_fixed_stems(
            [source.stem for source in val_sources],
            int(self.hyperparams.EPOCH_OVERLAY_STEMS),
            int(self.hyperparams.BASE_SEED),
        )
        outputs_dir: Path = Path(self.config.OUTPUT_SAMPLES_DIRECTORY)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        (outputs_dir / "epoch_overlay_stems.txt").write_text(
            "\n".join(overlay_stems) + "\n", encoding="utf-8"
        )
        return data_yaml, train_sources, cond_norm, overlay_stems

    def _clear_ultralytics_cache(self, live_root: Path) -> int:
        """Delete stale Ultralytics ``*.cache`` files under the live root.

        The label cache records the train stem set; the stem set can change
        between runs (different caps), so a cache from a previous run must not
        survive. Train pixels change every epoch, but stems/labels/shapes do
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

    # -- telemetry -----------------------------------------------------------

    def mirror_results_to_telemetry(
        self, results_path: str, after_epoch: int = 0, live: bool = False
    ) -> int:
        """Copy Ultralytics per-epoch rows into the shared telemetry CSV.

        Only rows with ``epoch > after_epoch`` are mirrored, so the epoch-end
        callback can mirror incrementally while training runs (the dashboard
        polls the CSV live) and the final sweep picks up exactly the rest -
        no epoch is ever mirrored twice.

        Args:
            results_path (str): Path to Ultralytics ``results.csv``.
            after_epoch (int): Skip rows at or below this epoch number
                (already mirrored earlier).
            live (bool): True from the per-epoch callback: mirror every new row
                so the dashboard plots while training runs. False applies
                ``TELEMETRY_INTERVAL`` thinning (used by the final sweep).

        Returns:
            int: Number of epochs mirrored by this call.
        """
        rows: list[dict[str, str]] = read_results_csv(results_path)
        mirrored: int = 0
        for row in rows:
            metrics: dict[str, float] | None = _map_results_row(row, self.log)
            if metrics is None:
                continue
            epoch_num: int = int(metrics["epoch"])
            if epoch_num <= after_epoch:
                continue
            # Write every TELEMETRY_INTERVAL-th epoch plus the final row when
            # thinning; live callbacks always mirror their own fresh row.
            is_last: bool = row is rows[-1]
            if (
                live
                or epoch_num % int(self.hyperparams.TELEMETRY_INTERVAL) == 0
                or is_last
            ):
                self.telemetry_writer.log(**metrics)
                mirrored += 1
        if mirrored:
            self.log.info(f"Mirrored {mirrored} epochs to telemetry.")
        return mirrored

    def copy_training_artifacts(self, save_dir: str) -> None:
        """Copy Ultralytics plots into the outputs dir for dashboard inspection.

        Args:
            save_dir (str): Ultralytics run directory (holds results.png,
                PR_curve.png, F1_curve.png, confusion_matrix.png, labels.jpg).
        """
        out_dir: Path = Path(self.config.OUTPUT_SAMPLES_DIRECTORY)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in _ULTRALYTICS_ARTIFACTS:
            src: Path = Path(save_dir) / name
            if src.is_file():
                shutil.copy(str(src), str(out_dir / f"{Path(save_dir).name}_{name}"))
                self.log.info(f"Copied artifact {name} to outputs.")

    # -- training ------------------------------------------------------------

    def train(self) -> None:
        """Finetune YOLO26 on live diffusion overlays with full telemetry."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run "
                "`./venv/bin/pip install ultralytics` (see YOLO/README.md)."
            ) from exc

        # The per-epoch redraw syncs the trainer's in-memory dataset labels
        # (see apply_label_lines_to_dataset). With worker processes each
        # worker holds its own fork of that list, so the sync would silently
        # not reach training: fail fast instead of training on stale boxes.
        if int(self.hyperparams.WORKERS) != 0:
            raise ValueError(
                f"WORKERS must be 0 for live redraw training (got "
                f"{self.hyperparams.WORKERS}): worker processes would snapshot "
                "the train labels at fork time, so per-epoch redrawn boxes "
                "would never reach the loss. Set WORKERS = 0 in "
                "YOLO/hyperparameters.toml."
            )

        stack: FrozenRenderStack = FrozenRenderStack(
            self.config, self.hyperparams, self.log
        )
        redrawer: LiveEpochRedrawer = LiveEpochRedrawer(
            stack, self.hyperparams, self.log
        )
        (
            data_yaml,
            train_sources,
            cond_norm,
            overlay_stems,
        ) = self._prepare_live_dataset(stack, redrawer)
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)

        model = YOLO(str(self.hyperparams.YOLO_MODEL))
        self.log.info(f"Loaded pretrained weights: {self.hyperparams.YOLO_MODEL}")
        # Separate wrapper for the per-epoch overlay predictions: the trainer's
        # EMA module is a raw nn.Module without a ``predict`` API, so the EMA
        # state dict is loaded into this wrapper to visualize exactly the
        # weights validation scored. Its architecture is swapped in lazily on
        # the first epoch end (see _resolve_overlay_model).
        overlay_model = YOLO(str(self.hyperparams.YOLO_MODEL))
        overlay_ready: list[bool] = [False]
        save_dir: str = str(
            Path(self.config.OUTPUT_SAMPLES_DIRECTORY).resolve()
            / f"ultralytics_{self.config.runtime.RUN_ID}"
        )
        run_dir: str = os.path.join(save_dir, "train")
        # Ultralytics appends results.csv once per epoch under its run dir;
        # mirroring it at epoch end keeps the dashboard live while training.
        ultralytics_results: str = os.path.join(run_dir, "results.csv")

        live_root: Path = Path(self.config.LIVE_DATASET_ROOT)
        train_img_dir: Path = live_root / "images" / "train"
        train_lbl_dir: Path = live_root / "labels" / "train"
        manifest_path: Path = Path(self.config.LIVE_MANIFEST_PATH)
        val_img_dir: Path = live_root / "images" / "val"
        val_lbl_dir: Path = live_root / "labels" / "val"
        outputs_dir: Path = Path(self.config.OUTPUT_SAMPLES_DIRECTORY)
        base_seed: int = int(self.hyperparams.BASE_SEED)
        num_epochs: int = int(self.hyperparams.NUM_EPOCHS)
        # Idempotency guard for the epoch-end callback: tracked via the
        # results.csv row count (see the callback), so it also absorbs the
        # extra on_fit_epoch_end firing from Ultralytics' post-training
        # final_eval.
        handled_epochs: list[int] = [0]

        def _on_fit_epoch_end(trainer: object) -> None:
            """Epoch-end callback: telemetry + GT/pred panel + next-epoch draw."""
            # Dedup key: results.csv gains exactly one row per completed
            # training epoch (save_metrics runs before this callback), so the
            # row count is immune to the epoch bump Ultralytics applies inside
            # its post-training final_eval (which re-fires this callback).
            completed: int = len(read_results_csv(ultralytics_results))
            if completed <= handled_epochs[0]:
                return
            handled_epochs[0] = completed
            epoch: int = completed  # 1-based epoch number just finished
            self.log.info(f"Epoch {epoch} finished (validation metrics on disk).")

            # a) fresh telemetry row -> dashboard CSV (live, no thinning).
            self.mirror_results_to_telemetry(
                ultralytics_results, after_epoch=epoch - 1, live=True
            )

            # b) GT vs predicted boxes on the fixed val subset, using the
            #    weights this epoch was validated with (EMA when available).
            predict_model: object = _resolve_overlay_model(
                trainer, model, overlay_model, overlay_ready
            )
            summary: dict[str, float] = save_epoch_overlays(
                predict_model,
                overlay_stems,
                val_img_dir,
                val_lbl_dir,
                outputs_dir,
                epoch,
                self.class_names,
                float(self.hyperparams.EPOCH_OVERLAY_CONF),
                int(self.hyperparams.EPOCH_OVERLAY_MAX_BOXES),
                int(self.hyperparams.EPOCH_OVERLAY_MONTAGE_COLS),
                int(self.hyperparams.IMG_SIZE),
            )
            self.log.info(
                f"Epoch {epoch} overlays: {int(summary['images'])} images, "
                f"{int(summary['gt_boxes'])} GT boxes, "
                f"{int(summary['pred_boxes'])} predicted boxes "
                f"(peak conf {summary['peak_conf']:.4f}); montage in "
                f"{outputs_dir}/montage_epoch_{epoch:02d}.png."
            )

            # c) fresh diffusion draw for the next epoch's pixels. Skipped on
            #    the final or early-stopped epoch: nothing will read them.
            stopping: bool = bool(getattr(trainer, "stop", False))
            if stopping or epoch >= num_epochs:
                self.log.info(
                    f"Epoch {epoch} is the last epoch; skipping next-epoch redraw."
                )
                return
            stats = redrawer.redraw_epoch(
                train_sources,
                cond_norm,
                epoch,
                train_img_dir,
                train_lbl_dir,
                manifest_path,
            )
            # The pixels are new on disk; the boxes must follow into the
            # trainer's in-memory dataset, because Ultralytics snapshots
            # labels once at dataloader construction and only re-reads images.
            synced: int = apply_label_lines_to_dataset(
                trainer.train_loader.dataset,  # type: ignore[attr-defined]
                stats.label_lines_by_stem,
            )
            self.log.info(
                f"Epoch {epoch} redraw for epoch {epoch + 1}: "
                f"{stats.images} images, {stats.boxes} boxes in "
                f"{stats.seconds:.1f}s (seed={stats.seed}); "
                f"synced {synced} in-memory label entries."
            )

        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

        model.train(
            data=data_yaml,
            epochs=num_epochs,
            imgsz=int(self.hyperparams.IMG_SIZE),
            batch=int(self.hyperparams.BATCH_SIZE),
            lr0=float(self.hyperparams.LR0),
            optimizer=str(self.hyperparams.OPTIMIZER),
            patience=int(self.hyperparams.PATIENCE),
            device=self.device_str,
            workers=int(self.hyperparams.WORKERS),
            seed=base_seed,
            # Nominal batch size drives gradient accumulation
            # (accumulate = round(nbs / batch)). Pinning it equal to the batch
            # size disables accumulation, so every batch updates the weights;
            # the Ultralytics default nbs=64 would otherwise step only once
            # every `accumulate` batches and can starve a small train set
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
        # Final sweep: mirrors whatever the epoch-end callbacks have not yet
        # covered (e.g. rows flushed after the last callback, or everything
        # when callbacks never fired).
        self.mirror_results_to_telemetry(
            ultralytics_results, after_epoch=handled_epochs[0]
        )
        self.copy_training_artifacts(run_dir)

        # Persist the finetuned weights next to the run's other artifacts.
        best_src: str = os.path.join(run_dir, "weights", "best.pt")
        if os.path.isfile(best_src):
            best_dst: str = os.path.join(
                self.config.MODEL_SAVE_PATH, f"yolo_{self.config.runtime.RUN_ID}_best.pt"
            )
            shutil.copy(best_src, best_dst)
            self.log.info(f"Saved best weights to {best_dst}")

        # Final validation summary on the deterministic val split, plus an
        # unbiased report on the held-out test split (never early-stopped on).
        metrics = model.val(data=data_yaml, device=self.device_str, verbose=False)
        self._log_final_metrics("FINAL VAL", metrics)
        test_img_dir: Path = live_root / "images" / "test"
        if test_img_dir.is_dir() and any(test_img_dir.glob("*.png")):
            test_metrics = model.val(
                data=data_yaml, split="test", device=self.device_str, verbose=False
            )
            self._log_final_metrics("FINAL TEST", test_metrics)
        else:
            self.log.warning("No test split; skipping final test report.")

    def _log_final_metrics(self, label: str, metrics: object) -> None:
        """Log the headline detection metrics of a validation run.

        Args:
            label (str): Prefix for the log line (e.g. "FINAL VAL").
            metrics (object): Ultralytics validation metrics object.
        """
        try:
            self.log.info(
                f"{label}: mAP50={metrics.box.map50:.4f} "  # type: ignore[attr-defined]
                f"mAP50-95={metrics.box.map:.4f} "  # type: ignore[attr-defined]
                f"precision={metrics.box.mp:.4f} "  # type: ignore[attr-defined]
                f"recall={metrics.box.mr:.4f}"  # type: ignore[attr-defined]
            )
        except AttributeError:
            self.log.info(f"{label} metrics: {metrics}")


def main() -> None:
    """Parse CLI args and launch the live-diffusion YOLO finetune run."""
    parser = argparse.ArgumentParser(
        description="Finetune YOLO26 on live diffusion overlays."
    )
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    args: argparse.Namespace = parser.parse_args()
    YOLORunner(mode=args.mode).train()


if __name__ == "__main__":
    main()
