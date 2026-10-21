"""missing_yolo trainer: one run, one missing component.

Flow (no prebuilt dataset is ever reused across runs; everything a run needs is
generated inside its own dataset tree):

1. Resolve the run's component from ``MISSING_COMPONENT`` / ``--component``
   (:func:`missing_yolo.dataset.resolve_component_variant`), load the frozen
   generator and validate it against this run's knobs
   (:class:`missing_yolo.dataset.FrozenMaskDiffusionStack`).
2. Index ``work/train`` for training and carve ``work/validate`` into two
   family-disjoint halves (val / test). Every sample is checked for the
   dataset's own ``labels.txt`` before anything expensive happens, and those
   boxes become the run's detection targets.
3. Render val and test ONCE from the frozen generator with fixed seeds (skipped
   when a content signature says they are already current), encode both
   generator conditions for all three splits once, then draw the train pixels
   for epoch 1.
4. Train Ultralytics YOLO26. At the end of every epoch
   (``on_fit_epoch_end``) the runner mirrors the fresh ``results.csv`` row into
   the project telemetry CSV so the dashboard plots live, publishes GT-vs-
   prediction overlays for a FIXED val subset using that epoch's EMA weights,
   and redraws every train image from a fresh generator draw for the next
   epoch.
5. After training: mirror whatever telemetry is left, copy the Ultralytics
   curves, persist the best weights beside the run artifacts, and report the
   final val plus the held-out test metrics.

Labels do NOT move: every target comes from ``labels.txt``, which is fixed per
sample, while only the pixels are regenerated each epoch. Ultralytics reads the
label files once when the train dataloader is built, so the once-read labels
stay correct for the whole run - no in-memory patching is needed (see
``missing_yolo/README.md`` for why that is safe here and not in ``YOLO/``).

Budget guard: Ultralytics only steps the optimizer once per
gradient-accumulation group, so a train set too small for the configured ``nbs``
would silently do nothing. :func:`validate_training_budget` (reused from
``YOLO.runner``) fails fast before the expensive first draw in that case.

The ``ultralytics`` import is lazy (inside :meth:`MissingYoloRunner.train`) so
unit tests run without the YOLO dependency installed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

# Ensure PyTorch falls back to CPU for ops unsupported on MPS (e.g.
# torchvision::nms) so validation and prediction do not fail on macOS Apple
# Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Project root so ``helpers/`` and ``missing_yolo`` import from any cwd.
SCRIPT_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from helpers.config import load_config
from helpers.hyperparams import load_hyperparams
from helpers.logger import setup_logger
from helpers.telemetry_writer import TelemetryWriter
from missing_yolo.dataset import (
    ArrangementGenerator,
    ComponentPaths,
    ComponentVariant,
    ConditionCache,
    FrozenMaskDiffusionStack,
    GenerationStats,
    carve_family_splits,
    component_paths,
    require_label_files,
    resolve_component_variant,
)
from missing_yolo.labels import LabelIndex, build_label_index
from YOLO.dataset import SampleFolder, list_sample_folders, write_data_yaml
from YOLO.overlay import choose_fixed_stems, save_epoch_overlays
from YOLO.runner import (
    ULTRALYTICS_ARTIFACTS,
    map_results_row,
    read_results_csv,
    resolve_overlay_model,
    resolve_train_device,
    validate_training_budget,
)

# Ultralytics detection metrics, mirrored verbatim from results.csv. Shared with
# the YOLO stage so both dashboards plot the same column names.
DETECTION_TELEMETRY_FIELDS: tuple[str, ...] = (
    "train_box", "train_cls", "train_dfl",
    "val_box", "val_cls", "val_dfl",
    "precision", "recall", "mAP50", "mAP50_95", "lr",
)

# Generation telemetry columns. Prefixed to keep them distinguishable from the
# detection metrics in one CSV.
GENERATION_TELEMETRY_FIELDS: tuple[str, ...] = (
    "draw_index",
    "draw_seed",
    "draw_images",
    "draw_seconds",
    "agreement_scored",
    "agreement_mean",
)

# Values written into a telemetry cell the pass did not measure. Empty, never
# 0.0: an unmeasured column must not read as a measured zero on the dashboard.
_UNMEASURED: str = ""

# Config keys that are filesystem paths. Relative ones are anchored to the
# repository root (not the current working directory) so the trainer behaves
# identically no matter where it is launched from: ``train_missing_yolo.sh`` cds
# to the root, but a bare ``python missing_yolo/run.py`` from elsewhere must not
# scatter logs, telemetry and checkpoints across the filesystem. Absolute paths
# (e.g. the dataset root on another disk) pass through. The per-component
# generator paths are anchored separately, by ``resolve_component_variant``.
_PATH_CONFIG_KEYS: tuple[str, ...] = (
    "DEBUG_LOG_FILE_PATH",
    "MODEL_SAVE_PATH",
    "TELEMETRY_LOG_DIRECTORY",
    "OUTPUT_SAMPLES_DIRECTORY",
    "LIVE_DATASET_BASE_ROOT",
    "VALIDATION_OUTPUT_DIRECTORY",
)


def anchor_config_paths(config: SimpleNamespace) -> None:
    """Resolve relative config paths against the repository root in place.

    Args:
        config (SimpleNamespace): Parsed config whose path keys are rewritten to
            absolute paths anchored at the repo root. Missing keys are left
            untouched (the run fails later with a clear KeyError naming the key).
    """
    for key in _PATH_CONFIG_KEYS:
        if not hasattr(config, key):
            continue
        path: Path = Path(str(getattr(config, key)))
        setattr(config, key, str(path if path.is_absolute() else ROOT_DIR / path))
    # The project logger writes to this file on setup, so its directory must
    # exist even on a fresh clone.
    Path(str(config.DEBUG_LOG_FILE_PATH)).parent.mkdir(parents=True, exist_ok=True)


def telemetry_fields(hyperparams: SimpleNamespace) -> list[str]:
    """Build the telemetry column order for this run.

    One CSV carries two kinds of row, distinguished by ``row_kind``: a
    ``detection`` row holds one epoch's Ultralytics metrics, a ``generation``
    row holds the agreement of the pixels drawn for an epoch. Cells the row did
    not measure stay empty, so one row never claims a metric it does not have.

    Args:
        hyperparams (SimpleNamespace): Run hyperparameters, read for the
            agreement thresholds whose columns are appended.

    Returns:
        list[str]: Column order handed to :class:`TelemetryWriter`.
    """
    fields: list[str] = ["row_kind", "epoch", *DETECTION_TELEMETRY_FIELDS]
    fields.extend(GENERATION_TELEMETRY_FIELDS)
    fields.extend(
        f"agreement_frac_{float(threshold):g}"
        for threshold in hyperparams.AGREEMENT_BOX_IOU_THRESHOLDS
    )
    return fields


def generation_row(stats: GenerationStats, train_for_epoch: int) -> dict[str, object]:
    """Turn one generation pass into a telemetry row.

    Args:
        stats (GenerationStats): The pass's provenance and agreement record.
        train_for_epoch (int): 1-based epoch the drawn pixels will train. Draw
            index ``d`` trains epoch ``d + 1``, so the row sits one step ahead
            of the detection row of the epoch that triggered it.

    Returns:
        dict[str, object]: Row keyed by telemetry column name. The
            ``row_kind`` marks it as a generation row; the caller fills the
            unmeasured detection columns with :data:`_UNMEASURED`.
    """
    row: dict[str, object] = {
        "row_kind": "generation",
        "epoch": int(train_for_epoch),
        **{field: _UNMEASURED for field in DETECTION_TELEMETRY_FIELDS},
        "draw_index": int(stats.epoch),
        "draw_seed": int(stats.seed),
        "draw_images": int(stats.images),
        "draw_seconds": f"{stats.seconds:.1f}",
        "agreement_scored": int(stats.agreement_scored),
        "agreement_mean": (
            _UNMEASURED if stats.agreement_mean is None else f"{stats.agreement_mean:.6f}"
        ),
    }
    for threshold in stats.agreement_fractions:
        value: float = stats.agreement_fractions[threshold]
        row[f"agreement_frac_{threshold}"] = (
            _UNMEASURED if value != value else f"{value:.6f}"  # NaN check
        )
    return row


class MissingYoloRunner:
    """Runner wiring the frozen generator, YOLO26 finetune and telemetry."""

    def __init__(self, mode: str, component: str | None = None) -> None:
        """Load config/hyperparams, resolve the component and set up logging.

        Args:
            mode (str): Run mode ("debug" or "prod") selecting config overrides.
            component (str | None): Component to run, overriding the
                ``MISSING_COMPONENT`` hyperparameter. None keeps the TOML value,
                which is the default path a run script takes.
        """
        self.config: SimpleNamespace = load_config(mode, str(SCRIPT_DIR / "config.toml"))
        anchor_config_paths(self.config)
        self.hyperparams: SimpleNamespace = load_hyperparams(
            str(SCRIPT_DIR / "hyperparameters.toml")
        )
        self.component: str = (
            str(component) if component else str(self.hyperparams.MISSING_COMPONENT)
        )
        # Resolve before the logger exists so a bad component fails with a
        # message naming the valid ones, rather than inside the logger setup.
        self.variant: ComponentVariant = resolve_component_variant(
            self.config, self.component
        )
        self.log: SimpleNamespace = setup_logger(self.config)
        self.telemetry_writer: TelemetryWriter = TelemetryWriter(
            fieldnames=telemetry_fields(self.hyperparams),
            directory=str(self.config.TELEMETRY_LOG_DIRECTORY),
            config=self.config,
        )
        self.device_str: str = resolve_train_device(str(self.config.DEVICE), self.log)
        self.paths: ComponentPaths = component_paths(self.config, self.variant)
        self.class_names: list[str] = [self.variant.detector_class_name]
        self.log.info(
            f"Starting missing_yolo run_id: {self.config.runtime.RUN_ID} on device: "
            f"{self.device_str}; component '{self.component}' "
            f"(class '{self.variant.detector_class_name}', labels "
            f"{list(self.variant.label_names)}) from generator "
            f"{self.variant.mask_diffusion_checkpoint.name}; dataset root "
            f"{self.paths.root}."
        )
        self.log.info(
            "Splits: train = SOURCE_TRAIN_SPLIT_DIR (the frozen generator was "
            "trained on it, so train metrics are generation metrics); val and "
            "test = family-disjoint halves of SOURCE_VAL_SPLIT_DIR, which the "
            "generator never saw."
        )

    # -- dataset preparation -------------------------------------------------

    def _index_sources(self, split_dir: str, cap: int, stack: FrozenMaskDiffusionStack, role: str) -> list[SampleFolder]:
        """Index one source split and check it can be labelled.

        Args:
            split_dir (str): Split folder holding one sub-folder per sample.
            cap (int): Pool cap over samples; negative keeps every sample.
            stack (FrozenMaskDiffusionStack): Provides the required mask filenames.
            role (str): Split role name for log messages.

        Returns:
            list[SampleFolder]: Indexed sample folders of the split.
        """
        sources: list[SampleFolder] = list_sample_folders(
            split_dir,
            stack.floorplan_group.mask_file_name,
            stack.target_group.mask_file_name,
            cap,
        )
        require_label_files(sources, str(self.config.GROUND_TRUTH_LABEL_FILE))
        self.log.info(f"Indexed {len(sources)} {role} samples from {split_dir}.")
        return sources

    def _index_val_test(
        self, stack: FrozenMaskDiffusionStack
    ) -> tuple[list[SampleFolder], list[SampleFolder]]:
        """Index the validate split and carve it into val and test halves.

        Args:
            stack (FrozenMaskDiffusionStack): Provides the required mask filenames.

        Returns:
            tuple[list[SampleFolder], list[SampleFolder]]: ``(val, test)``
                sources, family-disjoint and capped individually.
        """
        val_split_dir: str = str(self.config.SOURCE_VAL_SPLIT_DIR)
        pooled: list[SampleFolder] = self._index_sources(
            val_split_dir, -1, stack, "validate (pooled)"
        )
        val_sources, test_sources = carve_family_splits(
            pooled,
            float(self.hyperparams.VAL_FAMILY_FRACTION),
            int(self.hyperparams.BASE_SEED),
        )
        val_cap: int = int(self.hyperparams.VAL_SAMPLE_CAP)
        test_cap: int = int(self.hyperparams.TEST_SAMPLE_CAP)
        val_sources = val_sources[:val_cap] if val_cap >= 0 else val_sources
        test_sources = test_sources[:test_cap] if test_cap >= 0 else test_sources
        overlap: set[str] = {source.stem for source in val_sources} & {
            source.stem for source in test_sources
        }
        if overlap:
            raise ValueError(
                f"val and test share {len(overlap)} samples (e.g. "
                f"{sorted(overlap)[:3]}); the split is not family-disjoint."
            )
        self.log.info(
            f"Carved {val_split_dir} into {len(val_sources)} val and "
            f"{len(test_sources)} test samples by layout family."
        )
        return val_sources, test_sources

    def _label_index(self, sources: list[SampleFolder], role: str) -> LabelIndex:
        """Build one split's ``labels.txt`` target index.

        Args:
            sources (list[SampleFolder]): Split sources, in image write order.
            role (str): Split role name for log messages.

        Returns:
            LabelIndex: Targets per stem, with the run's single class id.
        """
        with ThreadPoolExecutor(
            max_workers=max(1, int(self.hyperparams.RENDER_IO_WORKERS))
        ) as pool:
            index: LabelIndex = build_label_index(
                sources,
                str(self.config.GROUND_TRUTH_LABEL_FILE),
                self.variant.label_names,
                0,
                pool,
            )
        self.log.info(
            f"{role} targets: {index.positive_samples} positive samples "
            f"({index.background_samples} kept as background), "
            f"{index.total_boxes} boxes of class "
            f"'{self.variant.detector_class_name}'."
        )
        return index

    def _prepare_dataset(
        self, stack: FrozenMaskDiffusionStack
    ) -> tuple[str, list[SampleFolder], ConditionCache, ArrangementGenerator, list[str]]:
        """Generate the dataset: one-shot val/test, epoch-1 train pixels.

        Args:
            stack (FrozenMaskDiffusionStack): The run's frozen generator.

        Returns:
            tuple[str, list[SampleFolder], ConditionCache, ArrangementGenerator,
                list[str]]: data.yaml path, train sources, cached train
                conditions, the train redrawer, and the fixed overlay stems.

        Raises:
            OSError: If Ultralytics' stale caches or the previous manifest
                cannot be cleared.
        """
        for split in ("train", "val", "test"):
            self.paths.image_dir(split).mkdir(parents=True, exist_ok=True)
            self.paths.label_dir(split).mkdir(parents=True, exist_ok=True)

        train_sources: list[SampleFolder] = self._index_sources(
            str(self.config.SOURCE_TRAIN_SPLIT_DIR),
            int(self.hyperparams.SOURCE_SAMPLE_CAP),
            stack,
            "train",
        )
        val_sources, test_sources = self._index_val_test(stack)

        # Budget guard first: it is a pure arithmetic check, and running it before
        # any generation keeps a too-small train set from costing an hour of
        # diffusion before failing.
        validate_training_budget(
            len(train_sources),
            int(self.hyperparams.BATCH_SIZE),
            int(self.hyperparams.NOMINAL_BATCH_SIZE),
            int(self.hyperparams.NUM_EPOCHS),
            self.log,
        )

        # Fresh stale-artifact sweep: a train PNG or label left over from a
        # previous run (different stems) would leak into Ultralytics' file list,
        # and a label cache from another stem set would mislabel this one.
        self._clear_ultralytics_cache()
        if self.paths.manifest.is_file():
            self.paths.manifest.unlink()

        data_yaml: str = write_data_yaml(
            str(self.paths.root), str(self.paths.data_yaml), self.class_names
        )
        self.log.info(f"Wrote {data_yaml} with classes {self.class_names}.")

        # Conditions are cached once per split: the GT masks never change, so
        # only the generator noise varies per epoch.
        train_cache: ConditionCache = stack.encode_conditions(train_sources)
        val_cache: ConditionCache = stack.encode_conditions(val_sources)
        test_cache: ConditionCache = stack.encode_conditions(test_sources)

        # Val and test are drawn ONCE each with their own fixed seeds, so their
        # pixels - and every metric read off them - are stable for the run.
        val_generator: ArrangementGenerator = ArrangementGenerator(
            stack, self._label_index(val_sources, "val"), self.hyperparams, self.log
        )
        val_generator.render_split_once(
            val_sources,
            val_cache,
            int(self.hyperparams.VAL_RENDER_SEED),
            "val",
            self.paths.image_dir("val"),
            self.paths.label_dir("val"),
            self.paths.manifest,
        )
        test_generator: ArrangementGenerator = ArrangementGenerator(
            stack, self._label_index(test_sources, "test"), self.hyperparams, self.log
        )
        test_generator.render_split_once(
            test_sources,
            test_cache,
            int(self.hyperparams.TEST_RENDER_SEED),
            "test",
            self.paths.image_dir("test"),
            self.paths.label_dir("test"),
            self.paths.manifest,
        )

        train_generator: ArrangementGenerator = ArrangementGenerator(
            stack, self._label_index(train_sources, "train"), self.hyperparams, self.log
        )
        epoch_one: GenerationStats = train_generator.redraw_train_epoch(
            train_sources,
            train_cache,
            0,
            self.paths.image_dir("train"),
            self.paths.label_dir("train"),
            self.paths.manifest,
        )
        self._log_generation(epoch_one, 1)

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
        return data_yaml, train_sources, train_cache, train_generator, overlay_stems

    def _clear_ultralytics_cache(self) -> int:
        """Delete stale Ultralytics ``*.cache`` files under the run's root.

        The label cache records the stem set, which can change between runs
        (different caps or components), so a cache from a previous run must not
        survive. Train pixels change every epoch, but stems, labels and shapes
        do not, so no per-epoch clearing is needed.

        Returns:
            int: Number of cache files removed.

        Raises:
            OSError: If a cache file cannot be removed; a cache whose stem set
                disagrees with this one would mislabel the whole run.
        """
        removed: int = 0
        if self.paths.root.is_dir():
            for cache_file in sorted(self.paths.root.rglob("*.cache")):
                cache_file.unlink()
                removed += 1
        if removed:
            self.log.info(f"Removed {removed} stale Ultralytics cache files.")
        return removed

    # -- telemetry -----------------------------------------------------------

    def _log_generation(self, stats: GenerationStats, train_for_epoch: int) -> None:
        """Append one generation row to the telemetry CSV.

        Args:
            stats (GenerationStats): The pass's record.
            train_for_epoch (int): 1-based epoch the drawn pixels will train.
        """
        row: dict[str, object] = {
            key: _UNMEASURED for key in self.telemetry_writer.writer.fieldnames
        }
        row.update(generation_row(stats, train_for_epoch))
        self.telemetry_writer.log(**row)
        self.log.info(
            f"Telemetry: generation draw {stats.epoch} -> epoch {train_for_epoch} "
            f"recorded (agreement mean {_format(stats.agreement_mean)}, "
            f"fractions {stats.agreement_fractions})."
        )

    def _log_detection(self, results_path: str, epoch: int) -> bool:
        """Append one detection row (one epoch's Ultralytics metrics).

        Args:
            results_path (str): Path to Ultralytics ``results.csv``.
            epoch (int): 1-based epoch number just finished.

        Returns:
            bool: True when a row was written, False when the row was malformed
                (already logged) and nothing was recorded.
        """
        rows: list[dict[str, str]] = read_results_csv(results_path)
        if epoch > len(rows):
            return False
        metrics: dict[str, float] | None = map_results_row(rows[epoch - 1], self.log)
        if metrics is None:
            return False
        row: dict[str, object] = {
            key: _UNMEASURED for key in self.telemetry_writer.writer.fieldnames
        }
        row["row_kind"] = "detection"
        row["epoch"] = epoch
        row.update({field: metrics[field] for field in DETECTION_TELEMETRY_FIELDS})
        self.telemetry_writer.log(**row)
        return True

    def copy_training_artifacts(self, run_dir: str) -> None:
        """Copy Ultralytics plots into the outputs dir for dashboard inspection.

        Args:
            run_dir (str): Ultralytics run directory (holds results.png,
                PR_curve.png, F1_curve.png, confusion_matrix.png, labels.jpg).
        """
        out_dir: Path = Path(self.config.OUTPUT_SAMPLES_DIRECTORY)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ULTRALYTICS_ARTIFACTS:
            src: Path = Path(run_dir) / name
            if src.is_file():
                shutil.copy(str(src), str(out_dir / f"{Path(run_dir).name}_{name}"))
                self.log.info(f"Copied artifact {name} to outputs.")

    # -- training ------------------------------------------------------------

    def train(self) -> None:
        """Train YOLO26 on live missing-component renders with full telemetry."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run "
                "`./venv/bin/pip install ultralytics` (see missing_yolo/README.md)."
            ) from exc

        if int(self.hyperparams.WORKERS) != 0:
            raise ValueError(
                f"WORKERS must be 0 (got {self.hyperparams.WORKERS}): worker "
                "processes would keep serving the previous epoch's pixels from "
                "forked state while this process rewrites the train PNGs. Set "
                "WORKERS = 0 in missing_yolo/hyperparameters.toml."
            )

        stack: FrozenMaskDiffusionStack = FrozenMaskDiffusionStack(
            self.config, self.hyperparams, self.variant, self.log
        )
        (
            data_yaml,
            train_sources,
            train_cache,
            train_generator,
            overlay_stems,
        ) = self._prepare_dataset(stack)
        os.makedirs(str(self.config.MODEL_SAVE_PATH), exist_ok=True)

        model = YOLO(str(self.hyperparams.YOLO_MODEL))
        self.log.info(f"Loaded pretrained weights: {self.hyperparams.YOLO_MODEL}")
        # Separate wrapper for the per-epoch overlay predictions: the trainer's
        # EMA module is a raw nn.Module without a ``predict`` API, so the EMA
        # state dict is loaded into this wrapper to visualize exactly the
        # weights validation scored. Its architecture is swapped in lazily on
        # the first epoch end (see YOLO.runner.resolve_overlay_model).
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

        outputs_dir: Path = Path(self.config.OUTPUT_SAMPLES_DIRECTORY)
        num_epochs: int = int(self.hyperparams.NUM_EPOCHS)
        # Idempotency guard for the epoch-end callback, tracked via the
        # results.csv row count (see the callback), so it also absorbs the extra
        # on_fit_epoch_end firing of Ultralytics' post-training final_eval.
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

            # a) fresh detection row -> dashboard CSV (written before the
            #    minutes-long redraw, so the curve grows while training runs).
            self._log_detection(ultralytics_results, epoch)

            # b) GT vs predicted boxes on the fixed val subset, using the
            #    weights this epoch was validated with (EMA when available).
            predict_model: object = resolve_overlay_model(
                trainer, model, overlay_model, overlay_ready
            )
            summary: dict[str, float] = save_epoch_overlays(
                predict_model,
                overlay_stems,
                self.paths.image_dir("val"),
                self.paths.label_dir("val"),
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

            # c) fresh generator draw for the next epoch's pixels. Skipped on the
            #    final or early-stopped epoch: nothing will read them. Labels
            #    need no syncing: they come from labels.txt and never change.
            stopping: bool = bool(getattr(trainer, "stop", False))
            if stopping or epoch >= num_epochs:
                self.log.info(
                    f"Epoch {epoch} is the last epoch; skipping next-epoch draw."
                )
                return
            stats: GenerationStats = train_generator.redraw_train_epoch(
                train_sources,
                train_cache,
                epoch,
                self.paths.image_dir("train"),
                self.paths.label_dir("train"),
                self.paths.manifest,
            )
            self._log_generation(stats, epoch + 1)

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
            seed=int(self.hyperparams.BASE_SEED),
            # Nominal batch size drives gradient accumulation
            # (accumulate = round(nbs / batch)). Pinning it equal to the batch
            # size disables accumulation, so every batch updates the weights;
            # the Ultralytics default nbs=64 would otherwise step only once
            # every `accumulate` batches and can starve a small train set
            # (see validate_training_budget).
            nbs=int(self.hyperparams.NOMINAL_BATCH_SIZE),
            # cache=False is load-bearing: the train PNGs are rewritten every
            # epoch and any snapshot cache would freeze the first epoch's
            # pixels forever.
            cache=False,
            # Internal augmentation is pinned off for geometry (the per-epoch
            # generation already supplies the variation); hsv kept.
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
        # Final sweep: mirror any epoch whose row never reached the CSV (e.g. a
        # row flushed after the last callback, or everything when the callback
        # never fired).
        for epoch in range(handled_epochs[0] + 1, num_epochs + 1):
            self._log_detection(ultralytics_results, epoch)
        self.copy_training_artifacts(run_dir)

        # Persist the finetuned weights next to the run's other artifacts, named
        # after the component so the three runs are distinguishable on disk.
        best_src: str = os.path.join(run_dir, "weights", "best.pt")
        if os.path.isfile(best_src):
            best_dst: str = os.path.join(
                str(self.config.MODEL_SAVE_PATH),
                f"missing_yolo_{self.component}_{self.config.runtime.RUN_ID}_best.pt",
            )
            shutil.copy(best_src, best_dst)
            self.log.info(f"Saved best weights to {best_dst}")

        # Final validation summary, plus an unbiased report on the held-out test
        # half (never early-stopped on, and never seen by the generator). The
        # project is pinned so these runs' artifacts land in this run's outputs
        # directory instead of Ultralytics' global runs/ folder.
        metrics = model.val(
            data=data_yaml,
            device=self.device_str,
            verbose=False,
            project=save_dir,
            name="val",
        )
        self._log_final_metrics("FINAL VAL", metrics)
        test_metrics = model.val(
            data=data_yaml,
            split="test",
            device=self.device_str,
            verbose=False,
            project=save_dir,
            name="test",
        )
        self._log_final_metrics("FINAL TEST (generator-unseen)", test_metrics)
        self.log.info(
            f"Run summary: component '{self.component}' (class "
            f"'{self.variant.detector_class_name}'), train {len(train_sources)} "
            f"samples from {self.config.SOURCE_TRAIN_SPLIT_DIR}, val/test "
            f"carved from {self.config.SOURCE_VAL_SPLIT_DIR}, dataset root "
            f"{self.paths.root}, manifest {self.paths.manifest}, weights "
            f"{best_src if os.path.isfile(best_src) else 'NOT SAVED'}."
        )

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


def _format(value: float | None) -> str:
    """Format an optional telemetry value for a log line.

    Args:
        value (float | None): The value, None when unmeasured.

    Returns:
        str: Six-decimal text, or ``"n/a"`` when unmeasured - never ``0.0``.
    """
    return "n/a" if value is None else f"{value:.6f}"


def main() -> None:
    """Parse CLI args and launch one missing-component detector run."""
    parser = argparse.ArgumentParser(
        description="Train YOLO26 on live missing-component diffusion renders."
    )
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    parser.add_argument(
        "--component",
        default=None,
        help="Component to train (overrides MISSING_COMPONENT in "
        "missing_yolo/hyperparameters.toml).",
    )
    args: argparse.Namespace = parser.parse_args()
    MissingYoloRunner(mode=args.mode, component=args.component).train()


if __name__ == "__main__":
    main()