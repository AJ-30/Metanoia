"""Standalone per-component validation for the missing_yolo detectors.

One run produces one self-contained report folder per missing component:

    missing_yolo/validation_outputs/<component>/
        dataset/                  regenerated val renders + labels + data.yaml
        ultralytics/val/          Ultralytics curves, confusion matrix, saved predictions
        samples/                  24 balanced raw/annotated reconstructions + montages
        per_image.csv             one row per sample and confidence threshold
        metrics.json              every number the report needs
        metrics.txt               human-readable report

The pass deliberately re-renders the val half from the frozen generator instead
of reusing whatever is already on disk. That makes the report reproducible from
``SOURCE_VAL_SPLIT_DIR`` and the component's generator checkpoint alone, and it
means the pixels the detector is scored on are exactly the pixels this run
describes. ``work/test`` is never touched: the generator saw it, so it cannot
yield an honest number.

The detector weights are not hard-coded here. They come from the shared
``components.toml`` registry, and this module fails if that row's generator does
not match the generator declared in ``missing_yolo/config.toml``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from PIL import Image

# Ultralytics NMS uses a torchvision op without an MPS kernel; fall back to CPU
# for that op while keeping the rest of the run on the configured device.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Project root so ``evaluator``/``helpers``/``missing_yolo``/``YOLO`` import when
# launched as ``python missing_yolo/validate.py`` from any directory.
_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_ROOT_DIR: Path = _SCRIPT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from evaluator.variants import (
    ComponentVariant as SharedComponentVariant,
    load_component_variants,
    resolve_component_variant as resolve_shared_variant,
)
from evaluator.yolo_eval import detector_class_names
from helpers.config import load_config
from helpers.hyperparams import load_hyperparams
from helpers.logger import setup_logger
from missing_yolo.dataset import (
    ArrangementGenerator,
    ComponentVariant,
    FrozenMaskDiffusionStack,
    GenerationStats,
    RENDER_SIGNATURE_FILE,
    carve_family_splits,
    render_signature,
    require_label_files,
    resolve_component_variant,
)
from missing_yolo.labels import LabelIndex, build_label_index
from missing_yolo.runner import anchor_config_paths
from missing_yolo.validation_metrics import (
    choose_balanced_stems,
    evaluate_split,
    read_prediction_labels,
    threshold_key,
)
from YOLO.dataset import (
    SampleFolder,
    list_sample_folders,
    write_data_yaml,
    write_render_signature,
)
from YOLO.overlay import draw_boxes_on_image, make_montage
from YOLO.runner import resolve_train_device

# Per-image CSV columns. Long format: one row per (stem, confidence threshold),
# so the same file answers both "what happened to this image?" and "how did the
# threshold change the answer?" without a second spreadsheet.
_PER_IMAGE_FIELDS: tuple[str, ...] = (
    "stem",
    "conf_threshold",
    "is_positive",
    "gt_boxes",
    "pred_boxes",
    "tp",
    "fp",
    "fn",
    "best_iou",
    "mean_matched_iou",
    "best_center_distance_px",
    "max_confidence",
    "status",
)
# Sample manifest columns. Paths are relative to the component output folder so
# the manifest stays valid if the whole report tree is moved.
_SAMPLE_FIELDS: tuple[str, ...] = (
    "stem",
    "is_positive",
    "gt_boxes",
    "pred_boxes",
    "tp",
    "fp",
    "fn",
    "best_iou",
    "mean_matched_iou",
    "best_center_distance_px",
    "max_confidence",
    "status",
    "reconstruction",
    "annotated",
)


@dataclass(frozen=True)
class ValidationPaths:
    """Filesystem layout of one component's validation report.

    Attributes:
        root (Path): Component folder, e.g. ``validation_outputs/tv``.
        dataset_root (Path): Regenerated Ultralytics dataset root.
        image_dir (Path): Rendered val images.
        label_dir (Path): Rendered val labels.
        data_yaml (Path): Ultralytics dataset description.
        manifest (Path): Render provenance CSV.
        ultralytics_dir (Path): Parent of the Ultralytics val run.
        samples_dir (Path): Raw and annotated sample images.
        per_image_csv (Path): Long-format per-image metrics.
        metrics_json (Path): Machine-readable component report.
        metrics_txt (Path): Human-readable component report.
    """

    root: Path
    dataset_root: Path
    image_dir: Path
    label_dir: Path
    data_yaml: Path
    manifest: Path
    ultralytics_dir: Path
    samples_dir: Path
    per_image_csv: Path
    metrics_json: Path
    metrics_txt: Path


def build_paths(root: Path) -> ValidationPaths:
    """Build the fixed report layout under one component folder.

    Args:
        root (Path): Component output folder.

    Returns:
        ValidationPaths: Every path the run writes.
    """
    dataset_root: Path = root / "dataset"
    return ValidationPaths(
        root=root,
        dataset_root=dataset_root,
        image_dir=dataset_root / "images" / "val",
        label_dir=dataset_root / "labels" / "val",
        data_yaml=dataset_root / "data.yaml",
        manifest=dataset_root / "redraw_manifest.csv",
        ultralytics_dir=root / "ultralytics",
        samples_dir=root / "samples",
        per_image_csv=root / "per_image.csv",
        metrics_json=root / "metrics.json",
        metrics_txt=root / "metrics.txt",
    )


def _jsonable(value: Any) -> Any:
    """Convert numpy/scalar/path values into strict JSON-compatible values.

    Args:
        value (Any): Value from an Ultralytics metrics object or a report dict.

    Returns:
        Any: A JSON-serializable equivalent. Non-finite floats become None so a
            report never emits NaN/Infinity.
    """
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write one pretty-printed, sorted JSON document.

    Args:
        path (Path): Destination path.
        payload (Any): Python object to serialize.
    """
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _format_optional(value: Any, digits: int = 4) -> str:
    """Format an optional number for a report line.

    Args:
        value (Any): Number, or None/empty when the metric does not apply.
        digits (int): Decimal places.

    Returns:
        str: Formatted number or ``"n/a"``.
    """
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_statistics(statistics: dict[str, float] | None) -> str:
    """Format one distribution as mean/median/p10..p90.

    Args:
        statistics (dict[str, float] | None): Output of ``summarize_values``.

    Returns:
        str: One-line summary, or ``"n/a"`` when empty.
    """
    if not statistics:
        return "n/a"
    return (
        f"mean {_format_optional(statistics.get('mean'))} | "
        f"median {_format_optional(statistics.get('median'))} | "
        f"p10 {_format_optional(statistics.get('p10'))}.."
        f"p90 {_format_optional(statistics.get('p90'))}"
    )


def _csv_value(value: Any) -> str:
    """Format one value for the per-image CSV without writing ``None``.

    Args:
        value (Any): Cell value.

    Returns:
        str: Empty for None, ``true``/``false`` for booleans, six decimals for
            floats, plain text otherwise.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _safe_index(values: Any, index: int) -> float | None:
    """Read one class entry from a possibly-empty Ultralytics metric array.

    Args:
        values (Any): Sequence or numpy array.
        index (int): Class index.

    Returns:
        float | None: The value, or None when the index is unavailable.
    """
    try:
        return float(values[index])
    except (IndexError, TypeError, KeyError, ValueError):
        return None


def collect_ultralytics_metrics(
    metrics: object, class_names: Sequence[str]
) -> dict[str, Any]:
    """Extract every reportable number from an Ultralytics validation result.

    Args:
        metrics (object): Ultralytics ``DetMetrics`` returned by ``model.val``.
        class_names (Sequence[str]): Ordered class vocabulary of the detector.

    Returns:
        dict[str, Any]: Mean metrics, per-class metrics, speed, results dict,
            save directory and confusion matrix. Missing fields are None.
    """
    box: Any = getattr(metrics, "box", None)
    names: dict[int, str] = {
        int(index): str(name)
        for index, name in getattr(metrics, "names", {}).items()
    }
    per_class: dict[str, dict[str, float | None]] = {}
    for class_id in sorted(names):
        per_class[names[class_id]] = {
            "precision": _safe_index(getattr(box, "p", []), class_id),
            "recall": _safe_index(getattr(box, "r", []), class_id),
            "f1": _safe_index(getattr(box, "f1", []), class_id),
            "mAP50": _safe_index(getattr(box, "ap50", []), class_id),
            "mAP50_95": _safe_index(getattr(box, "ap", []), class_id),
        }

    matrix_value: Any = getattr(
        getattr(metrics, "confusion_matrix", None), "matrix", None
    )
    matrix: list[list[int]] | None = None
    confusion_counts: dict[str, int] | None = None
    if matrix_value is not None:
        matrix = [[int(value) for value in row] for row in matrix_value.tolist()]
        if len(matrix) == 2 and len(matrix[0]) == 2:
            # Ultralytics stores rows=true class, columns=predicted class, with
            # the last row/column representing background.
            confusion_counts = {
                "tp": matrix[0][0],
                "fn": matrix[0][1],
                "fp": matrix[1][0],
                "tn": matrix[1][1],
            }

    speed: dict[str, float] = {
        str(key): float(value)
        for key, value in getattr(metrics, "speed", {}).items()
    }
    fitness: float | None = None
    if box is not None and hasattr(box, "fitness"):
        try:
            fitness = float(box.fitness())
        except (TypeError, ValueError):
            fitness = None

    return {
        "mean": {
            "precision": float(getattr(box, "mp", 0.0)) if box is not None else None,
            "recall": float(getattr(box, "mr", 0.0)) if box is not None else None,
            "f1": (
                None
                if box is None
                else _f1_ratio(
                    float(getattr(box, "mp", 0.0)),
                    float(getattr(box, "mr", 0.0)),
                )
            ),
            "mAP50": float(getattr(box, "map50", 0.0)) if box is not None else None,
            "mAP50_95": float(getattr(box, "map", 0.0)) if box is not None else None,
            "mAP75": float(getattr(box, "map75", 0.0)) if box is not None else None,
            "fitness": fitness,
        },
        "per_class": per_class,
        "class_names": list(class_names),
        "speed_ms_per_image": speed,
        "results_dict": {
            str(key): float(value)
            for key, value in getattr(metrics, "results_dict", {}).items()
            if isinstance(value, (int, float))
        },
        "save_dir": (
            str(getattr(metrics, "save_dir"))
            if getattr(metrics, "save_dir", None) is not None
            else None
        ),
        "confusion_matrix": {
            "labels": [*class_names, "background"],
            "matrix": matrix,
            "counts": confusion_counts,
        },
    }


def _f1_ratio(precision: float, recall: float) -> float | None:
    """Return F1 from precision and recall.

    Args:
        precision (float): Precision in ``0..1``.
        recall (float): Recall in ``0..1``.

    Returns:
        float | None: F1, or None when both are zero.
    """
    denominator: float = precision + recall
    return None if denominator == 0.0 else 2.0 * precision * recall / denominator


class MissingYoloValidator:
    """Regenerate one component's val split and report its detector.

    Attributes:
        config (SimpleNamespace): Runtime config.
        hyperparams (SimpleNamespace): Validation and rendering knobs.
        log (SimpleNamespace): Project logger.
        component (str): Component key being validated.
        variant (ComponentVariant): Generator/label row from missing_yolo config.
        shared_variant (SharedComponentVariant): Detector/generator row from components.toml.
        weights (Path): Detector weights this run scores.
        output_root (Path): Parent of all component report folders.
        component_root (Path): This component's report folder.
        paths (ValidationPaths): Fixed layout inside ``component_root``.
        cap (int): Val sample cap; negative keeps every sample.
        device_str (str): Ultralytics device string.
    """

    def __init__(
        self,
        config: SimpleNamespace,
        hyperparams: SimpleNamespace,
        log: SimpleNamespace,
        component: str,
        cap: int | None,
        output_root: Path,
        device: str | None,
    ) -> None:
        """Resolve the component, its two model rows and this run's paths.

        Args:
            config (SimpleNamespace): Runtime config; ``DEVICE`` may be
                overridden by ``device``.
            hyperparams (SimpleNamespace): Validation hyperparameters.
            log (SimpleNamespace): Configured project logger.
            component (str): Component key from ``components.toml``.
            cap (int | None): Val sample cap, or None to use
                ``VAL_SAMPLE_CAP``.
            output_root (Path): Parent folder for all component reports.
            device (str | None): Optional device override.

        Raises:
            FileNotFoundError: If the detector weights are absent.
            ValueError: If the shared registry and missing_yolo config disagree
                about the generator for this component, or the cap is invalid.
        """
        self.config = config
        self.hyperparams = hyperparams
        self.log = log
        self.component = component
        self.output_root = output_root
        self.component_root = output_root / component
        self.paths = build_paths(self.component_root)
        self.cap = int(cap) if cap is not None else int(hyperparams.VAL_SAMPLE_CAP)
        if self.cap < -1:
            raise ValueError(
                f"Validation cap must be -1 (all samples) or non-negative, got "
                f"{self.cap}."
            )
        if device is not None:
            config.DEVICE = str(device)

        self.variant: ComponentVariant = resolve_component_variant(config, component)
        registry: dict[str, SharedComponentVariant] = load_component_variants()
        self.shared_variant: SharedComponentVariant = resolve_shared_variant(
            registry, component
        )
        self.weights: Path = self.shared_variant.detector_weights
        self._validate_pairing()
        self.device_str: str = resolve_train_device(
            str(config.DEVICE), self.log
        )

    def _validate_pairing(self) -> None:
        """Fail fast if the two component registries disagree.

        ``missing_yolo/config.toml`` owns the generator row used for rendering;
        ``components.toml`` owns the detector weights. If their generator paths
        differ, the report would score a detector against a different
        generator's pixels, so the run refuses to start.

        Raises:
            FileNotFoundError: If the detector weights file is absent.
            ValueError: If the generators differ.
        """
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"Detector weights for component '{self.component}' are missing: "
                f"{self.weights}. Fix that row of components.toml."
            )
        registry_generator: Path = self.shared_variant.mask_diffusion_checkpoint.resolve()
        config_generator: Path = self.variant.mask_diffusion_checkpoint.resolve()
        if registry_generator != config_generator:
            raise ValueError(
                f"Component '{self.component}' points at two different "
                f"generators: components.toml says {registry_generator}, "
                f"missing_yolo/config.toml says {config_generator}. Fix one "
                "row before validating."
            )

    def run(self) -> dict[str, Any]:
        """Run the whole regeneration, validation and reporting pass.

        Returns:
            dict[str, Any]: The component report written to ``metrics.json``,
                plus the run's elapsed time.

        Raises:
            ImportError: If Ultralytics is not installed.
            FileNotFoundError: If the val split or its labels are incomplete.
            ValueError: If the detector vocabulary does not match the selected
                component.
        """
        started: float = time.time()
        self._prepare_outputs()
        self.log.info(
            f"Validating component '{self.component}' "
            f"(class '{self.variant.detector_class_name}') with weights "
            f"{self.weights}; output {self.component_root}."
        )

        stack: FrozenMaskDiffusionStack = FrozenMaskDiffusionStack(
            self.config, self.hyperparams, self.variant, self.log
        )
        val_sources, test_samples = self._index_val_sources(stack)
        label_index: LabelIndex = self._label_index(val_sources)

        condition_started: float = time.time()
        conditions = stack.encode_conditions(val_sources)
        condition_seconds: float = time.time() - condition_started
        generator: ArrangementGenerator = ArrangementGenerator(
            stack, label_index, self.hyperparams, self.log
        )
        # generate_split (not render_split_once) is deliberate: the report must
        # always regenerate its pixels, even when a previous run left a current-
        # looking render behind.
        stats: GenerationStats = generator.generate_split(
            sources=val_sources,
            cache=conditions,
            seed=int(self.hyperparams.VAL_RENDER_SEED),
            split="val",
            epoch=0,
            image_dir=self.paths.image_dir,
            label_dir=self.paths.label_dir,
            manifest_path=self.paths.manifest,
        )
        signature: str = render_signature(
            val_sources,
            self.variant,
            stack.overlay_alpha,
            stack.image_size,
            int(self.hyperparams.VAL_RENDER_SEED),
        )
        write_render_signature(
            self.paths.image_dir / RENDER_SIGNATURE_FILE, signature
        )
        data_yaml: str = write_data_yaml(
            str(self.paths.dataset_root),
            str(self.paths.data_yaml),
            [self.variant.detector_class_name],
        )
        self.log.info(
            f"Regenerated {stats.images} val images in {stats.seconds:.1f}s "
            f"(conditions {condition_seconds:.1f}s, seed "
            f"{self.hyperparams.VAL_RENDER_SEED}); data.yaml {data_yaml}."
        )

        model: Any = self._load_detector()
        ultralytics_metrics: Any = self._run_ultralytics_validation(
            model, data_yaml
        )
        official: dict[str, Any] = collect_ultralytics_metrics(
            ultralytics_metrics, [self.variant.detector_class_name]
        )
        prediction_dir: Path = (
            Path(str(official["save_dir"])) / "labels"
            if official.get("save_dir")
            else self.paths.ultralytics_dir / "val" / "labels"
        )
        if not prediction_dir.is_dir():
            self.log.warning(
                f"Ultralytics wrote no prediction labels directory at "
                f"{prediction_dir}; every image will be treated as having no "
                "predictions. This is expected only if the detector emitted no "
                "box above VALIDATION_VAL_CONF."
            )

        thresholds, primary_threshold = self._thresholds()
        threshold_summaries, per_image_rows = evaluate_split(
            stems=[source.stem for source in val_sources],
            ground_truth_dir=self.paths.label_dir,
            prediction_dir=prediction_dir,
            conf_thresholds=thresholds,
            iou_threshold=float(self.hyperparams.VALIDATION_MATCH_IOU),
            image_size=int(self.hyperparams.RENDER_IMAGE_SIZE),
        )
        self._write_per_image(per_image_rows)
        sample_info: dict[str, Any] = self._write_samples(
            val_sources=val_sources,
            label_index=label_index,
            per_image_rows=per_image_rows,
            primary_threshold=primary_threshold,
            prediction_dir=prediction_dir,
        )

        summary: dict[str, Any] = {
            "component": self.component,
            "class_name": self.variant.detector_class_name,
            "split": "val",
            "source_split_directory": str(self.config.SOURCE_VAL_SPLIT_DIR),
            "generator_checkpoint": str(self.variant.mask_diffusion_checkpoint),
            "detector_weights": str(self.weights),
            "output_directory": str(self.component_root),
            "dataset": {
                "samples": len(val_sources),
                "positive_samples": label_index.positive_samples,
                "background_samples": label_index.background_samples,
                "ground_truth_boxes": label_index.total_boxes,
                "boxes_per_positive": (
                    label_index.total_boxes / label_index.positive_samples
                    if label_index.positive_samples
                    else None
                ),
                "val_family_fraction": float(
                    self.hyperparams.VAL_FAMILY_FRACTION
                ),
                "base_seed": int(self.hyperparams.BASE_SEED),
                "render_seed": int(self.hyperparams.VAL_RENDER_SEED),
                "condition_seconds": condition_seconds,
                "render_seconds": stats.seconds,
                "cap": self.cap,
                "test_samples_not_rendered": test_samples,
                "data_yaml": str(self.paths.data_yaml),
                "manifest": str(self.paths.manifest),
            },
            "generation_agreement": {
                "scored_samples": stats.agreement_scored,
                "mean_box_iou": stats.agreement_mean,
                "fraction_above_iou": stats.agreement_fractions,
            },
            "ultralytics": official,
            "thresholded": threshold_summaries,
            "primary_threshold": primary_threshold,
            "iou_threshold": float(self.hyperparams.VALIDATION_MATCH_IOU),
            "per_image_csv": str(self.paths.per_image_csv),
            "samples": sample_info,
            "elapsed_seconds": time.time() - started,
        }
        write_json(self.paths.metrics_json, summary)
        self.paths.metrics_txt.write_text(
            "\n".join(self.format_component_report(summary)) + "\n",
            encoding="utf-8",
        )
        self.log.info(
            f"Component '{self.component}' report written: "
            f"{self.paths.metrics_json} and {self.paths.metrics_txt}."
        )
        return summary

    def _prepare_outputs(self) -> None:
        """Create this component's output tree, removing a previous report.

        The validation run is defined as "always regenerate", so a previous
        component folder is replaced rather than merged. Removing it also
        prevents stale prediction files from an older weights version from
        leaking into the new metrics.
        """
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.component_root.exists():
            self.log.info(
                f"Removing previous validation report at {self.component_root}."
            )
            shutil.rmtree(self.component_root)
        self.paths.dataset_root.mkdir(parents=True, exist_ok=True)
        self.paths.image_dir.mkdir(parents=True, exist_ok=True)
        self.paths.label_dir.mkdir(parents=True, exist_ok=True)
        self.paths.ultralytics_dir.mkdir(parents=True, exist_ok=True)
        self.paths.samples_dir.mkdir(parents=True, exist_ok=True)

    def _index_val_sources(
        self, stack: FrozenMaskDiffusionStack
    ) -> tuple[list[SampleFolder], int]:
        """Index the validate pool and carve the same val half the trainer used.

        Args:
            stack (FrozenMaskDiffusionStack): Provides the required mask
                filenames.

        Returns:
            tuple[list[SampleFolder], int]: The capped val sources and the
                number of test samples that were carved but deliberately not
                rendered.

        Raises:
            FileNotFoundError: If the source split or any labels.txt is absent.
            ValueError: If the cap removes every sample or val/test overlap.
        """
        pooled: list[SampleFolder] = list_sample_folders(
            str(self.config.SOURCE_VAL_SPLIT_DIR),
            stack.floorplan_group.mask_file_name,
            stack.target_group.mask_file_name,
            -1,
        )
        require_label_files(pooled, str(self.config.GROUND_TRUTH_LABEL_FILE))
        val_sources, test_sources = carve_family_splits(
            pooled,
            float(self.hyperparams.VAL_FAMILY_FRACTION),
            int(self.hyperparams.BASE_SEED),
        )
        if self.cap >= 0:
            val_sources = val_sources[: self.cap]
        if not val_sources:
            raise ValueError(
                f"Validation cap {self.cap} left no val samples. Use -1 for the "
                "entire split."
            )
        overlap: set[str] = {source.stem for source in val_sources} & {
            source.stem for source in test_sources
        }
        if overlap:
            raise ValueError(
                f"val and test share {len(overlap)} samples (e.g. "
                f"{sorted(overlap)[:3]}); the split is not family-disjoint."
            )
        self.log.info(
            f"Carved {len(pooled)} validate samples into {len(val_sources)} val "
            f"(rendered) and {len(test_sources)} test (not rendered, "
            "generator-unseen but reserved for the training report)."
        )
        return val_sources, len(test_sources)

    def _label_index(self, sources: list[SampleFolder]) -> LabelIndex:
        """Build the component's labels.txt index for the val split.

        Args:
            sources (list[SampleFolder]): Val sources in render order.

        Returns:
            LabelIndex: Targets per stem with the single detector class id.
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
            f"Val targets: {index.positive_samples} positive "
            f"({index.background_samples} background), "
            f"{index.total_boxes} boxes of class "
            f"'{self.variant.detector_class_name}'."
        )
        return index

    def _load_detector(self) -> Any:
        """Load the selected detector and verify its class vocabulary.

        Returns:
            Any: Ultralytics ``YOLO`` wrapper for the component's weights.

        Raises:
            ImportError: If ultralytics is not installed.
            ValueError: If the weights are not the single class this component
                expects, which would attribute every metric to the wrong name.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run "
                "`./venv/bin/pip install ultralytics` (see "
                "missing_yolo/README.md)."
            ) from exc
        model: Any = YOLO(str(self.weights))
        names: list[str] = detector_class_names(model)
        expected: list[str] = [self.variant.detector_class_name]
        if names != expected:
            raise ValueError(
                f"Detector weights {self.weights} declare classes {names}, but "
                f"component '{self.component}' expects {expected}. The "
                "components.toml row and the weights do not belong together."
            )
        return model

    def _run_ultralytics_validation(
        self, model: Any, data_yaml: str
    ) -> Any:
        """Run one Ultralytics validation pass over the regenerated val split.

        Args:
            model (Any): Loaded detector.
            data_yaml (str): Dataset description written by this run.

        Returns:
            Any: Ultralytics validation metrics object.
        """
        self.log.info(
            f"Running Ultralytics val on {self.component} "
            f"(batch {self.hyperparams.VALIDATION_BATCH_SIZE}, "
            f"imgsz {self.hyperparams.IMG_SIZE}, device {self.device_str})..."
        )
        return model.val(
            data=data_yaml,
            split="val",
            device=self.device_str,
            imgsz=int(self.hyperparams.IMG_SIZE),
            batch=int(self.hyperparams.VALIDATION_BATCH_SIZE),
            classes=[0],
            conf=float(self.hyperparams.VALIDATION_VAL_CONF),
            iou=float(self.hyperparams.VALIDATION_NMS_IOU),
            max_det=int(self.hyperparams.VALIDATION_MAX_DETECTIONS),
            save_txt=True,
            save_conf=True,
            plots=True,
            project=str(self.paths.ultralytics_dir),
            name="val",
            exist_ok=True,
            verbose=False,
        )

    def _thresholds(self) -> tuple[list[float], float]:
        """Return the sorted confidence thresholds and the primary one.

        The primary threshold is always included, so the sample panel and the
        per-image CSV have a row to read even if the configured list omits it.

        Returns:
            tuple[list[float], float]: ``(thresholds, primary_threshold)``.

        Raises:
            ValueError: If the primary threshold is outside ``0..1``.
        """
        thresholds: list[float] = [
            float(value)
            for value in self.hyperparams.VALIDATION_CONF_THRESHOLDS
        ]
        primary: float = float(self.hyperparams.VALIDATION_REPORT_CONF)
        if not 0.0 <= primary <= 1.0:
            raise ValueError(
                f"VALIDATION_REPORT_CONF must lie in 0..1, got {primary}."
            )
        thresholds.append(primary)
        return sorted({value for value in thresholds}), primary

    def _write_per_image(self, rows: Sequence[dict[str, object]]) -> None:
        """Write the long-format per-image metrics CSV.

        Args:
            rows (Sequence[dict[str, object]]): Rows from
                :func:`missing_yolo.validation_metrics.evaluate_split`.
        """
        with self.paths.per_image_csv.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=_PER_IMAGE_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: _csv_value(row.get(field)) for field in _PER_IMAGE_FIELDS}
                )

    def _write_samples(
        self,
        val_sources: Sequence[SampleFolder],
        label_index: LabelIndex,
        per_image_rows: Sequence[dict[str, object]],
        primary_threshold: float,
        prediction_dir: Path,
    ) -> dict[str, Any]:
        """Write the balanced raw/annotated sample panel and its manifest.

        Args:
            val_sources (Sequence[SampleFolder]): Val samples in render order.
            label_index (LabelIndex): Positivity of every stem.
            per_image_rows (Sequence[dict[str, object]]): Per-image metrics at
                every threshold.
            primary_threshold (float): Threshold the annotated panel uses.
            prediction_dir (Path): Ultralytics saved prediction labels.

        Returns:
            dict[str, Any]: Counts and paths for the component summary.
        """
        positive_stems: list[str] = [
            stem
            for stem, labels in label_index.by_stem.items()
            if labels.positive
        ]
        negative_stems: list[str] = [
            stem
            for stem, labels in label_index.by_stem.items()
            if not labels.positive
        ]
        selected: list[tuple[str, bool]] = choose_balanced_stems(
            positive_stems,
            negative_stems,
            int(self.hyperparams.VALIDATION_SAMPLE_POSITIVES),
            int(self.hyperparams.VALIDATION_SAMPLE_NEGATIVES),
            int(self.hyperparams.VALIDATION_SAMPLE_SEED),
        )
        rows_by_stem: dict[str, dict[str, object]] = {
            str(row["stem"]): row
            for row in per_image_rows
            if float(row["conf_threshold"]) == float(primary_threshold)
        }
        manifest_rows: list[dict[str, object]] = []
        raw_paths: list[Path] = []
        annotated_paths: list[Path] = []
        for stem, is_positive in selected:
            image_path: Path = self.paths.image_dir / f"{stem}.png"
            label_path: Path = self.paths.label_dir / f"{stem}.txt"
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Rendered validation image is missing: {image_path}."
                )
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Rendered validation label is missing: {label_path}."
                )
            split_dir: Path = self.paths.samples_dir / (
                "positive" if is_positive else "negative"
            )
            split_dir.mkdir(parents=True, exist_ok=True)
            raw_path: Path = split_dir / f"{stem}_reconstruction.png"
            annotated_path: Path = split_dir / f"{stem}_annotated.png"
            shutil.copy2(image_path, raw_path)

            image: Image.Image = Image.open(image_path).convert("RGB")
            predictions = read_prediction_labels(
                prediction_dir / f"{stem}.txt", class_id=0
            )
            kept = [
                prediction
                for prediction in predictions
                if prediction.confidence >= float(primary_threshold)
            ]
            kept.sort(key=lambda prediction: prediction.confidence, reverse=True)
            kept = kept[: int(self.hyperparams.VALIDATION_MAX_BOXES)]
            pixel_predictions: list[tuple[float, float, float, float, int, float]] = [
                (
                    *prediction.box.to_pixels(image.width),
                    0,
                    prediction.confidence,
                )
                for prediction in kept
            ]
            gt_lines: list[str] = label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            annotated: Image.Image = draw_boxes_on_image(
                image,
                gt_lines,
                pixel_predictions,
                [self.variant.detector_class_name],
            )
            annotated.save(
                str(annotated_path),
                compress_level=int(self.hyperparams.RENDER_PNG_COMPRESS_LEVEL),
            )
            raw_paths.append(raw_path)
            annotated_paths.append(annotated_path)

            outcome: dict[str, object] = rows_by_stem.get(
                stem,
                {
                    "stem": stem,
                    "is_positive": is_positive,
                    "gt_boxes": 0,
                    "pred_boxes": len(kept),
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "best_iou": None,
                    "mean_matched_iou": None,
                    "best_center_distance_px": None,
                    "max_confidence": (
                        kept[0].confidence if kept else None
                    ),
                    "status": "unknown",
                },
            )
            manifest_rows.append(
                {
                    **{field: outcome.get(field) for field in _SAMPLE_FIELDS},
                    "stem": stem,
                    "is_positive": is_positive,
                    "reconstruction": str(
                        raw_path.relative_to(self.component_root)
                    ),
                    "annotated": str(
                        annotated_path.relative_to(self.component_root)
                    ),
                }
            )

        manifest_path: Path = self.paths.samples_dir / "sample_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_SAMPLE_FIELDS)
            writer.writeheader()
            for row in manifest_rows:
                writer.writerow(
                    {field: _csv_value(row.get(field)) for field in _SAMPLE_FIELDS}
                )

        montage_raw: Path = self.paths.samples_dir / "montage_reconstructions.png"
        montage_annotated: Path = self.paths.samples_dir / "montage_annotated.png"
        if raw_paths:
            make_montage(
                raw_paths,
                montage_raw,
                int(self.hyperparams.VALIDATION_MONTAGE_COLS),
            )
        if annotated_paths:
            make_montage(
                annotated_paths,
                montage_annotated,
                int(self.hyperparams.VALIDATION_MONTAGE_COLS),
            )
        self.log.info(
            f"Saved {len(selected)} sample images "
            f"({len(positive_stems)} positives / {len(negative_stems)} "
            f"backgrounds available) to {self.paths.samples_dir}."
        )
        return {
            "written": len(selected),
            "requested_positives": int(
                self.hyperparams.VALIDATION_SAMPLE_POSITIVES
            ),
            "requested_negatives": int(
                self.hyperparams.VALIDATION_SAMPLE_NEGATIVES
            ),
            "available_positives": len(positive_stems),
            "available_negatives": len(negative_stems),
            "manifest": str(manifest_path),
            "montage_reconstructions": (
                str(montage_raw) if raw_paths else None
            ),
            "montage_annotated": (
                str(montage_annotated) if annotated_paths else None
            ),
        }

    def format_component_report(self, summary: dict[str, Any]) -> list[str]:
        """Build the human-readable component report.

        Args:
            summary (dict[str, Any]): The component summary written as JSON.

        Returns:
            list[str]: Report lines, ready to join with newlines.
        """
        dataset: dict[str, Any] = summary["dataset"]
        agreement: dict[str, Any] = summary["generation_agreement"]
        official: dict[str, Any] = summary["ultralytics"]
        mean: dict[str, Any] = official["mean"]
        primary: str = threshold_key(float(summary["primary_threshold"]))
        primary_stats: dict[str, Any] = summary["thresholded"][primary]
        lines: list[str] = [
            "",
            "=" * 78,
            "        MISSING-COMPONENT YOLO VALIDATION REPORT",
            "=" * 78,
            f"Component          : {summary['component']}",
            f"Detector class     : {summary['class_name']}",
            f"Generator          : {summary['generator_checkpoint']}",
            f"Detector weights   : {summary['detector_weights']}",
            f"Split              : {summary['split']} (always regenerated)",
            f"Source split       : {summary['source_split_directory']}",
            f"Samples            : {dataset['samples']} "
            f"({dataset['positive_samples']} positive / "
            f"{dataset['background_samples']} background)",
            f"Ground-truth boxes : {dataset['ground_truth_boxes']} "
            f"(mean {_format_optional(dataset['boxes_per_positive'])} per positive)",
            f"Render             : seed {dataset['render_seed']}, "
            f"{dataset['render_seconds']:.1f}s for {dataset['samples']} images "
            f"(conditions {dataset['condition_seconds']:.1f}s)",
            f"Test half          : {dataset['test_samples_not_rendered']} samples "
            "carved but not rendered (reserved for the training report)",
            "-" * 78,
            "Generator/labels.txt agreement on the regenerated val split",
            "-" * 78,
            f"  scored samples   : {agreement['scored_samples']}",
            f"  mean box IoU     : {_format_optional(agreement['mean_box_iou'])}",
            "  fraction >= IoU  : "
            + (
                ", ".join(
                    f"{threshold}: {_format_optional(value)}"
                    for threshold, value in agreement[
                        "fraction_above_iou"
                    ].items()
                )
                or "n/a"
            ),
            "-" * 78,
            "Ultralytics validation (single class; official mAP/precision/recall)",
            "-" * 78,
            f"  precision        : {_format_optional(mean['precision'])}",
            f"  recall           : {_format_optional(mean['recall'])}",
            f"  F1               : {_format_optional(mean['f1'])}",
            f"  mAP@50           : {_format_optional(mean['mAP50'])}",
            f"  mAP@50-95        : {_format_optional(mean['mAP50_95'])}",
            f"  mAP@75           : {_format_optional(mean['mAP75'])}",
            f"  fitness          : {_format_optional(mean['fitness'])}",
            f"  speed (ms/img)   : "
            f"pre {_format_optional(official['speed_ms_per_image'].get('preprocess'))} | "
            f"infer {_format_optional(official['speed_ms_per_image'].get('inference'))} | "
            f"loss {_format_optional(official['speed_ms_per_image'].get('loss'))} | "
            f"post {_format_optional(official['speed_ms_per_image'].get('postprocess'))}",
        ]
        per_class: dict[str, dict[str, Any]] = official["per_class"]
        for class_name, class_metrics in per_class.items():
            lines.append(
                f"  class {class_name:<16}: "
                f"P {_format_optional(class_metrics['precision'])} | "
                f"R {_format_optional(class_metrics['recall'])} | "
                f"mAP50 {_format_optional(class_metrics['mAP50'])} | "
                f"mAP50-95 {_format_optional(class_metrics['mAP50_95'])}"
            )
        confusion: dict[str, Any] = official["confusion_matrix"]
        if confusion.get("matrix"):
            labels: list[str] = confusion["labels"]
            lines.append("  confusion matrix (rows=true, columns=predicted):")
            lines.append("    " + " " * 18 + "  ".join(f"{name:>10}" for name in labels))
            for label, row in zip(labels, confusion["matrix"], strict=True):
                lines.append(
                    f"    {label:<18}" + "  ".join(f"{int(value):>10}" for value in row)
                )
            counts = confusion.get("counts") or {}
            lines.append(
                "    counts: "
                f"TP {counts.get('tp', 'n/a')} | FP {counts.get('fp', 'n/a')} | "
                f"FN {counts.get('fn', 'n/a')} | TN {counts.get('tn', 'n/a')}"
            )
        lines += [
            "-" * 78,
            "Thresholded box and image metrics (greedy match at IoU "
            f"{_format_optional(summary['iou_threshold'], 2)})",
            "-" * 78,
            "  conf   TP    FP    FN    prec   rec    F1     boxJ   "
            "imgAcc detRate FA-rate spec   balAcc",
        ]
        for key in sorted(
            summary["thresholded"],
            key=lambda item: float(item),
        ):
            stats = summary["thresholded"][key]
            box_metrics = stats["box_metrics"]
            image_metrics = stats["image_metrics"]
            counts = stats["box_counts"]
            lines.append(
                f"  {key:<6} {counts['tp']:<5} {counts['fp']:<5} "
                f"{counts['fn']:<5} "
                f"{_format_optional(box_metrics['precision']):<6} "
                f"{_format_optional(box_metrics['recall']):<6} "
                f"{_format_optional(box_metrics['f1']):<6} "
                f"{_format_optional(box_metrics['jaccard']):<6} "
                f"{_format_optional(image_metrics['accuracy']):<6} "
                f"{_format_optional(image_metrics['positive_detection_rate']):<7} "
                f"{_format_optional(image_metrics['false_alarm_rate']):<7} "
                f"{_format_optional(image_metrics['specificity']):<6} "
                f"{_format_optional(image_metrics['balanced_accuracy'])}"
            )
        lines += [
            "-" * 78,
            f"Primary threshold {primary}: localisation of matched boxes",
            "-" * 78,
            "  IoU              : "
            + _format_statistics(
                primary_stats["localization"]["matched_iou"]
            ),
            "  centre dist (px) : "
            + _format_statistics(
                primary_stats["localization"]["center_distance_px"]
            ),
            "  L1 on (cx,cy,w,h): "
            + _format_statistics(
                primary_stats["localization"]["l1_norm_xywh"]
            ),
            f"  mean confidence  : TP "
            f"{_format_optional(primary_stats['confidence']['mean_true_positive'])} "
            f"| FP {_format_optional(primary_stats['confidence']['mean_false_positive'])}",
            f"  image outcomes   : "
            f"detected {primary_stats['image_counts']['detected_positive']} / "
            f"{primary_stats['image_counts']['positive']} positive | "
            f"missed {primary_stats['image_counts']['missed_positive']} | "
            f"false alarms {primary_stats['image_counts']['false_alarm']} / "
            f"{primary_stats['image_counts']['background']} background | "
            f"correctly empty {primary_stats['image_counts']['correct_empty']}",
            "-" * 78,
            "Artifacts",
            "-" * 78,
            f"  metrics JSON     : {summary['output_directory']}/metrics.json",
            f"  per-image CSV    : {summary['per_image_csv']}",
            f"  samples          : {summary['samples']['manifest']}",
            f"  Ultralytics      : {official.get('save_dir') or 'n/a'}",
            "=" * 78,
            "",
        ]
        return lines


def _component_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    """Reduce a component report to the fields the run summary compares.

    Args:
        summary (dict[str, Any]): Full component report.

    Returns:
        dict[str, Any]: Component name, split size, official metrics and the
            primary-threshold metrics.
    """
    primary_key: str = threshold_key(float(summary["primary_threshold"]))
    primary_stats: dict[str, Any] = summary["thresholded"][primary_key]
    return {
        "component": summary["component"],
        "class_name": summary["class_name"],
        "samples": summary["dataset"]["samples"],
        "positive_samples": summary["dataset"]["positive_samples"],
        "background_samples": summary["dataset"]["background_samples"],
        "ground_truth_boxes": summary["dataset"]["ground_truth_boxes"],
        "mAP50": summary["ultralytics"]["mean"]["mAP50"],
        "mAP50_95": summary["ultralytics"]["mean"]["mAP50_95"],
        "precision": summary["ultralytics"]["mean"]["precision"],
        "recall": summary["ultralytics"]["mean"]["recall"],
        "primary_threshold": summary["primary_threshold"],
        "thresholded_box_metrics": primary_stats["box_metrics"],
        "thresholded_image_metrics": primary_stats["image_metrics"],
        "primary_box_counts": primary_stats["box_counts"],
        "output_directory": summary["output_directory"],
    }


def _summary_lines(
    summaries: Sequence[dict[str, Any]], output_root: Path, elapsed: float
) -> list[str]:
    """Build the cross-component comparison table.

    Args:
        summaries (Sequence[dict[str, Any]]): Full component reports.
        output_root (Path): Parent folder of the three reports.
        elapsed (float): Whole-run wall clock in seconds.

    Returns:
        list[str]: Summary lines.
    """
    lines: list[str] = [
        "",
        "=" * 100,
        "        MISSING-COMPONENT YOLO VALIDATION SUMMARY",
        "=" * 100,
        f"Output root : {output_root}",
        f"Elapsed     : {elapsed:.1f}s",
        "-" * 100,
        "component    samples positives background boxes  mAP50  mAP50-95 "
        "prec   rec    conf  TP    FP    FN    F1     imgAcc",
    ]
    for summary in summaries:
        row: dict[str, Any] = _component_summary_row(summary)
        box_metrics: dict[str, Any] = row["thresholded_box_metrics"]
        image_metrics: dict[str, Any] = row["thresholded_image_metrics"]
        counts: dict[str, Any] = row["primary_box_counts"]
        lines.append(
            f"{row['component']:<12} {row['samples']:<7} "
            f"{row['positive_samples']:<9} {row['background_samples']:<10} "
            f"{row['ground_truth_boxes']:<6} "
            f"{_format_optional(row['mAP50']):<6} "
            f"{_format_optional(row['mAP50_95']):<9} "
            f"{_format_optional(row['precision']):<6} "
            f"{_format_optional(row['recall']):<6} "
            f"{row['primary_threshold']:<5} "
            f"{counts['tp']:<5} {counts['fp']:<5} {counts['fn']:<5} "
            f"{_format_optional(box_metrics['f1']):<6} "
            f"{_format_optional(image_metrics['accuracy'])}"
        )
    lines += [
        "-" * 100,
        "Metric notes",
        "-" * 100,
        "  mAP/precision/recall are Ultralytics' official values (low-confidence",
        "  val pass); thresholded TP/FP/FN use greedy conf-order matching at the",
        "  configured IoU. Image accuracy = (detected positive images + correctly",
        "  empty background images) / all images. Balanced accuracy averages the",
        "  positive detection rate and specificity. Undefined values are n/a.",
        "=" * 100,
        "",
    ]
    return lines


def write_run_summary(
    output_root: Path,
    summaries: Sequence[dict[str, Any]],
    elapsed: float,
) -> None:
    """Write the run-level JSON, text summary and output guide.

    Args:
        output_root (Path): Parent folder of all component reports.
        summaries (Sequence[dict[str, Any]]): Full component reports.
        elapsed (float): Whole-run wall clock in seconds.
    """
    rows: list[dict[str, Any]] = [
        _component_summary_row(summary) for summary in summaries
    ]
    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed,
        "output_root": str(output_root),
        "components": rows,
    }
    write_json(output_root / "summary.json", payload)
    (output_root / "summary.txt").write_text(
        "\n".join(_summary_lines(summaries, output_root, elapsed)) + "\n",
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        _output_readme(output_root, summaries), encoding="utf-8"
    )


def _output_readme(
    output_root: Path, summaries: Sequence[dict[str, Any]]
) -> str:
    """Return the static guide that explains the report tree and metrics.

    Args:
        output_root (Path): Parent folder of all component reports.
        summaries (Sequence[dict[str, Any]]): Full component reports.

    Returns:
        str: Markdown guide written as ``README.md``.
    """
    component_names: str = ", ".join(
        summary["component"] for summary in summaries
    )
    return f"""# missing_yolo validation reports

Generated by `missing_yolo/validate.py` (or `./validate_missing_yolo.sh`).
Components in this run: **{component_names}**.

Every component folder was produced end to end: the val half of
`SOURCE_VAL_SPLIT_DIR` was re-rendered from the component's frozen
`mask_diffusion` checkpoint, then the matching `missing_yolo` detector was run
over **all** regenerated val images. Nothing was trained and no image was
reused from a previous render.

## Folder layout

```
{output_root}/
├── summary.txt                 cross-component table
├── summary.json                machine-readable cross-component table
├── README.md                   this file
└── <component>/
    ├── metrics.txt             human-readable report
    ├── metrics.json            every number in the report
    ├── per_image.csv           one row per sample and confidence threshold
    ├── dataset/                regenerated val render + labels + data.yaml
    ├── ultralytics/val/        curves, confusion matrix, saved predictions
    └── samples/                balanced raw and GT-vs-prediction images
```

## Metric definitions

* **mAP@50 / mAP@50-95 / precision / recall** — Ultralytics' official validation
  metrics. They come from the same low-confidence val pass but are independent
  of the thresholded tables below.
* **Thresholded TP/FP/FN** — predictions are sorted by confidence and greedily
  matched to ground-truth boxes at `VALIDATION_MATCH_IOU` (default 0.5), at each
  configured confidence threshold.
* **precision / recall / F1** — standard box-level ratios at that threshold.
* **boxJ** — `TP / (TP + FP + FN)`, a box-level Jaccard score.
* **imgAcc** — image-level accuracy: `(positive images with at least one TP +
  background images with no prediction) / all images`.
* **detRate** — fraction of positive images with at least one TP.
* **FA-rate** — fraction of background images with at least one false positive.
* **spec** — specificity: fraction of background images correctly left empty.
* **balAcc** — mean of `detRate` and `spec`.
* **matched IoU / centre distance / L1** — localisation quality of matched boxes
  at the primary threshold. `n/a` means there were no matches (not zero).
* **agreement mean / fractions** — box-IoU between the component the generator
  actually drew and the dataset's `labels.txt` box, over positive samples. This
  measures label noise injected by keeping every generated draw.

## Why the val half only

`work/test` is intentionally not used: the frozen generator was trained on it,
so it cannot give an honest held-out number. The val half is generator-unseen.
The test half of `work/validate` is carved consistently but not rendered here;
the training report already owns that number.
"""


def main() -> None:
    """Parse CLI arguments and run validation for one or all components."""
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate each missing component's val split and report its "
            "missing_yolo detector on every image."
        )
    )
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    parser.add_argument(
        "--component",
        default="all",
        help="Component key from components.toml, or 'all' (default).",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help=(
            "Cap the val samples per component for a smoke run; -1 keeps the "
            "entire val split (the default from VAL_SAMPLE_CAP)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Override the validation output root (default: "
            "VALIDATION_OUTPUT_DIRECTORY from missing_yolo/config.toml)."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cuda", "cpu"],
        default=None,
        help="Override DEVICE from missing_yolo/config.toml.",
    )
    args: argparse.Namespace = parser.parse_args()

    config: SimpleNamespace = load_config(
        args.mode, str(_SCRIPT_DIR / "config.toml")
    )
    anchor_config_paths(config)
    hyperparams: SimpleNamespace = load_hyperparams(
        str(_SCRIPT_DIR / "hyperparameters.toml")
    )
    if args.device is not None:
        config.DEVICE = str(args.device)
    log: SimpleNamespace = setup_logger(config)

    registry: dict[str, SharedComponentVariant] = load_component_variants()
    components: list[str] = (
        list(registry) if str(args.component) == "all" else [str(args.component)]
    )
    # Fail on an unknown component before spending time rendering the first one.
    for component in components:
        resolve_shared_variant(registry, component)

    output_root: Path = (
        Path(str(args.output)).expanduser().resolve()
        if args.output
        else Path(str(config.VALIDATION_OUTPUT_DIRECTORY))
    )
    output_root.mkdir(parents=True, exist_ok=True)
    log.info(
        f"Starting missing_yolo validation for {components} in "
        f"{args.mode} mode on {config.DEVICE}; output {output_root}."
    )

    run_started: float = time.time()
    summaries: list[dict[str, Any]] = []
    for component in components:
        validator: MissingYoloValidator = MissingYoloValidator(
            config=config,
            hyperparams=hyperparams,
            log=log,
            component=component,
            cap=args.cap,
            output_root=output_root,
            device=args.device,
        )
        summaries.append(validator.run())

    elapsed: float = time.time() - run_started
    write_run_summary(output_root, summaries, elapsed)
    log.info(
        f"Validation complete for {components} in {elapsed:.1f}s. "
        f"Reports: {output_root}/summary.txt and one metrics.txt per "
        "component folder."
    )


if __name__ == "__main__":
    main()
