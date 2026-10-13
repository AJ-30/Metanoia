"""Validation module evaluating YOLO on diffusion-drawn test floorplans.

This script benchmarks a trained YOLO detector on the testing split
conditioned through the frozen conditional diffusion model (the same pipeline
used during training). It optionally compares the performance side-by-side
against the deterministic real ground-truth test split.

Pipeline:
1. Index test sample folders from ``LIVE_SOURCE_TEST_SPLIT_DIR``.
2. Load the frozen rendering stack (condition VAE, target VAE, conditional
   diffusion model).
3. Encode the test floorplan conditions once.
4. Run the conditional diffusion model to sample furniture arrangements on
   the test floorplans.
5. Decode sampled latents and blend furniture onto the floorplans.
6. Extract bounding boxes using connected components on the diffusion output
   (identical to the live training pipeline).
7. Run Ultralytics validation (``model.val``) against the diffusion test set.
8. Run Ultralytics validation (``model.val``) against the real ground-truth
   test set (if available) for direct benchmarking.
9. Generate a side-by-side comparison table and save overlay montages.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

# Ensure PyTorch falls back to CPU for ops unsupported on MPS (e.g. torchvision::nms)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Project root setup so ``helpers/`` and ``YOLO.*`` import reliably.
SCRIPT_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from helpers.config import load_config
from helpers.hyperparams import load_hyperparams
from helpers.logger import setup_logger
from YOLO.dataset import (
    DeterministicSplitRenderer,
    FrozenRenderStack,
    LiveEpochRedrawer,
    RedrawStats,
    SampleFolder,
    clear_generated_split,
    list_sample_folders,
    write_data_yaml,
)
from YOLO.overlay import (
    choose_fixed_stems,
    draw_boxes_on_image,
    make_montage,
    predict_boxes,
)
from YOLO.predict import labels_dir_for, read_gt_lines, resolve_weights_path
from YOLO.runner import anchor_config_paths, resolve_train_device


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the diffusion test validation script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO on test floorplans using diffusion model outputs."
    )
    parser.add_argument(
        "--mode",
        choices=["debug", "prod"],
        default="debug",
        help="Runtime configuration mode (debug or prod).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to YOLO weights (.pt). Defaults to newest yolo_*_best.pt in MODEL_SAVE_PATH.",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=None,
        help="Sample cap for the test set. Overrides TEST_SAMPLE_CAP from hyperparameters.toml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for diffusion draw. Overrides BASE_SEED from hyperparameters.toml.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--save-montage",
        action="store_true",
        default=True,
        help="Generate and save visual prediction montages on sample test images.",
    )
    parser.add_argument(
        "--montage-samples",
        type=int,
        default=16,
        help="Number of test images to include in visual contact sheet montages.",
    )
    parser.add_argument(
        "--skip-gt",
        action="store_true",
        default=False,
        help="Skip evaluating on the real ground-truth test set (only evaluate on diffusion).",
    )
    return parser.parse_args()


def prepare_eval_directories(eval_root: Path) -> tuple[Path, Path, Path]:
    """Create and clean directories for the diffusion test dataset.

    Args:
        eval_root (Path): Root folder where diffusion test dataset is saved.

    Returns:
        tuple[Path, Path, Path]: Paths to images directory, labels directory,
            and data.yaml file.
    """
    img_dir: Path = eval_root / "images" / "test"
    lbl_dir: Path = eval_root / "labels" / "test"
    yaml_path: Path = eval_root / "data.yaml"

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing generated artifacts in this evaluation folder to prevent leakage
    clear_generated_split(img_dir, lbl_dir)
    return img_dir, lbl_dir, yaml_path


def evaluate_split(
    model: Any,
    yaml_path: str,
    device_str: str,
    img_size: int,
    batch_size: int,
    log: SimpleNamespace,
    label: str,
) -> dict[str, float]:
    """Run Ultralytics validation on a specified dataset split and return metrics.

    Args:
        model (Any): Loaded Ultralytics YOLO model instance.
        yaml_path (str): Path to data.yaml dataset descriptor.
        device_str (str): Target device string ('mps', 'cuda', or 'cpu').
        img_size (int): Image size for validation inference.
        batch_size (int): Batch size for validation loader.
        log (SimpleNamespace): Logger instance for status reporting.
        label (str): Human-readable label for logging context.

    Returns:
        dict[str, float]: Dictionary containing headline metrics:
            mAP50, mAP50_95, precision, and recall.
    """
    log.info(f"Running YOLO validation on {label}...")
    metrics: Any = model.val(
        data=yaml_path,
        split="test",
        device=device_str,
        imgsz=img_size,
        batch=batch_size,
        verbose=False,
    )

    result_dict: dict[str, float] = {
        "mAP50": float(getattr(metrics.box, "map50", 0.0)),
        "mAP50_95": float(getattr(metrics.box, "map", 0.0)),
        "precision": float(getattr(metrics.box, "mp", 0.0)),
        "recall": float(getattr(metrics.box, "mr", 0.0)),
    }

    log.info(
        f"{label} Metrics: "
        f"mAP50={result_dict['mAP50']:.4f}, "
        f"mAP50-95={result_dict['mAP50_95']:.4f}, "
        f"precision={result_dict['precision']:.4f}, "
        f"recall={result_dict['recall']:.4f}"
    )
    return result_dict


def render_evaluation_montage(
    model: Any,
    image_paths: list[Path],
    labels_dir: Path,
    output_image_path: Path,
    class_names: list[str],
    conf_threshold: float,
    max_boxes: int,
    montage_cols: int,
    img_size: int,
    log: SimpleNamespace,
) -> None:
    """Render a visual montage comparing ground-truth targets with model predictions.

    Args:
        model (Any): Loaded YOLO model used for inference.
        image_paths (list[Path]): List of image paths to process and montage.
        labels_dir (Path): Directory containing matching YOLO label files (.txt).
        output_image_path (Path): Path where the resulting montage will be saved.
        class_names (list[str]): Class vocabulary mapping class ID to display name.
        conf_threshold: (float): Confidence score threshold for predicted boxes.
        max_boxes (int): Maximum predicted boxes to display per image.
        montage_cols (int): Number of columns in the montage grid.
        img_size (int): Inference image dimension.
        log (SimpleNamespace): Logger instance.
    """
    annotated_images: list[Image.Image] = []
    for img_path in image_paths:
        if not img_path.is_file():
            continue
        lbl_path: Path = labels_dir / f"{img_path.stem}.txt"
        gt_lines: list[str] = read_gt_lines(lbl_path)
        pred_boxes, _ = predict_boxes(
            model=model,
            image_path=img_path,
            conf=conf_threshold,
            max_boxes=max_boxes,
            imgsz=img_size,
        )

        with Image.open(img_path) as raw_img:
            annotated: Image.Image = draw_boxes_on_image(
                img=raw_img.convert("RGB"),
                gt_lines=gt_lines,
                pred_boxes=pred_boxes,
                class_names=class_names,
            )
            annotated_images.append(annotated)

    if not annotated_images:
        log.warning("No images available to create montage; skipping.")
        return

    montage: Image.Image = make_montage(annotated_images, cols=montage_cols)
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(str(output_image_path))
    log.info(f"Saved visual evaluation montage to {output_image_path}")


def format_summary_table(
    weights_path: str,
    test_sample_count: int,
    diff_metrics: dict[str, float],
    gt_metrics: dict[str, float] | None,
) -> str:
    """Format evaluation metrics into an aligned ASCII comparison table.

    Args:
        weights_path (str): Path to the evaluated YOLO weights file.
        test_sample_count (int): Number of evaluated test samples.
        diff_metrics (dict[str, float]): Metrics evaluated on diffusion outputs.
        gt_metrics (dict[str, float] | None): Metrics evaluated on real ground-truth.

    Returns:
        str: Multi-line formatted ASCII table string.
    """
    lines: list[str] = [
        "",
        "=" * 82,
        "                      YOLO VALIDATION BENCHMARK REPORT",
        "=" * 82,
        f"Weights File: {weights_path}",
        f"Evaluated Test Samples: {test_sample_count}",
        "-" * 82,
        f"{'Metric':<16} | {'Diffusion-Generated Test Split':<30} | {'Real Ground-Truth Test Split':<28}",
        "-" * 82,
    ]

    metrics_keys: list[tuple[str, str]] = [
        ("mAP@50", "mAP50"),
        ("mAP@50-95", "mAP50_95"),
        ("Precision", "precision"),
        ("Recall", "recall"),
    ]

    for display_name, key in metrics_keys:
        diff_val_str: str = f"{diff_metrics.get(key, 0.0):.4f}"
        gt_val_str: str = (
            f"{gt_metrics.get(key, 0.0):.4f}" if gt_metrics is not None else "N/A (Skipped)"
        )
        lines.append(f"{display_name:<16} | {diff_val_str:<30} | {gt_val_str:<28}")

    lines.append("=" * 82)
    lines.append("")
    return "\n".join(lines)


class DiffusionTestValidator:
    """Validator orchestrating the live diffusion test run and GT comparison."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize configurations, paths, device, and logger.

        Args:
            args (argparse.Namespace): Command-line options.
        """
        self.args: argparse.Namespace = args
        self.config: SimpleNamespace = load_config(
            args.mode, str(SCRIPT_DIR / "config.toml")
        )
        anchor_config_paths(self.config)
        self.hyperparams: SimpleNamespace = load_hyperparams(
            str(SCRIPT_DIR / "hyperparameters.toml")
        )
        self.log: SimpleNamespace = setup_logger(self.config)
        self.device_str: str = resolve_train_device(self.config.DEVICE, self.log)
        self.class_names: list[str] = list(self.hyperparams.CLASS_NAMES)

        # Output directory resolution
        if args.output_dir:
            self.eval_output_dir: Path = Path(args.output_dir).resolve()
        else:
            self.eval_output_dir = (
                Path(self.config.OUTPUT_SAMPLES_DIRECTORY).resolve()
                / f"diffusion_test_eval_{self.config.runtime.RUN_ID}"
            )
        self.eval_output_dir.mkdir(parents=True, exist_ok=True)

        # Sample cap and seed overrides
        self.sample_cap: int = (
            args.cap
            if args.cap is not None
            else int(self.hyperparams.TEST_SAMPLE_CAP)
        )
        self.seed: int = (
            args.seed
            if args.seed is not None
            else int(self.hyperparams.BASE_SEED)
        )

        self.log.info(
            f"Initialized DiffusionTestValidator on {self.device_str}. "
            f"Output directory: {self.eval_output_dir}"
        )

    def run(self) -> None:
        """Execute the complete validation benchmark pipeline."""
        # Lazy import of ultralytics to allow unit test execution without dependencies
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Install it via pip in the environment."
            ) from exc

        # 1. Resolve and load YOLO model weights
        weights_path: Path = resolve_weights_path(
            Path(self.config.MODEL_SAVE_PATH), self.args.weights
        )
        self.log.info(f"Loading YOLO detector weights from: {weights_path}")
        model: Any = YOLO(str(weights_path))

        # 2. Load the frozen rendering stack (Diffusion + VAEs)
        stack: FrozenRenderStack = FrozenRenderStack(
            self.config, self.hyperparams, self.log
        )

        # 3. Index test split sample folders
        test_split_src: str = str(self.config.LIVE_SOURCE_TEST_SPLIT_DIR)
        if not Path(test_split_src).is_dir():
            raise FileNotFoundError(
                f"Source test split folder does not exist: {test_split_src}"
            )

        test_sources: list[SampleFolder] = list_sample_folders(
            split_dir=test_split_src,
            cond_mask_file=stack.floorplan_mask_file,
            target_mask_file=stack.arrangement_mask_file,
            cap=self.sample_cap,
        )
        self.log.info(
            f"Indexed {len(test_sources)} test samples from {test_split_src} (cap={self.sample_cap})."
        )

        # 4. Generate Diffusion-Drawn Test Split (Mirroring Training Pipeline)
        diff_img_dir, diff_lbl_dir, diff_yaml_path = prepare_eval_directories(
            self.eval_output_dir / "diffusion_dataset"
        )
        data_yaml_str: str = write_data_yaml(
            live_root=str(self.eval_output_dir / "diffusion_dataset"),
            yaml_path=str(diff_yaml_path),
            class_names=self.class_names,
        )

        self.log.info("Encoding floorplan condition latents for test set...")
        cond_norm: torch.Tensor = stack.encode_source_conditions(test_sources)

        self.log.info(
            f"Sampling diffusion arrangements and rendering test overlays (seed={self.seed})..."
        )
        redrawer: LiveEpochRedrawer = LiveEpochRedrawer(
            stack, self.hyperparams, self.log
        )
        redraw_stats: RedrawStats = redrawer.redraw_epoch(
            sources=test_sources,
            cond_norm=cond_norm,
            epoch=0,
            dst_img_dir=diff_img_dir,
            dst_lbl_dir=diff_lbl_dir,
            manifest_path=self.eval_output_dir / "redraw_manifest.csv",
        )
        self.log.info(
            f"Generated {redraw_stats.images} diffusion test images with "
            f"{redraw_stats.boxes} derived bounding boxes in {redraw_stats.seconds:.1f}s."
        )

        # 5. Evaluate YOLO on the Diffusion-Generated Test Split
        img_size: int = int(self.hyperparams.IMG_SIZE)
        batch_size: int = int(self.hyperparams.BATCH_SIZE)
        diff_metrics: dict[str, float] = evaluate_split(
            model=model,
            yaml_path=data_yaml_str,
            device_str=self.device_str,
            img_size=img_size,
            batch_size=batch_size,
            log=self.log,
            label="Diffusion-Generated Test Split",
        )

        # 6. Evaluate YOLO on the Real Ground-Truth Test Split (if not skipped)
        gt_metrics: dict[str, float] | None = None
        if not self.args.skip_gt:
            live_root: Path = Path(self.config.LIVE_DATASET_ROOT)
            gt_img_dir: Path = live_root / "images" / "test"
            gt_lbl_dir: Path = live_root / "labels" / "test"
            gt_yaml_str: str = write_data_yaml(
                live_root=str(live_root),
                yaml_path=str(self.config.LIVE_DATA_YAML_PATH),
                class_names=self.class_names,
            )

            # Ensure the deterministic GT split is rendered and current
            gt_renderer: DeterministicSplitRenderer = DeterministicSplitRenderer(
                stack, self.hyperparams, self.log
            )
            gt_renderer.render_split(test_sources, gt_img_dir, gt_lbl_dir)

            gt_metrics = evaluate_split(
                model=model,
                yaml_path=gt_yaml_str,
                device_str=self.device_str,
                img_size=img_size,
                batch_size=batch_size,
                log=self.log,
                label="Real Ground-Truth Test Split",
            )

        # 7. Generate Visual Montages for Qualitative Visibility
        if self.args.save_montage and self.args.montage_samples > 0:
            sample_stems: list[str] = choose_fixed_stems(
                sources=[source.stem for source in test_sources],
                count=self.args.montage_samples,
                seed=self.seed,
            )
            diff_sample_paths: list[Path] = [
                diff_img_dir / f"{stem}.png" for stem in sample_stems
            ]

            conf_thresh: float = float(self.hyperparams.EPOCH_OVERLAY_CONF)
            max_boxes: int = int(self.hyperparams.EPOCH_OVERLAY_MAX_BOXES)
            cols: int = int(self.hyperparams.EPOCH_OVERLAY_MONTAGE_COLS)

            render_evaluation_montage(
                model=model,
                image_paths=diff_sample_paths,
                labels_dir=diff_lbl_dir,
                output_image_path=self.eval_output_dir / "montage_diffusion_test.png",
                class_names=self.class_names,
                conf_threshold=conf_thresh,
                max_boxes=max_boxes,
                montage_cols=cols,
                img_size=img_size,
                log=self.log,
            )

            if gt_metrics is not None:
                gt_sample_paths: list[Path] = [
                    Path(self.config.LIVE_DATASET_ROOT) / "images" / "test" / f"{stem}.png"
                    for stem in sample_stems
                ]
                gt_labels_dir: Path = (
                    Path(self.config.LIVE_DATASET_ROOT) / "labels" / "test"
                )
                render_evaluation_montage(
                    model=model,
                    image_paths=gt_sample_paths,
                    labels_dir=gt_labels_dir,
                    output_image_path=self.eval_output_dir / "montage_gt_test.png",
                    class_names=self.class_names,
                    conf_threshold=conf_thresh,
                    max_boxes=max_boxes,
                    montage_cols=cols,
                    img_size=img_size,
                    log=self.log,
                )

        # 8. Print and Persist Benchmark Summary Report
        summary_text: str = format_summary_table(
            weights_path=str(weights_path),
            test_sample_count=len(test_sources),
            diff_metrics=diff_metrics,
            gt_metrics=gt_metrics,
        )
        print(summary_text)
        self.log.info(summary_text)

        summary_data: dict[str, Any] = {
            "timestamp": time.time(),
            "weights_path": str(weights_path),
            "test_sample_count": len(test_sources),
            "seed": self.seed,
            "diffusion_test_metrics": diff_metrics,
            "ground_truth_test_metrics": gt_metrics,
        }
        summary_file: Path = self.eval_output_dir / "validation_summary.json"
        summary_file.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
        self.log.info(f"Saved benchmark summary JSON to {summary_file}")


def main() -> None:
    """CLI entry point for the diffusion test validation script."""
    args: argparse.Namespace = parse_arguments()
    validator = DiffusionTestValidator(args)
    validator.run()


if __name__ == "__main__":
    main()
