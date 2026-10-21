"""The detector side of the evaluation: dataset layout, validation and predictions.

Two detector passes are needed and they answer different questions:

* :func:`run_detector_validation` runs Ultralytics' own validation over every
  evaluated image, restricted to the missing component's class, and returns the
  headline metrics (mAP@50, mAP@50-95, precision, recall) for that class.
* :func:`predict_component_boxes` returns the component's boxes per image, which
  is what the box loss is computed from.

The images and labels are written to ``evaluator/outputs/eval_<run>/dataset`` -
never into the live dataset the detector was trained on, which would both
destroy training data and make the numbers meaningless.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image

from YOLO.dataset import clear_generated_split, write_data_yaml, write_label_file

# Pixel-space prediction tuple returned to the metric layer:
# ``(x1, y1, x2, y2, class_id, confidence)`` in the ORIGINAL image's pixels.
PixelBox = tuple[float, float, float, float, int, float]


def detector_class_names(model: Any) -> list[str]:
    """Read the selected detector's own class vocabulary, ordered by class id.

    The evaluator now judges per-component ``missing_yolo`` detectors as well as
    the multi-class YOLO detector, and those two do not share a vocabulary: a
    missing_yolo model has exactly one class (id 0), while the multi-class model
    has ten. The weights are the only authority on what an id means, so their
    names are read here instead of being assumed from a stage's TOML.

    Args:
        model (Any): Loaded Ultralytics detector exposing ``names`` as either a
            mapping of id -> name or an ordered list.

    Returns:
        list[str]: Class names indexed by class id.

    Raises:
        ValueError: If the model exposes no class names, because every filtered
            metric would then be attributed to an unknown class.
    """
    raw_names: Any = getattr(model, "names", None)
    if isinstance(raw_names, dict):
        if not raw_names:
            raise ValueError("The detector exposes an empty class vocabulary.")
        indices: list[int] = sorted(int(index) for index in raw_names)
        if indices != list(range(len(indices))):
            raise ValueError(
                f"The detector's class ids are not contiguous from 0: {indices}."
            )
        class_names = [str(raw_names[index]) for index in indices]
    elif isinstance(raw_names, (list, tuple)):
        class_names = [str(name) for name in raw_names]
    else:
        raise ValueError(
            "The detector exposes no class names; its predictions could not be "
            "attributed to the missing component."
        )
    if not class_names:
        raise ValueError("The detector exposes an empty class vocabulary.")
    return class_names


def validate_detector_names(model: Any, class_names: list[str]) -> None:
    """Check that the loaded weights share the evaluator's class vocabulary.

    A class id means whatever the weights were trained to mean. If the ontology
    read from the YOLO hyperparameters does not match the weights, every
    filtered metric would be attributed to the wrong furniture, so this fails
    before anything is evaluated.

    Args:
        model (Any): Loaded Ultralytics detector.
        class_names (list[str]): Expected ordered class vocabulary.

    Raises:
        ValueError: If the weights' class names differ from the vocabulary.
    """
    model_names: dict[int, str] = {
        int(index): str(name) for index, name in model.names.items()
    }
    expected: dict[int, str] = {index: name for index, name in enumerate(class_names)}
    if model_names != expected:
        raise ValueError(
            "The detector weights do not use the ontology in "
            "YOLO/hyperparameters.toml.\n"
            f"  weights : {model_names}\n"
            f"  expected: {expected}\n"
            "Pass --weights for the run that matches this ontology."
        )


def prepare_eval_dataset(
    eval_root: Path, class_names: list[str]
) -> tuple[Path, Path, str]:
    """Create one Ultralytics dataset for the evaluated images and labels.

    Args:
        eval_root (Path): Root folder for the generated dataset
            (``<output>/eval_<run_id>/dataset``).
        class_names (list[str]): Ordered class vocabulary, index == class id.

    Returns:
        tuple[Path, Path, str]: Images directory, labels directory and the
            absolute path of the written ``data.yaml``.
    """
    image_dir: Path = eval_root / "images" / "test"
    label_dir: Path = eval_root / "labels" / "test"
    data_yaml_path: Path = eval_root / "data.yaml"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    # Clear first: a changed sample set must not leave orphan images behind.
    clear_generated_split(image_dir, label_dir)
    yaml_path: str = write_data_yaml(
        live_root=str(eval_root), yaml_path=str(data_yaml_path), class_names=class_names
    )
    return image_dir, label_dir, yaml_path


def write_eval_image(
    image: Image.Image, destination: Path, compress_level: int
) -> None:
    """Write one evaluated image.

    Args:
        image (Image.Image): The composite image the detector will judge.
        destination (Path): Destination PNG path.
        compress_level (int): zlib level 0-9; pixels are identical either way.
    """
    image.convert("RGB").save(str(destination), compress_level=int(compress_level))


def write_eval_labels(label_path: Path, lines: list[str]) -> None:
    """Write one evaluated label file, empty when the component is absent.

    Args:
        label_path (Path): Destination ``.txt`` path.
        lines (list[str]): ``"<cls> cx cy w h"`` lines for the component only.
    """
    write_label_file(label_path, lines)


@torch.no_grad()
def predict_component_boxes(
    model: Any,
    image_path: Path,
    confidence: float,
    image_size: int,
    max_detections: int,
    class_id: int,
) -> list[PixelBox]:
    """Run the detector on one image and keep only the component's boxes.

    Boxes come back in the original image's pixels (Ultralytics undoes its own
    letterboxing), which is the domain the ground-truth boxes live in.

    Args:
        model (Any): Loaded Ultralytics detector.
        image_path (Path): Evaluated composite image.
        confidence (float): Confidence threshold the predictions are kept at.
        image_size (int): Inference size the detector is run at.
        max_detections (int): ``max_det`` cap, highest-confidence first.
        class_id (int): Class index of the missing component.

    Returns:
        list[PixelBox]: The component's detections as
            ``(x1, y1, x2, y2, class_id, confidence)`` in pixels.
    """
    results: Any = model(
        str(image_path),
        conf=confidence,
        imgsz=image_size,
        max_det=max_detections,
        verbose=False,
    )[0]
    if getattr(results, "boxes", None) is None:
        return []
    xyxy: Any = results.boxes.xyxy.cpu().tolist()
    classes: Any = results.boxes.cls.cpu().tolist()
    confidences: Any = results.boxes.conf.cpu().tolist()
    predictions: list[PixelBox] = []
    for box, detected_class, box_conf in zip(xyxy, classes, confidences):
        if int(detected_class) != int(class_id):
            continue
        predictions.append(
            (float(box[0]), float(box[1]), float(box[2]), float(box[3]), int(class_id), float(box_conf))
        )
    return predictions


@torch.no_grad()
def run_detector_validation(
    model: Any,
    data_yaml_path: str,
    device_str: str,
    image_size: int,
    batch_size: int,
    class_id: int,
    log: SimpleNamespace,
    label: str,
) -> dict[str, float]:
    """Validate the detector on the evaluated split, filtered to one class.

    Args:
        model (Any): Loaded Ultralytics detector.
        data_yaml_path (str): ``data.yaml`` of the evaluated dataset.
        device_str (str): Device string for Ultralytics (``mps``/``cuda``/``cpu``).
        image_size (int): Inference size for validation.
        batch_size (int): Validation batch size.
        class_id (int): Class index of the missing component.
        log (SimpleNamespace): Configured project logger.
        label (str): Human-readable label used in the logs.

    Returns:
        dict[str, float]: ``mAP50``, ``mAP50-95``, ``precision`` and ``recall``.
    """
    log.info(f"Running detector validation on {label} (class {class_id} only)...")
    metrics: Any = model.val(
        data=data_yaml_path,
        split="test",
        device=device_str,
        imgsz=image_size,
        batch=batch_size,
        classes=[int(class_id)],
        verbose=False,
    )
    results: dict[str, float] = {
        "mAP50": float(getattr(metrics.box, "map50", 0.0)),
        "mAP50_95": float(getattr(metrics.box, "map", 0.0)),
        "precision": float(getattr(metrics.box, "mp", 0.0)),
        "recall": float(getattr(metrics.box, "mr", 0.0)),
    }
    log.info(
        f"{label}: mAP@50={results['mAP50']:.4f}, mAP@50-95={results['mAP50_95']:.4f}, "
        f"precision={results['precision']:.4f}, recall={results['recall']:.4f}"
    )
    return results
