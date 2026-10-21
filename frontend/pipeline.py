"""From an upload to the picture YOLO drew boxes on.

The path an uploaded mask travels is deliberately the same one a dataset sample
travels, so nothing about the interface can quietly change the result:

1. the two uploads are validated as class-id masks of the right range and are
   re-saved under the exact filenames the frozen encoders read;
2. the sampler encodes both conditions, samples the missing component and decodes
   it into a class-id mask (argmax over channels, no tuned threshold);
3. the composite is built - floorplan, real furniture minus the missing
   component, generated component on top - and the detector is run on it;
4. an optional uploaded ``labels.txt`` is matched against the detector's boxes,
   and both the ground truth (green) and the predictions (red) are drawn on the
   composite;
5. the panels, the detection table and the comparison metrics are written out.

A missing furniture upload becomes a blank layout, which is what "generate the
component into an empty floor plan" means: the blank mask still goes through the
partial VAE's layout, so it is the model being honest about an empty room rather
than a special-cased path.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image
from PIL import UnidentifiedImageError as PILUnidentifiedImageError
import torch

from evaluator.box_metrics import PredictedBox, pixel_boxes_to_predictions
from evaluator.compositor import (
    build_composite,
    build_conditioned_overlay,
    render_layout_ids,
    tensor_to_image,
)
from evaluator.dataset_gt import Box
from evaluator.ontology import ComponentOntology
from evaluator.visualize import annotate_composite, render_component_image
from evaluator.yolo_eval import predict_component_boxes
from frontend.comparison import (
    DetectionComparison,
    compare_boxes,
    read_ground_truth_labels,
)
from mask_diffusion.inference import (
    ACCEPTED_MASK_IMAGE_MODES,
    MaskDiffusionSampler,
    load_class_ids,
)

# Filenames of the panels the result page shows.
FLOORPLAN_PANEL: str = "floorplan.png"
CONDITIONED_PANEL: str = "conditioned.png"
COMPONENT_PANEL: str = "component_draw_{draw}.png"
COMPOSITE_PANEL: str = "composite_draw_{draw}.png"
GT_OVERLAY_PANEL: str = "composite_draw_{draw}_ground_truth.png"
ANNOTATED_PANEL: str = "composite_draw_{draw}_boxes.png"

# Class id of the furniture background, i.e. an empty layout.
BLANK_CLASS_ID: int = 0


def _accept_single_channel(path: Path, role: str) -> None:
    """Reject an upload that is not a single-channel class-id mask up front.

    ``load_class_ids`` already validates the image mode, but its message quotes
    the file's server-side path. Checking here keeps the answer about the
    upload's shape instead of about where the server happened to save it.

    Args:
        path (Path): Uploaded PNG.
        role (str): Human-readable role for the error message.

    Raises:
        ValueError: If the file is not a class-id mask (single channel, mode L
            or P). A colour photo or a render is the common mistake.
    """
    with Image.open(path) as image:
        if image.mode not in ACCEPTED_MASK_IMAGE_MODES:
            raise ValueError(
                f"The {role} upload is an image of mode '{image.mode}', not a "
                "single-channel class-id mask. Export the segmentation PNG in "
                "mode L, where each pixel is one class id, and try again."
            )


def validate_and_load_mask(
    path: Path, num_classes: int, image_size: int, role: str
) -> torch.Tensor:
    """Read an uploaded mask and check it is a class-id mask in range.

    Args:
        path (Path): Uploaded PNG, any size, single-channel (mode L or P).
        num_classes (int): Number of classes the layout expects, so an id above
            ``num_classes - 1`` is rejected instead of silently clamping to a
            neighbouring class.
        image_size (int): Square side length the frozen VAEs were trained at.
        role (str): Human-readable role for the error message, "floorplan" or
            "furniture".

    Returns:
        torch.Tensor: int64 ``(image_size, image_size)`` class-id tensor.

    Raises:
        ValueError: If the file is not a readable single-channel class-id mask,
            or holds an id outside the layout's range.
    """
    try:
        _accept_single_channel(path, role)
        ids: torch.Tensor = load_class_ids(path, image_size)
    except PILUnidentifiedImageError as exc:
        raise ValueError(
            f"The {role} upload is not a readable PNG image. Export a single-channel "
            "class-id mask (mode L, one class id per pixel) and try again."
        ) from exc
    maximum: int = int(ids.max())
    if maximum >= num_classes:
        raise ValueError(
            f"The {role} mask holds class id {maximum}, but that layout has "
            f"{num_classes} classes (ids 0..{num_classes - 1}). Export it as a "
            "class-id PNG (mode L, one id per pixel), not as a colour photo."
        )
    return ids


def save_upload_as(ids: torch.Tensor, destination: Path) -> None:
    """Write one validated mask under the filename the encoders expect.

    Args:
        ids (torch.Tensor): ``(S, S)`` int64 class ids.
        destination (Path): Destination PNG path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ids.to(torch.uint8).numpy(), mode="L").save(str(destination))


def build_blank_mask(shape: tuple[int, int]) -> torch.Tensor:
    """Build an all-background class-id mask of the given shape.

    Args:
        shape (tuple[int, int]): ``(height, width)`` of the blank mask.

    Returns:
        torch.Tensor: int64 tensor of zeros, i.e. every pixel background.
    """
    return torch.full(shape, BLANK_CLASS_ID, dtype=torch.int64)


def stage_upload(
    context: Any,
    request_id: str,
    floorplan_path: Path,
    furniture_path: Path | None,
    draws: int | None = None,
    ground_truth_path: Path | None = None,
) -> dict[str, Any]:
    """Encode, generate, composite, detect and optionally score one upload.

    Args:
        context (Any): The cached ``FrontendContext``.
        request_id (str): Folder name this request's artifacts are written to.
        floorplan_path (Path): Uploaded floorplan segmentation mask.
        furniture_path (Path | None): Uploaded furniture segmentation mask, or
            ``None`` to generate the component into an empty layout.
        draws (int | None): Number of independent diffusion chains, or ``None``
            to use the stage default. Clamped to ``1..MAX_DRAWS``.
        ground_truth_path (Path | None): Optional uploaded YOLO ``labels.txt``
            describing the component in the composite, or ``None`` when the
            user did not upload one. The file is validated before any costly
            generation starts.

    Returns:
        dict[str, Any]: The result payload: panel filenames, the per-draw
        detections and comparisons, the component identity, timings and the
        generated size.

    Raises:
        ValueError: If a mask is not a class-id mask in the expected range, or
            the uploaded label file is malformed.
        FileNotFoundError: If the uploaded label file vanished after upload.
    """
    started: float = time.time()
    sampler: MaskDiffusionSampler = context.sampler
    hyperparams: SimpleNamespace = context.hyperparams
    ontology: ComponentOntology = context.ontology
    image_size: int = sampler.image_size
    draws = _clamp_draws(
        draws if draws is not None else int(hyperparams.DEFAULT_DRAWS),
        int(hyperparams.MAX_DRAWS),
    )
    # Parse the optional labels before encoding: a bad label file should fail in
    # a second, not after a full diffusion chain has already run.
    ground_truth: list[Box] = (
        read_ground_truth_labels(
            ground_truth_path,
            class_id=ontology.yolo_class_id,
            max_boxes=int(hyperparams.MAX_GROUND_TRUTH_BOXES),
            label_names=ontology.label_names,
        )
        if ground_truth_path is not None
        else []
    )

    sample_dir: Path = _prepare_sample_dir(
        context, request_id, floorplan_path, furniture_path, image_size
    )
    conditions: dict[str, Any] = sampler.normalize_conditions(
        sampler.encode_conditions([sample_dir], int(hyperparams.ENCODE_BATCH_SIZE))
    )
    masks: Any = sampler.generate_channel_masks(
        conditions,
        num_draws=draws,
        seed=int(hyperparams.SEED),
        chunk_size=int(hyperparams.GENERATION_CHUNK),
        chain_offset=0,
    )

    output_root: Path = Path(str(context.config.OUTPUT_DIRECTORY)) / request_id
    output_root.mkdir(parents=True, exist_ok=True)
    panels: dict[str, str] = {}

    # Panel 1: the uploaded floorplan, rendered through the layout's colours.
    _save_panel(
        tensor_to_image(
            render_layout_ids(
                conditions["floorplan_ids"][0:1],
                sampler.floorplan_layout.groups[0],
                sampler.floorplan_layout.num_channels,
            )
        ),
        output_root / FLOORPLAN_PANEL,
    )
    panels["floorplan"] = FLOORPLAN_PANEL

    # Panel 2: the real furniture the generator had to work with, with the
    # missing component folded into the background exactly as the encoder sees it.
    _save_panel(
        build_conditioned_overlay(
            conditions["floorplan_ids"][0:1],
            conditions["furniture_ids"][0:1],
            sampler,
            float(hyperparams.OVERLAY_ALPHA),
        ),
        output_root / CONDITIONED_PANEL,
    )
    panels["conditioned"] = CONDITIONED_PANEL

    detections: list[dict[str, Any]] = []
    for draw in range(draws):
        generated = masks[draw : draw + 1]
        # Panel 3: the component on its own, in its own colour.
        panel_3: str = COMPONENT_PANEL.format(draw=draw)
        render_component_image(generated, sampler).save(
            str(output_root / panel_3)
        )
        panels[f"component_draw_{draw}"] = panel_3

        # Panel 4: the composite the detector judges, then the same image with
        # the uploaded ground truth only, then ground truth and predictions.
        composite: Image.Image = build_composite(
            conditions["floorplan_ids"][0:1],
            conditions["furniture_ids"][0:1],
            generated,
            sampler,
            float(hyperparams.OVERLAY_ALPHA),
            float(hyperparams.OVERLAY_ALPHA),
        )
        panel_4: str = COMPOSITE_PANEL.format(draw=draw)
        panel_4_boxes: str = ANNOTATED_PANEL.format(draw=draw)
        composite.convert("RGB").save(str(output_root / panel_4))
        panels[f"composite_draw_{draw}"] = panel_4
        found: list[tuple[float, float, float, float, int, float]] = (
            predict_component_boxes(
                model=context.model,
                image_path=output_root / panel_4,
                confidence=float(hyperparams.YOLO_CONF),
                image_size=int(hyperparams.YOLO_IMGSZ),
                max_detections=int(hyperparams.YOLO_MAX_BOXES),
                class_id=ontology.yolo_class_id,
            )
        )
        comparison: DetectionComparison | None = None
        if ground_truth_path is not None:
            panel_4_gt: str = GT_OVERLAY_PANEL.format(draw=draw)
            annotate_composite(
                composite,
                ground_truth,
                [],
                context.class_names,
                ontology.yolo_class_id,
            ).save(str(output_root / panel_4_gt))
            panels[f"ground_truth_draw_{draw}"] = panel_4_gt
            comparison = compare_boxes(
                ground_truth,
                pixel_boxes_to_predictions(
                    found, ontology.yolo_class_id, image_size
                ),
                iou_threshold=float(hyperparams.MATCH_IOU_THRESHOLD),
                image_size=image_size,
            )
        annotate_composite(
            composite,
            ground_truth,
            found,
            context.class_names,
            ontology.yolo_class_id,
        ).save(str(output_root / panel_4_boxes))
        panels[f"boxes_draw_{draw}"] = panel_4_boxes
        detections.append(
            {
                "draw": draw,
                "found": [
                    {
                        "class": context.class_names[int(box[4])],
                        "confidence": float(box[5]),
                        "cx": (float(box[0]) + float(box[2])) / 2.0 / image_size,
                        "cy": (float(box[1]) + float(box[3])) / 2.0 / image_size,
                        "w": (float(box[2]) - float(box[0])) / image_size,
                        "h": (float(box[3]) - float(box[1])) / image_size,
                    }
                    for box in found
                ],
                "comparison": comparison,
            }
        )

    return {
        "request_id": request_id,
        "panels": panels,
        "detections": detections,
        "component": ontology.name,
        "component_label": context.class_names[ontology.yolo_class_id],
        "draws": draws,
        "elapsed_seconds": time.time() - started,
        "furniture_uploaded": furniture_path is not None,
        "ground_truth_uploaded": ground_truth_path is not None,
        "ground_truth_boxes": len(ground_truth),
        "generated_pixels": int(masks[0].sum()),
    }


def _clamp_draws(requested: int, maximum: int) -> int:
    """Clamp the requested draw count to the stage's allowed band.

    Args:
        requested (int): Draw count the form asked for.
        maximum (int): Upper bound from ``MAX_DRAWS``; each draw is a full
            1000-step reverse chain, so the cap bounds one request's cost.

    Returns:
        int: ``requested`` limited to ``1..maximum``.
    """
    if requested < 1:
        return 1
    return min(int(requested), int(maximum))


def _prepare_sample_dir(
    context: Any,
    request_id: str,
    floorplan_path: Path,
    furniture_path: Path | None,
    image_size: int,
) -> Path:
    """Stage both uploads in one folder named the way the encoders expect.

    Args:
        context (Any): The cached frontend context.
        request_id (str): Folder name for this request's staged masks.
        floorplan_path (Path): Uploaded floorplan mask.
        furniture_path (Path | None): Uploaded furniture mask, or ``None``.
        image_size (int): Square side length of the frozen encoders' domain.

    Returns:
        Path: The sample folder holding the two named masks.

    Raises:
        ValueError: If a mask is not a class-id mask in the expected range.
    """
    sampler: MaskDiffusionSampler = context.sampler
    staged: Path = (
        Path(str(context.config.UPLOAD_DIRECTORY)) / request_id / "sample"
    )
    staged.mkdir(parents=True, exist_ok=True)
    floorplan_ids = validate_and_load_mask(
        floorplan_path, sampler.floorplan_layout.num_channels, image_size, "floorplan"
    )
    save_upload_as(floorplan_ids, staged / str(context.config.FLOORPLAN_MASK_FILE))
    if furniture_path is not None:
        furniture_ids = validate_and_load_mask(
            furniture_path,
            sampler.target_layout.num_channels,
            image_size,
            "furniture",
        )
    else:
        furniture_ids = build_blank_mask((image_size, image_size))
    save_upload_as(
        furniture_ids, staged / str(context.config.FURNITURE_MASK_FILE)
    )
    return staged


def _save_panel(image: Image.Image, destination: Path) -> None:
    """Write one visualization panel.

    Args:
        image (Image.Image): Panel to save.
        destination (Path): Destination PNG path.
    """
    image.convert("RGB").save(str(destination))
