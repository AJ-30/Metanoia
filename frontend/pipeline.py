"""From an upload to the picture YOLO drew boxes on.

The path an uploaded mask travels is deliberately the same one a dataset sample
travels, so nothing about the interface can quietly change the result:

1. the two uploads are validated as class-id masks of the right range and are
   re-saved under the exact filenames the frozen encoders read;
2. the sampler encodes both conditions, samples the missing component with every
   requested chain and decodes each chain into a class-id mask; the primary mask
   is the per-pixel average of the chains' missing-component softmax
   probabilities, thresholded once;
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

``stage_inpaint`` reuses the same encode/sample/composite/detect machinery for
the editor: it runs every component model on one edited mask and merges the
per-component ensemble masks into a single furniture layout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
    build_ground_truth_overlay,
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
from frontend.inpainting import (
    merge_component_mask,
    merge_inpainted_components,
    validate_inpaint_edit,
)
from mask_diffusion.inference import (
    ACCEPTED_MASK_IMAGE_MODES,
    ChannelGeneration,
    MaskDiffusionSampler,
    average_probability_maps,
    load_class_ids,
    threshold_probability_maps,
)
from YOLO.overlay import draw_boxes_on_image

# Filenames of the panels the result page shows.
FLOORPLAN_PANEL: str = "floorplan.png"
CONDITIONED_PANEL: str = "conditioned.png"
COMPONENT_PANEL: str = "component_{suffix}.png"
COMPOSITE_PANEL: str = "composite_{suffix}.png"
GT_OVERLAY_PANEL: str = "composite_{suffix}_ground_truth.png"
ANNOTATED_PANEL: str = "composite_{suffix}_boxes.png"
ENSEMBLE_SUFFIX: str = "ensemble"

# Panel filenames of the inpainting result page. The component panels are
# formatted with the component key and one of the four kinds below.
INPAINT_FLOORPLAN_PANEL: str = "floorplan.png"
INPAINT_ORIGINAL_PANEL: str = "original_furniture.png"
INPAINT_EDITED_PANEL: str = "edited_furniture.png"
INPAINT_ERASE_PANEL: str = "erase_overlay.png"
INPAINT_COMPONENT_PANEL: str = "component_{component}_{kind}.png"
INPAINT_MERGED_FURNITURE_PANEL: str = "merged_furniture.png"
INPAINT_MERGED_COMPOSITE_PANEL: str = "merged_composite.png"
INPAINT_MERGED_BOXES_PANEL: str = "merged_boxes.png"

# Class id of the furniture background, i.e. an empty layout.
BLANK_CLASS_ID: int = 0


@dataclass(frozen=True)
class InpaintComponentRun:
    """One component's ensemble result and the masks needed for the merge.

    Attributes:
        result (dict[str, Any]): The component payload the result page renders.
        channel (int): Furniture class id the component owns.
        generated_mask (torch.Tensor): ``(H, W)`` bool ensemble mask.
        probability (torch.Tensor): ``(H, W)`` averaged ensemble probability,
            used to resolve overlaps between components.
    """

    result: dict[str, Any]
    channel: int
    generated_mask: torch.Tensor
    probability: torch.Tensor


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
        draws (int | None): Number of independent diffusion chains to ensemble,
            or ``None`` to use the stage default. Clamped to
            ``1..MAX_DRAWS``.
        ground_truth_path (Path | None): Optional uploaded YOLO ``labels.txt``
            describing the component in the composite, or ``None`` when the
            user did not upload one. The file is validated before any costly
            generation starts.

    Returns:
        dict[str, Any]: The result payload: panel filenames, the ensemble and
        per-draw detections/comparisons, the component identity, timings, the
        ensemble threshold and the ensemble-generated size.

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
    generation: ChannelGeneration = sampler.generate_channel_outputs(
        conditions,
        num_draws=draws,
        seed=int(hyperparams.SEED),
        chunk_size=int(hyperparams.GENERATION_CHUNK),
        chain_offset=0,
    )
    draw_masks: torch.Tensor = generation.ids == sampler.missing_channel
    ensemble_probability: torch.Tensor = average_probability_maps(
        generation.probabilities, draws
    )
    ensemble_mask: torch.Tensor = threshold_probability_maps(
        ensemble_probability, float(hyperparams.ENSEMBLE_PROBABILITY_THRESHOLD)
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

    # The ensemble is the primary result: one mask built from the average
    # missing-component probability of every chain. The per-draw panels stay as
    # diagnostics, so the page still shows what each individual chain produced.
    detections: list[dict[str, Any]] = [
        _render_result(
            context=context,
            output_root=output_root,
            conditions=conditions,
            generated=ensemble_mask[0:1],
            suffix=ENSEMBLE_SUFFIX,
            panels=panels,
            ground_truth=ground_truth,
            ground_truth_path=ground_truth_path,
            is_ensemble=True,
            draw=None,
            members=draws,
        )
    ]
    for draw in range(draws):
        detections.append(
            _render_result(
                context=context,
                output_root=output_root,
                conditions=conditions,
                generated=draw_masks[draw : draw + 1],
                suffix=f"draw_{draw}",
                panels=panels,
                ground_truth=ground_truth,
                ground_truth_path=ground_truth_path,
                is_ensemble=False,
                draw=draw,
                members=None,
            )
        )

    return {
        "request_id": request_id,
        "panels": panels,
        "detections": detections,
        "component": ontology.name,
        "component_label": context.class_names[ontology.yolo_class_id],
        "draws": draws,
        "ensemble_probability_threshold": float(
            hyperparams.ENSEMBLE_PROBABILITY_THRESHOLD
        ),
        "elapsed_seconds": time.time() - started,
        "furniture_uploaded": furniture_path is not None,
        "ground_truth_uploaded": ground_truth_path is not None,
        "ground_truth_boxes": len(ground_truth),
        "generated_pixels": int(ensemble_mask[0].sum()),
    }


def stage_inpaint(
    contexts: dict[str, Any],
    request_id: str,
    floorplan_path: Path,
    original_furniture_path: Path,
    edited_ids: torch.Tensor,
    draws: int | None = None,
) -> dict[str, Any]:
    """Run every component model on one edited mask and merge the results.

    The edited mask is the only furniture layout the models see. It is validated
    against the original upload so the request can only erase the three
    component classes, and each component model is then run one at a time with
    the existing diffusion ensemble. The three generated masks are merged into
    the edited layout with the inpainting rule; the merged layout is composited
    and judged by every component detector.

    Args:
        contexts (dict[str, Any]): Component key -> cached
            ``FrontendContext``. Iteration order is the merge tie-break order.
        request_id (str): Folder name this request's artifacts are written to.
        floorplan_path (Path): Uploaded floorplan segmentation mask.
        original_furniture_path (Path): Uploaded furniture mask before erasing.
        edited_ids (torch.Tensor): ``(H, W)`` int64 furniture mask after the
            browser's eraser was applied. Only erasable component pixels may
            differ from the original.
        draws (int | None): Independent diffusion chains per component, or
            ``None`` to use the stage default. Clamped to ``1..MAX_DRAWS``.

    Returns:
        dict[str, Any]: Panel filenames, per-component results, merged
        detections, the draw count, the ensemble threshold, the merge policy and
        the elapsed time.

    Raises:
        ValueError: If the contexts are empty, a mask is not a class-id mask in
            the expected range, or the edited mask changed a protected pixel.
        FileNotFoundError: If a model checkpoint or detector weights file is
            absent while a context is being used.
    """
    if not contexts:
        raise ValueError("Inpainting needs at least one component context.")

    started: float = time.time()
    first_context: Any = next(iter(contexts.values()))
    sampler: MaskDiffusionSampler = first_context.sampler
    hyperparams: SimpleNamespace = first_context.hyperparams
    image_size: int = sampler.image_size
    draws = _clamp_draws(
        draws if draws is not None else int(hyperparams.DEFAULT_DRAWS),
        int(hyperparams.MAX_DRAWS),
    )

    floorplan_ids: torch.Tensor = validate_and_load_mask(
        floorplan_path,
        sampler.floorplan_layout.num_channels,
        image_size,
        "floorplan",
    )
    original_ids: torch.Tensor = validate_and_load_mask(
        original_furniture_path,
        sampler.target_layout.num_channels,
        image_size,
        "furniture",
    )
    edited_ids = _validate_edited_ids(
        edited_ids, image_size, sampler.target_layout.num_channels
    )
    component_channels: list[int] = [
        int(context.sampler.missing_channel) for context in contexts.values()
    ]
    if len(set(component_channels)) != len(component_channels):
        raise ValueError(
            "Two inpainting components resolve to the same furniture channel; "
            "each component must own a distinct class id."
        )
    validate_inpaint_edit(original_ids, edited_ids, set(component_channels))
    protect_original: bool = bool(
        getattr(hyperparams, "INPAINT_PROTECT_ORIGINAL", True)
    )

    output_root: Path = (
        Path(str(first_context.config.OUTPUT_DIRECTORY)) / request_id
    )
    output_root.mkdir(parents=True, exist_ok=True)
    panels: dict[str, str] = _save_inpaint_input_panels(
        sampler,
        hyperparams,
        floorplan_ids,
        original_ids,
        edited_ids,
        output_root,
    )
    runs: list[InpaintComponentRun] = [
        _run_inpaint_component(
            context=context,
            component_name=component_name,
            floorplan_ids=floorplan_ids,
            edited_ids=edited_ids,
            output_root=output_root,
            draws=draws,
            hyperparams=hyperparams,
            protect_original=protect_original,
        )
        for component_name, context in contexts.items()
    ]
    merged_found: list[dict[str, Any]] = _render_merged_inpaint(
        contexts=contexts,
        sampler=sampler,
        hyperparams=hyperparams,
        floorplan_ids=floorplan_ids,
        edited_ids=edited_ids,
        runs=runs,
        output_root=output_root,
        panels=panels,
        protect_original=protect_original,
    )

    return {
        "request_id": request_id,
        "panels": panels,
        "components": [run.result for run in runs],
        "merged_found": merged_found,
        "draws": draws,
        "ensemble_probability_threshold": float(
            hyperparams.ENSEMBLE_PROBABILITY_THRESHOLD
        ),
        "elapsed_seconds": time.time() - started,
        "image_size": image_size,
        "protect_original": protect_original,
    }


def _save_inpaint_input_panels(
    sampler: MaskDiffusionSampler,
    hyperparams: SimpleNamespace,
    floorplan_ids: torch.Tensor,
    original_ids: torch.Tensor,
    edited_ids: torch.Tensor,
    output_root: Path,
) -> dict[str, str]:
    """Write the four input panels of the inpainting result page.

    Args:
        sampler (MaskDiffusionSampler): Supplies both layouts and their colours.
        hyperparams (SimpleNamespace): Stage numbers, including the erase
            overlay opacity.
        floorplan_ids (torch.Tensor): ``(H, W)`` floorplan class ids.
        original_ids (torch.Tensor): ``(H, W)`` furniture ids before erasing.
        edited_ids (torch.Tensor): ``(H, W)`` furniture ids after erasing.
        output_root (Path): Request output directory.

    Returns:
        dict[str, str]: Panel key -> filename for the result page.
    """
    panels: dict[str, str] = {}

    floorplan_rgb: torch.Tensor = render_layout_ids(
        floorplan_ids.unsqueeze(0),
        sampler.floorplan_layout.groups[0],
        sampler.floorplan_layout.num_channels,
    )
    _save_panel(
        tensor_to_image(floorplan_rgb), output_root / INPAINT_FLOORPLAN_PANEL
    )
    panels["floorplan"] = INPAINT_FLOORPLAN_PANEL

    original_rgb: torch.Tensor = render_layout_ids(
        original_ids.unsqueeze(0),
        sampler.target_group,
        sampler.target_layout.num_channels,
    )
    _save_panel(
        tensor_to_image(original_rgb), output_root / INPAINT_ORIGINAL_PANEL
    )
    panels["original_furniture"] = INPAINT_ORIGINAL_PANEL

    edited_rgb: torch.Tensor = render_layout_ids(
        edited_ids.unsqueeze(0),
        sampler.target_group,
        sampler.target_layout.num_channels,
    )
    _save_panel(
        tensor_to_image(edited_rgb), output_root / INPAINT_EDITED_PANEL
    )
    panels["edited_furniture"] = INPAINT_EDITED_PANEL

    erase_mask: torch.Tensor = original_ids != edited_ids
    _save_panel(
        _build_erase_overlay(
            edited_rgb,
            erase_mask,
            float(hyperparams.INPAINT_ERASE_OVERLAY_ALPHA),
        ),
        output_root / INPAINT_ERASE_PANEL,
    )
    panels["erase_overlay"] = INPAINT_ERASE_PANEL
    return panels


def _run_inpaint_component(
    context: Any,
    component_name: str,
    floorplan_ids: torch.Tensor,
    edited_ids: torch.Tensor,
    output_root: Path,
    draws: int,
    hyperparams: SimpleNamespace,
    protect_original: bool,
) -> InpaintComponentRun:
    """Run one component model, render its panels and collect its merge data.

    Args:
        context (Any): Cached ``FrontendContext`` for this component.
        component_name (str): Registry key of the component.
        floorplan_ids (torch.Tensor): ``(H, W)`` floorplan class ids.
        edited_ids (torch.Tensor): ``(H, W)`` edited furniture class ids.
        output_root (Path): Request output directory.
        draws (int): Independent diffusion chains to ensemble.
        hyperparams (SimpleNamespace): Stage numbers.
        protect_original (bool): Whether original furniture wins over generated
            pixels in the component merge.

    Returns:
        InpaintComponentRun: The result payload plus the masks needed for the
        global merge.
    """
    component_sampler: MaskDiffusionSampler = context.sampler
    channel: int = int(component_sampler.missing_channel)

    # Stage the same two masks under the encoders' filenames for this
    # component. Each component has its own checkpoint, so the sample folder is
    # repeated rather than shared.
    sample_dir: Path = output_root / "samples" / component_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_upload_as(
        floorplan_ids,
        sample_dir / str(context.config.FLOORPLAN_MASK_FILE),
    )
    save_upload_as(
        edited_ids,
        sample_dir / str(context.config.FURNITURE_MASK_FILE),
    )
    conditions: dict[str, Any] = component_sampler.normalize_conditions(
        component_sampler.encode_conditions(
            [sample_dir], int(hyperparams.ENCODE_BATCH_SIZE)
        )
    )
    generation: ChannelGeneration = component_sampler.generate_channel_outputs(
        conditions,
        num_draws=draws,
        seed=int(hyperparams.SEED),
        chunk_size=int(hyperparams.GENERATION_CHUNK),
        chain_offset=0,
    )
    ensemble_probability: torch.Tensor = average_probability_maps(
        generation.probabilities, draws
    )
    ensemble_mask: torch.Tensor = threshold_probability_maps(
        ensemble_probability,
        float(hyperparams.ENSEMBLE_PROBABILITY_THRESHOLD),
    )

    # Keep every chain visible as a diagnostic. The ensemble above is the
    # primary result; these thumbnails show what each independent chain
    # produced before the probabilities were averaged.
    draw_masks: torch.Tensor = generation.ids == channel
    draw_results: list[dict[str, Any]] = []
    for draw in range(draws):
        draw_panel: str = INPAINT_COMPONENT_PANEL.format(
            component=component_name, kind=f"draw_{draw}"
        )
        render_component_image(
            draw_masks[draw : draw + 1], component_sampler
        ).save(str(output_root / draw_panel))
        draw_results.append(
            {
                "index": draw,
                "panel": draw_panel,
                "generated_pixels": int(draw_masks[draw].sum()),
            }
        )

    component_final: torch.Tensor = merge_component_mask(
        edited_ids,
        channel,
        ensemble_mask[0],
        protect_original=protect_original,
    )

    generated_panel: str = INPAINT_COMPONENT_PANEL.format(
        component=component_name, kind="generated"
    )
    final_panel: str = INPAINT_COMPONENT_PANEL.format(
        component=component_name, kind="final"
    )
    composite_panel: str = INPAINT_COMPONENT_PANEL.format(
        component=component_name, kind="composite"
    )
    boxes_panel: str = INPAINT_COMPONENT_PANEL.format(
        component=component_name, kind="boxes"
    )
    render_component_image(
        ensemble_mask[0:1], component_sampler
    ).save(str(output_root / generated_panel))
    render_component_image(
        component_final.unsqueeze(0), component_sampler
    ).save(str(output_root / final_panel))

    component_composite: Image.Image = build_composite(
        conditions["floorplan_ids"][0:1],
        edited_ids.unsqueeze(0),
        component_final.unsqueeze(0),
        component_sampler,
        float(hyperparams.OVERLAY_ALPHA),
        float(hyperparams.OVERLAY_ALPHA),
    )
    component_composite.convert("RGB").save(str(output_root / composite_panel))
    found: list[tuple[float, float, float, float, int, float]] = (
        predict_component_boxes(
            model=context.model,
            image_path=output_root / composite_panel,
            confidence=float(hyperparams.YOLO_CONF),
            image_size=int(hyperparams.YOLO_IMGSZ),
            max_detections=int(hyperparams.YOLO_MAX_BOXES),
            class_id=context.ontology.yolo_class_id,
        )
    )
    annotate_composite(
        component_composite,
        [],
        found,
        context.class_names,
        context.ontology.yolo_class_id,
    ).save(str(output_root / boxes_panel))

    result: dict[str, Any] = {
        "name": component_name,
        "label": context.class_names[context.ontology.yolo_class_id],
        "channel": channel,
        "panels": {
            "generated": generated_panel,
            "final": final_panel,
            "composite": composite_panel,
            "boxes": boxes_panel,
        },
        "generated_pixels": int(ensemble_mask.sum()),
        "preserved_pixels": int((edited_ids == channel).sum()),
        "final_pixels": int(component_final.sum()),
        "draws": draw_results,
        "found": _found_payload(
            found, context.class_names, component_sampler.image_size
        ),
    }
    return InpaintComponentRun(
        result=result,
        channel=channel,
        generated_mask=ensemble_mask[0],
        probability=ensemble_probability[0],
    )


def _render_merged_inpaint(
    contexts: dict[str, Any],
    sampler: MaskDiffusionSampler,
    hyperparams: SimpleNamespace,
    floorplan_ids: torch.Tensor,
    edited_ids: torch.Tensor,
    runs: list[InpaintComponentRun],
    output_root: Path,
    panels: dict[str, str],
    protect_original: bool,
) -> list[dict[str, Any]]:
    """Merge the component runs, render the merged layout and detect it.

    Args:
        contexts (dict[str, Any]): Component key -> cached context, in tie-break
            order and with each component's detector.
        sampler (MaskDiffusionSampler): Supplies both layouts.
        hyperparams (SimpleNamespace): Stage numbers.
        floorplan_ids (torch.Tensor): ``(H, W)`` floorplan class ids.
        edited_ids (torch.Tensor): ``(H, W)`` edited furniture class ids.
        runs (list[InpaintComponentRun]): Per-component ensemble results.
        output_root (Path): Request output directory.
        panels (dict[str, str]): Panel map to extend in place.
        protect_original (bool): Whether original furniture wins over generated
            pixels in the global merge.

    Returns:
        list[dict[str, Any]]: Detections from every component detector on the
        merged composite.
    """
    component_channels: dict[str, int] = {
        run.result["name"]: run.channel for run in runs
    }
    generated_masks: dict[int, torch.Tensor] = {
        run.channel: run.generated_mask for run in runs
    }
    probabilities: dict[int, torch.Tensor] = {
        run.channel: run.probability for run in runs
    }
    merged_ids: torch.Tensor = merge_inpainted_components(
        edited_ids,
        component_channels,
        generated_masks,
        probabilities,
        protect_original=protect_original,
    )
    merged_rgb: torch.Tensor = render_layout_ids(
        merged_ids.unsqueeze(0),
        sampler.target_group,
        sampler.target_layout.num_channels,
    )
    _save_panel(
        tensor_to_image(merged_rgb),
        output_root / INPAINT_MERGED_FURNITURE_PANEL,
    )
    panels["merged_furniture"] = INPAINT_MERGED_FURNITURE_PANEL

    merged_composite: Image.Image = build_ground_truth_overlay(
        floorplan_ids.unsqueeze(0),
        merged_ids.unsqueeze(0),
        sampler,
        float(hyperparams.OVERLAY_ALPHA),
    )
    merged_composite_path: Path = output_root / INPAINT_MERGED_COMPOSITE_PANEL
    _save_panel(merged_composite, merged_composite_path)
    panels["merged_composite"] = INPAINT_MERGED_COMPOSITE_PANEL

    # Run every component detector on the merged composite. The per-component
    # detectors are single-class, so their class id is remapped to the
    # component's index to let one draw call label all three vocabularies.
    merged_found: list[dict[str, Any]] = []
    merged_pred_boxes: list[tuple[float, float, float, float, int, float]] = []
    merged_class_names: list[str] = []
    for index, (component_name, context) in enumerate(contexts.items()):
        label: str = context.class_names[context.ontology.yolo_class_id]
        merged_class_names.append(label)
        found: list[tuple[float, float, float, float, int, float]] = (
            predict_component_boxes(
                model=context.model,
                image_path=merged_composite_path,
                confidence=float(hyperparams.YOLO_CONF),
                image_size=int(hyperparams.YOLO_IMGSZ),
                max_detections=int(hyperparams.YOLO_MAX_BOXES),
                class_id=context.ontology.yolo_class_id,
            )
        )
        for box in found:
            merged_pred_boxes.append(
                (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    index,
                    float(box[5]),
                )
            )
            payload: dict[str, Any] = _found_payload(
                [box], context.class_names, sampler.image_size
            )[0]
            payload["component"] = component_name
            merged_found.append(payload)

    draw_boxes_on_image(
        merged_composite,
        [],
        merged_pred_boxes,
        merged_class_names,
    ).save(str(output_root / INPAINT_MERGED_BOXES_PANEL))
    panels["merged_boxes"] = INPAINT_MERGED_BOXES_PANEL
    return merged_found


def _validate_edited_ids(
    edited_ids: torch.Tensor, image_size: int, num_classes: int
) -> torch.Tensor:
    """Check the browser's raw class-id tensor before any model runs.

    Args:
        edited_ids (torch.Tensor): Tensor decoded from the form's base64 raw
            byte array.
        image_size (int): Square side length of the model's domain.
        num_classes (int): Number of furniture classes, background included.

    Returns:
        torch.Tensor: The same data as an int64 ``(image_size, image_size)``
        tensor.

    Raises:
        ValueError: If the shape or a class id is outside the layout's range.
    """
    tensor: torch.Tensor = edited_ids.to(torch.int64)
    expected: tuple[int, int] = (int(image_size), int(image_size))
    if tuple(tensor.shape) != expected:
        raise ValueError(
            "The edited furniture mask has shape "
            f"{tuple(tensor.shape)}, but this model expects {expected}."
        )
    minimum: int = int(tensor.min())
    maximum: int = int(tensor.max())
    if minimum < 0 or maximum >= int(num_classes):
        raise ValueError(
            "The edited furniture mask holds a class id outside "
            f"0..{int(num_classes) - 1}."
        )
    return tensor


def _build_erase_overlay(
    edited_rgb: torch.Tensor, erase_mask: torch.Tensor, alpha: float
) -> Image.Image:
    """Draw the erased pixels over the edited layout.

    Args:
        edited_rgb (torch.Tensor): ``(1, 3, H, W)`` RGB rendering of the edited
            furniture layout.
        erase_mask (torch.Tensor): ``(H, W)`` bool mask of erased pixels.
        alpha (float): Opacity of the red overlay in ``0..1``.

    Returns:
        Image.Image: The edited layout with a translucent red erase layer.
    """
    base: Image.Image = tensor_to_image(edited_rgb).convert("RGBA")
    mask_image: Image.Image = Image.fromarray(
        (erase_mask.to(torch.uint8) * 255).numpy(), mode="L"
    )
    red: Image.Image = Image.new(
        "RGBA", base.size, (255, 0, 0, int(round(255.0 * float(alpha))))
    )
    overlay: Image.Image = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay.paste(red, (0, 0), mask_image)
    return Image.alpha_composite(base, overlay).convert("RGB")


def _found_payload(
    found: list[tuple[float, float, float, float, int, float]],
    class_names: list[str],
    image_size: int,
) -> list[dict[str, Any]]:
    """Convert pixel-space detector boxes to the result page's JSON shape.

    Args:
        found (list[tuple[float, float, float, float, int, float]]): Detector
            boxes as ``(x1, y1, x2, y2, class_id, confidence)`` in pixels.
        class_names (list[str]): Detector's class vocabulary.
        image_size (int): Square side length used to normalize the box.

    Returns:
        list[dict[str, Any]]: Boxes with a class label and normalized centre
        and size.
    """
    return [
        {
            "class": class_names[int(box[4])],
            "confidence": float(box[5]),
            "cx": (float(box[0]) + float(box[2])) / 2.0 / image_size,
            "cy": (float(box[1]) + float(box[3])) / 2.0 / image_size,
            "w": (float(box[2]) - float(box[0])) / image_size,
            "h": (float(box[3]) - float(box[1])) / image_size,
        }
        for box in found
    ]


def _render_result(
    context: Any,
    output_root: Path,
    conditions: dict[str, Any],
    generated: torch.Tensor,
    suffix: str,
    panels: dict[str, str],
    ground_truth: list[Box],
    ground_truth_path: Path | None,
    is_ensemble: bool,
    draw: int | None,
    members: int | None,
) -> dict[str, Any]:
    """Render, detect and score one result (a draw or the ensemble).

    Args:
        context (Any): The cached frontend context.
        output_root (Path): Request output directory.
        conditions (dict[str, Any]): Normalized conditions of the upload.
        generated (torch.Tensor): ``(1, H, W)`` component mask to composite.
        suffix (str): Panel filename suffix, ``"ensemble"`` or ``"draw_<d>"``.
        panels (dict[str, str]): Panel filename map to extend in place.
        ground_truth (list[Box]): Parsed uploaded ground-truth boxes.
        ground_truth_path (Path | None): Uploaded labels path, or ``None``.
        is_ensemble (bool): Whether this result is the ensemble.
        draw (int | None): Draw index, or ``None`` for the ensemble.
        members (int | None): Number of ensemble members, or ``None``.

    Returns:
        dict[str, Any]: Detection payload for the result page.
    """
    sampler: MaskDiffusionSampler = context.sampler
    hyperparams: SimpleNamespace = context.hyperparams
    ontology: ComponentOntology = context.ontology
    image_size: int = sampler.image_size

    # Panel 3: the component on its own, in its own colour.
    panel_3: str = COMPONENT_PANEL.format(suffix=suffix)
    render_component_image(generated, sampler).save(str(output_root / panel_3))
    panels[f"component_{suffix}"] = panel_3

    # Panel 4: the composite the detector judges, then the same image with the
    # uploaded ground truth only, then ground truth and predictions.
    composite: Image.Image = build_composite(
        conditions["floorplan_ids"][0:1],
        conditions["furniture_ids"][0:1],
        generated,
        sampler,
        float(hyperparams.OVERLAY_ALPHA),
        float(hyperparams.OVERLAY_ALPHA),
    )
    panel_4: str = COMPOSITE_PANEL.format(suffix=suffix)
    panel_4_boxes: str = ANNOTATED_PANEL.format(suffix=suffix)
    composite.convert("RGB").save(str(output_root / panel_4))
    panels[f"composite_{suffix}"] = panel_4
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
        panel_4_gt: str = GT_OVERLAY_PANEL.format(suffix=suffix)
        annotate_composite(
            composite,
            ground_truth,
            [],
            context.class_names,
            ontology.yolo_class_id,
        ).save(str(output_root / panel_4_gt))
        panels[f"ground_truth_{suffix}"] = panel_4_gt
        comparison = compare_boxes(
            ground_truth,
            pixel_boxes_to_predictions(found, ontology.yolo_class_id, image_size),
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
    panels[f"boxes_{suffix}"] = panel_4_boxes
    return {
        "draw": draw,
        "is_ensemble": is_ensemble,
        "panel_suffix": suffix,
        "members": members,
        "found": _found_payload(found, context.class_names, image_size),
        "comparison": comparison,
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
