"""Visualization: what the evaluator produced, shown next to what it should be.

Three kinds of figure are written, all under ``evaluator/outputs/...``:

* ``eval_composite_draw_<d>_<run>.png`` / ``eval_composite_ensemble_<run>.png``
  - the images the detector actually saw, annotated with the dataset's
  ground-truth box (green) and the detector's prediction (red), one column per
  sample.
* ``eval_compare_<run>.png`` - the side-by-side panel, five columns per sample:
  the true layout, the layout the model was conditioned on, the true component,
  the ensemble-generated component and the annotated ensemble composite. This
  is the figure that answers "did the component land in the right place, in the
  right shape, in the right room" without reading a single number.
* ``eval_best_<result>_<run>.png`` and ``eval_worst_<result>_<run>.png`` - the
  best and worst ranked samples by box IoU for each draw and the ensemble, so
  the tail of the distribution is as visible as its mean.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image

from evaluator.box_metrics import BoxLoss
from evaluator.compositor import component_rgb
from evaluator.dataset_gt import Box
from evaluator.results import SampleRecord, result_slug
from mask_diffusion.inference import MaskDiffusionSampler
from YOLO.overlay import draw_boxes_on_image, make_montage

# Background colour of an empty montage cell.
EMPTY_CELL_COLOR: tuple[int, int, int] = (16, 16, 16)
# Directory (under the run root) holding the annotated composite copies.
ANNOTATED_DIRECTORY: str = "annotated"


def choose_visualized_stems(
    stems: list[str], count: int, seed: int
) -> list[str]:
    """Choose the fixed subset of samples that gets visualized.

    Args:
        stems (list[str]): All evaluated sample stems, in sorted order.
        count (int): Subset size; negative or ``>= len`` keeps every stem.
        seed (int): Run seed, so the same subset is shown on every rerun.

    Returns:
        list[str]: The chosen stems, in sorted order.
    """
    if not stems:
        return []
    if count < 0 or count >= len(stems):
        return list(stems)
    rng: random.Random = random.Random(seed)
    return sorted(rng.sample(stems, count))


def annotate_composite(
    image: Image.Image,
    ground_truth: list[Box],
    predicted_pixels: list[tuple[float, float, float, float, int, float]],
    class_names: list[str],
    class_id: int,
) -> Image.Image:
    """Draw the ground-truth and predicted boxes over one composite image.

    Args:
        image (Image.Image): Composite image to annotate.
        ground_truth (list[Box]): Dataset ground-truth boxes, normalized.
        predicted_pixels (list[tuple[float, float, float, float, int, float]]):
            Predicted boxes in pixels of the same image.
        class_names (list[str]): Class vocabulary, for the box captions.
        class_id (int): Class index the ys belong to.

    Returns:
        Image.Image: Annotated copy of the composite.
    """
    gt_lines: list[str] = [
        f"{class_id} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}"
        for box in ground_truth
    ]
    return draw_boxes_on_image(
        img=image,
        gt_lines=gt_lines,
        pred_boxes=list(predicted_pixels),
        class_names=class_names,
    )


def build_annotated_montage(
    annotated_paths: list[Path],
    output_path: Path,
    columns: int,
) -> Path | None:
    """Tile annotated composites into one contact sheet.

    Args:
        annotated_paths (list[Path]): Annotated images, row-major order.
        output_path (Path): Destination PNG.
        columns (int): Columns of the tile grid.

    Returns:
        Path | None: The written path, or ``None`` when there was nothing to tile.
    """
    if not annotated_paths:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    make_montage([Path(path) for path in annotated_paths], output_path, columns)
    return output_path


def build_sample_strip(
    ground_truth_overlay: Image.Image,
    conditioned_overlay: Image.Image,
    ground_truth_component: Image.Image,
    generated_component: Image.Image,
    annotated_composite: Image.Image,
) -> Image.Image:
    """Lay out one sample's five comparison images side by side.

    Args:
        ground_truth_overlay (Image.Image): Floorplan + the true full layout.
        conditioned_overlay (Image.Image): Floorplan + the true layout minus the
            missing component.
        ground_truth_component (Image.Image): The true component alone.
        generated_component (Image.Image): The generated component alone.
        annotated_composite (Image.Image): The evaluated image with both boxes.

    Returns:
        Image.Image: A five-panel-wide strip of one sample.
    """
    panels: list[Image.Image] = [
        ground_truth_overlay,
        conditioned_overlay,
        ground_truth_component,
        generated_component,
        annotated_composite,
    ]
    cell_width, cell_height = panels[0].size
    strip: Image.Image = Image.new("RGB", (cell_width * len(panels), cell_height), EMPTY_CELL_COLOR)
    for index, panel in enumerate(panels):
        strip.paste(panel.convert("RGB"), (index * cell_width, 0))
    return strip


def rank_samples(
    losses: dict[str, list[BoxLoss]], count: int
) -> tuple[list[str], list[str]]:
    """Split samples into the best and worst by mean box IoU.

    Samples with no matched box sort as worst (IoU 0 by definition), so a
    component the detector never finds is visible at the bottom of the ranking
    rather than disappearing from it.

    Args:
        losses (dict[str, list[BoxLoss]]): Per-sample list of matched losses.
        count (int): How many to keep at each end; negative keeps all.

    Returns:
        tuple[list[str], list[str]]: Best stems and worst stems.
    """
    scored: list[tuple[float, str]] = [
        (sum(item.iou for item in items) / len(items) if items else 0.0, stem)
        for stem, items in losses.items()
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if count < 0:
        return [stem for _score, stem in scored], []
    return (
        [stem for _score, stem in scored[:count]],
        [stem for _score, stem in scored[-count:][::-1]],
    )


def annotate_record_path(
    record: SampleRecord,
    result_key: int,
    class_names: list[str],
    class_id: int,
    output_root: Path,
) -> Path:
    """Copy one evaluated composite with both boxes drawn on it.

    Args:
        record (SampleRecord): Sample to annotate.
        result_key (int): Draw index, or :data:`ENSEMBLE_RESULT`.
        class_names (list[str]): Detector class vocabulary.
        class_id (int): Class index of the component.
        output_root (Path): Run output root.

    Returns:
        Path: Path of the written annotated image.
    """
    image_path: Path = record.image_paths[result_key]
    with Image.open(image_path) as opened:
        composite: Image.Image = opened.convert("RGB")
    annotated: Image.Image = annotate_composite(
        composite,
        record.ground_truth,
        record.predictions[result_key],
        class_names,
        class_id,
    )
    annotation_dir: Path = output_root / ANNOTATED_DIRECTORY
    annotation_dir.mkdir(parents=True, exist_ok=True)
    destination: Path = (
        annotation_dir / f"{record.stem}__{result_slug(result_key)}.png"
    )
    annotated.save(str(destination))
    return destination


def build_compare_montage(compare_dir: Path, output_path: Path, columns: int) -> Path | None:
    """Tile the comparison strips into one figure.

    Args:
        compare_dir (Path): Directory holding the per-sample strips.
        output_path (Path): Destination PNG.
        columns (int): Columns of the tile grid (columns of strips, each of
            which is already five panels wide).

    Returns:
        Path | None: The written figure, or ``None`` when no strip was written.
    """
    strips: list[Path] = sorted(compare_dir.glob("*.png"))
    if not strips:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    make_montage(strips, output_path, columns)
    return output_path


def relative_path(path: Path, root: Path) -> str:
    """Path of a figure relative to the run root, for the summary.

    Args:
        path (Path): Absolute figure path.
        root (Path): Run root.

    Returns:
        str: Relative path string, falling back to the absolute path.
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def render_component_image(
    mask: torch.Tensor,
    sampler: MaskDiffusionSampler,
) -> Image.Image:
    """Render a component mask in the component's own colour on black.

    Args:
        mask (torch.Tensor): Boolean mask of one component instance, shaped
            ``(1, H, W)``.
        sampler (MaskDiffusionSampler): Supplies the target layout, the
            component's channel colours and the channel count.

    Returns:
        Image.Image: The mask drawn in the component's colour on black.
    """
    tensor: torch.Tensor = mask.reshape(1, 1, *mask.shape[-2:]).to(torch.float32)
    rgb: torch.Tensor = component_rgb(
        tensor,
        sampler.target_group,
        sampler.missing_channel,
        sampler.target_layout.num_channels,
    )
    return Image.fromarray(
        (rgb[0].clamp(0.0, 1.0).permute(1, 2, 0).mul(255).to(torch.uint8).numpy())
    )
