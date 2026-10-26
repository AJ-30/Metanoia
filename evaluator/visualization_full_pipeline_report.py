"""Qualitative full-pipeline report for every registered missing component.

One run answers a deliberately visual question for each component in
``components.toml``:

    Given a real floorplan whose missing component has been erased, what does
    the diffusion generate, what does the matching YOLO detector say about it,
    and how does both compare with the dataset's ground truth?

The report is one contact sheet per component. Every row is one validation
sample, every column is one stage of the pipeline:

1. **conditioned floorplan overlay** - the real furniture with the missing
   component erased, i.e. exactly what the model was conditioned on;
2. **diffusion reconstruction** - the same scene with the generated component
   composited in, i.e. the image the detector is asked to judge;
3. **YOLO vs ground truth** - the detector's boxes in red and the dataset's
   ``labels.txt`` boxes in green, drawn on the reconstruction;
4. **ground truth** - the real layout with the missing component still present.

The same seeded five validation samples are used for every component, and only
samples whose ``labels.txt`` contains every selected component are eligible.
That keeps the rows comparable while guaranteeing that every column has a real
ground-truth component to show.

The script is read-only with respect to the dataset. It loads the frozen
diffusion checkpoint and detector named by ``components.toml`` and writes only
under ``FULL_PIPELINE_OUTPUT_DIRECTORY``.
"""

from __future__ import annotations

import os

# Ultralytics' NMS uses a torchvision op without an MPS kernel. This must be set
# before torch is imported, so it sits above every project import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import gc
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from evaluator.channel_metrics import channel_scores  # noqa: E402
from evaluator.compositor import (  # noqa: E402
    build_composite,
    build_conditioned_overlay,
    build_ground_truth_overlay,
)
from evaluator.dataset_gt import Box, read_component_boxes  # noqa: E402
from evaluator.ontology import ComponentOntology, resolve_component_ontology  # noqa: E402
from evaluator.settings import (  # noqa: E402
    PATH_CONFIG_KEYS,
    anchor_path_keys,
    load_detector_ontology,
)
from evaluator.variants import (  # noqa: E402
    ComponentVariant,
    apply_component_variant,
    load_component_variants,
    resolve_component_variant,
)
from evaluator.visualize import (  # noqa: E402
    annotate_composite,
    render_component_image,
)
from evaluator.yolo_eval import (  # noqa: E402
    PixelBox,
    detector_class_names,
    predict_component_boxes,
)
from helpers.config import load_config  # noqa: E402
from helpers.hyperparams import load_hyperparams  # noqa: E402
from helpers.logger import setup_logger  # noqa: E402
from mask_diffusion.inference import MaskDiffusionSampler, default_device  # noqa: E402
from mask_diffusion.runner import set_reproducibility_seed  # noqa: E402
from YOLO.dataset import SampleFolder, list_sample_folders  # noqa: E402

SCRIPT_TOML: str = "config.toml"
HYPERPARAMS_TOML: str = "hyperparameters.toml"
PANELS_SUBDIRECTORY: str = "panels"
SUMMARY_FILENAME: str = "report.json"
README_FILENAME: str = "README.md"

# Text drawn above each of the four columns. Kept together so the order of the
# figure and the explanation in the README cannot drift apart.
COLUMN_HEADERS: tuple[str, ...] = (
    "Missing component\n(conditioned floorplan)",
    "Diffusion\nreconstruction",
    "YOLO (red) vs GT (green)\nlabels.txt boxes",
    "Ground truth",
)


@dataclass(frozen=True)
class GridStyle:
    """Pixel layout and colors of the contact sheet.

    Attributes:
        header_height (int): Height of the column-header band.
        title_height (int): Height of the title band above the headers.
        row_label_width (int): Width of the numbered row-label column.
        font_size (int): Point size of headers and row labels.
        title_font_size (int): Point size of the title.
        padding (int): Margin around title text and separators.
        background (tuple[int, int, int]): Canvas color behind every panel.
        text (tuple[int, int, int]): Header and row-label text color.
        separator (tuple[int, int, int]): Color of the thin grid lines.
    """

    header_height: int
    title_height: int
    row_label_width: int
    font_size: int
    title_font_size: int
    padding: int
    background: tuple[int, int, int]
    text: tuple[int, int, int]
    separator: tuple[int, int, int]

    @classmethod
    def from_hyperparams(cls, hyperparams: SimpleNamespace) -> "GridStyle":
        """Build the style from the report's TOML keys.

        Args:
            hyperparams (SimpleNamespace): Parsed ``hyperparameters.toml``.

        Returns:
            GridStyle: The configured style.

        Raises:
            ValueError: If a size is not positive, which would make the grid
                unreadable or fail inside PIL.
        """
        style: GridStyle = cls(
            header_height=int(hyperparams.FULL_PIPELINE_HEADER_HEIGHT),
            title_height=int(hyperparams.FULL_PIPELINE_TITLE_HEIGHT),
            row_label_width=int(hyperparams.FULL_PIPELINE_ROW_LABEL_WIDTH),
            font_size=int(hyperparams.FULL_PIPELINE_FONT_SIZE),
            title_font_size=int(hyperparams.FULL_PIPELINE_TITLE_FONT_SIZE),
            padding=int(hyperparams.FULL_PIPELINE_PADDING),
            background=(int(hyperparams.FULL_PIPELINE_BACKGROUND_GRAY),) * 3,
            text=(int(hyperparams.FULL_PIPELINE_TEXT_GRAY),) * 3,
            separator=(int(hyperparams.FULL_PIPELINE_SEPARATOR_GRAY),) * 3,
        )
        if min(
            style.header_height,
            style.title_height,
            style.row_label_width,
            style.font_size,
            style.title_font_size,
        ) <= 0:
            raise ValueError(
                "Every FULL_PIPELINE_* size in hyperparameters.toml must be "
                f"positive, got {style}."
            )
        return style


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load PIL's bundled font at ``size``, with a version-safe fallback.

    Args:
        size (int): Requested point size.

    Returns:
        ImageFont.ImageFont | ImageFont.FreeTypeFont: A usable font object.
    """
    try:
        return ImageFont.load_default(size=int(size))
    except TypeError:
        # Pillow before 10.1 has no sized default font. The report stays
        # readable, just with the fixed-size bitmap font.
        return ImageFont.load_default()


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options that override the TOML defaults.

    Args:
        argv (list[str] | None): Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        argparse.Namespace: Parsed options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Render the qualitative full-pipeline report: conditioned layout, "
            "diffusion reconstruction, YOLO-vs-GT boxes and ground truth."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["debug", "prod"],
        default="debug",
        help="Runtime configuration mode (debug or prod).",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help="Components to render; defaults to every row of components.toml.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Validation samples per component; overrides FULL_PIPELINE_SAMPLES.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed of the sample choice and the diffusion chain.",
    )
    parser.add_argument(
        "--validation-dir",
        type=str,
        default=None,
        help="Folder-per-sample validation split; overrides VALIDATION_IMAGES_DIR.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Report directory; overrides FULL_PIPELINE_OUTPUT_DIRECTORY.",
    )
    return parser.parse_args(argv)


def apply_overrides(
    config: SimpleNamespace,
    hyperparams: SimpleNamespace,
    args: argparse.Namespace,
) -> None:
    """Copy command-line overrides onto the loaded settings.

    Args:
        config (SimpleNamespace): Config object to mutate in place.
        hyperparams (SimpleNamespace): Hyperparameter object to mutate in place.
        args (argparse.Namespace): Parsed command-line options.
    """
    if args.samples is not None:
        hyperparams.FULL_PIPELINE_SAMPLES = int(args.samples)
    if args.seed is not None:
        hyperparams.SEED = int(args.seed)
    if args.validation_dir is not None:
        config.VALIDATION_IMAGES_DIR = str(Path(args.validation_dir).expanduser().resolve())
    if args.output_dir is not None:
        config.FULL_PIPELINE_OUTPUT_DIRECTORY = str(
            Path(args.output_dir).expanduser().resolve()
        )


def select_component_names(
    available: Sequence[str], requested: Sequence[str] | None
) -> list[str]:
    """Resolve the components a run should render, failing on a typo.

    Args:
        available (Sequence[str]): Component keys declared by ``components.toml``.
        requested (Sequence[str] | None): CLI selection, or ``None`` for all.

    Returns:
        list[str]: Selected component keys, in registry order.

    Raises:
        ValueError: If a requested component is not registered.
    """
    if requested is None:
        return list(available)
    unknown: list[str] = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown component(s) {unknown}. components.toml declares {list(available)}."
        )
    # Registry order keeps output ordering stable even when the CLI lists the
    # components in another order.
    return [name for name in available if name in set(requested)]


def load_selected_detector(
    weights_path: str, log: SimpleNamespace
) -> tuple[Any, list[str]]:
    """Load one detector and read its own class vocabulary.

    Args:
        weights_path (str): Path of the selected ``.pt`` weights.
        log (SimpleNamespace): Configured project logger.

    Returns:
        tuple[Any, list[str]]: Loaded Ultralytics detector and ordered class
            names, index == class id.

    Raises:
        FileNotFoundError: If the weights file is absent.
    """
    from ultralytics import YOLO  # lazy so pure helpers import without it

    path: Path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Detector weights not found: {path}.")
    model: Any = YOLO(str(path))
    class_names: list[str] = detector_class_names(model)
    log.info(f"Loaded detector {path} with classes {class_names}.")
    return model, class_names


def build_sampler(config: SimpleNamespace, log: SimpleNamespace) -> MaskDiffusionSampler:
    """Build the frozen diffusion sampler for the selected component.

    Args:
        config (SimpleNamespace): Config whose component selection already set
            ``MASK_DIFFUSION_CHECKPOINT_PATH`` and the missing component name.
        log (SimpleNamespace): Configured project logger.

    Returns:
        MaskDiffusionSampler: Sampler on the configured device.
    """
    device: torch.device = default_device(str(config.DEVICE), log)
    sampler: MaskDiffusionSampler = MaskDiffusionSampler(
        checkpoint_path=str(config.MASK_DIFFUSION_CHECKPOINT_PATH),
        missing_component_name=str(config.MISSING_COMPONENT_CLASS_NAME),
        class_group_prefix=str(config.TARGET_CLASS_GROUP_PREFIX),
        mask_files={
            "floorplan": str(config.FLOORPLAN_MASK_FILE),
            "partial": str(config.FURNITURE_MASK_FILE),
            "target": str(config.FURNITURE_MASK_FILE),
        },
        color_index_dir=Path(str(config.MASK_COLOR_INDEX_DIR)),
        color_index_files={
            "floorplan": str(config.FLOORPLAN_COLOR_INDEX_FILE),
            "partial": str(config.FURNITURE_COLOR_INDEX_FILE),
            "target": str(config.FURNITURE_COLOR_INDEX_FILE),
        },
        device=device,
        log=log,
        fallback_color_dir=Path(str(config.VALIDATION_IMAGES_DIR)).parent,
    )
    return sampler


def component_label_names(
    hyperparams: SimpleNamespace, component_names: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Read every component's dataset-label names from the hyperparameters.

    Args:
        hyperparams (SimpleNamespace): Parsed ``hyperparameters.toml``.
        component_names (Sequence[str]): Components the report will render.

    Returns:
        dict[str, tuple[str, ...]]: Component -> names used by ``labels.txt``.

    Raises:
        ValueError: If a component has no entry in ``COMPONENT_LABEL_NAMES``,
            which would silently make every sample look like a negative.
    """
    table: dict[str, list[str]] = dict(getattr(hyperparams, "COMPONENT_LABEL_NAMES", {}))
    labels: dict[str, tuple[str, ...]] = {}
    for name in component_names:
        names: tuple[str, ...] = tuple(table.get(name, []))
        if not names:
            raise ValueError(
                f"Component '{name}' has no COMPONENT_LABEL_NAMES entry; the "
                "validation samples cannot be checked for ground truth."
            )
        labels[name] = names
    return labels


def read_present_component_names(sample_dir: Path, label_filename: str) -> set[str]:
    """Read the component names present in one sample's ``labels.txt``.

    Args:
        sample_dir (Path): Sample folder holding the label file.
        label_filename (str): Label filename inside the folder.

    Returns:
        set[str]: Every class-name token that appears on a non-empty line.

    Raises:
        FileNotFoundError: If the sample has no label file.
    """
    label_path: Path = sample_dir / label_filename
    if not label_path.is_file():
        raise FileNotFoundError(f"Sample {sample_dir} has no {label_filename}.")
    present: set[str] = set()
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts: list[str] = raw.split()
        if parts:
            present.add(parts[0])
    return present


def choose_shared_samples(
    samples: Sequence[SampleFolder],
    label_names_by_component: dict[str, tuple[str, ...]],
    count: int,
    seed: int,
    label_filename: str,
) -> list[SampleFolder]:
    """Choose validation samples that contain every selected component.

    The shuffle is seeded, so rerunning with the same configuration shows the
    same samples. Candidate labels are read only until enough positives are
    found, which keeps this cheap even when the split has thousands of folders.

    Args:
        samples (Sequence[SampleFolder]): Every usable validation sample.
        label_names_by_component (dict[str, tuple[str, ...]]): Component ->
            dataset-label names.
        count (int): Number of samples to keep; must be positive.
        seed (int): Shuffle seed.
        label_filename (str): Dataset label filename inside each sample.

    Returns:
        list[SampleFolder]: Chosen samples, sorted by stem.

    Raises:
        ValueError: If ``count`` is not positive or the split does not hold
            enough samples containing every selected component.
    """
    if count < 1:
        raise ValueError(f"FULL_PIPELINE_SAMPLES must be >= 1, got {count}.")
    required: list[tuple[str, set[str]]] = [
        (name, set(names)) for name, names in label_names_by_component.items()
    ]
    shuffled: list[SampleFolder] = list(samples)
    random.Random(int(seed)).shuffle(shuffled)
    chosen: list[SampleFolder] = []
    for sample in shuffled:
        present: set[str] = read_present_component_names(sample.src_dir, label_filename)
        if all(present & names for _name, names in required):
            chosen.append(sample)
            if len(chosen) == count:
                break
    if len(chosen) < count:
        raise ValueError(
            f"Only {len(chosen)} validation samples contain every selected "
            f"component {list(label_names_by_component)}, but {count} were "
            "requested. Lower FULL_PIPELINE_SAMPLES or add dataset samples."
        )
    return sorted(chosen, key=lambda sample: sample.stem)


class ComponentReportRenderer:
    """Render one component's contact sheet from a frozen sampler and detector.

    The renderer owns no training state: it encodes the five sample folders,
    walks one reverse chain, composites the generated component, asks the
    detector, and writes the figure plus the individual panels.

    Args:
        sampler (MaskDiffusionSampler): Frozen component generator.
        model (Any): Loaded Ultralytics detector for the same component.
        class_names (list[str]): Detector class vocabulary.
        ontology (ComponentOntology): Resolved component across vocabularies.
        hyperparams (SimpleNamespace): Stage knobs.
        label_filename (str): Dataset label file inside each sample folder.
        component_name (str): Component key, used for titles and paths.
    """

    def __init__(
        self,
        sampler: MaskDiffusionSampler,
        model: Any,
        class_names: list[str],
        ontology: ComponentOntology,
        hyperparams: SimpleNamespace,
        label_filename: str,
        component_name: str,
    ) -> None:
        self.sampler: MaskDiffusionSampler = sampler
        self.model: Any = model
        self.class_names: list[str] = class_names
        self.ontology: ComponentOntology = ontology
        self.hyperparams: SimpleNamespace = hyperparams
        self.label_filename: str = label_filename
        self.component_name: str = component_name

    def render(
        self,
        samples: Sequence[SampleFolder],
        conditions: dict[str, torch.Tensor],
        generated_masks: torch.Tensor,
        output_root: Path,
        style: GridStyle,
    ) -> dict[str, Any]:
        """Render and write the whole contact sheet for this component.

        Args:
            samples (Sequence[SampleFolder]): Samples, in row order.
            conditions (dict[str, torch.Tensor]): Normalized conditions of the
                same samples, as returned by the sampler.
            generated_masks (torch.Tensor): Bool ``(N, H, W)`` generated
                component masks, one per sample.
            output_root (Path): Report root.
            style (GridStyle): Contact-sheet style.

        Returns:
            dict[str, Any]: Manifest entry for this component.

        Raises:
            ValueError: If the sample, condition and mask counts disagree.
        """
        if not (len(samples) == conditions["floorplan_ids"].shape[0] == generated_masks.shape[0]):
            raise ValueError(
                "Samples, floorplan conditions and generated masks must have "
                f"the same count, got {len(samples)}, "
                f"{conditions['floorplan_ids'].shape[0]} and "
                f"{generated_masks.shape[0]}."
            )
        rows: list[dict[str, Any]] = []
        panels: list[list[Image.Image]] = []
        for index, sample in enumerate(samples):
            row, row_panels = self._render_sample(
                sample=sample,
                floorplan_ids=conditions["floorplan_ids"][index : index + 1],
                furniture_ids=conditions["furniture_ids"][index : index + 1],
                generated_mask=generated_masks[index : index + 1],
                output_root=output_root,
            )
            rows.append(row)
            panels.append(row_panels)

        title: str = (
            f"{self.component_name} - full pipeline - "
            f"{len(samples)} validation samples - draw 0"
        )
        figure: Image.Image = build_report_grid(
            panels=panels,
            headers=list(COLUMN_HEADERS),
            row_labels=[str(index + 1) for index in range(len(panels))],
            title=title,
            style=style,
        )
        # Save the finished sheet at the report root, next to report.json and
        # README.md, so the three deliverable images are the first thing a
        # reader sees. The individual panels stay under panels/.
        figure_path: Path = output_root / f"full_pipeline_{self.component_name}.png"
        figure.save(
            str(figure_path),
            compress_level=int(self.hyperparams.PNG_COMPRESS_LEVEL),
        )
        return {
            "component": self.component_name,
            "figure": str(figure_path.relative_to(output_root)),
            "missing_channel": int(self.sampler.missing_channel),
            "detector_class_id": int(self.ontology.yolo_class_id),
            "label_names": list(self.ontology.label_names),
            "rows": rows,
        }

    def _render_sample(
        self,
        sample: SampleFolder,
        floorplan_ids: torch.Tensor,
        furniture_ids: torch.Tensor,
        generated_mask: torch.Tensor,
        output_root: Path,
    ) -> tuple[dict[str, Any], list[Image.Image]]:
        """Render the four panels of one sample and write its individual images.

        Args:
            sample (SampleFolder): Sample being rendered.
            floorplan_ids (torch.Tensor): ``(1, H, W)`` floorplan class ids.
            furniture_ids (torch.Tensor): ``(1, H, W)`` real furniture class ids.
            generated_mask (torch.Tensor): ``(1, H, W)`` bool generated component.
            output_root (Path): Report root.

        Returns:
            tuple[dict[str, Any], list[Image.Image]]: Manifest row and the four
                panels in column order.
        """
        ground_truth: list[Box] = read_component_boxes(
            sample.src_dir, self.label_filename, self.ontology.label_names
        )
        furniture_alpha: float = float(self.hyperparams.COMPOSITE_OVERLAY_ALPHA)
        generated_alpha: float = float(self.hyperparams.GENERATED_OVERLAY_ALPHA)
        conditioned: Image.Image = build_conditioned_overlay(
            floorplan_ids, furniture_ids, self.sampler, furniture_alpha
        )
        reconstruction: Image.Image = build_composite(
            floorplan_ids,
            furniture_ids,
            generated_mask,
            self.sampler,
            furniture_alpha,
            generated_alpha,
        )
        ground_truth_overlay: Image.Image = build_ground_truth_overlay(
            floorplan_ids, furniture_ids, self.sampler, furniture_alpha
        )
        generated_component: Image.Image = render_component_image(
            generated_mask, self.sampler
        )
        true_component: Image.Image = render_component_image(
            furniture_ids == self.sampler.missing_channel, self.sampler
        )

        panel_dir: Path = (
            output_root / PANELS_SUBDIRECTORY / self.component_name / sample.stem
        )
        panel_dir.mkdir(parents=True, exist_ok=True)
        reconstruction_path: Path = panel_dir / "2_diffusion_reconstruction.png"
        reconstruction.save(
            str(reconstruction_path),
            compress_level=int(self.hyperparams.PNG_COMPRESS_LEVEL),
        )
        predictions: list[PixelBox] = predict_component_boxes(
            model=self.model,
            image_path=reconstruction_path,
            confidence=float(self.hyperparams.YOLO_CONF),
            image_size=int(self.hyperparams.YOLO_IMGSZ),
            max_detections=int(self.hyperparams.YOLO_MAX_BOXES),
            class_id=self.ontology.yolo_class_id,
        )
        annotated: Image.Image = annotate_composite(
            reconstruction,
            ground_truth,
            predictions,
            self.class_names,
            self.ontology.yolo_class_id,
        )
        dice, iou = channel_scores(
            generated_mask, furniture_ids == self.sampler.missing_channel
        )

        self._save_panel(conditioned, panel_dir / "1_conditioned_floorplan.png")
        self._save_panel(generated_component, panel_dir / "2_generated_component.png")
        self._save_panel(annotated, panel_dir / "3_yolo_vs_ground_truth.png")
        self._save_panel(ground_truth_overlay, panel_dir / "4_ground_truth.png")
        self._save_panel(true_component, panel_dir / "4_true_component.png")

        row: dict[str, Any] = {
            "stem": sample.stem,
            "gt_boxes": len(ground_truth),
            "pred_boxes": len(predictions),
            "max_confidence": max(
                (float(prediction[5]) for prediction in predictions), default=0.0
            ),
            "channel_dice": float(dice),
            "channel_iou": float(iou),
        }
        return row, [conditioned, reconstruction, annotated, ground_truth_overlay]

    def _save_panel(self, image: Image.Image, path: Path) -> None:
        """Write one individual panel with the configured PNG compression.

        Args:
            image (Image.Image): Panel to write.
            path (Path): Destination PNG.
        """
        image.convert("RGB").save(
            str(path), compress_level=int(self.hyperparams.PNG_COMPRESS_LEVEL)
        )


def build_report_grid(
    panels: Sequence[Sequence[Image.Image]],
    headers: Sequence[str],
    row_labels: Sequence[str],
    title: str,
    style: GridStyle,
) -> Image.Image:
    """Tile the per-sample panels into the numbered four-column contact sheet.

    Args:
        panels (Sequence[Sequence[Image.Image]]): Rows of panels, each row in
            column order.
        headers (Sequence[str]): Column headers, one per column.
        row_labels (Sequence[str]): Short labels printed beside every row.
        title (str): Figure title.
        style (GridStyle): Pixel layout and colors.

    Returns:
        Image.Image: The assembled RGB contact sheet.

    Raises:
        ValueError: If the rows are empty, ragged, have the wrong number of
            columns, or their panels do not share one size.
    """
    if not panels:
        raise ValueError("Cannot build a report grid with no rows.")
    column_count: int = len(headers)
    for index, row in enumerate(panels):
        if len(row) != column_count:
            raise ValueError(
                f"Row {index} has {len(row)} panels but {column_count} headers."
            )
    if len(row_labels) != len(panels):
        raise ValueError(
            f"Got {len(row_labels)} row labels for {len(panels)} rows."
        )
    panel_width, panel_height = panels[0][0].size
    for row in panels:
        for panel in row:
            if panel.size != (panel_width, panel_height):
                raise ValueError(
                    "Every panel must share one size; got "
                    f"{(panel_width, panel_height)} and {panel.size}."
                )

    grid_width: int = style.row_label_width + column_count * panel_width
    grid_height: int = (
        style.title_height + style.header_height + len(panels) * panel_height
    )
    canvas: Image.Image = Image.new(
        "RGB", (grid_width, grid_height), style.background
    )
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(canvas)
    draw.text(
        (style.padding, style.title_height // 2),
        title,
        fill=style.text,
        font=load_font(style.title_font_size),
        anchor="lm",
    )

    header_top: int = style.title_height
    for column, header in enumerate(headers):
        draw.multiline_text(
            (
                style.row_label_width + column * panel_width + panel_width / 2,
                header_top + style.header_height / 2,
            ),
            header,
            fill=style.text,
            font=load_font(style.font_size),
            anchor="mm",
            align="center",
            spacing=2,
        )

    body_top: int = style.title_height + style.header_height
    for row_index, row in enumerate(panels):
        row_top: int = body_top + row_index * panel_height
        draw.text(
            (style.row_label_width / 2, row_top + panel_height / 2),
            row_labels[row_index],
            fill=style.text,
            font=load_font(style.font_size),
            anchor="mm",
        )
        for column, panel in enumerate(row):
            canvas.paste(
                panel.convert("RGB"),
                (style.row_label_width + column * panel_width, row_top),
            )

    _draw_grid_lines(
        draw=draw,
        columns=column_count,
        rows=len(panels),
        panel_size=(panel_width, panel_height),
        body_top=body_top,
        header_top=header_top,
        grid_width=grid_width,
        grid_height=grid_height,
        style=style,
    )
    return canvas


def _draw_grid_lines(
    draw: ImageDraw.ImageDraw,
    columns: int,
    rows: int,
    panel_size: tuple[int, int],
    body_top: int,
    header_top: int,
    grid_width: int,
    grid_height: int,
    style: GridStyle,
) -> None:
    """Draw the thin separators that make the four columns scannable.

    Args:
        draw (ImageDraw.ImageDraw): Canvas drawing context.
        columns (int): Number of panel columns.
        rows (int): Number of panel rows.
        panel_size (tuple[int, int]): ``(width, height)`` of one panel.
        body_top (int): y of the first panel row.
        header_top (int): y of the header band.
        grid_width (int): Canvas width.
        grid_height (int): Canvas height.
        style (GridStyle): Provides the separator color.
    """
    panel_width, panel_height = panel_size
    draw.line(
        [(0, header_top), (grid_width, header_top)], fill=style.separator, width=1
    )
    draw.line(
        [(0, body_top), (grid_width, body_top)], fill=style.separator, width=1
    )
    for column in range(columns + 1):
        x: int = style.row_label_width + column * panel_width
        if x > 0:
            draw.line([(x, header_top), (x, grid_height)], fill=style.separator, width=1)
    for row in range(rows + 1):
        y: int = body_top + row * panel_height
        draw.line([(0, y), (grid_width, y)], fill=style.separator, width=1)


def write_readme(output_root: Path, summary: dict[str, Any]) -> Path:
    """Write the human-readable explanation of the report.

    Args:
        output_root (Path): Report root.
        summary (dict[str, Any]): Manifest returned by the run.

    Returns:
        Path: Path of the written README.
    """
    lines: list[str] = [
        "# Full-pipeline qualitative report",
        "",
        "Generated by `evaluator/visualization_full_pipeline_report.py`.",
        "",
        "Every component figure has **one row per validation sample** and four",
        "columns:",
        "",
        "1. **Missing component (conditioned floorplan)** - real furniture with",
        "   the missing component erased; what the diffusion model was given.",
        "2. **Diffusion reconstruction** - the same scene with the generated",
        "   component composited in; the image YOLO was asked to judge.",
        "3. **YOLO (red) vs GT (green) labels.txt boxes** - detector boxes in",
        "   red, dataset ground-truth boxes in green.",
        "4. **Ground truth** - the real layout, missing component included.",
        "",
        "The samples are shared by every component: they come from a seeded",
        "shuffle of the validation split, keeping only samples whose",
        "`labels.txt` contains every component being rendered.",
        "",
        f"- Seed: `{summary['seed']}`",
        f"- Samples per component: `{summary['samples_per_component']}`",
        f"- Validation split: `{summary['validation_dir']}`",
        "",
        "## Figures",
        "",
    ]
    for component in summary["components"]:
        sample_list: str = ", ".join(
            f"`{row['stem']}`" for row in component["rows"]
        )
        lines.extend(
            [
                f"### {component['component']}",
                "",
                f"- Figure: `{component['figure']}`",
                f"- Checkpoint: `{component['checkpoint']}`",
                f"- Detector: `{component['detector_weights']}`",
                f"- Missing mask channel: `{component['missing_channel']}`",
                f"- Samples: {sample_list}",
                "",
            ]
        )
    readme_path: Path = output_root / README_FILENAME
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def render_component(
    config: SimpleNamespace,
    hyperparams: SimpleNamespace,
    log: SimpleNamespace,
    variant: ComponentVariant,
    samples: Sequence[SampleFolder],
    style: GridStyle,
) -> dict[str, Any]:
    """Render one component's report from its registered checkpoint and detector.

    Args:
        config (SimpleNamespace): Config copied for this component.
        hyperparams (SimpleNamespace): Stage knobs.
        log (SimpleNamespace): Configured project logger.
        variant (ComponentVariant): Selected registry row.
        samples (Sequence[SampleFolder]): Shared validation samples.
        style (GridStyle): Contact-sheet style.

    Returns:
        dict[str, Any]: Manifest entry for the component.
    """
    apply_component_variant(config, variant)
    sampler: MaskDiffusionSampler = build_sampler(config, log)
    model: Any
    class_names: list[str]
    model, class_names = load_selected_detector(str(config.YOLO_WEIGHTS_PATH), log)
    _mask_classes, aliases = load_detector_ontology(str(config.YOLO_HYPERPARAMETERS_PATH))
    ontology = resolve_component_ontology(
        name=str(config.MISSING_COMPONENT_CLASS_NAME),
        group_prefix=str(config.TARGET_CLASS_GROUP_PREFIX),
        mask_channel=sampler.missing_channel,
        class_names=class_names,
        aliases=aliases,
        label_names_table=dict(getattr(hyperparams, "COMPONENT_LABEL_NAMES", {})),
    )
    sample_dirs: list[Path] = [sample.src_dir for sample in samples]
    conditions: dict[str, torch.Tensor] = sampler.normalize_conditions(
        sampler.encode_conditions(
            sample_dirs, int(hyperparams.ENCODE_BATCH_SIZE)
        )
    )
    generated_masks: torch.Tensor = sampler.generate_channel_masks(
        conditions,
        num_draws=1,
        seed=int(hyperparams.SEED),
        chunk_size=int(hyperparams.GENERATION_CHUNK),
        chain_offset=0,
    )
    renderer = ComponentReportRenderer(
        sampler=sampler,
        model=model,
        class_names=class_names,
        ontology=ontology,
        hyperparams=hyperparams,
        label_filename=str(config.GROUND_TRUTH_LABEL_FILE),
        component_name=variant.name,
    )
    entry: dict[str, Any] = renderer.render(
        samples=samples,
        conditions=conditions,
        generated_masks=generated_masks,
        output_root=Path(str(config.FULL_PIPELINE_OUTPUT_DIRECTORY)),
        style=style,
    )
    entry["checkpoint"] = str(variant.mask_diffusion_checkpoint)
    entry["detector_weights"] = str(variant.detector_weights)

    # Free the three frozen VAEs and detector before the next component. The
    # report renders only five chains per component, so reloading is cheaper
    # than risking a long-lived three-model stack on a laptop.
    del renderer, generated_masks, conditions, sampler, model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return entry


def run_report(
    config: SimpleNamespace,
    hyperparams: SimpleNamespace,
    log: SimpleNamespace,
    requested_components: Sequence[str] | None,
) -> dict[str, Any]:
    """Run the whole full-pipeline report.

    Args:
        config (SimpleNamespace): Anchored config.
        hyperparams (SimpleNamespace): Stage knobs.
        log (SimpleNamespace): Configured project logger.
        requested_components (Sequence[str] | None): CLI component selection.

    Returns:
        dict[str, Any]: Summary written to ``report.json``.
    """
    started: float = time.time()
    variants: dict[str, ComponentVariant] = load_component_variants()
    component_names: list[str] = select_component_names(
        list(variants), requested_components
    )
    labels: dict[str, tuple[str, ...]] = component_label_names(
        hyperparams, component_names
    )
    style: GridStyle = GridStyle.from_hyperparams(hyperparams)
    samples: list[SampleFolder] = list_sample_folders(
        split_dir=str(config.VALIDATION_IMAGES_DIR),
        cond_mask_file=str(config.FLOORPLAN_MASK_FILE),
        target_mask_file=str(config.FURNITURE_MASK_FILE),
        cap=-1,
    )
    log.info(
        f"Indexed {len(samples)} validation samples from "
        f"{config.VALIDATION_IMAGES_DIR}."
    )
    chosen: list[SampleFolder] = choose_shared_samples(
        samples=samples,
        label_names_by_component=labels,
        count=int(hyperparams.FULL_PIPELINE_SAMPLES),
        seed=int(hyperparams.SEED),
        label_filename=str(config.GROUND_TRUTH_LABEL_FILE),
    )
    log.info(
        f"Using {len(chosen)} shared validation samples: "
        f"{[sample.stem for sample in chosen]}."
    )

    output_root: Path = Path(str(config.FULL_PIPELINE_OUTPUT_DIRECTORY))
    output_root.mkdir(parents=True, exist_ok=True)
    components: list[dict[str, Any]] = []
    for name in component_names:
        log.info(f"Rendering full-pipeline report for '{name}'...")
        # Use a shallow copy: a component selection only ever rewrites the two
        # model paths and the missing component name, so the dataset settings
        # remain shared.
        component_config: SimpleNamespace = SimpleNamespace(**vars(config))
        component_config.runtime = config.runtime
        components.append(
            render_component(
                config=component_config,
                hyperparams=hyperparams,
                log=log,
                variant=resolve_component_variant(variants, name),
                samples=chosen,
                style=style,
            )
        )

    summary: dict[str, Any] = {
        "seed": int(hyperparams.SEED),
        "samples_per_component": len(chosen),
        "validation_dir": str(config.VALIDATION_IMAGES_DIR),
        "elapsed_seconds": time.time() - started,
        "components": components,
    }
    summary_path: Path = output_root / SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    write_readme(output_root, summary)
    log.info(f"Full-pipeline report written to {output_root}.")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Launch the full-pipeline report from the command line.

    Args:
        argv (list[str] | None): Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        int: Process exit code, ``0`` on success.
    """
    script_dir: Path = Path(__file__).resolve().parent
    args: argparse.Namespace = parse_arguments(argv)
    config: SimpleNamespace = load_config(args.mode, str(script_dir / SCRIPT_TOML))
    anchor_path_keys(config, PATH_CONFIG_KEYS)
    hyperparams: SimpleNamespace = load_hyperparams(
        str(script_dir / HYPERPARAMS_TOML)
    )
    apply_overrides(config, hyperparams, args)
    set_reproducibility_seed(int(hyperparams.SEED))
    log: SimpleNamespace = setup_logger(config)
    summary: dict[str, Any] = run_report(
        config=config,
        hyperparams=hyperparams,
        log=log,
        requested_components=args.components,
    )
    print(f"Full-pipeline report: {config.FULL_PIPELINE_OUTPUT_DIRECTORY}")
    for component in summary["components"]:
        print(f"  {component['component']}: {component['figure']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
