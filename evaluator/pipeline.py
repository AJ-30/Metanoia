"""The evaluation pipeline: generate, composite, detect, score.

The order of operations is fixed by the question being answered, so it is kept
linear and explicit here:

1. **resolve** the missing component in the mask, detector and dataset-label
   vocabularies, and load the frozen sampler plus the detector;
2. **index** the test split (every sample, negatives included, so false
   positives are measured and not assumed away);
3. **stream** the split in blocks: encode both conditioning latents once per
   block, then per draw run the full reverse chain, decode it into a class-id
   mask and a missing-component probability map, and build the erased ground
   truth plus the generated component composite; the probability maps are
   averaged across draws and thresholded into the primary ensemble composite;
4. **score** every (sample, result) - the dataset's ground-truth box against the
   detector's box, the final detected box against the layout's other furniture,
   and the generated channel against the true channel - for each draw and for
   the ensemble;
5. **validate** the detector over the whole evaluated split, per draw and for the
   ensemble, restricted to the missing component's class;
6. **visualize** and **report**.

Only one block's masks are ever in memory, so the full 10k test split runs with
the same code path as the smoke pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from evaluator.box_metrics import (
    BoxLoss,
    MatchOutcome,
    PredictedBox,
    match_boxes,
    pixel_boxes_to_predictions,
)
from evaluator.channel_metrics import channel_scores
from evaluator.compositor import (
    build_composite,
    build_conditioned_overlay,
    build_ground_truth_overlay,
)
from evaluator.dataset_gt import (
    Box,
    boxes_to_yolo_lines,
    read_component_boxes,
    read_other_component_boxes,
)
from evaluator.ontology import ComponentOntology, resolve_component_ontology
from evaluator.overlap_metrics import overlap_for_final_prediction
from evaluator.report import format_report
from evaluator.results import (
    ENSEMBLE_RESULT,
    DrawAccumulator,
    SampleRecord,
    aggregate_per_draw,
    result_slug,
)
from evaluator.settings import load_detector_ontology
from evaluator.telemetry import SampleTelemetry, write_report_table, write_summary
from evaluator.visualize import (
    annotate_composite,
    annotate_record_path,
    build_annotated_montage,
    build_compare_montage,
    build_sample_strip,
    choose_visualized_stems,
    rank_samples,
    relative_path,
    render_component_image,
)
from evaluator.yolo_eval import (
    PixelBox,
    detector_class_names,
    prepare_eval_dataset,
    predict_component_boxes,
    run_detector_validation,
    write_eval_image,
    write_eval_labels,
)
from mask_diffusion.inference import (
    MaskDiffusionSampler,
    default_device,
    threshold_probability_maps,
)
from mask_diffusion.runner import set_reproducibility_seed
from YOLO.dataset import SampleFolder, list_sample_folders

# Directory (under the run root) holding the side-by-side comparison strips.
COMPARE_DIRECTORY: str = "compare"
# Directory holding the annotated copies of the evaluated images.
ANNOTATED_DIRECTORY: str = "annotated"


@dataclass(frozen=True)
class _EvalDatasets:
    """Ultralytics datasets written by one run.

    Attributes:
        image_dirs (list[Path]): Evaluated images, one directory per draw.
        label_dirs (list[Path]): Matching label files, one directory per draw.
        yaml_paths (list[str]): ``data.yaml`` path per draw.
        ensemble_image_dir (Path): Evaluated images of the ensemble result.
        ensemble_label_dir (Path): Matching label files of the ensemble result.
        ensemble_yaml_path (str): ``data.yaml`` path of the ensemble result.
    """

    image_dirs: list[Path]
    label_dirs: list[Path]
    yaml_paths: list[str]
    ensemble_image_dir: Path
    ensemble_label_dir: Path
    ensemble_yaml_path: str


@dataclass
class _BlockState:
    """Everything the block loop needs, so its signatures stay short.

    Attributes:
        samples (list[SampleFolder]): Every evaluated sample, in order.
        sampler (MaskDiffusionSampler): Frozen sampler of the run under test.
        ontology (ComponentOntology): The missing component, resolved.
        hyperparams (SimpleNamespace): Stage knobs.
        model (Any): Detector under test.
        image_dirs (list[Path]): Destination of the evaluated images, one per
            draw, so each draw's detector metrics are computed over that draw's
            own images only.
        label_dirs (list[Path]): Destination of the evaluated labels, one per
            draw.
        ensemble_image_dir (Path): Destination of the ensemble's evaluated images.
        ensemble_label_dir (Path): Destination of the ensemble's labels.
        telemetry (SampleTelemetry): Per-sample CSV writer.
        accumulators (dict[int, DrawAccumulator]): One accumulator per draw,
            plus :data:`ENSEMBLE_RESULT` for the ensemble.
        records (dict[str, SampleRecord]): Per-sample records for visualization.
        output_root (Path): Run output root.
        label_filename (str): Dataset label filename inside each sample folder.
        class_names (list[str]): Detector class vocabulary.
        device_str (str): Device string passed to Ultralytics.
        viz_stems (set[str]): Samples that also get a comparison strip.
        other_ground_truth (dict[str, list[Box]]): Cache of each sample's
            non-target ground-truth boxes, read once and reused by every draw
            and the ensemble.
    """

    samples: list[SampleFolder]
    sampler: MaskDiffusionSampler
    ontology: ComponentOntology
    hyperparams: SimpleNamespace
    model: Any
    image_dirs: list[Path]
    label_dirs: list[Path]
    ensemble_image_dir: Path
    ensemble_label_dir: Path
    telemetry: SampleTelemetry
    accumulators: dict[int, DrawAccumulator]
    records: dict[str, SampleRecord]
    output_root: Path
    label_filename: str
    class_names: list[str]
    device_str: str
    viz_stems: set[str] = field(default_factory=set)
    other_ground_truth: dict[str, list[Box]] = field(default_factory=dict)


def run_evaluation(
    args: SimpleNamespace,
    config: SimpleNamespace,
    hyperparams: SimpleNamespace,
    log: SimpleNamespace,
) -> dict[str, Any]:
    """Run the whole evaluation and write every artifact.

    Args:
        args (SimpleNamespace): Parsed command-line options (overrides).
        config (SimpleNamespace): Paths, device and component identity.
        hyperparams (SimpleNamespace): Every tunable number of this stage.
        log (SimpleNamespace): Configured project logger.

    Returns:
        dict[str, Any]: The summary that was written to
            ``evaluation_summary.json``.
    """
    started: float = time.time()
    _validate_ensemble_settings(hyperparams)
    device = default_device(str(config.DEVICE), log)
    set_reproducibility_seed(int(hyperparams.SEED))

    # The selected variant decides WHICH detector is loaded, so the detector's
    # own class names are the authority on what a class id means. Aliases still
    # come from the YOLO stage's ontology because the mask vocabulary spells
    # some components differently (dining_set vs Dining_table).
    model: Any
    class_names: list[str]
    model, class_names = _load_detector(config, log)
    _, aliases = load_detector_ontology(str(config.YOLO_HYPERPARAMETERS_PATH))

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
        fallback_color_dir=Path(str(config.TEST_IMAGES_DIR)).parent,
    )
    log.info(
        "Sampling with the frozen VAEs the checkpoint recorded: "
        + ", ".join(
            f"{key}={sampler.checkpoint[f'{key}_checkpoint']}"
            for key in ("floorplan", "partial", "target")
        )
        + "."
    )
    ontology: ComponentOntology = resolve_component_ontology(
        name=str(config.MISSING_COMPONENT_CLASS_NAME),
        group_prefix=str(config.TARGET_CLASS_GROUP_PREFIX),
        mask_channel=sampler.missing_channel,
        class_names=class_names,
        aliases=aliases,
        label_names_table=dict(getattr(hyperparams, "COMPONENT_LABEL_NAMES", {})),
    )
    log.info(
        f"Missing component '{ontology.name}': mask channel {ontology.mask_channel}, "
        f"detector class {ontology.yolo_class_id} "
        f"('{class_names[ontology.yolo_class_id]}'), dataset label names "
        f"{list(ontology.label_names)}."
    )

    samples: list[SampleFolder] = _index_samples(config, hyperparams)
    output_root: Path = _output_root(config, args)
    draws: int = int(hyperparams.DRAWS_PER_SAMPLE)
    datasets: _EvalDatasets = _prepare_datasets(output_root, draws, class_names)
    telemetry: SampleTelemetry = SampleTelemetry(
        str(config.TELEMETRY_LOG_DIRECTORY), config
    )
    accumulators: dict[int, DrawAccumulator] = {
        draw: DrawAccumulator() for draw in range(draws)
    }
    accumulators[ENSEMBLE_RESULT] = DrawAccumulator()
    block_state: _BlockState = _BlockState(
        samples=samples,
        sampler=sampler,
        ontology=ontology,
        hyperparams=hyperparams,
        model=model,
        image_dirs=datasets.image_dirs,
        label_dirs=datasets.label_dirs,
        ensemble_image_dir=datasets.ensemble_image_dir,
        ensemble_label_dir=datasets.ensemble_label_dir,
        telemetry=telemetry,
        accumulators=accumulators,
        records={},
        output_root=output_root,
        label_filename=str(config.GROUND_TRUTH_LABEL_FILE),
        class_names=class_names,
        device_str=str(config.DEVICE),
        viz_stems=set(
            choose_visualized_stems(
                [sample.stem for sample in samples],
                int(hyperparams.VIZ_SAMPLES),
                int(hyperparams.SEED),
            )
        ),
    )

    _run_blocks(block_state, log)

    detection, ensemble_detection = _validate_detector(
        model=model,
        datasets=datasets,
        ontology=ontology,
        hyperparams=hyperparams,
        device_str=str(config.DEVICE),
        log=log,
        skip=bool(getattr(args, "skip_yolo", False)),
    )

    figures: list[str] = _build_figures(block_state, class_names)
    # Publish the figures to the stage's outputs root (the directory the shared
    # dashboard serves), named with the run id, so several evaluation runs can
    # coexist there the way the VAE stage's grids do.
    published: list[str] = _publish_figures(
        figures,
        output_root,
        Path(str(config.OUTPUT_SAMPLES_DIRECTORY)),
        str(config.runtime.RUN_ID),
        log,
    )
    summary: dict[str, Any] = _build_summary(
        config=config,
        hyperparams=hyperparams,
        ontology=ontology,
        class_names=class_names,
        accumulators=accumulators,
        detection=detection,
        ensemble_detection=ensemble_detection,
        sample_count=len(samples),
        output_root=output_root,
        figures=figures,
        published_figures=published,
        elapsed=time.time() - started,
        sampler=sampler,
    )
    write_summary(output_root / "evaluation_summary.json", summary)
    write_report_table(output_root / "evaluation_report.txt", format_report(summary))
    log.info(
        f"Evaluation finished in {time.time() - started:.1f}s; artifacts in {output_root}"
    )
    return summary


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def _load_detector(
    config: SimpleNamespace, log: SimpleNamespace
) -> tuple[Any, list[str]]:
    """Load the selected detector and read the vocabulary it was trained with.

    Args:
        config (SimpleNamespace): Config whose ``YOLO_WEIGHTS_PATH`` was set by
            the component selection (or by ``--weights``).
        log (SimpleNamespace): Configured project logger.

    Returns:
        tuple[Any, list[str]]: The loaded Ultralytics detector and its ordered
            class names, index == class id.

    Raises:
        FileNotFoundError: If the weights file is absent.
        ValueError: If the detector exposes no class vocabulary.
    """
    from ultralytics import YOLO  # lazy import so unit tests run without it

    weights_path: Path = Path(str(config.YOLO_WEIGHTS_PATH))
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Detector weights not found: {weights_path}. Select a component "
            "declared in components.toml, or pass --weights explicitly."
        )
    model: Any = YOLO(str(weights_path))
    class_names: list[str] = detector_class_names(model)
    log.info(
        f"Loaded detector weights from {weights_path} with classes {class_names}"
    )
    return model, class_names


def _index_samples(
    config: SimpleNamespace, hyperparams: SimpleNamespace
) -> list[SampleFolder]:
    """Index the test split the evaluation runs over.

    Args:
        config (SimpleNamespace): Config with the split path and mask filenames.
        hyperparams (SimpleNamespace): Stage knobs (sample cap).

    Returns:
        list[SampleFolder]: Every evaluated sample folder, sorted by name.
    """
    return list_sample_folders(
        split_dir=str(config.TEST_IMAGES_DIR),
        cond_mask_file=str(config.FLOORPLAN_MASK_FILE),
        target_mask_file=str(config.FURNITURE_MASK_FILE),
        cap=int(hyperparams.SAMPLE_CAP),
    )


def _validate_ensemble_settings(hyperparams: SimpleNamespace) -> None:
    """Fail before model loading when the ensemble configuration is unusable.

    Args:
        hyperparams (SimpleNamespace): Stage knobs read from the TOML.

    Raises:
        ValueError: If there is no draw to ensemble, or the threshold cannot
            split a probability map into a mask.
    """
    draws: int = int(hyperparams.DRAWS_PER_SAMPLE)
    if draws < 1:
        raise ValueError(f"DRAWS_PER_SAMPLE must be >= 1, got {draws}.")
    threshold: float = float(hyperparams.ENSEMBLE_PROBABILITY_THRESHOLD)
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "ENSEMBLE_PROBABILITY_THRESHOLD must be strictly between 0 and 1, "
            f"got {threshold}."
        )


def _prepare_datasets(
    output_root: Path, draws: int, class_names: list[str]
) -> _EvalDatasets:
    """Prepare one Ultralytics dataset per draw plus one for the ensemble.

    Each result's images and labels live in their own ``<root>/dataset_*``
    directory and are cleared first, so detector validation of draw 0 can never
    see draw 1 and the ensemble cannot see a single chain's images.

    Args:
        output_root (Path): Run output root.
        draws (int): Number of per-draw datasets to prepare.
        class_names (list[str]): Ordered detector vocabulary.

    Returns:
        _EvalDatasets: Per-draw and ensemble image/label directories and
            ``data.yaml`` paths.
    """
    image_dirs: list[Path] = []
    label_dirs: list[Path] = []
    yaml_paths: list[str] = []
    for draw in range(draws):
        image_dir, label_dir, yaml_path = prepare_eval_dataset(
            output_root / f"dataset_draw_{draw}", class_names
        )
        image_dirs.append(image_dir)
        label_dirs.append(label_dir)
        yaml_paths.append(yaml_path)
    ensemble_image_dir, ensemble_label_dir, ensemble_yaml_path = prepare_eval_dataset(
        output_root / "dataset_ensemble", class_names
    )
    return _EvalDatasets(
        image_dirs=image_dirs,
        label_dirs=label_dirs,
        yaml_paths=yaml_paths,
        ensemble_image_dir=ensemble_image_dir,
        ensemble_label_dir=ensemble_label_dir,
        ensemble_yaml_path=ensemble_yaml_path,
    )


def _output_root(config: SimpleNamespace, args: SimpleNamespace) -> Path:
    """Resolve and create this run's output directory.

    Args:
        config (SimpleNamespace): Config with the output directory.
        args (SimpleNamespace): Optional ``--output-dir`` override.

    Returns:
        Path: The created output root.
    """
    if getattr(args, "output_dir", None):
        root: Path = Path(str(args.output_dir)).resolve()
    else:
        root = (
            Path(str(config.OUTPUT_SAMPLES_DIRECTORY))
            / f"eval_{config.runtime.RUN_ID}"
        )
    (root / "dataset").mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Block streaming
# ---------------------------------------------------------------------------
def _run_blocks(block_state: _BlockState, log: SimpleNamespace) -> None:
    """Stream the split in blocks, scoring every (sample, result) pair.

    Args:
        block_state (_BlockState): Shared state of the block loop.
        log (SimpleNamespace): Configured project logger.
    """
    block_size: int = max(1, int(block_state.hyperparams.BLOCK_SIZE))
    draws: int = int(block_state.hyperparams.DRAWS_PER_SAMPLE)
    total: int = len(block_state.samples)
    log.info(
        f"Evaluating {total} test samples x {draws} draws plus the ensemble "
        f"(block {block_size})."
    )
    for start in range(0, total, block_size):
        stop: int = min(start + block_size, total)
        _evaluate_block(block_state, start, stop)
        log.info(f"Scored samples {start + 1}..{stop} of {total}.")


def _evaluate_block(block_state: _BlockState, start: int, stop: int) -> None:
    """Encode, generate, composite, detect and score one block of samples.

    Each draw is generated once and decoded into both the per-draw argmax mask
    and the missing-component probability map. The probability maps are summed
    across draws, averaged, and thresholded into the ensemble mask, so the
    expensive reverse chains are never repeated for the ensemble.

    Args:
        block_state (_BlockState): Shared state of the block loop.
        start (int): First sample index of the block.
        stop (int): Index past the last sample of the block.
    """
    sampler: MaskDiffusionSampler = block_state.sampler
    hyperparams: SimpleNamespace = block_state.hyperparams
    draws: int = int(hyperparams.DRAWS_PER_SAMPLE)
    block_dirs: list[Path] = [sample.src_dir for sample in block_state.samples[start:stop]]
    conditions: dict[str, Any] = sampler.normalize_conditions(
        sampler.encode_conditions(block_dirs, int(hyperparams.ENCODE_BATCH_SIZE))
    )
    ensemble_probability_sum: Any = None
    for draw in range(draws):
        # One chain per sample. The sampler seeds every chunk from this seed and
        # from the chain offset, so a block boundary never changes which noise
        # lands on which sample.
        outputs: Any = sampler.generate_channel_outputs(
            conditions,
            num_draws=1,
            seed=int(hyperparams.SEED) + draw * int(hyperparams.DRAW_SEED_STRIDE),
            chunk_size=int(hyperparams.GENERATION_CHUNK),
            chain_offset=start,
        )
        draw_mask: Any = outputs.ids == sampler.missing_channel
        _score_result(
            block_state,
            start,
            stop,
            conditions,
            masks=draw_mask,
            probabilities=outputs.probabilities,
            result_key=draw,
        )
        ensemble_probability_sum = (
            outputs.probabilities
            if ensemble_probability_sum is None
            else ensemble_probability_sum + outputs.probabilities
        )
    if ensemble_probability_sum is None:
        raise ValueError("DRAWS_PER_SAMPLE must be >= 1 to build an ensemble.")
    ensemble_probability: Any = ensemble_probability_sum / draws
    ensemble_mask: Any = threshold_probability_maps(
        ensemble_probability, float(hyperparams.ENSEMBLE_PROBABILITY_THRESHOLD)
    )
    _score_result(
        block_state,
        start,
        stop,
        conditions,
        masks=ensemble_mask,
        probabilities=ensemble_probability,
        result_key=ENSEMBLE_RESULT,
    )


def _other_ground_truth(block_state: _BlockState, sample: SampleFolder) -> list[Box]:
    """Read and cache the non-target furniture boxes of one sample.

    Args:
        block_state (_BlockState): Shared state of the block loop.
        sample (SampleFolder): Sample whose label file is read.

    Returns:
        list[Box]: Ground-truth boxes of every named object that is not the
            missing component.
    """
    cached: list[Box] | None = block_state.other_ground_truth.get(sample.stem)
    if cached is not None:
        return cached
    boxes: list[Box] = read_other_component_boxes(
        sample.src_dir,
        block_state.label_filename,
        block_state.ontology.label_names,
    )
    block_state.other_ground_truth[sample.stem] = boxes
    return boxes


def _score_result(
    block_state: _BlockState,
    start: int,
    stop: int,
    conditions: dict[str, Any],
    masks: Any,
    probabilities: Any,
    result_key: int,
) -> None:
    """Build, detect and score every sample of one result (draw or ensemble).

    Args:
        block_state (_BlockState): Shared state of the block loop.
        start (int): First sample index of the block.
        stop (int): Index past the last sample of the block.
        conditions (dict[str, Any]): Normalized conditions of the block.
        masks (Any): Component masks of the block, ``(n, H, W)`` bool.
        probabilities (Any): Missing-component probability maps of the block,
            ``(n, H, W)`` float.
        result_key (int): Draw index, or :data:`ENSEMBLE_RESULT` for the ensemble.
    """
    hyperparams: SimpleNamespace = block_state.hyperparams
    sampler: MaskDiffusionSampler = block_state.sampler
    image_size: int = sampler.image_size
    for offset, index in enumerate(range(start, stop)):
        sample: SampleFolder = block_state.samples[index]
        ground_truth: list[Box] = read_component_boxes(
            sample.src_dir, block_state.label_filename, block_state.ontology.label_names
        )
        other_ground_truth: list[Box] = _other_ground_truth(block_state, sample)
        generated: Any = masks[offset : offset + 1]
        composite: Any = build_composite(
            conditions["floorplan_ids"][offset : offset + 1],
            conditions["furniture_ids"][offset : offset + 1],
            generated,
            sampler,
            float(hyperparams.COMPOSITE_OVERLAY_ALPHA),
            float(hyperparams.GENERATED_OVERLAY_ALPHA),
        )
        image_path: Path = _write_evaluated_image(
            block_state, sample, result_key, composite, ground_truth
        )
        predicted_pixels: list[PixelBox] = _predict_component(block_state, image_path)
        predictions: list[PredictedBox] = pixel_boxes_to_predictions(
            predicted_pixels, block_state.ontology.yolo_class_id, image_size
        )
        outcome: MatchOutcome = match_boxes(
            ground_truth,
            predictions,
            float(hyperparams.BOX_MATCH_IOU),
            image_size,
        )
        overlap: float | None = overlap_for_final_prediction(
            predictions, other_ground_truth
        )
        scores: tuple[float, float] = channel_scores(
            generated, conditions["furniture_ids"][offset : offset + 1] == sampler.missing_channel
        )
        _record_sample(
            block_state,
            sample=sample,
            result_key=result_key,
            ground_truth=ground_truth,
            other_ground_truth=other_ground_truth,
            overlap=overlap,
            outcome=outcome,
            scores=scores,
            image_path=image_path,
            predictions=predictions,
            image_size=image_size,
            generated_pixels=int(generated.sum()),
            probability_mean=float(probabilities[offset].mean()),
        )
        if sample.stem in block_state.viz_stems and result_key == ENSEMBLE_RESULT:
            _write_comparison_strip(
                block_state,
                sample=sample,
                conditions=conditions,
                offset=offset,
                generated=generated,
                predicted_pixels=predicted_pixels,
                image_path=image_path,
            )


def _result_directories(
    block_state: _BlockState, result_key: int
) -> tuple[Path, Path]:
    """Pick the image and label directory of one result.

    Args:
        block_state (_BlockState): Shared state of the block loop.
        result_key (int): Draw index, or :data:`ENSEMBLE_RESULT`.

    Returns:
        tuple[Path, Path]: Image directory and label directory.
    """
    if result_key == ENSEMBLE_RESULT:
        return block_state.ensemble_image_dir, block_state.ensemble_label_dir
    return block_state.image_dirs[result_key], block_state.label_dirs[result_key]


def _write_evaluated_image(
    block_state: _BlockState,
    sample: SampleFolder,
    result_key: int,
    composite: Any,
    ground_truth: list[Box],
) -> Path:
    """Write one evaluated image and its label file.

    The image name carries the result key, so every draw and the ensemble stay
    addressable and each label file matches its image one to one.

    Args:
        block_state (_BlockState): Shared state of the block loop.
        sample (SampleFolder): Sample being written.
        result_key (int): Draw index, or :data:`ENSEMBLE_RESULT`.
        composite (Any): The composite image to write.
        ground_truth (list[Box]): Dataset ground-truth boxes of the component.

    Returns:
        Path: Path of the written image.
    """
    image_dir, label_dir = _result_directories(block_state, result_key)
    stem: str = f"{sample.stem}__{result_slug(result_key)}"
    image_path: Path = image_dir / f"{stem}.png"
    label_path: Path = label_dir / f"{stem}.txt"
    write_eval_image(
        composite, image_path, int(block_state.hyperparams.PNG_COMPRESS_LEVEL)
    )
    write_eval_labels(
        label_path, boxes_to_yolo_lines(ground_truth, block_state.ontology.yolo_class_id)
    )
    return image_path


def _predict_component(block_state: _BlockState, image_path: Path) -> list[PixelBox]:
    """Run the detector on one evaluated image, keeping only the component.

    Args:
        block_state (_BlockState): Shared state of the block loop.
        image_path (Path): Evaluated composite image.

    Returns:
        list[PixelBox]: Pixel-space boxes of the missing component.
    """
    return predict_component_boxes(
        model=block_state.model,
        image_path=image_path,
        confidence=float(block_state.hyperparams.YOLO_CONF),
        image_size=int(block_state.hyperparams.YOLO_IMGSZ),
        max_detections=int(block_state.hyperparams.YOLO_MAX_BOXES),
        class_id=block_state.ontology.yolo_class_id,
    )


def _record_sample(
    block_state: _BlockState,
    sample: SampleFolder,
    result_key: int,
    ground_truth: list[Box],
    other_ground_truth: list[Box],
    overlap: float | None,
    outcome: MatchOutcome,
    scores: tuple[float, float],
    image_path: Path,
    predictions: list[PredictedBox],
    image_size: int,
    generated_pixels: int,
    probability_mean: float,
) -> None:
    """Fold one (sample, result) measurement into telemetry and the record.

    Args:
        block_state (_BlockState): Shared state of the block loop.
        sample (SampleFolder): Sample being recorded.
        result_key (int): Draw index, or :data:`ENSEMBLE_RESULT`.
        ground_truth (list[Box]): Dataset ground-truth boxes of the missing
            component.
        other_ground_truth (list[Box]): Dataset ground-truth boxes of every
            other named object in the layout.
        overlap (float | None): Fraction of the highest-confidence target
            prediction covered by other furniture, or ``None`` when no target
            box was predicted.
        outcome (MatchOutcome): Matched/missed/false-positive outcome.
        scores (tuple[float, float]): Channel ``(dice, iou)``.
        image_path (Path): Written composite image.
        predictions (list[PredictedBox]): Detector predictions.
        image_size (int): Render domain side length.
        generated_pixels (int): Number of pixels in the generated mask.
        probability_mean (float): Mean missing-component probability.
    """
    matched: list[BoxLoss] = [
        match.loss for match in outcome.matches if match.loss is not None
    ]
    # The row describes ONE match: the best one. Taking max IoU, min L1 and min
    # centre distance from three different matches would mix three different
    # placements into one row for a sample with more than one instance.
    best: BoxLoss | None = (
        max(matched, key=lambda loss: loss.iou) if matched else None
    )
    block_state.accumulators[result_key].add(
        ground_truth, outcome, scores, overlap
    )
    block_state.telemetry.log(
        stem=sample.stem,
        draw=result_key,
        gt_boxes=len(ground_truth),
        other_gt_boxes=len(other_ground_truth),
        pred_boxes=len(predictions),
        matched_boxes=len(matched),
        missed_boxes=sum(1 for match in outcome.matches if match.loss is None),
        false_positives=len(outcome.false_positives),
        overlap=overlap,
        best_iou=best.iou if best is not None else 0.0,
        best_l1_norm=best.l1_norm if best is not None else 0.0,
        best_center_distance=best.center_distance if best is not None else 0.0,
        channel_dice=scores[0],
        channel_iou=scores[1],
        generated_pixels=generated_pixels,
        probability_mean=probability_mean,
    )
    record: SampleRecord = block_state.records.setdefault(
        sample.stem, SampleRecord(stem=sample.stem)
    )
    record.ground_truth = ground_truth
    record.image_paths[result_key] = image_path
    record.predictions[result_key] = _to_pixel_predictions(predictions, image_size)
    record.losses[result_key] = matched


def _to_pixel_predictions(
    predictions: list[PredictedBox], image_size: int
) -> list[PixelBox]:
    """Convert normalized predictions back to pixel tuples for annotation.

    Args:
        predictions (list[PredictedBox]): Normalized predictions.
        image_size (int): Render domain side length.

    Returns:
        list[PixelBox]: ``(x1, y1, x2, y2, class_id, confidence)`` in pixels, the
            class id being ``-1`` because the annotation only draws one class.
    """
    pixels: list[PixelBox] = []
    for prediction in predictions:
        x1, y1, x2, y2 = prediction.box.to_pixels(image_size)
        pixels.append((x1, y1, x2, y2, -1, prediction.confidence))
    return pixels


def _write_comparison_strip(
    block_state: _BlockState,
    sample: SampleFolder,
    conditions: dict[str, Any],
    offset: int,
    generated: Any,
    predicted_pixels: list[PixelBox],
    image_path: Path,
) -> None:
    """Write one sample's five-panel comparison strip.

    Args:
        block_state (_BlockState): Shared state of the block loop.
        sample (SampleFolder): Sample being visualized.
        conditions (dict[str, Any]): Normalized conditions of the block.
        offset (int): Sample offset inside the block.
        generated (Any): Generated component mask of this sample.
        predicted_pixels (list[PixelBox]): Predictions for the annotated panel.
        image_path (Path): The ensemble composite the strip annotates.
    """
    sampler: MaskDiffusionSampler = block_state.sampler
    hyperparams: SimpleNamespace = block_state.hyperparams
    floorplan_ids = conditions["floorplan_ids"][offset : offset + 1]
    furniture_ids = conditions["furniture_ids"][offset : offset + 1]
    with Image.open(image_path) as opened:
        composite: Image.Image = opened.convert("RGB")
    ground_truth: list[Box] = read_component_boxes(
        sample.src_dir, block_state.label_filename, block_state.ontology.label_names
    )
    annotated: Any = annotate_composite(
        composite,
        ground_truth,
        predicted_pixels,
        block_state.class_names,
        block_state.ontology.yolo_class_id,
    )
    strip: Any = build_sample_strip(
        ground_truth_overlay=build_ground_truth_overlay(
            floorplan_ids,
            furniture_ids,
            sampler,
            float(hyperparams.COMPOSITE_OVERLAY_ALPHA),
        ),
        conditioned_overlay=build_conditioned_overlay(
            floorplan_ids,
            furniture_ids,
            sampler,
            float(hyperparams.COMPOSITE_OVERLAY_ALPHA),
        ),
        ground_truth_component=render_component_image(
            furniture_ids == sampler.missing_channel, sampler
        ),
        generated_component=render_component_image(generated, sampler),
        annotated_composite=annotated,
    )
    compare_dir: Path = block_state.output_root / COMPARE_DIRECTORY
    compare_dir.mkdir(parents=True, exist_ok=True)
    strip.save(str(compare_dir / f"{sample.stem}.png"))


def _validate_detector(
    model: Any,
    datasets: _EvalDatasets,
    ontology: ComponentOntology,
    hyperparams: SimpleNamespace,
    device_str: str,
    log: SimpleNamespace,
    skip: bool,
) -> tuple[list[dict[str, float] | None], dict[str, float] | None]:
    """Validate the detector per draw and on the ensemble, one class only.

    Every result owns a dataset directory. Sharing one would validate the union
    of the draws and make every draw's metrics identical, which is precisely the
    spread this stage exists to measure.

    Args:
        model (Any): Detector under test.
        datasets (_EvalDatasets): Per-draw and ensemble datasets.
        ontology (ComponentOntology): The missing component.
        hyperparams (SimpleNamespace): Stage knobs.
        device_str (str): Device string for Ultralytics.
        log (SimpleNamespace): Configured project logger.
        skip (bool): When true, no validation runs (channel metrics only).

    Returns:
        tuple[list[dict[str, float] | None], dict[str, float] | None]: One
            metrics dict per draw plus the ensemble's metrics, each ``None`` when
            validation was skipped.
    """
    draws: int = int(hyperparams.DRAWS_PER_SAMPLE)
    if skip:
        log.warning("--skip-yolo given: no detector metrics are computed.")
        return [None] * draws, None
    per_draw: list[dict[str, float] | None] = [
        run_detector_validation(
            model=model,
            data_yaml_path=datasets.yaml_paths[draw],
            device_str=device_str,
            image_size=int(hyperparams.YOLO_IMGSZ),
            batch_size=int(hyperparams.YOLO_BATCH),
            class_id=ontology.yolo_class_id,
            log=log,
            label=f"missing component '{ontology.name}', draw {draw}",
        )
        for draw in range(draws)
    ]
    ensemble: dict[str, float] = run_detector_validation(
        model=model,
        data_yaml_path=datasets.ensemble_yaml_path,
        device_str=device_str,
        image_size=int(hyperparams.YOLO_IMGSZ),
        batch_size=int(hyperparams.YOLO_BATCH),
        class_id=ontology.yolo_class_id,
        log=log,
        label=f"missing component '{ontology.name}', ensemble",
    )
    return per_draw, ensemble


def _build_figures(block_state: _BlockState, class_names: list[str]) -> list[str]:
    """Build every visualization the run writes.

    Three kinds of figure are produced for every result (each draw and the
    ensemble):

    * ``eval_composite_<result>.png`` - the evaluated images with the
      ground-truth box in green and the prediction in red;
    * ``eval_best_<result>`` / ``eval_worst_<result>`` - the ranked ends of the
      box-IoU distribution;
    * ``eval_compare`` - the five-panel strip of the fixed visualization subset,
      which shows the conditioning, the true component, the ensemble-generated
      component and the ensemble composite together.

    Args:
        block_state (_BlockState): Shared state of the block loop.
        class_names (list[str]): Detector class vocabulary.

    Returns:
        list[str]: Written figure paths, relative to the run root.
    """
    hyperparams: SimpleNamespace = block_state.hyperparams
    output_root: Path = block_state.output_root
    columns: int = int(hyperparams.MONTAGE_COLS)
    class_id: int = block_state.ontology.yolo_class_id
    figures: list[str] = []
    result_keys: list[int] = [
        ENSEMBLE_RESULT,
        *range(int(hyperparams.DRAWS_PER_SAMPLE)),
    ]
    for result_key in result_keys:
        slug: str = result_slug(result_key)
        annotated: list[Path] = [
            annotate_record_path(
                block_state.records[stem],
                result_key,
                class_names,
                class_id,
                output_root,
            )
            for stem in sorted(block_state.records)
            if stem in block_state.viz_stems
        ]
        montage: Path | None = build_annotated_montage(
            annotated, output_root / f"eval_composite_{slug}.png", columns
        )
        if montage is not None:
            figures.append(relative_path(montage, output_root))
        best_stems, worst_stems = rank_samples(
            _losses_by_stem(block_state, result_key),
            int(hyperparams.VIZ_RANKED_COUNT),
        )
        ranked: tuple[tuple[str, list[str]], ...] = (
            ("best", best_stems),
            ("worst", worst_stems),
        )
        for name, stems in ranked:
            ranked_paths: list[Path] = [
                annotate_record_path(
                    block_state.records[stem],
                    result_key,
                    class_names,
                    class_id,
                    output_root,
                )
                for stem in stems
            ]
            ranked_montage: Path | None = build_annotated_montage(
                ranked_paths, output_root / f"eval_{name}_{slug}.png", columns
            )
            if ranked_montage is not None:
                figures.append(relative_path(ranked_montage, output_root))
    compare_path: Path | None = build_compare_montage(
        block_state.output_root / COMPARE_DIRECTORY,
        block_state.output_root / "eval_compare.png",
        columns,
    )
    if compare_path is not None:
        figures.append(relative_path(compare_path, output_root))
    return figures


def _publish_figures(
    figures: list[str],
    output_root: Path,
    dashboard_dir: Path,
    run_id: str,
    log: SimpleNamespace,
) -> list[str]:
    """Copy the run's figures to the directory the shared dashboard serves.

    The dashboard reads one flat image directory (its ``--recon-dir``), so a
    per-run folder is invisible to it. Copying each figure there under
    ``<figure_stem>_<run_id>.png`` keeps every evaluation run's figures side by
    side, exactly the way the VAE stage's grids are named, without changing the
    run folder's own self-contained copies.

    Args:
        figures (list[str]): Figure paths relative to ``output_root``.
        output_root (Path): The run's output root.
        dashboard_dir (Path): Stage outputs directory the dashboard serves.
        run_id (str): Run id, embedded in each copied filename.
        log (SimpleNamespace): Configured project logger.

    Returns:
        list[str]: Published filenames, in the same order as ``figures``.
    """
    published: list[str] = []
    for figure in figures:
        source: Path = output_root / figure
        destination: Path = dashboard_dir / f"{source.stem}_{run_id}{source.suffix}"
        try:
            destination.write_bytes(source.read_bytes())
        except OSError as exc:
            # A figure that cannot be published is a logging problem, not a
            # reason to lose the evaluation's numbers.
            log.warning(f"Could not publish figure {figure} to {destination}: {exc}")
            published.append("")
            continue
        published.append(destination.name)
    return published


def _losses_by_stem(
    block_state: _BlockState, result_key: int
) -> dict[str, list[BoxLoss]]:
    """Collect each sample's matched-box losses for one result.

    Args:
        block_state (_BlockState): Shared state of the block loop.
        result_key (int): Draw index, or :data:`ENSEMBLE_RESULT`, to rank by.

    Returns:
        dict[str, list[BoxLoss]]: Per-sample matched losses; samples without a
            match hold an empty list, which sorts them as worst.
    """
    return {
        stem: record.losses.get(result_key, [])
        for stem, record in block_state.records.items()
    }


def _build_summary(
    config: SimpleNamespace,
    hyperparams: SimpleNamespace,
    ontology: ComponentOntology,
    class_names: list[str],
    accumulators: dict[int, DrawAccumulator],
    detection: list[dict[str, float] | None],
    ensemble_detection: dict[str, float] | None,
    sample_count: int,
    output_root: Path,
    figures: list[str],
    published_figures: list[str],
    elapsed: float,
    sampler: MaskDiffusionSampler,
) -> dict[str, Any]:
    """Assemble the report that is written as JSON and as a text table.

    Args:
        config (SimpleNamespace): Paths and component identity.
        hyperparams (SimpleNamespace): Stage knobs.
        ontology (ComponentOntology): The missing component, resolved.
        class_names (list[str]): Detector class vocabulary.
        accumulators (dict[int, DrawAccumulator]): Per-draw accumulators plus
            :data:`ENSEMBLE_RESULT`.
        detection (list[dict[str, float] | None]): Per-draw detector metrics.
        ensemble_detection (dict[str, float] | None): Ensemble detector metrics.
        sample_count (int): Number of evaluated samples.
        output_root (Path): Run output root.
        figures (list[str]): Written figure paths, relative to ``output_root``.
        published_figures (list[str]): Filenames copied to the dashboard's image
            directory, in the same order as ``figures``.
        elapsed (float): Wall-clock seconds the run took.
        sampler (MaskDiffusionSampler): The loaded sampler, for provenance.

    Returns:
        dict[str, Any]: The full summary document.
    """
    per_draw: list[dict[str, Any]] = [
        accumulators[draw].summary(detection[draw])
        for draw in range(int(hyperparams.DRAWS_PER_SAMPLE))
    ]
    return {
        "run_id": str(config.runtime.RUN_ID),
        "elapsed_seconds": elapsed,
        "configuration": {
            "mask_diffusion_checkpoint": str(config.MASK_DIFFUSION_CHECKPOINT_PATH),
            "mask_diffusion_epoch": sampler.checkpoint.get("epoch"),
            "yolo_weights": str(config.YOLO_WEIGHTS_PATH),
            "split_dir": str(config.TEST_IMAGES_DIR),
            "device": str(config.DEVICE),
            "samples_evaluated": sample_count,
            "draws_per_sample": int(hyperparams.DRAWS_PER_SAMPLE),
            "ensemble_probability_threshold": float(
                hyperparams.ENSEMBLE_PROBABILITY_THRESHOLD
            ),
            "seed": int(hyperparams.SEED),
            "block_size": int(hyperparams.BLOCK_SIZE),
            "composite_overlay_alpha": float(hyperparams.COMPOSITE_OVERLAY_ALPHA),
            "generated_overlay_alpha": float(hyperparams.GENERATED_OVERLAY_ALPHA),
            "box_match_iou": float(hyperparams.BOX_MATCH_IOU),
            "yolo_conf": float(hyperparams.YOLO_CONF),
            "yolo_imgsz": int(hyperparams.YOLO_IMGSZ),
        },
        "missing_component": {
            "name": ontology.name,
            "mask_channel": ontology.mask_channel,
            "qualified_mask_name": ontology.qualified_name,
            "detector_class_id": ontology.yolo_class_id,
            "detector_class_name": class_names[ontology.yolo_class_id],
            "dataset_label_names": list(ontology.label_names),
        },
        "ensemble": accumulators[ENSEMBLE_RESULT].summary(ensemble_detection),
        "per_draw": per_draw,
        "aggregate": aggregate_per_draw(per_draw),
        "output_root": str(output_root),
        "figures": figures,
        "published_figures": published_figures,
    }
