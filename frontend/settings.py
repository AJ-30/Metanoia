"""Path and settings handling for the frontend stage.

The evaluator anchors its own config keys; the frontend needs a slightly different
set (uploads and outputs instead of splits and telemetry), so it declares its own
list here rather than borrowing one that would silently skip its keys. Anchoring
matters for more than tidiness: Flask resolves a *relative* ``send_from_directory``
path against the app's root path, which is this package's directory, so an
unanchored output directory 404s in the browser even though the file is on disk.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from evaluator.ontology import ComponentOntology, resolve_component_ontology
from evaluator.settings import load_detector_ontology
from helpers.config import load_config
from helpers.hyperparams import load_hyperparams
from helpers.logger import setup_logger
from mask_diffusion.inference import MaskDiffusionSampler, default_device

# This stage's path keys, anchored at the repository root.
PATH_CONFIG_KEYS: tuple[str, ...] = (
    "DEBUG_LOG_FILE_PATH",
    "UPLOAD_DIRECTORY",
    "OUTPUT_DIRECTORY",
    "MASK_DIFFUSION_CHECKPOINT_PATH",
    "MASK_COLOR_INDEX_DIR",
    "YOLO_WEIGHTS_PATH",
    "YOLO_HYPERPARAMETERS_PATH",
)

SCRIPT_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SCRIPT_DIR.parent

# One cached context per run mode, guarded by a lock because the first upload
# may arrive while another request is still building the context.
_CONTEXTS: dict[str, "FrontendContext"] = {}
_LOCK: threading.Lock = threading.Lock()


@dataclass(frozen=True)
class FrontendContext:
    """The loaded stage: sampler, detector, ontology and settings.

    Attributes:
        config (SimpleNamespace): Anchored paths and component identity.
        hyperparams (SimpleNamespace): Stage numbers.
        sampler (MaskDiffusionSampler): Frozen sampler of the configured run.
        ontology (ComponentOntology): The missing component, resolved.
        model (Any): Loaded Ultralytics detector.
        class_names (list[str]): Detector class vocabulary.
    """

    config: SimpleNamespace
    hyperparams: SimpleNamespace
    sampler: MaskDiffusionSampler
    ontology: ComponentOntology
    model: Any
    class_names: list[str]


def anchor_path_keys(
    config: SimpleNamespace, keys: tuple[str, ...] = PATH_CONFIG_KEYS
) -> None:
    """Rewrite relative path keys of ``config`` to absolute paths, in place.

    Args:
        config (SimpleNamespace): Parsed config object to mutate.
        keys (tuple[str, ...]): Config keys to anchor; missing keys are skipped.
    """
    for key in keys:
        if not hasattr(config, key):
            continue
        path: Path = Path(str(getattr(config, key)))
        setattr(config, key, str(path if path.is_absolute() else ROOT_DIR / path))
    # The project logger writes to this file on setup, so its directory must
    # exist even on a fresh clone.
    Path(str(config.DEBUG_LOG_FILE_PATH)).parent.mkdir(parents=True, exist_ok=True)


def _build_context(
    config: SimpleNamespace, hyperparams: SimpleNamespace, log: SimpleNamespace
) -> FrontendContext:
    """Load the sampler, the detector and the resolved ontology.

    Args:
        config (SimpleNamespace): Config with paths and component identity.
        hyperparams (SimpleNamespace): Stage numbers.
        log (SimpleNamespace): Configured project logger.

    Returns:
        FrontendContext: The ready context.

    Raises:
        FileNotFoundError: If a checkpoint, the weights or a color index is absent.
        ValueError: On any dimension or ontology disagreement.
    """
    device: torch.device = default_device(str(config.DEVICE), log)
    class_names, aliases = load_detector_ontology(
        str(config.YOLO_HYPERPARAMETERS_PATH)
    )
    sampler = MaskDiffusionSampler(
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
    )
    ontology = resolve_component_ontology(
        name=str(config.MISSING_COMPONENT_CLASS_NAME),
        group_prefix=str(config.TARGET_CLASS_GROUP_PREFIX),
        mask_channel=sampler.missing_channel,
        class_names=class_names,
        aliases=aliases,
        label_names_table=dict(getattr(hyperparams, "COMPONENT_LABEL_NAMES", {})),
    )
    from evaluator.yolo_eval import validate_detector_names
    from ultralytics import YOLO  # lazy: only needed when an upload arrives

    weights_path = Path(str(config.YOLO_WEIGHTS_PATH))
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Detector weights not found: {weights_path}. Fix YOLO_WEIGHTS_PATH "
            "in frontend/config.toml."
        )
    model = YOLO(str(weights_path))
    # The ontology must be the detector's own, otherwise the boxes it draws would
    # carry the wrong class names.
    validate_detector_names(model, class_names)
    log.info(
        f"Frontend ready: component '{ontology.name}' from "
        f"{sampler.checkpoint_path}, detector {weights_path} on {device}."
    )
    return FrontendContext(
        config=config,
        hyperparams=hyperparams,
        sampler=sampler,
        ontology=ontology,
        model=model,
        class_names=class_names,
    )


def get_context(mode: str = "debug") -> FrontendContext:
    """Return the cached context for a run mode, building it on first use.

    Args:
        mode (str): Runtime configuration mode (``debug`` or ``prod``), selecting
            the matching table of ``frontend/config.toml``.

    Returns:
        FrontendContext: The cached, read-only context of this process.

    Raises:
        FileNotFoundError: If a checkpoint, the weights or a color index is absent.
        ValueError: On any dimension or ontology disagreement.
    """
    cached: FrontendContext | None = _CONTEXTS.get(mode)
    if cached is not None:
        return cached
    with _LOCK:
        if mode in _CONTEXTS:  # another request built it while we waited
            return _CONTEXTS[mode]
        config: SimpleNamespace = load_config(
            mode, str(SCRIPT_DIR / "config.toml")
        )
        anchor_path_keys(config, PATH_CONFIG_KEYS)
        hyperparams: SimpleNamespace = load_hyperparams(
            str(SCRIPT_DIR / "hyperparameters.toml")
        )
        log: SimpleNamespace = setup_logger(config)
        _CONTEXTS[mode] = _build_context(config, hyperparams, log)
    return _CONTEXTS[mode]
