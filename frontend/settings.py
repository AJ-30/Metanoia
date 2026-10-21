"""Path and settings handling for the frontend stage.

The evaluator anchors its own config keys; the frontend needs a slightly different
set (uploads and outputs instead of splits and telemetry), so it declares its own
list here rather than borrowing one that would silently skip its keys. Anchoring
matters for more than tidiness: Flask resolves a *relative* ``send_from_directory``
path against the app's root path, which is this package's directory, so an
unanchored output directory 404s in the browser even though the file is on disk.

One context is built per SELECTED COMPONENT and cached, because each component
pairs its own diffusion checkpoint with its own detector. Building a context is
the expensive part (three frozen VAEs plus a detector), so a component is loaded
the first time it is chosen and reused afterwards.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from evaluator.ontology import ComponentOntology, resolve_component_ontology
from evaluator.settings import load_detector_ontology
from evaluator.variants import (
    apply_component_variant,
    component_names,
    load_component_variants,
    resolve_component_variant,
)
from evaluator.yolo_eval import detector_class_names
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

# Directory holding the stage's TOML files, and the repository root.
SCRIPT_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SCRIPT_DIR.parent

# One cached context per (run mode, component), guarded by a lock because the
# first upload may arrive while another request is still building its context.
_CONTEXTS: dict[tuple[str, str], "FrontendContext"] = {}
_LOCK: threading.Lock = threading.Lock()


@dataclass(frozen=True)
class FrontendContext:
    """The loaded stage: sampler, detector, ontology and settings.

    Attributes:
        config (SimpleNamespace): Anchored paths and the SELECTED component's
            model paths.
        hyperparams (SimpleNamespace): Stage numbers.
        sampler (MaskDiffusionSampler): Frozen sampler of the selected
            component's diffusion run.
        ontology (ComponentOntology): The selected component, resolved across
            the mask, detector and dataset-label vocabularies.
        model (Any): Loaded Ultralytics detector trained for this component.
        class_names (list[str]): That detector's own class vocabulary.
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


def load_stage_config(mode: str) -> SimpleNamespace:
    """Load and anchor the stage config without loading any model.

    The upload page, the result page and the folder servers only need paths and
    the default component. Building a full context for those would load a VAE
    stack before the first upload, so they use this instead.

    Args:
        mode (str): Runtime configuration mode (``debug`` or ``prod``).

    Returns:
        SimpleNamespace: Anchored config of that mode.
    """
    config: SimpleNamespace = load_config(mode, str(SCRIPT_DIR / "config.toml"))
    anchor_path_keys(config, PATH_CONFIG_KEYS)
    return config


def available_components() -> list[str]:
    """List the components the shared registry makes selectable.

    Returns:
        list[str]: Component keys in registry order.

    Raises:
        FileNotFoundError: If ``components.toml`` is absent.
        KeyError: If it declares no components.
    """
    return component_names(load_component_variants())


def default_component(config: SimpleNamespace) -> str:
    """Return the component a fresh page selects.

    Args:
        config (SimpleNamespace): Stage config with
            ``MISSING_COMPONENT_CLASS_NAME``.

    Returns:
        str: The configured default component key.
    """
    return str(config.MISSING_COMPONENT_CLASS_NAME)


def _build_context(
    config: SimpleNamespace,
    hyperparams: SimpleNamespace,
    log: SimpleNamespace,
    component: str,
) -> FrontendContext:
    """Load the sampler, the detector and the resolved ontology for one component.

    Args:
        config (SimpleNamespace): Config with paths; the selected component's
            two model paths are written onto this copy.
        hyperparams (SimpleNamespace): Stage numbers.
        log (SimpleNamespace): Configured project logger.
        component (str): Registry key of the component to load.

    Returns:
        FrontendContext: The ready context for that component.

    Raises:
        FileNotFoundError: If a checkpoint, the weights or a color index is absent.
        KeyError: If the component is not declared in the registry.
        ValueError: On any dimension or ontology disagreement.
    """
    variant = resolve_component_variant(load_component_variants(), component)
    # Work on a copy so two contexts never share the two model-path keys.
    selected: SimpleNamespace = copy.deepcopy(config)
    apply_component_variant(selected, variant)

    device: torch.device = default_device(str(selected.DEVICE), log)
    sampler = MaskDiffusionSampler(
        checkpoint_path=str(selected.MASK_DIFFUSION_CHECKPOINT_PATH),
        missing_component_name=str(selected.MISSING_COMPONENT_CLASS_NAME),
        class_group_prefix=str(selected.TARGET_CLASS_GROUP_PREFIX),
        mask_files={
            "floorplan": str(selected.FLOORPLAN_MASK_FILE),
            "partial": str(selected.FURNITURE_MASK_FILE),
            "target": str(selected.FURNITURE_MASK_FILE),
        },
        color_index_dir=Path(str(selected.MASK_COLOR_INDEX_DIR)),
        color_index_files={
            "floorplan": str(selected.FLOORPLAN_COLOR_INDEX_FILE),
            "partial": str(selected.FURNITURE_COLOR_INDEX_FILE),
            "target": str(selected.FURNITURE_COLOR_INDEX_FILE),
        },
        device=device,
        log=log,
    )

    from ultralytics import YOLO  # lazy: only needed when a component is selected

    weights_path = Path(str(selected.YOLO_WEIGHTS_PATH))
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Detector weights not found: {weights_path}. Fix the '{component}' "
            "row of components.toml."
        )
    model = YOLO(str(weights_path))
    # The selected weights are the authority on their own vocabulary: a
    # missing_yolo detector has one class (id 0), the multi-class YOLO model has
    # ten. Aliases still come from the YOLO stage's ontology because mask names
    # differ from some class names (dining_set -> Dining_table).
    class_names: list[str] = detector_class_names(model)
    _, aliases = load_detector_ontology(str(selected.YOLO_HYPERPARAMETERS_PATH))
    ontology = resolve_component_ontology(
        name=str(selected.MISSING_COMPONENT_CLASS_NAME),
        group_prefix=str(selected.TARGET_CLASS_GROUP_PREFIX),
        mask_channel=sampler.missing_channel,
        class_names=class_names,
        aliases=aliases,
        label_names_table=dict(getattr(hyperparams, "COMPONENT_LABEL_NAMES", {})),
    )
    log.info(
        f"Frontend ready: component '{ontology.name}' from "
        f"{sampler.checkpoint_path}, detector {weights_path} "
        f"(class {ontology.yolo_class_id} '{class_names[ontology.yolo_class_id]}') "
        f"on {device}."
    )
    return FrontendContext(
        config=selected,
        hyperparams=hyperparams,
        sampler=sampler,
        ontology=ontology,
        model=model,
        class_names=class_names,
    )


def get_context(
    mode: str = "debug", component: str | None = None
) -> FrontendContext:
    """Return the cached context of one component, building it on first use.

    Args:
        mode (str): Runtime configuration mode (``debug`` or ``prod``), selecting
            the matching table of ``frontend/config.toml``.
        component (str | None): Component key to load, or ``None`` to use the
            configured default (the form's initial selection).

    Returns:
        FrontendContext: The cached, read-only context of that component.

    Raises:
        FileNotFoundError: If the registry, a checkpoint, the weights or a color
            index is absent.
        KeyError: If the component is not declared in the registry.
        ValueError: On any dimension or ontology disagreement.
    """
    config: SimpleNamespace = load_stage_config(mode)
    selected: str = str(component) if component else default_component(config)
    key: tuple[str, str] = (mode, selected)
    cached: FrontendContext | None = _CONTEXTS.get(key)
    if cached is not None:
        return cached
    with _LOCK:
        if key in _CONTEXTS:  # another request built it while we waited
            return _CONTEXTS[key]
        hyperparams: SimpleNamespace = load_hyperparams(
            str(SCRIPT_DIR / "hyperparameters.toml")
        )
        log: SimpleNamespace = setup_logger(config)
        _CONTEXTS[key] = _build_context(config, hyperparams, log, selected)
    return _CONTEXTS[key]
