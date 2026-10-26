"""Path and settings handling for the evaluator stage.

The evaluator runs from the repository root (``run_eval.sh`` cds there), but a
user can also launch it from anywhere, so every path key of ``config.toml`` is
anchored at the repository root once, up front. That keeps the debug log, the
telemetry directory and the frozen checkpoint paths meaningful no matter which
directory the run was started from, and it is the same policy the YOLO stage
follows with ``anchor_config_paths``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from helpers.hyperparams import load_hyperparams

# Repository root, resolved from this file's location so it never depends on
# the working directory.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Config keys whose value is a filesystem path. Every one of these is rewritten
# to an absolute path anchored at ROOT_DIR when it is relative.
PATH_CONFIG_KEYS: tuple[str, ...] = (
    "DEBUG_LOG_FILE_PATH",
    "OUTPUT_SAMPLES_DIRECTORY",
    "TELEMETRY_LOG_DIRECTORY",
    "MASK_DIFFUSION_CHECKPOINT_PATH",
    "TEST_IMAGES_DIR",
    "VALIDATION_IMAGES_DIR",
    "FULL_PIPELINE_OUTPUT_DIRECTORY",
    "MASK_COLOR_INDEX_DIR",
    "YOLO_WEIGHTS_PATH",
    "YOLO_HYPERPARAMETERS_PATH",
)


def anchor_path_keys(config: SimpleNamespace, keys: tuple[str, ...] = PATH_CONFIG_KEYS) -> None:
    """Rewrite relative path keys of ``config`` to absolute paths, in place.

    Args:
        config (SimpleNamespace): Parsed config object to mutate.
        keys (tuple[str, ...]): Config keys to anchor. Missing keys are skipped
            so adding a key to the TOML cannot break an older call site.
    """
    for key in keys:
        if not hasattr(config, key):
            continue
        path: Path = Path(str(getattr(config, key)))
        setattr(config, key, str(path if path.is_absolute() else ROOT_DIR / path))
    # The project logger writes to this file on setup, so its directory must
    # exist even on a fresh clone.
    Path(str(config.DEBUG_LOG_FILE_PATH)).parent.mkdir(parents=True, exist_ok=True)


def load_detector_ontology(hyperparams_path: str) -> tuple[list[str], dict[str, str]]:
    """Read the detector's class vocabulary and aliases from the YOLO stage.

    The evaluator never redeclares the ontology: the class ids in the label
    files it writes must mean exactly what the weights under test learned, so
    they are read from the same TOML the detector was trained with.

    Args:
        hyperparams_path (str): Path of ``YOLO/hyperparameters.toml``.

    Returns:
        tuple[list[str], dict[str, str]]: Ordered class vocabulary (index ==
            class id) and the source-name -> class-name alias table.

    Raises:
        KeyError: If the YOLO hyperparameters declare no ``CLASS_NAMES``.
    """
    yolo_hyperparams: SimpleNamespace = load_hyperparams(hyperparams_path)
    if not hasattr(yolo_hyperparams, "CLASS_NAMES"):
        raise KeyError(
            f"{hyperparams_path} declares no CLASS_NAMES; the detector ontology "
            "cannot be resolved."
        )
    class_names: list[str] = [str(name) for name in yolo_hyperparams.CLASS_NAMES]
    aliases: dict[str, str] = {
        str(source): str(target)
        for source, target in dict(getattr(yolo_hyperparams, "CLASS_ALIASES", {})).items()
    }
    return class_names, aliases
