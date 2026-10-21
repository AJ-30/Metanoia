"""Per-missing-component model selection, shared by the evaluator and the frontend.

The two consumers ask the same question - "for component X, which generator and
which detector?" - so the answer lives in ONE registry (``components.toml`` at
the repository root) instead of being copied into both stages' configs. That
keeps the pairing of a generated component with the detector trained on it from
drifting apart.

The registry is loaded on demand, and a selected row is applied to a stage's
config by :func:`apply_component_variant`, which also anchors relative paths and
fails fast when either file is absent. Once applied, the rest of the pipeline
keeps using the same ``MASK_DIFFUSION_CHECKPOINT_PATH`` and
``YOLO_WEIGHTS_PATH`` keys it always used, so no other module needs to know that
the selection exists.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

# Repository root, resolved from this file's location so the registry is found
# no matter which directory a stage was launched from.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
# The shared registry. One file for every stage that needs the pairing.
DEFAULT_REGISTRY_PATH: Path = ROOT_DIR / "components.toml"

# Registry keys every component row must declare.
REQUIRED_VARIANT_KEYS: tuple[str, ...] = (
    "mask_diffusion_checkpoint",
    "detector_weights",
)


@dataclass(frozen=True)
class ComponentVariant:
    """One component paired with the models that generate and judge it.

    Attributes:
        name (str): Component key, e.g. ``tv``, ``showcase`` or ``dining_set``.
        mask_diffusion_checkpoint (Path): Checkpoint of the diffusion run that
            generates this component.
        detector_weights (Path): ``missing_yolo`` weights trained to localise
            this component. The detector is single-class, so its class id is 0.
    """

    name: str
    mask_diffusion_checkpoint: Path
    detector_weights: Path


def anchor_path(path: str | Path) -> Path:
    """Resolve a registry or CLI path against the repository root.

    Args:
        path (str | Path): Absolute path, or a repository-root relative path
            such as ``./missing_yolo/models/tv_best.pt``.

    Returns:
        Path: Absolute path, expanded for ``~`` when the caller typed one.
    """
    candidate: Path = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT_DIR / candidate


def load_component_variants(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, ComponentVariant]:
    """Read every component -> models row of the registry.

    Args:
        registry_path (Path): TOML file holding ``[default.COMPONENT_VARIANTS.*]``.

    Returns:
        dict[str, ComponentVariant]: Component key -> resolved paths, in the
            order the file declares them (the order the frontend lists them).

    Raises:
        FileNotFoundError: If the registry file is absent.
        KeyError: If the registry declares no variants, or a row is incomplete.
    """
    registry_path = Path(registry_path)
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"Component registry not found: {registry_path}. It pairs every "
            "missing component with its diffusion checkpoint and detector."
        )
    with open(registry_path, "rb") as handle:
        data: dict[str, object] = tomllib.load(handle)
    default_table: dict[str, object] = dict(data.get("default", {}))
    raw_variants: dict[str, object] = dict(
        default_table.get("COMPONENT_VARIANTS", {})
    )
    if not raw_variants:
        raise KeyError(
            f"{registry_path} declares no [default.COMPONENT_VARIANTS.*] rows; "
            "at least one component must be selectable."
        )

    variants: dict[str, ComponentVariant] = {}
    for name, raw_row in raw_variants.items():
        row: dict[str, object] = dict(raw_row)
        missing: list[str] = [key for key in REQUIRED_VARIANT_KEYS if not row.get(key)]
        if missing:
            raise KeyError(
                f"Component variant '{name}' in {registry_path} is missing "
                f"{missing}; every row needs a checkpoint and detector weights."
            )
        variants[str(name)] = ComponentVariant(
            name=str(name),
            mask_diffusion_checkpoint=anchor_path(str(row["mask_diffusion_checkpoint"])),
            detector_weights=anchor_path(str(row["detector_weights"])),
        )
    return variants


def resolve_component_variant(
    variants: dict[str, ComponentVariant], name: str
) -> ComponentVariant:
    """Look up one component, failing with the valid choices.

    Args:
        variants (dict[str, ComponentVariant]): Registry loaded by
            :func:`load_component_variants`.
        name (str): Component key to select.

    Returns:
        ComponentVariant: The matching row.

    Raises:
        KeyError: If no row has that key. The message lists every valid key so
            the user does not have to open the registry.
    """
    variant: ComponentVariant | None = variants.get(name)
    if variant is None:
        raise KeyError(
            f"Unknown missing component '{name}'. Select one of "
            f"{sorted(variants)} in components.toml (or via --component)."
        )
    return variant


def apply_component_variant(
    config: SimpleNamespace, variant: ComponentVariant
) -> None:
    """Copy one variant's models onto a stage config, in place.

    The rest of the pipeline reads the historical ``MASK_DIFFUSION_CHECKPOINT_PATH``
    and ``YOLO_WEIGHTS_PATH`` keys, so selecting a component changes those two
    values and nothing else. Both files are checked here, before any sampling.

    Args:
        config (SimpleNamespace): Stage config to mutate.
        variant (ComponentVariant): Selected component and its model paths.

    Raises:
        FileNotFoundError: If the variant's checkpoint or weights are absent.
        ValueError: If a path is a directory rather than a file.
    """
    for role, path in (
        ("diffusion checkpoint", variant.mask_diffusion_checkpoint),
        ("detector weights", variant.detector_weights),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"The {role} for component '{variant.name}' is missing: {path}. "
                "Fix that row of components.toml or retrain the model."
            )
        if not path.is_file():
            raise ValueError(
                f"The {role} for component '{variant.name}' is not a file: {path}."
            )
    config.MISSING_COMPONENT_CLASS_NAME = variant.name
    config.MASK_DIFFUSION_CHECKPOINT_PATH = str(variant.mask_diffusion_checkpoint)
    config.YOLO_WEIGHTS_PATH = str(variant.detector_weights)


def select_component(
    config: SimpleNamespace, name: str | None = None
) -> ComponentVariant:
    """Resolve and apply the component a stage should run.

    Args:
        config (SimpleNamespace): Stage config. Its
            ``MISSING_COMPONENT_CLASS_NAME`` is the default when ``name`` is
            ``None``; it is overwritten with the selected key.
        name (str | None): Explicit selection (CLI flag or form field), or
            ``None`` to keep the configured default.

    Returns:
        ComponentVariant: The applied row.

    Raises:
        FileNotFoundError: If the registry or a selected model file is absent.
        KeyError: If the component is not declared in the registry.
        ValueError: If a selected model path is not a file.
    """
    variants: dict[str, ComponentVariant] = load_component_variants()
    selected_name: str = (
        str(name) if name is not None else str(config.MISSING_COMPONENT_CLASS_NAME)
    )
    variant: ComponentVariant = resolve_component_variant(variants, selected_name)
    apply_component_variant(config, variant)
    return variant


def component_names(variants: dict[str, ComponentVariant] | None = None) -> list[str]:
    """List the selectable components, for a dropdown or an error message.

    Args:
        variants (dict[str, ComponentVariant] | None): Already-loaded registry,
            or ``None`` to load it.

    Returns:
        list[str]: Component keys in registry order.

    Raises:
        FileNotFoundError: If the registry is absent and must be loaded.
        KeyError: If the registry declares no components.
    """
    loaded: dict[str, ComponentVariant] = (
        variants if variants is not None else load_component_variants()
    )
    return list(loaded)
