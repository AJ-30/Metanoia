"""Ontology resolution: which class is the missing component, and what is it called?

The evaluator has to answer the same question in three different vocabularies,
and getting any of them wrong would silently score the wrong channel:

* the **mask** class id of the missing component (the target VAE's channel),
* the **detector** class id of that component (an index into the YOLO
  ``CLASS_NAMES`` the weights were trained on),
* the **dataset label** names that component is written as inside each sample's
  ``labels.txt`` (which are not the mask color index's names: the mask says
  ``dining_set`` where the labels say ``Dining_table_big``).

Everything is resolved once, at startup, and a failure is a startup error - the
same policy the YOLO stage uses for its own map.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from YOLO.dataset import normalize_class_name


@dataclass(frozen=True)
class ComponentOntology:
    """The missing component expressed in the three vocabularies at once.

    Attributes:
        name (str): Plain component name, e.g. ``tv``.
        qualified_name (str): Name as the mask layout spells it, e.g.
            ``furniture/tv``.
        mask_channel (int): Channel index the frozen target VAE decodes to.
        yolo_class_id (int): Class index inside the detector's ontology.
        label_names (tuple[str, ...]): Names this component is written as in the
            dataset's own ``labels.txt``.
    """

    name: str
    qualified_name: str
    mask_channel: int
    yolo_class_id: int
    label_names: tuple[str, ...] = field(default_factory=tuple)


def resolve_yolo_class_id(
    yolo_class_name: str, class_names: list[str], aliases: dict[str, str]
) -> int:
    """Resolve one component name onto the detector's class vocabulary.

    The name may be a ``CLASS_NAMES`` entry, or a source/mask name folded onto
    one by ``CLASS_ALIASES`` (e.g. ``dining_set`` -> ``Dining_table``). Anything
    that matches nothing, or more than one class, fails fast: a class id that
    means the wrong furniture would make every filtered metric meaningless.

    Args:
        yolo_class_name (str): Component name to resolve.
        class_names (list[str]): Ordered detector vocabulary.
        aliases (dict[str, str]): Source name -> ``CLASS_NAMES`` entry table.

    Returns:
        int: Index of the resolved class in ``class_names``.

    Raises:
        ValueError: If the name resolves to no detector class, or to several.
    """
    by_key: dict[str, list[str]] = {}
    for name in class_names:
        by_key.setdefault(normalize_class_name(name), []).append(name)
    key: str = normalize_class_name(yolo_class_name)
    aliased: str | None = aliases.get(yolo_class_name)
    matches: list[str] = (
        [aliased] if aliased is not None else list(by_key.get(key, []))
    )
    if not matches:
        raise ValueError(
            f"Component '{yolo_class_name}' is not in the detector ontology "
            f"{class_names} and no CLASS_ALIASES entry folds it onto one."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Component '{yolo_class_name}' matches several detector classes "
            f"{matches}; the detector could not tell them apart."
        )
    return class_names.index(matches[0])


def resolve_component_ontology(
    name: str,
    group_prefix: str,
    mask_channel: int,
    class_names: list[str],
    aliases: dict[str, str],
    label_names_table: dict[str, list[str]],
) -> ComponentOntology:
    """Resolve the missing component across the mask, detector and label vocabularies.

    Args:
        name (str): Plain component name, e.g. ``tv``.
        group_prefix (str): Prefix the mask layout's class names carry, e.g.
            ``furniture``; ``""`` when they carry none.
        mask_channel (int): Channel index the sampler resolved for it.
        class_names (list[str]): Detector vocabulary the weights were trained on.
        aliases (dict[str, str]): Detector aliases, e.g. ``dining_set``.
        label_names_table (dict[str, list[str]]): Component name -> dataset
            ``labels.txt`` names, from ``COMPONENT_LABEL_NAMES``.

    Returns:
        ComponentOntology: The resolved component.

    Raises:
        ValueError: If the component is absent from a vocabulary, or the
            detector's own class list does not agree with the ontology table.
    """
    qualified: str = f"{group_prefix}/{name}" if group_prefix else name
    normal_key: str = normalize_class_name(name)
    label_names: tuple[str, ...] = tuple(
        label_names_table.get(name)
        or label_names_table.get(normal_key)
        or []
    )
    if not label_names:
        raise ValueError(
            f"Component '{name}' has no dataset label names in "
            f"COMPONENT_LABEL_NAMES {sorted(label_names_table)}. The ground-truth "
            "boxes are read from the dataset's own labels, so this component "
            "cannot be evaluated until its label names are listed."
        )
    yolo_class_id: int = resolve_yolo_class_id(name, class_names, aliases)
    return ComponentOntology(
        name=name,
        qualified_name=qualified,
        mask_channel=mask_channel,
        yolo_class_id=yolo_class_id,
        label_names=label_names,
    )
