"""Class taxonomy shared by rasterization, case generation and visualization.

A plan is stored as two *instance layers*:

* ``structure`` layer: walls, windows, doors (thin elements, drawn on top).
* ``room`` layer: room regions (living, bedroom, ..., stair).

Each layer is an ``uint8`` image whose pixel value is a layer-local instance
id (0 = empty). The class of every instance is kept in the plan metadata.
Semantic maps have one channel per class, ordered structure classes first,
then room classes (see ``params.toml`` [taxonomy]).
"""

from dataclasses import dataclass
from types import SimpleNamespace

STRUCTURE_LAYER: str = "structure"
ROOM_LAYER: str = "room"
LAYERS: tuple[str, str] = (STRUCTURE_LAYER, ROOM_LAYER)


@dataclass(frozen=True)
class Schema:
    """Immutable description of classes, channels, component types and colours.

    Attributes:
        structure_classes (tuple[str, ...]): Classes of the structure layer, in
            drawing order (later overwrites earlier).
        room_classes (tuple[str, ...]): Classes of the room layer.
        component_type (dict[str, str]): Class -> component type
            ("wall", "door", "window", "room", "other").
        colors (dict[str, str]): Class (plus "background", "mask_outline") ->
            hex colour.
    """

    structure_classes: tuple[str, ...]
    room_classes: tuple[str, ...]
    component_type: dict[str, str]
    colors: dict[str, str]

    @classmethod
    def from_params(cls, params: SimpleNamespace) -> "Schema":
        """Build the schema from the loaded ``params.toml``.

        Args:
            params (SimpleNamespace): Parameters with ``taxonomy`` and ``colors``.

        Returns:
            Schema: Validated schema.

        Raises:
            ValueError: If a class lacks a component type or colour, or a class
                name appears in both layers.
        """
        schema = cls(
            structure_classes=tuple(params.taxonomy.STRUCTURE_CLASSES),
            room_classes=tuple(params.taxonomy.ROOM_CLASSES),
            component_type=dict(vars(params.taxonomy.component_type)),
            colors=dict(vars(params.colors)),
        )
        schema._validate()
        return schema

    def _validate(self) -> None:
        """Fail fast on an inconsistent taxonomy.

        Raises:
            ValueError: On overlapping layers or missing type/colour entries.
        """
        overlap: set[str] = set(self.structure_classes) & set(self.room_classes)
        if overlap:
            raise ValueError(f"Classes in both layers: {sorted(overlap)}")
        for name in self.channels:
            if name not in self.component_type:
                raise ValueError(f"Class '{name}' has no [taxonomy.component_type] entry")
            if name not in self.colors:
                raise ValueError(f"Class '{name}' has no [colors] entry")

    @property
    def channels(self) -> tuple[str, ...]:
        """Semantic channel names: structure classes then room classes."""
        return self.structure_classes + self.room_classes

    @property
    def num_channels(self) -> int:
        """Number of semantic channels."""
        return len(self.channels)

    def channel_index(self, class_name: str) -> int:
        """Channel index of a class.

        Args:
            class_name (str): Class name.

        Returns:
            int: Index into the semantic channel axis.

        Raises:
            ValueError: If the class is unknown.
        """
        return self.channels.index(class_name)

    def layer_of(self, class_name: str) -> str:
        """Instance layer that stores a class.

        Args:
            class_name (str): Class name.

        Returns:
            str: ``STRUCTURE_LAYER`` or ``ROOM_LAYER``.

        Raises:
            ValueError: If the class is unknown.
        """
        if class_name in self.structure_classes:
            return STRUCTURE_LAYER
        if class_name in self.room_classes:
            return ROOM_LAYER
        raise ValueError(f"Unknown class '{class_name}'")
