"""
Mask layout, channel encoding and colorized rendering for mask_VAE.

The dataset stores segmentation masks as single-channel 8-bit (``L``) PNGs of
plain integer class ids (``0`` is background, higher ids follow the color map
key order). The model never sees those integers: it sees a one-vs-rest binary
tensor, one channel per class, and it is trained with binary cross-entropy
against that tensor.

This module owns three concerns and nothing else:

1. **Layout** - read the ``*_color_index.json`` maps, decide which channels
   exist and in which order (:class:`MaskLayout`).
2. **Encoding** - turn a set of per-group integer masks into the one-vs-rest
   channel tensor the model consumes (:func:`encode_masks_to_channels`).
3. **Rendering** - turn either an encoded tensor or raw model logits into RGB
   using the color map, so the saved grids the dashboard serves are colored
   floor plans rather than grayscale class-id maps
   (:func:`render_group_rgb`).

Channel layout for ``MASK_GROUPS = ["floorplan", "furniture"]``:

===========  =========================================================
Global       Meaning
channel
===========  =========================================================
``0``        Shared background: on wherever the pixel is ``0`` in
             *every* selected mask.
``1 .. 11``  ``floorplan`` classes 1..11, in color-map key order.
``12 .. 20`` ``furniture`` classes 1..9, in color-map key order.
===========  =========================================================

The background channel is shared on purpose: the two source masks are rendered
from the same layout, so a pixel is background for the model exactly when it is
background in both. This also makes the model's output an explicit background
channel, as opposed to "all classes off", which is what a one-vs-rest target
without a background channel would mean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

# Suffix appended to a group name to build a globally unique class label for
# logs and per-class telemetry columns, e.g. "floorplan/living".
GROUP_NAME_SEPARATOR: str = "/"

# Number of hex digits in a color such as "#38bdf8".
HEX_COLOR_LENGTH: int = 7

# Single channel depth of the source masks. Guards against silently reading an
# RGB render as if it were a class-id map.
EXPECTED_MASK_IMAGE_MODE: str = "L"


@dataclass(frozen=True)
class MaskGroup:
    """
    One semantic mask family, e.g. "floorplan", with its class table.

    The class table is a straight copy of the group's ``*_color_index.json``:
    local class id ``0`` is background and ids ``1..N-1`` follow the file's key
    order, which is the authoritative class ordering for that mask.

    Attributes:
        name (str): Group name as written in ``config.toml``'s ``MASK_GROUPS``.
        mask_file_name (str): Mask filename inside each sample folder.
        class_names (tuple[str, ...]): One name per LOCAL class id, index 0 is
            the background name of this group.
        colors (tuple[tuple[int, int, int], ...]): One ``(R, G, B)`` triple in
            ``0..255`` per LOCAL class id, parallel to ``class_names``.
        channel_indices (tuple[int, ...]): Global model channel that each LOCAL
            class id is written to. Index 0 is always global channel 0 (the
            shared background); the rest are the group's own foreground
            channels. Selected with ``Tensor.index_select`` to get this group's
            logits in local class order.
    """

    name: str
    mask_file_name: str
    class_names: tuple[str, ...]
    colors: tuple[tuple[int, int, int], ...]
    channel_indices: tuple[int, ...]

    @property
    def num_classes(self) -> int:
        """Number of classes in this group, background included."""
        return len(self.class_names)

    @property
    def local_labels(self) -> tuple[str, ...]:
        """
        Globally unique display labels, one per local class id.

        Prefixed with the group name because two groups may legitimately share a
        class name (the furniture map gives both ``couch`` and ``chair`` the
        same color), and telemetry columns must stay unique.
        """
        return tuple(
            f"{self.name}{GROUP_NAME_SEPARATOR}{name}"
            for name in self.class_names
        )

    def color_tensor(self, device: torch.device) -> torch.Tensor:
        """
        Build the ``(num_classes, 3, 1, 1)`` lookup table that maps this group's
        local class ids to colors, scaled to the ``0..1`` range float image
        tensors use.

        Args:
            device (torch.device): Device the LUT is allocated on, so the
                indexed result lands next to the tensors being rendered.

        Returns:
            torch.Tensor: Float LUT of shape ``(num_classes, 3, 1, 1)``.
        """
        colors: torch.Tensor = torch.tensor(
            self.colors, dtype=torch.float32, device=device
        )
        return (colors / 255.0).unsqueeze(-1).unsqueeze(-1)


@dataclass(frozen=True)
class MaskLayout:
    """
    The full channel layout of one mask_VAE run.

    Built once at startup from ``config.toml`` plus the color index JSONs, then
    handed to the dataset, the model and the grid renderers so that every one of
    them agrees on which channel means what.

    Attributes:
        groups (tuple[MaskGroup, ...]): Selected mask groups, in the order they
            occupy the model's channels.
        class_names (tuple[str, ...]): One label per GLOBAL channel, so
            ``class_names[channel]`` is the class that channel predicts.
        num_channels (int): Required encoder input and decoder output channels.
        foreground_channels (tuple[int, ...]): Global channels of every
            foreground class, i.e. all channels except the background one.
    """

    groups: tuple[MaskGroup, ...]
    class_names: tuple[str, ...]
    num_channels: int
    foreground_channels: tuple[int, ...]

    def describe(self) -> str:
        """
        Render the layout as a single multi-line log block.

        Returns:
            str: One line per group plus a channel count, listing the global
                channel range each group occupies.
        """
        lines: list[str] = [
            f"Mask layout: {len(self.groups)} group(s), "
            f"{self.num_channels} channels, "
            f"{len(self.foreground_channels)} foreground"
        ]
        for group in self.groups:
            lines.append(
                f"  - {group.name}: {group.mask_file_name} -> global channels "
                f"{list(group.channel_indices)} "
                f"({group.num_classes} classes: {', '.join(group.local_labels)})"
            )
        return "\n".join(lines)


def parse_hex_color(hex_color: object, context: str) -> tuple[int, int, int]:
    """
    Parse a ``"#rrggbb"`` string into an ``(R, G, B)`` triple.

    Args:
        hex_color (object): Value read from the color index JSON. Validated to
            be a string so a malformed map fails here instead of producing a
            silently black region in the rendered grids.
        context (str): Human readable origin of the value, used in error
            messages, e.g. ``"furniture_color_index.json['3']['color']"``.

    Returns:
        tuple[int, int, int]: Red, green and blue components in ``0..255``.

    Raises:
        TypeError: If ``hex_color`` is not a string.
        ValueError: If the string is not ``#`` followed by six hex digits.
    """
    if not isinstance(hex_color, str):
        raise TypeError(f"{context} must be a string, got {hex_color!r}.")
    if len(hex_color) != HEX_COLOR_LENGTH or not hex_color.startswith("#"):
        raise ValueError(
            f"{context} must be a '#rrggbb' string of length "
            f"{HEX_COLOR_LENGTH}, got {hex_color!r}."
        )
    try:
        return (
            int(hex_color[1:3], 16),
            int(hex_color[3:5], 16),
            int(hex_color[5:7], 16),
        )
    except ValueError as exc:
        raise ValueError(
            f"{context} must be a '#rrggbb' string of hex digits, got {hex_color!r}."
        ) from exc


def load_color_index(color_index_path: Path) -> tuple[list[str], list[tuple[int, int, int]]]:
    """
    Read a ``*_color_index.json`` map into parallel class-name and color lists.

    The file maps the numeric class id, as a string, to ``{"name", "color"}``.
    Ids must be contiguous from ``0`` because the mask PNGs use the integer
    value directly as a list index later on; a gap would silently shift every
    color, so it is rejected here.

    Args:
        color_index_path (Path): Path of the JSON file to read.

    Returns:
        tuple[list[str], list[tuple[int, int, int]]]: Class names indexed by
            class id, and ``(R, G, B)`` triples indexed by the same class id.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the class ids are not contiguous starting at ``0``.
        TypeError: If an entry is not a mapping with a string ``name``.
    """
    if not color_index_path.is_file():
        raise FileNotFoundError(
            f"Mask color index file not found: {color_index_path}. It is part of the "
            "dataset and documents how to turn mask class ids into display colors."
        )

    with open(color_index_path, "r", encoding="utf-8") as handle:
        raw_index: object = json.load(handle)

    if not isinstance(raw_index, dict):
        raise TypeError(
            f"{color_index_path} must contain a JSON object mapping class id -> "
            f"{{'name', 'color'}}, got {type(raw_index).__name__}."
        )

    class_ids: list[int] = sorted(int(key) for key in raw_index)
    expected_ids: list[int] = list(range(len(class_ids)))
    if class_ids != expected_ids:
        raise KeyError(
            f"{color_index_path} must define class ids {expected_ids[0]}..{expected_ids[-1]} "
            f"with no gaps, got {class_ids}."
        )

    class_names: list[str] = []
    colors: list[tuple[int, int, int]] = []
    for class_id in class_ids:
        entry: object = raw_index[str(class_id)]
        context: str = f"{color_index_path.name}[{class_id}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{context} must be a JSON object, got {entry!r}.")
        name: object = entry.get("name")
        if not isinstance(name, str) or not name:
            raise TypeError(f"{context}['name'] must be a non-empty string, got {name!r}.")
        class_names.append(name)
        colors.append(
            parse_hex_color(entry.get("color"), context=f"{context}['color']")
        )

    if not class_names:
        raise KeyError(f"{color_index_path} defines no classes.")
    return class_names, colors


def resolve_color_index_path(
    group_name: str,
    color_index_dir: Path,
    fallback_dir: Path,
) -> Path:
    """
    Locate a group's color index JSON inside an already-resolved directory.

    Args:
        group_name (str): Mask group name, used in the error message.
        color_index_dir (Path): Directory configured as
            ``MASK_COLOR_INDEX_DIR``, searched first.
        fallback_dir (Path): Directory searched when the first one misses, so a
            moved dataset only needs the split paths updated.

    Returns:
        Path: The resolved color index JSON path.

    Raises:
        FileNotFoundError: If the file is in neither directory.
    """
    file_name: str = f"{group_name}_color_index.json"
    candidate: Path = color_index_dir / file_name
    if candidate.is_file():
        return candidate

    fallback_candidate: Path = fallback_dir / file_name
    if fallback_candidate.is_file():
        return fallback_candidate

    raise FileNotFoundError(
        f"No color index '{file_name}' for mask group '{group_name}' in "
        f"{color_index_dir} or in the dataset root fallback {fallback_dir}. "
        "Set MASK_COLOR_INDEX_DIR in mask_VAE/config.toml to the directory that "
        "holds it."
    )


def resolve_mask_group_names(config: SimpleNamespace) -> list[str]:
    """
    Resolve the ordered mask group names from the configured mode.

    Primary source is ``MASK_MODE`` looked up in the ``MASK_MODES`` table
    (e.g. ``"floorplan" -> ["floorplan"]``, ``"arrangement" -> ["furniture"]``,
    ``"both" -> ["floorplan", "furniture"]``). When ``MASK_MODE`` is absent,
    falls back to the legacy explicit ``MASK_GROUPS`` list so old configs keep
    working.

    Args:
        config (SimpleNamespace): Parsed ``config.toml``. Must expose either
            ``MASK_MODE`` plus ``MASK_MODES``, or ``MASK_GROUPS``.

    Returns:
        list[str]: Ordered group names, e.g. ``["floorplan", "furniture"]``.

    Raises:
        AttributeError: If neither ``MASK_MODE`` nor ``MASK_GROUPS`` is present.
        KeyError: If the mode is not a key of the ``MASK_MODES`` table.
        TypeError: If the mode entry is not a list of non-empty strings.
        ValueError: If the resolved list is empty or contains duplicates.
    """
    if hasattr(config, "MASK_MODE"):
        mode_name: str = str(config.MASK_MODE)
        modes_table: object = _require_config_key(config, "MASK_MODES")
        if not isinstance(modes_table, dict):
            raise TypeError(
                "mask_VAE/config.toml must define a [default.MASK_MODES] table "
                f"mapping mode name -> group list, got {type(modes_table).__name__}."
            )
        if mode_name not in modes_table:
            raise KeyError(
                f"MASK_MODE={mode_name!r} is not in the [default.MASK_MODES] "
                f"table. Known modes: {sorted(modes_table.keys())}."
            )
        raw_groups: object = modes_table[mode_name]
        if not isinstance(raw_groups, list):
            raise TypeError(
                f"[default.MASK_MODES].{mode_name} must be a list of group "
                f"names, got {type(raw_groups).__name__}."
            )
        group_names: list[str] = []
        for entry in raw_groups:
            if not isinstance(entry, str) or not entry:
                raise TypeError(
                    f"[default.MASK_MODES].{mode_name} must hold non-empty "
                    f"strings, got {entry!r}."
                )
            group_names.append(entry)
        source_label: str = f"MASK_MODE={mode_name!r}"
    elif hasattr(config, "MASK_GROUPS"):
        # Legacy path: explicit list, kept for backward compatibility.
        raw_legacy: object = getattr(config, "MASK_GROUPS")
        if not isinstance(raw_legacy, list):
            raise TypeError(
                f"MASK_GROUPS must be a list of group names, got {type(raw_legacy).__name__}."
            )
        group_names = [str(name) for name in raw_legacy]
        source_label = "MASK_GROUPS"
    else:
        raise AttributeError(
            "mask_VAE/config.toml must define MASK_MODE (preferred) or "
            "MASK_GROUPS (legacy) in the active table."
        )

    if not group_names:
        raise ValueError(
            f"{source_label} in mask_VAE/config.toml must list at least one "
            "mask group."
        )
    if len(set(group_names)) != len(group_names):
        raise ValueError(
            f"{source_label} in mask_VAE/config.toml contains duplicates: {group_names}."
        )
    return group_names


def build_mask_layout(
    config: SimpleNamespace,
    fallback_color_index_dir: Path,
) -> MaskLayout:
    """
    Build the :class:`MaskLayout` selected by ``config.toml``.

    The class ids of every selected mask group are laid out as follows: global
    channel ``0`` is the shared background, and each group then claims as many
    channels as it has foreground classes, in resolved mode order
    (:func:`resolve_mask_group_names`).

    Args:
        config (SimpleNamespace): Parsed ``config.toml``. Must expose
            ``MASK_MODE`` plus ``MASK_MODES`` (or legacy ``MASK_GROUPS``),
            ``MASK_FILES`` and ``MASK_COLOR_INDEX_FILES`` in the active table,
            plus ``MASK_COLOR_INDEX_DIR``.
        fallback_color_index_dir (Path): Directory searched for the color index
            JSONs when ``MASK_COLOR_INDEX_DIR`` does not hold them. In practice
            the parent of the split folder.

    Returns:
        MaskLayout: The validated channel layout.

    Raises:
        AttributeError: If a required config key is missing.
        ValueError: If the resolved group list is empty or names a group twice.
    """
    group_names: list[str] = resolve_mask_group_names(config)

    mask_files: object = _require_config_key(config, "MASK_FILES")
    color_index_files: object = _require_config_key(config, "MASK_COLOR_INDEX_FILES")
    color_index_dir: Path = Path(str(_require_config_key(config, "MASK_COLOR_INDEX_DIR")))

    groups: list[MaskGroup] = []
    global_class_names: list[str] = []
    # Global channel 0 is reserved for the shared background up front, so the
    # first group's foreground classes start at 1.
    next_channel: int = 1
    for group_name in group_names:
        mask_file_name: str = _lookup_table_entry(mask_files, group_name, "MASK_FILES")
        color_index_file_name: str = _lookup_table_entry(
            color_index_files, group_name, "MASK_COLOR_INDEX_FILES"
        )
        color_index_path: Path = (
            color_index_dir / color_index_file_name
            if (color_index_dir / color_index_file_name).is_file()
            else resolve_color_index_path(
                group_name, color_index_dir, fallback_color_index_dir
            )
        )
        class_names, colors = load_color_index(color_index_path)

        num_foreground: int = len(class_names) - 1
        # Local class 0 maps to the shared background channel; local classes
        # 1..N map to this group's own consecutive channels.
        channel_indices: tuple[int, ...] = (0,) + tuple(
            range(next_channel, next_channel + num_foreground)
        )
        next_channel += num_foreground

        group = MaskGroup(
            name=group_name,
            mask_file_name=mask_file_name,
            class_names=tuple(class_names),
            colors=tuple(colors),
            channel_indices=channel_indices,
        )
        groups.append(group)
        # Channel 0 keeps the plain "background" name; every foreground class is
        # labelled with its group so the names stay unique.
        global_class_names.extend(group.local_labels[1:])

    return MaskLayout(
        groups=tuple(groups),
        class_names=("background", *global_class_names),
        num_channels=next_channel,
        foreground_channels=tuple(range(1, next_channel)),
    )


def _require_config_key(config: SimpleNamespace, key: str) -> object:
    """
    Read a config key that must be present in the active table, with an error
    message that names the table-placement pitfall.

    Args:
        config (SimpleNamespace): Parsed config.
        key (str): Key to read.

    Returns:
        object: The raw config value.

    Raises:
        AttributeError: If the key is absent from the active table.
    """
    if not hasattr(config, key):
        raise AttributeError(
            f"mask_VAE/config.toml must define {key} in the active table. A key "
            "placed after [debug]/[prod] belongs to that mode's table, not to "
            "[default]."
        )
    return getattr(config, key)


def _lookup_table_entry(
    table: object,
    group_name: str,
    table_key: str,
) -> str:
    """
    Look a mask group up in a TOML table of filenames.

    Args:
        table (object): The parsed table, e.g. ``[default.MASK_FILES]``.
        group_name (str): Key to look up.
        table_key (str): Config key the table came from, for the error message.

    Returns:
        str: The filename, guaranteed non-empty.

    Raises:
        TypeError: If ``table`` is not a mapping or the entry is not a string.
        KeyError: If the group is not in the table.
    """
    if not isinstance(table, dict):
        raise TypeError(
            f"mask_VAE/config.toml must define a [default.{table_key}] table, "
            f"got {type(table).__name__}."
        )
    if group_name not in table:
        raise KeyError(
            f"MASK_GROUPS asks for group '{group_name}' but [default.{table_key}] "
            f"only defines {sorted(table.keys())}."
        )
    file_name: object = table[group_name]
    if not isinstance(file_name, str) or not file_name:
        raise TypeError(
            f"[default.{table_key}].{group_name} must be a non-empty filename, "
            f"got {file_name!r}."
        )
    return file_name


def encode_masks_to_channels(
    mask_arrays: list[torch.Tensor],
    layout: MaskLayout,
) -> torch.Tensor:
    """
    Turn per-group integer masks into the one-vs-rest binary channel tensor.

    Every pixel gets exactly one active channel *per group*, plus the shared
    background channel when it is background in all groups. The result is what
    the encoder receives and what the binary cross-entropy is computed against.

    Args:
        mask_arrays (list[torch.Tensor]): One ``(H, W)`` or ``(B, H, W)`` int64
            tensor of class ids per group, ordered like ``layout.groups``. All
            entries must share a batch dimension: a single unbatched mask is
            promoted to a batch of one.
        layout (MaskLayout): Channel layout to encode into.

    Returns:
        torch.Tensor: Float32 ``(B, num_channels, H, W)`` tensor of zeros and
            ones, the binary target for the loss and the encoder input.

    Raises:
        ValueError: If the number of masks does not match the number of groups,
            or the masks disagree on batch size or spatial size.
    """
    if len(mask_arrays) != len(layout.groups):
        raise ValueError(
            f"Expected one mask per configured group ({len(layout.groups)}), got "
            f"{len(mask_arrays)}."
        )

    batched: list[torch.Tensor] = [_as_batched_int_mask(array) for array in mask_arrays]
    reference: torch.Tensor = batched[0]
    # Every mask must share the batch and spatial shape of the first one, or the
    # scatter into the shared channel tensor below would be undefined.
    for group, array in zip(layout.groups[1:], batched[1:], strict=True):
        if array.shape != reference.shape:
            raise ValueError(
                f"Mask '{group.mask_file_name}' has shape {tuple(array.shape)} but "
                f"'{layout.groups[0].mask_file_name}' has "
                f"{tuple(reference.shape)}. All masks of a sample must share the "
                "same size."
            )

    batch_size, height, width = reference.shape
    channels: torch.Tensor = torch.zeros(
        (batch_size, layout.num_channels, height, width), dtype=torch.float32
    )

    for group, array in zip(layout.groups, batched, strict=True):
        # One-hot over this group's own class ids, so the last axis is ordered
        # exactly like MaskGroup.class_names.
        local_one_hot: torch.Tensor = torch.nn.functional.one_hot(
            array, num_classes=group.num_classes
        ).permute(0, 3, 1, 2).to(torch.float32)
        # Scatter the local classes onto the global channels this group owns.
        channels[:, group.channel_indices, :, :] = local_one_hot

    # The loop above wrote each group's own background into global channel 0.
    # Replace it with the shared definition: background in every group.
    any_foreground: torch.Tensor = channels[:, layout.foreground_channels].amax(dim=1)
    channels[:, 0:1, :, :] = 1.0 - any_foreground
    return channels


def encode_masks_to_single_channel(
    mask_arrays: list[torch.Tensor],
    layout: MaskLayout,
) -> torch.Tensor:
    """
    Stack per-group integer masks into the float tensor the mask encoder reads.

    One channel per group holding that group's class ids normalized to
    ``0..1`` by ``num_classes - 1``, so groups with different class counts
    share the same input scale. Values stay exact fractions of the integer
    ids; no rounding is applied.

    Args:
        mask_arrays (list[torch.Tensor]): One ``(H, W)`` or ``(B, H, W)`` int
            tensor of class ids per group, ordered like ``layout.groups``.
        layout (MaskLayout): Channel layout defining the group order.

    Returns:
        torch.Tensor: Float32 ``(B, num_groups, H, W)`` tensor in ``0..1``.

    Raises:
        ValueError: If the number of masks does not match the number of groups,
            or the masks disagree on batch size or spatial size.
    """
    if len(mask_arrays) != len(layout.groups):
        raise ValueError(
            f"Expected one mask per configured group ({len(layout.groups)}), got "
            f"{len(mask_arrays)}."
        )

    batched: list[torch.Tensor] = [_as_batched_int_mask(array) for array in mask_arrays]
    reference: torch.Tensor = batched[0]
    for group, array in zip(layout.groups[1:], batched[1:], strict=True):
        if array.shape != reference.shape:
            raise ValueError(
                f"Mask '{group.mask_file_name}' has shape {tuple(array.shape)} but "
                f"'{layout.groups[0].mask_file_name}' has "
                f"{tuple(reference.shape)}. All masks of a sample must share the "
                "same size."
            )

    channels: list[torch.Tensor] = []
    for group, array in zip(layout.groups, batched, strict=True):
        # Normalize by the group's own id range so a floorplan id and a
        # furniture id of the same number mean the same input magnitude.
        scale: float = float(max(group.num_classes - 1, 1))
        channels.append(array.to(torch.float32).unsqueeze(1) / scale)
    return torch.cat(channels, dim=1)


def _as_batched_int_mask(mask_array: torch.Tensor) -> torch.Tensor:
    """
    Normalize a mask to a contiguous int64 tensor of shape ``(B, H, W)``.

    Args:
        mask_array (torch.Tensor): ``(H, W)`` or ``(B, H, W)`` class-id tensor.

    Returns:
        torch.Tensor: Batched int64 class-id tensor.

    Raises:
        ValueError: If the input is not a 2-D or 3-D tensor.
    """
    if mask_array.ndim == 2:
        return mask_array.unsqueeze(0).to(torch.int64)
    if mask_array.ndim == 3:
        return mask_array.to(torch.int64)
    raise ValueError(
        f"A mask must be (H, W) or (B, H, W), got shape {tuple(mask_array.shape)}."
    )


def render_group_rgb(
    channel_tensor: torch.Tensor,
    group: MaskGroup,
    device: torch.device,
) -> torch.Tensor:
    """
    Render one mask group as an RGB image batch using the dataset's color map.

    ``channel_tensor`` may be either raw one-vs-rest binary targets or model
    logits: in both cases an ``argmax`` over the group's own channels is the
    predicted class id, so the model needs no extra calibration step before it
    can be looked at.

    Args:
        channel_tensor (torch.Tensor): ``(B, num_channels, H, W)`` tensor of
            binary targets or logits for the whole model.
        group (MaskGroup): Group to render, which selects both the channels and
            the color table.
        device (torch.device): Device the color lookup table is allocated on.

    Returns:
        torch.Tensor: Float32 ``(B, 3, H, W)`` RGB image in the ``0..1`` range,
            ready for ``torchvision.utils.make_grid`` and ``ToPILImage``.
    """
    channel_index: torch.Tensor = torch.tensor(
        group.channel_indices, dtype=torch.long, device=device
    )
    # (B, num_channels, H, W) -> (B, group.num_classes, H, W) in local class order.
    local_scores: torch.Tensor = channel_tensor.index_select(1, channel_index)
    class_ids: torch.Tensor = local_scores.argmax(dim=1)

    color_lut: torch.Tensor = group.color_tensor(device)
    batch_size, height, width = class_ids.shape
    # index_select on a flat id vector is the cheapest way to gather colors.
    flat_ids: torch.Tensor = class_ids.reshape(-1)
    flat_colors: torch.Tensor = color_lut.index_select(0, flat_ids)
    return flat_colors.reshape(batch_size, height, width, 3).permute(0, 3, 1, 2)
