"""Pure mask rules for the frontend's inpainting mode.

The inpainting page is a small editor on top of the existing missing-component
ensemble: the user erases pixels of the three furniture components that the
model knows, the page sends the edited class-id mask, and the pipeline runs one
component model at a time before merging the three generated masks.

This module owns only the rules that can be tested without a model:

1. an edit is legal when it changes a pixel of an erasable component to
   background and changes nothing else;
2. a component's merged mask keeps the pixels that survived the erasure and
   fills background pixels with the generated component; with protection
   disabled, the literal union lets generated pixels claim any pixel that is not
   that component's remaining original;
3. when two generated components claim the same pixel, the one with the higher
   ensemble probability wins.

Keeping these rules here (rather than in the Flask route or the model code)
makes the merge behaviour reviewable and reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

# The furniture layout's background class. Erasing a pixel means writing this
# id, which is also what the partial encoder uses for a missing component.
BACKGROUND_CLASS_ID: int = 0


def load_furniture_palette(palette_path: Path) -> list[dict[str, Any]]:
    """Read the furniture colour index used by the canvas editor.

    Args:
        palette_path (Path): JSON file mapping class-id strings to an object
            with ``name`` and ``color`` (``#rrggbb``).

    Returns:
        list[dict[str, Any]]: One entry per class, sorted by class id. Each
        entry has ``id`` (int), ``name`` (str), ``rgb`` (tuple[int, int, int])
        and ``hex`` (str).

    Raises:
        FileNotFoundError: If the palette file does not exist.
        ValueError: If the JSON shape or a colour is malformed.
    """
    if not palette_path.is_file():
        raise FileNotFoundError(f"Furniture colour index not found: {palette_path}")
    raw: Any = json.loads(palette_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"Furniture colour index {palette_path} must be a JSON object."
        )

    entries: list[dict[str, Any]] = []
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"Furniture colour index entry {key!r} must be a JSON object."
            )
        try:
            class_id: int = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Furniture colour index key {key!r} is not an integer class id."
            ) from exc
        color: str = str(value.get("color", "")).strip().lstrip("#")
        if len(color) != 6:
            raise ValueError(
                f"Furniture colour index entry {key!r} has colour "
                f"{value.get('color')!r}; expected #rrggbb."
            )
        try:
            rgb: tuple[int, int, int] = (
                int(color[0:2], 16),
                int(color[2:4], 16),
                int(color[4:6], 16),
            )
        except ValueError as exc:
            raise ValueError(
                f"Furniture colour index entry {key!r} has a non-hex colour "
                f"{value.get('color')!r}."
            ) from exc
        entries.append(
            {
                "id": class_id,
                "name": str(value.get("name", f"class_{class_id}")),
                "rgb": rgb,
                "hex": f"#{color.lower()}",
            }
        )
    entries.sort(key=lambda entry: int(entry["id"]))
    return entries


def erasable_channels_from_palette(
    palette: list[dict[str, Any]], component_names: list[str]
) -> set[int]:
    """Find the class ids the editor is allowed to erase.

    Args:
        palette (list[dict[str, Any]]): Output of
            :func:`load_furniture_palette`.
        component_names (list[str]): Registry keys of the components the page
            runs, e.g. ``["tv", "showcase", "dining_set"]``.

    Returns:
        set[int]: Class ids whose palette name matches one of the component
        names. The browser uses this set to refuse painting over furniture the
        model does not own.
    """
    wanted: set[str] = set(component_names)
    return {
        int(entry["id"])
        for entry in palette
        if str(entry["name"]) in wanted
    }


def validate_inpaint_edit(
    original_ids: torch.Tensor,
    edited_ids: torch.Tensor,
    erasable_channels: set[int],
) -> None:
    """Check that an edited furniture mask only erased erasable components.

    A legal edit changes a pixel whose original class is one of
    ``erasable_channels`` to :data:`BACKGROUND_CLASS_ID`. Every other change is
    rejected, because the inpainting mode is not a general mask editor: it may
    only remove the components whose models the page runs.

    Args:
        original_ids (torch.Tensor): ``(H, W)`` int64 original furniture ids.
        edited_ids (torch.Tensor): ``(H, W)`` int64 edited furniture ids.
        erasable_channels (set[int]): Class ids the page may erase.

    Raises:
        ValueError: If the shapes differ, or if any changed pixel is not an
            erasable component changed to background.
    """
    if original_ids.shape != edited_ids.shape:
        raise ValueError(
            "The original and edited furniture masks must have the same shape; "
            f"got {tuple(original_ids.shape)} and {tuple(edited_ids.shape)}."
        )
    if not erasable_channels:
        raise ValueError("No erasable component channels were supplied.")

    changed: torch.Tensor = original_ids != edited_ids
    if not bool(changed.any()):
        return

    allowed: torch.Tensor = torch.zeros_like(changed)
    for channel in erasable_channels:
        allowed |= (original_ids == int(channel)) & (
            edited_ids == BACKGROUND_CLASS_ID
        )
    illegal: torch.Tensor = changed & ~allowed
    if bool(illegal.any()):
        raise ValueError(
            "Inpaint editing may only erase the selected component classes "
            f"({sorted(erasable_channels)}) to background; "
            f"{int(illegal.sum())} pixel(s) changed in another way."
        )


def merge_component_mask(
    base_ids: torch.Tensor,
    channel: int,
    generated_mask: torch.Tensor,
    protect_original: bool = True,
) -> torch.Tensor:
    """Apply the inpainting merge rule to one component.

    The rule is ``remaining_original OR generated``. With
    ``protect_original=True`` the generated pixels are restricted to background
    pixels, so a component model cannot overwrite furniture that survived the
    erasure. Setting it to ``False`` follows the literal set rule and lets a
    generated component claim any pixel that is not its own remaining original
    channel.

    Args:
        base_ids (torch.Tensor): ``(H, W)`` int64 edited furniture ids.
        channel (int): Component class id in the furniture layout.
        generated_mask (torch.Tensor): ``(H, W)`` bool generated mask for this
            component.
        protect_original (bool): Whether original non-background pixels win
            over generated pixels. Defaults to ``True``.

    Returns:
        torch.Tensor: ``(H, W)`` bool mask of the component after the merge.

    Raises:
        ValueError: If the tensors do not have the same spatial shape.
    """
    if base_ids.shape != generated_mask.shape:
        raise ValueError(
            "The base furniture mask and the generated component mask must "
            f"have the same shape; got {tuple(base_ids.shape)} and "
            f"{tuple(generated_mask.shape)}."
        )
    generated_mask = generated_mask.to(
        device=base_ids.device, dtype=torch.bool
    )
    remaining_original: torch.Tensor = base_ids == int(channel)
    if protect_original:
        candidate: torch.Tensor = generated_mask & (
            base_ids == BACKGROUND_CLASS_ID
        )
    else:
        candidate = generated_mask & ~remaining_original
    return remaining_original | candidate


def merge_inpainted_components(
    base_ids: torch.Tensor,
    component_channels: dict[str, int],
    generated_masks: dict[int, torch.Tensor],
    probabilities: dict[int, torch.Tensor],
    protect_original: bool = True,
) -> torch.Tensor:
    """Merge every generated component into one furniture mask.

    Each component contributes ``remaining_original OR generated``. When two
    components generate the same background pixel, the pixel goes to the
    component with the higher ensemble probability; a tie keeps the component
    that appears first in ``component_channels``. Original pixels always win
    when ``protect_original`` is true, which is the default because the three
    component models are trained separately and a stray generated TV must not
    delete a sofa that the user did not erase.

    Args:
        base_ids (torch.Tensor): ``(H, W)`` int64 edited furniture ids.
        component_channels (dict[str, int]): Component name -> furniture class
            id, in merge-priority order for ties.
        generated_masks (dict[int, torch.Tensor]): Class id -> ``(H, W)`` bool
            generated mask.
        probabilities (dict[int, torch.Tensor]): Class id -> ``(H, W)`` float
            ensemble probability map, used to resolve overlaps.
        protect_original (bool): Whether original non-background pixels win
            over generated pixels. Defaults to ``True``.

    Returns:
        torch.Tensor: ``(H, W)`` int64 merged furniture ids.

    Raises:
        ValueError: If a component has no generated mask/probability, or if a
            tensor shape disagrees with ``base_ids``.
    """
    if not component_channels:
        return base_ids.clone()

    merged: torch.Tensor = base_ids.clone()
    best_probability: torch.Tensor = torch.full(
        base_ids.shape,
        -1.0,
        dtype=torch.float32,
        device=base_ids.device,
    )
    best_channel: torch.Tensor = torch.zeros_like(base_ids)

    for component_name, channel in component_channels.items():
        channel = int(channel)
        if channel not in generated_masks:
            raise ValueError(
                f"No generated mask was supplied for component "
                f"{component_name!r} (channel {channel})."
            )
        if channel not in probabilities:
            raise ValueError(
                f"No ensemble probability map was supplied for component "
                f"{component_name!r} (channel {channel})."
            )
        generated: torch.Tensor = generated_masks[channel].to(
            device=base_ids.device, dtype=torch.bool
        )
        probability: torch.Tensor = probabilities[channel].to(
            device=base_ids.device, dtype=torch.float32
        )
        if (
            generated.shape != base_ids.shape
            or probability.shape != base_ids.shape
        ):
            raise ValueError(
                f"Component {component_name!r} has shapes "
                f"{tuple(generated.shape)} and {tuple(probability.shape)}, "
                f"but the furniture mask is {tuple(base_ids.shape)}."
            )

        if protect_original:
            candidate: torch.Tensor = generated & (
                base_ids == BACKGROUND_CLASS_ID
            )
        else:
            candidate = generated & (base_ids != channel)

        # Strictly greater keeps the first component on a tie, which makes the
        # merge deterministic for a given registry order.
        better: torch.Tensor = candidate & (probability > best_probability)
        best_probability[better] = probability[better]
        best_channel[better] = channel

    fill: torch.Tensor = best_channel != BACKGROUND_CLASS_ID
    merged[fill] = best_channel[fill]
    return merged
