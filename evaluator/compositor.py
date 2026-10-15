"""Compositing the image YOLO will judge.

For every test sample and every diffusion draw the evaluator builds ONE image:

1. the floorplan the layout belongs to, rendered from its class ids;
2. the real furniture, with the missing component's pixels remapped to
   background - exactly the layout the freezing partial encoder saw, so no
   ground-truth trace of the component can leak into the image;
3. the generated component painted on top.

Everything is drawn at the same blend weight the detector was trained with
(``COMPOSITE_OVERLAY_ALPHA``), because an evaluated image that does not look
like the training domain would measure the domain shift rather than the
component. The generated component is layered last and wins every overlap, so a
misplaced component is never hidden behind the furniture it was placed against.
"""

from __future__ import annotations

import torch
from PIL import Image

from mask_diffusion.inference import MaskDiffusionSampler
from mask_diffusion.runner import class_ids_to_one_hot, overlay_component_on_base
from mask_VAE.masking import MaskGroup, render_group_rgb

# Background channel of every group layout: local class id 0.
BACKGROUND_CHANNEL: int = 0


def erase_component_channel(one_hot: torch.Tensor, channel: int) -> torch.Tensor:
    """Remove one component's channel from a one-hot mask batch.

    The component's pixels are folded into the background channel, which is
    exactly the rule the partial VAE's layout applies when it encodes the same
    mask (``remap_ids_to_background``). Keeping the two identical is what makes
    the evaluated image a faithful picture of the conditioning: nothing of the
    real component can survive anywhere in it.

    Args:
        one_hot (torch.Tensor): ``(B, C, H, W)`` binary mask batch with exactly
            one active channel per pixel.
        channel (int): Channel to erase; must not be the background channel.

    Returns:
        torch.Tensor: A new ``(B, C, H, W)`` tensor in which ``channel`` is all
            zeros and the background channel carries its former pixels.

    Raises:
        ValueError: If ``channel`` is the background channel.
    """
    if channel == BACKGROUND_CHANNEL:
        raise ValueError(
            "The background channel cannot be erased; erasing it would leave the "
            "mask with no background at all."
        )
    erased: torch.Tensor = one_hot.clone()
    erased[:, BACKGROUND_CHANNEL] = erased[:, BACKGROUND_CHANNEL] + erased[:, channel]
    erased[:, channel] = 0.0
    return erased


def render_layout_ids(
    ids: torch.Tensor,
    group: MaskGroup,
    num_channels: int,
    erase_channel: int | None = None,
) -> torch.Tensor:
    """Render class ids as an RGB image batch through one group's colour table.

    Args:
        ids (torch.Tensor): ``(B, H, W)`` int64 class ids.
        group (MaskGroup): Group whose colours and channel order are used.
        num_channels (int): Channel count of the layout that ``ids`` belong to.
        erase_channel (int | None): Optional channel to render as background.

    Returns:
        torch.Tensor: ``(B, 3, H, W)`` float RGB in ``0..1`` on CPU.
    """
    one_hot: torch.Tensor = class_ids_to_one_hot(ids, num_channels)
    if erase_channel is not None:
        one_hot = erase_component_channel(one_hot, erase_channel)
    cpu: torch.device = torch.device("cpu")
    return render_group_rgb(one_hot.to(cpu), group, cpu)


def component_rgb(
    mask: torch.Tensor, group: MaskGroup, channel: int, num_channels: int
) -> torch.Tensor:
    """Render a single component's binary mask in that component's own colour.

    Args:
        mask (torch.Tensor): ``(B, H, W)`` bool/float mask, non-zero where the
            component was generated.
        group (MaskGroup): Group supplying the component's colour.
        channel (int): Channel of the component inside its layout.
        num_channels (int): Channel count of the layout the mask belongs to.

    Returns:
        torch.Tensor: ``(B, 3, H, W)`` RGB in ``0..1``, coloured only where the
            mask is set and background (black) everywhere else.
    """
    binary: torch.Tensor = mask.reshape(mask.shape[0], 1, *mask.shape[-2:]).to(
        torch.float32
    )
    tensor: torch.Tensor = torch.zeros(
        (binary.shape[0], num_channels, *binary.shape[-2:]), dtype=torch.float32
    )
    tensor[:, channel] = binary[:, 0]
    tensor[:, BACKGROUND_CHANNEL] = 1.0 - binary[:, 0]
    return render_group_rgb(tensor, group, torch.device("cpu"))


def tensor_to_image(rgb: torch.Tensor) -> Image.Image:
    """Convert an RGB float batch to PIL images.

    Args:
        rgb (torch.Tensor): ``(B, 3, H, W)`` float tensor in ``0..1``.

    Returns:
        Image.Image: Only the first image of the batch (the compositor works on
            one sample at a time).
    """
    return Image.fromarray(
        (rgb[0].clamp(0.0, 1.0).permute(1, 2, 0).mul(255).to(torch.uint8).numpy())
    )


def build_composite(
    floorplan_ids: torch.Tensor,
    furniture_ids: torch.Tensor,
    generated_mask: torch.Tensor,
    sampler: MaskDiffusionSampler,
    furniture_alpha: float,
    generated_alpha: float,
) -> Image.Image:
    """Build the one image YOLO is evaluated on.

    Args:
        floorplan_ids (torch.Tensor): ``(1, H, W)`` int64 floorplan class ids.
        furniture_ids (torch.Tensor): ``(1, H, W)`` int64 furniture class ids of
            the real layout, missing component still present.
        generated_mask (torch.Tensor): ``(1, H, W)`` bool mask of the generated
            component for this draw.
        sampler (MaskDiffusionSampler): Supplies the two layouts, their channel
            counts and the missing channel.
        furniture_alpha (float): Blend weight of the erased real furniture over
            the floorplan.
        generated_alpha (float): Blend weight of the generated component over
            the furnished floorplan.

    Returns:
        Image.Image: The 256x256 RGB composite image.
    """
    channel: int = sampler.missing_channel
    floorplan_group: MaskGroup = sampler.floorplan_layout.groups[0]
    floor_rgb: torch.Tensor = render_layout_ids(
        floorplan_ids, floorplan_group, sampler.floorplan_layout.num_channels
    )
    furniture_rgb: torch.Tensor = render_layout_ids(
        furniture_ids,
        sampler.target_group,
        sampler.target_layout.num_channels,
        erase_channel=channel,
    )
    # Foreground is taken from the ids rather than by testing the rendered RGB
    # for non-black: the ids say exactly which pixels are furniture.
    furniture_foreground: torch.Tensor = (
        (furniture_ids != 0) & (furniture_ids != channel)
    ).to(floor_rgb.dtype).unsqueeze(1)
    furnished: torch.Tensor = overlay_component_on_base(
        floor_rgb, furniture_rgb, furniture_foreground, furniture_alpha
    )
    generated: torch.Tensor = component_rgb(
        generated_mask,
        sampler.target_group,
        channel,
        sampler.target_layout.num_channels,
    )
    composite: torch.Tensor = overlay_component_on_base(
        furnished, generated, generated_mask.to(furnished.dtype), generated_alpha
    )
    return tensor_to_image(composite)


def build_ground_truth_overlay(
    floorplan_ids: torch.Tensor,
    furniture_ids: torch.Tensor,
    sampler: MaskDiffusionSampler,
    alpha: float,
) -> Image.Image:
    """Build the full-layout reference image, missing component included.

    This is what the composite would look like if the component were real: it
    is the side of every visualization the composite is judged against.

    Args:
        floorplan_ids (torch.Tensor): ``(1, H, W)`` int64 floorplan class ids.
        furniture_ids (torch.Tensor): ``(1, H, W)`` int64 furniture class ids.
        sampler (MaskDiffusionSampler): Supplies both layouts.
        alpha (float): Blend weight of furniture over the floorplan.

    Returns:
        Image.Image: The 256x256 RGB floorplan + full-layout overlay.
    """
    floorplan_group: MaskGroup = sampler.floorplan_layout.groups[0]
    floor_rgb: torch.Tensor = render_layout_ids(
        floorplan_ids, floorplan_group, sampler.floorplan_layout.num_channels
    )
    furniture_rgb: torch.Tensor = render_layout_ids(
        furniture_ids, sampler.target_group, sampler.target_layout.num_channels
    )
    foreground: torch.Tensor = (furniture_ids != 0).to(floor_rgb.dtype).unsqueeze(1)
    return tensor_to_image(
        overlay_component_on_base(floor_rgb, furniture_rgb, foreground, alpha)
    )


def build_conditioned_overlay(
    floorplan_ids: torch.Tensor,
    furniture_ids: torch.Tensor,
    sampler: MaskDiffusionSampler,
    alpha: float,
) -> Image.Image:
    """Build the layout the model was actually conditioned on.

    Every furniture channel except the missing component, over the floorplan -
    the visible statement of "what the model had to work with".

    Args:
        floorplan_ids (torch.Tensor): ``(1, H, W)`` int64 floorplan class ids.
        furniture_ids (torch.Tensor): ``(1, H, W)`` int64 furniture class ids.
        sampler (MaskDiffusionSampler): Supplies both layouts.
        alpha (float): Blend weight of furniture over the floorplan.

    Returns:
        Image.Image: The 256x256 RGB floorplan + conditioned-layout overlay.
    """
    channel: int = sampler.missing_channel
    floorplan_group: MaskGroup = sampler.floorplan_layout.groups[0]
    floor_rgb: torch.Tensor = render_layout_ids(
        floorplan_ids, floorplan_group, sampler.floorplan_layout.num_channels
    )
    furniture_rgb: torch.Tensor = render_layout_ids(
        furniture_ids,
        sampler.target_group,
        sampler.target_layout.num_channels,
        erase_channel=channel,
    )
    foreground: torch.Tensor = (
        (furniture_ids != 0) & (furniture_ids != channel)
    ).to(floor_rgb.dtype).unsqueeze(1)
    return tensor_to_image(
        overlay_component_on_base(floor_rgb, furniture_rgb, foreground, alpha)
    )
