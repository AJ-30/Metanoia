"""Training objective: foreground-weighted v-prediction MSE + auxiliary type loss.

Most target pixels are empty (-1 everywhere except the missing part), so an
unweighted loss is dominated by "predict nothing". Pixels within
``FOREGROUND_DILATION_PX`` of the missing part therefore weigh
``1 + FOREGROUND_WEIGHT`` times more. The weighted mean is normalized by the
total weight, so the loss scale does not depend on the missing part's size.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from Reconstruction.model.diffusion import GaussianDiffusion


@dataclass
class LossTerms:
    """Loss values of one batch.

    Attributes:
        total (torch.Tensor): Scalar loss to back-propagate.
        diffusion (torch.Tensor): Weighted v-MSE (detached).
        type_ce (torch.Tensor): Type cross-entropy (detached).
        type_accuracy (torch.Tensor): Type head accuracy (detached).
    """

    total: torch.Tensor
    diffusion: torch.Tensor
    type_ce: torch.Tensor
    type_accuracy: torch.Tensor


def to_model_scale(maps: torch.Tensor) -> torch.Tensor:
    """Map {0, 1} semantic maps to {-1, +1}.

    Args:
        maps (torch.Tensor): Maps in [0, 1].

    Returns:
        torch.Tensor: Maps in [-1, 1].
    """
    return maps * 2.0 - 1.0


def foreground_weights(target: torch.Tensor, weight: float, dilation_px: int) -> torch.Tensor:
    """Per-pixel loss weights emphasizing the missing part and its surroundings.

    Args:
        target (torch.Tensor): (B, C, H, W) missing part in {0, 1}.
        weight (float): Extra weight of foreground pixels.
        dilation_px (int): Foreground dilation radius (square).

    Returns:
        torch.Tensor: (B, 1, H, W) weights >= 1.
    """
    foreground: torch.Tensor = target.amax(dim=1, keepdim=True)
    if dilation_px > 0:
        foreground = F.max_pool2d(foreground, 2 * dilation_px + 1, stride=1, padding=dilation_px)
    return 1.0 + weight * foreground


def compute_loss(model: nn.Module, diffusion: GaussianDiffusion, incomplete: torch.Tensor, target: torch.Tensor,
                 type_labels: torch.Tensor, cfg: SimpleNamespace) -> LossTerms:
    """Sample timesteps and noise, run the model, and compute the loss.

    Args:
        model (nn.Module): ``ConditionalUNet`` (possibly DDP-wrapped).
        diffusion (GaussianDiffusion): Diffusion process.
        incomplete (torch.Tensor): (B, C, H, W) input plan in {0, 1}.
        target (torch.Tensor): (B, C, H, W) missing part in {0, 1}.
        type_labels (torch.Tensor): (B,) missing-type class indices.
        cfg (SimpleNamespace): ``[loss]`` hyperparameters.

    Returns:
        LossTerms: Loss values.
    """
    x0: torch.Tensor = to_model_scale(target)
    t: torch.Tensor = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=x0.device)
    noise: torch.Tensor = torch.randn_like(x0)
    v_pred, type_logits = model(diffusion.q_sample(x0, t, noise), to_model_scale(incomplete), t)
    weights: torch.Tensor = foreground_weights(target, cfg.FOREGROUND_WEIGHT, cfg.FOREGROUND_DILATION_PX)
    per_pixel: torch.Tensor = (v_pred.float() - diffusion.v_target(x0, t, noise)).pow(2).mean(dim=1, keepdim=True)
    diffusion_loss: torch.Tensor = (weights * per_pixel).sum() / weights.expand_as(per_pixel).sum()
    type_ce: torch.Tensor = F.cross_entropy(type_logits.float(), type_labels)
    accuracy: torch.Tensor = (type_logits.argmax(dim=1) == type_labels).float().mean()
    total: torch.Tensor = diffusion_loss + cfg.TYPE_LOSS_WEIGHT * type_ce
    return LossTerms(total, diffusion_loss.detach(), type_ce.detach(), accuracy.detach())
