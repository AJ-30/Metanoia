"""Training loss: foreground-weighted v-prediction MSE + auxiliary missing-type loss.

Same loss as ``model/objective.py`` (kept here, with the type accuracy that
``train.py`` logs, so this package does not depend on that file's current state).
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from Reconstruction.model.diffusion import GaussianDiffusion
from Reconstruction.model.objective import foreground_weights, to_model_scale


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


def compute_loss(model: nn.Module, diffusion: GaussianDiffusion, incomplete: torch.Tensor, target: torch.Tensor,
                 type_labels: torch.Tensor, cfg: SimpleNamespace) -> LossTerms:
    """Sample timesteps and noise, run the model, and compute the loss.

    Args:
        model (nn.Module): Model (possibly DDP-wrapped) returning (v, type logits).
        diffusion (GaussianDiffusion): Diffusion process.
        incomplete (torch.Tensor): (B, C, H, W) input plans in {0, 1}.
        target (torch.Tensor): (B, C, H, W) missing part in {0, 1}.
        type_labels (torch.Tensor): (B,) missing-type class indices.
        cfg (SimpleNamespace): ``[loss]`` hyperparameters.

    Returns:
        LossTerms: Loss terms.
    """
    x0 = to_model_scale(target)
    t = torch.randint(0, diffusion.timesteps, (x0.shape[0],), device=x0.device)
    noise = torch.randn_like(x0)
    v_pred, type_logits = model(diffusion.q_sample(x0, t, noise), to_model_scale(incomplete), t)
    weights = foreground_weights(target, cfg.FOREGROUND_WEIGHT, cfg.FOREGROUND_DILATION_PX)
    per_pixel = (v_pred.float() - diffusion.v_target(x0, t, noise)).pow(2).mean(dim=1, keepdim=True)
    diffusion_loss = (weights * per_pixel).sum() / weights.expand_as(per_pixel).sum()
    type_ce = F.cross_entropy(type_logits.float(), type_labels)
    accuracy = (type_logits.argmax(dim=1) == type_labels).float().mean()
    total = diffusion_loss + cfg.TYPE_LOSS_WEIGHT * type_ce
    return LossTerms(total, diffusion_loss.detach(), type_ce.detach(), accuracy.detach())
