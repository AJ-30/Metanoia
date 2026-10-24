"""Generate missing parts for a batch of incomplete plans.

Shared by validation during training and by ``predict.py``, so both produce
predictions in exactly the same way.
"""

import torch
from torch import nn

from Reconstruction.model.diffusion import GaussianDiffusion, StepCallback
from Reconstruction.model.objective import to_model_scale


@torch.no_grad()
def sample_missing(model: nn.Module, diffusion: GaussianDiffusion, incomplete: torch.Tensor, steps: int,
                   clip_x0: bool, generator: torch.Generator | None, autocast_dtype: torch.dtype | None,
                   on_step: StepCallback | None = None) -> torch.Tensor:
    """DDIM-sample the missing part of each plan in a batch.

    Args:
        model (nn.Module): Unwrapped ``ConditionalUNet`` (e.g. the EMA copy).
        diffusion (GaussianDiffusion): Diffusion process.
        incomplete (torch.Tensor): (B, C, H, W) input plans in {0, 1}, on the model's device.
        steps (int): DDIM steps.
        clip_x0 (bool): Clip x0 estimates to [-1, 1].
        generator (torch.Generator | None): RNG for the initial noise (reproducible samples).
        autocast_dtype (torch.dtype | None): Mixed-precision dtype, or None for fp32.
        on_step (StepCallback | None): Receives (t, x0 estimate in [-1, 1]) at every step.

    Returns:
        torch.Tensor: (B, C, H, W) float32 scores in [0, 1].
    """
    condition: torch.Tensor = to_model_scale(incomplete)

    def denoise(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x_t.device.type, dtype=autocast_dtype or torch.float32, enabled=autocast_dtype is not None):
            return model(x_t, condition, t)[0]

    x0: torch.Tensor = diffusion.ddim_sample(denoise, tuple(incomplete.shape), steps, incomplete.device, generator,
                                             clip_x0, on_step)
    return ((x0.float() + 1.0) / 2.0).clamp(0.0, 1.0)
