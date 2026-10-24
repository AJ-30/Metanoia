"""Gaussian diffusion with a cosine schedule, v-parameterization and DDIM sampling.

Data scale: every semantic map in {0, 1} is mapped to {-1, +1}. The model
generates only the missing part x0 = 2R - 1, conditioned on the incomplete
plan 2I - 1 (see ``ConditionalUNet``).

v-parameterization (Salimans & Ho 2022): with a_t = sqrt(alpha_bar_t) and
s_t = sqrt(1 - alpha_bar_t),

    x_t = a_t x0 + s_t eps,   v = a_t eps - s_t x0,
    x0  = a_t x_t - s_t v,    eps = s_t x_t + a_t v.

Sampling uses deterministic DDIM (eta = 0) on an evenly spaced subset of the
training timesteps.
"""

import math
from typing import Callable
import torch

# model(x_t, t) -> predicted v;
DenoiseFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
StepCallback = Callable[[int, torch.Tensor], None]


def cosine_alpha_bar(timesteps: int, max_beta: float) -> torch.Tensor:
    """Cumulative signal fractions alpha_bar_t of the cosine schedule.

    Args:
        timesteps (int): Number of diffusion steps T.
        max_beta (float): Upper clip for per-step betas (avoids singularity at t = T).

    Returns:
        torch.Tensor: float64 (T,) alpha_bar values, decreasing from ~1 to ~0.
    """
    offset: float = 0.008  # keeps betas from being too small near t = 0
    f = lambda t: math.cos((t / timesteps + offset) / (1 + offset) * math.pi / 2) ** 2
    betas: list[float] = [min(1 - f(i + 1) / f(i), max_beta) for i in range(timesteps)]
    return torch.cumprod(1 - torch.tensor(betas, dtype=torch.float64), dim=0)


class GaussianDiffusion:
    """Forward noising, training targets and DDIM sampling."""

    def __init__(self, timesteps: int, max_beta: float) -> None:
        """Pre-compute the schedule.

        Args:
            timesteps (int): Number of diffusion steps T.
            max_beta (float): Beta clip of the cosine schedule.
        """
        self.timesteps: int = timesteps
        alpha_bar= cosine_alpha_bar(timesteps, max_beta)
        self.sqrt_alpha_bar= alpha_bar.sqrt().float()
        self.sqrt_one_minus= (1 - alpha_bar).sqrt().float()

    def _coef(self, values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """To broadcast per-sample schedule to multiply later to img (B, C, H, W).
        for (B,)timesteps to get (B, 1, 1, 1) values

        """
        return values.to(t.device)[t][:, None, None, None]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Noise clean data to step t.

        Args:
            x0 (torch.Tensor): (B, C, H, W) clean data in [-1, 1].
            t (torch.Tensor): (B,) timesteps.
            noise (torch.Tensor): Standard normal noise like ``x0``.

        Returns:
            torch.Tensor: x_t.
        """
        return self._coef(self.sqrt_alpha_bar, t) * x0 + self._coef(self.sqrt_one_minus, t) * noise

    def v_target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Training target v for (x0, noise, t).

        Args:
            x0 (torch.Tensor): Clean data.
            t (torch.Tensor): Timesteps.
            noise (torch.Tensor): Noise used in ``q_sample``.

        Returns:
            torch.Tensor: v.
        """
        return self._coef(self.sqrt_alpha_bar, t) * noise - self._coef(self.sqrt_one_minus, t) * x0

    def x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Recover the clean-data estimate from a v prediction.

        Args:
            x_t (torch.Tensor): Noisy data.
            t (torch.Tensor): Timesteps.
            v (torch.Tensor): Pred v.

        Returns:
            torch.Tensor: x0 estimate.
        """
        return self._coef(self.sqrt_alpha_bar, t) * x_t - self._coef(self.sqrt_one_minus, t) * v

    @torch.no_grad()
    def ddim_sample(self, denoise: DenoiseFn, shape: tuple[int, ...], steps: int, device: torch.device,
                    generator: torch.Generator | None, clip_x0: bool, on_step: StepCallback | None = None) -> torch.Tensor:
        """Deterministic DDIM sampling from pure noise.

        Args:
            denoise (DenoiseFn): Returns predicted v for (x_t, t).
            shape (tuple[int, ...]): Output shape (B, C, H, W).
            steps (int): Number of sampling steps (<= T).
            device (torch.device): Device.
            generator (torch.Generator | None): RNG for the initial noise.
            clip_x0 (bool): Clip x0 estimates to [-1, 1].
            on_step (StepCallback | None): Receives (t, x0 estimate) at every
                step, e.g. to record how the prediction forms while denoising.

        Returns:
            torch.Tensor: (B, C, H, W) samples in [-1, 1] (approximately).
        """
        assert 1 <= steps <= self.timesteps, "steps must be in [1, T]"
        x = torch.randn(shape, device=device, generator=generator)
        schedule: list[int] = torch.linspace(self.timesteps - 1, 0, steps).round().long().tolist()
        # ^chooses which 50 of 1k timesteps to visit when sampling
        for i, step in enumerate(schedule):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            # During training t is random per image. During sampling every image in the batch is at the same point of the denoising process, so the same step number is repeated B times.
            x0 = self.x0_from_v(x, t, denoise(x, t).float())
            if clip_x0:
                x0 = x0.clamp(-1.0, 1.0)
            if on_step is not None:
                on_step(step, x0)
            if i == len(schedule) - 1:
                return x0
            a_t, s_t = self._coef(self.sqrt_alpha_bar, t), self._coef(self.sqrt_one_minus, t)
            eps = (x - a_t * x0) / s_t
            t_prev = torch.full_like(t, schedule[i + 1])
            x = self._coef(self.sqrt_alpha_bar, t_prev) * x0 + self._coef(self.sqrt_one_minus, t_prev) * eps
        return x
