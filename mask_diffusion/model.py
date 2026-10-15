"""Missing-component diffusion model over frozen mask VAE latent vectors.

The goal is to generate a MISSING furniture component (the TV) rather than to
generate a whole arrangement. Three frozen mask VAEs define the spaces this
module works in, all with 128-dim flat latents:

* ``floorplan``  - encodes ``floorplan_segmentation.png`` (18 class ids).
* ``partial``    - encodes ``furniture_segmentation.png`` through a layout whose
  ``tv`` class is remapped to background, so the TV is genuinely absent.
* ``target``     - encodes the SAME ``furniture_segmentation.png`` through a
  layout that keeps the TV. This is the diffusion space.

The model therefore learns ``p(z_target | z_floorplan, z_partial)`` with a
denoising diffusion probabilistic model (DDPM, Ho et al. 2020) whose denoiser
is conditioned on TWO latent vectors instead of one. Sampling runs the reverse
chain from pure noise and the result is decoded by the frozen target decoder,
which now also contains the TV.

Training runs TWO objectives in sequence (see ``LOSS_SCHEDULE`` in the
hyperparameters, and the runner for the loop):

* the **latent MSE** - this model's native output, the noise estimate, scored
  against the noise that was actually added. Timesteps span the FULL schedule.
* the **pixel BCE** - ``x0_from_eps`` estimates the clean latent, the frozen
  target decoder turns it into mask logits, and those logits feed a weighted
  composite (``MSE_WEIGHT`` on the noise pair, ``BCE_WEIGHT`` on the whole mask,
  ``TV_BCE_WEIGHT`` on the missing channel). That is why the decoder must stay
  in the autograd graph, and why this epoch caps its timesteps at
  ``BCE_TRAIN_T_MAX`` - a decoded BCE is only meaningful while the estimate is
  still close to the latent manifold.

Conditioning is concatenation: ``[x_t, z_floorplan, z_partial]`` is projected
to the hidden width at the entry, and each residual block additionally re-adds
projected time, floorplan and partial embeddings so every layer can weigh the
three signals differently.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

# The shared time embedding and beta schedule are pure functions of their
# arguments and identical to the ones conditional_diffusion needs. Importing
# them keeps a single implementation in the repo instead of a copy that can
# silently drift apart.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from conditional_diffusion.model import (  # noqa: E402
    SinusoidalTimeEmbedding,
    build_linear_beta_schedule,
)


class MaskDenoiserBlock(nn.Module):
    """One residual block conditioned on the timestep and BOTH latent conditions.

    Pre-norm layout: the hidden state is layer-normed, shifted by projected
    time, projected floorplan and projected partial embeddings, passed through
    a two-layer feedforward with dropout, and added back to the input so deep
    stacks keep a clean gradient path. Dropout is training-only.
    """

    def __init__(
        self,
        hidden_dim: int,
        time_emb_dim: int,
        cond_dim: int,
        dropout: float,
    ) -> None:
        """Initialize the block layers.

        Args:
            hidden_dim (int): Width of the denoiser hidden representation.
            time_emb_dim (int): Size of the incoming time embedding vector.
            cond_dim (int): Width of each per-condition embedding entering the
                block. Both condition embeddings share this width.
            dropout (float): Dropout probability inside the feedforward.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        # Each block re-projects every conditioning signal so different layers
        # can rely on them differently.
        self.time_projection = nn.Linear(time_emb_dim, hidden_dim)
        self.floorplan_projection = nn.Linear(cond_dim, hidden_dim)
        self.partial_projection = nn.Linear(cond_dim, hidden_dim)
        self.feedforward = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        time_emb: torch.Tensor,
        floorplan_emb: torch.Tensor,
        partial_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Run one doubly-conditioned residual step.

        Args:
            hidden (torch.Tensor): Hidden states, shape ``(batch_size, hidden_dim)``.
            time_emb (torch.Tensor): Time embeddings, ``(batch_size, time_emb_dim)``.
            floorplan_emb (torch.Tensor): Floorplan embeddings, ``(batch_size, cond_dim)``.
            partial_emb (torch.Tensor): Missing-component-free layout embeddings,
                shape ``(batch_size, cond_dim)``.

        Returns:
            torch.Tensor: Updated hidden states, same shape as ``hidden``.
        """
        conditioned: torch.Tensor = (
            self.norm(hidden)
            + self.time_projection(time_emb)
            + self.floorplan_projection(floorplan_emb)
            + self.partial_projection(partial_emb)
        )
        return hidden + self.feedforward(conditioned)


class MaskDenoiser(nn.Module):
    """MLP predicting diffusion noise from ``(x_t, t, z_floorplan, z_partial)``.

    The noisy latent and both conditions are concatenated at the entry, each
    condition additionally gets its own embedding reused by every block, and a
    linear head maps back to the latent dimension. The head is zero-initialized
    so the first prediction is the data mean rather than large random noise.
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        hidden_dim: int,
        num_layers: int,
        time_emb_dim: int,
        max_period: int,
        dropout: float,
    ) -> None:
        """Initialize the doubly-conditioned denoiser.

        Args:
            latent_dim (int): Dimensionality of the target (TV-inclusive) latents.
            cond_dim (int): Dimensionality of EACH condition latent. The two
                conditions are concatenated, so the entry projection is
                ``latent_dim + 2 * cond_dim`` wide.
            hidden_dim (int): Width of the hidden MLP representation.
            num_layers (int): Number of stacked residual blocks.
            time_emb_dim (int): Size of the timestep embedding.
            max_period (int): Longest sinusoid period of the time embedding.
            dropout (float): Dropout probability at the entry projection and
                inside each residual block.

        Raises:
            ValueError: If ``num_layers`` is smaller than 1.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim, max_period)
        # Entry projection sees the noisy target and BOTH conditions together.
        self.input_projection = nn.Linear(latent_dim + 2 * cond_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.floorplan_embedding = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.partial_embedding = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                MaskDenoiserBlock(hidden_dim, time_emb_dim, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)
        # Zero-init the head: predicting ~0 noise at step 0 is a far better
        # starting point than a randomly projected tensor.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        floorplan_latents: torch.Tensor,
        partial_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise present in a batch of noisy target latents.

        Args:
            noisy_latents (torch.Tensor): Noisy target batch ``x_t`` of shape
                ``(batch_size, latent_dim)``.
            timesteps (torch.Tensor): Integer timesteps ``(batch_size,)``.
            floorplan_latents (torch.Tensor): Floorplan conditions, shape
                ``(batch_size, cond_dim)``.
            partial_latents (torch.Tensor): TV-free arrangement conditions,
                shape ``(batch_size, cond_dim)``.

        Returns:
            torch.Tensor: Predicted noise, same shape as ``noisy_latents``.
        """
        time_emb: torch.Tensor = self.time_embedding(timesteps)
        floorplan_emb: torch.Tensor = self.floorplan_embedding(floorplan_latents)
        partial_emb: torch.Tensor = self.partial_embedding(partial_latents)
        hidden: torch.Tensor = self.input_dropout(
            self.input_projection(
                torch.cat([noisy_latents, floorplan_latents, partial_latents], dim=1)
            )
        )
        for block in self.blocks:
            hidden = block(hidden, time_emb, floorplan_emb, partial_emb)
        return self.output_projection(self.output_norm(hidden))


class MaskComponentDiffusion(nn.Module):
    """DDPM over TV-inclusive arrangement latents, conditioned on two latents.

    Holds the conditional denoiser and the fixed noise schedule as buffers, so
    ``.to(device)`` moves them along and checkpoints capture them.

    The training path is deliberately pixel-supervised: ``x0_from_eps`` gives the
    estimated clean latent, which the runner feeds to the frozen target decoder
    and scores with BCE. ``reverse_step``/``sample_conditional`` are used for
    evaluation and visualization only.
    """

    def __init__(self, archparams: SimpleNamespace) -> None:
        """Initialize the denoiser and precompute the schedule buffers.

        Args:
            archparams (SimpleNamespace): Exposing ``LATENT_DIM``, ``COND_DIM``,
                ``HIDDEN_DIM``, ``NUM_LAYERS``, ``TIME_EMB_DIM``,
                ``TIME_MAX_PERIOD``, ``DROPOUT``, ``NUM_TIMESTEPS``,
                ``BETA_START`` and ``BETA_END``.
        """
        super().__init__()
        self.num_timesteps: int = int(archparams.NUM_TIMESTEPS)
        self.latent_dim: int = int(archparams.LATENT_DIM)
        self.cond_dim: int = int(archparams.COND_DIM)
        self.denoiser = MaskDenoiser(
            latent_dim=self.latent_dim,
            cond_dim=self.cond_dim,
            hidden_dim=int(archparams.HIDDEN_DIM),
            num_layers=int(archparams.NUM_LAYERS),
            time_emb_dim=int(archparams.TIME_EMB_DIM),
            max_period=int(archparams.TIME_MAX_PERIOD),
            dropout=float(archparams.DROPOUT),
        )

        betas: torch.Tensor = build_linear_beta_schedule(
            self.num_timesteps,
            float(archparams.BETA_START),
            float(archparams.BETA_END),
        )
        alphas: torch.Tensor = 1.0 - betas
        alpha_bar: torch.Tensor = torch.cumprod(alphas, dim=0)
        # alpha_bar_prev[-1] is defined as 1 for the closed-form posterior
        # variance at the final step of the chain.
        alpha_bar_prev: torch.Tensor = torch.cat(
            [torch.ones(1, dtype=torch.float32), alpha_bar[:-1]]
        )
        posterior_variance: torch.Tensor = (
            betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("posterior_variance", posterior_variance)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        floorplan_latents: torch.Tensor,
        partial_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise of a noisy target batch under both conditions.

        Args:
            noisy_latents (torch.Tensor): ``x_t``, shape ``(batch_size, latent_dim)``.
            timesteps (torch.Tensor): Integer timesteps ``(batch_size,)``.
            floorplan_latents (torch.Tensor): Floorplan conditions, ``(batch_size, cond_dim)``.
            partial_latents (torch.Tensor): TV-free layout conditions, ``(batch_size, cond_dim)``.

        Returns:
            torch.Tensor: Predicted noise, same shape as ``noisy_latents``.
        """
        return self.denoiser(noisy_latents, timesteps, floorplan_latents, partial_latents)

    def q_sample(
        self,
        clean_latents: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Draw from the forward process ``q(x_t | x_0)`` in closed form.

        Uses ``x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise``.

        Args:
            clean_latents (torch.Tensor): Clean target latents ``x_0``, ``(batch_size, latent_dim)``.
            timesteps (torch.Tensor): Integer timesteps ``(batch_size,)``.
            noise (torch.Tensor): Standard Gaussian noise shaped like ``clean_latents``.

        Returns:
            torch.Tensor: Noisy latents ``x_t``, same shape as ``clean_latents``.
        """
        sqrt_bar: torch.Tensor = self.sqrt_alpha_bar[timesteps].unsqueeze(1)
        sqrt_one_minus_bar: torch.Tensor = self.sqrt_one_minus_alpha_bar[timesteps].unsqueeze(1)
        return sqrt_bar * clean_latents + sqrt_one_minus_bar * noise

    def x0_from_eps(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Invert the forward process to estimate the clean latent ``x_0``.

        This is the quantity the runner decodes with the frozen target decoder.
        It is differentiable with respect to ``predicted_noise``, which is the
        whole reason the pixel-space BCE can supervise the denoiser at all.

        Args:
            noisy_latents (torch.Tensor): ``x_t``, shape ``(batch_size, latent_dim)``.
            timesteps (torch.Tensor): Integer timesteps ``(batch_size,)``.
            predicted_noise (torch.Tensor): Predicted noise, same shape as ``noisy_latents``.

        Returns:
            torch.Tensor: Estimated ``x_0``, same shape as ``noisy_latents``.
        """
        sqrt_bar: torch.Tensor = self.sqrt_alpha_bar[timesteps].unsqueeze(1)
        sqrt_one_minus_bar: torch.Tensor = self.sqrt_one_minus_alpha_bar[timesteps].unsqueeze(1)
        return (noisy_latents - sqrt_one_minus_bar * predicted_noise) / sqrt_bar

    def reverse_step(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        floorplan_latents: torch.Tensor,
        partial_latents: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Take one conditional DDPM reverse step from ``x_t`` to ``x_{t-1}``.

        Applies the Ho et al. posterior mean with the predicted noise and adds
        scaled fresh noise, except at step 0 where the chain is deterministic.

        Args:
            noisy_latents (torch.Tensor): Current ``x_t``, ``(batch_size, latent_dim)``.
            timesteps (torch.Tensor): Integer timesteps ``(batch_size,)``.
            floorplan_latents (torch.Tensor): Floorplan conditions, ``(batch_size, cond_dim)``.
            partial_latents (torch.Tensor): TV-free layout conditions, ``(batch_size, cond_dim)``.
            generator (torch.Generator | None): Optional device-resident RNG.
                When given, the fresh noise is drawn from it so an evaluation
                caller can make a chain reproducible without depending on the
                global RNG state; ``None`` keeps the default global-generator
                behaviour used by training.

        Returns:
            torch.Tensor: Previous-step latents ``x_{t-1}``, same shape as the input.
        """
        predicted_noise: torch.Tensor = self.denoiser(
            noisy_latents, timesteps, floorplan_latents, partial_latents
        )
        beta_t: torch.Tensor = self.betas[timesteps].unsqueeze(1)
        alpha_t: torch.Tensor = self.alphas[timesteps].unsqueeze(1)
        sqrt_one_minus_bar: torch.Tensor = self.sqrt_one_minus_alpha_bar[timesteps].unsqueeze(1)
        mean: torch.Tensor = (
            noisy_latents - beta_t / sqrt_one_minus_bar * predicted_noise
        ) / torch.sqrt(alpha_t)
        # No fresh noise once the chain has reached its final step.
        is_nonzero_step: torch.Tensor = (timesteps > 0).float().unsqueeze(1)
        variance: torch.Tensor = self.posterior_variance[timesteps].unsqueeze(1)
        fresh_noise: torch.Tensor = torch.randn(
            noisy_latents.shape,
            device=noisy_latents.device,
            generator=generator,
        )
        return mean + is_nonzero_step * torch.sqrt(variance.clamp_min(1e-20)) * fresh_noise

    @torch.no_grad()
    def sample_conditional(
        self,
        floorplan_latents: torch.Tensor,
        partial_latents: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate TV-inclusive arrangement latents from both conditions.

        Runs the FULL reverse chain from pure noise, holding both conditions
        fixed throughout.

        Args:
            floorplan_latents (torch.Tensor): Floorplan conditions on the model
                device, shape ``(num_samples, cond_dim)``.
            partial_latents (torch.Tensor): TV-free layout conditions, shape
                ``(num_samples, cond_dim)``.
            generator (torch.Generator | None): Optional device-resident RNG used
                for the initial noise and every reverse step, so an evaluation
                caller can reproduce a chain exactly. ``None`` keeps the default
                global-generator behaviour used by training.

        Returns:
            torch.Tensor: Generated latents, shape ``(num_samples, latent_dim)``.
        """
        device: torch.device = floorplan_latents.device
        num_samples: int = floorplan_latents.shape[0]
        latents: torch.Tensor = torch.randn(
            num_samples, self.latent_dim, device=device, generator=generator
        )
        # Walking backwards means the last update lands on x_0 exactly.
        for step in reversed(range(self.num_timesteps)):
            timesteps: torch.Tensor = torch.full(
                (num_samples,), step, device=device, dtype=torch.long
            )
            latents = self.reverse_step(
                latents, timesteps, floorplan_latents, partial_latents, generator
            )
        return latents