"""
Conditional DDPM diffusion model over frozen mask_VAE latent vectors.

The floorplan mask VAE compresses each floorplan mask to a flat condition
vector ``c`` (``COND_DIM``) and the arrangement mask VAE compresses the paired
furniture mask of the SAME layout to a target vector ``x_0`` (``LATENT_DIM``).
This module learns ``p(x_0 | c)`` with a denoising diffusion probabilistic
model (DDPM, Ho et al. 2020): the forward process adds Gaussian noise to
``x_0`` on a fixed linear schedule, and the condition-aware MLP denoiser
learns to predict the added noise from ``(x_t, t, c)``. Sampling runs the
reverse chain from pure noise conditioned on a floorplan latent and the result
is decoded with the frozen arrangement decoder.

Conditioning is concatenation: ``[x_t, c]`` is linearly projected to the
hidden width, and each residual block adds projected time AND condition
embeddings. All schedule and architecture knobs come from
``conditional_diffusion/hyperparameters.toml`` via the ``archparams``
namespace; nothing is hardcoded here.
"""

import math
from types import SimpleNamespace

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a small MLP.

    Maps an integer timestep to a dense vector so the denoiser can condition
    its prediction on the noise level, mirroring the Transformer positional
    encoding (Vaswani et al. 2017) plus a two-layer projection.
    """

    def __init__(self, embedding_dim: int, max_period: int) -> None:
        """Initialize the projection layers.

        Args:
            embedding_dim (int): Size of the output embedding vector.
            max_period (int): Longest sinusoid period; controls the frequency
                spacing of the embedding.
        """
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.max_period: int = max_period
        # Two-layer projection that mixes the raw sinusoids into a learned
        # time representation. SiLU keeps gradients smooth for small values.
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed a batch of integer timesteps.

        Args:
            timesteps (torch.Tensor): Long tensor of shape (batch_size,)
                holding diffusion steps in [0, num_timesteps).

        Returns:
            torch.Tensor: Time embeddings of shape (batch_size, embedding_dim).
        """
        # Frequencies spaced geometrically from 1 to 1/max_period over half
        # the embedding width; the other half comes from the cosine terms.
        half_dim: int = self.embedding_dim // 2
        frequency_exponents: torch.Tensor = (
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / half_dim
        )
        frequencies: torch.Tensor = torch.exp(
            -math.log(self.max_period) * frequency_exponents
        )
        # Outer product of timestep magnitudes and frequencies gives the
        # sinusoid arguments, one row per timestep in the batch.
        angles: torch.Tensor = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        raw_embedding: torch.Tensor = torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=1
        )
        return self.projection(raw_embedding)


class ConditionalDenoiserBlock(nn.Module):
    """One time- and condition-aware residual block of the denoiser MLP.

    Applies LayerNorm, shifts by the projected time AND condition embeddings,
    runs a two-layer feedforward with dropout, and adds the input back so
    deeper stacks keep a clean gradient path. Dropout is training-only.
    """

    def __init__(self, hidden_dim: int, time_emb_dim: int, dropout: float) -> None:
        """Initialize the block layers.

        Args:
            hidden_dim (int): Width of the denoiser hidden representation.
            time_emb_dim (int): Size of the incoming time embedding vector.
                The condition embedding shares this width (projected once at
                the denoiser entry), so one width covers both signals.
            dropout (float): Dropout probability inside the feedforward.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        # Each block re-projects time and condition so every layer can use
        # the noise level and the floorplan signal differently.
        self.time_projection = nn.Linear(time_emb_dim, hidden_dim)
        self.cond_projection = nn.Linear(hidden_dim, hidden_dim)
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
        cond_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Run one conditional residual step.

        Args:
            hidden (torch.Tensor): Hidden states, shape (batch_size, hidden_dim).
            time_emb (torch.Tensor): Time embeddings, shape (batch_size, time_emb_dim).
            cond_emb (torch.Tensor): Condition embeddings, shape (batch_size, hidden_dim).

        Returns:
            torch.Tensor: Updated hidden states, same shape as the input.
        """
        # Pre-norm layout, then shift by both signals.
        conditioned: torch.Tensor = (
            self.norm(hidden)
            + self.time_projection(time_emb)
            + self.cond_projection(cond_emb)
        )
        return hidden + self.feedforward(conditioned)


class ConditionalLatentDenoiser(nn.Module):
    """MLP that predicts diffusion noise from (noisy_target, timestep, condition).

    Architecture: concat ``[x_t, c]`` is projected to ``hidden_dim``; the
    condition is separately embedded to ``hidden_dim`` for per-block injection;
    a stack of residual blocks shares one time embedding; a linear head maps
    back to ``latent_dim``. The output head is zero-initialized so training
    starts from a near-identity prediction.
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
        """Initialize the conditional denoiser.

        Args:
            latent_dim (int): Dimensionality of the arrangement (target) latents.
            cond_dim (int): Dimensionality of the floorplan condition latents.
            hidden_dim (int): Width of the hidden MLP representation.
            num_layers (int): Number of stacked residual blocks.
            time_emb_dim (int): Size of the timestep embedding.
            max_period (int): Longest sinusoid period of the time embedding.
            dropout (float): Dropout probability (entry projection and blocks).
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim, max_period)
        # Joint entry projection sees noisy target and condition together.
        self.input_projection = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        # Dedicated condition embedding reused by every block.
        self.cond_embedding = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [ConditionalDenoiserBlock(hidden_dim, time_emb_dim, dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)
        # Zero-init the head so the first predictions are ~0 (predicting the
        # data mean) instead of large random noise targets.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        cond_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise component of noisy arrangement latents.

        Args:
            noisy_latents (torch.Tensor): Noisy target batch, shape (batch_size, latent_dim).
            timesteps (torch.Tensor): Long timesteps, shape (batch_size,).
            cond_latents (torch.Tensor): Floorplan condition batch, shape (batch_size, cond_dim).

        Returns:
            torch.Tensor: Predicted noise, same shape as ``noisy_latents``.
        """
        time_emb: torch.Tensor = self.time_embedding(timesteps)
        cond_emb: torch.Tensor = self.cond_embedding(cond_latents)
        hidden: torch.Tensor = self.input_dropout(
            self.input_projection(torch.cat([noisy_latents, cond_latents], dim=1))
        )
        for block in self.blocks:
            hidden = block(hidden, time_emb, cond_emb)
        return self.output_projection(self.output_norm(hidden))


def build_linear_beta_schedule(
    num_timesteps: int, beta_start: float, beta_end: float
) -> torch.Tensor:
    """Build a linearly spaced DDPM noise schedule.

    Args:
        num_timesteps (int): Number of diffusion steps.
        beta_start (float): Noise variance at step 0.
        beta_end (float): Noise variance at the final step.

    Returns:
        torch.Tensor: Per-step variances of shape (num_timesteps,).
    """
    if num_timesteps < 1:
        raise ValueError(f"num_timesteps must be >= 1, got {num_timesteps}.")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError(
            f"Require 0 < beta_start < beta_end < 1, got {beta_start=} {beta_end=}."
        )
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)


class ConditionalLatentDiffusion(nn.Module):
    """DDPM wrapper holding the conditional denoiser and the fixed schedule.

    The schedule tensors are registered as buffers so ``.to(device)`` moves
    them with the model and checkpoints capture them for reproducibility.
    """

    def __init__(self, archparams: SimpleNamespace) -> None:
        """Initialize the denoiser and precompute the schedule buffers.

        Args:
            archparams (SimpleNamespace): Object exposing LATENT_DIM,
                COND_DIM, HIDDEN_DIM, NUM_LAYERS, TIME_EMB_DIM,
                TIME_MAX_PERIOD, DROPOUT, NUM_TIMESTEPS, BETA_START and
                BETA_END.
        """
        super().__init__()
        self.num_timesteps: int = int(archparams.NUM_TIMESTEPS)
        self.latent_dim: int = int(archparams.LATENT_DIM)
        self.cond_dim: int = int(archparams.COND_DIM)
        self.denoiser = ConditionalLatentDenoiser(
            latent_dim=self.latent_dim,
            cond_dim=self.cond_dim,
            hidden_dim=int(archparams.HIDDEN_DIM),
            num_layers=int(archparams.NUM_LAYERS),
            time_emb_dim=int(archparams.TIME_EMB_DIM),
            max_period=int(archparams.TIME_MAX_PERIOD),
            dropout=float(archparams.DROPOUT),
        )

        betas: torch.Tensor = build_linear_beta_schedule(
            self.num_timesteps, float(archparams.BETA_START), float(archparams.BETA_END)
        )
        alphas: torch.Tensor = 1.0 - betas
        alpha_bar: torch.Tensor = torch.cumprod(alphas, dim=0)
        # Previous-step cumulative product, with alpha_bar[-1] defined as 1
        # for the closed-form posterior variance at step 0.
        alpha_bar_prev: torch.Tensor = torch.cat(
            [torch.ones(1, dtype=torch.float32), alpha_bar[:-1]]
        )
        # DDPM posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t).
        posterior_variance: torch.Tensor = (
            betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer(
            "sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar)
        )
        self.register_buffer("posterior_variance", posterior_variance)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        cond_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise of a noisy target batch given floorplan conditions.

        Args:
            noisy_latents (torch.Tensor): Shape (batch_size, latent_dim).
            timesteps (torch.Tensor): Long timesteps, shape (batch_size,).
            cond_latents (torch.Tensor): Floorplan conditions, shape (batch_size, cond_dim).

        Returns:
            torch.Tensor: Predicted noise, same shape as ``noisy_latents``.
        """
        return self.denoiser(noisy_latents, timesteps, cond_latents)

    def q_sample(
        self, clean_latents: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Sample from the forward process q(x_t | x_0) in closed form.

        Uses x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise.

        Args:
            clean_latents (torch.Tensor): Clean arrangement latents x_0, shape (batch_size, latent_dim).
            timesteps (torch.Tensor): Long timesteps, shape (batch_size,).
            noise (torch.Tensor): Standard Gaussian noise, same shape as ``clean_latents``.

        Returns:
            torch.Tensor: Noisy latents x_t, same shape as ``clean_latents``.
        """
        # Gather the per-sample schedule scalars and broadcast over latent_dim.
        sqrt_bar: torch.Tensor = self.sqrt_alpha_bar[timesteps].unsqueeze(1)
        sqrt_one_minus_bar: torch.Tensor = self.sqrt_one_minus_alpha_bar[
            timesteps
        ].unsqueeze(1)
        return sqrt_bar * clean_latents + sqrt_one_minus_bar * noise

    def reverse_step(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        cond_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Take one conditional DDPM reverse step from x_t to x_{t-1}.

        Applies the Ho et al. posterior mean with the predicted noise and adds
        scaled fresh noise, except at step 0 where the chain is deterministic.

        Args:
            noisy_latents (torch.Tensor): Current latents x_t, shape (batch_size, latent_dim).
            timesteps (torch.Tensor): Long timesteps, shape (batch_size,).
            cond_latents (torch.Tensor): Floorplan conditions, shape (batch_size, cond_dim).

        Returns:
            torch.Tensor: Previous-step latents x_{t-1}, same shape as the input.
        """
        predicted_noise: torch.Tensor = self.denoiser(noisy_latents, timesteps, cond_latents)
        beta_t: torch.Tensor = self.betas[timesteps].unsqueeze(1)
        alpha_t: torch.Tensor = self.alphas[timesteps].unsqueeze(1)
        sqrt_one_minus_bar: torch.Tensor = self.sqrt_one_minus_alpha_bar[
            timesteps
        ].unsqueeze(1)
        # Posterior mean: undo the forward corruption with the predicted noise.
        mean: torch.Tensor = (
            noisy_latents - beta_t / sqrt_one_minus_bar * predicted_noise
        ) / torch.sqrt(alpha_t)
        # No fresh noise is injected once the chain reaches step 0.
        is_nonzero_step: torch.Tensor = (timesteps > 0).float().unsqueeze(1)
        variance: torch.Tensor = self.posterior_variance[timesteps].unsqueeze(1)
        fresh_noise: torch.Tensor = torch.randn_like(noisy_latents)
        return mean + is_nonzero_step * torch.sqrt(variance.clamp_min(1e-20)) * fresh_noise

    @torch.no_grad()
    def sample_conditional(
        self, cond_latents: torch.Tensor
    ) -> torch.Tensor:
        """Generate arrangement latents conditioned on floorplan latents.

        Runs the full reverse chain from pure noise, keeping the condition
        fixed throughout the chain.

        Args:
            cond_latents (torch.Tensor): Floorplan conditions on the model
                device, shape (num_samples, cond_dim).

        Returns:
            torch.Tensor: Generated arrangement latents, shape (num_samples, latent_dim).
        """
        device: torch.device = cond_latents.device
        num_samples: int = cond_latents.shape[0]
        latents: torch.Tensor = torch.randn(num_samples, self.latent_dim, device=device)
        # Walk the chain backwards so the last update lands exactly on x_0.
        for step in reversed(range(self.num_timesteps)):
            timesteps: torch.Tensor = torch.full(
                (num_samples,), step, device=device, dtype=torch.long
            )
            latents = self.reverse_step(latents, timesteps, cond_latents)
        return latents
