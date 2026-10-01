"""
Convolutional VAE over semantic segmentation masks, trained with BCE.

Structurally this is the same autoencoder as :mod:`VAE.model`: a convolutional
encoder produces the mean and log-variance of a flat latent Gaussian, and a
convolutional decoder maps a sampled latent back to a full-resolution tensor.
The encoder is imported from :mod:`VAE.model` and reused unchanged, because it
is activation-agnostic: it ends in two linear heads and never sees what the
decoded tensor is supposed to represent.

What differs from the image VAE is entirely on the output side:

* The decoder emits ``num_channels`` **logits**, one per class, with no output
  activation. The image VAE ends in a ``Sigmoid`` because its target is a
  ``0..1`` pixel; a mask target is a one-vs-rest binary map scored with
  ``binary_cross_entropy_with_logits``, which is numerically stable and already
  fuses the sigmoid, so adding one here would only destroy the gradients.
* The reconstruction term of the loss is BCE instead of MSE + L1, because the
  target is a set of independent binary indicators rather than a continuous
  image. The KL term is unchanged.

The latent stays a flat ``LATENT_DIM`` vector with no spatial axes, matching
:mod:`VAE.model` and keeping these latents interchangeable with the ones the
Diffusion and YOLO stages consume.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from VAE.model import Encoder

# Downsampling stages in the encoder, i.e. the power of two an input side length
# must be divisible by. The decoder mirrors it with transposed convolutions.
DOWNSAMPLE_STAGES: int = 4


class MaskDecoder(nn.Module):
    """
    Decoder that maps a flat latent vector to per-class mask logits.

    The mirror image of :class:`VAE.model.Encoder`'s four downsampling stages,
    with one difference that matters: the output has no activation function.
    The caller applies ``binary_cross_entropy_with_logits`` (or an ``argmax`` for
    display), so keeping the decoder linear is what makes the BCE gradient
    correct.

    Args:
        latent_dim (int): Dimensionality of the flat latent vector.
        base_channels (int): Channel count of the first decoder convolution,
            matching the encoder's deepest layer.
        num_channels (int): Number of class channels the decoder predicts, one
            per class of the configured mask layout.
        image_size (int): Side length of the decoded output. Must be
            ``base_size * 2 ** DOWNSAMPLE_STAGES`` for a consistent
            ``base_size`` feature map, i.e. divisible by 16.
        dropout (float): Dropout probability applied to the decoder features
            and to the latent. Only active in training mode.

    Raises:
        ValueError: If ``image_size`` is not divisible by ``2 ** 4``.
        ValueError: If ``num_channels`` or ``base_channels`` is not positive.
    """

    def __init__(
        self,
        latent_dim: int,
        base_channels: int,
        num_channels: int,
        image_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        stride: int = 2**DOWNSAMPLE_STAGES
        if image_size % stride != 0:
            raise ValueError(
                f"IMAGE_SIZE must be divisible by {stride} for the 4 downsampling "
                f"stages of the encoder, got {image_size}."
            )
        if num_channels < 1 or base_channels < 1:
            raise ValueError(
                f"num_channels and base_channels must be positive, got "
                f"{num_channels} and {base_channels}."
            )

        # Feature-map geometry of the encoder's deepest output, stored so the
        # flat latent can be reshaped without recomputing any division.
        self.spatial_size: int = image_size // stride
        self.base_channels: int = base_channels
        self.latent_dropout: nn.Dropout = nn.Dropout(dropout)
        self.fc: nn.Linear = nn.Linear(
            latent_dim, base_channels * self.spatial_size * self.spatial_size
        )

        # Four transposed convolutions undo the encoder's four stride-2 stages,
        # restoring the original spatial size. The channel counts mirror
        # VAE/model.py so both decoders have the same capacity profile.
        self.deconv_layers: nn.Sequential = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels, base_channels // 2, kernel_size=3, stride=2,
                padding=1, output_padding=1,
            ),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(
                base_channels // 2, base_channels // 4, kernel_size=3, stride=2,
                padding=1, output_padding=1,
            ),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(
                base_channels // 4, base_channels // 8, kernel_size=3, stride=2,
                padding=1, output_padding=1,
            ),
            nn.SiLU(),
            nn.ConvTranspose2d(
                base_channels // 8, base_channels // 8, kernel_size=5, stride=2,
                padding=2, output_padding=1,
            ),
            nn.SiLU(),
            # Final 5x5 convolution mixes the upsampled features into per-class
            # scores. No activation: these are logits.
            nn.Conv2d(base_channels // 8, num_channels, kernel_size=5, padding=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a batch of latent vectors into per-class mask logits.

        Args:
            z (torch.Tensor): ``(batch, latent_dim)`` latent vectors.

        Returns:
            torch.Tensor: ``(batch, num_channels, image_size, image_size)``
                unnormalized logits.

        Raises:
            ValueError: If ``z`` is not a ``(batch, latent_dim)`` tensor.
        """
        if z.ndim != 2:
            raise ValueError(f"The latent must be (batch, latent_dim), got {tuple(z.shape)}.")

        h: torch.Tensor = self.latent_dropout(z)
        h = self.fc(h)
        h = h.view(h.shape[0], self.base_channels, self.spatial_size, self.spatial_size)
        return self.deconv_layers(h)


class MaskVAEModel(nn.Module):
    """
    VAE that encodes and decodes one-vs-rest semantic mask tensors.

    The whole hyperparameters namespace is taken rather than individual scalars,
    matching :class:`VAE.model.VAEModel` and the rest of the project: adding a
    knob stays a TOML edit instead of a signature change.

    Args:
        archparams (SimpleNamespace): Architecture hyperparameters. Must expose
            ``MASK_CHANNELS``, ``LATENT_DIM``, ``BASE_CHANNELS``, ``IMAGE_SIZE``,
            ``DROPOUT`` and ``LATENT_ADDITIONAL_NOISE``.

    Attributes:
        num_channels (int): Class channels the decoder predicts.
        latent_dim (int): Dimensionality of the flat latent vector.
        latent_additional_noise (float): Std-dev of the extra latent noise
            added during training.
        encoder (Encoder): Reused VAE encoder producing ``(mu, logvar)``.
        decoder (MaskDecoder): Mask decoder producing logits.
    """

    def __init__(self, archparams: SimpleNamespace) -> None:
        super().__init__()
        self.num_channels: int = int(archparams.MASK_CHANNELS)
        self.latent_dim: int = int(archparams.LATENT_DIM)
        self.latent_additional_noise: float = float(archparams.LATENT_ADDITIONAL_NOISE)

        self.encoder: Encoder = Encoder(
            in_channels=self.num_channels,
            latent_dim=self.latent_dim,
            base_channels=int(archparams.BASE_CHANNELS),
            image_size=int(archparams.IMAGE_SIZE),
            dropout=float(archparams.DROPOUT),
        )
        self.decoder: MaskDecoder = MaskDecoder(
            latent_dim=self.latent_dim,
            base_channels=int(archparams.BASE_CHANNELS),
            num_channels=self.num_channels,
            image_size=int(archparams.IMAGE_SIZE),
            dropout=float(archparams.DROPOUT),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample a latent vector with the reparameterization trick.

        Sampling from ``N(mu, exp(logvar))`` is written as ``mu + eps * std`` so
        the result stays differentiable with respect to ``mu`` and ``logvar``.

        Args:
            mu (torch.Tensor): ``(batch, latent_dim)`` posterior means.
            logvar (torch.Tensor): ``(batch, latent_dim)`` posterior log-variances.

        Returns:
            torch.Tensor: ``(batch, latent_dim)`` sampled latent vectors.
        """
        std: torch.Tensor = torch.exp(0.5 * logvar)
        eps: torch.Tensor = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct a batch of mask tensors through the latent bottleneck.

        Args:
            x (torch.Tensor): ``(batch, num_channels, image_size, image_size)``
                binary one-vs-rest masks.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The reconstructed
                logits shaped like ``x``, plus the posterior ``mu`` and
                ``logvar``.

        Raises:
            ValueError: If the input's channel count is not what the decoder was
                built for.
        """
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"This model decodes {self.num_channels} mask channels but got "
                f"{x.shape[1]}. MASK_MODE in mask_VAE/config.toml and "
                "the checkpoint's channel count disagree (checkpoints from a "
                "different mode are unloadable)."
            )

        mu, logvar = self.encoder(x)
        z: torch.Tensor = self.reparameterize(mu, logvar)

        # Small Gaussian jitter on the latent during training keeps the decoder
        # from overfitting the exact posterior samples of a small dataset.
        if self.training:
            noise: torch.Tensor = torch.randn_like(z) * self.latent_additional_noise
            z = z + noise

        reconstruction: torch.Tensor = self.decoder(z)
        return reconstruction, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latents straight into mask logits, skipping the encoder.

        Used to sample the latent prior, which is what the generation grids show.

        Args:
            z (torch.Tensor): ``(batch, latent_dim)`` latent vectors.

        Returns:
            torch.Tensor: ``(batch, num_channels, image_size, image_size)``
                unnormalized logits.
        """
        return self.decoder(z)


def mask_vae_loss(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
    bce_pos_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Beta-VAE loss whose reconstruction term is a per-class binary cross-entropy.

    Args:
        x (torch.Tensor): ``(batch, num_channels, H, W)`` binary one-vs-rest
            target masks, valued ``0.0`` or ``1.0``.
        reconstruction (torch.Tensor): ``(batch, num_channels, H, W)`` decoder
            logits for the same samples.
        mu (torch.Tensor): ``(batch, latent_dim)`` posterior means.
        logvar (torch.Tensor): ``(batch, latent_dim)`` posterior log-variances.
        kl_weight (float): Beta weight of the KL term. ``0.0`` removes the
            latent regularization entirely.
        bce_pos_weight (float): Weight of positive targets in the BCE. ``1.0``
            is plain BCE. Mask pixels are dominated by background, so raising
            this makes a missed foreground class cost more than a false positive.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The total
            loss, the BCE term, the KL term, and the KL divided by the latent
            dimension. All four are per-image sums, matching the reduction
            convention of :func:`VAE.model.vae_loss`, so the magnitudes are
            comparable with the image VAE's telemetry.

    Raises:
        ValueError: If the shapes of ``x`` and ``reconstruction`` disagree.
    """
    if x.shape != reconstruction.shape:
        raise ValueError(
            f"Target shape {tuple(x.shape)} and reconstruction shape "
            f"{tuple(reconstruction.shape)} must match."
        )

    # Per-image sums rather than a flat mean: the reduction convention of
    # VAE/model.py, so a full-size loss and a long loss stay comparable.
    bce_loss: torch.Tensor = nn.functional.binary_cross_entropy_with_logits(
        reconstruction,
        x,
        pos_weight=torch.full_like(x, bce_pos_weight),
        reduction="sum",
    ) / x.shape[0]
    kl_loss: torch.Tensor = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss: torch.Tensor = bce_loss + kl_weight * kl_loss
    # Per-dimension KL is the standard posterior-collapse diagnostic: when it
    # collapses towards zero the encoder has stopped using the latent.
    kl_per_dim: torch.Tensor = kl_loss / mu.shape[1]
    return total_loss, bce_loss, kl_loss, kl_per_dim
