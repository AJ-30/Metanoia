"""
Convolutional Variational Autoencoder (VAE) model.

A standard VAE that encodes an input image into a low-dimensional latent
distribution (parameterized by mean and log-variance) and decodes a sampled
latent vector back into a reconstruction of the input image.

The model expects square input images whose side length is divisible by
2 ** NUM_STAGES, because the convolutional encoder applies NUM_STAGES
downsampling stages (stride 2).

The network is dense in the sense that it is both deep and wide: it stacks
``NUM_STAGES`` down/up-sampling blocks and grows the channel count geometrically
from ``BASE_CHANNELS / 2 ** (NUM_STAGES - 1)`` up to ``BASE_CHANNELS``.
"""

from types import SimpleNamespace

import torch
from torch import nn


def _build_stage_channels(base_channels: int, num_stages: int) -> list[int]:
    """Build the per-stage channel counts for a dense downsampling path.

    The first stage uses ``base_channels / 2 ** (num_stages - 1)`` channels and
    each subsequent stage doubles them, ending exactly at ``base_channels`` for
    the deepest stage.

    Args:
        base_channels (int): Number of channels in the deepest stage.
        num_stages (int): Number of down/up-sampling stages.

    Returns:
        list[int]: Channel count of each stage, length ``num_stages``, strictly
        increasing and ending at ``base_channels``.
    """
    # The shallowest stage must be a whole number of channels (no fractional
    # widths), so the geometry has to divide evenly.
    if base_channels % (2 ** (num_stages - 1)) != 0:
        raise ValueError(
            f"base_channels ({base_channels}) must be divisible by 2**(num_stages - 1) "
            f"(= {2 ** (num_stages - 1)} for num_stages={num_stages})"
        )
    channels: list[int] = [base_channels // (2 ** (num_stages - 1))]
    for _ in range(1, num_stages):
        channels.append(channels[-1] * 2)
    return channels


class Encoder(nn.Module):
    """Convolutional encoder that maps an image to latent distribution parameters.

    Downsamples the input ``num_stages`` times (stride-2 convolutions) and then
    maps the flattened feature map to the mean and log-variance of the latent
    Gaussian. Applies dropout after every convolution block and before the
    linear heads to regularize the encoding (active only while training).
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        base_channels: int,
        image_size: int,
        num_stages: int,
        dropout: float,
    ) -> None:
        """Initialize the encoder layers.

        Args:
            in_channels (int): Number of channels in the input image.
            latent_dim (int): Dimensionality of the latent space.
            base_channels (int): Number of channels in the deepest convolutional layer.
            image_size (int): Height/width of the (square) input image.
            num_stages (int): Number of stride-2 downsampling stages (network depth).
            dropout (float): Dropout probability applied after each activation.
        """
        super().__init__()

        # Each block (kernel=4, stride=2, padding=1) halves the spatial size.
        # The channel count grows geometrically from base / 2**(num_stages - 1)
        # up to base_channels, so the encoder is both deeper and wider as
        # num_stages / base_channels grow. Dropout2d drops whole feature maps,
        # which regularizes more strongly than pixel-wise dropout.
        stage_channels: list[int] = _build_stage_channels(base_channels, num_stages)
        conv_blocks: list[nn.Module] = []
        for stage_index, output_channels in enumerate(stage_channels):
            input_channels: int = in_channels if stage_index == 0 else stage_channels[stage_index - 1]
            conv_blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Dropout2d(dropout),
                ]
            )
        self.conv_layers = nn.Sequential(*conv_blocks)

        # After num_stages downsampling stages the spatial size is
        # image_size / (2 ** num_stages). Reject images that do not divide
        # evenly so no silent spatial-size truncation can occur.
        downsampling_factor: int = 2 ** num_stages
        if image_size % downsampling_factor != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by 2**num_stages "
                f"({downsampling_factor} for num_stages={num_stages})"
            )
        spatial_size: int = image_size // downsampling_factor
        self.flattened_size: int = base_channels * spatial_size * spatial_size

        # Linear heads producing the parameters of the latent Gaussian. The
        # plain Dropout operates on the flattened vector before the heads.
        self.fc_dropout = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an image batch into the latent Gaussian parameters.

        Args:
            x (torch.Tensor): Image batch of shape (batch_size, in_channels, image_size, image_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (mu, logvar) each of shape (batch_size, latent_dim).
        """
        h: torch.Tensor = self.conv_layers(x)
        h = h.flatten(start_dim=1)
        h = self.fc_dropout(h)
        mu: torch.Tensor = self.fc_mu(h)
        logvar: torch.Tensor = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Transposed-convolutional decoder that maps latent vectors back to images."""

    def __init__(
        self,
        latent_dim: int,
        base_channels: int,
        out_channels: int,
        image_size: int,
        num_stages: int,
        dropout: float,
    ) -> None:
        """Initialize the decoder layers.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            base_channels (int): Number of channels used by the deepest decoder layer.
            out_channels (int): Number of channels of the reconstructed image (e.g. 3 for RGB).
            image_size (int): Height/width of the (square) output image.
            num_stages (int): Number of stride-2 upsampling stages (network depth).
            dropout (float): Dropout probability applied after each activation.
        """
        super().__init__()

        # Store the feature-map geometry so the flat fc output can be reshaped
        # back into a spatial tensor without any float arithmetic.
        downsampling_factor: int = 2 ** num_stages
        if image_size % downsampling_factor != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by 2**num_stages "
                f"({downsampling_factor} for num_stages={num_stages})"
            )
        self.spatial_size: int = image_size // downsampling_factor
        self.base_channels: int = base_channels
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(latent_dim, base_channels * self.spatial_size * self.spatial_size)

        # Reverse of the encoder: num_stages upsampling stages (stride 2) restore
        # the original spatial size. Channels shrink back geometrically from
        # base_channels to base / 2**(num_stages - 1); the last stage emits the
        # image channels. Sigmoid keeps outputs in the [0, 1] range. Dropout2d
        # again regularizes the feature maps while training.
        up_sampling_channels: list[int] = list(
            reversed(_build_stage_channels(base_channels, num_stages))
        )
        deconv_blocks: list[nn.Module] = []
        for stage_index, input_channels in enumerate(up_sampling_channels):
            output_channels: int = (
                out_channels if stage_index == len(up_sampling_channels) - 1 else up_sampling_channels[stage_index + 1]
            )
            deconv_blocks.append(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1)
            )
            if stage_index == len(up_sampling_channels) - 1:
                deconv_blocks.append(nn.Sigmoid())
            else:
                deconv_blocks.append(nn.ReLU())
                deconv_blocks.append(nn.Dropout2d(dropout))
        self.deconv_layers = nn.Sequential(*deconv_blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent batch into reconstructed images.

        Args:
            z (torch.Tensor): Latent vectors of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Reconstructed images of shape
                (batch_size, out_channels, image_size, image_size) with values in [0, 1].
        """
        h: torch.Tensor = self.fc_dropout(z)
        h = self.fc(h)
        # Reshape the flat feature vector back into (channels, spatial, spatial)
        # before feeding it to the transposed convolutions.
        h = h.view(h.shape[0], self.base_channels, self.spatial_size, self.spatial_size)
        return self.deconv_layers(h)


class VAEModel(nn.Module):
    """Variational Autoencoder combining the encoder and decoder."""

    def __init__(self, archparams: SimpleNamespace) -> None:
        """Initialize the VAE.

        Args:
            archparams (SimpleNamespace): Object exposing the architectural parameters
                IMAGE_SIZE, OUT_CHANNELS, LATENT_DIM, BASE_CHANNELS, NUM_STAGES and DROPOUT.
        """
        super().__init__()
        self.latent_dim: int = archparams.LATENT_DIM

        self.encoder = Encoder(
            in_channels=archparams.OUT_CHANNELS,
            latent_dim=archparams.LATENT_DIM,
            base_channels=archparams.BASE_CHANNELS,
            image_size=archparams.IMAGE_SIZE,
            num_stages=archparams.NUM_STAGES,
            dropout=archparams.DROPOUT,
        )
        self.decoder = Decoder(
            latent_dim=archparams.LATENT_DIM,
            base_channels=archparams.BASE_CHANNELS,
            out_channels=archparams.OUT_CHANNELS,
            image_size=archparams.IMAGE_SIZE,
            num_stages=archparams.NUM_STAGES,
            dropout=archparams.DROPOUT,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Apply the reparameterization trick: z = mu + sigma * eps.

        Args:
            mu (torch.Tensor): Mean of the latent distribution, shape (batch_size, latent_dim).
            logvar (torch.Tensor): Log-variance of the latent distribution, shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Sampled latent vectors, shape (batch_size, latent_dim).
        """
        # Standard deviation derived from log-variance. See VAE (Kingma & Welling, 2013).
        std: torch.Tensor = torch.exp(0.5 * logvar)
        eps: torch.Tensor = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass: encode, sample and reconstruct.

        Args:
            x (torch.Tensor): Input image batch, shape (batch_size, out_channels, image_size, image_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                (reconstruction, mu, logvar) where reconstruction has the same
                shape as x and mu/logvar have shape (batch_size, latent_dim).
        """
        mu, logvar = self.encoder(x)
        z: torch.Tensor = self.reparameterize(mu, logvar)
        reconstruction: torch.Tensor = self.decoder(z)
        return reconstruction, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode raw latent vectors into images (used for sampling/generation).

        Args:
            z (torch.Tensor): Latent vectors, shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Generated images, shape (batch_size, out_channels, image_size, image_size).
        """
        return self.decoder(z)


def vae_loss(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
    l1_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the VAE loss: MSE + L1 reconstruction and weighted KL divergence.

    The reconstruction term combines mean squared error and mean absolute error,
    both reduced per image (per-sample sum over all pixels and channels). The
    reported ``recon_loss`` is the composite MSE + l1_weight * L1, so telemetry
    reflects the actual reconstruction objective being optimized. The KL term is
    the closed-form KL between the encoder's Gaussian and the N(0, 1) prior.

    Args:
        x (torch.Tensor): Original input images, shape (batch_size, out_channels, image_size, image_size).
        reconstruction (torch.Tensor): Reconstructed images, same shape as x.
        mu (torch.Tensor): Encoder mean, shape (batch_size, latent_dim).
        logvar (torch.Tensor): Encoder log-variance, shape (batch_size, latent_dim).
        kl_weight (float): Weight applied to the KL divergence term (beta-VAE style).
        l1_weight (float): Weight applied to the L1 term alongside the MSE term.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            (total_loss, recon_loss, kl_loss) where total_loss is
            recon_loss + kl_weight * kl_loss and recon_loss is the composite
            MSE + l1_weight * L1 reconstruction loss, reduced over the batch.
    """
    # Mean squared error between input and reconstruction.
    mse_loss: torch.Tensor = nn.functional.mse_loss(reconstruction, x, reduction="sum") / x.shape[0]

    # Mean absolute error between input and reconstruction (sharpens edges).
    l1_term: torch.Tensor = nn.functional.l1_loss(reconstruction, x, reduction="sum") / x.shape[0]

    # Combined reconstruction objective: MSE regularized by a weighted L1 term.
    recon_loss: torch.Tensor = mse_loss + l1_weight * l1_term

    # Closed-form KL divergence between N(mu, sigma^2) and N(0, 1).
    kl_loss: torch.Tensor = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    total_loss: torch.Tensor = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss