"""
Convolutional Variational Autoencoder (VAE) model.

A standard VAE that encodes an input image into a low-dimensional latent
distribution (parameterized by mean and log-variance) and decodes a sampled
latent vector back into a reconstruction of the input image.

The model expects square input images whose side length is divisible by 16,
because the convolutional encoder applies 4 downsampling stages (stride 2).
"""

from types import SimpleNamespace

import torch
from torch import nn


class Encoder(nn.Module):
    """Convolutional encoder that maps an image to latent distribution parameters.

    Downsamples the input 4 times (stride-2 convolutions) and then maps the
    flattened feature map to the mean and log-variance of the latent Gaussian.
    Applies dropout after every convolution block and before the linear heads
    to regularize the encoding (active only while training).
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        base_channels: int,
        image_size: int,
        dropout: float,
    ) -> None:
        """Initialize the encoder layers.

        Args:
            in_channels (int): Number of channels in the input image.
            latent_dim (int): Dimensionality of the latent space.
            base_channels (int): Number of channels in the deepest convolutional layer.
            image_size (int): Height/width of the (square) input image.
            dropout (float): Dropout probability applied after each activation.
        """
        super().__init__()

        # Each block (kernel=4, stride=2, padding=1) halves the spatial size.
        # Dropout2d drops whole feature maps, which regularizes more strongly
        # than pixel-wise dropout for convolutional features.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, base_channels // 8,
                      kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(base_channels // 8, base_channels //
                      4, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(base_channels // 4, base_channels//2,
                      kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(base_channels // 2, base_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Dropout2d(dropout),
        )

        # After 4 downsampling stages the spatial size is image_size / 16.
        spatial_size: int = image_size // 16
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
        dropout: float,
    ) -> None:
        """Initialize the decoder layers.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            base_channels (int): Number of channels used by the deepest decoder layer.
            out_channels (int): Number of channels of the reconstructed image (e.g. 3 for RGB).
            image_size (int): Height/width of the (square) output image.
            dropout (float): Dropout probability applied after each activation.
        """
        super().__init__()

        # Store the feature-map geometry so the flat fc output can be reshaped
        # back into a spatial tensor without any float arithmetic.
        self.spatial_size: int = image_size // 16
        self.base_channels: int = base_channels
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(latent_dim, base_channels *
                            self.spatial_size * self.spatial_size)

        # Reverse of the encoder: 4 upsampling stages (stride 2) restore the
        # original spatial size. Sigmoid keeps outputs in the [0, 1] range.
        # Dropout2d again regularizes the feature maps while training.
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(base_channels // 4, base_channels // 8,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.SiLU(),

            nn.ConvTranspose2d(base_channels // 8, out_channels,
                               kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(),

            nn.Conv2d(out_channels, out_channels,
                      kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
        )

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
        h = h.view(h.shape[0], self.base_channels,
                   self.spatial_size, self.spatial_size)
        return self.deconv_layers(h)


class VAEModel(nn.Module):
    """Variational Autoencoder combining the encoder and decoder."""

    def __init__(self, archparams: SimpleNamespace) -> None:
        """Initialize the VAE.

        Args:
            archparams (SimpleNamespace): Object exposing the architectural parameters
                IMAGE_SIZE, OUT_CHANNELS, LATENT_DIM, BASE_CHANNELS and DROPOUT.
        """
        super().__init__()
        self.latent_dim: int = archparams.LATENT_DIM
        self.latent_additional_noise: float = archparams.LATENT_ADDITIONAL_NOISE

        self.encoder = Encoder(
            in_channels=archparams.OUT_CHANNELS,
            latent_dim=archparams.LATENT_DIM,
            base_channels=archparams.BASE_CHANNELS,
            image_size=archparams.IMAGE_SIZE,
            dropout=archparams.DROPOUT,
        )
        self.decoder = Decoder(
            latent_dim=archparams.LATENT_DIM,
            base_channels=archparams.BASE_CHANNELS,
            out_channels=archparams.OUT_CHANNELS,
            image_size=archparams.IMAGE_SIZE,
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

        # Add small Gaussian noise to the latent vector during training to regularize the decoder.
        if self.training:
            noise: torch.Tensor = torch.randn_like(
                z) * self.latent_additional_noise  # Small noise for regularization
            z = z + noise
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
    mse_loss: torch.Tensor = nn.functional.mse_loss(
        reconstruction, x, reduction="sum") / x.shape[0]

    # Mean absolute error between input and reconstruction (sharpens edges).
    l1_term: torch.Tensor = nn.functional.l1_loss(
        reconstruction, x, reduction="sum") / x.shape[0]

    # Combined reconstruction objective: MSE regularized by a weighted L1 term.
    recon_loss: torch.Tensor = mse_loss + l1_weight * l1_term

    # Closed-form KL divergence between N(mu, sigma^2) and N(0, 1).
    kl_loss: torch.Tensor = torch.mean(-0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    total_loss: torch.Tensor = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss
