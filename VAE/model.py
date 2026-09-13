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
    """

    def __init__(self, in_channels: int, latent_dim: int, base_channels: int, image_size: int) -> None:
        """Initialize the encoder layers.

        Args:
            in_channels (int): Number of channels in the input image.
            latent_dim (int): Dimensionality of the latent space.
            base_channels (int): Number of channels in the deepest convolutional layer.
            image_size (int): Height/width of the (square) input image.
        """
        super().__init__()

        # Each block (kernel=4, stride=2, padding=1) halves the spatial size.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, base_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels // 8, base_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels // 4, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels // 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # After 4 downsampling stages the spatial size is image_size / 16.
        spatial_size: int = image_size // 16
        self.flattened_size: int = base_channels * spatial_size * spatial_size

        # Linear heads producing the parameters of the latent Gaussian.
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
        mu: torch.Tensor = self.fc_mu(h)
        logvar: torch.Tensor = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Transposed-convolutional decoder that maps latent vectors back to images."""

    def __init__(self, latent_dim: int, base_channels: int, out_channels: int, image_size: int) -> None:
        """Initialize the decoder layers.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            base_channels (int): Number of channels used by the deepest decoder layer.
            out_channels (int): Number of channels of the reconstructed image (e.g. 3 for RGB).
            image_size (int): Height/width of the (square) output image.
        """
        super().__init__()

        # Store the feature-map geometry so the flat fc output can be reshaped
        # back into a spatial tensor without any float arithmetic.
        self.spatial_size: int = image_size // 16
        self.base_channels: int = base_channels
        self.fc = nn.Linear(latent_dim, base_channels * self.spatial_size * self.spatial_size)

        # Reverse of the encoder: 4 upsampling stages (stride 2) restore the
        # original spatial size. Sigmoid keeps outputs in the [0, 1] range.
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels // 4, base_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels // 8, out_channels, kernel_size=4, stride=2, padding=1),
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
        h: torch.Tensor = self.fc(z)
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
                IMAGE_SIZE, OUT_CHANNELS, LATENT_DIM and BASE_CHANNELS.
        """
        super().__init__()
        self.latent_dim: int = archparams.LATENT_DIM

        self.encoder = Encoder(
            in_channels=archparams.OUT_CHANNELS,
            latent_dim=archparams.LATENT_DIM,
            base_channels=archparams.BASE_CHANNELS,
            image_size=archparams.IMAGE_SIZE,
        )
        self.decoder = Decoder(
            latent_dim=archparams.LATENT_DIM,
            base_channels=archparams.BASE_CHANNELS,
            out_channels=archparams.OUT_CHANNELS,
            image_size=archparams.IMAGE_SIZE,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the VAE loss: reconstruction (MSE) + weighted KL divergence.

    Args:
        x (torch.Tensor): Original input images, shape (batch_size, out_channels, image_size, image_size).
        reconstruction (torch.Tensor): Reconstructed images, same shape as x.
        mu (torch.Tensor): Encoder mean, shape (batch_size, latent_dim).
        logvar (torch.Tensor): Encoder log-variance, shape (batch_size, latent_dim).
        kl_weight (float): Weight applied to the KL divergence term (beta-VAE style).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            (total_loss, recon_loss, kl_loss) where recon_loss and kl_loss are
            reduced over the batch.
    """
    # Mean squared error between input and reconstruction.
    recon_loss: torch.Tensor = nn.functional.mse_loss(reconstruction, x, reduction="sum") / x.shape[0]

    # Closed-form KL divergence between N(mu, sigma^2) and N(0, 1).
    kl_loss: torch.Tensor = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    total_loss: torch.Tensor = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss