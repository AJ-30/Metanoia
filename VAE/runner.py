"""
Training runner for the VAE model.

Loads configuration and hyperparameters from TOML files, builds the image
dataset and dataloader, runs the training loop on the configured device
(MPS / CUDA / CPU) and logs progress using the shared project logger and
telemetry writer from the root `helpers/` package.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torchvision import transforms

# Project root directory so that the shared `helpers/` package (logger, config,
# telemetry writer) can be imported no matter which directory the runner is
# launched from.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from helpers.config import load_config
from helpers.hyperparams import load_hyperparams
from helpers.logger import setup_logger
from helpers.telemetry_writer import TelemetryWriter

from VAE.model import VAEModel, vae_loss

# Directory of this file, used to resolve relative TOML / output paths.
SCRIPT_DIR: Path = Path(__file__).resolve().parent


class FloorPlanDataset(Dataset):
    """PyTorch Dataset that lazy-loads and preprocesses floor-plan images.

    Each RGBA image is composited onto a solid background (to drop the alpha
    channel), resized to a square and converted to a float tensor in [0, 1].
    """

    def __init__(
        self,
        image_dir: str,
        image_size: int,
        background_color: list[int],
        max_samples: int,
    ) -> None:
        """Initialize the dataset.

        Args:
            image_dir (str): Directory containing the PNG/JPEG images.
            image_size (int): Target side length of the resized square images.
            background_color (list[int]): RGBA tuple used as the backdrop when
                flattening the alpha channel.
            max_samples (int): Cap on the number of images loaded. Negative
                values mean "use every image in the directory".
        """
        self.image_paths: list[str] = sorted(
            str(p) for p in Path(image_dir).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        # Optionally cap the dataset size, so quick debug runs stay cheap.
        if max_samples >= 0 and len(self.image_paths) > max_samples:
            self.image_paths = self.image_paths[:max_samples]

        # Compose transforms: resize, flatten alpha onto a white backdrop, tensor.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(self._flatten_alpha),
            transforms.ToTensor(),
        ])
        self.background: tuple[int, int, int, int] = tuple(background_color)

    def _flatten_alpha(self, img: Image.Image) -> Image.Image:
        """Composite the RGBA image onto the configured background color.

        Args:
            img (Image.Image): RGBA image to flatten.

        Returns:
            Image.Image: RGB image with the alpha channel baked onto the background.
        """
        if img.mode != "RGBA":
            return img.convert("RGB")
        backdrop: Image.Image = Image.new("RGBA", img.size, self.background)
        return Image.alpha_composite(backdrop, img).convert("RGB")

    def __len__(self) -> int:
        """Return the number of images in the dataset.

        Returns:
            int: Dataset size.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Load and preprocess the image at the given index.

        Args:
            index (int): Index of the image to load.

        Returns:
            torch.Tensor: Tensor of shape (out_channels, image_size, image_size) in [0, 1].
        """
        img: Image.Image = Image.open(self.image_paths[index])
        return self.transform(img)


def resolve_device(requested_device: str, log: SimpleNamespace) -> torch.device:
    """Resolve the requested device string to a usable PyTorch device.

    Falls back to CPU (with a warning) when the requested accelerator is not
    available on the current machine.

    Args:
        requested_device (str): One of "mps", "cuda" or "cpu".
        log (SimpleNamespace): Configured logger for the fallback warning.

    Returns:
        torch.device: The device to use for training.
    """
    if requested_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    log.warning(
        f"Requested device '{requested_device}' is not available on this machine. Falling back to CPU."
    )
    return torch.device("cpu")


def save_reconstruction_grid(
    model: VAEModel,
    sample_batch: torch.Tensor,
    device: torch.device,
    save_path: Path,
    log: SimpleNamespace,
) -> None:
    """Reconstruct a sample batch and save an original-vs-reconstructed grid image.

    Args:
        model (VAEModel): The VAE model used for reconstruction.
        sample_batch (torch.Tensor): Ground-truth image batch, shape (n, c, h, w) in [0, 1].
        device (torch.device): Device the model lives on.
        save_path (Path): Destination path for the saved grid PNG.
        log (SimpleNamespace): Configured logger.
    """
    model.eval()
    with torch.no_grad():
        batch: torch.Tensor = sample_batch.to(device)
        reconstruction, _, _ = model(batch)
        reconstruction = reconstruction.cpu()

    # Stack originals above their reconstructions so they are easy to compare.
    comparison: torch.Tensor = torch.cat([sample_batch, reconstruction], dim=0)
    grid: torch.Tensor = make_grid(comparison, nrow=sample_batch.shape[0], padding=2, normalize=True)
    # make_grid returns a (3, h, w) tensor; de-normalize to 0-255 for saving.
    grid_img = transforms.ToPILImage()(grid)
    grid_img.save(str(save_path))
    model.train()
    log.info(f"Saved reconstruction grid to {save_path}")


class VAERunner:
    """Runner that wires together config, hyperparameters, model and training loop."""

    def __init__(self) -> None:
        """Initialize the runner: load config/hyperparams, set up the device and model."""
        self.config_file: str = str(SCRIPT_DIR / "config.toml")
        self.hyperparams_file: str = str(SCRIPT_DIR / "hyperparameters.toml")

        parser = argparse.ArgumentParser(description="Train the VAE on floor-plan images.")
        parser.add_argument(
            "--mode", help="Run mode (debug/prod)", choices=["debug", "prod"], default="debug"
        )
        args: argparse.Namespace = parser.parse_args()

        self.config: SimpleNamespace = load_config(args.mode, self.config_file)
        self.hyperparams: SimpleNamespace = load_hyperparams(self.hyperparams_file)
        self.log = setup_logger(self.config)

        # Telemetry fieldnames for the shared CSV telemetry writer.
        self.telemetry_fieldnames: list[str] = [
            "epoch", "train_loss", "recon_loss", "kl_loss",
        ]

        self.device: torch.device = resolve_device(self.config.DEVICE, self.log)
        self.log.info(
            f"Starting VAE training with run_id: <red>{self.config.runtime.RUN_ID}</red> "
            f"at time: {datetime.fromtimestamp(self.config.runtime.START_TIME)} on device: {self.device}"
        )

        # Training state.
        self.model: VAEModel = VAEModel(self.hyperparams).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams.LEARNING_RATE)
        self.telemetry_writer = TelemetryWriter(
            fieldnames=self.telemetry_fieldnames,
            directory=self.config.TELEMETRY_LOG_DIRECTORY,
            config=self.config,
        )

    def build_dataloader(self) -> DataLoader[FloorPlanDataset]:
        """Build the dataset and dataloader from config/hyperparameters.

        Returns:
            DataLoader: Configured dataloader for the floor-plan images.
        """
        dataset = FloorPlanDataset(
            image_dir=self.config.IMAGE_DIR,
            image_size=self.hyperparams.IMAGE_SIZE,
            background_color=self.hyperparams.BACKGROUND_COLOR,
            max_samples=self.hyperparams.MAX_SAMPLES,
        )
        self.log.info(f"Dataset size: {len(dataset)} images from {self.config.IMAGE_DIR}")
        return DataLoader(
            dataset,
            batch_size=self.hyperparams.BATCH_SIZE,
            shuffle=True,
            num_workers=self.hyperparams.NUM_WORKERS,
            # Only CUDA supports pinned host memory; MPS/silicon ignores it.
            pin_memory=self.device.type == "cuda",
        )

    def training_step(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one optimization step on a single batch.

        Args:
            batch (torch.Tensor): Input image batch, shape (n, c, h, w).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                (total_loss, recon_loss, kl_loss) averaged over the batch.
        """
        batch = batch.to(self.device)
        reconstruction, mu, logvar = self.model(batch)
        total_loss, recon_loss, kl_loss = vae_loss(
            batch, reconstruction, mu, logvar, kl_weight=self.hyperparams.KL_WEIGHT
        )
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), recon_loss.detach(), kl_loss.detach()

    def run_epoch(self, dataloader: DataLoader[FloorPlanDataset]) -> tuple[float, float, float]:
        """Iterate over the whole dataloader once and report averaged losses.

        Args:
            dataloader (DataLoader): Dataloader to iterate over.

        Returns:
            tuple[float, float, float]: Averaged (train_loss, recon_loss, kl_loss) for the epoch.
        """
        self.model.train()
        total_loss_sum: float = 0.0
        recon_loss_sum: float = 0.0
        kl_loss_sum: float = 0.0
        num_batches: int = 0

        for batch in dataloader:
            total_loss, recon_loss, kl_loss = self.training_step(batch)
            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1

        return (
            total_loss_sum / num_batches,
            recon_loss_sum / num_batches,
            kl_loss_sum / num_batches,
        )

    def save_checkpoint(self, epoch: int) -> None:
        """Save the current model weights to the configured model directory.

        Args:
            epoch (int): Current epoch, used in the checkpoint file name.
        """
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        checkpoint_path: str = os.path.join(
            self.config.MODEL_SAVE_PATH,
            f"vae_{self.config.runtime.RUN_ID}_epoch_{epoch}.pt",
        )
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "hyperparameters": vars(self.hyperparams),
            },
            checkpoint_path,
        )
        self.log.info(f"Saved checkpoint to {checkpoint_path}")

    def train(self) -> None:
        """Run the full training loop and log telemetry at intervals."""
        torch.autograd.set_detect_anomaly(self.config.IS_DEBUG_TORCH)

        dataloader: DataLoader[FloorPlanDataset] = self.build_dataloader()
        os.makedirs(self.config.OUTPUT_SAMPLES_DIRECTORY, exist_ok=True)

        # Hold onto one fixed batch for periodic reconstruction visualization.
        fixed_batch: torch.Tensor = next(iter(dataloader))[: self.hyperparams.NUM_RECON_SAMPLES]

        self.log.info(f"Torch version: {torch.__version__}, total images: {len(dataloader.dataset)}")
        execution_start: float = time.time()

        for epoch in range(1, self.hyperparams.NUM_EPOCHS + 1):
            epoch_start: float = time.time()
            train_loss, recon_loss, kl_loss = self.run_epoch(dataloader)

            # Periodic telemetry + checkpointing as configured.
            if epoch % self.config.TELEMETRY_INTERVAL == 0 or epoch == self.hyperparams.NUM_EPOCHS:
                self.telemetry_writer.log(
                    epoch=epoch, train_loss=train_loss, recon_loss=recon_loss, kl_loss=kl_loss
                )
                self.save_checkpoint(epoch)

                # Save a reconstruction grid every SAVE_AFTER_INTERVAL epochs.
                if epoch % self.config.SAVE_AFTER_INTERVAL == 0:
                    sample_path: Path = (
                        Path(self.config.OUTPUT_SAMPLES_DIRECTORY)
                        / f"reconstruction_epoch_{epoch}_{self.config.runtime.RUN_ID}.png"
                    )
                    save_reconstruction_grid(self.model, fixed_batch, self.device, sample_path, self.log)

            epoch_time: float = time.time() - epoch_start
            total_elapsed: float = time.time() - execution_start
            self.log.info(
                f"Epoch {epoch}/{self.hyperparams.NUM_EPOCHS} | "
                f"loss: {train_loss:.4f} | recon: {recon_loss:.4f} | kl: {kl_loss:.4f} | "
                f"epoch_time: {epoch_time:.1f}s | total: {total_elapsed:.1f}s"
            )

        self.log.success("VAE training finished.")