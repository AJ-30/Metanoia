"""
Training runner for the latent diffusion model.

Pipeline: images are encoded once with a *frozen* pretrained VAE encoder into
flat latent vectors (15k x 256 floats is only ~15 MB, so the whole split is
cached in RAM as tensors). The DDPM denoiser then trains purely on those
cached latents, which keeps every epoch cheap (MLP only, no convolutions).
Sampling runs the reverse chain from noise and decodes the result with the
frozen VAE decoder, so generated floor plans can be inspected visually.

Telemetry goes to its own directory (``Diffusion/telemetry_logs/``, one CSV
per run) and per-epoch generation grids go to ``Diffusion/outputs/``, so the
diffusion dashboard server watches files that are fully isolated from the
VAE run. No reconstruction grids are produced; the dashboard shows
diffusion generations only.

Dataset and device helpers are reused from ``VAE.runner`` (single source of
truth for image preprocessing) instead of being duplicated here.
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.utils import make_grid

# Project root, so the shared `helpers/` package and the `VAE` package are
# importable no matter which directory the runner is launched from.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from helpers.config import load_config
from helpers.hyperparams import load_hyperparams
from helpers.logger import setup_logger
from helpers.telemetry_writer import TelemetryWriter

from Diffusion.model import LatentDiffusion
from VAE.model import VAEModel
from VAE.runner import FloorPlanDataset, resolve_device

# Directory of this file, used to resolve relative TOML / output paths.
SCRIPT_DIR: Path = Path(__file__).resolve().parent


def set_reproducibility_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs for reproducible runs.

    Args:
        seed (int): Seed applied to ``random`` and ``torch`` (CPU + CUDA).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_vae_num_stages(state_dict: dict[str, torch.Tensor]) -> int:
    """Infer the VAE depth for old checkpoints that predate NUM_STAGES.

    Each encoder stage owns exactly one convolution whose weight key looks
    like ``encoder.conv_layers.<N>.weight`` with N a multiple of 3
    (conv + ReLU + Dropout2d per stage), so counting those keys recovers the
    number of downsampling stages.

    Args:
        state_dict (dict[str, torch.Tensor]): VAE checkpoint state dict.

    Returns:
        int: Number of encoder downsampling stages.
    """
    conv_weight_keys: list[str] = [
        key
        for key in state_dict
        if key.startswith("encoder.conv_layers.") and key.endswith(".weight")
    ]
    if len(conv_weight_keys) == 0:
        raise ValueError("VAE checkpoint has no encoder conv weights; not a VAE checkpoint.")
    return len(conv_weight_keys)


def build_vae_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> tuple[VAEModel, SimpleNamespace]:
    """Rebuild the exact VAE architecture stored in a checkpoint and load it frozen.

    The architecture is reconstructed from the hyperparameters embedded in the
    checkpoint itself (not from the current ``VAE/hyperparameters.toml``), so
    fully trained older runs stay usable even after the VAE config moved on.
    Checkpoints saved before ``NUM_STAGES`` existed fall back to inferring the
    depth from the encoder weights.

    Args:
        checkpoint_path (str): Path to the ``.pt`` VAE checkpoint.
        device (torch.device): Device to place the frozen VAE on.

    Returns:
        tuple[VAEModel, SimpleNamespace]: (frozen VAE in eval mode, VAE archparams).
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")
    checkpoint: dict[str, object] = torch.load(checkpoint_path, map_location=device)
    saved_hyperparams: dict[str, object] = checkpoint["hyperparameters"]
    state_dict: dict[str, torch.Tensor] = checkpoint["model_state_dict"]

    # Older checkpoints lack NUM_STAGES; recover the depth from the weights.
    num_stages: int = saved_hyperparams.get("NUM_STAGES", infer_vae_num_stages(state_dict))
    archparams = SimpleNamespace(
        IMAGE_SIZE=saved_hyperparams["IMAGE_SIZE"],
        OUT_CHANNELS=saved_hyperparams["OUT_CHANNELS"],
        LATENT_DIM=saved_hyperparams["LATENT_DIM"],
        BASE_CHANNELS=saved_hyperparams["BASE_CHANNELS"],
        NUM_STAGES=num_stages,
        DROPOUT=saved_hyperparams["DROPOUT"],
    )
    vae = VAEModel(archparams)
    vae.load_state_dict(state_dict)
    # Freeze every VAE parameter: the encoder/decoder never train here.
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae.to(device), archparams


def save_image_grid(images: torch.Tensor, nrow: int, save_path: Path) -> None:
    """Save a batch of [0, 1] images as a single grid PNG.

    Args:
        images (torch.Tensor): Image batch, shape (n, c, h, w) in [0, 1].
        nrow (int): Images per grid row.
        save_path (Path): Destination path for the saved PNG.
    """
    grid: torch.Tensor = make_grid(images.cpu(), nrow=nrow, padding=2, normalize=True)
    grid_image = transforms.ToPILImage()(grid)
    grid_image.save(str(save_path))


class DiffusionRunner:
    """Runner that wires together config, frozen VAE, diffusion model and training loop."""

    def __init__(self) -> None:
        """Initialize the runner: config, device, frozen VAE, diffusion model, telemetry."""
        self.config_file: str = str(SCRIPT_DIR / "config.toml")
        self.hyperparams_file: str = str(SCRIPT_DIR / "hyperparameters.toml")

        parser = argparse.ArgumentParser(description="Train diffusion on frozen VAE latents.")
        parser.add_argument(
            "--mode", help="Run mode (debug/prod)", choices=["debug", "prod"], default="debug"
        )
        args: argparse.Namespace = parser.parse_args()

        self.config: SimpleNamespace = load_config(args.mode, self.config_file)
        self.hyperparams: SimpleNamespace = load_hyperparams(self.hyperparams_file)
        self.log = setup_logger(self.config)
        set_reproducibility_seed(self.hyperparams.SEED)

        # Separate telemetry fields from the VAE run: diffusion tracks the
        # composite noise-prediction loss (MSE + L1_WEIGHT * L1) on train and
        # test latents, plus the train-time L1 component for visibility.
        self.telemetry_fieldnames: list[str] = ["epoch", "train_loss", "l1_loss", "test_loss"]

        self.device: torch.device = resolve_device(self.config.DEVICE, self.log)
        self.log.info(
            f"Starting diffusion training with run_id: <red>{self.config.runtime.RUN_ID}</red> "
            f"at time: {datetime.fromtimestamp(self.config.runtime.START_TIME)} on device: {self.device}"
        )

        # Frozen VAE supplies latents (encoder) and renders samples (decoder).
        self.vae: VAEModel
        self.vae_archparams: SimpleNamespace
        self.vae, self.vae_archparams = build_vae_from_checkpoint(
            self.config.VAE_CHECKPOINT_PATH, self.device
        )
        self.log.info(
            f"Loaded frozen VAE from {self.config.VAE_CHECKPOINT_PATH} "
            f"(latent_dim={self.vae_archparams.LATENT_DIM}, "
            f"num_stages={self.vae_archparams.NUM_STAGES})"
        )

        # Fail fast when the diffusion width does not match the VAE latents;
        # training on silently mismatched shapes would waste the whole run.
        if self.hyperparams.LATENT_DIM != self.vae_archparams.LATENT_DIM:
            raise ValueError(
                f"Diffusion LATENT_DIM ({self.hyperparams.LATENT_DIM}) does not match "
                f"the VAE checkpoint latent dim ({self.vae_archparams.LATENT_DIM}). "
                f"Update Diffusion/hyperparameters.toml or point VAE_CHECKPOINT_PATH "
                f"at a matching VAE checkpoint."
            )

        self.model: LatentDiffusion = LatentDiffusion(self.hyperparams).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hyperparams.LEARNING_RATE
        )
        self.telemetry_writer = TelemetryWriter(
            fieldnames=self.telemetry_fieldnames,
            directory=self.config.TELEMETRY_LOG_DIRECTORY,
            config=self.config,
        )

        # Dataset-level latent standardization (mean/std over encoder outputs).
        # Computed once before training; stored on CPU and saved in checkpoints.
        self.latent_mean: torch.Tensor = torch.zeros(self.hyperparams.LATENT_DIM)
        self.latent_std: torch.Tensor = torch.ones(self.hyperparams.LATENT_DIM)

    def _make_image_dataloader(
        self,
        image_dir: str,
        max_samples: int,
        shuffle: bool,
    ) -> DataLoader[FloorPlanDataset]:
        """Create a dataloader over one image split folder.

        Args:
            image_dir (str): Directory with the split's images.
            max_samples (int): Cap on the dataset size (negative = all images).
            shuffle (bool): Whether to shuffle batches (True for train).

        Returns:
            DataLoader: Configured image dataloader for that split.
        """
        preload: bool = bool(self.hyperparams.PRELOAD_IMAGES)
        self.log.info(
            f"Building image dataset from {image_dir} (preload into RAM: {preload})..."
        )
        dataset = FloorPlanDataset(
            image_dir=image_dir,
            image_size=self.vae_archparams.IMAGE_SIZE,
            background_color=self.hyperparams.BACKGROUND_COLOR,
            max_samples=max_samples,
            preload=preload,
        )
        return DataLoader(
            dataset,
            batch_size=self.hyperparams.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=self.hyperparams.NUM_WORKERS,
            pin_memory=self.device.type == "cuda",
        )

    @torch.no_grad()
    def _encode_images(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Encode an image batch to latent vectors with the frozen VAE.

        Args:
            image_batch (torch.Tensor): Images, shape (n, c, h, w) in [0, 1].

        Returns:
            torch.Tensor: Latent vectors on CPU, shape (n, latent_dim).
        """
        # The VAE stays frozen, so no gradients are tracked here; only the
        # denoiser trains. USE_MU_ONLY picks the deterministic encoder mean,
        # otherwise reparameterized samples act as stochastic augmentation.
        batch: torch.Tensor = image_batch.to(self.device)
        mu: torch.Tensor
        logvar: torch.Tensor
        mu, logvar = self.vae.encoder(batch)
        if self.hyperparams.USE_MU_ONLY:
            latents: torch.Tensor = mu
        else:
            latents = self.vae.reparameterize(mu, logvar)
        return latents.cpu()

    def compute_latent_stats(self, image_loader: DataLoader[FloorPlanDataset]) -> None:
        """Estimate latent mean/std over train images for standardization.

        Streams encoder outputs (capped at LATENT_STATS_SAMPLES images) with a
        running sum so the full split never needs to sit in memory twice.

        Args:
            image_loader (DataLoader): Train image dataloader (shuffle=True is fine).
        """
        if not self.hyperparams.NORMALIZE_LATENTS:
            self.log.info("NORMALIZE_LATENTS=false; training on raw VAE latents.")
            return
        sum_latents: torch.Tensor = torch.zeros(self.hyperparams.LATENT_DIM)
        sum_squares: torch.Tensor = torch.zeros(self.hyperparams.LATENT_DIM)
        counted_images: int = 0
        for image_batch in image_loader:
            latents: torch.Tensor = self._encode_images(image_batch)
            # Respect the configured cap so stats stay cheap on huge datasets.
            remaining: int = self.hyperparams.LATENT_STATS_SAMPLES - counted_images
            if remaining <= 0:
                break
            latents = latents[:remaining]
            sum_latents += latents.sum(dim=0)
            sum_squares += (latents ** 2).sum(dim=0)
            counted_images += latents.shape[0]
        if counted_images == 0:
            raise ValueError("Train image loader is empty; cannot compute latent stats.")
        self.latent_mean = sum_latents / counted_images
        variance: torch.Tensor = sum_squares / counted_images - self.latent_mean ** 2
        # Floor the std so near-constant latent dims cannot blow up on division.
        self.latent_std = variance.clamp_min(0.0).sqrt().clamp_min(self.hyperparams.LATENT_STD_MIN)
        self.log.info(
            f"Latent stats over N={counted_images}: mean~{self.latent_mean.mean().item():.4f} "
            f"std~{self.latent_std.mean().item():.4f}"
        )

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Standardize latents with the dataset mean/std (identity when disabled).

        Args:
            latents (torch.Tensor): Raw VAE latents, shape (n, latent_dim).

        Returns:
            torch.Tensor: Standardized latents, same shape.
        """
        if not self.hyperparams.NORMALIZE_LATENTS:
            return latents
        return (latents - self.latent_mean) / self.latent_std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Invert the dataset standardization before VAE decoding.

        Args:
            latents (torch.Tensor): Standardized latents, shape (n, latent_dim).

        Returns:
            torch.Tensor: Raw-scale latents the VAE decoder expects.
        """
        if not self.hyperparams.NORMALIZE_LATENTS:
            return latents
        return latents * self.latent_std.to(latents.device) + self.latent_mean.to(latents.device)

    def encode_split_to_latents(
        self, image_loader: DataLoader[FloorPlanDataset], split_name: str
    ) -> torch.Tensor:
        """Encode a whole image split once into cached (normalized) latents.

        Caching keeps diffusion epochs MLP-only: the convolutional VAE encoder
        runs a single time instead of on every epoch.

        Args:
            image_loader (DataLoader): Image dataloader for the split.
            split_name (str): Human-readable split name for logging.

        Returns:
            torch.Tensor: Normalized latents on CPU, shape (num_images, latent_dim).
        """
        encoded_chunks: list[torch.Tensor] = []
        for image_batch in image_loader:
            encoded_chunks.append(self.normalize_latents(self._encode_images(image_batch)))
        latents: torch.Tensor = torch.cat(encoded_chunks, dim=0)
        self.log.info(f"Encoded {split_name} split to latents: {tuple(latents.shape)}")
        return latents

    def build_latent_loaders(self) -> tuple[DataLoader[TensorDataset], DataLoader[TensorDataset]]:
        """Build cached latent dataloaders for train and test.

        Returns:
            tuple[DataLoader, DataLoader]: (train latent loader, test latent loader).
        """
        train_img_dl: DataLoader[FloorPlanDataset] = self._make_image_dataloader(
            image_dir=self.config.TRAIN_IMAGES_DIR,
            max_samples=self.hyperparams.MAX_SAMPLES,
            shuffle=True,
        )
        test_img_dl: DataLoader[FloorPlanDataset] = self._make_image_dataloader(
            image_dir=self.config.TEST_IMAGES_DIR,
            max_samples=-1,
            shuffle=False,
        )
        self.log.info(
            f"Train images: {len(train_img_dl.dataset)} | Test images: {len(test_img_dl.dataset)}"
        )

        self.compute_latent_stats(train_img_dl)
        train_latents: torch.Tensor = self.encode_split_to_latents(train_img_dl, "train")
        test_latents: torch.Tensor = self.encode_split_to_latents(test_img_dl, "test")

        train_latent_dl: DataLoader[TensorDataset] = DataLoader(
            TensorDataset(train_latents),
            batch_size=self.hyperparams.BATCH_SIZE,
            shuffle=True,
        )
        test_latent_dl: DataLoader[TensorDataset] = DataLoader(
            TensorDataset(test_latents),
            batch_size=self.hyperparams.BATCH_SIZE,
            shuffle=False,
        )
        return train_latent_dl, test_latent_dl

    def diffusion_loss(self, clean_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the composite noise-prediction loss on a clean latent batch.

        Samples a random timestep and Gaussian noise per image, corrupts the
        latents with the forward process, and scores the denoiser's noise
        prediction with MSE plus a weighted L1 term (the L1 term makes the
        prediction robust to outlier noise draws, mirroring the VAE loss).

        Args:
            clean_latents (torch.Tensor): Clean latents x_0 on the device,
                shape (batch_size, latent_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                (total_loss, mse_loss, l1_loss) scalars with gradients for
                the denoiser, where total = mse + L1_WEIGHT * l1.
        """
        batch_size: int = clean_latents.shape[0]
        timesteps: torch.Tensor = torch.randint(
            0, self.model.num_timesteps, (batch_size,), device=clean_latents.device
        )
        noise: torch.Tensor = torch.randn_like(clean_latents)
        noisy_latents: torch.Tensor = self.model.q_sample(clean_latents, timesteps, noise)
        predicted_noise: torch.Tensor = self.model(noisy_latents, timesteps)
        # Mean-reduced error terms so the loss scale is batch-size independent.
        mse_loss: torch.Tensor = nn.functional.mse_loss(predicted_noise, noise)
        l1_loss: torch.Tensor = nn.functional.l1_loss(predicted_noise, noise)
        total_loss: torch.Tensor = mse_loss + self.hyperparams.L1_WEIGHT * l1_loss
        return total_loss, mse_loss, l1_loss

    def training_step(self, latent_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one optimization step on a single latent batch.

        Args:
            latent_batch (torch.Tensor): Clean latent batch (CPU cache).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Detached (total_loss, l1_loss)
                scalars for logging.
        """
        clean_latents: torch.Tensor = latent_batch.to(self.device)
        total_loss, _, l1_loss = self.diffusion_loss(clean_latents)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), l1_loss.detach()

    def run_epoch(self, dataloader: DataLoader[TensorDataset]) -> tuple[float, float]:
        """Train over the whole latent dataloader once and average the losses.

        Args:
            dataloader (DataLoader): Train latent dataloader.

        Returns:
            tuple[float, float]: Epoch-averaged (train_loss, l1_loss).
        """
        self.model.train()
        total_sum: float = 0.0
        l1_sum: float = 0.0
        num_batches: int = 0
        for (latent_batch,) in dataloader:
            total_loss, l1_loss = self.training_step(latent_batch)
            total_sum += total_loss.item()
            l1_sum += l1_loss.item()
            num_batches += 1
        return total_sum / num_batches, l1_sum / num_batches

    @torch.no_grad()
    def evaluate_test(
        self, dataloader: DataLoader[TensorDataset], max_samples: int | None = None
    ) -> dict[str, float]:
        """Average the composite noise-prediction loss over cached test latents.

        Args:
            dataloader (DataLoader): Test latent dataloader (shuffle=False).
            max_samples (int | None): Optional cap on evaluated latents
                (e.g. TEST_LOSS_SAMPLES for the per-epoch check). None = all.

        Returns:
            dict[str, float]: {"test_loss", "l1_loss", "samples"} averaged per image.
        """
        self.model.eval()
        total_sum: float = 0.0
        l1_sum: float = 0.0
        counted: int = 0
        for (latent_batch,) in dataloader:
            remaining: int | None = None if max_samples is None else max_samples - counted
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None:
                latent_batch = latent_batch[:remaining]
            clean_latents: torch.Tensor = latent_batch.to(self.device)
            total_loss, _, l1_loss = self.diffusion_loss(clean_latents)
            num_images: int = clean_latents.shape[0]
            total_sum += total_loss.item() * num_images
            l1_sum += l1_loss.item() * num_images
            counted += num_images
        if counted == 0:
            self.log.error("Test latent set is empty - cannot evaluate the model.")
            return {"test_loss": 0.0, "l1_loss": 0.0, "samples": 0.0}
        return {"test_loss": total_sum / counted, "l1_loss": l1_sum / counted, "samples": float(counted)}

    @torch.no_grad()
    def save_generation_grid(self, epoch: int) -> None:
        """Sample novel latents from noise and save them as a decoded image grid.

        The latents come from the full reverse chain starting at pure noise,
        are denormalized back to VAE scale, and are decoded with the frozen
        VAE decoder. The file follows the ``generation_epoch_<N>_<run>.png``
        convention so the generic dashboard's "Generations" panel picks it up
        with no code change.

        Args:
            epoch (int): Current epoch, used in the grid file name.
        """
        output_dir = Path(self.config.OUTPUT_SAMPLES_DIRECTORY)
        output_dir.mkdir(parents=True, exist_ok=True)

        sampled: torch.Tensor = self.model.sample(
            self.hyperparams.NUM_GEN_SAMPLES, self.hyperparams.LATENT_DIM, self.device
        )
        generated: torch.Tensor = self.vae.decode(self.denormalize_latents(sampled)).cpu()
        gen_path: Path = output_dir / f"generation_epoch_{epoch}_{self.config.runtime.RUN_ID}.png"
        save_image_grid(generated, self.hyperparams.GRID_NROW, gen_path)
        self.log.info(f"Saved generation grid to {gen_path}")

    def save_checkpoint(self, epoch: int) -> None:
        """Save the denoiser weights plus latent stats for reproducible sampling.

        Args:
            epoch (int): Current epoch, used in the checkpoint file name.
        """
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        checkpoint_path: str = os.path.join(
            self.config.MODEL_SAVE_PATH,
            f"diffusion_{self.config.runtime.RUN_ID}_epoch_{epoch}.pt",
        )
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "hyperparameters": vars(self.hyperparams),
                "latent_mean": self.latent_mean,
                "latent_std": self.latent_std,
                "vae_checkpoint": self.config.VAE_CHECKPOINT_PATH,
                "vae_hyperparameters": vars(self.vae_archparams),
            },
            checkpoint_path,
        )
        self.log.info(f"Saved checkpoint to {checkpoint_path}")

    def train(self) -> None:
        """Run the full training loop with per-epoch test evaluation.

        Encodes both splits to cached latents once, then every epoch trains on
        all train latents, evaluates the composite loss on a fixed test
        subset, writes a telemetry row, saves a generation grid for the
        dashboard, and checkpoints periodically.
        """
        torch.autograd.set_detect_anomaly(self.config.IS_DEBUG_TORCH)
        os.makedirs(self.config.OUTPUT_SAMPLES_DIRECTORY, exist_ok=True)

        train_dl, test_dl = self.build_latent_loaders()

        self.log.info(f"Torch version: {torch.__version__}")
        execution_start: float = time.time()

        for epoch in range(1, self.hyperparams.NUM_EPOCHS + 1):
            epoch_start: float = time.time()

            # 1) Train on all cached train latents (cheap MLP-only epoch).
            train_loss, l1_loss = self.run_epoch(train_dl)

            # 2) Evaluate on the configured test subset (e.g. 100 latents).
            test_metrics: dict[str, float] = self.evaluate_test(
                test_dl, max_samples=self.hyperparams.TEST_LOSS_SAMPLES
            )

            # 3) Telemetry row every TELEMETRY_INTERVAL epochs (plus the final
            # epoch) so the dashboard updates live.
            if (
                epoch % self.hyperparams.TELEMETRY_INTERVAL == 0
                or epoch == self.hyperparams.NUM_EPOCHS
            ):
                self.telemetry_writer.log(
                    epoch=epoch,
                    train_loss=train_loss,
                    l1_loss=l1_loss,
                    test_loss=test_metrics["test_loss"],
                )

            # 4) Generation grid every epoch (visible in the server dashboard).
            self.save_generation_grid(epoch)

            # 5) Periodic checkpoints plus one at the end of training.
            if epoch % self.hyperparams.CHECKPOINT_INTERVAL == 0 or epoch == self.hyperparams.NUM_EPOCHS:
                self.save_checkpoint(epoch)

            epoch_time: float = time.time() - epoch_start
            total_elapsed: float = time.time() - execution_start
            self.log.info(
                f"Epoch {epoch}/{self.hyperparams.NUM_EPOCHS} | "
                f"train_loss: {train_loss:.4f} (l1: {l1_loss:.4f}) | "
                f"test_loss: {test_metrics['test_loss']:.4f} "
                f"(N={int(test_metrics['samples'])}) | "
                f"epoch_time: {epoch_time:.1f}s | total: {total_elapsed:.1f}s"
            )

        # Final evaluation over ALL cached test latents.
        final_metrics: dict[str, float] = self.evaluate_test(test_dl)
        self.save_checkpoint(self.hyperparams.NUM_EPOCHS)
        self.log.success(
            f"FINAL TEST RESULT on full test set (N={int(final_metrics['samples'])}): "
            f"test_loss={final_metrics['test_loss']:.4f} | "
            f"l1_loss={final_metrics['l1_loss']:.4f}"
        )
