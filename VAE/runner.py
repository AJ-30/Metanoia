"""
Training runner for the VAE model.

Loads configuration and hyperparameters from TOML files, builds separate
train and test dataloaders from the split dataset folders, runs the training
loop on the configured device (MPS / CUDA / CPU), and evaluates the model on
the test split after every epoch and once on the full test set at the end.

The dataset is folder-per-sample (one folder holding every render of a layout).
Which render is learned — the full render (``full_png``) or the floorplan-only
render (``floorplan_only``) — is selected by ``TRAIN_IMAGE_VARIANT`` in
``config.toml``; the render name itself lives in the
``[default.IMAGE_VARIANTS]`` table of the same file, never hardcoded here.

Progress is logged with the shared project logger, written to telemetry CSVs
via helpers/telemetry_writer, and reconstruction grids of fixed test samples
are saved every epoch so the dashboard server can display them.
"""

from VAE.model import VAEModel, vae_loss
from helpers.telemetry_writer import TelemetryWriter
from helpers.logger import setup_logger
from helpers.hyperparams import load_hyperparams
from helpers.config import load_config
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torchvision import transforms

# Project root directory so that the shared `helpers/` package (logger, config,
# telemetry writer) can be imported no matter which directory the runner is
# launched from.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# Directory of this file, used to resolve relative TOML / output paths.
SCRIPT_DIR: Path = Path(__file__).resolve().parent

# Name of the config.toml key holding the selected render variant for the two
# trainers (VAE, Diffusion). The YOLO augmentation pipeline reads source samples
# and uses its own key, SOURCE_IMAGE_VARIANT (YOLO/config.toml).
TRAIN_IMAGE_VARIANT_KEY: str = "TRAIN_IMAGE_VARIANT"


class FloorPlanDataset(Dataset):
    """PyTorch Dataset that lazy-loads and preprocesses floor-plan images.

    Expected on-disk layout (one folder per sample, see the dataset README):

    .. code-block:: text

        <split_dir>/
          <hash>_original/            # or <hash>_augmented_<NNN>
            full.png                  # image_file_name = "full.png"
            floorplan_only.png        # image_file_name = "floorplan_only.png"
            no_tv.png
            ground_truth.svg
            labels.txt

    Each sample folder holds every render of that layout, so ``image_file_name``
    selects which one this dataset reads (``full_png`` or ``floorplan_only`` in
    the TOML; the mapping lives in ``config.toml`` under
    ``[default.IMAGE_VARIANTS]``). Each RGBA image is composited onto a solid
    background (to drop the alpha channel), resized to a square and converted
    to a float tensor in [0, 1]. Optionally preloads every image into RAM (as
    uint8) to avoid disk I/O during training and keep the GPU fed.

    This class is the single source of truth for image preprocessing and is
    reused as-is by the Diffusion runner, which takes its ``image_size`` from
    the frozen VAE checkpoint.
    """

    def __init__(
        self,
        split_dir: str,
        image_file_name: str,
        image_size: int,
        background_color: list[int],
        max_samples: int,
        preload: bool = False,
    ) -> None:
        """Initialize the dataset.

        Args:
            split_dir (str): Split folder (e.g. ``.../dataset/train``) holding
                one sub-folder per sample.
            image_file_name (str): Render to read from inside every sample
                folder (e.g. ``"full.png"`` or ``"floorplan_only.png"``). A
                sample folder that lacks it is skipped.
            image_size (int): Target side length of the resized square images.
            background_color (list[int]): RGBA tuple used as the backdrop when
                flattening the alpha channel.
            max_samples (int): Cap on the number of images loaded. Negative
                values mean "use every sample folder in the split".
            preload (bool): When True, decode every image into RAM once at
                construction (as compact uint8 tensors) so training epochs no
                longer hit the disk. This is the main fix for GPU starvation.

        Raises:
            FileNotFoundError: If ``split_dir`` does not exist, or if no sample
                folder in it provides ``image_file_name``. Failing loudly here
                is essential: a silently empty dataset only surfaces much later
                as a division by zero inside the training loop.
        """
        self.split_dir: str = split_dir
        self.image_file_name: str = image_file_name
        split_path: Path = Path(split_dir)
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Dataset split directory missing: {split_dir}")

        # One image per sample folder: enumerate sub-directories and keep those
        # that actually carry the selected render. Iterdir() is deliberately
        # non-recursive so a stray nested folder is never mistaken for a sample.
        self.image_paths: list[str] = sorted(
            str(sample_dir / image_file_name)
            for sample_dir in split_path.iterdir()
            if sample_dir.is_dir() and (sample_dir / image_file_name).is_file()
        )
        if not self.image_paths:
            raise FileNotFoundError(
                f"No sample folder under {split_dir} provides '{image_file_name}'. "
                "Check the split path and the configured image variant."
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

        # Forward-reference so the cache list can be typed.
        self.cache: list[torch.Tensor] | None = None
        if preload:
            # Decode all images once into RAM. Each cached image is 3*S*S bytes
            # (uint8 instead of float32), so 15k x 3*256*256 is ~2.9 GB. Turn
            # PRELOAD_IMAGES off when the split outgrows the machine's RAM.
            self.cache = [self._load_uint8(path) for path in self.image_paths]

    def _load_float(self, image_path: str) -> torch.Tensor:
        """Decode one image file into a [0, 1] float tensor.

        Args:
            image_path (str): Path of the image to load.

        Returns:
            torch.Tensor: Preprocessed tensor of shape (3, image_size, image_size) in [0, 1].
        """
        img: Image.Image = Image.open(image_path)
        return self.transform(img)

    def _load_uint8(self, image_path: str) -> torch.Tensor:
        """Decode one image file into a compact uint8 tensor.

        Args:
            image_path (str): Path of the image to load.

        Returns:
            torch.Tensor: Preprocessed tensor of shape (3, image_size, image_size) in [0, 255].
        """
        # Rounding the float [0, 1] tensor to uint8 saves 4x memory vs float32.
        return (self._load_float(image_path) * 255.0).round().to(torch.uint8)

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

        Serves from the preloaded RAM cache when available (avoids disk reads
        and repeated PNG decoding during training), otherwise decodes on demand.

        Args:
            index (int): Index of the image to load.

        Returns:
            torch.Tensor: Tensor of shape (out_channels, image_size, image_size) in [0, 1].
        """
        if self.cache is not None:
            return self.cache[index].float() / 255.0
        return self._load_float(self.image_paths[index])


def resolve_image_file_name(config: SimpleNamespace, variant_key: str) -> str:
    """Resolve the configured render variant to a filename inside a sample folder.

    Reads two keys of the model ``config.toml`` (not ``hyperparameters.toml``),
    so no render filename is ever hardcoded in Python: the variant key (e.g.
    ``TRAIN_IMAGE_VARIANT = "full_png"``) selects an entry of the
    ``IMAGE_VARIANTS`` table (e.g. ``{full_png = "full.png", ...}``) declared as
    ``[default.IMAGE_VARIANTS]`` in the TOML file. Shared by the VAE, Diffusion
    and YOLO runners, which each keep their own copy of the table and name their
    variant key after what they read (``TRAIN_IMAGE_VARIANT`` for the two
    trainers, ``SOURCE_IMAGE_VARIANT`` for the YOLO augmentation pipeline).

    Args:
        config (SimpleNamespace): Config namespace produced by
            ``helpers.config.load_config``, holding ``variant_key`` (str) and
            ``IMAGE_VARIANTS`` (dict[str, str]).
        variant_key (str): Name of the ``config.toml`` key holding the selected
            variant name, e.g. ``"TRAIN_IMAGE_VARIANT"``. A wrong or missing key
            raises ``AttributeError`` naming it, so TOML drift fails fast.

    Returns:
        str: Filename of the selected render, e.g. ``"full.png"``.

    Raises:
        AttributeError: If the config file does not define ``variant_key`` (the
            key must live in the ``[default]`` table of ``config.toml``, not in a
            ``[debug]``/``[prod]`` table).
        KeyError: If the variant is not a key of the ``IMAGE_VARIANTS`` table
            (fail fast on TOML typos).
        TypeError: If ``IMAGE_VARIANTS`` is not a mapping of variant name to
            filename, or the selected entry is not a non-empty string.
    """
    if not hasattr(config, variant_key):
        raise AttributeError(
            f"config.toml must define {variant_key} in its [default] table "
            "(a key placed after [debug]/[prod] belongs to that mode's table, "
            "not to [default])."
        )
    variant: str = str(getattr(config, variant_key))
    variants: object = getattr(config, "IMAGE_VARIANTS", None)
    if not isinstance(variants, dict):
        raise TypeError(
            "config.toml must define a [default.IMAGE_VARIANTS] table mapping "
            f"variant name -> render filename, got {type(variants).__name__}."
        )
    if variant not in variants:
        raise KeyError(
            f"{variant_key}='{variant}' is not in the [default.IMAGE_VARIANTS] "
            f"table. Known variants: {sorted(variants.keys())}."
        )
    file_name: object = variants[variant]
    if not isinstance(file_name, str) or not file_name:
        raise TypeError(
            f"[default.IMAGE_VARIANTS].{variant} must be a non-empty filename, "
            f"got {file_name!r}."
        )
    return file_name


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

    The grid stacks the ground-truth test images in the top row and their
    one-shot reconstructions in the bottom row, so training progress can be
    inspected visually.

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
    grid: torch.Tensor = make_grid(
        comparison, nrow=sample_batch.shape[0], padding=2, normalize=True)
    grid_img = transforms.ToPILImage()(grid)
    grid_img.save(str(save_path))
    model.train()
    log.info(f"Saved reconstruction grid to {save_path}")


def save_generation_grid(
    model: VAEModel,
    num_samples: int,
    latent_dim: int,
    image_size: int,
    device: torch.device,
    save_path: Path,
    log: SimpleNamespace,
) -> None:
    """Generate novel images by sampling latent vectors and save a single grid image.

    Contrary to reconstructions (which pass fixed test images through the
    encoder), generations draw ``num_samples`` vectors from the prior
    N(0, I) latent distribution and decode them straight to images. This shows
    what the trained latent space produces from scratch.

    Args:
        model (VAEModel): The VAE model used for generation.
        num_samples (int): Number of images to generate.
        latent_dim (int): Dimensionality of the model's latent space.
        image_size (int): Side length of the generated images.
        device (torch.device): Device the model lives on.
        save_path (Path): Destination path for the saved grid PNG.
        log (SimpleNamespace): Configured logger.
    """
    model.eval()
    with torch.no_grad():
        # Re-draw random latents each epoch so the generated content evolves.
        z: torch.Tensor = torch.randn(num_samples, latent_dim, device=device)
        generated: torch.Tensor = model.decode(z).cpu()

    # Non-square / non-power-of-2 grid cell counts fall back to nrow=min(10, n).
    nrow: int = min(num_samples, 10)
    grid: torch.Tensor = make_grid(
        generated, nrow=nrow, padding=2, normalize=True)
    grid_img = transforms.ToPILImage()(grid)
    grid_img.save(str(save_path))
    model.train()
    log.info(f"Saved generation grid to {save_path}")


class VAERunner:
    """Runner that wires together config, hyperparameters, model and training loop."""

    def __init__(self) -> None:
        """Initialize the runner: load config/hyperparams, set up the device and model."""
        self.config_file: str = str(SCRIPT_DIR / "config.toml")
        self.hyperparams_file: str = str(SCRIPT_DIR / "hyperparameters.toml")

        parser = argparse.ArgumentParser(
            description="Train the VAE on floor-plan images.")
        parser.add_argument(
            "--mode", help="Run mode (debug/prod)", choices=["debug", "prod"], default="debug"
        )
        args: argparse.Namespace = parser.parse_args()

        self.config: SimpleNamespace = load_config(args.mode, self.config_file)
        self.hyperparams: SimpleNamespace = load_hyperparams(
            self.hyperparams_file)
        self.log = setup_logger(self.config)

        # Telemetry fieldnames for the shared CSV telemetry writer. test_loss is
        # written every epoch so the dashboard shows the test-loss curve too.
        self.telemetry_fieldnames: list[str] = [
            "epoch", "train_loss", "recon_loss", "kl_loss", "test_loss",
        ]

        self.device: torch.device = resolve_device(
            self.config.DEVICE, self.log)
        self.log.info(
            f"Starting VAE training with run_id: <red>{self.config.runtime.RUN_ID}</red> "
            f"at time: {datetime.fromtimestamp(self.config.runtime.START_TIME)} on device: {self.device}"
        )

        # Training state.
        self.model: VAEModel = VAEModel(self.hyperparams).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hyperparams.LEARNING_RATE)
        self.telemetry_writer = TelemetryWriter(
            fieldnames=self.telemetry_fieldnames,
            directory=self.config.TELEMETRY_LOG_DIRECTORY,
            config=self.config,
        )

    def _make_dataloader(
        self,
        split_dir: str,
        max_samples: int,
        shuffle: bool,
    ) -> DataLoader[FloorPlanDataset]:
        """Create a single dataset + dataloader for one split folder.

        Args:
            split_dir (str): Split folder holding one sub-folder per sample.
            max_samples (int): Cap on the dataset size (negative = all images).
            shuffle (bool): Whether to shuffle batches (True for train).

        Returns:
            DataLoader: Configured dataloader for that split.
        """
        preload: bool = bool(self.hyperparams.PRELOAD_IMAGES)
        image_file_name: str = resolve_image_file_name(
            self.config, TRAIN_IMAGE_VARIANT_KEY)
        self.log.info(
            f"Building dataset from {split_dir} using render '{image_file_name}' "
            f"(variant '{self.config.TRAIN_IMAGE_VARIANT}', "
            f"preload into RAM: {preload})..."
        )
        dataset = FloorPlanDataset(
            split_dir=split_dir,
            image_file_name=image_file_name,
            image_size=self.hyperparams.IMAGE_SIZE,
            background_color=self.hyperparams.BACKGROUND_COLOR,
            max_samples=max_samples,
            preload=preload,
        )
        return DataLoader(
            dataset,
            batch_size=self.hyperparams.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=self.hyperparams.NUM_WORKERS,
            # Only CUDA supports pinned host memory; MPS/silicon ignores it.
            pin_memory=self.device.type == "cuda",
        )

    def build_dataloaders(self) -> tuple[DataLoader[FloorPlanDataset], DataLoader[FloorPlanDataset]]:
        """Build the train and test dataloaders from the split dataset folders.

        Returns:
            tuple[DataLoader, DataLoader]: (train_dataloader, test_dataloader).
        """
        # Train uses all training images (MAX_SAMPLES caps it when debugging);
        # test always uses every test image so the final evaluation is complete.
        train_dl: DataLoader[FloorPlanDataset] = self._make_dataloader(
            split_dir=self.config.TRAIN_IMAGES_DIR,
            max_samples=self.hyperparams.MAX_SAMPLES,
            shuffle=True,
        )
        test_dl: DataLoader[FloorPlanDataset] = self._make_dataloader(
            split_dir=self.config.TEST_IMAGES_DIR,
            max_samples=-1,
            shuffle=False,  # deterministic subset for the per-epoch test loss
        )
        self.log.info(
            f"Train images: {len(train_dl.dataset)} | Test images: {len(test_dl.dataset)}"
        )
        return train_dl, test_dl

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
            batch, reconstruction, mu, logvar,
            kl_weight=self.hyperparams.KL_WEIGHT,
            l1_weight=self.hyperparams.L1_WEIGHT,
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

    def evaluate_test(
        self,
        dataloader: DataLoader[FloorPlanDataset],
        max_samples: int | None = None,
    ) -> dict[str, float]:
        """Average the VAE loss over test images in no-grad mode.

        Args:
            dataloader (DataLoader): The test dataloader (shuffle=False).
            max_samples (int | None): Optional cap on how many test images to
                evaluate (e.g. 100 for the per-epoch check). None = entire test set.

        Returns:
            dict[str, float]: Averaged {"test_loss", "recon_loss", "kl_loss",
                "samples"} weighted by the number of evaluated images.
        """
        self.model.eval()
        total_samples: int = 0
        sum_total: float = 0.0
        sum_recon: float = 0.0
        sum_kl: float = 0.0

        with torch.no_grad():
            for batch in dataloader:
                # Slice to exactly the remaining requested samples so the
                # per-epoch test loss covers exactly TEST_LOSS_SAMPLES images.
                if max_samples is not None:
                    remaining: int = max_samples - total_samples
                    if remaining <= 0:
                        break
                    batch = batch[:remaining]

                batch = batch.to(self.device)
                reconstruction, mu, logvar = self.model(batch)
                total_loss, recon_loss, kl_loss = vae_loss(
                    batch, reconstruction, mu, logvar,
                    kl_weight=self.hyperparams.KL_WEIGHT,
                    l1_weight=self.hyperparams.L1_WEIGHT,
                )
                num_images: int = batch.shape[0]
                sum_total += total_loss.item() * num_images
                sum_recon += recon_loss.item() * num_images
                sum_kl += kl_loss.item() * num_images
                total_samples += num_images

        self.model.train()

        if total_samples == 0:
            self.log.error("Test set is empty - cannot evaluate the model.")
            return {"test_loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0, "samples": 0.0}

        # Losses from vae_loss are already per-image averages; weight by count.
        return {
            "test_loss": sum_total / total_samples,
            "recon_loss": sum_recon / total_samples,
            "kl_loss": sum_kl / total_samples,
            "samples": float(total_samples),
        }

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
        """Run the full training loop with per-epoch test evaluation.

        Every epoch: train on all train images, evaluate test loss on a fixed
        subset of TEST_LOSS_SAMPLES images, save a reconstruction grid of 10
        fixed test samples and a generation grid of NUM_GEN_SAMPLES fresh
        latent samples, and write a telemetry row. After the last epoch the
        final test loss is computed over the entire test set.
        """
        torch.autograd.set_detect_anomaly(self.config.IS_DEBUG_TORCH)
        os.makedirs(self.config.OUTPUT_SAMPLES_DIRECTORY, exist_ok=True)

        train_dl, test_dl = self.build_dataloaders()

        # Hold onto one fixed batch of TEST images for the per-epoch
        # reconstruction grids, so progress is always shown on the same samples.
        fixed_batch: torch.Tensor = next(iter(test_dl))[
            : self.hyperparams.NUM_RECON_SAMPLES]

        self.log.info(f"Torch version: {torch.__version__}")
        execution_start: float = time.time()

        for epoch in range(1, self.hyperparams.NUM_EPOCHS + 1):
            epoch_start: float = time.time()

            # 1) Train on all training images.
            train_loss, recon_loss, kl_loss = self.run_epoch(train_dl)

            # 2) Evaluate test loss on the configured sample subset (e.g. 100).
            test_metrics: dict[str, float] = self.evaluate_test(
                test_dl, max_samples=self.hyperparams.TEST_LOSS_SAMPLES
            )

            # 3) Write telemetry every TELEMETRY_INTERVAL epochs (plus the final
            # epoch) so the dashboard updates live. With interval 1 a row is
            # written after every single epoch.
            if (
                epoch % self.hyperparams.TELEMETRY_INTERVAL == 0
                or epoch == self.hyperparams.NUM_EPOCHS
            ):
                self.telemetry_writer.log(
                    epoch=epoch,
                    train_loss=train_loss,
                    recon_loss=recon_loss,
                    kl_loss=kl_loss,
                    test_loss=test_metrics["test_loss"],
                )

            # 4) Save reconstruction + generation grids every epoch (visible in
            # the server dashboard).
            sample_path: Path = (
                Path(self.config.OUTPUT_SAMPLES_DIRECTORY)
                / f"reconstruction_epoch_{epoch}_{self.config.runtime.RUN_ID}.png"
            )
            save_reconstruction_grid(
                self.model, fixed_batch, self.device, sample_path, self.log)

            gen_path: Path = (
                Path(self.config.OUTPUT_SAMPLES_DIRECTORY)
                / f"generation_epoch_{epoch}_{self.config.runtime.RUN_ID}.png"
            )
            save_generation_grid(
                model=self.model,
                num_samples=self.hyperparams.NUM_GEN_SAMPLES,
                latent_dim=self.hyperparams.LATENT_DIM,
                image_size=self.hyperparams.IMAGE_SIZE,
                device=self.device,
                save_path=gen_path,
                log=self.log,
            )

            # 5) Periodic checkpoints (saved every CHECKPOINT_INTERVAL epochs plus at the
            # end of training).
            if epoch % self.hyperparams.CHECKPOINT_INTERVAL == 0 or epoch == self.hyperparams.NUM_EPOCHS:
                self.save_checkpoint(epoch)

            epoch_time: float = time.time() - epoch_start
            total_elapsed: float = time.time() - execution_start
            self.log.info(
                f"Epoch {epoch}/{self.hyperparams.NUM_EPOCHS} | "
                f"train_loss: {train_loss:.4f} | test_loss: {test_metrics['test_loss']:.4f} "
                f"(N={int(test_metrics['samples'])}) | "
                f"epoch_time: {epoch_time:.1f}s | total: {total_elapsed:.1f}s"
            )

        # Final evaluation over the ENTIRE test set.
        final_metrics: dict[str, float] = self.evaluate_test(test_dl)
        self.save_checkpoint(self.hyperparams.NUM_EPOCHS)
        self.log.success(
            f"FINAL TEST RESULT on full test set (N={int(final_metrics['samples'])}): "
            f"test_loss={final_metrics['test_loss']:.4f} | "
            f"recon_loss={final_metrics['recon_loss']:.4f} | "
            f"kl_loss={final_metrics['kl_loss']:.4f}"
        )
