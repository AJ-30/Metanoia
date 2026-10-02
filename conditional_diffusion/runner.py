"""
Training runner for conditional latent diffusion: floorplan -> arrangement.

Pipeline: each sample folder provides BOTH a floorplan and a furniture
segmentation mask of the same layout. Both masks are encoded once with their
respective FROZEN mask_VAE encoders (floorplan VAE supplies the condition
vector ``c``, arrangement VAE supplies the diffusion target ``x_0``) and the
pairs are cached in RAM as tensors (70k x 256 floats is only ~72 MB per
space). The conditional DDPM denoiser then trains purely on those cached
pairs, which keeps every epoch cheap (MLP only, no convolutions). Sampling
encodes a floorplan mask to ``c``, runs the reverse chain from noise
conditioned on ``c``, and decodes with the frozen arrangement decoder.

Every ``GRID_INTERVAL`` epochs the runner saves a single unified grid with
``3 + NUM_GEN_DRAWS`` rows per sample column — full render, floorplan ground
truth, arrangement ground truth, then one overlay row per fresh noise draw —
under the ``reconstruction_epoch_<N>_<run>.png`` convention so the generic
dashboard's "Reconstructions" panel picks it up with no server change.

Mask helpers (integer-id loading, one-vs-rest encoding, colorized rendering)
are reused from ``mask_VAE.masking`` instead of being duplicated here.
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.utils import make_grid

# Project root, so the shared `helpers/` package and the `mask_VAE` package
# are importable no matter which directory the runner is launched from.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from helpers.config import load_config
from helpers.hyperparams import load_hyperparams
from helpers.logger import setup_logger
from helpers.telemetry_writer import TelemetryWriter

from conditional_diffusion.model import ConditionalLatentDiffusion
from mask_VAE.masking import (
    MaskGroup,
    MaskLayout,
    encode_masks_to_channels,
    load_color_index,
    render_group_rgb,
)
from mask_VAE.model import MaskVAEModel
from VAE.runner import resolve_device

# Directory of this file, used to resolve relative TOML / output paths.
SCRIPT_DIR: Path = Path(__file__).resolve().parent

# Image modes accepted when opening a source mask. Palette images are accepted
# because Pillow reports them as "P", but they are converted to single channel
# before use: the payload is class ids either way, not colors.
ACCEPTED_MASK_IMAGE_MODES: frozenset[str] = frozenset({"L", "P"})

# Single channel depth of the source masks after conversion.
MASK_IMAGE_MODE: str = "L"

# Resampling filter for class-id masks. Never an interpolating filter: it
# would blend neighbouring class ids into values that belong to no class.
MASK_RESAMPLE: Image.Resampling = Image.Resampling.NEAREST


def set_reproducibility_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs for reproducible runs.

    Args:
        seed (int): Seed applied to ``random`` and ``torch`` (CPU + CUDA).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_color_index_file(config_value: str, color_index_dir: Path, fallback_dir: Path) -> Path:
    """Locate a color index JSON, falling back to the dataset root.

    Args:
        config_value (str): Filename from ``config.toml``.
        color_index_dir (Path): Configured ``MASK_COLOR_INDEX_DIR``, searched first.
        fallback_dir (Path): Parent of the train split, searched on a miss.

    Returns:
        Path: Resolved color index JSON path.

    Raises:
        FileNotFoundError: If the file is in neither directory.
    """
    candidate: Path = color_index_dir / config_value
    if candidate.is_file():
        return candidate
    fallback_candidate: Path = fallback_dir / config_value
    if fallback_candidate.is_file():
        return fallback_candidate
    raise FileNotFoundError(
        f"Mask color index '{config_value}' not found in {color_index_dir} "
        f"or in the dataset root fallback {fallback_dir}."
    )


def build_frozen_mask_vae(
    checkpoint_path: str,
    color_index_path: Path,
    mask_file_name: str,
    device: torch.device,
) -> tuple[MaskVAEModel, SimpleNamespace, MaskLayout]:
    """Rebuild a mask VAE from its checkpoint and freeze it for conditioning.

    The architecture is reconstructed from the hyperparameters embedded in
    the checkpoint itself (not from the current TOML), so older runs stay
    usable even after the mask_VAE config moved on. The render layout is
    rebuilt from the checkpoint's own channel indices plus the dataset's
    color index JSON, so grids use the dataset's authoritative colors.

    Args:
        checkpoint_path (str): Path to the ``mask_vae_<run>_epoch_<N>.pt`` file.
        color_index_path (Path): Dataset color index JSON for this VAE's group.
        mask_file_name (str): Mask filename inside each sample folder.
        device (torch.device): Device to place the frozen VAE on.

    Returns:
        tuple[MaskVAEModel, SimpleNamespace, MaskLayout]: Frozen VAE in eval
            mode, its archparams, and its single-group render/encode layout.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Mask VAE checkpoint not found: {checkpoint_path}")
    checkpoint: dict[str, object] = torch.load(checkpoint_path, map_location=device)
    saved_hyperparams: dict[str, object] = checkpoint["hyperparameters"]
    state_dict: dict[str, torch.Tensor] = checkpoint["model_state_dict"]
    archparams = SimpleNamespace(
        MASK_CHANNELS=saved_hyperparams["MASK_CHANNELS"],
        LATENT_DIM=saved_hyperparams["LATENT_DIM"],
        BASE_CHANNELS=saved_hyperparams["BASE_CHANNELS"],
        IMAGE_SIZE=saved_hyperparams["IMAGE_SIZE"],
        DROPOUT=saved_hyperparams["DROPOUT"],
        LATENT_ADDITIONAL_NOISE=saved_hyperparams.get("LATENT_ADDITIONAL_NOISE", 0.0),
    )
    vae = MaskVAEModel(archparams)
    vae.load_state_dict(state_dict)
    # Freeze every parameter: neither encoder trains here.
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    # Single-group layout for this checkpoint: channel indices come from the
    # checkpoint, names and colors from the dataset's color index.
    group_names: list[str] = list(checkpoint["mask_groups"])
    if len(group_names) != 1:
        raise ValueError(
            f"Conditional diffusion needs single-group mask VAEs, but "
            f"{checkpoint_path} holds groups {group_names}."
        )
    group_name: str = group_names[0]
    channel_indices: tuple[int, ...] = tuple(
        checkpoint["mask_channel_indices"][group_name]
    )
    class_names, colors = load_color_index(color_index_path)
    if len(class_names) != len(channel_indices):
        raise ValueError(
            f"{color_index_path} defines {len(class_names)} classes but "
            f"{checkpoint_path} reserves {len(channel_indices)} channels."
        )
    group = MaskGroup(
        name=group_name,
        mask_file_name=mask_file_name,
        class_names=tuple(class_names),
        colors=tuple(colors),
        channel_indices=channel_indices,
    )
    layout = MaskLayout(
        groups=(group,),
        class_names=tuple(f"{group_name}/{name}" for name in class_names[1:]),
        num_channels=len(channel_indices),
        foreground_channels=tuple(range(1, len(channel_indices))),
    )
    # Prepend the shared background label so class_names indexes by channel.
    layout = MaskLayout(
        groups=layout.groups,
        class_names=("background", *layout.class_names),
        num_channels=layout.num_channels,
        foreground_channels=layout.foreground_channels,
    )
    return vae.to(device), archparams, layout


class PairedMaskDataset(Dataset):
    """Torch dataset yielding paired floorplan/arrangement mask tensors.

    One item is the (floorplan, arrangement) one-vs-rest binary tensors of a
    single layout, each shaped ``(num_channels, image_size, image_size)`` and
    valued ``0.0`` or ``1.0``. Only sample folders providing BOTH masks are
    kept, so every pair describes the same layout. Streaming only (no RAM
    cache): a cached pair costs ~1.9 MB and the full split would need ~133 GB.
    Masks are encoded to latents once and cached there instead.
    """

    def __init__(
        self,
        split_dir: str,
        cond_mask_file: str,
        target_mask_file: str,
        image_file: str,
        cond_layout: MaskLayout,
        target_layout: MaskLayout,
        image_size: int,
        max_samples: int,
    ) -> None:
        """Index the sample folders of one split that provide both masks.

        Args:
            split_dir (str): Split folder holding one sub-folder per sample.
            cond_mask_file (str): Floorplan mask filename inside each sample.
            target_mask_file (str): Furniture mask filename inside each sample.
            image_file (str): Full render filename inside each sample, shown
                as the top grid row (not required for indexing).
            cond_layout (MaskLayout): Single-group floorplan layout for encoding.
            target_layout (MaskLayout): Single-group arrangement layout for encoding.
            image_size (int): Square side length the masks are resized to.
            max_samples (int): Cap on samples; negative keeps every sample.

        Raises:
            ValueError: If ``image_size`` is not positive.
            FileNotFoundError: If the split is missing or no sample provides
                both masks.
        """
        if image_size < 1:
            raise ValueError(f"image_size must be positive, got {image_size}.")
        self.cond_mask_file: str = cond_mask_file
        self.target_mask_file: str = target_mask_file
        self.image_file: str = image_file
        self.cond_layout: MaskLayout = cond_layout
        self.target_layout: MaskLayout = target_layout
        self.image_size: int = image_size

        split_path: Path = Path(split_dir)
        if not split_path.is_dir():
            raise FileNotFoundError(f"Dataset split directory missing: {split_dir}.")
        self.sample_dirs: list[Path] = sorted(
            candidate
            for candidate in split_path.iterdir()
            if candidate.is_dir()
            and (candidate / cond_mask_file).is_file()
            and (candidate / target_mask_file).is_file()
        )
        if not self.sample_dirs:
            raise FileNotFoundError(
                f"No sample folder under {split_dir} provides both "
                f"'{cond_mask_file}' and '{target_mask_file}'."
            )
        if max_samples >= 0:
            self.sample_dirs = self.sample_dirs[:max_samples]

    def __len__(self) -> int:
        """Return the number of paired samples.

        Returns:
            int: Length of the resolved sample list.
        """
        return len(self.sample_dirs)

    def image_path(self, index: int) -> Path:
        """Return the full render path of one sample.

        Args:
            index (int): Position in the sorted sample list.

        Returns:
            Path: Path of the sample's full render (may not exist; the grid
                renderer falls back to a blank cell then).
        """
        return self.sample_dirs[index] / self.image_file

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the encoded mask pair of one sample.

        Args:
            index (int): Position in the sorted sample list.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (floorplan, arrangement)
                Float32 binary tensors of shape ``(cond_channels, S, S)`` and
                ``(target_channels, S, S)``.
        """
        sample_dir: Path = self.sample_dirs[index]
        cond_ids: torch.Tensor = self._load_mask(sample_dir / self.cond_mask_file)
        target_ids: torch.Tensor = self._load_mask(sample_dir / self.target_mask_file)
        cond: torch.Tensor = encode_masks_to_channels([cond_ids], self.cond_layout).squeeze(0)
        target: torch.Tensor = encode_masks_to_channels([target_ids], self.target_layout).squeeze(0)
        return cond, target

    def _load_mask(self, mask_path: Path) -> torch.Tensor:
        """Read one mask PNG as a class-id tensor.

        Args:
            mask_path (Path): Path of the ``*_segmentation.png`` file.

        Returns:
            torch.Tensor: int64 ``(image_size, image_size)`` class-id tensor,
                resized with nearest-neighbour interpolation.

        Raises:
            ValueError: If the image is not a single-channel class-id mask.
        """
        with Image.open(mask_path) as image:
            if image.mode not in ACCEPTED_MASK_IMAGE_MODES:
                raise ValueError(
                    f"{mask_path} has image mode {image.mode!r}, but a "
                    f"{MASK_IMAGE_MODE!r} integer class-id mask is required."
                )
            resized: Image.Image = image.convert(MASK_IMAGE_MODE).resize(
                (self.image_size, self.image_size),
                resample=MASK_RESAMPLE,
            )
            array: np.ndarray = np.array(resized, dtype=np.uint8)
        return torch.from_numpy(array.astype(np.int64))


def overlay_furniture_on_floorplan(
    floor_rgb: torch.Tensor,
    furn_rgb: torch.Tensor,
    furn_ids: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Blend predicted furniture pixels over a floorplan base image.

    Furniture background pixels stay transparent so rooms and walls show
    through; only pixels where the predicted arrangement class is not
    background are blended with weight ``alpha``.

    Args:
        floor_rgb (torch.Tensor): ``(B, 3, H, W)`` floorplan base in ``0..1``.
        furn_rgb (torch.Tensor): ``(B, 3, H, W)`` arrangement render in ``0..1``.
        furn_ids (torch.Tensor): ``(B, H, W)`` predicted arrangement class ids.
        alpha (float): Blend weight of furniture foreground pixels.

    Returns:
        torch.Tensor: ``(B, 3, H, W)`` overlay in ``0..1``.
    """
    foreground: torch.Tensor = (furn_ids != 0).to(floor_rgb.dtype).unsqueeze(1)
    weight: torch.Tensor = foreground * alpha
    return floor_rgb * (1.0 - weight) + furn_rgb * weight


def render_full_image_batch(
    image_paths: list[Path],
    height: int,
    width: int,
    background: tuple[int, int, int, int],
) -> torch.Tensor:
    """Render full images to an RGB batch for the top grid row.

    Each RGBA render is composited onto ``background`` to drop the alpha
    channel, then resized to ``(height, width)`` with bilinear filtering (a
    photo-style render has no class ids to corrupt, unlike the
    nearest-neighbour mask path). A missing or unreadable file becomes a
    blank white cell so one bad file never kills the training loop.

    Args:
        image_paths (list[Path]): Full render paths, one per grid column.
        height (int): Target cell height in pixels.
        width (int): Target cell width in pixels.
        background (tuple[int, int, int, int]): RGBA backdrop for flattening.

    Returns:
        torch.Tensor: Float32 ``(B, 3, H, W)`` RGB batch in ``0..1``.
    """
    cells: list[torch.Tensor] = []
    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                if image.mode != "RGBA":
                    rgb: Image.Image = image.convert("RGB")
                else:
                    backdrop: Image.Image = Image.new("RGBA", image.size, background)
                    rgb = Image.alpha_composite(backdrop, image).convert("RGB")
                cell: Image.Image = rgb.resize((width, height), resample=Image.Resampling.BILINEAR)
        except (OSError, ValueError):
            cell = Image.new("RGB", (width, height), (255, 255, 255))
        cells.append(transforms.ToTensor()(cell))
    return torch.stack(cells, dim=0)


class ConditionalDiffusionRunner:
    """Runner wiring frozen mask VAEs, conditional diffusion and the grid loop."""

    def __init__(self, mode: str | None = None) -> None:
        """Load config/hyperparams, frozen VAEs, diffusion model and telemetry.

        Args:
            mode (str | None): Run mode selecting the ``[debug]`` or ``[prod]``
                table of ``config.toml``. Parsed from ``--mode`` when None.
        """
        self.config_file: str = str(SCRIPT_DIR / "config.toml")
        self.hyperparams_file: str = str(SCRIPT_DIR / "hyperparameters.toml")

        if mode is None:
            parser = argparse.ArgumentParser(
                description="Train floorplan-conditioned diffusion on frozen mask VAE latents.")
            parser.add_argument(
                "--mode",
                help="Run mode (debug/prod)",
                choices=["debug", "prod"],
                default="debug",
            )
            mode = parser.parse_args().mode

        self.config: SimpleNamespace = load_config(mode, self.config_file)
        self.hyperparams: SimpleNamespace = load_hyperparams(self.hyperparams_file)
        self.log = setup_logger(self.config)
        set_reproducibility_seed(int(self.hyperparams.SEED))

        self.telemetry_fieldnames: list[str] = ["epoch", "train_loss", "l1_loss", "test_loss"]

        self.device: torch.device = resolve_device(self.config.DEVICE, self.log)
        self.log.info(
            f"Starting conditional diffusion with run_id: "
            f"<red>{self.config.runtime.RUN_ID}</red> at time: "
            f"{datetime.fromtimestamp(self.config.runtime.START_TIME)} on device: {self.device}"
        )

        # Frozen VAEs: floorplan supplies the condition, arrangement supplies
        # the diffusion space and the decoder for visualization.
        color_dir: Path = Path(str(self.config.MASK_COLOR_INDEX_DIR))
        fallback_dir: Path = Path(str(self.config.TRAIN_IMAGES_DIR)).parent
        self.cond_vae: MaskVAEModel
        self.cond_arch: SimpleNamespace
        self.cond_layout: MaskLayout
        self.cond_vae, self.cond_arch, self.cond_layout = build_frozen_mask_vae(
            str(self.config.COND_VAE_CHECKPOINT_PATH),
            resolve_color_index_file(str(self.config.COND_COLOR_INDEX_FILE), color_dir, fallback_dir),
            str(self.config.COND_MASK_FILE),
            self.device,
        )
        self.target_vae: MaskVAEModel
        self.target_arch: SimpleNamespace
        self.target_layout: MaskLayout
        self.target_vae, self.target_arch, self.target_layout = build_frozen_mask_vae(
            str(self.config.TARGET_VAE_CHECKPOINT_PATH),
            resolve_color_index_file(str(self.config.TARGET_COLOR_INDEX_FILE), color_dir, fallback_dir),
            str(self.config.TARGET_MASK_FILE),
            self.device,
        )
        self.log.info(
            f"Frozen condition VAE: {self.config.COND_VAE_CHECKPOINT_PATH} "
            f"(channels={self.cond_arch.MASK_CHANNELS}, latent={self.cond_arch.LATENT_DIM})"
        )
        self.log.info(
            f"Frozen target VAE: {self.config.TARGET_VAE_CHECKPOINT_PATH} "
            f"(channels={self.target_arch.MASK_CHANNELS}, latent={self.target_arch.LATENT_DIM})"
        )

        # Fail fast on shape drift between the two frozen encoders and this
        # run's hyperparameters; silent mismatches would waste the whole run.
        if int(self.cond_arch.IMAGE_SIZE) != int(self.target_arch.IMAGE_SIZE):
            raise ValueError(
                f"Frozen VAEs disagree on IMAGE_SIZE: condition "
                f"{self.cond_arch.IMAGE_SIZE} vs target {self.target_arch.IMAGE_SIZE}."
            )
        if int(self.cond_arch.IMAGE_SIZE) != int(self.hyperparams.IMAGE_SIZE):
            raise ValueError(
                f"hyperparameters IMAGE_SIZE ({self.hyperparams.IMAGE_SIZE}) does not "
                f"match the frozen VAEs ({self.cond_arch.IMAGE_SIZE})."
            )
        if int(self.hyperparams.COND_DIM) != int(self.cond_arch.LATENT_DIM):
            raise ValueError(
                f"COND_DIM ({self.hyperparams.COND_DIM}) does not match the "
                f"condition checkpoint latent dim ({self.cond_arch.LATENT_DIM})."
            )
        if int(self.hyperparams.LATENT_DIM) != int(self.target_arch.LATENT_DIM):
            raise ValueError(
                f"LATENT_DIM ({self.hyperparams.LATENT_DIM}) does not match the "
                f"target checkpoint latent dim ({self.target_arch.LATENT_DIM})."
            )

        self.model: ConditionalLatentDiffusion = ConditionalLatentDiffusion(
            self.hyperparams
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.hyperparams.LEARNING_RATE)
        )
        self.telemetry_writer = TelemetryWriter(
            fieldnames=self.telemetry_fieldnames,
            directory=self.config.TELEMETRY_LOG_DIRECTORY,
            config=self.config,
        )

        # Dataset-level standardization stats, estimated once before training.
        self.latent_mean: torch.Tensor = torch.zeros(int(self.hyperparams.LATENT_DIM))
        self.latent_std: torch.Tensor = torch.ones(int(self.hyperparams.LATENT_DIM))
        self.cond_mean: torch.Tensor = torch.zeros(int(self.hyperparams.COND_DIM))
        self.cond_std: torch.Tensor = torch.ones(int(self.hyperparams.COND_DIM))

    def _make_paired_dataloader(
        self,
        split_dir: str,
        max_samples: int,
        shuffle: bool,
        batch_size: int,
    ) -> DataLoader[PairedMaskDataset]:
        """Create a dataloader over paired masks of one split.

        Args:
            split_dir (str): Split folder holding one sub-folder per sample.
            max_samples (int): Cap on samples; negative keeps all.
            shuffle (bool): Whether to shuffle batches (True for train).
            batch_size (int): Mask pairs per batch.

        Returns:
            DataLoader[PairedMaskDataset]: Loader yielding
                ``(floor, furn)`` binary mask batches.
        """
        dataset = PairedMaskDataset(
            split_dir=split_dir,
            cond_mask_file=str(self.config.COND_MASK_FILE),
            target_mask_file=str(self.config.TARGET_MASK_FILE),
            image_file=str(self.config.FULL_IMAGE_FILE),
            cond_layout=self.cond_layout,
            target_layout=self.target_layout,
            image_size=int(self.hyperparams.IMAGE_SIZE),
            max_samples=max_samples,
        )
        self.log.info(f"Paired masks from {split_dir}: {len(dataset)} samples.")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(self.hyperparams.NUM_WORKERS),
            pin_memory=self.device.type == "cuda",
        )

    @torch.no_grad()
    def _encode_pair(
        self, cond_masks: torch.Tensor, target_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one mask pair to latent vectors with the frozen VAEs.

        Args:
            cond_masks (torch.Tensor): Floorplan masks, shape (n, Cc, H, W).
            target_masks (torch.Tensor): Arrangement masks, shape (n, Ct, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (condition, target) latent
                vectors on CPU, each shape (n, dim).
        """
        cond_mu: torch.Tensor
        cond_logvar: torch.Tensor
        cond_mu, cond_logvar = self.cond_vae.encoder(cond_masks.to(self.device))
        target_mu: torch.Tensor
        target_logvar: torch.Tensor
        target_mu, target_logvar = self.target_vae.encoder(target_masks.to(self.device))
        if bool(self.hyperparams.USE_MU_ONLY):
            cond: torch.Tensor = cond_mu
            target: torch.Tensor = target_mu
        else:
            cond = self.cond_vae.reparameterize(cond_mu, cond_logvar)
            target = self.target_vae.reparameterize(target_mu, target_logvar)
        return cond.cpu(), target.cpu()

    def compute_latent_stats(self, image_loader: DataLoader[PairedMaskDataset]) -> None:
        """Estimate mean/std of both latent spaces over train pairs.

        Streams frozen-encoder outputs (capped at LATENT_STATS_SAMPLES pairs)
        with running sums so the full split never sits in memory twice.

        Args:
            image_loader (DataLoader[PairedMaskDataset]): Train mask loader.
        """
        if not bool(self.hyperparams.NORMALIZE_LATENTS) and not bool(
            self.hyperparams.NORMALIZE_COND
        ):
            self.log.info("Normalization disabled; training on raw latents.")
            return
        latent_dim: int = int(self.hyperparams.LATENT_DIM)
        cond_dim: int = int(self.hyperparams.COND_DIM)
        sum_lat: torch.Tensor = torch.zeros(latent_dim)
        sum_sq_lat: torch.Tensor = torch.zeros(latent_dim)
        sum_cond: torch.Tensor = torch.zeros(cond_dim)
        sum_sq_cond: torch.Tensor = torch.zeros(cond_dim)
        counted: int = 0
        for cond_masks, target_masks in image_loader:
            remaining: int = int(self.hyperparams.LATENT_STATS_SAMPLES) - counted
            if remaining <= 0:
                break
            cond_masks = cond_masks[:remaining]
            target_masks = target_masks[:remaining]
            cond, target = self._encode_pair(cond_masks, target_masks)
            sum_lat += target.sum(dim=0)
            sum_sq_lat += (target ** 2).sum(dim=0)
            sum_cond += cond.sum(dim=0)
            sum_sq_cond += (cond ** 2).sum(dim=0)
            counted += target.shape[0]
        if counted == 0:
            raise ValueError("Train mask loader is empty; cannot compute latent stats.")
        std_min: float = float(self.hyperparams.LATENT_STD_MIN)
        self.latent_mean = sum_lat / counted
        variance: torch.Tensor = sum_sq_lat / counted - self.latent_mean ** 2
        self.latent_std = variance.clamp_min(0.0).sqrt().clamp_min(std_min)
        self.cond_mean = sum_cond / counted
        cond_var: torch.Tensor = sum_sq_cond / counted - self.cond_mean ** 2
        self.cond_std = cond_var.clamp_min(0.0).sqrt().clamp_min(std_min)
        self.log.info(
            f"Latent stats over N={counted}: target mean~{self.latent_mean.mean().item():.4f} "
            f"std~{self.latent_std.mean().item():.4f} | cond mean~{self.cond_mean.mean().item():.4f} "
            f"std~{self.cond_std.mean().item():.4f}"
        )

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Standardize arrangement latents (identity when disabled).

        Args:
            latents (torch.Tensor): Raw target latents, shape (n, latent_dim).

        Returns:
            torch.Tensor: Standardized latents, same shape.
        """
        if not bool(self.hyperparams.NORMALIZE_LATENTS):
            return latents
        return (latents - self.latent_mean) / self.latent_std

    def normalize_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Standardize floorplan conditions (identity when disabled).

        Args:
            cond (torch.Tensor): Raw condition latents, shape (n, cond_dim).

        Returns:
            torch.Tensor: Standardized conditions, same shape.
        """
        if not bool(self.hyperparams.NORMALIZE_COND):
            return cond
        return (cond - self.cond_mean) / self.cond_std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Invert target standardization before arrangement decoding.

        Args:
            latents (torch.Tensor): Standardized latents, shape (n, latent_dim).

        Returns:
            torch.Tensor: Raw-scale latents the arrangement decoder expects.
        """
        if not bool(self.hyperparams.NORMALIZE_LATENTS):
            return latents
        return latents * self.latent_std.to(latents.device) + self.latent_mean.to(latents.device)

    def encode_split_to_latents(
        self, image_loader: DataLoader[PairedMaskDataset], split_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a whole split once into cached (condition, target) latents.

        Caching keeps diffusion epochs MLP-only: both convolutional encoders
        run a single time instead of on every epoch.

        Args:
            image_loader (DataLoader[PairedMaskDataset]): Mask loader.
            split_name (str): Human-readable split name for logging.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Normalized (conditions, targets)
                on CPU, shapes (N, cond_dim) and (N, latent_dim).
        """
        cond_chunks: list[torch.Tensor] = []
        target_chunks: list[torch.Tensor] = []
        encoded_pairs: int = 0
        log_every: int = int(self.hyperparams.ENCODE_LOG_INTERVAL)
        for batch_index, (cond_masks, target_masks) in enumerate(image_loader):
            cond, target = self._encode_pair(cond_masks, target_masks)
            cond_chunks.append(self.normalize_cond(cond))
            target_chunks.append(self.normalize_latents(target))
            encoded_pairs += target.shape[0]
            if (batch_index + 1) % log_every == 0:
                self.log.info(
                    f"Encoded {split_name} pairs: {encoded_pairs} "
                    f"({batch_index + 1} batches)..."
                )
        cond_latents: torch.Tensor = torch.cat(cond_chunks, dim=0)
        target_latents: torch.Tensor = torch.cat(target_chunks, dim=0)
        self.log.info(
            f"Encoded {split_name} split to latents: cond {tuple(cond_latents.shape)}, "
            f"target {tuple(target_latents.shape)}"
        )
        return cond_latents, target_latents

    def latent_cache_fingerprint(self, train_samples: int, test_samples: int) -> dict[str, object]:
        """Build the fingerprint identifying the cached latent pairs.

        Every input that changes the encoded values is covered: both frozen
        checkpoint files (path + mtime + size), both split dirs, the mask
        size, the mu-vs-sample switch, the sample caps, the normalization
        switches and the resulting sample counts. A run whose fingerprint
        matches the stored one reuses the cache; anything else re-encodes.

        Args:
            train_samples (int): Number of paired train samples listed.
            test_samples (int): Number of paired test samples listed.

        Returns:
            dict[str, object]: JSON-comparable fingerprint of the encode inputs.
        """
        cond_ckpt: str = str(self.config.COND_VAE_CHECKPOINT_PATH)
        target_ckpt: str = str(self.config.TARGET_VAE_CHECKPOINT_PATH)
        cond_stat: os.stat_result = os.stat(cond_ckpt)
        target_stat: os.stat_result = os.stat(target_ckpt)
        return {
            "cond_ckpt": cond_ckpt,
            "cond_ckpt_mtime": cond_stat.st_mtime,
            "cond_ckpt_size": cond_stat.st_size,
            "target_ckpt": target_ckpt,
            "target_ckpt_mtime": target_stat.st_mtime,
            "target_ckpt_size": target_stat.st_size,
            "train_dir": str(self.config.TRAIN_IMAGES_DIR),
            "test_dir": str(self.config.TEST_IMAGES_DIR),
            "image_size": int(self.hyperparams.IMAGE_SIZE),
            "use_mu_only": bool(self.hyperparams.USE_MU_ONLY),
            "max_samples": int(self.hyperparams.MAX_SAMPLES),
            "normalize_latents": bool(self.hyperparams.NORMALIZE_LATENTS),
            "normalize_cond": bool(self.hyperparams.NORMALIZE_COND),
            "stats_samples": int(self.hyperparams.LATENT_STATS_SAMPLES),
            "train_samples": train_samples,
            "test_samples": test_samples,
        }

    def _latent_cache_paths(self) -> tuple[Path, Path]:
        """Return the train/test latent cache file paths.

        Returns:
            tuple[Path, Path]: (train cache path, test cache path).
        """
        cache_dir: Path = Path(str(self.config.LATENT_CACHE_DIR))
        return cache_dir / "train_pairs.pt", cache_dir / "test_pairs.pt"

    def try_load_latent_cache(
        self, train_samples: int, test_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Load cached latent pairs when the fingerprint matches this run.

        A match restores the stored standardization stats as well, so the
        stats pass is skipped too. Any missing/corrupt/mismatched cache
        returns None so the caller falls back to encoding from scratch.

        Args:
            train_samples (int): Number of paired train samples listed.
            test_samples (int): Number of paired test samples listed.

        Returns:
            tuple or None: (train_cond, train_target, test_cond, test_target)
                on CPU when the cache hits, else None.
        """
        train_path, test_path = self._latent_cache_paths()
        if not train_path.is_file() or not test_path.is_file():
            return None
        try:
            expected: dict[str, object] = self.latent_cache_fingerprint(train_samples, test_samples)
            train_blob: dict[str, object] = torch.load(str(train_path), map_location="cpu")
            test_blob: dict[str, object] = torch.load(str(test_path), map_location="cpu")
            if train_blob.get("fingerprint") != expected or test_blob.get("fingerprint") != expected:
                self.log.info("Latent cache fingerprint mismatch; re-encoding from scratch.")
                return None
            self.latent_mean = train_blob["latent_mean"]
            self.latent_std = train_blob["latent_std"]
            self.cond_mean = train_blob["cond_mean"]
            self.cond_std = train_blob["cond_std"]
            return (
                train_blob["cond"],
                train_blob["target"],
                test_blob["cond"],
                test_blob["target"],
            )
        except (OSError, RuntimeError, KeyError, ValueError, TypeError) as exc:
            self.log.warning(f"Ignoring unreadable latent cache ({exc}); re-encoding from scratch.")
            return None

    def save_latent_cache(
        self,
        train_cond: torch.Tensor,
        train_target: torch.Tensor,
        test_cond: torch.Tensor,
        test_target: torch.Tensor,
        train_samples: int,
        test_samples: int,
    ) -> None:
        """Write freshly encoded latent pairs plus stats and fingerprint.

        Args:
            train_cond (torch.Tensor): Cached train conditions on CPU.
            train_target (torch.Tensor): Cached train targets on CPU.
            test_cond (torch.Tensor): Cached test conditions on CPU.
            test_target (torch.Tensor): Cached test targets on CPU.
            train_samples (int): Number of paired train samples encoded.
            test_samples (int): Number of paired test samples encoded.
        """
        train_path, test_path = self._latent_cache_paths()
        train_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint: dict[str, object] = self.latent_cache_fingerprint(train_samples, test_samples)
        stats: dict[str, object] = {
            "fingerprint": fingerprint,
            "latent_mean": self.latent_mean,
            "latent_std": self.latent_std,
            "cond_mean": self.cond_mean,
            "cond_std": self.cond_std,
        }
        torch.save({**stats, "cond": train_cond, "target": train_target}, str(train_path))
        torch.save({**stats, "cond": test_cond, "target": test_target}, str(test_path))
        self.log.info(f"Saved latent cache to {train_path.parent}")

    def build_latent_loaders(
        self,
    ) -> tuple[
        DataLoader[TensorDataset],
        DataLoader[TensorDataset],
        tuple[torch.Tensor, torch.Tensor, list[Path]],
    ]:
        """Build cached paired-latent loaders plus the fixed grid batch.

        The frozen-encoder pass runs only on a cache miss: when the stored
        fingerprint matches this run's config, dataset and checkpoints, the
        latent pairs and stats are loaded from ``LATENT_CACHE_DIR`` instead.

        Returns:
            tuple: (train latent loader, test latent loader, fixed triple),
                where the triple is (floorplan masks, arrangement masks,
                full render paths) of ``NUM_GRID_SAMPLES`` test pairs.
        """
        encode_batch: int = int(self.hyperparams.IMAGE_ENCODE_BATCH_SIZE)
        train_img_dl: DataLoader[PairedMaskDataset] = self._make_paired_dataloader(
            split_dir=str(self.config.TRAIN_IMAGES_DIR),
            max_samples=int(self.hyperparams.MAX_SAMPLES),
            shuffle=True,
            batch_size=encode_batch,
        )
        test_img_dl: DataLoader[PairedMaskDataset] = self._make_paired_dataloader(
            split_dir=str(self.config.TEST_IMAGES_DIR),
            max_samples=-1,
            shuffle=False,
            batch_size=encode_batch,
        )

        cached = self.try_load_latent_cache(len(train_img_dl.dataset), len(test_img_dl.dataset))
        if cached is not None:
            self.log.info(
                f"Reusing cached latents from {self.config.LATENT_CACHE_DIR} "
                "(fingerprint match; skipping the encoder pass)."
            )
            train_cond, train_target, test_cond, test_target = cached
        else:
            self.compute_latent_stats(train_img_dl)
            train_cond, train_target = self.encode_split_to_latents(train_img_dl, "train")
            test_cond, test_target = self.encode_split_to_latents(test_img_dl, "test")
            self.save_latent_cache(
                train_cond,
                train_target,
                test_cond,
                test_target,
                len(train_img_dl.dataset),
                len(test_img_dl.dataset),
            )

        train_latent_dl: DataLoader[TensorDataset] = DataLoader(
            TensorDataset(train_cond, train_target),
            batch_size=int(self.hyperparams.BATCH_SIZE),
            shuffle=True,
        )
        test_latent_dl: DataLoader[TensorDataset] = DataLoader(
            TensorDataset(test_cond, test_target),
            batch_size=int(self.hyperparams.BATCH_SIZE),
            shuffle=False,
        )

        # Fixed test pairs back the unified grid, so every saved picture shows
        # the same floorplans and progress stays comparable across epochs.
        fixed_pair: tuple[torch.Tensor, torch.Tensor, list[Path]] = self._fixed_pair_batch(
            test_img_dl, int(self.hyperparams.NUM_GRID_SAMPLES), "unified"
        )
        return train_latent_dl, test_latent_dl, fixed_pair

    @staticmethod
    def _fixed_pair_batch(
        test_img_dl: DataLoader[PairedMaskDataset], count: int, kind: str
    ) -> tuple[torch.Tensor, torch.Tensor, list[Path]]:
        """Stack the first ``count`` test pairs as a fixed grid batch.

        Args:
            test_img_dl (DataLoader[PairedMaskDataset]): Test mask loader.
            count (int): Number of pairs to stack.
            kind (str): Grid kind for the error message.

        Returns:
            tuple: (floorplan masks, arrangement masks, full render paths),
                masks on CPU.

        Raises:
            ValueError: If the test split is empty.
        """
        dataset: PairedMaskDataset = test_img_dl.dataset
        take: int = min(count, len(dataset))
        if take == 0:
            raise ValueError(f"Test split is empty; cannot build the {kind} grid batch.")
        cond_masks: torch.Tensor = torch.stack([dataset[i][0] for i in range(take)])
        target_masks: torch.Tensor = torch.stack([dataset[i][1] for i in range(take)])
        image_paths: list[Path] = [dataset.image_path(i) for i in range(take)]
        return cond_masks, target_masks, image_paths

    def diffusion_loss(
        self, clean_targets: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the conditional noise-prediction loss on one latent pair batch.

        Samples a random timestep and Gaussian noise per layout, corrupts the
        arrangement latents with the forward process, and scores the denoiser's
        noise prediction (which also sees the floorplan condition) with MSE
        plus a weighted L1 term.

        Args:
            clean_targets (torch.Tensor): Clean arrangement latents x_0 on the
                device, shape (batch_size, latent_dim).
            cond (torch.Tensor): Floorplan conditions on the device, shape
                (batch_size, cond_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                (total_loss, mse_loss, l1_loss) scalars, where
                total = mse + L1_WEIGHT * l1.
        """
        batch_size: int = clean_targets.shape[0]
        timesteps: torch.Tensor = torch.randint(
            0, self.model.num_timesteps, (batch_size,), device=clean_targets.device
        )
        noise: torch.Tensor = torch.randn_like(clean_targets)
        noisy: torch.Tensor = self.model.q_sample(clean_targets, timesteps, noise)
        predicted: torch.Tensor = self.model(noisy, timesteps, cond)
        # Mean-reduced error terms so the loss scale is batch-size independent.
        mse_loss: torch.Tensor = torch.nn.functional.mse_loss(predicted, noise)
        l1_loss: torch.Tensor = torch.nn.functional.l1_loss(predicted, noise)
        total_loss: torch.Tensor = mse_loss + float(self.hyperparams.L1_WEIGHT) * l1_loss
        return total_loss, mse_loss, l1_loss

    def training_step(
        self, cond_batch: torch.Tensor, target_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one optimization step on a single paired-latent batch.

        Args:
            cond_batch (torch.Tensor): Condition batch from the CPU cache.
            target_batch (torch.Tensor): Clean arrangement batch from the cache.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Detached (total_loss, l1_loss).
        """
        cond: torch.Tensor = cond_batch.to(self.device)
        target: torch.Tensor = target_batch.to(self.device)
        total_loss, _, l1_loss = self.diffusion_loss(target, cond)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), l1_loss.detach()

    def run_epoch(self, dataloader: DataLoader[TensorDataset]) -> tuple[float, float]:
        """Train over the whole paired-latent loader once and average losses.

        Args:
            dataloader (DataLoader[TensorDataset]): Train loader of (cond, target).

        Returns:
            tuple[float, float]: Epoch-averaged (train_loss, l1_loss).
        """
        self.model.train()
        total_sum: float = 0.0
        l1_sum: float = 0.0
        num_batches: int = 0
        for cond_batch, target_batch in dataloader:
            total_loss, l1_loss = self.training_step(cond_batch, target_batch)
            total_sum += total_loss.item()
            l1_sum += l1_loss.item()
            num_batches += 1
        if num_batches == 0:
            raise ValueError("Train latent loader is empty.")
        return total_sum / num_batches, l1_sum / num_batches

    @torch.no_grad()
    def evaluate_test(
        self, dataloader: DataLoader[TensorDataset], max_samples: int | None = None
    ) -> dict[str, float]:
        """Average the conditional loss over cached test pairs.

        Args:
            dataloader (DataLoader[TensorDataset]): Test loader (shuffle=False).
            max_samples (int | None): Cap on evaluated pairs. None = all.

        Returns:
            dict[str, float]: {"test_loss", "l1_loss", "samples"} per pair.
        """
        self.model.eval()
        total_sum: float = 0.0
        l1_sum: float = 0.0
        counted: int = 0
        for cond_batch, target_batch in dataloader:
            remaining: int | None = None if max_samples is None else max_samples - counted
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None:
                cond_batch = cond_batch[:remaining]
                target_batch = target_batch[:remaining]
            cond: torch.Tensor = cond_batch.to(self.device)
            target: torch.Tensor = target_batch.to(self.device)
            total_loss, _, l1_loss = self.diffusion_loss(target, cond)
            num_pairs: int = target.shape[0]
            total_sum += total_loss.item() * num_pairs
            l1_sum += l1_loss.item() * num_pairs
            counted += num_pairs
        if counted == 0:
            self.log.error("Test latent set is empty - cannot evaluate the model.")
            return {"test_loss": 0.0, "l1_loss": 0.0, "samples": 0.0}
        return {"test_loss": total_sum / counted, "l1_loss": l1_sum / counted, "samples": float(counted)}

    @torch.no_grad()
    @torch.no_grad()
    def save_unified_grid(
        self,
        fixed_cond_masks: torch.Tensor,
        fixed_target_masks: torch.Tensor,
        fixed_image_paths: list[Path],
        epoch: int,
    ) -> None:
        """Save the single unified grid: GT rows plus generated draws.

        Rows (each of ``NUM_GRID_SAMPLES`` columns): full rendered image,
        floorplan ground truth, arrangement ground truth, then one overlay
        row per fresh noise draw (``NUM_GEN_DRAWS`` draws of the predicted
        arrangement blended over the floorplan base). The GT rows judge
        fidelity; the draw rows show the distribution
        p(arrangement | floorplan). Saved as
        ``reconstruction_epoch_<N>_<run>.png`` so the generic dashboard's
        "Reconstructions" panel picks it up with no server change.

        Args:
            fixed_cond_masks (torch.Tensor): Fixed floorplan masks on CPU,
                shape (n, Cc, H, W).
            fixed_target_masks (torch.Tensor): Paired arrangement masks on CPU,
                shape (n, Ct, H, W).
            fixed_image_paths (list[Path]): Full render paths, one per column.
            epoch (int): Current epoch, used in the grid file name.
        """
        output_dir = Path(str(self.config.OUTPUT_SAMPLES_DIRECTORY))
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        cond, _ = self._encode_pair(fixed_cond_masks, fixed_target_masks)
        cond_norm: torch.Tensor = self.normalize_cond(cond).to(self.device)
        num_conds: int = cond_norm.shape[0]
        draws: int = int(self.hyperparams.NUM_GEN_DRAWS)
        # Tile each condition `draws` times so one reverse chain covers every
        # (floorplan, noise draw) combination in a single batched pass.
        tiled_cond: torch.Tensor = cond_norm.repeat_interleave(draws, dim=0)
        sampled: torch.Tensor = self.model.sample_conditional(tiled_cond)
        pred_logits: torch.Tensor = self.target_vae.decode(
            self.denormalize_latents(sampled)
        ).cpu()

        cpu: torch.device = torch.device("cpu")
        cond_group: MaskGroup = self.cond_layout.groups[0]
        target_group: MaskGroup = self.target_layout.groups[0]
        _, _, cell_height, cell_width = fixed_cond_masks.shape
        background: tuple[int, int, int, int] = tuple(self.hyperparams.BACKGROUND_COLOR)
        full_rgb: torch.Tensor = render_full_image_batch(
            fixed_image_paths, cell_height, cell_width, background
        )
        floor_rgb: torch.Tensor = render_group_rgb(fixed_cond_masks, cond_group, cpu)
        furn_gt_rgb: torch.Tensor = render_group_rgb(fixed_target_masks, target_group, cpu)
        # Argmax over the target group's own channels gives predicted ids in
        # local class order, matching the color table.
        target_index: torch.Tensor = torch.tensor(
            target_group.channel_indices, dtype=torch.long
        )
        overlay_alpha: float = float(self.hyperparams.OVERLAY_ALPHA)
        num_target_channels: int = pred_logits.shape[1]
        draw_logits: torch.Tensor = pred_logits.view(num_conds, draws, num_target_channels, *pred_logits.shape[2:])
        overlay_rows: list[torch.Tensor] = []
        for draw in range(draws):
            draw_rgb: torch.Tensor = render_group_rgb(draw_logits[:, draw], target_group, cpu)
            draw_ids: torch.Tensor = draw_logits[:, draw].index_select(1, target_index).argmax(dim=1)
            overlay_rows.append(
                overlay_furniture_on_floorplan(floor_rgb, draw_rgb, draw_ids, overlay_alpha)
            )

        grid: torch.Tensor = make_grid(
            torch.cat([full_rgb, floor_rgb, furn_gt_rgb, *overlay_rows], dim=0),
            nrow=num_conds,
            padding=2,
            normalize=False,
        )
        # normalize=False matters: rows are already 0..1 color-table output and
        # make_grid's default rescaling would shift every dataset color.
        grid_path: Path = output_dir / f"reconstruction_epoch_{epoch}_{self.config.runtime.RUN_ID}.png"
        transforms.ToPILImage()(grid).save(str(grid_path))
        self.log.info(f"Saved unified grid to {grid_path}")
        self.model.train()

    def save_checkpoint(self, epoch: int) -> None:
        """Save the denoiser weights plus latent stats for reproducible sampling.

        Args:
            epoch (int): Current epoch, used in the checkpoint file name.
        """
        os.makedirs(str(self.config.MODEL_SAVE_PATH), exist_ok=True)
        checkpoint_path: str = os.path.join(
            str(self.config.MODEL_SAVE_PATH),
            f"conditional_diffusion_{self.config.runtime.RUN_ID}_epoch_{epoch}.pt",
        )
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "hyperparameters": vars(self.hyperparams),
                "latent_mean": self.latent_mean,
                "latent_std": self.latent_std,
                "cond_mean": self.cond_mean,
                "cond_std": self.cond_std,
                "cond_checkpoint": str(self.config.COND_VAE_CHECKPOINT_PATH),
                "target_checkpoint": str(self.config.TARGET_VAE_CHECKPOINT_PATH),
            },
            checkpoint_path,
        )
        self.log.info(f"Saved checkpoint to {checkpoint_path}")

    def train(self) -> None:
        """Run the full training loop with the periodic unified grid.

        Encodes both splits to cached pairs once, then every epoch trains on
        all train pairs, evaluates the conditional loss on a fixed test
        subset, writes a telemetry row, and every ``GRID_INTERVAL`` epochs
        saves the single unified grid (full render, floorplan, arrangement
        GT, generated draws). Checkpoints periodically.
        """
        torch.autograd.set_detect_anomaly(bool(self.config.IS_DEBUG_TORCH))
        os.makedirs(str(self.config.OUTPUT_SAMPLES_DIRECTORY), exist_ok=True)

        train_dl, test_dl, (fixed_cond, fixed_target, fixed_images) = (
            self.build_latent_loaders()
        )

        self.log.info(f"Torch version: {torch.__version__}")
        execution_start: float = time.time()

        for epoch in range(1, int(self.hyperparams.NUM_EPOCHS) + 1):
            epoch_start: float = time.time()

            # 1) Train on all cached train pairs (cheap MLP-only epoch).
            train_loss, l1_loss = self.run_epoch(train_dl)

            # 2) Evaluate on the configured test subset.
            test_metrics: dict[str, float] = self.evaluate_test(
                test_dl, max_samples=int(self.hyperparams.TEST_LOSS_SAMPLES)
            )

            # 3) Telemetry row so the dashboard updates live.
            if (
                epoch % int(self.hyperparams.TELEMETRY_INTERVAL) == 0
                or epoch == int(self.hyperparams.NUM_EPOCHS)
            ):
                self.telemetry_writer.log(
                    epoch=epoch,
                    train_loss=train_loss,
                    l1_loss=l1_loss,
                    test_loss=test_metrics["test_loss"],
                )

            # 4) Unified grid every GRID_INTERVAL epochs (plus the final
            # epoch): full render, floorplan, arrangement GT, generated draws.
            if epoch % int(self.hyperparams.GRID_INTERVAL) == 0 or epoch == int(
                self.hyperparams.NUM_EPOCHS
            ):
                self.save_unified_grid(fixed_cond, fixed_target, fixed_images, epoch)

            # 5) Periodic checkpoints plus one at the end of training.
            if epoch % int(self.hyperparams.CHECKPOINT_INTERVAL) == 0 or epoch == int(
                self.hyperparams.NUM_EPOCHS
            ):
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

        # Final evaluation over ALL cached test pairs.
        final_metrics: dict[str, float] = self.evaluate_test(test_dl)
        self.save_checkpoint(int(self.hyperparams.NUM_EPOCHS))
        self.log.success(
            f"FINAL TEST RESULT on full test set (N={int(final_metrics['samples'])}): "
            f"test_loss={final_metrics['test_loss']:.4f} | "
            f"l1_loss={final_metrics['l1_loss']:.4f}"
        )
