"""Runner for missing-component (TV) generation in a frozen mask VAE latent space.

This wires three FROZEN mask_VAE checkpoints, a doubly-conditioned latent
diffusion model, and a pixel-space objective into one training loop.

Data flow per sample
--------------------
One sample folder holds a class-id ``furniture_segmentation.png`` and a
class-id ``floorplan_segmentation.png``. The furniture mask is encoded twice,
through two layouts built from two different frozen checkpoints:

* the PARTIAL layout (fair-burro) records ``furniture/tv`` as excluded, so
  ``remap_ids_to_background`` erases the TV and the resulting latent describes a
  layout with the TV genuinely missing;
* the TARGET layout (chief-yeti) keeps the TV, and its latent is the diffusion
  target.

The floorplan mask is encoded once (mint-akita) as the spatial condition. Both
furniture encoders see the identical class-id PNG, so the ONLY information
separating the condition from the target is the missing component - which is
exactly the missing-component generation task.

Objective
---------
Training runs two objectives in sequence, never summed (see LOSS_SCHEDULE):

1. **Latent MSE** (``mse``, the default for most epochs). Standard DDPM
   noise-prediction MSE::

       t ~ U[0, NUM_TIMESTEPS);  x_t = q_sample(z_target, t, eps)
       loss = MSE(denoiser(x_t, t, z_floorplan, z_partial), eps)

   The frozen decoder is not in the graph at all, so memory is dominated by the
   128-d latents, the batch size can be an order of magnitude larger, and the
   FULL timestep schedule is trainable - nothing needs truncating.

2. **Pixel BCE** (``bce``, one epoch per cycle). Only the missing component::

       x0_hat = denormalize((x_t - sqrt(1-abar_t)*eps_hat) / sqrt(abar_t))
       logits = chief_yeti.decoder(x0_hat)
       loss   = TV_LOSS_WEIGHT * BCE(logits[:, tv], gt[:, tv])

   The other ten furniture classes get NO pixel-space term at all: the latent MSE
   has no notion of which class matters, and a whole-mask BCE would be dominated
   by the ~97% background that is trivially predicted. The deliverable is the
   missing component, so the pixel objective supervises exactly that channel.

   The decoder is frozen but NOT under ``no_grad``, so the pixel-space gradient
   flows back through it into the latent and into the denoiser. That is also why
   this epoch runs at a small batch size, and why its timesteps stay capped at
   ``BCE_TRAIN_T_MAX`` - a decoded BCE is meaningless at high noise.

Evaluation mirrors the split: the latent MSE is scored every epoch (no decoder,
so it is cheap and the curve is continuous across the switch), while the pixel
pass is throttled to ``PIXEL_EVAL_INTERVAL`` during the MSE phase.

Caching
-------
The three convolutional encoders run exactly once per split. What is cached per
sample is the triple of latents (already standardized) plus the resized uint8
class-id mask. The mask lives in a memory-mapped ``.npy`` so the training loop
never decodes a PNG and never needs the whole array resident. A fingerprint over
every input that changes the encoded values invalidates the cache automatically.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from conditional_diffusion.runner import (  # noqa: E402
    build_frozen_mask_vae,
    resolve_color_index_file,
)
from helpers.config import load_config  # noqa: E402
from helpers.hyperparams import load_hyperparams  # noqa: E402
from helpers.logger import setup_logger  # noqa: E402
from helpers.telemetry_writer import TelemetryWriter  # noqa: E402
from mask_VAE.masking import (  # noqa: E402
    MaskGroup,
    MaskLayout,
    encode_masks_to_channels,
    encode_masks_to_single_channel,
    render_group_rgb,
)
from mask_VAE.metrics import (  # noqa: E402
    ClassOverlap,
    confusion_matrix,
    overlap_from_confusion,
)
from mask_VAE.model import MaskVAEModel  # noqa: E402
from mask_VAE.per_class_telemetry import PerClassTelemetryWriter  # noqa: E402
from mask_diffusion.model import MaskComponentDiffusion  # noqa: E402
from VAE.runner import resolve_device  # noqa: E402

SCRIPT_DIR: Path = Path(__file__).resolve().parent

# Image modes accepted when opening a source mask. Palette images are accepted
# because Pillow reports them as "P", but they are converted to single channel
# before use: the payload is class ids either way, not colors.
ACCEPTED_MASK_IMAGE_MODES: frozenset[str] = frozenset({"L", "P"})
MASK_IMAGE_MODE: str = "L"
# Resampling filter for class-id masks. Never an interpolating filter: it would
# blend neighbouring class ids into values belonging to no class.
MASK_RESAMPLE: Image.Resampling = Image.Resampling.NEAREST

# Filename stems of the cached artifacts. The latent blob is a torch file (small,
# holds the statistics and the fingerprint); the class ids are a raw uint8 array
# so they can be memory-mapped instead of read into RAM.
LATENT_CACHE_STEM_TRAIN: str = "train_triplets"
LATENT_CACHE_STEM_TEST: str = "test_triplets"

# Keys of the standardized-latent bundle stored alongside the fingerprint. Kept
# in one place so the save and restore paths cannot drift apart.
STATISTIC_KEYS: tuple[str, ...] = (
    "latent_mean",
    "latent_std",
    "floorplan_mean",
    "floorplan_std",
    "partial_mean",
    "partial_std",
)

# Latent-space names mapped to the config key holding their checkpoint and the
# hyperparameter holding their expected width. Drives both the checkpoint
# loading and the fail-fast dimension checks.
FROZEN_VAE_KEYS: tuple[tuple[str, str], ...] = (
    ("floorplan", "FLOORPLAN_VAE_CHECKPOINT_PATH"),
    ("partial", "PARTIAL_VAE_CHECKPOINT_PATH"),
    ("target", "TARGET_VAE_CHECKPOINT_PATH"),
)

# Valid values for LOSS_SCHEDULE. "mse_bce_cycle" repeats
# MSE_EPOCHS_PER_CYCLE latent epochs followed by one pixel epoch; "mse_then_bce"
# trains the cheap latent objective once, then switches permanently.
LOSS_SCHEDULES: frozenset[str] = frozenset(
    {"mse_bce_cycle", "mse_then_bce", "mse_only", "bce_only"}
)

# The schedules that must train the pixel objective at least once, and so must
# be checked against it at startup. "mse_only" and "bce_only" are constant by
# definition and need no such check.
SCHEDULES_WITH_PIXEL_EPOCHS: frozenset[str] = frozenset(
    {"mse_bce_cycle", "mse_then_bce"}
)


def set_reproducibility_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs so a run is reproducible.

    Args:
        seed (int): Seed applied to ``random`` and ``torch`` (CPU + CUDA).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_ids_to_one_hot(
    class_ids: torch.Tensor,
    num_channels: int,
) -> torch.Tensor:
    """Turn class-id masks into the one-vs-rest binary target the BCE needs.

    Mirrors :func:`mask_VAE.masking.encode_masks_to_channels` restricted to a
    SINGLE-GROUP layout, which is all this runner ever holds. It is written out
    rather than reused for one concrete reason: that helper allocates its output
    on CPU, so calling it every training step would copy an ``11 x 256 x 256``
    target across the device boundary per batch. Behaviour is identical -
    exactly one active channel per pixel, ordered like the layout's
    ``class_names`` - because the one-hot expansion and the shared-background
    rule coincide when there is a single group and no excluded class, which
    :meth:`MaskDiffusionRunner.verify_one_hot_matches_mask_vae` asserts.

    Args:
        class_ids (torch.Tensor): ``(B, H, W)`` integer class ids on any device.
        num_channels (int): Number of class channels, background included.

    Returns:
        torch.Tensor: Float32 ``(B, num_channels, H, W)`` tensor of zeros and
            ones, on the same device as ``class_ids``.

    Raises:
        ValueError: If an id lies outside ``[0, num_channels)``.
    """
    batch_size, height, width = class_ids.shape
    del batch_size, height, width
    out_of_range: torch.Tensor = (class_ids < 0) | (class_ids >= num_channels)
    if bool(out_of_range.any()):
        raise ValueError(
            f"Class ids must lie in [0, {num_channels}); saw min "
            f"{int(class_ids.min())} and max {int(class_ids.max())}."
        )
    # one_hot over the id axis, then permute so channels come first: the cheapest
    # way to get an exclusive per-pixel class assignment.
    one_hot: torch.Tensor = F.one_hot(
        class_ids.to(torch.long), num_classes=num_channels
    ).permute(0, 3, 1, 2)
    return one_hot.to(torch.float32)


def overlay_component_on_base(
    base_rgb: torch.Tensor,
    component_rgb: torch.Tensor,
    component_mask: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Blend one component's color over a base RGB image where it was predicted.

    Used by the third visualization block: the base is the ground-truth layout
    render and only the missing component's pixels are painted, so a misplaced
    TV shows up against the true furniture instead of being buried under it.

    Args:
        base_rgb (torch.Tensor): ``(B, 3, H, W)`` base image in ``0..1``.
        component_rgb (torch.Tensor): ``(B, 3, H, W)`` component render in
            ``0..1``; only read where ``component_mask`` is set.
        component_mask (torch.Tensor): ``(B, H, W)`` or ``(B, 1, H, W)`` mask,
            non-zero where the component was predicted.
        alpha (float): Blend weight; ``1.0`` is a hard paint, ``0.0`` shows the
            base alone.

    Returns:
        torch.Tensor: ``(B, 3, H, W)`` blended image in ``0..1``.
    """
    if component_mask.ndim == 3:
        component_mask = component_mask.unsqueeze(1)
    weight: torch.Tensor = component_mask.to(base_rgb.dtype) * alpha
    return base_rgb * (1.0 - weight) + component_rgb * weight


def _fmt(value: float | None) -> str:
    """Format an optional metric for the per-epoch log line.

    Telemetry distinguishes "not measured this epoch" from zero by writing an
    empty cell, and the log line should do the same rather than printing ``0.00``.

    Args:
        value (float | None): The metric, or None when it does not apply.

    Returns:
        str: ``"n/a"`` for None, otherwise the value at four decimals.
    """
    return "n/a" if value is None else f"{value:.4f}"


class MissingComponentDataset(Dataset[dict[str, torch.Tensor]]):
    """Cached (target latent, floorplan latent, TV-free latent, GT mask) samples.

    The three latents come from the frozen encoders and are already
    standardized; the ground truth is kept as uint8 class ids rather than as an
    11-channel float tensor because uint8 is 4x smaller and the one-hot
    expansion is a couple of cheap comparisons on the training device. The id
    array is memory-mapped, so the (roughly 4.6 GB at full scale) mask cache
    never has to be resident.

    Args:
        latents (dict[str, torch.Tensor]): Standardized ``target``, ``floorplan``
            and ``partial`` latent tensors, all ``(N, latent_dim)``, in a stable
            order shared with ``masks``.
        masks (np.ndarray): ``(N, S, S)`` uint8 class-id array, memory-mapped.
        floorplan_ids (np.ndarray): ``(N, S, S)`` uint8 floorplan class-id array.
            Only the visualization needs this, and the floorplan masks are tiny
            next to the 18-channel one-hot expansion, so they are cached with
            the same fingerprint as the rest.
        sample_dirs (list[Path]): Source folder of each sample, in the same
            order. Kept so a grid can locate the sample's rendered images.
        image_size (int): Side length the class ids were resized to.
    """

    def __init__(
        self,
        latents: dict[str, torch.Tensor],
        masks: np.ndarray,
        floorplan_ids: np.ndarray,
        sample_dirs: list[Path],
        image_size: int,
    ) -> None:
        """Index the cached artifacts of one split.

        Args:
            latents (dict[str, torch.Tensor]): Standardized latents keyed
                ``"target"``, ``"floorplan"`` and ``"partial"``.
            masks (np.ndarray): ``(N, S, S)`` uint8 class-id array.
            image_size (int): Side length the class ids were resized to.

        Raises:
            ValueError: If the id arrays are not ``(N, S, S)`` uint8, or the
                latent tensors disagree on their sample count.
        """
        counts: set[int] = {tensor.shape[0] for tensor in latents.values()}
        if len(counts) != 1:
            raise ValueError(
                f"Latent tensors have inconsistent lengths {sorted(counts)}; they "
                "must all cover the same samples in the same order."
            )
        num_samples: int = counts.pop()
        expected_shape: tuple[int, int, int] = (num_samples, image_size, image_size)
        for name, array in (("furniture", masks), ("floorplan", floorplan_ids)):
            if array.shape != expected_shape or array.dtype != np.uint8:
                raise ValueError(
                    f"Cached {name} class-id array must be {expected_shape} uint8, got "
                    f"{array.shape} {array.dtype}."
                )
        if len(sample_dirs) != num_samples:
            raise ValueError(
                f"Got {len(sample_dirs)} sample folders for {num_samples} cached samples."
            )
        self.latents: dict[str, torch.Tensor] = latents
        self.masks: np.ndarray = masks
        self.floorplan_ids: np.ndarray = floorplan_ids
        self.sample_dirs: list[Path] = list(sample_dirs)

    def __len__(self) -> int:
        """Return the number of cached samples.

        Returns:
            int: Length of the cached sample list.
        """
        return int(self.masks.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one sample's three latents and its two class-id masks.

        Args:
            index (int): Position in the cached sample list.

        Returns:
            dict[str, torch.Tensor]: ``{"target", "floorplan", "partial",
                "class_ids", "floorplan_class_ids"}``; the latents are 1-D float32
                and the two id tensors are int64 ``(S, S)``.

        Raises:
            IndexError: If ``index`` is outside the cached range.
        """
        num_samples: int = len(self)
        if index < 0 or index >= num_samples:
            raise IndexError(f"Sample index {index} outside [0, {num_samples}).")
        return {
            "target": self.latents["target"][index],
            "floorplan": self.latents["floorplan"][index],
            "partial": self.latents["partial"][index],
            "class_ids": torch.from_numpy(self.masks[index].astype(np.int64)),
            "floorplan_class_ids": torch.from_numpy(
                self.floorplan_ids[index].astype(np.int64)
            ),
        }


class MaskDiffusionRunner:
    """Trainer wiring three frozen mask VAEs, diffusion and the eval loop."""

    def __init__(self, mode: str | None = None) -> None:
        """Load config/hyperparams, the three frozen VAEs, model and telemetry.

        Args:
            mode (str | None): Run mode selecting the ``[debug]`` or ``[prod]``
                table of ``config.toml``. Parsed from ``--mode`` when None.

        Raises:
            ValueError: If a latent dimension or image size disagrees with the
                hyperparameters, or the missing component cannot be resolved.
        """
        self.config_file: str = str(SCRIPT_DIR / "config.toml")
        self.hyperparams_file: str = str(SCRIPT_DIR / "hyperparameters.toml")

        if mode is None:
            parser = argparse.ArgumentParser(
                description="Train missing-component diffusion on frozen mask VAE latents."
            )
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

        self.telemetry_fieldnames: list[str] = [
            "epoch",
            "train_loss",
            "train_mse",
            "train_tv_bce",
            "test_mse",
            "test_tv_bce",
            "test_tv_dice",
            "sample_tv_dice",
            "sample_tv_iou",
        ]

        self.device: torch.device = resolve_device(self.config.DEVICE, self.log)
        started: float = float(self.config.runtime.START_TIME)
        self.log.info(
            f"Starting mask diffusion with run_id: <red>{self.config.runtime.RUN_ID}"
            f"</red> started at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started))} on "
            f"device: {self.device}"
        )

        self.frozen_vaes: dict[str, MaskVAEModel]
        self.frozen_arch: dict[str, SimpleNamespace]
        self.floorplan_layout: MaskLayout
        self.partial_layout: MaskLayout
        self.target_layout: MaskLayout
        self.target_group: MaskGroup
        self._load_frozen_vaes()
        self.missing_channel: int = self._resolve_missing_channel()
        # Objective schedule. The two objectives are never summed: each epoch is
        # either a cheap latent-MSE epoch or an expensive pixel-BCE epoch. These
        # read only the hyperparameters, so they are set before validation.
        self.loss_schedule: str = str(self.hyperparams.LOSS_SCHEDULE)
        self.mse_phase_epochs: int = int(self.hyperparams.MSE_PHASE_EPOCHS)
        self.mse_epochs_per_cycle: int = int(self.hyperparams.MSE_EPOCHS_PER_CYCLE)
        self.mse_batch_size: int = int(self.hyperparams.MSE_BATCH_SIZE)
        self.bce_batch_size: int = int(self.hyperparams.BCE_BATCH_SIZE)
        # Timestep cap for the pixel phase only; the MSE phase trains the full
        # schedule because a latent objective is meaningful at every noise level.
        self.bce_train_t_max: int = int(self.hyperparams.BCE_TRAIN_T_MAX)
        self._validate_dimensions()

        self.model: MaskComponentDiffusion = MaskComponentDiffusion(self.hyperparams).to(self.device)
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.hyperparams.LEARNING_RATE)
        )
        self.grad_clip: float = float(self.hyperparams.GRAD_CLIP)
        # Logged after the model exists: the summary reports the schedule it was
        # actually built with.
        self._log_setup()

        # Standardization statistics, one pair per latent space. Filled by the
        # stats pass, or restored from a matching cache.
        self.latent_mean: torch.Tensor = torch.zeros(int(self.hyperparams.LATENT_DIM))
        self.latent_std: torch.Tensor = torch.ones(int(self.hyperparams.LATENT_DIM))
        self.floorplan_mean: torch.Tensor = torch.zeros(int(self.hyperparams.COND_DIM))
        self.floorplan_std: torch.Tensor = torch.ones(int(self.hyperparams.COND_DIM))
        self.partial_mean: torch.Tensor = torch.zeros(int(self.hyperparams.COND_DIM))
        self.partial_std: torch.Tensor = torch.ones(int(self.hyperparams.COND_DIM))

        self.telemetry: TelemetryWriter = TelemetryWriter(
            self.telemetry_fieldnames,
            directory=str(self.config.TELEMETRY_LOG_DIRECTORY),
            config=self.config,
        )
        self.per_class_telemetry: PerClassTelemetryWriter = PerClassTelemetryWriter(
            class_names=list(self.target_group.local_labels),
            directory=str(self.config.PER_CLASS_TELEMETRY_LOG_DIRECTORY),
            config=self.config,
        )
        self.last_sample_tv_dice: float | None = None
        self.last_sample_tv_iou: float | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _load_frozen_vaes(self) -> None:
        """Build the three frozen mask VAEs and their single-group layouts.

        Each is rebuilt from its OWN checkpoint's embedded hyperparameters, so
        these stay loadable after the mask_VAE config has moved on. The partial
        layout inherits the TV exclusion recorded inside the fair-burro
        checkpoint, which is what makes its encoder input TV-free.

        Raises:
            ValueError: If a checkpoint path does not exist on disk.
        """
        color_dir: Path = Path(str(self.config.MASK_COLOR_INDEX_DIR))
        fallback_dir: Path = Path(str(self.config.TRAIN_IMAGES_DIR)).parent
        floorplan_colors: Path = resolve_color_index_file(
            str(self.config.FLOORPLAN_COLOR_INDEX_FILE), color_dir, fallback_dir
        )
        furniture_colors: Path = resolve_color_index_file(
            str(self.config.FURNITURE_COLOR_INDEX_FILE), color_dir, fallback_dir
        )
        mask_files: dict[str, str] = {
            "floorplan": str(self.config.FLOORPLAN_MASK_FILE),
            "partial": str(self.config.PARTIAL_MASK_FILE),
            "target": str(self.config.TARGET_MASK_FILE),
        }
        color_files: dict[str, Path] = {
            "floorplan": floorplan_colors,
            "partial": furniture_colors,
            "target": furniture_colors,
        }
        self.frozen_vaes = {}
        self.frozen_arch = {}
        layouts: dict[str, MaskLayout] = {}
        for key, config_key in FROZEN_VAE_KEYS:
            checkpoint_path: str = str(getattr(self.config, config_key))
            if not os.path.isfile(checkpoint_path):
                raise ValueError(
                    f"Frozen {key} mask VAE checkpoint not found: {checkpoint_path}. "
                    f"Run ids in a config path must match the file on disk exactly."
                )
            vae, arch, layout = build_frozen_mask_vae(
                checkpoint_path, color_files[key], mask_files[key], self.device
            )
            self.frozen_vaes[key] = vae
            self.frozen_arch[key] = arch
            layouts[key] = layout
        self.floorplan_layout = layouts["floorplan"]
        self.partial_layout = layouts["partial"]
        self.target_layout = layouts["target"]
        self.target_group = self.target_layout.groups[0]

    def _resolve_missing_channel(self) -> int:
        """Look the missing component's channel up by name, never by number.

        Returns:
            int: Index of the missing component's channel in the target
                decoder's output.

        Raises:
            ValueError: If the name is absent from the target layout's classes,
                or the partial layout excludes nothing.
        """
        plain_name: str = str(self.config.MISSING_COMPONENT_CLASS_NAME)
        prefix: str = str(self.config.TARGET_CLASS_GROUP_PREFIX)
        qualified: str = f"{prefix}/{plain_name}" if prefix else plain_name
        if qualified not in self.target_layout.class_names:
            raise ValueError(
                f"Missing component '{qualified}' is not one of the target layout's "
                f"classes {list(self.target_layout.class_names)}. Check "
                "MISSING_COMPONENT_CLASS_NAME and TARGET_CLASS_GROUP_PREFIX in "
                "mask_diffusion/config.toml."
            )
        channel: int = self.target_layout.class_names.index(qualified)
        # A partial layout that does NOT actually exclude the missing component
        # would silently reduce this to unconditional generation, so it is
        # rejected here rather than discovered from confusing visualizations.
        if not self.partial_layout.excluded_channels:
            raise ValueError(
                f"The partial VAE at {self.config.PARTIAL_VAE_CHECKPOINT_PATH} excludes "
                "no class, so its layout would still contain the missing component. "
                "Point PARTIAL_VAE_CHECKPOINT_PATH at a filtered checkpoint."
            )
        return channel

    def _validate_dimensions(self) -> None:
        """Fail fast on any latent-dimension or image-size disagreement.

        Raises:
            ValueError: If any frozen VAE's latent width or image size differs
                from what the hyperparameters declare, or if the objective
                schedule is malformed.
        """
        image_size: int = int(self.hyperparams.IMAGE_SIZE)
        expected_widths: dict[str, int] = {
            "target": int(self.hyperparams.LATENT_DIM),
            "floorplan": int(self.hyperparams.COND_DIM),
            "partial": int(self.hyperparams.COND_DIM),
        }
        for key, config_key in FROZEN_VAE_KEYS:
            expected: int = expected_widths[key]
            actual: int = int(self.frozen_arch[key].LATENT_DIM)
            if actual != expected:
                raise ValueError(
                    f"The {key} VAE at {getattr(self.config, config_key)} has "
                    f"LATENT_DIM {actual} but the hyperparameters declare {expected}."
                )
            actual_size: int = int(self.frozen_arch[key].IMAGE_SIZE)
            if actual_size != image_size:
                raise ValueError(
                    f"The {key} VAE was trained at IMAGE_SIZE {actual_size} but the "
                    f"hyperparameters declare {image_size}."
                )
        if self.loss_schedule not in LOSS_SCHEDULES:
            raise ValueError(
                f"LOSS_SCHEDULE must be one of {sorted(LOSS_SCHEDULES)}, got "
                f"{self.loss_schedule!r}."
            )
        if self.mse_phase_epochs < 0:
            raise ValueError(
                f"MSE_PHASE_EPOCHS must be >= 0, got {self.mse_phase_epochs}."
            )
        # BCE_TRAIN_T_MAX only bounds the pixel phase, so it is checked against
        # the full schedule rather than against the MSE phase's range.
        if not 0 < self.bce_train_t_max <= int(self.hyperparams.NUM_TIMESTEPS):
            raise ValueError(
                f"BCE_TRAIN_T_MAX must lie in (0, NUM_TIMESTEPS], got "
                f"{self.bce_train_t_max}."
            )
        for key in (
            "MSE_BATCH_SIZE",
            "BCE_BATCH_SIZE",
            "EVAL_BATCH_SIZE",
            "PIXEL_EVAL_BATCH_SIZE",
            "MSE_EPOCHS_PER_CYCLE",
        ):
            if int(getattr(self.hyperparams, key)) < 1:
                raise ValueError(f"{key} must be >= 1, got {getattr(self.hyperparams, key)}.")
        # Silently running a schedule with no pixel epoch would be a confusing way
        # to discover a typo, so it is rejected at startup. This only applies to
        # the schedules that are supposed to include the pixel objective;
        # "mse_only" is by definition all latent MSE and "bce_only" is all pixel.
        num_epochs: int = int(self.hyperparams.NUM_EPOCHS)
        if self.loss_schedule in SCHEDULES_WITH_PIXEL_EPOCHS and (
            self.objective_for_epoch(num_epochs) != "bce"
        ):
            raise ValueError(
                f"LOSS_SCHEDULE '{self.loss_schedule}' would never train the pixel "
                f"objective: the last epoch is {num_epochs}, which resolves to "
                f"'mse'. Raise NUM_EPOCHS, lower MSE_PHASE_EPOCHS or "
                f"MSE_EPOCHS_PER_CYCLE, or use 'mse_only'."
            )

    def objective_for_epoch(self, epoch: int) -> str:
        """Return which objective an epoch trains with.

        For ``mse_bce_cycle`` the pattern repeats every
        ``MSE_EPOCHS_PER_CYCLE + 1`` epochs: the first ``N`` are latent MSE and
        the last is pixel BCE. With ``N = 5`` that is MSE at epochs 1-5, BCE at 6,
        MSE at 7-11, BCE at 12, and so on. Only the final BCE epoch of a run may
        be truncated mid-cycle, which is harmless.

        Args:
            epoch (int): 1-based epoch number.

        Returns:
            str: ``"mse"`` for the latent-space objective, ``"bce"`` for the
                pixel-space objective.
        """
        if self.loss_schedule == "mse_only":
            return "mse"
        if self.loss_schedule == "bce_only":
            return "bce"
        if self.loss_schedule == "mse_bce_cycle":
            cycle_length: int = self.mse_epochs_per_cycle + 1
            return "bce" if epoch % cycle_length == 0 else "mse"
        return "mse" if epoch <= self.mse_phase_epochs else "bce"

    def bce_epochs_planned(self, num_epochs: int) -> int:
        """Count how many of the given epochs train the pixel objective.

        Args:
            num_epochs (int): Total number of epochs the run will execute.

        Returns:
            int: Number of pixel-space epochs in that run.
        """
        return sum(
            1
            for epoch in range(1, num_epochs + 1)
            if self.objective_for_epoch(epoch) == "bce"
        )

    def batch_size_for_objective(self, objective: str) -> int:
        """Return the batch size an objective should run at.

        The two phases differ by an order of magnitude because only the BCE
        phase holds the frozen decoder's activations.

        Args:
            objective (str): ``"mse"`` or ``"bce"``.

        Returns:
            int: Batch size for that objective.

        Raises:
            ValueError: If ``objective`` is not a known objective.
        """
        if objective == "mse":
            return self.mse_batch_size
        if objective == "bce":
            return self.bce_batch_size
        raise ValueError(f"Unknown objective {objective!r}; expected 'mse' or 'bce'.")

    def _log_setup(self) -> None:
        """Log the resolved setup, including the alpha_bar training stops at.

        The ``BCE_TRAIN_T_MAX`` summary matters when reading telemetry later:
        past that step the schedule's signal-to-noise ratio is too low for the
        decoded BCE to mean anything, so a plateau there is expected rather than
        a bug. The MSE phase trains the full schedule and is unaffected.
        """
        missing_name: str = self.target_layout.class_names[self.missing_channel]
        # excluded_class_names is a MaskGroup property; the layout only exposes
        # the channel indices.
        excluded: list[str] = list(self.partial_layout.groups[0].excluded_class_names)
        self.log.info(
            f"Missing component: '{missing_name}' -> channel {self.missing_channel} of "
            f"{self.target_layout.num_channels}. The partial layout excludes {excluded}."
        )
        alpha_bar: float = float(self.model.alpha_bar[self.bce_train_t_max - 1])
        amplification: float = 1.0 / max(alpha_bar**0.5, 1e-12)
        num_epochs: int = int(self.hyperparams.NUM_EPOCHS)
        bce_epochs: int = self.bce_epochs_planned(num_epochs)
        if self.loss_schedule == "mse_bce_cycle":
            schedule_note: str = (
                f"repeating cycle of {self.mse_epochs_per_cycle} latent-MSE epochs then "
                f"1 pixel-BCE epoch; {bce_epochs} of {num_epochs} epochs train the "
                f"pixel objective"
            )
        elif self.loss_schedule == "mse_then_bce":
            schedule_note: str = (
                f"epochs 1-{self.mse_phase_epochs} use latent MSE, the remainder use "
                f"pixel BCE; {bce_epochs} of {num_epochs} epochs train the pixel objective"
            )
        else:
            schedule_note: str = f"every epoch uses {self.objective_for_epoch(1)}"
        self.log.info(
            f"Objective schedule '{self.loss_schedule}': {schedule_note}. "
            f"Batch size {self.mse_batch_size} (mse) / {self.bce_batch_size} (bce)."
        )
        self.log.info(
            f"MSE phase trains timesteps [0, {self.model.num_timesteps}); the BCE phase "
            f"is capped at [0, {self.bce_train_t_max}) where alpha_bar is "
            f"{alpha_bar:.4f}, so the x0 estimate is amplified {amplification:.2f}x. "
            f"Sampling walks all {self.model.num_timesteps} steps."
        )
        self.log.info(
            f"BCE epoch loss = {float(self.hyperparams.TV_LOSS_WEIGHT)} * "
            f"BCE(decoded_tv, gt_tv) on the missing component's channel ONLY, "
            f"pos_weight={float(self.hyperparams.TV_BCE_POS_WEIGHT)}. The other "
            "channels get no pixel-space term; the latent MSE is the only thing "
            "supervising them."
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _standardize(
        latents: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        enabled: bool,
    ) -> torch.Tensor:
        """Apply per-dimension standardization when enabled.

        Args:
            latents (torch.Tensor): ``(N, D)`` raw latents on CPU.
            mean (torch.Tensor): ``(D,)`` per-dimension mean.
            std (torch.Tensor): ``(D,)`` per-dimension std.
            enabled (bool): The corresponding ``NORMALIZE_*`` switch.

        Returns:
            torch.Tensor: Standardized latents, or the untouched input.
        """
        if not enabled:
            return latents
        return (latents - mean) / std

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Standardize target (diffusion space) latents.

        Args:
            latents (torch.Tensor): ``(n, LATENT_DIM)`` latents.

        Returns:
            torch.Tensor: Standardized latents, or the input when disabled.
        """
        return self._standardize(
            latents,
            self.latent_mean,
            self.latent_std,
            bool(self.hyperparams.NORMALIZE_LATENTS),
        )

    def normalize_floorplan(self, latents: torch.Tensor) -> torch.Tensor:
        """Standardize floorplan condition latents.

        Args:
            latents (torch.Tensor): ``(n, COND_DIM)`` latents.

        Returns:
            torch.Tensor: Standardized latents, or the input when disabled.
        """
        return self._standardize(
            latents,
            self.floorplan_mean,
            self.floorplan_std,
            bool(self.hyperparams.NORMALIZE_FLOORPLAN),
        )

    def normalize_partial(self, latents: torch.Tensor) -> torch.Tensor:
        """Standardize TV-free layout condition latents.

        Args:
            latents (torch.Tensor): ``(n, COND_DIM)`` latents.

        Returns:
            torch.Tensor: Standardized latents, or the input when disabled.
        """
        return self._standardize(
            latents,
            self.partial_mean,
            self.partial_std,
            bool(self.hyperparams.NORMALIZE_PARTIAL),
        )

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Invert target standardization before the frozen decoder is applied.

        Args:
            latents (torch.Tensor): Standardized ``(n, LATENT_DIM)`` latents on
                any device.

        Returns:
            torch.Tensor: Raw-scale latents the target decoder expects.
        """
        if not bool(self.hyperparams.NORMALIZE_LATENTS):
            return latents
        return (
            latents * self.latent_std.to(latents.device) + self.latent_mean.to(latents.device)
        )

    def compute_latent_stats(
        self, latents: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Estimate per-dimension mean/std for each of the three latent spaces.

        The three frozen VAEs have independent latent geometries, so sharing one
        set of statistics would mis-scale two of them. Each std is floored at
        ``LATENT_STD_MIN`` so a dead latent dimension cannot divide by ~0.

        Args:
            latents (dict[str, torch.Tensor]): Raw ``target``/``floorplan``/
                ``partial`` latents, all ``(N, D)``.

        Returns:
            dict[str, torch.Tensor]: One ``mean``/``std`` pair per space, keyed
                ``latent_*``, ``floorplan_*`` and ``partial_*``.
        """
        limit: int = int(self.hyperparams.LATENT_STATS_SAMPLES)
        floor: float = float(self.hyperparams.LATENT_STD_MIN)
        stats: dict[str, torch.Tensor] = {}
        for key in ("target", "floorplan", "partial"):
            sample: torch.Tensor = latents[key][:limit]
            mean: torch.Tensor = sample.mean(dim=0)
            std: torch.Tensor = sample.std(dim=0, unbiased=False).clamp_min(floor)
            prefix: str = "latent" if key == "target" else key
            stats[f"{prefix}_mean"] = mean
            stats[f"{prefix}_std"] = std
            self.log.info(
                f"{key} latent stats over {sample.shape[0]} samples: "
                f"mean|mu|={float(mean.abs().mean()):.4f} "
                f"mean|std|={float(std.mean()):.4f} min|std|={float(std.min()):.4f}"
            )
        return stats

    def _apply_latent_stats(self, train_latents: dict[str, torch.Tensor]) -> None:
        """Store the standardization statistics estimated on the train split.

        Args:
            train_latents (dict[str, torch.Tensor]): Raw train-split latents.
        """
        for key, value in self.compute_latent_stats(train_latents).items():
            setattr(self, key, value)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def list_sample_dirs(self, split_dir: str) -> list[Path]:
        """List the sample folders of one split that provide both masks.

        Args:
            split_dir (str): Split folder holding one sub-folder per sample.

        Returns:
            list[Path]: Sorted sample folders.

        Raises:
            FileNotFoundError: If the split is missing, or no sample provides
                both segmentation masks.
        """
        floor_mask: str = str(self.config.FLOORPLAN_MASK_FILE)
        target_mask: str = str(self.config.TARGET_MASK_FILE)
        split_path: Path = Path(split_dir)
        if not split_path.is_dir():
            raise FileNotFoundError(f"Dataset split directory missing: {split_dir}.")
        # is_dir() also drops the stray .DS_Store that sits in these splits.
        sample_dirs: list[Path] = sorted(
            candidate
            for candidate in split_path.iterdir()
            if candidate.is_dir()
            and (candidate / floor_mask).is_file()
            and (candidate / target_mask).is_file()
        )
        if not sample_dirs:
            raise FileNotFoundError(
                f"No sample folder under {split_dir} provides both '{floor_mask}' "
                f"and '{target_mask}'."
            )
        return sample_dirs

    def _load_class_ids(self, mask_path: Path) -> torch.Tensor:
        """Read one mask PNG as an int64 class-id tensor.

        Args:
            mask_path (Path): Path of the ``*_segmentation.png`` file.

        Returns:
            torch.Tensor: int64 ``(S, S)`` class-id tensor, resized with
                nearest-neighbour interpolation.

        Raises:
            ValueError: If the image is not a single-channel class-id mask.
        """
        image_size: int = int(self.hyperparams.IMAGE_SIZE)
        with Image.open(mask_path) as image:
            if image.mode not in ACCEPTED_MASK_IMAGE_MODES:
                raise ValueError(
                    f"{mask_path} has image mode {image.mode!r}, but a "
                    f"{MASK_IMAGE_MODE!r} integer class-id mask is required."
                )
            resized: Image.Image = image.convert(MASK_IMAGE_MODE).resize(
                (image_size, image_size), resample=MASK_RESAMPLE
            )
            array: np.ndarray = np.array(resized, dtype=np.uint8)
        return torch.from_numpy(array.astype(np.int64))

    def _encode_latent(self, vae: MaskVAEModel, mask_input: torch.Tensor) -> torch.Tensor:
        """Encode a masked batch to a latent, optionally taking the posterior mean.

        ``USE_MU_ONLY = true`` picks ``mu``, which is deterministic and makes a
        run reproducible; ``false`` draws a reparameterized sample, which
        augments the data at the cost of run-to-run variation.

        Args:
            vae (MaskVAEModel): Frozen mask VAE.
            mask_input (torch.Tensor): ``(B, enc_channels, S, S)`` encoder input.

        Returns:
            torch.Tensor: ``(B, LATENT_DIM)`` latents on the VAE's device.
        """
        with torch.no_grad():
            if bool(self.hyperparams.USE_MU_ONLY):
                return vae.encoder(mask_input)[0]
            mu, logvar = vae.encoder(mask_input)
            return vae.reparameterize(mu, logvar)

    def _encode_batch(self, sample_dirs: list[Path]) -> dict[str, torch.Tensor]:
        """Encode one batch of samples with all three frozen encoders.

        The same furniture class-id tensor is pushed through the partial and the
        target layout; only the layout differs, so the TV is removed by the
        partial layout's exclusion rule rather than by editing pixels here.

        Args:
            sample_dirs (list[Path]): Sample folders of this batch.

        Returns:
            dict[str, torch.Tensor]: ``target``, ``partial`` and ``floorplan``
                latents of shape ``(B, 128)``, plus ``class_ids`` ``(B, S, S)``
                (furniture) and ``floorplan_class_ids`` ``(B, S, S)`` int64. The
                furniture ids drive the loss and both id arrays are cached so the
                visualization can render either without touching the disk.
        """
        floor_ids: list[torch.Tensor] = []
        furniture_ids: list[torch.Tensor] = []
        for sample_dir in sample_dirs:
            floor_ids.append(
                self._load_class_ids(sample_dir / str(self.config.FLOORPLAN_MASK_FILE))
            )
            furniture_ids.append(
                self._load_class_ids(sample_dir / str(self.config.PARTIAL_MASK_FILE))
            )
        floor_batch: torch.Tensor = torch.stack(floor_ids).to(self.device)
        furniture_batch: torch.Tensor = torch.stack(furniture_ids).to(self.device)

        def encode_with(vae: MaskVAEModel, layout: MaskLayout, ids: torch.Tensor) -> torch.Tensor:
            """Encode class ids through one layout with one frozen encoder.

            Args:
                vae (MaskVAEModel): Frozen mask VAE.
                layout (MaskLayout): Layout defining exclusions and class count.
                ids (torch.Tensor): ``(B, S, S)`` class ids.

            Returns:
                torch.Tensor: ``(B, LATENT_DIM)`` latents.
            """
            return self._encode_latent(vae, encode_masks_to_single_channel([ids], layout))

        return {
            "target": encode_with(self.frozen_vaes["target"], self.target_layout, furniture_batch),
            "partial": encode_with(self.frozen_vaes["partial"], self.partial_layout, furniture_batch),
            "floorplan": encode_with(self.frozen_vaes["floorplan"], self.floorplan_layout, floor_batch),
            "class_ids": furniture_batch,
            "floorplan_class_ids": floor_batch,
        }

    def encode_split(
        self, sample_dirs: list[Path], split_name: str
    ) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, list[Path]]:
        """Encode a whole split once into raw latents plus class-id masks.

        Args:
            sample_dirs (list[Path]): Sample folders of the split.
            split_name (str): Human-readable name used in progress logs.

        Returns:
            tuple: Raw (unstandardized) ``target``/``floorplan``/``partial``
                latents on CPU, the ``(N, S, S)`` uint8 FURNITURE class-id array,
                and the matching floorplan class-id array, plus the folders that
                actually survived. Samples without the missing component are
                dropped when configured to do so, and the filter applies to both
                id arrays so they stay aligned.
        """
        batch_size: int = int(self.hyperparams.IMAGE_ENCODE_BATCH_SIZE)
        max_samples: int = int(self.hyperparams.MAX_SAMPLES)
        require_component: bool = bool(self.hyperparams.REQUIRE_MISSING_COMPONENT_SAMPLES)
        log_every: int = max(1, int(self.hyperparams.ENCODE_LOG_INTERVAL))
        chunks: dict[str, list[torch.Tensor]] = {
            "target": [],
            "floorplan": [],
            "partial": [],
        }
        mask_chunks: list[np.ndarray] = []
        floorplan_chunks: list[np.ndarray] = []
        # The folders that survive filtering, in cache order. Tracked because the
        # drop filter and MAX_SAMPLES mean the kept rows are NOT simply a prefix
        # of the input list, so a grid cannot reconstruct this afterwards.
        kept_dirs: list[Path] = []
        kept: int = 0
        skipped: int = 0
        start: float = time.time()

        for batch_index, offset in enumerate(range(0, len(sample_dirs), batch_size)):
            batch_dirs: list[Path] = sample_dirs[offset:offset + batch_size]
            if max_samples >= 0 and kept + len(batch_dirs) > max_samples:
                batch_dirs = batch_dirs[:max(0, max_samples - kept)]
            if not batch_dirs:
                break
            encoded: dict[str, torch.Tensor] = self._encode_batch(batch_dirs)
            class_ids: torch.Tensor = encoded["class_ids"]
            floor_ids: torch.Tensor = encoded["floorplan_class_ids"]
            # Indices kept in THIS batch, so kept_dirs stays row-aligned with the
            # id arrays appended below.
            keep_indices: list[int] = list(range(len(batch_dirs)))
            if require_component:
                present: torch.Tensor = (
                    (class_ids == self.missing_channel).flatten(1).any(dim=1)
                )
                if not bool(present.all()):
                    skipped += int((~present).sum())
                    # Both id arrays must be filtered together or the cached
                    # floorplan masks would drift out of alignment.
                    class_ids = class_ids[present]
                    floor_ids = floor_ids[present]
                    encoded = {key: value[present] for key, value in encoded.items()}
                    keep_indices = [i for i in keep_indices if bool(present[i])]
            for key, value in encoded.items():
                if key in chunks:
                    chunks[key].append(value.detach().cpu())
            mask_chunks.append(class_ids.to(torch.uint8).cpu().numpy())
            floorplan_chunks.append(floor_ids.to(torch.uint8).cpu().numpy())
            # Aligns with the mask rows just appended: keep_indices decided both.
            kept_dirs.extend(batch_dirs[i] for i in keep_indices)
            kept += int(class_ids.shape[0])
            if batch_index % log_every == 0:
                rate: float = (offset + len(batch_dirs)) / max(time.time() - start, 1e-6)
                self.log.info(
                    f"Encoded {split_name}: {offset + len(batch_dirs)}/{len(sample_dirs)} "
                    f"scanned, {kept} kept, {skipped} dropped (no missing component), "
                    f"{rate:.1f} samples/s"
                )

        latents: dict[str, torch.Tensor] = {
            key: torch.cat(chunks[key], dim=0) for key in ("target", "floorplan", "partial")
        }
        masks: np.ndarray = np.concatenate(mask_chunks, axis=0)
        floorplan_ids: np.ndarray = np.concatenate(floorplan_chunks, axis=0)
        self.log.info(
            f"Encoded {split_name}: {kept} samples kept ({skipped} dropped), latents "
            f"{tuple(latents['target'].shape)}, furniture ids {masks.shape}, "
            f"floorplan ids {floorplan_ids.shape}, {time.time() - start:.1f}s"
        )
        return latents, masks, floorplan_ids, kept_dirs

    def latent_cache_fingerprint(self, counts: dict[str, int]) -> dict[str, object]:
        """Build the fingerprint identifying the cached split artifacts.

        Every input that changes the encoded values is covered: all three frozen
        checkpoint files (path + mtime + size), both split dirs, the mask size,
        the mu-vs-sample switch, the sample caps, the missing-component filter and
        the resulting per-split counts.

        Args:
            counts (dict[str, int]): Listed sample counts per split.

        Returns:
            dict[str, object]: JSON-comparable fingerprint of the encode inputs.
        """
        entries: dict[str, object] = {
            "train_dir": str(self.config.TRAIN_IMAGES_DIR),
            "test_dir": str(self.config.TEST_IMAGES_DIR),
            "image_size": int(self.hyperparams.IMAGE_SIZE),
            "use_mu_only": bool(self.hyperparams.USE_MU_ONLY),
            "max_samples": int(self.hyperparams.MAX_SAMPLES),
            "require_component": bool(self.hyperparams.REQUIRE_MISSING_COMPONENT_SAMPLES),
            "missing_component": str(self.config.MISSING_COMPONENT_CLASS_NAME),
            "stats_samples": int(self.hyperparams.LATENT_STATS_SAMPLES),
        }
        for key, config_key in FROZEN_VAE_KEYS:
            path: str = str(getattr(self.config, config_key))
            stat: os.stat_result = os.stat(path)
            entries[f"{key}_ckpt_path"] = path
            entries[f"{key}_ckpt_mtime"] = stat.st_mtime
            entries[f"{key}_ckpt_size"] = stat.st_size
        for split, count in counts.items():
            entries[f"{split}_samples"] = count
        return entries

    def _cache_paths(self, split: str) -> tuple[Path, Path, Path]:
        """Return the three cache file paths of one split.

        Args:
            split (str): ``"train"`` or ``"test"``.

        Returns:
            tuple[Path, Path, Path]: Latent ``.pt``, furniture class-id ``.npy``
                and floorplan class-id ``.npy`` paths.
        """
        cache_dir: Path = Path(str(self.config.LATENT_CACHE_DIR))
        stem: str = LATENT_CACHE_STEM_TRAIN if split == "train" else LATENT_CACHE_STEM_TEST
        return (
            cache_dir / f"{stem}.pt",
            cache_dir / f"{stem}_masks.npy",
            cache_dir / f"{stem}_floorplan.npy",
        )

    def try_load_latent_cache(
        self, counts: dict[str, int]
    ) -> dict[str, dict[str, Any]] | None:
        """Load both split caches when the fingerprint matches this run.

        A match also restores the standardization statistics, so the stats pass
        is skipped. Any missing, unreadable or mismatched cache returns None so
        the caller re-encodes from scratch.

        Args:
            counts (dict[str, int]): Listed sample counts per split.

        Returns:
            dict or None: ``{"train": {...}, "test": {...}}`` where each entry
                holds standardized ``target``/``floorplan``/``partial`` latents
                the memory-mapped ``"masks"`` array and the memory-mapped
                ``"floorplan_ids"`` array, else None.
        """
        paths: dict[str, tuple[Path, Path, Path]] = {
            split: self._cache_paths(split) for split in ("train", "test")
        }
        for latent_path, mask_path, floorplan_path in paths.values():
            if not latent_path.is_file() or not mask_path.is_file():
                return None
            if not floorplan_path.is_file():
                self.log.info(
                    f"Latent cache predates the floorplan ids ({floorplan_path}); "
                    "re-encoding."
                )
                return None
        try:
            expected: dict[str, object] = self.latent_cache_fingerprint(counts)
            restored: dict[str, dict[str, Any]] = {}
            statistics: dict[str, torch.Tensor] = {}
            for split, (latent_path, mask_path, floorplan_path) in paths.items():
                blob: dict[str, Any] = torch.load(str(latent_path), map_location="cpu")
                if blob.get("fingerprint") != expected:
                    self.log.info(f"Latent cache fingerprint mismatch on {split}; re-encoding.")
                    return None
                restored[split] = {
                    "target": blob["target"],
                    "floorplan": blob["floorplan"],
                    "partial": blob["partial"],
                    # Restored, not re-derived: the TV filter and MAX_SAMPLES
                    # mean the kept rows are not a prefix of the split listing.
                    "sample_dirs": [Path(text) for text in blob["sample_dirs"]],
                    # mmap so the multi-GB id arrays never become resident.
                    "masks": np.load(str(mask_path), mmap_mode="r"),
                    "floorplan_ids": np.load(str(floorplan_path), mmap_mode="r"),
                }
                # The statistics live in the blob, alongside the fingerprint -
                # not in the per-split entry above, which only carries samples.
                if not statistics:
                    statistics = {key: blob[key] for key in STATISTIC_KEYS}
            for key in STATISTIC_KEYS:
                setattr(self, key, statistics[key])
            self.log.info(
                f"Reused latent cache for both splits "
                f"({restored['train']['target'].shape[0]} train, "
                f"{restored['test']['target'].shape[0]} test samples)."
            )
            return restored
        except (OSError, RuntimeError, KeyError, ValueError, TypeError) as exc:
            self.log.warning(f"Ignoring unreadable latent cache ({exc}); re-encoding.")
            return None

    def save_latent_cache(
        self, cache: dict[str, dict[str, Any]], counts: dict[str, int]
    ) -> None:
        """Write the standardized latents, class-id arrays, stats and fingerprint.

        Args:
            cache (dict): Per-split ``target``/``floorplan``/``partial`` latents
                plus ``"masks"`` and ``"floorplan_ids"`` arrays, with the latents
                already standardized.
            counts (dict[str, int]): Listed sample counts per split.
        """
        fingerprint: dict[str, object] = self.latent_cache_fingerprint(counts)
        statistics: dict[str, torch.Tensor] = {
            key: getattr(self, key) for key in STATISTIC_KEYS
        }
        for split, entry in cache.items():
            latent_path, mask_path, floorplan_path = self._cache_paths(split)
            latent_path.parent.mkdir(parents=True, exist_ok=True)
            # ascontiguousarray because the caller holds a memory map, which
            # np.save cannot serialize.
            np.save(str(mask_path), np.ascontiguousarray(entry["masks"]))
            np.save(
                str(floorplan_path), np.ascontiguousarray(entry["floorplan_ids"])
            )
            torch.save(
                {
                    "fingerprint": fingerprint,
                    **statistics,
                    "target": entry["target"],
                    "floorplan": entry["floorplan"],
                    "partial": entry["partial"],
                    "sample_dirs": [str(d) for d in entry["sample_dirs"]],
                },
                str(latent_path),
            )
        self.log.info(f"Saved latent cache to {Path(str(self.config.LATENT_CACHE_DIR))}")

    def build_datasets(
        self,
    ) -> tuple[MissingComponentDataset, MissingComponentDataset]:
        """Encode both splits (or reuse the cache) and wrap them as datasets.

        Datasets rather than loaders, because the training loader is rebuilt each
        epoch: the two objectives need batch sizes an order of magnitude apart,
        and re-wrapping the same dataset per epoch is far cheaper than holding two
        loaders alive or slicing batches by hand.

        Returns:
            tuple: ``(train_dataset, test_dataset)``.
        """
        image_size: int = int(self.hyperparams.IMAGE_SIZE)
        train_dirs: list[Path] = self.list_sample_dirs(str(self.config.TRAIN_IMAGES_DIR))
        test_dirs: list[Path] = self.list_sample_dirs(str(self.config.TEST_IMAGES_DIR))
        counts: dict[str, int] = {"train": len(train_dirs), "test": len(test_dirs)}
        self.log.info(f"Listed {counts['train']} train and {counts['test']} test samples.")

        cache: dict[str, dict[str, Any]] | None = self.try_load_latent_cache(counts)
        if cache is None:
            raw: dict[str, dict[str, Any]] = {}
            for split, dirs in (("train", train_dirs), ("test", test_dirs)):
                latents, masks, floorplan_ids, kept_dirs = self.encode_split(dirs, split)
                raw[split] = {
                    **latents,
                    "masks": masks,
                    "floorplan_ids": floorplan_ids,
                    "sample_dirs": kept_dirs,
                }
            self._apply_latent_stats(raw["train"])
            cache = {
                "train": self._standardize_split(raw["train"]),
                "test": self._standardize_split(raw["test"]),
            }
            self.save_latent_cache(cache, counts)

        datasets: list[MissingComponentDataset] = []
        # The cached sample_dirs, not train_dirs/test_dirs: filtering and
        # MAX_SAMPLES mean those are longer than the cache.
        for split in ("train", "test"):
            entry: dict[str, Any] = cache[split]
            datasets.append(
                MissingComponentDataset(
                    latents={
                        "target": entry["target"],
                        "floorplan": entry["floorplan"],
                        "partial": entry["partial"],
                    },
                    masks=entry["masks"],
                    floorplan_ids=entry["floorplan_ids"],
                    sample_dirs=entry["sample_dirs"],
                    image_size=image_size,
                )
            )
            self.log.info(f"{split} dataset: {len(datasets[-1])} samples.")
        return datasets[0], datasets[1]

    def make_loader(
        self,
        dataset: MissingComponentDataset,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader[MissingComponentDataset]:
        """Wrap a cached dataset in a loader of the requested batch size.

        Args:
            dataset (MissingComponentDataset): Dataset to wrap.
            batch_size (int): Samples per batch.
            shuffle (bool): Whether to reshuffle every epoch (training only).

        Returns:
            DataLoader[MissingComponentDataset]: The loader.

        Raises:
            ValueError: If a training loader is requested with a batch larger
                than the dataset. Combined with ``drop_last`` that would yield
                ZERO batches, so the epoch would silently train on nothing.
        """
        if shuffle and batch_size > len(dataset):
            raise ValueError(
                f"Training batch size {batch_size} exceeds the {len(dataset)} cached "
                "samples, which with drop_last would produce no batches at all. "
                "Lower the batch size or raise MAX_SAMPLES."
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(self.hyperparams.NUM_WORKERS),
            pin_memory=self.device.type == "cuda",
            # drop_last on train keeps every batch a uniform size, so the
            # per-batch means average cleanly.
            drop_last=shuffle,
        )

    def _standardize_split(self, split: dict[str, Any]) -> dict[str, Any]:
        """Apply each latent space's own standardization to one split.

        Args:
            split (dict): Raw ``target``/``floorplan``/``partial`` latents plus
                the ``"masks"``, ``"floorplan_ids"`` and ``"sample_dirs"`` entries.

        Returns:
            dict: The same keys, with the three latents standardized and the id
                arrays and folder list passed through untouched.
        """
        return {
            "target": self.normalize_latents(split["target"]),
            "floorplan": self.normalize_floorplan(split["floorplan"]),
            "partial": self.normalize_partial(split["partial"]),
            "masks": split["masks"],
            "floorplan_ids": split["floorplan_ids"],
            "sample_dirs": split["sample_dirs"],
        }

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def _mask_bce(
        self, logits: torch.Tensor, targets: torch.Tensor, pos_weight: float
    ) -> torch.Tensor:
        """Score one set of channels with weighted mean binary cross-entropy.

        ``reduction="mean"`` averages over every pixel of every sample, which is
        why the positive-class weight matters: without it the loss is dominated
        by the (correct) background and a missing component barely registers.

        Args:
            logits (torch.Tensor): Decoder logits of any shape.
            targets (torch.Tensor): Binary targets of the same shape.
            pos_weight (float): Positive-class weight for this term.

        Returns:
            torch.Tensor: Scalar loss.
        """
        weight: torch.Tensor = torch.full_like(targets, pos_weight)
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=weight, reduction="mean"
        )

    def latent_mse_loss(
        self,
        target_latents: torch.Tensor,
        floorplan_latents: torch.Tensor,
        partial_latents: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Standard DDPM noise-prediction MSE in the 128-d latent space.

        This is the objective the model spends most epochs on. It is the textbook
        DDPM loss (Ho et al. 2020): the forward process corrupts ``x_0``, the
        denoiser predicts the noise, and the loss is the MSE between prediction
        and the noise actually added.

        Two properties make it the right default here:

        * **The frozen decoder is not in the graph at all**, so memory is
          dominated by the 128-d latents and ``MSE_BATCH_SIZE`` can be an order of
          magnitude larger than the pixel phase's.
        * **Timesteps are drawn from the full ``[0, NUM_TIMESTEPS)`` range.**
          Truncating the schedule was only ever necessary because a *decoded* BCE
          is meaningless at high noise; a latent objective is well defined at
          every noise level, so this phase trains the schedule sampling actually
          uses.

        Args:
            target_latents (torch.Tensor): Standardized ``x_0``, ``(B, LATENT_DIM)``.
            floorplan_latents (torch.Tensor): Standardized condition, ``(B, COND_DIM)``.
            partial_latents (torch.Tensor): Standardized TV-free condition,
                ``(B, COND_DIM)``.

        Returns:
            dict[str, torch.Tensor]: ``loss`` (equal to ``mse``) and ``mse``.
        """
        batch_size: int = target_latents.shape[0]
        timesteps: torch.Tensor = torch.randint(
            0, self.model.num_timesteps, (batch_size,), device=target_latents.device
        )
        noise: torch.Tensor = torch.randn_like(target_latents)
        noisy: torch.Tensor = self.model.q_sample(target_latents, timesteps, noise)
        predicted_noise: torch.Tensor = self.model(
            noisy, timesteps, floorplan_latents, partial_latents
        )
        mse: torch.Tensor = F.mse_loss(predicted_noise, noise)
        return {"loss": mse, "mse": mse}

    def pixel_bce_loss(
        self,
        target_latents: torch.Tensor,
        floorplan_latents: torch.Tensor,
        partial_latents: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the pixel-space missing-component loss for one batch.

        A timestep is drawn from ``[0, BCE_TRAIN_T_MAX)``, the forward process
        corrupts the target latent, the denoiser predicts the noise, the noise is
        inverted back into ``x_0_hat`` and the frozen target decoder turns that
        into mask logits. **Only the missing component's channel is scored.**

        That is deliberate. The other ten furniture classes get no pixel-space
        term at all, because the latent MSE has no notion of which class matters
        and a whole-mask BCE spends almost all of its weight on background: with
        ~97% background and ten classes sparser than the TV, the full term was
        ~90% correct-forever pixels. Since the deliverable is the missing
        component, the pixel objective supervises exactly that channel and
        nothing else.

        The decoder call is deliberately NOT wrapped in ``no_grad``: the whole
        point is that the pixel-space gradient reaches the denoiser through it.
        That is also why this epoch runs at a small batch size.

        Args:
            target_latents (torch.Tensor): Standardized ``x_0``, ``(B, LATENT_DIM)``.
            floorplan_latents (torch.Tensor): Standardized condition, ``(B, COND_DIM)``.
            partial_latents (torch.Tensor): Standardized TV-free condition,
                ``(B, COND_DIM)``.
            target_masks (torch.Tensor): One-vs-rest ground truth, ``(B, C, S, S)``
                valued ``0.0``/``1.0``.

        Returns:
            dict[str, torch.Tensor]: ``loss``, ``tv_bce``, the detached ``logits``
                for evaluation, the drawn ``timestep``, and the noise pair so the
                caller can report the latent MSE of the same forward pass for free.

        Raises:
            ValueError: If ``target_masks`` does not have the target's channel count.
        """
        if target_masks.shape[1] != self.target_layout.num_channels:
            raise ValueError(
                f"Ground truth has {target_masks.shape[1]} channels but the target "
                f"decoder predicts {self.target_layout.num_channels}."
            )
        batch_size: int = target_latents.shape[0]
        timesteps: torch.Tensor = torch.randint(
            0, self.bce_train_t_max, (batch_size,), device=target_latents.device
        )
        noise: torch.Tensor = torch.randn_like(target_latents)
        noisy: torch.Tensor = self.model.q_sample(target_latents, timesteps, noise)
        predicted_noise: torch.Tensor = self.model(
            noisy, timesteps, floorplan_latents, partial_latents
        )
        estimated_clean: torch.Tensor = self.model.x0_from_eps(
            noisy, timesteps, predicted_noise
        )
        # denormalize first: the frozen decoder only ever saw raw-scale latents.
        logits: torch.Tensor = self.frozen_vaes["target"].decode(
            self.denormalize_latents(estimated_clean)
        )
        tv_bce: torch.Tensor = self._mask_bce(
            logits[:, self.missing_channel],
            target_masks[:, self.missing_channel],
            float(self.hyperparams.TV_BCE_POS_WEIGHT),
        )
        loss: torch.Tensor = float(self.hyperparams.TV_LOSS_WEIGHT) * tv_bce
        return {
            "loss": loss,
            "tv_bce": tv_bce,
            "logits": logits.detach(),
            "timestep": timesteps,
            # Returned so the caller can report the latent MSE of the very same
            # forward pass, which keeps that curve continuous across the phase
            # switch at no extra compute.
            "noise": noise,
            "predicted_noise": predicted_noise,
        }

    def _targets_to_device(self, class_ids: torch.Tensor) -> torch.Tensor:
        """Expand class ids into the one-vs-rest BCE target on the training device.

        Args:
            class_ids (torch.Tensor): ``(B, S, S)`` int64 class ids.

        Returns:
            torch.Tensor: Float32 ``(B, C, S, S)`` binary target.
        """
        return class_ids_to_one_hot(class_ids.to(self.device), self.target_layout.num_channels)

    def training_step(
        self, batch: dict[str, torch.Tensor], objective: str
    ) -> dict[str, float | None]:
        """Run one optimization step on a batch with the given objective.

        Exactly one objective is active per step; they are never summed.

        Args:
            batch (dict[str, torch.Tensor]): Cached ``target``, ``floorplan``,
                ``partial`` latents and ``class_ids``.
            objective (str): ``"mse"`` for the latent-space loss, ``"bce"`` for
                the pixel-space loss.

        Returns:
            dict: ``loss`` plus the active objective's own terms. Keys belonging
                to the other objective are ``None``, never ``0.0``, so telemetry
                leaves an empty cell instead of recording a fake zero.

        Raises:
            ValueError: If ``objective`` is not a known objective.
        """
        self.optimizer.zero_grad(set_to_none=True)
        if objective == "mse":
            metrics: dict[str, torch.Tensor] = self.latent_mse_loss(
                target_latents=batch["target"].to(self.device),
                floorplan_latents=batch["floorplan"].to(self.device),
                partial_latents=batch["partial"].to(self.device),
            )
            metrics["loss"].backward()
            reported: dict[str, float | None] = {
                "loss": float(metrics["mse"].detach()),
                "mse": float(metrics["mse"].detach()),
                "tv_bce": None,
            }
        elif objective == "bce":
            metrics = self.pixel_bce_loss(
                target_latents=batch["target"].to(self.device),
                floorplan_latents=batch["floorplan"].to(self.device),
                partial_latents=batch["partial"].to(self.device),
                target_masks=self._targets_to_device(batch["class_ids"]),
            )
            metrics["loss"].backward()
            latent_mse: float = float(
                F.mse_loss(metrics["predicted_noise"].detach(), metrics["noise"])
            )
            reported = {
                "loss": float(metrics["loss"].detach()),
                # The MSE of this batch is measured for free from the same forward
                # pass, so both objectives stay visible across the phase boundary.
                "mse": latent_mse,
                "tv_bce": float(metrics["tv_bce"].detach()),
            }
        else:
            raise ValueError(f"Unknown objective {objective!r}; expected 'mse' or 'bce'.")
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return reported

    def run_epoch(
        self, loader: DataLoader[MissingComponentDataset], objective: str
    ) -> dict[str, float | None]:
        """Train for one pass over the split with a single objective.

        Args:
            loader (DataLoader[MissingComponentDataset]): Training loader, already
                sized for ``objective``.
            objective (str): ``"mse"`` or ``"bce"``.

        Returns:
            dict: Mean ``loss``, ``mse`` and ``tv_bce``. Keys that do not apply to
                ``objective`` are ``None`` rather than ``0.0``, so telemetry writes
                an empty cell instead of a misleading number.
        """
        self.model.train()
        totals: dict[str, float] = {"loss": 0.0, "mse": 0.0, "tv_bce": 0.0}
        batches: int = 0
        for batch in loader:
            metrics: dict[str, float | None] = self.training_step(batch, objective)
            for key in totals:
                value: float | None = metrics.get(key)
                if value is not None:
                    totals[key] += value
            batches += 1
        if batches == 0:
            self.log.error("Training loader produced no batches; nothing to train on.")
            return {"loss": 0.0, "mse": 0.0, "tv_bce": 0.0}
        active: set[str] = {"loss"} | ({"mse"} if objective == "mse" else {"tv_bce"})
        averaged: dict[str, float | None] = {
            key: (value / batches if key in active else None)
            for key, value in totals.items()
        }
        return averaged

    def _accumulate_confusion(
        self,
        confusion: torch.Tensor | None,
        logits: torch.Tensor,
        class_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Fold one batch's predicted-vs-true confusion into the running total.

        All channels are accumulated, not just the missing one, so the same
        matrix drives both the sampled TV score and per-class telemetry.
        Summing over batches before measuring makes the result a property of the
        whole evaluation set rather than an average of per-batch overlaps.

        Args:
            confusion (torch.Tensor | None): Running total, or None to start.
            logits (torch.Tensor): Decoded logits, ``(B, C, S, S)``.
            class_ids (torch.Tensor): ``(B, S, S)`` int64 ground-truth class ids.

        Returns:
            torch.Tensor: Updated ``(C, C)`` confusion matrix on CPU.
        """
        targets: torch.Tensor = self._targets_to_device(class_ids).cpu()
        matrix: torch.Tensor = confusion_matrix(
            logits.detach().cpu(),
            targets,
            self.target_layout.num_channels,
            layout=self.target_layout,
        )
        return matrix if confusion is None else confusion + matrix

    @torch.no_grad()
    def evaluate_latent_mse(
        self, loader: DataLoader[MissingComponentDataset], max_batches: int
    ) -> float:
        """Score the latent-space MSE on held-out data without training.

        Needs no decoder, so this is cheap enough to run every epoch in both
        phases and it is the curve that stays continuous across the objective
        switch.

        Args:
            loader (DataLoader[MissingComponentDataset]): Evaluation loader.
            max_batches (int): Cap on evaluated batches; negative means all.

        Returns:
            float: Mean noise-prediction MSE, or ``0.0`` if there were no batches.
        """
        self.model.eval()
        total: float = 0.0
        batches: int = 0
        for batch_index, batch in enumerate(loader):
            if 0 <= max_batches <= batch_index:
                break
            metrics: dict[str, torch.Tensor] = self.latent_mse_loss(
                target_latents=batch["target"].to(self.device),
                floorplan_latents=batch["floorplan"].to(self.device),
                partial_latents=batch["partial"].to(self.device),
            )
            total += float(metrics["mse"])
            batches += 1
        self.model.train()
        if batches == 0:
            self.log.error("Latent MSE evaluation had no batches.")
            return 0.0
        return total / batches

    @torch.no_grad()
    def evaluate_pixel(
        self, loader: DataLoader[MissingComponentDataset], max_batches: int
    ) -> dict[str, float]:
        """Score the pixel-space loss and TV overlap without training.

        Uses the same corrupted-latent objective as the BCE phase, so this is a
        like-for-like comparison, plus the exact predicted-vs-true TV confusion,
        which is the number that actually says whether the component was placed.

        Expensive (it decodes), so during the MSE phase it runs only every
        ``PIXEL_EVAL_INTERVAL`` epochs.

        Args:
            loader (DataLoader[MissingComponentDataset]): Evaluation loader.
            max_batches (int): Cap on evaluated batches; negative means all.

        Returns:
            dict[str, float]: ``loss``, ``tv_bce``, ``tv_dice``, ``tv_iou`` and
                ``samples``.
        """
        self.model.eval()
        totals: dict[str, float] = {"loss": 0.0, "tv_bce": 0.0}
        batches: int = 0
        confusion: torch.Tensor | None = None
        for batch_index, batch in enumerate(loader):
            if 0 <= max_batches <= batch_index:
                break
            metrics: dict[str, torch.Tensor] = self.pixel_bce_loss(
                target_latents=batch["target"].to(self.device),
                floorplan_latents=batch["floorplan"].to(self.device),
                partial_latents=batch["partial"].to(self.device),
                target_masks=self._targets_to_device(batch["class_ids"]),
            )
            for key in totals:
                totals[key] += float(metrics[key])
            # Reuse the logits the loss already produced instead of paying for a
            # second forward pass just to score the confusion.
            confusion = self._accumulate_confusion(
                confusion, metrics["logits"], batch["class_ids"]
            )
            batches += 1
        if batches == 0:
            self.log.error("Pixel evaluation had no batches.")
            return {
                "loss": 0.0, "tv_bce": 0.0,
                "tv_dice": 0.0, "tv_iou": 0.0, "samples": 0.0,
            }
        overlap: ClassOverlap = overlap_from_confusion(confusion)
        result: dict[str, float] = {key: value / batches for key, value in totals.items()}
        result["tv_dice"] = float(overlap.dice[self.missing_channel])
        result["tv_iou"] = float(overlap.iou[self.missing_channel])
        result["samples"] = float(
            batches * int(self.hyperparams.PIXEL_EVAL_BATCH_SIZE)
        )
        self.model.train()
        return result

    @torch.no_grad()
    def evaluate_per_class(
        self, loader: DataLoader[MissingComponentDataset], max_batches: int
    ) -> tuple[list[float], list[float]]:
        """Measure per-class IoU and Dice over the evaluated batches.

        The number this project actually cares about is the missing component,
        and an averaged loss hides which channel is holding the model back, so
        every channel is scored into its own telemetry column.

        Args:
            loader (DataLoader[MissingComponentDataset]): Evaluation loader.
            max_batches (int): Cap on evaluated batches; negative means all.

        Returns:
            tuple[list[float], list[float]]: Per-class IoU and Dice, indexed by
                channel like the layout's ``class_names``.
        """
        self.model.eval()
        confusion: torch.Tensor | None = None
        batches: int = 0
        for batch_index, batch in enumerate(loader):
            if 0 <= max_batches <= batch_index:
                break
            logits: torch.Tensor = self._predict_noisy_logits(batch)
            confusion = self._accumulate_confusion(confusion, logits, batch["class_ids"])
            batches += 1
        self.model.train()
        if confusion is None:
            self.log.error("Per-class evaluation had no batches.")
            absent: list[float] = [0.0] * self.target_layout.num_channels
            return absent, list(absent)
        overlap: ClassOverlap = overlap_from_confusion(confusion)
        iou: list[float] = [float(value) for value in overlap.iou]
        dice: list[float] = [float(value) for value in overlap.dice]
        self.log.info(
            f"Per-class Dice over {batches} batches: "
            + ", ".join(
                f"{name}={dice[index]:.3f}"
                for index, name in enumerate(self.target_group.class_names)
            )
        )
        return iou, dice

    @torch.no_grad()
    def _predict_noisy_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the BCE-phase forward pass and decode, without gradients.

        Used where the pixel loss itself is not wanted but its decoded masks are
        (the per-class evaluation), so the logits match BCE-phase training
        exactly. Timesteps are capped at ``BCE_TRAIN_T_MAX`` for the same reason
        training caps them: the estimate has to be decodable to mean anything.

        Args:
            batch (dict[str, torch.Tensor]): Batch with latents and class ids.

        Returns:
            torch.Tensor: Decoded logits, ``(B, C, S, S)``.
        """
        target: torch.Tensor = batch["target"].to(self.device)
        timesteps: torch.Tensor = torch.randint(
            0, self.bce_train_t_max, (target.shape[0],), device=self.device
        )
        noisy: torch.Tensor = self.model.q_sample(target, timesteps, torch.randn_like(target))
        noise_hat: torch.Tensor = self.model(
            noisy,
            timesteps,
            batch["floorplan"].to(self.device),
            batch["partial"].to(self.device),
        )
        clean_hat: torch.Tensor = self.model.x0_from_eps(noisy, timesteps, noise_hat)
        return self.frozen_vaes["target"].decode(self.denormalize_latents(clean_hat))

    @torch.no_grad()
    def sample_and_score(self, batch: dict[str, torch.Tensor], num_draws: int) -> dict[str, Any]:
        """Run the full reverse chain and return decoded generations plus GT.

        This is the honest end-to-end measurement: it uses the complete
        ``NUM_TIMESTEPS`` chain rather than the truncated training objective, so
        it is the number to trust about whether generation actually works.

        Args:
            batch (dict[str, torch.Tensor]): Fixed evaluation batch with latents
                and ``class_ids``.
            num_draws (int): Fresh noise draws per condition.

        Returns:
            dict: ``logits`` ``(B * num_draws, C, S, S)`` in
                sample-major/draw-minor order, ``targets`` ``(B, C, S, S)`` and
                the missing channel index.
        """
        self.model.eval()
        floorplan: torch.Tensor = batch["floorplan"].to(self.device)
        partial: torch.Tensor = batch["partial"].to(self.device)
        # repeat_interleave puts each sample's draws next to each other, so
        # draw d of every sample is a strided slice of the result.
        sampled: torch.Tensor = self.model.sample_conditional(
            floorplan.repeat_interleave(num_draws, dim=0),
            partial.repeat_interleave(num_draws, dim=0),
        )
        logits: torch.Tensor = self.frozen_vaes["target"].decode(
            self.denormalize_latents(sampled)
        )
        targets: torch.Tensor = self._targets_to_device(batch["class_ids"])
        self.model.train()
        return {"logits": logits, "targets": targets, "channel": self.missing_channel}

    @torch.no_grad()
    def _score_sampled(self, batch: dict[str, torch.Tensor]) -> tuple[float, float]:
        """Score the full reverse chain's missing-component overlap.

        Args:
            batch (dict[str, torch.Tensor]): Fixed evaluation batch.

        Returns:
            tuple[float, float]: Dice and IoU of the decoded samples.
        """
        num_draws: int = int(self.hyperparams.NUM_GEN_DRAWS)
        result: dict[str, Any] = self.sample_and_score(batch, num_draws)
        # sample_and_score interleaves the draws inside each sample, so the
        # ground truth is tiled the same way before it can be compared.
        targets: torch.Tensor = result["targets"].repeat_interleave(num_draws, dim=0)
        matrix: torch.Tensor = confusion_matrix(
            result["logits"].cpu(),
            targets.cpu(),
            self.target_layout.num_channels,
            layout=self.target_layout,
        )
        overlap: ClassOverlap = overlap_from_confusion(matrix)
        return float(overlap.dice[self.missing_channel]), float(overlap.iou[self.missing_channel])

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def _component_rgb(self, mask: torch.Tensor, channel: int) -> torch.Tensor:
        """Render a single component as an RGB batch on a background.

        Args:
            mask (torch.Tensor): ``(B, H, W)`` or ``(B, 1, H, W)`` binary mask.
            channel (int): Channel the component is written to.

        Returns:
            torch.Tensor: ``(B, 3, H, W)`` RGB in ``0..1``.
        """
        binary: torch.Tensor = mask.reshape(mask.shape[0], 1, *mask.shape[-2:]).to(torch.float32)
        tensor: torch.Tensor = torch.zeros(
            (binary.shape[0], self.target_layout.num_channels, *binary.shape[-2:]),
            dtype=torch.float32,
        )
        tensor[:, channel] = binary[:, 0]
        # Background is the complement, so argmax resolves to the component where
        # the mask is set and to background everywhere else.
        tensor[:, 0:1] = 1.0 - binary
        return render_group_rgb(tensor, self.target_group, torch.device("cpu"))

    def _input_layout_rgb(self, class_ids: torch.Tensor, channel: int) -> torch.Tensor:
        """Render the layout the model is actually GIVEN, missing component aside.

        That is the ground truth with the missing component's pixels remapped to
        background - exactly what the partial (fair-burro) encoder saw, because
        that encoder's layout carries the same exclusion. Showing it makes the
        conditioning visible: every other furniture class is present, and the one
        class the model has to invent is simply absent.

        The remap follows ``remap_ids_to_background``, the same rule the encoder
        uses, so this row cannot drift from the true input.

        Args:
            class_ids (torch.Tensor): ``(B, S, S)`` int64 ground-truth class ids.
            channel (int): Channel of the component to remove.

        Returns:
            torch.Tensor: ``(B, 3, H, W)`` RGB in ``0..1`` on CPU, with the
                component's pixels rendered as background.
        """
        cpu: torch.device = torch.device("cpu")
        erased: torch.Tensor = class_ids.detach().to(cpu).clone()
        erased[erased == channel] = 0
        targets: torch.Tensor = class_ids_to_one_hot(
            erased, self.target_layout.num_channels
        )
        return render_group_rgb(targets, self.target_group, cpu)

    @torch.no_grad()
    def save_visualization_grid(self, batch: dict[str, torch.Tensor], epoch: int) -> None:
        """Save the missing-component grid.

        Rows, ``NUM_VIZ_SAMPLES`` columns each:

        1. **ground truth** - the full arrangement, colour-coded;
        2. **floorplan + ground-truth arrangement** - the complete answer placed
           on the floorplan it belongs to;
        3. **floorplan + the INPUT layout** - every channel the model is actually
           given, i.e. all furniture EXCEPT the missing component. This is the
           information the conditioning encoder saw, and the one class the model
           has to invent is visibly absent;
        4..(3 + NUM_GEN_DRAWS) **floorplan + predicted component** - one row per
           fresh diffusion draw, showing ONLY the model's missing component
           painted on the bare floorplan.

        Rows 3 and 4+ read together as "given" + "generated" = the answer, so a
        column can be checked against row 2 in one glance.

        The base for row 3 and rows 4+ is the floorplan, which by construction
        contains no furniture at all. That is what makes row 4+ judgeable: every
        pixel there is the model's own, so a missing, misplaced or oversized
        component all show up, and there is no ground-truth component sitting
        underneath to be confused with it.

        Named ``reconstruction_epoch_<N>_<run_id>.png`` so the shared dashboard's
        "Reconstructions" panel picks it up with no server change.

        Args:
            batch (dict[str, torch.Tensor]): Fixed batch of ``NUM_VIZ_SAMPLES``,
                including the cached floorplan and furniture class ids.
            epoch (int): Current epoch, used in the file name.
        """
        output_dir: Path = Path(str(self.config.OUTPUT_SAMPLES_DIRECTORY))
        output_dir.mkdir(parents=True, exist_ok=True)
        draws: int = int(self.hyperparams.NUM_GEN_DRAWS)
        num_viz: int = int(self.hyperparams.NUM_VIZ_SAMPLES)
        alpha: float = float(self.hyperparams.OVERLAY_ALPHA)
        channel: int = self.missing_channel
        cpu: torch.device = torch.device("cpu")

        result: dict[str, Any] = self.sample_and_score(batch, draws)
        logits: torch.Tensor = result["logits"].cpu()
        targets: torch.Tensor = result["targets"].cpu()

        # Row 1: the ground-truth arrangement on its own.
        gt_rgb: torch.Tensor = render_group_rgb(targets, self.target_group, cpu)
        # The floorplan base, rendered from its cached class ids through the same
        # colour table as everything else in the grid.
        floor_rgb: torch.Tensor = render_group_rgb(
            class_ids_to_one_hot(
                batch["floorplan_class_ids"].cpu(),
                self.floorplan_layout.num_channels,
            ),
            self.floorplan_layout.groups[0],
            cpu,
        )
        # Row 2: the ground truth laid over that floorplan. Furniture background
        # is excluded so the rooms and walls show through.
        gt_foreground: torch.Tensor = (targets.argmax(dim=1) != 0).to(floor_rgb.dtype)
        gt_overlay: torch.Tensor = overlay_component_on_base(
            floor_rgb, gt_rgb, gt_foreground, alpha
        )

        # Row 3: the INPUT - every furniture channel the model is given, with the
        # missing component removed. Drawn from the class ids with the component
        # remapped to background, the same rule the partial encoder applies, so
        # this row is the conditioning and not an approximation of it.
        input_ids: torch.Tensor = batch["class_ids"].detach().to(cpu)
        input_rgb: torch.Tensor = self._input_layout_rgb(input_ids, channel)
        # Foreground is taken from the ids rather than by testing the rendered RGB
        # for non-black: the class ids say exactly which pixels are furniture, and
        # which of those are the component being removed.
        input_foreground: torch.Tensor = (
            (input_ids != 0) & (input_ids != channel)
        ).unsqueeze(1).to(floor_rgb.dtype)
        input_overlay: torch.Tensor = overlay_component_on_base(
            floor_rgb, input_rgb, input_foreground, alpha
        )

        # Rows 4+: only the missing component comes from the model, painted on the
        # bare floorplan. Nothing else is drawn, so nothing can be confused with it.
        prediction_rows: list[torch.Tensor] = []
        for draw in range(draws):
            # Draw d of every sample: strided because sample_and_score lays the
            # batch out sample-major with the draws adjacent to each sample.
            draw_logits: torch.Tensor = logits[draw::draws]
            pred_component: torch.Tensor = draw_logits[:, channel] > 0.0
            prediction_rows.append(
                overlay_component_on_base(
                    floor_rgb,
                    self._component_rgb(pred_component, channel),
                    pred_component,
                    alpha,
                )
            )

        grid: torch.Tensor = make_grid(
            torch.cat([gt_rgb, gt_overlay, input_overlay, *prediction_rows], dim=0),
            nrow=num_viz,
            padding=2,
            normalize=False,
        )
        # normalize=False matters: rows are already 0..1 color-table output and
        # make_grid's default rescaling would shift every dataset color.
        grid_path: Path = output_dir / (
            f"reconstruction_epoch_{epoch}_{self.config.runtime.RUN_ID}.png"
        )
        transforms.ToPILImage()(grid).save(str(grid_path))
        self.log.info(f"Saved visualization grid to {grid_path}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _fixed_batch(
        self, loader: DataLoader[MissingComponentDataset]
    ) -> dict[str, torch.Tensor]:
        """Take one deterministic batch for evaluation and visualization.

        The batch comes from the unshuffled test loader and is reused every
        epoch, so grid rows and sampled metrics are comparable across epochs.

        Args:
            loader (DataLoader[MissingComponentDataset]): Test loader.

        Returns:
            dict[str, torch.Tensor]: The first ``NUM_VIZ_SAMPLES`` samples.

        Raises:
            ValueError: If the test split holds fewer than ``NUM_VIZ_SAMPLES``.
        """
        num_viz: int = int(self.hyperparams.NUM_VIZ_SAMPLES)
        collected: dict[str, list[torch.Tensor]] = {
            "target": [],
            "floorplan": [],
            "partial": [],
            "class_ids": [],
            "floorplan_class_ids": [],
        }
        for batch in loader:
            for key in collected:
                collected[key].append(batch[key][:num_viz])
            if collected["target"][0].shape[0] >= num_viz:
                break
        combined: dict[str, torch.Tensor] = {
            key: torch.cat(value, dim=0)[:num_viz] for key, value in collected.items()
        }
        if combined["target"].shape[0] < num_viz:
            raise ValueError(
                f"Need at least NUM_VIZ_SAMPLES={num_viz} test samples but the fixed "
                f"batch holds {combined['target'].shape[0]}. Lower NUM_VIZ_SAMPLES."
            )
        return combined

    def save_checkpoint(self, epoch: int, objective: str | None = None) -> None:
        """Save weights, optimizer state, statistics and every frozen path.

        Args:
            epoch (int): Current epoch, used in the file name.
            objective (str | None): Which objective this epoch trained with,
                recorded so a checkpoint is self-describing about the phase it
                ended in.
        """
        os.makedirs(str(self.config.MODEL_SAVE_PATH), exist_ok=True)
        checkpoint_path: str = os.path.join(
            str(self.config.MODEL_SAVE_PATH),
            f"mask_diffusion_{self.config.runtime.RUN_ID}_epoch_{epoch}.pt",
        )
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "objective": objective,
                "loss_schedule": self.loss_schedule,
                "hyperparameters": vars(self.hyperparams),
                "missing_channel": self.missing_channel,
                "floorplan_checkpoint": str(self.config.FLOORPLAN_VAE_CHECKPOINT_PATH),
                "partial_checkpoint": str(self.config.PARTIAL_VAE_CHECKPOINT_PATH),
                "target_checkpoint": str(self.config.TARGET_VAE_CHECKPOINT_PATH),
                **{key: getattr(self, key) for key in STATISTIC_KEYS},
            },
            checkpoint_path,
        )
        self.log.info(f"Saved checkpoint to {checkpoint_path}")

    def verify_one_hot_matches_mask_vae(self) -> None:
        """Assert the local one-hot helper agrees with mask_VAE's own encoding.

        :func:`class_ids_to_one_hot` exists only to allocate on the training
        device; if it ever drifted from
        :func:`mask_VAE.masking.encode_masks_to_channels` the loss would be
        scoring against different targets than the VAE was trained on, silently.
        Checking it once at startup makes that a loud failure instead.

        Raises:
            AssertionError: If the two encodings disagree.
        """
        probe: torch.Tensor = torch.from_numpy(
            np.array([[0, 1, 2, 3], [4, 5, 0, 9]], dtype=np.int64)
        ).unsqueeze(0)
        expected: torch.Tensor = encode_masks_to_channels([probe], self.target_layout)
        actual: torch.Tensor = class_ids_to_one_hot(
            probe, self.target_layout.num_channels
        )
        if not bool(torch.equal(expected, actual)):
            raise AssertionError(
                "class_ids_to_one_hot disagrees with mask_VAE's encode_masks_to_channels; "
                "the pixel-space loss would be scored against the wrong targets."
            )
        self.log.info("One-hot target helper verified against mask_VAE's encoder.")

    def train(self) -> None:
        """Run the full training loop with telemetry, grids and checkpoints.

        The objective follows ``LOSS_SCHEDULE``: the cheap latent MSE first, then
        the expensive pixel BCE for a final stretch. Evaluation mirrors that
        split - the latent MSE is scored every epoch because it needs no decoder,
        while the pixel pass only runs on ``PIXEL_EVAL_INTERVAL`` during the MSE
        phase (and every epoch during the BCE phase, where it is the objective
        itself).

        A grid is saved before the first epoch so the pipeline can be inspected
        on an untrained model: if those rows look wrong, the bug is in the data
        wiring rather than in the training.
        """
        self.verify_one_hot_matches_mask_vae()
        num_epochs: int = int(self.hyperparams.NUM_EPOCHS)
        telemetry_interval: int = int(self.hyperparams.TELEMETRY_INTERVAL)
        checkpoint_interval: int = int(self.hyperparams.CHECKPOINT_INTERVAL)
        grid_interval: int = int(self.hyperparams.GRID_INTERVAL)
        sample_interval: int = int(self.hyperparams.SAMPLE_INTERVAL)
        per_class_interval: int = int(self.hyperparams.PER_CLASS_INTERVAL)
        pixel_eval_interval: int = int(self.hyperparams.PIXEL_EVAL_INTERVAL)
        eval_batch_size: int = int(self.hyperparams.EVAL_BATCH_SIZE)
        pixel_eval_batch_size: int = int(self.hyperparams.PIXEL_EVAL_BATCH_SIZE)
        eval_batches: int = max(1, int(self.hyperparams.TEST_LOSS_SAMPLES) // eval_batch_size)
        pixel_eval_batches: int = max(
            1, int(self.hyperparams.TEST_LOSS_SAMPLES) // pixel_eval_batch_size
        )

        train_dataset, test_dataset = self.build_datasets()
        # The fixed visualization batch comes from the unshuffled test split.
        fixed: dict[str, torch.Tensor] = self._fixed_batch(
            self.make_loader(test_dataset, eval_batch_size, shuffle=False)
        )
        self.save_visualization_grid(fixed, 0)

        # Retained across epochs so telemetry can carry the most recent value of
        # each metric forward; the inactive one is written as an empty cell.
        last_tv_dice: float | None = None
        last_tv_bce: float | None = None

        total_start: float = time.time()
        for epoch in range(1, num_epochs + 1):
            epoch_start: float = time.time()
            objective: str = self.objective_for_epoch(epoch)
            if epoch == 1 or self.objective_for_epoch(epoch - 1) != objective:
                # Fires on every alternation, so a cycling schedule shows its
                # rhythm in the log rather than only in the telemetry.
                self.log.info(
                    f"Epoch {epoch}: objective '{objective}' "
                    f"(batch {self.batch_size_for_objective(objective)})."
                )

            train_loader: DataLoader[MissingComponentDataset] = self.make_loader(
                train_dataset, self.batch_size_for_objective(objective), shuffle=True
            )
            train_metrics: dict[str, float | None] = self.run_epoch(train_loader, objective)

            # Latent MSE every epoch: cheap, and the curve stays continuous
            # across the objective switch.
            test_mse: float = self.evaluate_latent_mse(
                self.make_loader(test_dataset, eval_batch_size, shuffle=False), eval_batches
            )
            # Pixel pass: expensive, so throttled during the MSE phase.
            run_pixel_eval: bool = (
                objective == "bce" or epoch % pixel_eval_interval == 0 or epoch == num_epochs
            )
            if run_pixel_eval:
                pixel: dict[str, float] = self.evaluate_pixel(
                    self.make_loader(
                        test_dataset, pixel_eval_batch_size, shuffle=False
                    ),
                    pixel_eval_batches,
                )
                last_tv_dice = pixel["tv_dice"]
                last_tv_bce = pixel["tv_bce"]
            if epoch % sample_interval == 0 or epoch == num_epochs:
                self.last_sample_tv_dice, self.last_sample_tv_iou = self._score_sampled(fixed)
            if epoch % per_class_interval == 0 or epoch == num_epochs:
                iou, dice = self.evaluate_per_class(
                    self.make_loader(
                        test_dataset, pixel_eval_batch_size, shuffle=False
                    ),
                    pixel_eval_batches,
                )
                self.per_class_telemetry.log(epoch, iou, dice)

            self.log.info(
                f"Epoch {epoch:>5} [{objective}] | loss {train_metrics['loss']:.4f} "
                f"(mse {_fmt(train_metrics['mse'])}, tv {_fmt(train_metrics['tv_bce'])}) "
                f"| test mse {test_mse:.4f} tv_bce {_fmt(last_tv_bce)} "
                f"tv_dice {_fmt(last_tv_dice)} | {time.time() - epoch_start:.1f}s"
            )

            if epoch % telemetry_interval == 0 or epoch == num_epochs:
                self.telemetry.log(
                    epoch=epoch,
                    train_loss=train_metrics["loss"],
                    train_mse=train_metrics["mse"],
                    train_tv_bce=train_metrics["tv_bce"],
                    test_mse=test_mse,
                    test_tv_bce=last_tv_bce if objective == "bce" else None,
                    test_tv_dice=last_tv_dice,
                    sample_tv_dice=self.last_sample_tv_dice,
                    sample_tv_iou=self.last_sample_tv_iou,
                )
            if epoch % grid_interval == 0 or epoch == num_epochs:
                self.save_visualization_grid(fixed, epoch)
            if epoch % checkpoint_interval == 0 or epoch == num_epochs:
                self.save_checkpoint(epoch, objective)

        # The last epoch already checkpointed via the `or epoch == num_epochs`
        # branches above, so there is nothing to flush here but the files.
        self.close()
        self.log.info(
            f"Done in {(time.time() - total_start) / 60.0:.1f} min. Final test "
            f"tv_dice {_fmt(last_tv_dice)}, tv_bce {_fmt(last_tv_bce)}."
        )

    def close(self) -> None:
        """Flush and close both telemetry files.

        ``TelemetryWriter`` deliberately holds its file open for the whole run
        and exposes no close method, so the underlying handle is closed here
        directly; that is the only way to release it. Idempotent, because closing
        early and again at the end of a run is easy to do by accident.
        """
        if self.telemetry.file.closed:
            return
        self.telemetry.file.flush()
        self.telemetry.file.close()
        self.per_class_telemetry.close()