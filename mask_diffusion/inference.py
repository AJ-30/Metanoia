"""Inference-only sampling API for a trained missing-component diffusion run.

``mask_diffusion/runner.py`` is the training runner: it builds datasets, caches
latents and steps a loss. This module is the other half of the package - the
part an *evaluator* needs and nothing else. It loads a self-describing training
checkpoint and exposes exactly three operations:

1. :meth:`MaskDiffusionSampler.encode_conditions` - push sample folders through
   the frozen floorplan and TV-free encoders, giving the two conditioning
   latents of every sample.
2. :meth:`MaskDiffusionSampler.normalize_conditions` - standardize them with the
   statistics the checkpoint itself carries, so sampling matches the run.
3. :meth:`MaskDiffusionSampler.generate_masks` - run the full reverse chain for
   as many draws as asked and decode each draw into a class-id mask.

Why a separate module rather than a method on the runner:

* A checkpoint is **self-describing** (hyperparameters, latent statistics, the
  resolved missing channel and all three frozen VAE paths, see
  ``MaskDiffusionRunner.save_checkpoint``). Loading it here means an evaluation
  can never sample with a VAE or layout that differs from the run being judged.
* Nothing here writes, caches or trains, so importing it cannot perturb the
  training path, and the training runner does not import it back.

Only the *decode* half of the frozen target VAE is used for scoring; the
generated mask is the argmax over channels, so a pixel belongs to the missing
component exactly when the decoder assigns that channel the highest logit - the
same rule the training grids use, never a tuned threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from conditional_diffusion.runner import (  # noqa: E402
    build_frozen_mask_vae,
    resolve_color_index_file,
)
from mask_VAE.masking import (  # noqa: E402
    MaskGroup,
    MaskLayout,
    encode_masks_to_single_channel,
)
from mask_VAE.model import MaskVAEModel  # noqa: E402
from mask_diffusion.model import MaskComponentDiffusion  # noqa: E402
from mask_diffusion.runner import FROZEN_VAE_KEYS, STATISTIC_KEYS  # noqa: E402
from VAE.runner import resolve_device  # noqa: E402

# Mask image modes accepted when reading a source mask, and the resampling
# filter. Copied from the runner rather than imported as a private name, and
# kept identical: an interpolating filter would blend neighbouring class ids
# into values that belong to no class at all.
ACCEPTED_MASK_IMAGE_MODES: frozenset[str] = frozenset({"L", "P"})
MASK_IMAGE_MODE: str = "L"
MASK_RESAMPLE: Image.Resampling = Image.Resampling.NEAREST

# Seed mixing constant for the per-chunk reverse-chain RNG. Any large odd
# number works; it only has to keep a chunk's seed far from a neighbouring
# chunk's so two chunks cannot share a noise stream by accident.
CHUNK_SEED_STRIDE: int = 1_000_003


def load_class_ids(mask_path: Path, image_size: int) -> torch.Tensor:
    """Read one segmentation mask PNG as an int64 class-id tensor.

    Args:
        mask_path (Path): Path of the ``*_segmentation.png`` file.
        image_size (int): Square side length the frozen VAEs were trained at.

    Returns:
        torch.Tensor: int64 ``(image_size, image_size)`` class-id tensor,
            nearest-neighbour resized so no blended class ids appear.

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
            (image_size, image_size), resample=MASK_RESAMPLE
        )
        array: np.ndarray = np.array(resized, dtype=np.uint8)
    return torch.from_numpy(array.astype(np.int64))


def _anchor_path(path: str) -> str:
    """Resolve one path against the repository root when it is relative.

    The stages of this project record their checkpoint paths relative to the
    repository root, so anchoring is what makes a recorded path launchable from
    any working directory without changing which file is opened.

    Args:
        path (str): Absolute path (returned unchanged) or a repository-root
            relative path such as ``./mask_VAE/models/....pt``.

    Returns:
        str: Absolute path string.
    """
    candidate: Path = Path(path)
    return str(candidate if candidate.is_absolute() else ROOT_DIR / candidate)


def load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """Load one self-describing mask-diffusion checkpoint.

    Args:
        checkpoint_path (str): Path of the ``mask_diffusion_*_epoch_*.pt`` file.

    Returns:
        dict[str, Any]: The raw checkpoint dictionary, which must hold
            ``model_state_dict``, ``hyperparameters``, ``missing_channel`` and
            the three frozen-VAE paths plus the latent statistics.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the file predates the current checkpoint format.
    """
    path: Path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask diffusion checkpoint not found: {checkpoint_path}")
    checkpoint: dict[str, Any] = torch.load(str(path), map_location="cpu", weights_only=False)
    required: tuple[str, ...] = (
        "model_state_dict",
        "hyperparameters",
        "missing_channel",
        *(f"{key}_checkpoint" for key, _ in FROZEN_VAE_KEYS),
        *STATISTIC_KEYS,
    )
    missing: list[str] = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(
            f"Checkpoint {checkpoint_path} lacks {sorted(missing)}; it predates the "
            "current mask-diffusion checkpoint format."
        )
    return checkpoint


class MaskDiffusionSampler:
    """Frozen three-VAE stack plus a trained denoiser, for evaluation only.

    The class owns nothing that writes: it loads, encodes, samples and decodes.
    Every dimension, the missing component's channel and the layout of each
    frozen VAE come from the checkpoint or the VAE checkpoint itself, so a
    mismatch between what a run trained with and what an evaluation uses is a
    startup error rather than a silent change of meaning.

    Args:
        checkpoint_path (str): Self-describing mask-diffusion checkpoint.
        missing_component_name (str): Name of the component to generate (e.g.
            ``tv``), matched against the target layout's class names.
        class_group_prefix (str): Prefix those class names carry (e.g.
            ``furniture``); ``""`` for an unprefixed color index.
        mask_files (dict[str, str]): Mask filename per space, keyed
            ``floorplan``/``partial``/``target``.
        color_index_dir (Path): Directory searched first for each color index.
        color_index_files (dict[str, str]): Color index filename per space,
            keyed like ``mask_files``.
        device (torch.device): Device the frozen VAEs and the denoiser run on.
        log (SimpleNamespace): Configured project logger.
        fallback_color_dir (Path | None): Dataset root searched when a color
            index is absent from ``color_index_dir``; defaults to the parent of
            the parent of this file (the repository root).
        vae_checkpoint_overrides (dict[str, str] | None): Optional replacement
            VAE checkpoint paths keyed like ``mask_files``. Defaults to the
            paths the training checkpoint recorded, which is what keeps an
            evaluation on exactly the VAE it was trained against.

    Raises:
        FileNotFoundError: If a checkpoint or color index is missing.
        ValueError: If a dimension, the missing channel or the partial layout's
            exclusion disagrees with the checkpoint being loaded.
    """

    def __init__(
        self,
        checkpoint_path: str,
        missing_component_name: str,
        class_group_prefix: str,
        mask_files: dict[str, str],
        color_index_dir: Path,
        color_index_files: dict[str, str],
        device: torch.device,
        log: SimpleNamespace,
        fallback_color_dir: Path | None = None,
        vae_checkpoint_overrides: dict[str, str] | None = None,
    ) -> None:
        self.log: SimpleNamespace = log
        self.device: torch.device = device
        self.missing_component_name: str = str(missing_component_name)
        self.class_group_prefix: str = str(class_group_prefix)
        self.mask_files: dict[str, str] = dict(mask_files)
        self.color_index_files: dict[str, str] = dict(color_index_files)
        self.color_index_dir: Path = Path(color_index_dir)
        self.fallback_color_dir: Path = (
            Path(fallback_color_dir) if fallback_color_dir is not None else ROOT_DIR
        )
        self.checkpoint: dict[str, Any] = load_checkpoint(checkpoint_path)
        self.checkpoint_path: str = str(checkpoint_path)
        self.hyperparams: SimpleNamespace = SimpleNamespace(
            **self.checkpoint["hyperparameters"]
        )

        self.model: MaskComponentDiffusion = self._load_model()
        self.frozen_vaes: dict[str, MaskVAEModel] = {}
        self.frozen_arch: dict[str, SimpleNamespace] = {}
        self._layouts: dict[str, MaskLayout] = {}
        self._load_frozen_vaes(vae_checkpoint_overrides or {})
        self.target_layout: MaskLayout = self._layouts["target"]
        self.target_group: MaskGroup = self.target_layout.groups[0]
        self.partial_layout: MaskLayout = self._layouts["partial"]
        self.floorplan_layout: MaskLayout = self._layouts["floorplan"]
        self.missing_channel: int = self._resolve_missing_channel()
        self.image_size: int = self._validate_dimensions()
        self._restore_statistics()

        self.log.info(
            f"MaskDiffusionSampler ready from {self.checkpoint_path} on {self.device}: "
            f"missing component '{self.target_layout.class_names[self.missing_channel]}' "
            f"= channel {self.missing_channel} of {self.target_layout.num_channels}, "
            f"image size {self.image_size}, timesteps {self.model.num_timesteps}."
        )

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def _load_model(self) -> MaskComponentDiffusion:
        """Rebuild the denoiser and restore the checkpoint's weights.

        Returns:
            MaskComponentDiffusion: Frozen, eval-mode denoiser on ``self.device``.
        """
        model: MaskComponentDiffusion = MaskComponentDiffusion(self.hyperparams).to(
            self.device
        )
        model.load_state_dict(self.checkpoint["model_state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _load_frozen_vaes(self, overrides: dict[str, str]) -> None:
        """Build the three frozen mask VAEs named by the checkpoint.

        Recorded paths are anchored at the repository root when relative, so the
        run is launchable from any working directory and still loads exactly the
        VAE files it was trained against.

        Args:
            overrides (dict[str, str]): Optional replacement checkpoint paths,
                keyed ``floorplan``/``partial``/``target``.

        Raises:
            FileNotFoundError: If a resolved checkpoint or color index is absent.
        """
        self._layouts: dict[str, MaskLayout] = {}
        for key, _config_key in FROZEN_VAE_KEYS:
            recorded: str = str(self.checkpoint[f"{key}_checkpoint"])
            override: str | None = overrides.get(key)
            checkpoint_path: str = _anchor_path(override if override is not None else recorded)
            if override is not None and Path(override) != Path(recorded):
                self.log.info(
                    f"Using the {key} VAE override {checkpoint_path} instead of the "
                    f"checkpoint's recorded {recorded}."
                )
            color_path: Path = resolve_color_index_file(
                self.color_index_files[key], self.color_index_dir, self.fallback_color_dir
            )
            vae, arch, layout = build_frozen_mask_vae(
                checkpoint_path,
                color_path,
                self.mask_files[key],
                self.device,
            )
            self.frozen_vaes[key] = vae
            self.frozen_arch[key] = arch
            self._layouts[key] = layout

    def _resolve_missing_channel(self) -> int:
        """Resolve and verify the missing component's channel in the target space.

        The channel stored in the checkpoint is the verdict of the training run;
        re-deriving it here from the same layout proves the layout being loaded
        is the one that run used, so a stale VAE override cannot silently score
        a different channel.

        Returns:
            int: Channel index of the missing component.

        Raises:
            ValueError: If the re-derived channel differs from the stored one,
                or the partial layout does not exclude the component (which
                would reduce the task to unconditional generation).
        """
        plain_name: str = self.missing_component_name
        prefix: str = self.class_group_prefix
        qualified: str = f"{prefix}/{plain_name}" if prefix else plain_name
        if qualified not in self.target_layout.class_names:
            raise ValueError(
                f"The checkpoint stores hyperparameters naming missing component "
                f"'{qualified}', which is not one of the target layout's classes "
                f"{list(self.target_layout.class_names)}. The checkpoint and the "
                "target VAE do not belong together."
            )
        derived: int = self.target_layout.class_names.index(qualified)
        stored: int = int(self.checkpoint["missing_channel"])
        if derived != stored:
            raise ValueError(
                f"The checkpoint records missing channel {stored} but its layout "
                f"resolves '{qualified}' to channel {derived}. The target VAE at "
                f"{self.checkpoint['target_checkpoint']} is not the one that run "
                "was trained against."
            )
        if not self.partial_layout.excluded_channels:
            raise ValueError(
                f"The partial VAE at {self.checkpoint['partial_checkpoint']} excludes "
                "no class, so its layout would still contain the missing component "
                "and generation would be unconditional."
            )
        if derived not in self.partial_layout.excluded_channels:
            raise ValueError(
                f"The partial VAE at {self.checkpoint['partial_checkpoint']} excludes "
                f"channels {list(self.partial_layout.excluded_channels)}, which does not "
                f"include the missing channel {derived}. The conditioning would still "
                "contain the component."
            )
        return derived

    def _validate_dimensions(self) -> int:
        """Check every width and size against the checkpoint's hyperparameters.

        Returns:
            int: Shared image size, which must be the same for all three VAEs.

        Raises:
            ValueError: If any latent width or image size disagrees.
        """
        image_size: int = int(self.hyperparams.IMAGE_SIZE)
        expected_widths: dict[str, int] = {
            "target": int(self.hyperparams.LATENT_DIM),
            "floorplan": int(self.hyperparams.COND_DIM),
            "partial": int(self.hyperparams.COND_DIM),
        }
        for key, _config_key in FROZEN_VAE_KEYS:
            actual_width: int = int(self.frozen_arch[key].LATENT_DIM)
            if actual_width != expected_widths[key]:
                raise ValueError(
                    f"The {key} VAE has LATENT_DIM {actual_width} but the checkpoint "
                    f"declares {expected_widths[key]}."
                )
            actual_size: int = int(self.frozen_arch[key].IMAGE_SIZE)
            if actual_size != image_size:
                raise ValueError(
                    f"The {key} VAE was trained at IMAGE_SIZE {actual_size} but the "
                    f"checkpoint declares {image_size}."
                )
        return image_size

    def _restore_statistics(self) -> None:
        """Restore the per-dimension latent statistics the checkpoint carries.

        Standardizing with anything but the run's own statistics would shift the
        latent space the denoiser was trained in, so this is a copy of the
        checkpoint tensors rather than a recomputation.

        Raises:
            ValueError: If any stored std is non-positive, which would make
                standardization divide by (near) zero.
        """
        for key in STATISTIC_KEYS:
            tensor: torch.Tensor = self.checkpoint[key].detach().to(torch.float32)
            if key.endswith("_std") and bool((tensor <= 0).any()):
                raise ValueError(
                    f"Checkpoint {self.checkpoint_path} stores a non-positive value "
                    f"in '{key}'; its statistics are unusable for standardization."
                )
            setattr(self, key, tensor)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def load_sample_ids(self, sample_dir: Path) -> dict[str, torch.Tensor]:
        """Load both class-id masks of one sample folder.

        Args:
            sample_dir (Path): Folder holding ``floorplan`` and ``partial``
                (furniture) segmentation masks.

        Returns:
            dict[str, torch.Tensor]: ``floorplan`` and ``furniture`` int64
                ``(S, S)`` class-id tensors. The furniture mask is read once and
                pushed through both layouts, which is exactly what training did.

        Raises:
            FileNotFoundError: If a configured mask file is absent.
        """
        floorplan_path: Path = sample_dir / self.mask_files["floorplan"]
        furniture_path: Path = sample_dir / self.mask_files["partial"]
        for path in (floorplan_path, furniture_path):
            if not path.is_file():
                raise FileNotFoundError(f"Sample {sample_dir} is missing {path}.")
        return {
            "floorplan": load_class_ids(floorplan_path, self.image_size),
            "furniture": load_class_ids(furniture_path, self.image_size),
        }

    def encode_conditions(
        self, sample_dirs: list[Path], batch_size: int = 64
    ) -> dict[str, torch.Tensor]:
        """Encode sample folders into raw (unstandardized) condition latents.

        Only the two encoders that supply conditions are run. The target encoder
        is deliberately not used: an evaluation must not look at the answer.

        Args:
            sample_dirs (list[Path]): Sample folders, in the order the returned
                tensors are laid out.
            batch_size (int): Samples encoded per forward pass; bounds the
                ``(B, 1, S, S)`` encoder input memory.

        Returns:
            dict[str, torch.Tensor]: ``floorplan`` and ``partial`` raw latents
                ``(N, COND_DIM)`` on CPU, plus ``furniture_ids`` and
                ``floorplan_ids`` int64 ``(N, S, S)`` class-id tensors used for
                rendering and for the ground-truth channel.

        Raises:
            FileNotFoundError: If any sample folder lacks a configured mask.
        """
        if not sample_dirs:
            raise ValueError("encode_conditions needs at least one sample folder.")
        batch_size = max(1, int(batch_size))
        raw_floorplan: list[torch.Tensor] = []
        raw_partial: list[torch.Tensor] = []
        furniture_ids: list[torch.Tensor] = []
        floorplan_ids: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(sample_dirs), batch_size):
                batch_dirs: list[Path] = sample_dirs[start : start + batch_size]
                loaded: list[dict[str, torch.Tensor]] = [
                    self.load_sample_ids(sample_dir) for sample_dir in batch_dirs
                ]
                floor_batch: torch.Tensor = torch.stack(
                    [item["floorplan"] for item in loaded]
                ).to(self.device)
                furniture_batch: torch.Tensor = torch.stack(
                    [item["furniture"] for item in loaded]
                ).to(self.device)
                raw_floorplan.append(
                    self._encode_mu(self.frozen_vaes["floorplan"], self.floorplan_layout, floor_batch)
                    .cpu()
                )
                raw_partial.append(
                    self._encode_mu(self.frozen_vaes["partial"], self.partial_layout, furniture_batch)
                    .cpu()
                )
                furniture_ids.append(furniture_batch.cpu())
                floorplan_ids.append(floor_batch.cpu())
        return {
            "floorplan": torch.cat(raw_floorplan),
            "partial": torch.cat(raw_partial),
            "furniture_ids": torch.cat(furniture_ids),
            "floorplan_ids": torch.cat(floorplan_ids),
        }

    def _encode_mu(
        self, vae: MaskVAEModel, layout: MaskLayout, ids: torch.Tensor
    ) -> torch.Tensor:
        """Encode one batch through one frozen encoder, deterministically.

        ``USE_MU_ONLY`` mirrors the training switch: the posterior mean is
        taken so a rerun encodes the same masks to the same latents.

        Args:
            vae (MaskVAEModel): Frozen mask VAE.
            layout (MaskLayout): Layout defining this VAE's exclusions.
            ids (torch.Tensor): ``(B, S, S)`` int64 class ids.

        Returns:
            torch.Tensor: ``(B, LATENT_DIM)`` latents on the VAE's device.
        """
        encoded: torch.Tensor = encode_masks_to_single_channel([ids], layout)
        if bool(self.hyperparams.USE_MU_ONLY):
            return vae.encoder(encoded)[0]
        mu, logvar = vae.encoder(encoded)
        return vae.reparameterize(mu, logvar)

    def normalize_conditions(self, conditions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Standardize raw condition latents with the run's own statistics.

        Args:
            conditions (dict[str, torch.Tensor]): Output of
                :meth:`encode_conditions`.

        Returns:
            dict[str, torch.Tensor]: The same dictionary with ``floorplan`` and
                ``partial`` standardized; ``furniture_ids`` and
                ``floorplan_ids`` pass through untouched.
        """
        normalized: dict[str, torch.Tensor] = dict(conditions)
        normalized["floorplan"] = self._standardize(
            conditions["floorplan"],
            self.floorplan_mean,
            self.floorplan_std,
            bool(self.hyperparams.NORMALIZE_FLOORPLAN),
        )
        normalized["partial"] = self._standardize(
            conditions["partial"],
            self.partial_mean,
            self.partial_std,
            bool(self.hyperparams.NORMALIZE_PARTIAL),
        )
        return normalized

    @staticmethod
    def _standardize(
        latents: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, enabled: bool
    ) -> torch.Tensor:
        """Apply one per-dimension standardization, or return the input.

        Args:
            latents (torch.Tensor): ``(N, D)`` raw latents.
            mean (torch.Tensor): ``(D,)`` per-dimension mean.
            std (torch.Tensor): ``(D,)`` per-dimension std.
            enabled (bool): The run's corresponding ``NORMALIZE_*`` switch.

        Returns:
            torch.Tensor: Standardized latents, or the untouched input.
        """
        if not enabled:
            return latents
        return (latents - mean) / std

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_masks(
        self,
        conditions: dict[str, torch.Tensor],
        num_draws: int,
        seed: int,
        chunk_size: int = 64,
        chain_offset: int = 0,
    ) -> torch.Tensor:
        """Run the full reverse chain for every sample and draw, then decode.

        The whole chain is walked (``num_timesteps`` steps, no truncation), so
        this is the honest end-to-end generation rather than the truncated
        training objective. Each chunk's RNG is seeded from ``seed`` and the
        chunk's absolute chain index, so a rerun with the same arguments
        reproduces the same pixels.

        Args:
            conditions (dict[str, torch.Tensor]): Output of
                :meth:`normalize_conditions`, holding ``floorplan`` and
                ``partial`` latents of shape ``(N, COND_DIM)``.
            num_draws (int): Independent draws per sample, ``>= 1``.
            seed (int): Base seed; the chunk holding absolute chain ``k`` uses
                ``seed + CHUNK_SEED_STRIDE * (chain_offset + k // chunk_size)``.
            chunk_size (int): Chains per reverse-chain call. Fixed in config
                because it fixes the RNG stream; larger is faster per chain
                until the decoder's ``(C, S, S)`` logits dominate memory.
            chain_offset (int): Absolute chain index of the first row of
                ``conditions``. A caller streaming the split in blocks passes
                the block's start so two blocks never share a noise stream.

        Returns:
            torch.Tensor: int64 ``(N * num_draws, S, S)`` decoded class ids in
                sample-major/draw-minor order, i.e. draw ``d`` of every sample
                is the slice ``[d::num_draws]`` - the same convention
                ``MaskDiffusionRunner.sample_and_score`` uses.

        Raises:
            ValueError: If ``num_draws`` or ``chunk_size`` is below one.
        """
        if int(num_draws) < 1:
            raise ValueError(f"num_draws must be >= 1, got {num_draws}.")
        chunk_size = max(1, int(chunk_size))
        floorplan: torch.Tensor = conditions["floorplan"].repeat_interleave(
            int(num_draws), dim=0
        )
        partial: torch.Tensor = conditions["partial"].repeat_interleave(
            int(num_draws), dim=0
        )
        total_chains: int = floorplan.shape[0]
        decoded_ids: list[torch.Tensor] = []
        for start in range(0, total_chains, chunk_size):
            stop: int = min(start + chunk_size, total_chains)
            floor_chunk: torch.Tensor = floorplan[start:stop].to(self.device)
            partial_chunk: torch.Tensor = partial[start:stop].to(self.device)
            generator: torch.Generator = torch.Generator(device=self.device)
            absolute_chunk: int = int(chain_offset) + start // chunk_size
            generator.manual_seed(int(seed) + CHUNK_SEED_STRIDE * absolute_chunk)
            sampled: torch.Tensor = self.model.sample_conditional(
                floor_chunk, partial_chunk, generator
            )
            decoded_ids.append(self.decode_to_ids(sampled).cpu())
        if not decoded_ids:
            raise ValueError("No chains were generated.")
        return torch.cat(decoded_ids)

    @torch.no_grad()
    def decode_to_ids(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode standardized latents into per-pixel class ids.

        Args:
            latents (torch.Tensor): Standardized ``(B, LATENT_DIM)`` latents on
                the model device, as returned by the reverse chain.

        Returns:
            torch.Tensor: int64 ``(B, S, S)`` class ids, one argmax over the
                target decoder's channels per pixel.
        """
        logits: torch.Tensor = self.frozen_vaes["target"].decode(
            self.denormalize_latents(latents)
        )
        return logits.argmax(dim=1)

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Invert target standardization before the frozen decoder is applied.

        Args:
            latents (torch.Tensor): Standardized ``(B, LATENT_DIM)`` latents on
                any device.

        Returns:
            torch.Tensor: Raw-scale latents the target decoder expects.
        """
        if not bool(self.hyperparams.NORMALIZE_LATENTS):
            return latents
        return latents * self.latent_std.to(latents.device) + self.latent_mean.to(
            latents.device
        )

    def generate_channel_masks(
        self,
        conditions: dict[str, torch.Tensor],
        num_draws: int,
        seed: int,
        chunk_size: int = 64,
        chain_offset: int = 0,
    ) -> torch.Tensor:
        """Generate only the missing component's binary mask per draw.

        Args:
            conditions (dict[str, torch.Tensor]): Normalized conditions.
            num_draws (int): Independent draws per sample.
            seed (int): Base seed, see :meth:`generate_masks`.
            chunk_size (int): Chains per reverse-chain call.
            chain_offset (int): Absolute chain index of the first row.

        Returns:
            torch.Tensor: Bool ``(N * num_draws, S, S)``, true where the decoded
                argmax equals the missing component's channel. This is the
                generated component the evaluator composites and scores.
        """
        ids: torch.Tensor = self.generate_masks(
            conditions, num_draws, seed, chunk_size, chain_offset
        )
        return ids == self.missing_channel


def default_device(requested: str, log: SimpleNamespace) -> torch.device:
    """Resolve the configured device string to a usable PyTorch device.

    Args:
        requested (str): One of ``mps``, ``cuda`` or ``cpu``.
        log (SimpleNamespace): Configured project logger for fallback warnings.

    Returns:
        torch.device: Device to run the frozen VAEs and the denoiser on.
    """
    return resolve_device(requested, log)
