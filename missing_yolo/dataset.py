"""Image generation for missing_yolo: frozen mask-diffusion renders of layouts.

The images YOLO26 trains on are produced here and nowhere else. Nothing in this
module derives a detection target: it generates pixels (floorplan render with a
GENERATED arrangement blended over it, missing component included) and copies
the target lines out of the dataset's ``labels.txt`` via :mod:`missing_yolo.labels`.

Two collaborators, each with one job:

* :class:`FrozenMaskDiffusionStack` - the frozen generator. It rebuilds the
  missing-component denoiser and the three frozen mask VAEs **from the
  generator checkpoint itself** (architecture, channel layout, latent
  standardization statistics, and even the VAE file paths), so this stage
  cannot drift away from the exact stack the generator was trained against. It
  encodes the two conditions once per run, walks the full reverse chain per
  epoch, decodes through the frozen arrangement VAE and renders the overlay.
  Nothing here trains: every module is in eval mode with gradients disabled.

* :class:`ArrangementGenerator` - the writer. It turns a generated split into
  PNGs plus label files, measures how far the generated component lands from
  the ``labels.txt`` box (telemetry only - nothing is dropped or relabelled on
  that account) and records provenance in the run's manifest. The live train
  redraw and the one-shot val/test render are the same code path with different
  seeds, so the detector never sees an image distribution it was not trained on.

Splits: the detector trains on ``work/train`` and BOTH validates and tests on
family-disjoint halves of ``work/validate``. ``work/test`` is unused because the
frozen generator trained on it, so it could not yield an honest number.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

# Project root so ``mask_diffusion``/``mask_VAE``/``YOLO``/``evaluator`` import
# from any working directory (``python missing_yolo/run.py`` puts missing_yolo/
# on sys.path).
_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_ROOT_DIR: Path = _SCRIPT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from conditional_diffusion.runner import (
    build_frozen_mask_vae,
    overlay_furniture_on_floorplan,
    resolve_color_index_file,
)
from mask_VAE.masking import (
    MaskGroup,
    MaskLayout,
    encode_masks_to_channels,
    encode_masks_to_single_channel,
    render_group_rgb,
)
from mask_diffusion.model import MaskComponentDiffusion
from missing_yolo.labels import LabelIndex, component_agreement_iou
from VAE.runner import resolve_device
from YOLO.dataset import (
    SampleFolder,
    clear_generated_split,
    load_mask_ids_batch,
    split_is_current,
    write_render_signature,
)

# Sidecar written into a rendered split's image directory recording the content
# signature it was rendered with, so a later run with different stems, a
# different generator or different overlay settings re-renders instead of
# reusing pixels that no longer describe the run.
RENDER_SIGNATURE_FILE: str = ".render_signature"

# The three frozen mask VAEs a missing-component generator is built from, in the
# order their paths are stored in its checkpoint.
_FROZEN_VAE_KEYS: tuple[str, ...] = ("floorplan", "partial", "target")

# Per-dimension standardization statistics the generator checkpoint carries.
_STATISTIC_KEYS: tuple[str, ...] = (
    "latent_mean",
    "latent_std",
    "floorplan_mean",
    "floorplan_std",
    "partial_mean",
    "partial_std",
)

# Keys a self-describing mask_diffusion checkpoint must hold. A checkpoint
# missing any of them predates the current format and would silently sample in
# the wrong latent space.
_CHECKPOINT_KEYS: tuple[str, ...] = (
    "model_state_dict",
    "hyperparameters",
    "missing_channel",
    *(f"{key}_checkpoint" for key in _FROZEN_VAE_KEYS),
    *_STATISTIC_KEYS,
)

# CSV columns of the generation manifest, one row per generation pass.
_MANIFEST_FIELDS: tuple[str, ...] = (
    "split",
    "epoch",
    "seed",
    "images",
    "boxes",
    "agreement_scored",
    "agreement_mean",
    "seconds",
)


# ---------------------------------------------------------------------------
# Component variant and run paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentVariant:
    """One missing component's row of ``config.toml``'s variant table.

    Attributes:
        component (str): Variant key, e.g. ``"dining_set"``. This is what
            ``MISSING_COMPONENT``/``--component`` select.
        mask_diffusion_checkpoint (Path): Absolute path of the frozen
            generator. The checkpoint supplies the three VAE paths, the missing
            channel, the architecture and the latent statistics, so nothing
            else about the generator is configured.
        detector_class_name (str): The single class this run trains.
        label_names (tuple[str, ...]): Names ``labels.txt`` uses for the
            component. The dataset vocabulary differs from the mask vocabulary
            (three ``Dining_table_*`` sizes for one ``dining_set`` channel).
    """

    component: str
    mask_diffusion_checkpoint: Path
    detector_class_name: str
    label_names: tuple[str, ...]


@dataclass(frozen=True)
class ComponentPaths:
    """Filesystem layout of one component run's generated dataset.

    Each run gets its own tree (``<base>_<component>``) so the three component
    runs never overwrite each other's images, labels or manifest.

    Attributes:
        root (Path): Dataset root holding ``images/``, ``labels/`` and
            ``data.yaml``.
        data_yaml (Path): Ultralytics dataset description written from the
            run's single class.
        manifest (Path): Generation provenance CSV, one row per pass.
    """

    root: Path
    data_yaml: Path
    manifest: Path

    def image_dir(self, split: str) -> Path:
        """Return the image directory of one split.

        Args:
            split (str): Split role, ``"train"``/``"val"``/``"test"``.

        Returns:
            Path: ``<root>/images/<split>``.
        """
        return self.root / "images" / split

    def label_dir(self, split: str) -> Path:
        """Return the label directory of one split.

        Args:
            split (str): Split role, ``"train"``/``"val"``/``"test"``.

        Returns:
            Path: ``<root>/labels/<split>``.
        """
        return self.root / "labels" / split


def resolve_component_variant(config: SimpleNamespace, component: str) -> ComponentVariant:
    """Look the run's component row up and validate it, failing fast.

    Args:
        config (SimpleNamespace): Runtime config holding ``COMPONENT_VARIANTS``.
        component (str): Component key to run, e.g. ``"tv"``.

    Returns:
        ComponentVariant: The validated row, with the checkpoint path anchored
            at the repository root.

    Raises:
        KeyError: If ``component`` is not a row of the variant table. The
            message lists the keys that do exist, because the usual cause is a
            typo in ``MISSING_COMPONENT`` or ``--component``.
        ValueError: If the row is incomplete or its checkpoint is not on disk.
    """
    variants: dict[str, dict[str, object]] = dict(config.COMPONENT_VARIANTS)
    if component not in variants:
        raise KeyError(
            f"Unknown missing component {component!r}. "
            f"[COMPONENT_VARIANTS] in missing_yolo/config.toml declares "
            f"{sorted(variants)}."
        )
    row: dict[str, object] = variants[component]
    for key in ("mask_diffusion_checkpoint", "detector_class_name", "label_names"):
        if key not in row:
            raise ValueError(
                f"[COMPONENT_VARIANTS.{component}] in missing_yolo/config.toml is "
                f"missing '{key}'."
            )
    checkpoint: Path = Path(str(row["mask_diffusion_checkpoint"]))
    if not checkpoint.is_absolute():
        checkpoint = _ROOT_DIR / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Generator checkpoint for component '{component}' not found: "
            f"{checkpoint}. The run id in the config path must match the file "
            "on disk exactly."
        )
    class_name: str = str(row["detector_class_name"])
    if not class_name.strip():
        raise ValueError(
            f"[COMPONENT_VARIANTS.{component}].detector_class_name is empty; a "
            "run trains exactly one named class."
        )
    label_names: tuple[str, ...] = tuple(str(name) for name in row["label_names"])  # type: ignore[union-attr]
    return ComponentVariant(
        component=component,
        mask_diffusion_checkpoint=checkpoint,
        detector_class_name=class_name,
        label_names=label_names,
    )


def component_paths(config: SimpleNamespace, variant: ComponentVariant) -> ComponentPaths:
    """Build this run's dataset paths, namespaced by component.

    Args:
        config (SimpleNamespace): Runtime config with ``LIVE_DATASET_BASE_ROOT``.
        variant (ComponentVariant): The run's component row.

    Returns:
        ComponentPaths: Root, ``data.yaml`` and manifest of this run.
    """
    root: Path = Path(str(config.LIVE_DATASET_BASE_ROOT)) / variant.component
    return ComponentPaths(
        root=root,
        data_yaml=root / "data.yaml",
        manifest=root / "redraw_manifest.csv",
    )


# ---------------------------------------------------------------------------
# Split carving
# ---------------------------------------------------------------------------


def layout_family(stem: str) -> str:
    """Return the layout family a sample stem belongs to.

    The dataset stores every augmentation of one layout as its own folder
    (``<hash>_original``, ``<hash>_augmented_001``, ...). Splitting on the stem
    would therefore put augmented siblings of the same layout on both sides of
    a split, which is leakage of the most direct kind.

    Args:
        stem (str): Sample folder name.

    Returns:
        str: The family key, i.e. the stem with its augmentation suffix
            stripped.
    """
    if "_augmented_" in stem:
        return stem.rsplit("_augmented_", 1)[0]
    if stem.endswith("_original"):
        return stem[: -len("_original")]
    return stem


def carve_family_splits(
    sources: list[SampleFolder],
    val_fraction: float,
    seed: int,
) -> tuple[list[SampleFolder], list[SampleFolder]]:
    """Split one source split into family-disjoint val and test halves.

    Families are shuffled with a fixed seed, so the same seed always carves the
    same two halves: the split a run validated on is the split it can be
    re-validated against later. Samples keep their sorted order inside each
    half, which keeps the write order (and therefore the manifest) reproducible.

    Args:
        sources (list[SampleFolder]): Every sample of the source split.
        val_fraction (float): Fraction of FAMILIES assigned to validation; the
            rest becomes test.
        seed (int): Seed of the family shuffle.

    Returns:
        tuple[list[SampleFolder], list[SampleFolder]]: ``(val_sources,
            test_sources)``, both non-empty.

    Raises:
        ValueError: If ``val_fraction`` is outside ``(0, 1)`` or the split
            cannot yield two non-empty halves, which would silently turn one of
            them into a duplicate of the other.
    """
    if not 0.0 < float(val_fraction) < 1.0:
        raise ValueError(
            f"VAL_FAMILY_FRACTION must lie in (0, 1), got {val_fraction}."
        )
    families: list[str] = sorted({layout_family(source.stem) for source in sources})
    rng: random.Random = random.Random(seed)
    rng.shuffle(families)
    val_family_count: int = round(len(families) * float(val_fraction))
    val_family_count = min(max(val_family_count, 1), len(families) - 1)
    val_keys: set[str] = set(families[:val_family_count])
    val_sources: list[SampleFolder] = [
        source for source in sources if layout_family(source.stem) in val_keys
    ]
    test_sources: list[SampleFolder] = [
        source for source in sources if layout_family(source.stem) not in val_keys
    ]
    if not val_sources or not test_sources:
        raise ValueError(
            f"Carving {len(families)} families at VAL_FAMILY_FRACTION="
            f"{val_fraction} produced an empty half (val={len(val_sources)}, "
            f"test={len(test_sources)})."
        )
    return val_sources, test_sources


def require_label_files(sources: list[SampleFolder], label_filename: str) -> None:
    """Fail fast when any sample lacks the dataset's own label file.

    A missing ``labels.txt`` would silently turn a positive sample into a
    background one, and a run of those teaches the detector to miss the
    component. This check runs before the expensive generation instead.

    Args:
        sources (list[SampleFolder]): Samples about to be indexed for labels.
        label_filename (str): Label filename expected in every sample folder.

    Raises:
        FileNotFoundError: If at least one sample has no label file; the
            message names the first few offenders.
    """
    missing: list[str] = [
        source.stem
        for source in sources
        if not (source.src_dir / label_filename).is_file()
    ]
    if missing:
        shown: str = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} of {len(sources)} samples have no "
            f"'{label_filename}', starting with {shown}. Every sample must ship "
            "the dataset's own boxes; a sample without them cannot be labelled."
        )


# ---------------------------------------------------------------------------
# Render signature for the one-shot splits
# ---------------------------------------------------------------------------


def render_signature(
    sources: list[SampleFolder],
    variant: ComponentVariant,
    overlay_alpha: float,
    image_size: int,
    seed: int,
) -> str:
    """Build a content fingerprint describing what a one-shot render contains.

    Stems alone are not enough: pointing the run at a different generator, a
    different component's class name or a different seed changes every pixel
    while leaving the file names untouched, and the run would then validate
    against pixels from another configuration. The generator's size and
    modification time are part of the digest, so retraining the generator
    invalidates the render too.

    Args:
        sources (list[SampleFolder]): Samples the split must contain.
        variant (ComponentVariant): The run's component row.
        overlay_alpha (float): Overlay blend weight used for rendering.
        image_size (int): Rendered square side length.
        seed (int): Seed the split was drawn with.

    Returns:
        str: Stable hex digest of every input that affects render content.
    """
    checkpoint: Path = variant.mask_diffusion_checkpoint
    stat: os.stat_result = checkpoint.stat()
    payload: dict[str, object] = {
        "stems": sorted(source.stem for source in sources),
        "component": variant.component,
        "detector_class_name": variant.detector_class_name,
        "label_names": list(variant.label_names),
        "generator": str(checkpoint),
        "generator_size": int(stat.st_size),
        "generator_mtime_ns": int(stat.st_mtime_ns),
        "overlay_alpha": round(float(overlay_alpha), 6),
        "image_size": int(image_size),
        "seed": int(seed),
    }
    encoded: str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Frozen generator stack
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionCache:
    """Standardized generator conditions for a whole split, held on CPU.

    Attributes:
        floorplan (torch.Tensor): ``(N, cond_dim)`` standardized floorplan
            latents, in the order of the split's sources.
        partial (torch.Tensor): ``(N, cond_dim)`` standardized partial-layout
            latents (the same arrangement mask, with the component erased by
            the partial VAE's layout).
    """

    floorplan: torch.Tensor
    partial: torch.Tensor

    def __len__(self) -> int:
        """Return how many samples the cache covers.

        Returns:
            int: Number of cached condition rows.
        """
        return int(self.floorplan.shape[0])


def _load_generator_checkpoint(path: Path) -> dict[str, object]:
    """Load and validate one self-describing mask-diffusion checkpoint.

    A local loader rather than ``mask_diffusion.inference.load_checkpoint``:
    that module is the shared evaluation API, which this training stage must
    not depend on, and the key list it validates is the one reproduced here.

    Args:
        path (Path): Absolute path of the ``mask_diffusion_*_epoch_*.pt`` file.

    Returns:
        dict[str, object]: The checkpoint dictionary.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        KeyError: If it predates the current checkpoint format.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Generator checkpoint not found: {path}")
    checkpoint: dict[str, object] = torch.load(
        str(path), map_location="cpu", weights_only=False
    )
    missing: list[str] = [key for key in _CHECKPOINT_KEYS if key not in checkpoint]
    if missing:
        raise KeyError(
            f"Generator checkpoint {path} lacks {sorted(missing)}; it predates "
            "the current mask-diffusion checkpoint format."
        )
    return checkpoint


class FrozenMaskDiffusionStack:
    """Frozen missing-component generator: denoiser + three frozen mask VAEs.

    Everything is rebuilt from the generator checkpoint, never from this stage's
    TOMLs: the denoiser's architecture, the three VAE file paths, the layouts
    (including which class the partial VAE erases) and the latent
    standardization statistics. Only the render knobs (image size, overlay
    alpha, chunk sizes, seed stride) and the device come from
    ``missing_yolo/config.toml`` / ``missing_yolo/hyperparameters.toml``.

    Every module is in eval mode with gradients disabled; this class only ever
    runs forwards.
    """

    def __init__(
        self,
        config: SimpleNamespace,
        hyperparams: SimpleNamespace,
        variant: ComponentVariant,
        log: SimpleNamespace,
    ) -> None:
        """Load the frozen generator and validate it against this run's knobs.

        Args:
            config (SimpleNamespace): Runtime config with device, mask
                filenames and color index locations.
            hyperparams (SimpleNamespace): Render knobs and ``CHAIN_SEED_STRIDE``.
            variant (ComponentVariant): The run's component row.
            log (SimpleNamespace): Configured project logger.

        Raises:
            FileNotFoundError: If the generator checkpoint, a recorded VAE
                checkpoint or a color index JSON is missing.
            ValueError: On any disagreement between the checkpoint, the three
                VAEs and this run's render settings - including a partial VAE
                that does not actually erase the component, which would reduce
                the task to unconditional generation without any error.
        """
        self.log: SimpleNamespace = log
        self.variant: ComponentVariant = variant
        self.device: torch.device = resolve_device(str(config.DEVICE), log)
        self.image_size: int = int(hyperparams.RENDER_IMAGE_SIZE)
        self.overlay_alpha: float = float(hyperparams.RENDER_OVERLAY_ALPHA)
        self.chain_chunk: int = max(1, int(hyperparams.RENDER_CHAIN_CHUNK))
        self.decode_chunk: int = max(1, int(hyperparams.RENDER_DECODE_CHUNK))
        self.io_workers: int = max(1, int(hyperparams.RENDER_IO_WORKERS))
        self.png_compress_level: int = int(hyperparams.RENDER_PNG_COMPRESS_LEVEL)
        self.use_mu_only: bool = bool(hyperparams.RENDER_USE_MU_ONLY)
        self.chain_seed_stride: int = int(hyperparams.CHAIN_SEED_STRIDE)
        self.base_seed: int = int(hyperparams.BASE_SEED)

        self.checkpoint_path: Path = variant.mask_diffusion_checkpoint
        self.checkpoint: dict[str, object] = _load_generator_checkpoint(self.checkpoint_path)
        self.hyperparams: SimpleNamespace = SimpleNamespace(
            **dict(self.checkpoint["hyperparameters"])  # type: ignore[arg-type]
        )
        (
            (self.floorplan_vae, self.floorplan_arch, self.floorplan_layout),
            (self.partial_vae, self.partial_arch, self.partial_layout),
            (self.target_vae, self.target_arch, self.target_layout),
        ) = self._load_frozen_vaes(config)
        self.floorplan_group: MaskGroup = self.floorplan_layout.groups[0]
        self.target_group: MaskGroup = self.target_layout.groups[0]
        self.target_channel_index: torch.Tensor = torch.tensor(
            self.target_group.channel_indices, dtype=torch.long, device=self.device
        )
        self.missing_channel: int = self._resolve_missing_channel()
        self._validate_frozen_consistency()

        self.model: MaskComponentDiffusion = self._load_denoiser()
        self._restore_statistics()
        self.log.info(
            f"Frozen generator ready on {self.device}: {self.checkpoint_path.name} "
            f"-> missing component "
            f"'{self.target_layout.class_names[self.missing_channel]}' "
            f"(channel {self.missing_channel} of {self.target_layout.num_channels}), "
            f"image size {self.image_size}, "
            f"{int(self.hyperparams.NUM_TIMESTEPS)} reverse steps, class "
            f"'{self.variant.detector_class_name}'."
        )

    # -- startup -------------------------------------------------------------

    def _load_frozen_vaes(
        self, config: SimpleNamespace
    ) -> tuple[tuple[object, SimpleNamespace, MaskLayout], ...]:
        """Build the three frozen mask VAEs the generator was trained with.

        The file paths come from the checkpoint, so a ``mask_VAE`` config that
        moved on cannot silently change this stage's latent spaces. Relative
        paths inside a checkpoint are anchored at the repository root.

        Args:
            config (SimpleNamespace): Runtime config with mask filenames and
                color index locations.

        Returns:
            tuple[tuple[object, SimpleNamespace, MaskLayout], ...]: One
                ``(vae, archparams, layout)`` per key of :data:`_FROZEN_VAE_KEYS`,
                in that order.

        Raises:
            FileNotFoundError: If a recorded checkpoint or color index is
                missing.
        """
        color_dir: Path = Path(str(config.MASK_COLOR_INDEX_DIR))
        fallback_dir: Path = Path(str(config.SOURCE_TRAIN_SPLIT_DIR)).parent
        mask_files: dict[str, str] = {
            "floorplan": str(config.FLOORPLAN_MASK_FILE),
            # The partial and the target encoder see the SAME arrangement mask;
            # the component is removed by the partial layout, not by editing
            # pixels here.
            "partial": str(config.FURNITURE_MASK_FILE),
            "target": str(config.FURNITURE_MASK_FILE),
        }
        color_files: dict[str, str] = {
            "floorplan": str(config.FLOORPLAN_COLOR_INDEX_FILE),
            "partial": str(config.FURNITURE_COLOR_INDEX_FILE),
            "target": str(config.FURNITURE_COLOR_INDEX_FILE),
        }
        built: list[tuple[object, SimpleNamespace, MaskLayout]] = []
        for key in _FROZEN_VAE_KEYS:
            checkpoint_path: Path = Path(
                str(self.checkpoint[f"{key}_checkpoint"])
            )
            if not checkpoint_path.is_absolute():
                checkpoint_path = _ROOT_DIR / checkpoint_path
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Generator checkpoint {self.checkpoint_path.name} points its "
                    f"{key} VAE at {checkpoint_path}, which does not exist. The "
                    "recorded run id must match the file on disk exactly."
                )
            built.append(
                build_frozen_mask_vae(
                    str(checkpoint_path),
                    resolve_color_index_file(color_files[key], color_dir, fallback_dir),
                    mask_files[key],
                    self.device,
                )
            )
        return tuple(built)

    def _load_denoiser(self) -> MaskComponentDiffusion:
        """Rebuild the denoiser and restore the generator's weights.

        Returns:
            MaskComponentDiffusion: Frozen denoiser in eval mode on the run's
                device.
        """
        model: MaskComponentDiffusion = MaskComponentDiffusion(self.hyperparams).to(
            self.device
        )
        model.load_state_dict(self.checkpoint["model_state_dict"])  # type: ignore[arg-type]
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _restore_statistics(self) -> None:
        """Copy the generator's per-dimension standardization statistics.

        Sampling with anything but the run's own statistics would move the
        latent space the denoiser was trained in, so these tensors are copied
        from the checkpoint rather than recomputed.

        Raises:
            ValueError: If a stored standard deviation is non-positive, which
                would make standardization divide by (near) zero.
        """
        for key in _STATISTIC_KEYS:
            tensor: torch.Tensor = self.checkpoint[key].detach().to(torch.float32)  # type: ignore[union-attr]
            if key.endswith("_std") and bool((tensor <= 0).any()):
                raise ValueError(
                    f"Generator checkpoint {self.checkpoint_path} stores a "
                    f"non-positive value in '{key}'; its statistics are "
                    "unusable for standardization."
                )
            setattr(self, key, tensor)

    def _resolve_missing_channel(self) -> int:
        """Resolve the component's channel by NAME and cross-check the checkpoint.

        The channel is never taken from configuration: the target layout's class
        names are searched for the variant's component, and the index found must
        equal the one the checkpoint recorded. A mismatch means the variant row
        and the generator disagree about what is missing, which would produce a
        detector for a component the images do not contain.

        Returns:
            int: Index of the component's channel in the target decoder.

        Raises:
            ValueError: If the component is absent from the target layout, or
                the recorded channel disagrees with the resolved one.
        """
        names: tuple[str, ...] = tuple(self.target_layout.class_names)
        matches: list[int] = [
            index
            for index, name in enumerate(names)
            if name.rsplit("/", 1)[-1] == self.variant.component
        ]
        if not matches:
            raise ValueError(
                f"Component '{self.variant.component}' is not one of the target "
                f"VAE's classes {list(names)} (generator "
                f"{self.checkpoint_path.name}). Check MISSING_COMPONENT and the "
                "[COMPONENT_VARIANTS] row in missing_yolo/config.toml."
            )
        recorded: int = int(self.checkpoint["missing_channel"])  # type: ignore[arg-type]
        if recorded != matches[0]:
            raise ValueError(
                f"Generator {self.checkpoint_path.name} recorded the missing "
                f"component at channel {recorded} ({names[recorded]}), but "
                f"'{self.variant.component}' is channel {matches[0]} "
                f"({names[matches[0]]}). Point the variant row at the generator "
                "that actually generates this component."
            )
        return matches[0]

    def _validate_frozen_consistency(self) -> None:
        """Fail fast on any disagreement inside the frozen stack.

        Raises:
            ValueError: If the three VAEs disagree on image size or latent
                width, if the render size does not match them, if the target
                VAE excludes classes (it must be able to DRAW the component),
                if the partial VAE erases nothing, or if it erases something
                other than the component.
        """
        sizes: dict[str, int] = {
            "floorplan": int(self.floorplan_arch.IMAGE_SIZE),
            "partial": int(self.partial_arch.IMAGE_SIZE),
            "target": int(self.target_arch.IMAGE_SIZE),
        }
        if len(set(sizes.values())) != 1:
            raise ValueError(f"Frozen VAEs disagree on IMAGE_SIZE: {sizes}.")
        if next(iter(sizes.values())) != self.image_size:
            raise ValueError(
                f"RENDER_IMAGE_SIZE ({self.image_size}) does not match the frozen "
                f"VAEs ({next(iter(sizes.values()))})."
            )
        cond_dim: int = int(self.hyperparams.COND_DIM)
        for key, arch in (
            ("floorplan", self.floorplan_arch),
            ("partial", self.partial_arch),
        ):
            if int(arch.LATENT_DIM) != cond_dim:
                raise ValueError(
                    f"The {key} VAE has LATENT_DIM {int(arch.LATENT_DIM)} but the "
                    f"generator expects COND_DIM {cond_dim}."
                )
        target_dim: int = int(self.target_arch.LATENT_DIM)
        if target_dim != int(self.hyperparams.LATENT_DIM):
            raise ValueError(
                f"The target VAE has LATENT_DIM {target_dim} but the generator "
                f"expects LATENT_DIM {int(self.hyperparams.LATENT_DIM)}."
            )
        if self.target_layout.excluded_channels:
            raise ValueError(
                f"The target VAE excludes channels {self.target_layout.excluded_channels}; "
                "a filtered arrangement could never draw the component the "
                "detector is being trained to find."
            )
        erased: tuple[int, ...] = tuple(self.partial_layout.excluded_channels)
        if not erased:
            raise ValueError(
                "The partial VAE excludes no class, so its condition would still "
                "contain the component and the generator would be doing "
                "reconstruction instead of generation. Point the generator at a "
                "filtered checkpoint."
            )
        if erased != (self.missing_channel,):
            raise ValueError(
                f"The partial VAE erases channels {erased}, but the missing "
                f"component is channel {self.missing_channel}; the condition "
                "would differ from the target in some other class too."
            )

    # -- encoding ------------------------------------------------------------

    @staticmethod
    def _encoder_input(vae: object, layout: MaskLayout, ids: torch.Tensor) -> torch.Tensor:
        """Build the encoder input one VAE expects from its class-id masks.

        Newer mask VAEs read one normalized id channel per group; the legacy
        ones read the one-vs-rest binary channels. The checkpoint's own
        architecture decides which, so both stay loadable.

        Args:
            vae (object): Frozen mask VAE the input is built for.
            layout (MaskLayout): Layout of ``vae``.
            ids (torch.Tensor): int64 ``(B, S, S)`` class-id masks.

        Returns:
            torch.Tensor: Float encoder input on CPU.
        """
        if str(getattr(vae, "encoder_kind")) == "single":
            return encode_masks_to_single_channel([ids], layout)
        return encode_masks_to_channels([ids], layout)

    def load_floorplan_ids(
        self, sources: list[SampleFolder], pool: ThreadPoolExecutor
    ) -> torch.Tensor:
        """Load GT floorplan class-id masks for a chunk of sources.

        Args:
            sources (list[SampleFolder]): Sources to load (order preserved).
            pool (ThreadPoolExecutor): Worker pool doing the PNG loads.

        Returns:
            torch.Tensor: int64 ``(B, S, S)`` floorplan class-id tensor.
        """
        return load_mask_ids_batch(
            [source.src_dir / self.floorplan_group.mask_file_name for source in sources],
            self.image_size,
            pool,
        )

    def load_arrangement_ids(
        self, sources: list[SampleFolder], pool: ThreadPoolExecutor
    ) -> torch.Tensor:
        """Load GT arrangement class-id masks for a chunk of sources.

        Args:
            sources (list[SampleFolder]): Sources to load (order preserved).
            pool (ThreadPoolExecutor): Worker pool doing the PNG loads.

        Returns:
            torch.Tensor: int64 ``(B, S, S)`` arrangement class-id tensor.
        """
        return load_mask_ids_batch(
            [source.src_dir / self.target_group.mask_file_name for source in sources],
            self.image_size,
            pool,
        )

    @torch.no_grad()
    def encode_conditions(self, sources: list[SampleFolder]) -> ConditionCache:
        """Encode both generator conditions for a whole split, once per run.

        The floorplan is the spatial condition and the arrangement mask, encoded
        through the partial VAE's layout, is the "what is already there"
        condition with the component genuinely absent. Both conditions are
        derived from GT masks that never change, so they are cached once and
        only the diffusion noise varies per epoch.

        Masks are streamed one chunk at a time (reloaded from disk per chunk),
        so even the full pool never holds more than ``decode_chunk`` masks in
        memory; only the ``cond_dim`` float latents per image are kept.

        Args:
            sources (list[SampleFolder]): Split's samples, in image write order.

        Returns:
            ConditionCache: Standardized conditions on CPU, aligned with
                ``sources``.

        Raises:
            ValueError: If ``sources`` is empty.
        """
        if not sources:
            raise ValueError("Cannot encode conditions with no sources.")
        floor_rows: list[torch.Tensor] = []
        partial_rows: list[torch.Tensor] = []
        with ThreadPoolExecutor(max_workers=self.io_workers) as pool:
            for start in range(0, len(sources), self.decode_chunk):
                chunk: list[SampleFolder] = sources[start : start + self.decode_chunk]
                floor_ids: torch.Tensor = self.load_floorplan_ids(chunk, pool)
                arrangement_ids: torch.Tensor = self.load_arrangement_ids(chunk, pool)
                floor_mu: torch.Tensor = self._encode_mu(
                    self.floorplan_vae, self.floorplan_layout, floor_ids
                )
                partial_mu: torch.Tensor = self._encode_mu(
                    self.partial_vae, self.partial_layout, arrangement_ids
                )
                floor_rows.append(self._standardize(floor_mu, "floorplan"))
                partial_rows.append(self._standardize(partial_mu, "partial"))
                done: int = min(start + self.decode_chunk, len(sources))
                if done == len(sources) or (start // self.decode_chunk) % 20 == 0:
                    self.log.info(f"Condition encoding: {done}/{len(sources)}.")
        cache: ConditionCache = ConditionCache(
            floorplan=torch.cat(floor_rows, dim=0),
            partial=torch.cat(partial_rows, dim=0),
        )
        self.log.info(
            f"Cached {len(cache)} condition pairs "
            f"({int(cache.floorplan.shape[1])}-d each)."
        )
        return cache

    @torch.no_grad()
    def _encode_mu(
        self, vae: object, layout: MaskLayout, ids: torch.Tensor
    ) -> torch.Tensor:
        """Encode class-id masks into one latent per sample.

        Args:
            vae (object): Frozen mask VAE to encode with.
            layout (MaskLayout): Layout matching ``vae``.
            ids (torch.Tensor): int64 ``(B, S, S)`` class-id masks.

        Returns:
            torch.Tensor: ``(B, latent_dim)`` latents on CPU - the encoder mean,
                or a reparameterized sample when ``RENDER_USE_MU_ONLY`` is off.
        """
        encoder_input: torch.Tensor = self._encoder_input(vae, layout, ids).to(
            self.device
        )
        mu, logvar = vae.encoder(encoder_input)  # type: ignore[attr-defined]
        latents: torch.Tensor = (
            mu
            if self.use_mu_only
            else vae.reparameterize(mu, logvar)  # type: ignore[attr-defined]
        )
        return latents.cpu()

    def _standardize(self, latents: torch.Tensor, prefix: str) -> torch.Tensor:
        """Standardize latents with one condition's stored statistics.

        Args:
            latents (torch.Tensor): Raw latents ``(N, latent_dim)`` on CPU.
            prefix (str): ``"floorplan"`` or ``"partial"``, selecting the
                ``<prefix>_mean``/``<prefix>_std`` statistics and the matching
                normalization switch of the generator run.

        Returns:
            torch.Tensor: Standardized latents, same shape.

        Raises:
            ValueError: If ``prefix`` names no known condition, which would
                silently standardize with the wrong statistics.
        """
        if prefix not in ("floorplan", "partial"):
            raise ValueError(
                f"Unknown condition prefix {prefix!r}; expected 'floorplan' or "
                "'partial'."
            )
        if not bool(getattr(self.hyperparams, f"NORMALIZE_{prefix.upper()}", True)):
            return latents
        mean: torch.Tensor = getattr(self, f"{prefix}_mean")
        std: torch.Tensor = getattr(self, f"{prefix}_std")
        return (latents - mean) / std

    # -- generation ----------------------------------------------------------

    @torch.no_grad()
    def sample_latents(self, cache: ConditionCache, seed: int) -> torch.Tensor:
        """Walk the full reverse chain once per condition, seeded for a draw.

        Each chunk gets its own device generator seeded from ``seed +
        CHAIN_SEED_STRIDE * chunk_index``, so the same seed reproduces the same
        pixels no matter how the split was chunked, and two chunks can never
        share a noise stream.

        Args:
            cache (ConditionCache): Standardized conditions on CPU.
            seed (int): Base seed of this draw.

        Returns:
            torch.Tensor: Standardized arrangement latents on CPU, shape
                ``(N, LATENT_DIM)``.

        Raises:
            ValueError: If the cache is empty or its two conditions disagree on
                how many samples they cover.
        """
        if len(cache) == 0:
            raise ValueError("Cannot sample with no cached conditions.")
        if cache.floorplan.shape[0] != cache.partial.shape[0]:
            raise ValueError(
                f"Condition cache is misaligned: floorplan has "
                f"{cache.floorplan.shape[0]} rows, partial has "
                f"{cache.partial.shape[0]}."
            )
        chunks: list[torch.Tensor] = []
        for chunk_index, start in enumerate(range(0, len(cache), self.chain_chunk)):
            floor_chunk: torch.Tensor = cache.floorplan[
                start : start + self.chain_chunk
            ].to(self.device)
            partial_chunk: torch.Tensor = cache.partial[
                start : start + self.chain_chunk
            ].to(self.device)
            generator: torch.Generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed) + self.chain_seed_stride * chunk_index)
            latents: torch.Tensor = self.model.sample_conditional(
                floor_chunk, partial_chunk, generator
            )
            chunks.append(latents.cpu())
            done: int = min(start + self.chain_chunk, len(cache))
            self.log.info(
                f"Reverse chain: {done}/{len(cache)} samples (seed={seed})."
            )
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Invert the target standardization before decoding.

        Args:
            latents (torch.Tensor): Standardized latents ``(N, LATENT_DIM)``,
                on any device.

        Returns:
            torch.Tensor: Raw-scale latents the arrangement decoder expects.
        """
        if not bool(self.hyperparams.NORMALIZE_LATENTS):
            return latents
        return latents * self.latent_std.to(latents.device) + self.latent_mean.to(
            latents.device
        )

    @torch.no_grad()
    def render_overlay_chunk(
        self,
        floor_ids: torch.Tensor,
        arrangement_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend one chunk of generated arrangements over their floorplan base.

        The same code path the mask_diffusion and YOLO grids use: both renders
        come from the dataset's authoritative color tables and only furniture
        foreground pixels are blended, with the rooms and walls showing through.

        Args:
            floor_ids (torch.Tensor): int64 GT floorplan class ids ``(B, S, S)``.
            arrangement_logits (torch.Tensor): decoder logits
                ``(B, C, S, S)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: uint8 ``(B, 3, S, S)`` overlay on
                CPU in ``0..255`` and the int64 ``(B, S, S)`` generated class
                ids (0 = background).

        Raises:
            ValueError: If the two batches disagree on size.
        """
        if floor_ids.shape[0] != arrangement_logits.shape[0]:
            raise ValueError(
                f"Floorplan batch ({floor_ids.shape[0]}) and arrangement batch "
                f"({arrangement_logits.shape[0]}) must match."
            )
        floor_onehot: torch.Tensor = encode_masks_to_channels(
            [floor_ids], self.floorplan_layout
        ).to(self.device)
        logits: torch.Tensor = arrangement_logits.to(self.device)
        floor_rgb: torch.Tensor = render_group_rgb(
            floor_onehot, self.floorplan_group, self.device
        )
        furniture_rgb: torch.Tensor = render_group_rgb(
            logits, self.target_group, self.device
        )
        # The target VAE excludes nothing (validated at startup), so this argmax
        # is exactly the class-id map render_group_rgb painted from.
        arrangement_ids: torch.Tensor = logits.index_select(
            1, self.target_channel_index
        ).argmax(dim=1)
        overlay: torch.Tensor = overlay_furniture_on_floorplan(
            floor_rgb, furniture_rgb, arrangement_ids, self.overlay_alpha
        )
        overlay_u8: torch.Tensor = (
            (overlay.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu()
        )
        return overlay_u8, arrangement_ids.cpu()

    def component_mask(self, arrangement_ids: torch.Tensor) -> np.ndarray:
        """Extract the generated component's pixels from a chunk of class ids.

        Args:
            arrangement_ids (torch.Tensor): int64 ``(B, S, S)`` generated class
                ids.

        Returns:
            np.ndarray: Bool ``(B, S, S)`` array, true where the generator put
                the missing component.

        Raises:
            ValueError: If the class ids carry a channel the layout does not
                know, which would mean the decode and the layout disagree.
        """
        channel: int = int(arrangement_ids.max().item())
        if channel >= self.target_layout.num_channels:
            raise ValueError(
                f"Decoded class id {channel} is outside the target layout's "
                f"{self.target_layout.num_channels} classes."
            )
        return (arrangement_ids == self.missing_channel).numpy()


# ---------------------------------------------------------------------------
# Split writer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationStats:
    """Provenance and agreement record of one generation pass.

    Attributes:
        split (str): Split the pass wrote ("train"/"val"/"test").
        epoch (int): Draw index; ``0`` for the pre-training train draw and for
            the one-shot val/test renders.
        seed (int): Seed the reverse chains were seeded with.
        images (int): Images written.
        boxes (int): Label lines written across those images (background
            samples contribute none).
        seconds (float): Wall-clock cost of the pass.
        agreement_scored (int): Samples whose agreement could be measured, i.e.
            positives the generator drew something for.
        agreement_mean (float | None): Mean box-IoU between the generated
            component and the ``labels.txt`` box, or None when no sample could
            be scored.
        agreement_fractions (dict[str, float]): Fraction of scored samples above
            each configured IoU threshold, keyed by the threshold's text.
    """

    split: str
    epoch: int
    seed: int
    images: int
    boxes: int
    seconds: float
    agreement_scored: int
    agreement_mean: float | None
    agreement_fractions: dict[str, float]


def _write_row(
    overlay_u8: torch.Tensor,
    source: SampleFolder,
    image_dir: Path,
    label_dir: Path,
    lines: list[str],
    compress_level: int,
) -> None:
    """Write one generated PNG and its label file.

    Runs in a save-worker thread: only file encoding and I/O happen here, so
    concurrent calls are safe as long as each owns its own tensors and paths.

    Args:
        overlay_u8 (torch.Tensor): uint8 ``(3, S, S)`` overlay image.
        source (SampleFolder): Source whose stem names the output files.
        image_dir (Path): Destination images directory.
        label_dir (Path): Destination labels directory.
        lines (list[str]): Label lines from the dataset's ``labels.txt``; empty
            for a background sample, which gets an empty label file.
        compress_level (int): PNG zlib level.
    """
    image_hwc: np.ndarray = overlay_u8.permute(1, 2, 0).numpy()
    Image.fromarray(image_hwc, mode="RGB").save(
        str(image_dir / f"{source.stem}.png"), compress_level=compress_level
    )
    (label_dir / f"{source.stem}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )


class ArrangementGenerator:
    """Writes generated arrangements of a split as PNGs plus label files.

    The live train redraw and the one-shot val/test render share this one code
    path; only the seed and the signature gate differ. The labels written are
    always the dataset's ``labels.txt`` lines for the component - fixed per
    stem, which is why the label files never need rewriting while the pixels
    are rewritten every epoch, and why Ultralytics' once-read labels stay
    correct without any in-memory patching.
    """

    def __init__(
        self,
        stack: FrozenMaskDiffusionStack,
        label_index: LabelIndex,
        hyperparams: SimpleNamespace,
        log: SimpleNamespace,
    ) -> None:
        """Bind the frozen stack, the split's targets and the render knobs.

        Args:
            stack (FrozenMaskDiffusionStack): Frozen generator + renderer.
            label_index (LabelIndex): ``labels.txt`` targets of the split being
                written; every stem written must be present in it.
            hyperparams (SimpleNamespace): Render knobs and
                ``AGREEMENT_BOX_IOU_THRESHOLDS``.
            log (SimpleNamespace): Configured project logger.

        Raises:
            ValueError: If an agreement threshold is outside ``0..1``, which
                would make the reported fractions meaningless.
        """
        self.stack: FrozenMaskDiffusionStack = stack
        self.labels: LabelIndex = label_index
        self.workers: int = max(1, int(hyperparams.RENDER_IO_WORKERS))
        self.compress_level: int = int(hyperparams.RENDER_PNG_COMPRESS_LEVEL)
        self.base_seed: int = int(hyperparams.BASE_SEED)
        thresholds: list[float] = [
            float(value)
            for value in hyperparams.AGREEMENT_BOX_IOU_THRESHOLDS
        ]
        for threshold in thresholds:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "AGREEMENT_BOX_IOU_THRESHOLDS must hold values in 0..1, got "
                    f"{threshold}."
                )
        self.thresholds: tuple[float, ...] = tuple(thresholds)
        self.log: SimpleNamespace = log

    def redraw_train_epoch(
        self,
        sources: list[SampleFolder],
        cache: ConditionCache,
        epoch: int,
        image_dir: Path,
        label_dir: Path,
        manifest_path: Path,
    ) -> GenerationStats:
        """Redraw every train image from a fresh draw of the next epoch.

        Args:
            sources (list[SampleFolder]): The run's train sources (fixed stems).
            cache (ConditionCache): Cached conditions aligned with ``sources``.
            epoch (int): 0-based draw index; ``seed = BASE_SEED + epoch``.
            image_dir (Path): Live train images directory.
            label_dir (Path): Live train labels directory.
            manifest_path (Path): Provenance CSV appended with one row.

        Returns:
            GenerationStats: Provenance and agreement of the redraw.
        """
        return self.generate_split(
            sources=sources,
            cache=cache,
            seed=self.base_seed + int(epoch),
            split="train",
            epoch=int(epoch),
            image_dir=image_dir,
            label_dir=label_dir,
            manifest_path=manifest_path,
        )

    def render_split_once(
        self,
        sources: list[SampleFolder],
        cache: ConditionCache,
        seed: int,
        split: str,
        image_dir: Path,
        label_dir: Path,
        manifest_path: Path,
    ) -> GenerationStats | None:
        """Render a one-shot split, skipping a render that is already current.

        Val and test are drawn exactly once, so their pixels - and therefore
        every metric read off them - stay stable for the whole run. The mirror
        is reused only when the recorded content signature still matches (same
        stems, generator, component, overlay settings and seed); anything else
        re-renders rather than silently grading against stale pixels.

        Args:
            sources (list[SampleFolder]): Split sources to render.
            cache (ConditionCache): Cached conditions aligned with ``sources``.
            seed (int): Fixed seed this split is always drawn with.
            split (str): Split role name for logs and the manifest.
            image_dir (Path): Destination images directory.
            label_dir (Path): Destination labels directory.
            manifest_path (Path): Provenance CSV appended with one row.

        Returns:
            GenerationStats | None: The pass's record, or None when an existing
                render was already current and nothing was drawn.

        Raises:
            ValueError: If ``split`` is not a one-shot split.
        """
        if split == "train":
            raise ValueError(
                "The train split is redrawn every epoch; use redraw_train_epoch "
                "so the draw is seeded and logged as such."
            )
        signature: str = render_signature(
            sources,
            self.stack.variant,
            self.stack.overlay_alpha,
            self.stack.image_size,
            seed,
        )
        signature_path: Path = image_dir / RENDER_SIGNATURE_FILE
        if split_is_current(image_dir, sources, signature_path, signature):
            self.log.info(
                f"{split} render already current in {image_dir} "
                f"({len(sources)} images); skipping generation."
            )
            return None
        stats: GenerationStats = self.generate_split(
            sources=sources,
            cache=cache,
            seed=seed,
            split=split,
            epoch=0,
            image_dir=image_dir,
            label_dir=label_dir,
            manifest_path=manifest_path,
        )
        write_render_signature(signature_path, signature)
        return stats

    def generate_split(
        self,
        sources: list[SampleFolder],
        cache: ConditionCache,
        seed: int,
        split: str,
        epoch: int,
        image_dir: Path,
        label_dir: Path,
        manifest_path: Path,
    ) -> GenerationStats:
        """Generate one split's images and write their fixed labels.

        This is the pass every other entry point delegates to: sample the full
        reverse chain once per sample, decode, blend over the floorplan render,
        write the PNG, and copy the ``labels.txt`` lines across unchanged.

        Args:
            sources (list[SampleFolder]): Split sources (fixed stems).
            cache (ConditionCache): Cached conditions aligned with ``sources``.
            seed (int): Base seed of the reverse chains.
            split (str): Split role name for logs and the manifest.
            epoch (int): Draw index recorded in the manifest.
            image_dir (Path): Destination images directory.
            label_dir (Path): Destination labels directory.
            manifest_path (Path): Provenance CSV appended with one row.

        Returns:
            GenerationStats: Provenance and agreement of the pass.

        Raises:
            ValueError: If the sources are empty, misaligned with the cached
                conditions, or not all present in the label index.
        """
        if not sources:
            raise ValueError(f"Cannot generate the {split} split with no samples.")
        if len(cache) != len(sources):
            raise ValueError(
                f"Cached conditions ({len(cache)}) misaligned with {split} "
                f"sources ({len(sources)})."
            )
        unknown: list[str] = [
            source.stem
            for source in sources
            if source.stem not in self.labels.by_stem
        ]
        if unknown:
            raise ValueError(
                f"{len(unknown)} {split} samples are absent from the label "
                f"index, starting with {unknown[:3]}. The index and the source "
                "list must be built from the same samples."
            )
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        clear_generated_split(image_dir, label_dir)
        started: float = time.time()
        latents: torch.Tensor = self.stack.sample_latents(cache, seed)
        written: int = 0
        boxes: int = 0
        agreements: list[float] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for chunk_sources in _chunked(sources, self.stack.decode_chunk):
                floor_ids: torch.Tensor = self.stack.load_floorplan_ids(
                    chunk_sources, pool
                )
                chunk_latents: torch.Tensor = self.stack.denormalize_latents(
                    latents[written : written + len(chunk_sources)]
                ).to(self.stack.device)
                # Decode straight on the device: render_overlay_chunk consumes
                # the logits immediately, so no CPU round-trip is needed.
                logits: torch.Tensor = self.stack.target_vae.decode(chunk_latents)  # type: ignore[attr-defined]
                overlay_u8, arrangement_ids = self.stack.render_overlay_chunk(
                    floor_ids, logits
                )
                component_masks: np.ndarray = self.stack.component_mask(
                    arrangement_ids
                )
                # Targets are read once here and used for both the label file
                # and the box count, so the two can never disagree.
                lines_per_source: list[list[str]] = [
                    self.labels.lines_for(source.stem) for source in chunk_sources
                ]
                futures = [
                    pool.submit(
                        _write_row,
                        overlay_u8[index],
                        source,
                        image_dir,
                        label_dir,
                        lines,
                        self.compress_level,
                    )
                    for index, (source, lines) in enumerate(
                        zip(chunk_sources, lines_per_source, strict=True)
                    )
                ]
                wait(futures)
                # result() re-raises a worker failure: a sample that could not
                # be written must abort the run, never silently vanish from the
                # dataset.
                for future in futures:
                    future.result()
                boxes += sum(len(lines) for lines in lines_per_source)
                agreements.extend(
                    self._chunk_agreements(chunk_sources, component_masks)
                )
                written += len(chunk_sources)
                self.log.info(
                    f"{split} generation: {written}/{len(sources)} images "
                    f"(seed={seed})."
                )
        seconds: float = time.time() - started
        stats: GenerationStats = self._summarize(
            split=split,
            epoch=epoch,
            seed=seed,
            images=written,
            boxes=boxes,
            seconds=seconds,
            agreements=agreements,
        )
        _append_manifest_row(
            manifest_path,
            {
                "split": split,
                "epoch": epoch,
                "seed": seed,
                "images": stats.images,
                "boxes": stats.boxes,
                "agreement_scored": stats.agreement_scored,
                "agreement_mean": _format_optional(stats.agreement_mean),
                "seconds": f"{seconds:.1f}",
            },
        )
        self.log.info(
            f"{split} generation done: {stats.images} images, {stats.boxes} "
            f"labels in {seconds:.1f}s (seed={seed}); component agrees with "
            f"labels.txt at IoU>={stats.agreement_fractions} over "
            f"{stats.agreement_scored} scored samples "
            f"(mean {_format_optional(stats.agreement_mean)})."
        )
        return stats

    def _chunk_agreements(
        self, sources: list[SampleFolder], component_masks: np.ndarray
    ) -> list[float]:
        """Score one chunk's agreement between generated component and targets.

        Args:
            sources (list[SampleFolder]): The chunk's samples, in order.
            component_masks (np.ndarray): Bool ``(B, S, S)`` generated component
                masks, aligned with ``sources``.

        Returns:
            list[float]: One IoU per scored sample. Background samples (no box
                of the component) contribute nothing: there is nothing to agree
                with.
        """
        scored: list[float] = []
        for index, source in enumerate(sources):
            iou: float | None = component_agreement_iou(
                component_masks[index],
                self.labels.by_stem[source.stem].boxes,
                self.stack.image_size,
            )
            if iou is not None:
                scored.append(iou)
        return scored

    def _summarize(
        self,
        split: str,
        epoch: int,
        seed: int,
        images: int,
        boxes: int,
        seconds: float,
        agreements: list[float],
    ) -> GenerationStats:
        """Fold one pass's agreement samples into its statistics.

        Args:
            split (str): Split role name.
            epoch (int): Draw index.
            seed (int): Base seed of the pass.
            images (int): Images written.
            boxes (int): Label lines written.
            seconds (float): Wall-clock cost of the pass.
            agreements (list[float]): Per-sample agreement IoUs.

        Returns:
            GenerationStats: Provenance plus mean IoU and the per-threshold
                fractions.
        """
        mean: float | None = (
            sum(agreements) / len(agreements) if agreements else None
        )
        fractions: dict[str, float] = {
            f"{threshold:g}": (
                sum(1 for value in agreements if value >= threshold)
                / len(agreements)
                if agreements
                else float("nan")
            )
            for threshold in self.thresholds
        }
        return GenerationStats(
            split=split,
            epoch=epoch,
            seed=seed,
            images=images,
            boxes=boxes,
            seconds=seconds,
            agreement_scored=len(agreements),
            agreement_mean=mean,
            agreement_fractions=fractions,
        )


def _chunked(sources: list[SampleFolder], size: int) -> list[list[SampleFolder]]:
    """Split sources into consecutive chunks of at most ``size`` items.

    Args:
        sources (list[SampleFolder]): Sources to split, in write order.
        size (int): Maximum chunk size (>= 1).

    Returns:
        list[list[SampleFolder]]: Consecutive chunks, all but the last full.

    Raises:
        ValueError: If ``size`` is below one, which would loop forever.
    """
    if size < 1:
        raise ValueError(f"Chunk size must be >= 1, got {size}.")
    return [sources[start : start + size] for start in range(0, len(sources), size)]


def _format_optional(value: float | None) -> str:
    """Format an optional metric for a CSV cell or a log line.

    Args:
        value (float | None): The metric, None when it could not be measured.

    Returns:
        str: Six-decimal text, or ``""`` when unmeasured - never ``0.0``, which
            would read as a measured zero.
    """
    return "" if value is None else f"{value:.6f}"


def _append_manifest_row(manifest_path: Path, row: dict[str, object]) -> None:
    """Append one provenance row to the generation manifest CSV.

    Args:
        manifest_path (Path): Manifest file (header written on first call).
        row (dict[str, object]): Values matching :data:`_MANIFEST_FIELDS`.

    Raises:
        OSError: If the manifest cannot be created or appended to; a run whose
            provenance cannot be recorded must stop rather than continue
            silently.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header: bool = not manifest_path.is_file()
    with open(manifest_path, "a", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=list(_MANIFEST_FIELDS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)