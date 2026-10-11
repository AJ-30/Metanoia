"""Live conditional-diffusion overlay pipeline feeding YOLO train images.

Flow per training run (no fixed dataset is ever built):

  1. Every live source sample folder provides the GROUND TRUTH floorplan mask
     (``COND_MASK_FILE``), the GT arrangement mask (``TARGET_MASK_FILE``) and
     the GT boxes (``labels.txt`` in ``CLASS_NAMES`` name format).
  2. The train stems are chosen ONCE per run (seeded subset of the pool) and
     their GT floorplan masks are encoded ONCE with the frozen floorplan mask
     VAE into condition latents (cached: 128 floats per image).
  3. Before epoch 1 and at the end of every epoch, each train stem is redrawn:
     the cached condition goes through the frozen floorplan->arrangement
     diffusion model with a FRESH noise draw (``seed = BASE_SEED + epoch``),
     the sampled arrangement latent is decoded with the frozen arrangement
     (TV-inclusive) decoder, and the predicted furniture is blended over the
     GT floorplan base with the shared overlay helper. Only the PNG pixels
     change; stems never do, because Ultralytics indexes the file list once at
     train start.
  4. Train labels are derived FROM THE DRAWN ARRANGEMENT (connected components
     per furniture class, mapped through the mask->YOLO id table), so the boxes
     always describe the pixels next to them. Copying the GT ``labels.txt``
     instead is available via ``LIVE_REGENERATE_LABELS = false`` for ablations
     only: a fresh draw generally lands somewhere else, so those boxes
     contradict the image.
  5. Val/test are DETERMINISTIC GT overlays (GT floorplan + GT arrangement, no
     diffusion), labelled by the same routine, so all three splits share one
     convention and early stopping plus the final report stay stable across
     runs. Each mirror carries a signature; an ontology/alpha/size change
     re-renders instead of reusing labels written under the old class ids.

All rendering reuses the conditional-diffusion grid helpers
(``overlay_furniture_on_floorplan``) and the mask color tables
(``render_group_rgb``); nothing here duplicates that logic. Every tunable
constant lives in ``YOLO/config.toml`` / ``YOLO/hyperparameters.toml``.
"""

import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

# Project root so ``helpers/``, ``mask_VAE`` and ``conditional_diffusion``
# import from any working directory.
SCRIPT_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from conditional_diffusion.model import ConditionalLatentDiffusion
from conditional_diffusion.runner import (
    ACCEPTED_MASK_IMAGE_MODES,
    MASK_IMAGE_MODE,
    MASK_RESAMPLE,
    build_frozen_mask_vae,
    overlay_furniture_on_floorplan,
    resolve_color_index_file,
)
from mask_VAE.masking import (
    MaskGroup,
    MaskLayout,
    encode_masks_to_channels,
    onehot_to_single_channel,
    render_group_rgb,
)
from mask_VAE.model import MaskVAEModel
from VAE.runner import resolve_device
from YOLO.augment import (
    clear_generated_split,
    read_sample_labels,
    resolve_class_name_to_id,
)

# Sidecar written into a mirrored split's image directory recording the
# content signature it was rendered with, so a later run with a different
# ontology/alpha/size re-renders instead of trusting stale pixels and labels.
MIRROR_SIGNATURE_FILE: str = ".mirror_signature"

# Diffusion checkpoints store their own training hyperparameters dict; these
# keys are read from the checkpoint itself (never from YOLO TOMLs) to rebuild
# the denoiser architecture bit-for-bit.
_DIFFUSION_ARCH_KEYS: tuple[str, ...] = (
    "LATENT_DIM",
    "COND_DIM",
    "HIDDEN_DIM",
    "NUM_LAYERS",
    "TIME_EMB_DIM",
    "TIME_MAX_PERIOD",
    "DROPOUT",
    "NUM_TIMESTEPS",
    "BETA_START",
    "BETA_END",
)


@dataclass
class LiveSource:
    """One live source sample: GT masks on disk plus validated GT labels.

    Attributes:
        src_dir (Path): Sample folder holding both masks and ``labels.txt``.
        stem (str): Stable output stem (the folder name); never changes, so
            Ultralytics' once-indexed file list stays valid across regens.
        label_lines (list[str]): Validated YOLO label lines with numeric class
            ids read from the source ``labels.txt``. Used as the target only in
            the noisy-label ablation mode; by default labels are derived from
            the drawn pixels (see the module docstring).
    """

    src_dir: Path
    stem: str
    label_lines: list[str]


def list_live_sources(
    split_dir: str,
    cond_mask_file: str,
    target_mask_file: str,
    label_file_name: str,
    class_name_to_id: dict[str, int],
    max_sources: int,
) -> list[LiveSource]:
    """Enumerate live source folders providing BOTH masks and GT labels.

    Only folders carrying the floorplan mask, the arrangement mask AND the
    label file are returned, so a partially written sample can never reach
    the pipeline. Enumeration is a single non-recursive readdir sorted by
    folder name for reproducible seeding.

    Args:
        split_dir (str): Split folder holding one sub-folder per sample.
        cond_mask_file (str): Floorplan mask filename required in each sample.
        target_mask_file (str): Arrangement mask filename required in each
            sample (needed for the deterministic val/test overlays; train
            uses only the floorplan mask as the diffusion condition).
        label_file_name (str): GT label filename required in each sample.
        class_name_to_id (dict[str, int]): Class name -> id map used to
            validate/convert the GT labels once at indexing time.
        max_sources (int): Pool cap; negative keeps every sample.

    Returns:
        list[LiveSource]: Sorted live sources with validated GT labels.

    Raises:
        FileNotFoundError: If ``split_dir`` does not exist.
        ValueError: Propagated for malformed GT label lines.
    """
    split_path: Path = Path(split_dir)
    if not split_path.is_dir():
        raise FileNotFoundError(f"Live source split directory missing: {split_dir}")
    sample_dirs: list[Path] = sorted(
        candidate
        for candidate in split_path.iterdir()
        if candidate.is_dir()
        and (candidate / cond_mask_file).is_file()
        and (candidate / target_mask_file).is_file()
        and (candidate / label_file_name).is_file()
    )
    if max_sources >= 0:
        sample_dirs = sample_dirs[:max_sources]
    sources: list[LiveSource] = []
    for sample_dir in sample_dirs:
        # Validated once here; every regen copies these lines verbatim.
        label_lines: list[str] = read_sample_labels(
            sample_dir, label_file_name, class_name_to_id
        )
        sources.append(
            LiveSource(src_dir=sample_dir, stem=sample_dir.name, label_lines=label_lines)
        )
    return sources


def select_train_subset(
    pool: list[LiveSource],
    num_samples: int,
    seed: int,
) -> list[LiveSource]:
    """Choose the run's train stems once (seeded subset of the pool).

    The stems are fixed for the whole run while only their pixels change per
    epoch, because Ultralytics indexes the train file list once at train
    start and a changing stem set would break its dataloader mid-run.

    Args:
        pool (list[LiveSource]): Full indexed live pool (sorted).
        num_samples (int): Subset size; negative keeps the whole pool.
        seed (int): Run seed (``BASE_SEED``); draws are reproducible.

    Returns:
        list[LiveSource]: The run's train sources, in sorted-stem order.

    Raises:
        ValueError: If the pool is empty or ``num_samples`` exceeds it.
    """
    if not pool:
        raise ValueError("Live source pool is empty; cannot select train stems.")
    if num_samples < 0 or num_samples >= len(pool):
        return list(pool)
    rng: random.Random = random.Random(seed)
    chosen: list[LiveSource] = rng.sample(pool, num_samples)
    return sorted(chosen, key=lambda source: source.stem)


def load_mask_ids(mask_path: Path, image_size: int) -> torch.Tensor:
    """Read one segmentation mask PNG as a class-id tensor.

    Mirrors ``PairedMaskDataset._load_mask`` exactly (single-channel integer
    mask, nearest-neighbour resize so no blended class ids appear).

    Args:
        mask_path (Path): Path of the ``*_segmentation.png`` file.
        image_size (int): Square side length both frozen VAEs expect.

    Returns:
        torch.Tensor: int64 ``(image_size, image_size)`` class-id tensor.

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
            (image_size, image_size),
            resample=MASK_RESAMPLE,
        )
        array: np.ndarray = np.array(resized, dtype=np.uint8)
        return torch.from_numpy(array.astype(np.int64))


def load_source_mask_batch(
    sources: list[LiveSource],
    mask_file: str,
    layout: MaskLayout,
    image_size: int,
) -> torch.Tensor:
    """Load one mask file across sources into a one-vs-rest batch tensor.

    Args:
        sources (list[LiveSource]): Sources to load (order preserved).
        mask_file (str): Mask filename inside each sample folder.
        layout (MaskLayout): Single-group layout the masks are encoded with.
        image_size (int): Square side length both frozen VAEs expect.

    Returns:
        torch.Tensor: Float32 ``(N, C, S, S)`` one-vs-rest batch on CPU.
    """
    id_tensors: list[torch.Tensor] = [
        load_mask_ids(source.src_dir / mask_file, image_size) for source in sources
    ]
    # Encode one mask at a time (batch of one): encode_masks_to_channels only
    # ever sees single masks upstream (its DataLoader stacks the encoded
    # outputs), so per-mask calls stay on the exact production code path.
    encoded: list[torch.Tensor] = [
        encode_masks_to_channels([ids], layout) for ids in id_tensors
    ]
    return torch.cat(encoded, dim=0)


def overlay_row_to_pil(row: torch.Tensor) -> Image.Image:
    """Convert one overlay row tensor in [0, 1] to an RGB PIL image.

    Args:
        row (torch.Tensor): Float tensor of shape ``(3, S, S)`` in [0, 1].

    Returns:
        Image.Image: RGB PIL image of side S.
    """
    return to_pil_image(row.clamp(0.0, 1.0))


def normalize_class_name(name: str) -> str:
    """Normalize a class name for ontology matching across naming conventions.

    The mask color indexes use lowercase snake_case (``center_table``), the
    YOLO ontology uses CamelCase (``Center_table``) and the source SVGs use
    spaced words (``Dining table big``). All three normalize to the same key,
    so a single mapping can bridge them.

    Args:
        name (str): Raw class name from any of those sources.

    Returns:
        str: Lowercase alphanumeric key with separators removed.
    """
    return "".join(character for character in name.lower() if character.isalnum())


def build_mask_to_yolo_id_map(
    layout: MaskLayout,
    group: MaskGroup,
    class_names: list[str],
    aliases: dict[str, str] | None = None,
) -> dict[int, int]:
    """Map each furniture mask class id to its YOLO class id.

    Fails fast on any ambiguity rather than guessing: the mask ontology and the
    YOLO ontology must correspond one-to-one, otherwise boxes would silently
    carry the wrong class and the run would train on a shifted ontology.

    The arrangement mask has 10 furniture classes, of which ``dining_set``
    covers what the dataset names as three Dining_table sizes. Those sizes are
    not separable in the mask or the render, so they are expected to be folded
    onto one YOLO class via ``aliases``; a mask class with no alias and no
    same-named YOLO class is rejected here rather than mapped arbitrarily.

    Args:
        layout (MaskLayout): Full layout owning ``group`` (for its class names).
        group (MaskGroup): Furniture group whose local ids are mapped.
        class_names (list[str]): Ordered YOLO class vocabulary.
        aliases (dict[str, str] | None): Mask/source name -> CLASS_NAMES entry
            overrides (``CLASS_ALIASES``); used to fold names the mask cannot
            distinguish (e.g. ``dining_set`` -> ``Dining_table``).

    Returns:
        dict[int, int]: Furniture mask class id -> YOLO class id.

    Raises:
        ValueError: If a mask class has no YOLO counterpart, maps to several,
            or a YOLO class is unreachable (ontology mismatch).
    """
    yolo_by_key: dict[str, list[str]] = {}
    for yolo_name in class_names:
        yolo_by_key.setdefault(normalize_class_name(yolo_name), []).append(yolo_name)
    alias_by_key: dict[str, str] = {
        normalize_class_name(source): target for source, target in (aliases or {}).items()
    }
    resolved: dict[int, int] = {}
    unreachable: set[str] = {
        name for names in yolo_by_key.values() for name in names
    }
    for local_id, mask_name in enumerate(group.class_names):
        if local_id == 0:
            continue  # background is not a detection target
        short_name: str = mask_name.split("/")[-1]
        key: str = normalize_class_name(short_name)
        # An explicit alias wins; it is how several source names collapse onto
        # the single class the mask/render can actually represent.
        target: str | None = alias_by_key.get(key)
        matches: list[str] = [target] if target else yolo_by_key.get(key, [])
        if not matches:
            raise ValueError(
                f"Furniture mask class '{mask_name}' has no YOLO counterpart in "
                f"{class_names}. The mask ontology and CLASS_NAMES must match, or "
                "CLASS_ALIASES must fold the mask class onto one of them."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Furniture mask class '{mask_name}' matches several YOLO "
                f"classes {matches}. The mask cannot express this distinction, "
                "so per-instance boxes derived from it would be ambiguous."
            )
        if matches[0] not in class_names:
            raise ValueError(
                f"CLASS_ALIASES maps mask class '{mask_name}' to "
                f"{matches[0]!r}, which is not in CLASS_NAMES {class_names}."
            )
        resolved[local_id] = class_names.index(matches[0])
        unreachable.discard(matches[0])
    if unreachable:
        raise ValueError(
            f"YOLO classes {sorted(unreachable)} have no furniture mask "
            "counterpart, so no training box could ever carry them."
        )
    if len(resolved) != len(group.class_names) - 1:
        raise ValueError(
            f"Resolved {len(resolved)} furniture classes but the mask has "
            f"{len(group.class_names) - 1}; layout {layout.mode_name!r} is "
            "inconsistent."
        )
    return resolved


def connected_component_boxes(
    id_map: np.ndarray,
    min_area_px: int,
) -> list[tuple[int, int, int, int, int]]:
    """Extract per-instance bounding boxes from a class-id map.

    Each 4-connected region of a single class id becomes one box, which is how
    the dataset's own ``labels.txt`` defines instances (verified: GT labels
    match the GT furniture mask at IoU ~0.97).

    Args:
        id_map (np.ndarray): int array ``(H, W)`` of class ids, 0 = background.
        min_area_px (int): Regions smaller than this are dropped as decoder
            speckle rather than furniture.

    Returns:
        list[tuple[int, int, int, int, int]]: ``(class_id, x1, y1, x2, y2)``
            with inclusive-exclusive pixel bounds.
    """
    import cv2

    boxes: list[tuple[int, int, int, int, int]] = []
    for class_id in np.unique(id_map):
        if class_id == 0:
            continue
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            (id_map == class_id).astype(np.uint8), connectivity=4
        )
        for component in range(1, count):
            x, y, width, height, area = (int(v) for v in stats[component])
            if area < min_area_px:
                continue
            boxes.append((int(class_id), x, y, x + width, y + height))
    return boxes


def format_yolo_label_lines(
    boxes: list[tuple[int, int, int, int, int]],
    mask_to_yolo_id: dict[int, int],
    height: int,
    width: int,
) -> list[str]:
    """Convert pixel boxes into normalized YOLO ``cls cx cy w h`` label lines.

    Args:
        boxes (list[tuple[int, int, int, int, int]]): Boxes from
            :func:`connected_component_boxes`.
        mask_to_yolo_id (dict[int, int]): Furniture mask class id -> YOLO id.
        height (int): Image height in pixels (normalization denominator).
        width (int): Image width in pixels (normalization denominator).

    Returns:
        list[str]: One ``"<cls> <cx> <cy> <w> <h>"`` line per box.

    Raises:
        ValueError: If a box carries a class absent from the mapping.
    """
    lines: list[str] = []
    for class_id, x1, y1, x2, y2 in boxes:
        if class_id not in mask_to_yolo_id:
            raise ValueError(
                f"Derived box has furniture class {class_id}, which is not in "
                f"the mask->YOLO map {mask_to_yolo_id}."
            )
        box_width: float = (x2 - x1) / width
        box_height: float = (y2 - y1) / height
        center_x: float = (x1 + box_width * width / 2.0) / width
        center_y: float = (y1 + box_height * height / 2.0) / height
        lines.append(
            f"{mask_to_yolo_id[class_id]} {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )
    return lines


def write_live_data_yaml(
    live_root: str,
    yaml_path: str,
    class_names: list[str],
) -> str:
    """Write the Ultralytics ``data.yaml`` for the live dataset root.

    Same schema as the frozen-VAE pipeline (train/val/test image dirs plus
    the class table), so the trainer and ``model.val(split=...)`` work
    unchanged. Only the root differs.

    Args:
        live_root (str): Live dataset root holding ``images/{train,val,test}``.
        yaml_path (str): Destination yaml path.
        class_names (list[str]): Ordered class vocabulary (index == class id).

    Returns:
        str: Absolute path of the written yaml file.
    """
    root_abs: str = str(Path(live_root).resolve())
    yaml_file: Path = Path(yaml_path)
    yaml_file.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f'path: "{root_abs}"',
        "train: images/train",
        "val: images/val",
        # Held-out split for the final unbiased report (never early-stopped on).
        "test: images/test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for class_id, name in enumerate(class_names):
        lines.append(f"  {class_id}: {name}")
    yaml_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(yaml_file)


def mirror_signature(
    sources: list[LiveSource],
    class_names: list[str],
    aliases: dict[str, str] | None,
    overlay_alpha: float,
    image_size: int,
) -> str:
    """Build a content fingerprint describing what a mirror should contain.

    The mirror is skipped when it already looks complete, but stem equality
    alone is not enough: changing the class vocabulary, the aliases or the
    overlay settings changes the pixels and label ids while leaving the stems
    untouched. Without this signature a stale mirror from an older ontology
    survives and every downstream metric is computed against the wrong labels.

    Args:
        sources (list[LiveSource]): Sources the mirror should contain.
        class_names (list[str]): Ordered YOLO class vocabulary.
        aliases (dict[str, str] | None): Source-name aliases in force.
        overlay_alpha (float): Overlay blend weight used for rendering.
        image_size (int): Rendered square side length.

    Returns:
        str: Stable hex digest of every input that affects mirror content.
    """
    payload: dict[str, object] = {
        "stems": sorted(source.stem for source in sources),
        "class_names": list(class_names),
        "aliases": dict(sorted((aliases or {}).items())),
        "overlay_alpha": round(float(overlay_alpha), 6),
        "image_size": int(image_size),
    }
    encoded: str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_mirror_signature(signature_path: Path, signature: str) -> None:
    """Record the signature of the mirror currently on disk.

    Args:
        signature_path (Path): Destination of the sidecar file.
        signature (str): Digest from :func:`mirror_signature`.
    """
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    signature_path.write_text(f"{signature}\n", encoding="utf-8")


def read_mirror_signature(signature_path: Path) -> str | None:
    """Read the recorded mirror signature, or None when absent/unreadable.

    Args:
        signature_path (Path): Sidecar file written by
            :func:`write_mirror_signature`.

    Returns:
        str | None: Stored digest, or None when the file is missing or blank.
    """
    try:
        return signature_path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def live_split_complete(
    dst_img_dir: Path,
    sources: list[LiveSource],
    signature_path: Path | None = None,
    signature: str | None = None,
) -> bool:
    """Check whether a deterministic mirror already matches this run's inputs.

    Compares stem sets (cheap readdir, no image reads) AND, when a signature is
    supplied, the recorded content digest. A stem-only match is not enough:
    an ontology or alias change keeps the stems identical while invalidating
    every pixel and label id, which would silently train and validate against
    stale labels.

    Args:
        dst_img_dir (Path): Mirrored destination images directory.
        sources (list[LiveSource]): Expected sources for this split.
        signature_path (Path | None): Sidecar holding the previous signature.
        signature (str | None): Signature this run expects.

    Returns:
        bool: True when the destination matches both the stem set and the
            signature (signature checks are skipped when not supplied).
    """
    if signature_path is not None and signature is not None:
        if read_mirror_signature(signature_path) != signature:
            return False
    if not dst_img_dir.is_dir():
        return False
    expected: set[str] = {source.stem for source in sources}
    if not expected:
        return False
    actual: set[str] = {path.stem for path in dst_img_dir.glob("*.png")}
    return expected == actual


class LiveDiffusionPipeline:
    """Frozen floorplan->arrangement diffusion plus overlay rendering for YOLO.

    The two mask VAEs and the conditional denoiser are all frozen; this class
    only runs them forward (``no_grad``, ``eval``) to redraw train pixels.
    Diffusion architecture and latent standardization stats come from the
    diffusion checkpoint itself, so the pipeline matches the run that trained
    it even if YOLO TOMLs moved on. Only paths, the overlay alpha and batching
    come from YOLO TOMLs.
    """

    def __init__(
        self,
        config: SimpleNamespace,
        hyperparams: SimpleNamespace,
        log: SimpleNamespace,
    ) -> None:
        """Load the frozen VAEs, the diffusion denoiser and the stats.

        Args:
            config (SimpleNamespace): Runtime config with checkpoint paths,
                mask filenames and color index locations.
            hyperparams (SimpleNamespace): Live knobs (image size, overlay
                alpha, batch size, mu-vs-sample switch, PNG compression).
            log (SimpleNamespace): Configured project logger.

        Raises:
            FileNotFoundError: If any checkpoint or color index is missing.
            ValueError: On image-size / latent-dim drift between the frozen
                VAEs, the diffusion checkpoint and the YOLO TOMLs, or when
                the target VAE excludes classes (e.g. tv): a filtered
                arrangement would render systematic false-positive boxes.
        """
        self.log: SimpleNamespace = log
        self.image_size: int = int(hyperparams.LIVE_IMAGE_SIZE)
        self.overlay_alpha: float = float(hyperparams.LIVE_OVERLAY_ALPHA)
        self.batch_size: int = max(1, int(hyperparams.LIVE_DIFFUSION_BATCH_SIZE))
        self.use_mu_only: bool = bool(hyperparams.LIVE_USE_MU_ONLY)
        self.compress_level: int = int(hyperparams.LIVE_PNG_COMPRESS_LEVEL)
        # Whether train labels are regenerated from the pixels that were
        # actually drawn (True) or copied verbatim from the GT arrangement
        # (False). See regenerate_train_epoch for why this is load-bearing.
        self.regenerate_labels: bool = bool(hyperparams.LIVE_REGENERATE_LABELS)
        # Smallest connected region (in 256x256 pixels) treated as furniture
        # rather than decoder speckle when deriving boxes from a draw.
        self.min_instance_area: int = max(1, int(hyperparams.LIVE_MIN_INSTANCE_AREA))
        self.device: torch.device = resolve_device(str(config.DEVICE), log)

        color_dir: Path = Path(str(config.MASK_COLOR_INDEX_DIR))
        fallback_dir: Path = Path(str(config.LIVE_SOURCE_TRAIN_SPLIT_DIR)).parent
        self.cond_vae: MaskVAEModel
        self.cond_arch: SimpleNamespace
        self.cond_layout: MaskLayout
        self.cond_vae, self.cond_arch, self.cond_layout = build_frozen_mask_vae(
            str(config.COND_VAE_CHECKPOINT_PATH),
            resolve_color_index_file(
                str(config.COND_COLOR_INDEX_FILE), color_dir, fallback_dir
            ),
            str(config.COND_MASK_FILE),
            self.device,
        )
        self.target_vae: MaskVAEModel
        self.target_arch: SimpleNamespace
        self.target_layout: MaskLayout
        self.target_vae, self.target_arch, self.target_layout = build_frozen_mask_vae(
            str(config.TARGET_VAE_CHECKPOINT_PATH),
            resolve_color_index_file(
                str(config.TARGET_COLOR_INDEX_FILE), color_dir, fallback_dir
            ),
            str(config.TARGET_MASK_FILE),
            self.device,
        )
        if int(self.cond_arch.IMAGE_SIZE) != int(self.target_arch.IMAGE_SIZE):
            raise ValueError(
                "Frozen VAEs disagree on IMAGE_SIZE: condition "
                f"{self.cond_arch.IMAGE_SIZE} vs target {self.target_arch.IMAGE_SIZE}."
            )
        if int(self.cond_arch.IMAGE_SIZE) != self.image_size:
            raise ValueError(
                f"LIVE_IMAGE_SIZE ({self.image_size}) does not match the frozen "
                f"VAEs ({self.cond_arch.IMAGE_SIZE})."
            )
        if self.target_layout.excluded_channels:
            raise ValueError(
                "Target arrangement VAE excludes classes "
                f"(channels {self.target_layout.excluded_channels}); a filtered "
                "arrangement would train systematic false positives against the "
                "GT boxes. Use a TV-inclusive arrangement checkpoint."
            )

        ckpt_path: str = str(config.COND_DIFFUSION_CHECKPOINT_PATH)
        checkpoint: dict[str, object] = torch.load(ckpt_path, map_location="cpu")
        ckpt_hyper: dict[str, object] = checkpoint["hyperparameters"]  # type: ignore[assignment]
        try:
            archparams = SimpleNamespace(
                **{key: ckpt_hyper[key] for key in _DIFFUSION_ARCH_KEYS}
            )
        except KeyError as exc:
            raise ValueError(
                f"Diffusion checkpoint {ckpt_path} lacks architecture key {exc}; "
                "it predates the current conditional-diffusion format."
            ) from exc
        if int(archparams.COND_DIM) != int(self.cond_arch.LATENT_DIM):
            raise ValueError(
                f"Diffusion COND_DIM ({archparams.COND_DIM}) does not match the "
                f"condition VAE latent dim ({self.cond_arch.LATENT_DIM})."
            )
        if int(archparams.LATENT_DIM) != int(self.target_arch.LATENT_DIM):
            raise ValueError(
                f"Diffusion LATENT_DIM ({archparams.LATENT_DIM}) does not match "
                f"the target VAE latent dim ({self.target_arch.LATENT_DIM})."
            )
        self.model: ConditionalLatentDiffusion = ConditionalLatentDiffusion(archparams)
        self.model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[arg-type]
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        # Standardization stats + switches travel with the checkpoint so
        # sampling matches the diffusion run's own normalization.
        self.latent_mean: torch.Tensor = checkpoint["latent_mean"]  # type: ignore[assignment]
        self.latent_std: torch.Tensor = checkpoint["latent_std"]  # type: ignore[assignment]
        self.cond_mean: torch.Tensor = checkpoint["cond_mean"]  # type: ignore[assignment]
        self.cond_std: torch.Tensor = checkpoint["cond_std"]  # type: ignore[assignment]
        self.normalize_latents_on: bool = bool(ckpt_hyper.get("NORMALIZE_LATENTS", True))
        self.normalize_cond_on: bool = bool(ckpt_hyper.get("NORMALIZE_COND", True))
        # Furniture mask class id -> YOLO class id, resolved once so label
        # derivation can never silently drift onto a different ontology.
        self.class_names: list[str] = list(hyperparams.CLASS_NAMES)
        self.class_aliases: dict[str, str] = dict(
            getattr(hyperparams, "CLASS_ALIASES", {}) or {}
        )
        self.mask_to_yolo_id: dict[int, int] = build_mask_to_yolo_id_map(
            self.target_layout,
            self.target_layout.groups[0],
            self.class_names,
            self.class_aliases,
        )
        self.log.info(
            f"Live diffusion ready: {ckpt_path} "
            f"(cond {config.COND_VAE_CHECKPOINT_PATH}, "
            f"target {config.TARGET_VAE_CHECKPOINT_PATH}) on {self.device}. "
            f"Label source: "
            f"{'drawn arrangement (self-consistent)' if self.regenerate_labels else 'GT labels (noisy-label regime)'}; "
            f"mask->YOLO ids {self.mask_to_yolo_id}."
        )

    def normalize_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Standardize floorplan conditions (identity when disabled).

        Args:
            cond (torch.Tensor): Raw condition latents, shape (N, cond_dim).

        Returns:
            torch.Tensor: Standardized conditions, same shape.
        """
        if not self.normalize_cond_on:
            return cond
        return (cond - self.cond_mean) / self.cond_std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Invert target standardization before arrangement decoding.

        Args:
            latents (torch.Tensor): Standardized latents, shape (N, latent_dim).

        Returns:
            torch.Tensor: Raw-scale latents the arrangement decoder expects.
        """
        if not self.normalize_latents_on:
            return latents
        return (
            latents * self.latent_std.to(latents.device)
            + self.latent_mean.to(latents.device)
        )

    @torch.no_grad()
    def encode_conditions(self, cond_masks: torch.Tensor) -> torch.Tensor:
        """Encode floorplan mask batches into normalized condition latents.

        Runs the frozen condition encoder once per mask batch (mu when
        ``LIVE_USE_MU_ONLY`` is set, reparameterized samples otherwise) and
        standardizes with the checkpoint stats. Results are cached by the
        caller: GT floorplans never change, so only diffusion draws vary.

        Args:
            cond_masks (torch.Tensor): One-vs-rest floorplan masks on CPU,
                shape (N, Cc, S, S).

        Returns:
            torch.Tensor: Normalized condition latents on CPU, shape
                (N, cond_dim).
        """
        latents: list[torch.Tensor] = []
        for start in range(0, cond_masks.shape[0], self.batch_size):
            batch: torch.Tensor = cond_masks[start : start + self.batch_size]
            if self.cond_vae.encoder_kind == "single":
                encoder_input: torch.Tensor = onehot_to_single_channel(
                    batch, self.cond_layout
                )
            else:
                encoder_input = batch
            mu: torch.Tensor
            logvar: torch.Tensor
            mu, logvar = self.cond_vae.encoder(encoder_input.to(self.device))
            if self.use_mu_only:
                row_latents: torch.Tensor = mu
            else:
                row_latents = self.cond_vae.reparameterize(mu, logvar)
            normed: torch.Tensor = self.normalize_cond(row_latents.cpu())
            latents.append(normed)
        return torch.cat(latents, dim=0)

    @torch.no_grad()
    def encode_source_conditions(
        self, sources: list[LiveSource]
    ) -> torch.Tensor:
        """Encode GT floorplan masks of sources into cached condition latents.

        Streams one batch at a time (masks are reloaded from disk per batch),
        so even a full-pool subset never holds all masks in RAM; only the
        128-float latent per image is kept. Call once per run: GT floorplans
        never change, so only the diffusion draws vary per epoch.

        Args:
            sources (list[LiveSource]): The run's train sources (fixed stems).

        Returns:
            torch.Tensor: Normalized condition latents on CPU, shape
                (N, cond_dim), same order as ``sources``.

        Raises:
            ValueError: If ``sources`` is empty.
        """
        if not sources:
            raise ValueError("Cannot encode conditions with no sources.")
        latents: list[torch.Tensor] = []
        for start in range(0, len(sources), self.batch_size):
            batch_sources: list[LiveSource] = sources[start : start + self.batch_size]
            cond_masks: torch.Tensor = load_source_mask_batch(
                batch_sources,
                str(self.cond_layout.groups[0].mask_file_name),
                self.cond_layout,
                self.image_size,
            )
            latents.append(self.encode_conditions(cond_masks))
        stacked: torch.Tensor = torch.cat(latents, dim=0)
        self.log.info(f"Cached {stacked.shape[0]} condition latents.")
        return stacked

    @torch.no_grad()
    def sample_arrangement_logits(
        self, cond_norm: torch.Tensor, seed: int
    ) -> torch.Tensor:
        """Sample one arrangement per condition through the full reverse chain.

        The chain runs on CPU under a seeded RNG so every epoch's draw is
        reproducible from ``seed``; decoding then runs on the configured
        device in eval mode (deterministic given the sampled latents).

        Args:
            cond_norm (torch.Tensor): Normalized conditions on CPU, shape
                (N, cond_dim).
            seed (int): Epoch seed (``BASE_SEED + epoch``); reseeds Python
                and PyTorch so the draw varies per epoch but repeats per seed.

        Returns:
            torch.Tensor: Arrangement logits on CPU, shape (N, Ct, S, S).
        """
        random.seed(seed)
        torch.manual_seed(seed)
        self.model.eval()
        # Seed once per call (not per batch) so batching never changes draws:
        # consecutive batches continue the same RNG stream.
        sampled: torch.Tensor = self.model.sample_conditional(cond_norm)
        logits: list[torch.Tensor] = []
        for start in range(0, sampled.shape[0], self.batch_size):
            batch: torch.Tensor = self.denormalize_latents(
                sampled[start : start + self.batch_size]
            ).to(self.device)
            logits.append(self.target_vae.decode(batch).cpu())
        return torch.cat(logits, dim=0)

    @torch.no_grad()
    def render_overlays(
        self,
        cond_masks: torch.Tensor,
        arrangement_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Blend arrangements over the floorplan base (dataset colors).

        Same code path as the conditional-diffusion grids: the floorplan base
        and the furniture render both come from the datasets' authoritative
        color tables, and only predicted-foreground pixels are blended with
        ``LIVE_OVERLAY_ALPHA``.

        Args:
            cond_masks (torch.Tensor): GT one-vs-rest floorplan masks on CPU,
                shape (N, Cc, S, S).
            arrangement_logits (torch.Tensor): Sampled arrangement logits on
                CPU, shape (N, Ct, S, S).

        Returns:
            torch.Tensor: Float32 ``(N, 3, S, S)`` overlays in 0..1.

        Raises:
            ValueError: If the batch sizes disagree.
        """
        if cond_masks.shape[0] != arrangement_logits.shape[0]:
            raise ValueError(
                f"Condition batch ({cond_masks.shape[0]}) and arrangement batch "
                f"({arrangement_logits.shape[0]}) must match."
            )
        cpu: torch.device = torch.device("cpu")
        cond_group: MaskGroup = self.cond_layout.groups[0]
        target_group: MaskGroup = self.target_layout.groups[0]
        floor_rgb: torch.Tensor = render_group_rgb(cond_masks, cond_group, cpu)
        furn_rgb: torch.Tensor = render_group_rgb(arrangement_logits, target_group, cpu)
        target_index: torch.Tensor = torch.tensor(
            target_group.channel_indices, dtype=torch.long
        )
        furn_ids: torch.Tensor = arrangement_logits.index_select(
            1, target_index
        ).argmax(dim=1)
        return overlay_furniture_on_floorplan(
            floor_rgb, furn_rgb, furn_ids, self.overlay_alpha
        )

    def regenerate_train_epoch(
        self,
        sources: list[LiveSource],
        cond_norm: torch.Tensor,
        epoch: int,
        base_seed: int,
        train_img_dir: Path,
        train_lbl_dir: Path,
        manifest_path: Path,
    ) -> int:
        """Redraw every train stem from a fresh diffusion draw.

        Masks are reloaded from disk in batches (cheap) while the cached
        condition latents are reused; only the diffusion noise is fresh.
        PNG stems are stable so Ultralytics' file index survives the rewrite.

        Label source (``LIVE_REGENERATE_LABELS``, the load-bearing choice):

        * ``True`` (default) — boxes are derived from the SAME arrangement
          sample that produced the pixels (connected components per furniture
          class), so image and label always agree. This is a real detection
          task and is what val/test measure, so train and val stay in one
          domain.
        * ``False`` — the GT ``labels.txt`` boxes are copied verbatim while the
          pixels come from a fresh draw. Measured on this dataset, only ~27% of
          those boxes overlap the drawn furniture at IoU > 0.5 (vs ~94% for the
          deterministic GT overlays), because a new arrangement generally lands
          somewhere else. Kept as an ablation switch only.

        Args:
            sources (list[LiveSource]): The run's train sources (fixed stems).
            cond_norm (torch.Tensor): Cached normalized conditions on CPU,
                shape (N, cond_dim), same order as ``sources``.
            epoch (int): 1-based epoch number; ``seed = base_seed + epoch``.
            base_seed (int): Run seed (``BASE_SEED``).
            train_img_dir (Path): Live train images dir (created if needed).
            train_lbl_dir (Path): Live train labels dir (created if needed).
            manifest_path (Path): Provenance CSV appended with one row per
                stem per epoch (header written on first call).

        Returns:
            int: Number of train images rewritten.

        Raises:
            ValueError: If ``sources`` is empty or misaligned with cond latents.
        """
        if not sources:
            raise ValueError("Cannot regenerate a train epoch with no sources.")
        if cond_norm.shape[0] != len(sources):
            raise ValueError(
                f"Cached conditions ({cond_norm.shape[0]}) misaligned with "
                f"sources ({len(sources)})."
            )
        train_img_dir.mkdir(parents=True, exist_ok=True)
        train_lbl_dir.mkdir(parents=True, exist_ok=True)
        seed: int = base_seed + epoch
        arrangement_logits: torch.Tensor = self.sample_arrangement_logits(
            cond_norm, seed
        )
        target_index: torch.Tensor = torch.tensor(
            self.target_layout.groups[0].channel_indices, dtype=torch.long
        )
        rewritten: int = 0
        derived_boxes: int = 0
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_header: bool = not manifest_path.is_file()
        with open(manifest_path, "a", newline="", encoding="utf-8") as manifest_file:
            writer = csv.DictWriter(
                manifest_file,
                fieldnames=["epoch", "output", "source", "seed", "boxes", "label_source"],
            )
            if write_header:
                writer.writeheader()
            for start in range(0, len(sources), self.batch_size):
                batch_sources: list[LiveSource] = sources[start : start + self.batch_size]
                cond_masks: torch.Tensor = load_source_mask_batch(
                    batch_sources,
                    str(self.cond_layout.groups[0].mask_file_name),
                    self.cond_layout,
                    self.image_size,
                )
                batch_logits: torch.Tensor = arrangement_logits[
                    start : start + len(batch_sources)
                ]
                overlays: torch.Tensor = self.render_overlays(
                    cond_masks, batch_logits
                )
                # Same argmax the renderer used, so boxes describe the pixels
                # that were actually written rather than a re-decoded variant.
                drawn_ids: torch.Tensor = batch_logits.index_select(
                    1, target_index
                ).argmax(dim=1)
                for row, source, overlay in zip(drawn_ids, batch_sources, overlays):
                    out_path: Path = train_img_dir / f"{source.stem}.png"
                    overlay_row_to_pil(overlay).save(
                        str(out_path), compress_level=self.compress_level
                    )
                    if self.regenerate_labels:
                        label_lines: list[str] = format_yolo_label_lines(
                            connected_component_boxes(
                                row.numpy().astype(np.int32), self.min_instance_area
                            ),
                            self.mask_to_yolo_id,
                            self.image_size,
                            self.image_size,
                        )
                        derived_boxes += len(label_lines)
                        label_source: str = "drawn"
                    else:
                        label_lines = source.label_lines
                        derived_boxes += len(label_lines)
                        label_source = "gt"
                    with open(
                        train_lbl_dir / f"{source.stem}.txt", "w", encoding="utf-8"
                    ) as label_file:
                        label_file.write(
                            "\n".join(label_lines) + ("\n" if label_lines else "")
                        )
                    writer.writerow(
                        {
                            "epoch": epoch,
                            "output": f"{source.stem}.png",
                            "source": source.stem,
                            "seed": seed,
                            "boxes": len(label_lines),
                            "label_source": label_source,
                        }
                    )
                    rewritten += 1
        self.log.info(
            f"Live train epoch {epoch}: redrew {rewritten} overlays with "
            f"{derived_boxes} boxes (seed={seed}, labels="
            f"{'drawn' if self.regenerate_labels else 'gt'})."
        )
        return rewritten

    def mirror_deterministic_split(
        self,
        sources: list[LiveSource],
        dst_img_dir: Path,
        dst_lbl_dir: Path,
    ) -> int:
        """Render deterministic GT overlays (GT floorplan + GT arrangement).

        No diffusion is involved, so a completed mirror is skipped — but only
        when it also matches this run's signature (class vocabulary, aliases,
        overlay alpha, image size). An ontology change keeps the stems
        identical while invalidating every label id, so a stem-only check would
        silently keep validating against the previous run's labels.

        Args:
            sources (list[LiveSource]): Split sources to mirror.
            dst_img_dir (Path): Destination images directory.
            dst_lbl_dir (Path): Destination labels directory.

        Returns:
            int: Number of images mirrored (0 when the mirror was current).
        """
        signature: str = mirror_signature(
            sources,
            list(self.class_names),
            dict(getattr(self, "class_aliases", {}) or {}),
            self.overlay_alpha,
            self.image_size,
        )
        signature_path: Path = dst_img_dir / MIRROR_SIGNATURE_FILE
        if live_split_complete(dst_img_dir, sources, signature_path, signature):
            return 0
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        clear_generated_split(dst_img_dir, dst_lbl_dir)
        cpu: torch.device = torch.device("cpu")
        cond_group: MaskGroup = self.cond_layout.groups[0]
        target_group: MaskGroup = self.target_layout.groups[0]
        target_index: torch.Tensor = torch.tensor(
            target_group.channel_indices, dtype=torch.long
        )
        mirrored: int = 0
        for start in range(0, len(sources), self.batch_size):
            batch_sources: list[LiveSource] = sources[start : start + self.batch_size]
            cond_masks: torch.Tensor = load_source_mask_batch(
                batch_sources,
                str(cond_group.mask_file_name),
                self.cond_layout,
                self.image_size,
            )
            target_masks: torch.Tensor = load_source_mask_batch(
                batch_sources,
                str(target_group.mask_file_name),
                self.target_layout,
                self.image_size,
            )
            floor_rgb: torch.Tensor = render_group_rgb(cond_masks, cond_group, cpu)
            furn_rgb: torch.Tensor = render_group_rgb(target_masks, target_group, cpu)
            furn_ids: torch.Tensor = target_masks.index_select(
                1, target_index
            ).argmax(dim=1)
            overlays: torch.Tensor = overlay_furniture_on_floorplan(
                floor_rgb, furn_rgb, furn_ids, self.overlay_alpha
            )
            for index, (source, overlay) in enumerate(zip(batch_sources, overlays)):
                overlay_row_to_pil(overlay).save(
                    str(dst_img_dir / f"{source.stem}.png"),
                    compress_level=self.compress_level,
                )
                # Same derivation as train: boxes come from the arrangement
                # actually rendered, so all three splits share one convention
                # and a class id always matches the pixels it labels.
                label_lines: list[str] = format_yolo_label_lines(
                    connected_component_boxes(
                        furn_ids[index].numpy().astype(np.int32), self.min_instance_area
                    ),
                    self.mask_to_yolo_id,
                    self.image_size,
                    self.image_size,
                )
                with open(
                    dst_lbl_dir / f"{source.stem}.txt", "w", encoding="utf-8"
                ) as label_file:
                    label_file.write(
                        "\n".join(label_lines) + ("\n" if label_lines else "")
                    )
                mirrored += 1
        write_mirror_signature(signature_path, signature)
        self.log.info(f"Mirrored {mirrored} deterministic GT overlays to {dst_img_dir}.")
        return mirrored
