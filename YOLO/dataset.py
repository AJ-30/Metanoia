"""Dataset layer for YOLO detection on live diffusion-drawn floor plans.

This module owns every pixel and label the YOLO trainer ever reads, and
nothing else. It has three responsibilities:

1. **Frozen rendering stack** (:class:`FrozenRenderStack`) - loads the frozen
   floorplan mask VAE, the frozen (TV-inclusive) arrangement mask VAE and the
   frozen floorplan->arrangement conditional diffusion denoiser, all in eval
   mode with gradients disabled. Architecture and standardization statistics
   come from the checkpoints themselves, so the rendering matches the run
   that trained them even if the YOLO TOMLs moved on.

2. **Deterministic val/test overlays** (:class:`DeterministicSplitRenderer`) -
   renders GT floorplan + GT arrangement masks with the dataset's own color
   tables (the same overlay helper the conditional-diffusion grids use), once
   per configuration. A content signature records what was rendered; an
   ontology/alpha/size change re-renders instead of silently reusing labels
   written under old class ids. Deterministic pixels keep per-epoch validation
   metrics and early stopping stable.

3. **Live train redraws** (:class:`LiveEpochRedrawer`) - at the end of every
   epoch every train stem is redrawn from a FRESH diffusion draw conditioned on
   its GT floorplan mask (seed = ``BASE_SEED + epoch``), and the boxes are
   derived from the arrangement that was actually drawn (connected components
   per furniture class, mapped through the mask->YOLO id table). Image and
   target therefore always agree, which is what makes this a real detection
   task rather than a noisy-label regime. Stems never change (Ultralytics
   indexes the file list once at train start); only pixels and labels do.

Boxes on all three splits are derived by the same routine, so all splits share
one labelling convention. All tunables live in ``YOLO/config.toml`` /
``YOLO/hyperparameters.toml``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

# Project root so ``conditional_diffusion``/``mask_VAE``/``VAE`` import from
# any working directory (``python YOLO/run.py`` puts YOLO/ on sys.path).
_SCRIPT_DIR: Path = Path(__file__).resolve().parent
_ROOT_DIR: Path = _SCRIPT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

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
    encode_masks_to_single_channel,
    render_group_rgb,
)
from VAE.runner import resolve_device

# Sidecar written into a rendered split's image directory recording the
# content signature it was rendered with, so a later run with a different
# ontology/alpha/size/min-area re-renders instead of trusting stale pixels.
RENDER_SIGNATURE_FILE: str = ".render_signature"

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

# CSV columns of the per-epoch redraw manifest (one row per redraw per epoch).
_MANIFEST_FIELDS: tuple[str, ...] = (
    "epoch",
    "seed",
    "images",
    "boxes",
    "seconds",
)


# ---------------------------------------------------------------------------
# Ontology: source/mask class names <-> YOLO class ids
# ---------------------------------------------------------------------------


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


def resolve_class_name_to_id(
    class_names: list[str],
    aliases: dict[str, str],
) -> dict[str, int]:
    """Build the source-class-name -> YOLO-id map with alias folding.

    Aliases let several source names collapse onto the single YOLO class the
    mask/render can actually represent (e.g. the three ``Dining_table_*``
    sizes -> ``Dining_table``). A source name with no alias must match a
    ``CLASS_NAMES`` entry after normalization; anything else aborts here
    rather than silently training on a shifted ontology.

    Args:
        class_names (list[str]): Ordered YOLO class vocabulary.
        aliases (dict[str, str]): Source name -> CLASS_NAMES entry overrides.

    Returns:
        dict[str, int]: Normalized source name -> YOLO class id.

    Raises:
        ValueError: If a name resolves to no or several YOLO classes.
    """
    yolo_by_key: dict[str, list[str]] = {}
    for yolo_name in class_names:
        yolo_by_key.setdefault(normalize_class_name(yolo_name), []).append(yolo_name)
    for source, target in aliases.items():
        if target not in class_names:
            raise ValueError(
                f"CLASS_ALIASES maps '{source}' to {target!r}, which is not in "
                f"CLASS_NAMES {class_names}."
            )
    resolved: dict[str, int] = {}
    for key, candidates in yolo_by_key.items():
        if len(candidates) > 1:
            raise ValueError(
                f"YOLO class names {candidates} normalize to the same key "
                f"'{key}'; they would be indistinguishable to the source map."
            )
        resolved[key] = class_names.index(candidates[0])
    return resolved


def build_mask_to_yolo_id_map(
    layout: MaskLayout,
    group: MaskGroup,
    class_names: list[str],
    aliases: dict[str, str] | None = None,
) -> dict[int, int]:
    """Map each furniture mask class id to its YOLO class id.

    Fails fast on any ambiguity rather than guessing: the mask ontology and
    the YOLO ontology must correspond one-to-one, otherwise boxes would
    silently carry the wrong class and the run would train on a shifted
    ontology. Class 0 (background) is never a detection target and is skipped.

    Args:
        layout (MaskLayout): Full layout owning ``group`` (for its class names).
        group (MaskGroup): Furniture group whose local ids are mapped.
        class_names (list[str]): Ordered YOLO class vocabulary.
        aliases (dict[str, str] | None): Mask/source name -> CLASS_NAMES entry
            overrides (``CLASS_ALIASES``), e.g. ``dining_set`` ->
            ``Dining_table``.

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
    unreachable: set[str] = {name for names in yolo_by_key.values() for name in names}
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
    expected_furniture_classes: int = len(group.class_names) - 1
    if len(resolved) != expected_furniture_classes:
        raise ValueError(
            f"Resolved {len(resolved)} furniture classes but the mask has "
            f"{expected_furniture_classes}; layout {layout.mode_name!r} is "
            "inconsistent."
        )
    return resolved


# ---------------------------------------------------------------------------
# Source dataset indexing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleFolder:
    """One folder-per-sample source layout.

    Attributes:
        src_dir (Path): Sample folder holding both segmentation masks.
        stem (str): Stable output stem (the folder name). Never changes, so
            Ultralytics' once-indexed train file list stays valid across
            per-epoch redraws.
    """

    src_dir: Path
    stem: str


def list_sample_folders(
    split_dir: str,
    cond_mask_file: str,
    target_mask_file: str,
    cap: int,
) -> list[SampleFolder]:
    """Enumerate the sample folders of a split that carry both masks.

    Only folders providing the floorplan mask AND the arrangement mask are
    returned, so a partially written sample can never reach the pipeline.
    Enumeration is a single non-recursive readdir sorted by folder name for
    reproducible seeding. (Labels are never read: boxes are derived from the
    rendered arrangement, see the module docstring.)

    Args:
        split_dir (str): Split folder holding one sub-folder per sample.
        cond_mask_file (str): Floorplan mask filename required in each sample.
        target_mask_file (str): Arrangement mask filename required in each
            sample.
        cap (int): Pool cap over sample folders; negative keeps every sample.

    Returns:
        list[SampleFolder]: Sorted sample folders of the split.

    Raises:
        FileNotFoundError: If ``split_dir`` does not exist.
        ValueError: If no sample provides both masks.
    """
    split_path: Path = Path(split_dir)
    if not split_path.is_dir():
        raise FileNotFoundError(f"Source split directory missing: {split_dir}")
    sample_dirs: list[Path] = sorted(
        candidate
        for candidate in split_path.iterdir()
        if candidate.is_dir()
        and (candidate / cond_mask_file).is_file()
        and (candidate / target_mask_file).is_file()
    )
    if cap >= 0:
        sample_dirs = sample_dirs[:cap]
    if not sample_dirs:
        raise ValueError(
            f"Split has no usable samples: {split_dir}. Every sample folder "
            f"must carry '{cond_mask_file}' and '{target_mask_file}'."
        )
    return [
        SampleFolder(src_dir=sample_dir, stem=sample_dir.name)
        for sample_dir in sample_dirs
    ]


# ---------------------------------------------------------------------------
# Mask loading and box derivation
# ---------------------------------------------------------------------------


def load_mask_ids(mask_path: Path, image_size: int) -> torch.Tensor:
    """Read one segmentation mask PNG as a class-id tensor.

    Mirrors ``PairedMaskDataset._load_mask`` exactly: single-channel integer
    mask, nearest-neighbour resize so no blended class ids appear.

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


def load_mask_ids_batch(
    paths: list[Path],
    image_size: int,
    pool: ThreadPoolExecutor,
) -> torch.Tensor:
    """Load a batch of class-id masks in parallel and stack them.

    PNG decoding releases the GIL, so a small thread pool overlaps the file
    I/O of the next render chunk with the MPS work of the current one.

    Args:
        paths (list[Path]): Mask paths, order preserved in the output.
        image_size (int): Square side length both frozen VAEs expect.
        pool (ThreadPoolExecutor): Worker pool doing the loads.

    Returns:
        torch.Tensor: int64 ``(N, image_size, image_size)`` class-id tensor.
    """
    id_tensors: list[torch.Tensor] = list(
        pool.map(lambda path: load_mask_ids(path, image_size), paths)
    )
    return torch.stack(id_tensors, dim=0)


def connected_component_boxes(
    id_map: np.ndarray,
    min_area_px: int,
) -> list[tuple[int, int, int, int, int]]:
    """Extract per-instance bounding boxes from a class-id map.

    Each 4-connected region of a single class id becomes one box, which is how
    the dataset's own ``labels.txt`` defines instances (GT labels match the GT
    furniture mask at IoU ~0.97 by this definition).

    Args:
        id_map (np.ndarray): int array ``(H, W)`` of class ids, 0 = background.
        min_area_px (int): Regions smaller than this are dropped as decoder
            speckle rather than furniture.

    Returns:
        list[tuple[int, int, int, int, int]]: ``(class_id, x1, y1, x2, y2)``
            with inclusive-exclusive pixel bounds.
    """
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


def boxes_to_yolo_lines(
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


def write_label_file(label_path: Path, lines: list[str]) -> None:
    """Write one YOLO label file (empty file for a layout with no furniture).

    Args:
        label_path (Path): Destination ``.txt`` path.
        lines (list[str]): Label lines from :func:`boxes_to_yolo_lines`.
    """
    label_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )


def parse_label_lines_to_arrays(
    lines: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Parse YOLO label lines into the arrays Ultralytics keeps in memory.

    Ultralytics' dataset stores ``cls`` as an ``(n, 1)`` float array and
    ``bboxes`` as an ``(n, 4)`` float array of **normalized** ``cx cy w h``
    (``bbox_format="xywh"``, ``normalized=True``). Keeping that exact shape —
    including the empty ``(0, 1)`` / ``(0, 4)`` arrays for a layout with no
    furniture — is what allows the trainer's in-memory labels to be updated
    in place when the pixels are redrawn.

    Args:
        lines (list[str]): Label lines ``"<cls> <cx> <cy> <w> <h>"``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(cls, bboxes)`` float arrays of
            shapes ``(n, 1)`` and ``(n, 4)``; ``n == 0`` for no lines.

    Raises:
        ValueError: If a line does not hold exactly five numbers.
    """
    if not lines:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    rows: list[list[float]] = []
    for line in lines:
        parts: list[str] = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"Label line {line!r} must hold exactly 5 numbers "
                "(cls cx cy w h)."
            )
        try:
            rows.append([float(value) for value in parts])
        except ValueError as exc:
            raise ValueError(f"Label line {line!r} is not numeric: {exc}") from exc
    table: np.ndarray = np.asarray(rows, dtype=np.float32)
    return table[:, 0:1], table[:, 1:5]


def apply_label_lines_to_dataset(
    dataset: object,
    label_lines_by_stem: dict[str, list[str]],
) -> int:
    """Push freshly drawn boxes into the trainer's in-memory train dataset.

    Why this exists: Ultralytics reads the label files ONCE, when the train
    dataloader is built (``BaseDataset.__init__`` -> ``get_labels()``), and
    keeps the parsed boxes in ``dataset.labels`` for the whole run. Images are
    re-read from disk on every batch (``cache=False``), but boxes are NOT, so
    rewriting ``labels/*.txt`` on disk after each epoch would leave the model
    training on the FIRST draw's boxes while looking at every later draw's
    pixels — a pixel/target mismatch of exactly the kind this pipeline is
    built to avoid. Updating ``dataset.labels`` in place (the same dict
    structure ``cache_labels`` produces) keeps image and target in sync.

    Requires ``workers=0``: with worker processes each worker would hold its
    own fork of ``dataset.labels``. The runner pins ``WORKERS=0`` and fails
    fast otherwise.

    Args:
        dataset (object): The trainer's train dataset
            (``trainer.train_loader.dataset``); duck-typed as carrying
            ``im_files`` and ``labels`` in Ultralytics' format.
        label_lines_by_stem (dict[str, list[str]]): Label lines per image
            stem (basename without extension), as returned by
            :meth:`LiveEpochRedrawer.redraw_epoch`.

    Returns:
        int: Number of dataset entries updated.
    """
    updated: int = 0
    for index, image_file in enumerate(dataset.im_files):  # type: ignore[attr-defined]
        stem: str = Path(image_file).stem
        lines: list[str] | None = label_lines_by_stem.get(stem)
        if lines is None:
            continue
        cls, bboxes = parse_label_lines_to_arrays(lines)
        label: dict[str, object] = dataset.labels[index]  # type: ignore[attr-defined]
        label["cls"] = cls
        label["bboxes"] = bboxes
        updated += 1
    return updated


def write_data_yaml(
    live_root: str,
    yaml_path: str,
    class_names: list[str],
) -> str:
    """Write the Ultralytics ``data.yaml`` for the live dataset root.

    Schema matches the frozen-VAE pipeline (train/val/test image dirs plus the
    class table), so ``model.train()`` and ``model.val(split=...)`` work
    unchanged.

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


def clear_generated_split(img_dir: Path, lbl_dir: Path) -> None:
    """Delete generated PNG/label files so a changed split leaves no orphans.

    Args:
        img_dir (Path): Generated images directory of one split.
        lbl_dir (Path): Generated labels directory of one split.
    """
    for directory in (img_dir, lbl_dir):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix in {".png", ".txt"}:
                path.unlink()


# ---------------------------------------------------------------------------
# Render signature: when may a deterministic split be reused?
# ---------------------------------------------------------------------------


def render_signature(
    sources: list[SampleFolder],
    class_names: list[str],
    aliases: dict[str, str] | None,
    overlay_alpha: float,
    image_size: int,
    min_instance_area: int,
) -> str:
    """Build a content fingerprint describing what a render should contain.

    The mirror is skipped when it already looks complete, but stem equality
    alone is not enough: changing the class vocabulary, the aliases or the
    overlay settings changes the pixels and label ids while leaving the stems
    untouched. Without this signature a stale mirror from an older ontology
    survives and every downstream metric is computed against the wrong labels.

    Args:
        sources (list[SampleFolder]): Sources the split should contain.
        class_names (list[str]): Ordered YOLO class vocabulary.
        aliases (dict[str, str] | None): Source-name aliases in force.
        overlay_alpha (float): Overlay blend weight used for rendering.
        image_size (int): Rendered square side length.
        min_instance_area (int): Minimum box area in pixels.

    Returns:
        str: Stable hex digest of every input that affects render content.
    """
    payload: dict[str, object] = {
        "stems": sorted(source.stem for source in sources),
        "class_names": list(class_names),
        "aliases": dict(sorted((aliases or {}).items())),
        "overlay_alpha": round(float(overlay_alpha), 6),
        "image_size": int(image_size),
        "min_instance_area": int(min_instance_area),
    }
    encoded: str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_render_signature(signature_path: Path) -> str | None:
    """Read the recorded render signature, or None when absent/unreadable.

    Args:
        signature_path (Path): Sidecar file written by
            :func:`write_render_signature`.

    Returns:
        str | None: Stored digest, or None when the file is missing or blank.
    """
    try:
        return signature_path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def write_render_signature(signature_path: Path, signature: str) -> None:
    """Record the signature of the render currently on disk.

    Args:
        signature_path (Path): Destination of the sidecar file.
        signature (str): Digest from :func:`render_signature`.
    """
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    signature_path.write_text(f"{signature}\n", encoding="utf-8")


def split_is_current(
    dst_img_dir: Path,
    sources: list[SampleFolder],
    signature_path: Path | None = None,
    signature: str | None = None,
) -> bool:
    """Check whether a deterministic render already matches this run's inputs.

    Compares stem sets (cheap readdir, no image reads) AND, when a signature is
    supplied, the recorded content digest. A stem-only match is not enough: an
    ontology or alias change keeps the stems identical while invalidating every
    pixel and label id, which would silently validate against the previous
    run's labels.

    Args:
        dst_img_dir (Path): Rendered destination images directory.
        sources (list[SampleFolder]): Expected sources for this split.
        signature_path (Path | None): Sidecar holding the previous signature.
        signature (str | None): Signature this run expects.

    Returns:
        bool: True when the destination matches both the stem set and the
            signature (signature checks are skipped when not supplied).
    """
    if signature_path is not None and signature is not None:
        if read_render_signature(signature_path) != signature:
            return False
    if not dst_img_dir.is_dir():
        return False
    expected: set[str] = {source.stem for source in sources}
    if not expected:
        return False
    actual: set[str] = {path.stem for path in dst_img_dir.glob("*.png")}
    return expected == actual


@dataclass(frozen=True)
class RedrawStats:
    """Provenance record of one live train redraw.

    Attributes:
        epoch (int): 1-based epoch number whose pixels were drawn
            (``seed = BASE_SEED + epoch``; epoch 0 is the pre-training draw).
        seed (int): Seed the diffusion chain was reseeded with.
        images (int): Number of train images rewritten.
        boxes (int): Number of boxes written across all images.
        seconds (float): Wall-clock cost of the redraw.
        label_lines_by_stem (dict[str, list[str]]): The label lines written
            for each stem, in Ultralytics' in-memory format after parsing.
            The trainer needs these because Ultralytics snapshots labels once
            at dataloader construction (see :func:`apply_label_lines_to_dataset`).
    """

    epoch: int
    seed: int
    images: int
    boxes: int
    seconds: float
    label_lines_by_stem: dict[str, list[str]]


# ---------------------------------------------------------------------------
# Frozen rendering stack
# ---------------------------------------------------------------------------


class FrozenRenderStack:
    """Frozen floorplan VAE + arrangement VAE + conditional denoiser.

    The two mask VAEs and the conditional denoiser are all frozen; this class
    only runs them forward (``no_grad``, ``eval``) to redraw train pixels and
    to render the deterministic val/test overlays. Diffusion architecture and
    latent standardization stats come from the diffusion checkpoint itself, so
    rendering matches the run that trained it even if YOLO TOMLs moved on.
    Only paths, the overlay alpha, batching and the image size come from YOLO
    TOMLs, and the image size must equal both frozen VAEs' ``IMAGE_SIZE``.
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
            hyperparams (SimpleNamespace): Render knobs (image size, overlay
                alpha, chunk sizes, mu-vs-sample switch, min instance area,
                PNG compression) plus ``CLASS_NAMES``/``CLASS_ALIASES``.
            log (SimpleNamespace): Configured project logger.

        Raises:
            FileNotFoundError: If any checkpoint or color index is missing.
            ValueError: On image-size / latent-dim drift between the frozen
                VAEs, the diffusion checkpoint and the YOLO TOMLs, or when the
                target VAE excludes classes (e.g. tv): a filtered arrangement
                would render systematic false-positive boxes.
        """
        self.log: SimpleNamespace = log
        self.image_size: int = int(hyperparams.REDRAW_IMAGE_SIZE)
        self.overlay_alpha: float = float(hyperparams.REDRAW_OVERLAY_ALPHA)
        self.sample_chunk: int = max(1, int(hyperparams.REDRAW_SAMPLE_CHUNK))
        self.render_chunk: int = max(1, int(hyperparams.REDRAW_RENDER_CHUNK))
        self.min_instance_area: int = max(1, int(hyperparams.REDRAW_MIN_INSTANCE_AREA))
        self.png_compress_level: int = int(hyperparams.REDRAW_PNG_COMPRESS_LEVEL)
        self.use_mu_only: bool = bool(hyperparams.REDRAW_USE_MU_ONLY)
        self.io_workers: int = max(1, int(hyperparams.REDRAW_IO_WORKERS))
        self.device: torch.device = resolve_device(str(config.DEVICE), log)

        color_dir: Path = Path(str(config.MASK_COLOR_INDEX_DIR))
        fallback_dir: Path = Path(str(config.LIVE_SOURCE_TRAIN_SPLIT_DIR)).parent
        self.cond_vae: object
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
        self.target_vae: object
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
        self.cond_group: MaskGroup = self.cond_layout.groups[0]
        self.target_group: MaskGroup = self.target_layout.groups[0]
        self.floorplan_mask_file: str = self.cond_group.mask_file_name
        self.arrangement_mask_file: str = self.target_group.mask_file_name
        self.target_index: torch.Tensor = torch.tensor(
            self.target_group.channel_indices, dtype=torch.long, device=self.device
        )
        self._validate_frozen_consistency(config)

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
        # The reverse chain runs on the configured device (MPS on Apple
        # silicon); .to() moves the registered schedule buffers with it.
        self.model.to(self.device)
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
            self.target_group,
            self.class_names,
            self.class_aliases,
        )
        self.log.info(
            f"Frozen render stack ready on {self.device}: {ckpt_path} "
            f"(cond {config.COND_VAE_CHECKPOINT_PATH}, "
            f"target {config.TARGET_VAE_CHECKPOINT_PATH}); "
            f"mask->YOLO ids {self.mask_to_yolo_id}."
        )

    def _validate_frozen_consistency(self, config: SimpleNamespace) -> None:
        """Fail fast when the frozen VAEs disagree with the YOLO config.

        Args:
            config (SimpleNamespace): Runtime config (unused beyond context in
                error messages; kept for signature symmetry with __init__).

        Raises:
            ValueError: If the two VAEs disagree on image size, if the image
                size does not match ``REDRAW_IMAGE_SIZE``, or if the target
                VAE excludes classes.
        """
        del config  # consistency is checked against the frozen VAEs themselves
        if int(self.cond_arch.IMAGE_SIZE) != int(self.target_arch.IMAGE_SIZE):
            raise ValueError(
                "Frozen VAEs disagree on IMAGE_SIZE: condition "
                f"{self.cond_arch.IMAGE_SIZE} vs target {self.target_arch.IMAGE_SIZE}."
            )
        if int(self.cond_arch.IMAGE_SIZE) != self.image_size:
            raise ValueError(
                f"REDRAW_IMAGE_SIZE ({self.image_size}) does not match the frozen "
                f"VAEs ({self.cond_arch.IMAGE_SIZE})."
            )
        if self.target_layout.excluded_channels:
            raise ValueError(
                "Target arrangement VAE excludes classes "
                f"(channels {self.target_layout.excluded_channels}); a filtered "
                "arrangement would train systematic false positives against the "
                "GT boxes. Use a TV-inclusive arrangement checkpoint."
            )

    def load_floorplan_ids(
        self, sources: list[SampleFolder], pool: ThreadPoolExecutor
    ) -> torch.Tensor:
        """Load GT floorplan class-id masks for a chunk of sources.

        Args:
            sources (list[SampleFolder]): Sources to load (order preserved).
            pool (ThreadPoolExecutor): Worker pool doing the PNG loads.

        Returns:
            torch.Tensor: int64 ``(N, S, S)`` floorplan class-id tensor.
        """
        return load_mask_ids_batch(
            [source.src_dir / self.floorplan_mask_file for source in sources],
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
            torch.Tensor: int64 ``(N, S, S)`` arrangement class-id tensor.
        """
        return load_mask_ids_batch(
            [source.src_dir / self.arrangement_mask_file for source in sources],
            self.image_size,
            pool,
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
    def encode_source_conditions(
        self, sources: list[SampleFolder]
    ) -> torch.Tensor:
        """Encode GT floorplan masks of sources into cached condition latents.

        Streams one chunk at a time (masks are reloaded from disk per chunk),
        so even a full-pool subset never holds all masks in RAM; only the
        ``cond_dim`` float latent per image is kept. Call once per run: GT
        floorplans never change, so only the diffusion draws vary per epoch.

        Args:
            sources (list[SampleFolder]): The run's train sources (fixed stems).

        Returns:
            torch.Tensor: Normalized condition latents on CPU, shape
                ``(N, cond_dim)``, same order as ``sources``.

        Raises:
            ValueError: If ``sources`` is empty.
        """
        if not sources:
            raise ValueError("Cannot encode conditions with no sources.")
        latents: list[torch.Tensor] = []
        with ThreadPoolExecutor(max_workers=self.io_workers) as pool:
            for start in range(0, len(sources), self.render_chunk):
                chunk_sources: list[SampleFolder] = sources[
                    start : start + self.render_chunk
                ]
                floor_ids: torch.Tensor = self.load_floorplan_ids(chunk_sources, pool)
                # The single-channel MaskEncoder reads class-id channels, the
                # legacy one-hot encoder reads the binary channel tensor; the
                # checkpoint's own architecture decides which.
                if self.cond_vae.encoder_kind == "single":  # type: ignore[attr-defined]
                    encoder_input = encode_masks_to_single_channel(
                        [floor_ids], self.cond_layout
                    )
                else:
                    encoder_input = encode_masks_to_channels(
                        [floor_ids], self.cond_layout
                    )
                mu, logvar = self.cond_vae.encoder(encoder_input.to(self.device))  # type: ignore[attr-defined]
                row_latents: torch.Tensor = (
                    mu if self.use_mu_only
                    else self.cond_vae.reparameterize(mu, logvar)  # type: ignore[attr-defined]
                )
                latents.append(self.normalize_cond(row_latents.cpu()))
                done: int = min(start + self.render_chunk, len(sources))
                if done == len(sources) or (start // self.render_chunk) % 20 == 0:
                    self.log.info(
                        f"Condition encoding: {done}/{len(sources)} floorplans."
                    )
        stacked: torch.Tensor = torch.cat(latents, dim=0)
        self.log.info(f"Cached {stacked.shape[0]} condition latents.")
        return stacked

    @torch.no_grad()
    def sample_arrangements(
        self, cond_norm: torch.Tensor, seed: int
    ) -> torch.Tensor:
        """Sample one arrangement per condition through the full reverse chain.

        The chain runs on the configured device under a per-epoch reseeded
        RNG (``torch.manual_seed``), so every epoch's draw is reproducible
        from ``seed`` and varies per epoch. Seeding once per call (not per
        chunk) with a fixed chunk size keeps batching from changing draws:
        consecutive chunks continue the same RNG stream.

        Args:
            cond_norm (torch.Tensor): Normalized conditions on CPU, shape
                ``(N, cond_dim)``.
            seed (int): Epoch seed (``BASE_SEED + epoch``).

        Returns:
            torch.Tensor: Sampled standardized arrangement latents on CPU,
                shape ``(N, latent_dim)``.

        Raises:
            ValueError: If ``cond_norm`` is empty.
        """
        if cond_norm.shape[0] == 0:
            raise ValueError("Cannot sample arrangements with no conditions.")
        random.seed(seed)
        torch.manual_seed(seed)
        self.model.eval()
        chunks: list[torch.Tensor] = []
        for start in range(0, cond_norm.shape[0], self.sample_chunk):
            batch: torch.Tensor = cond_norm[start : start + self.sample_chunk].to(
                self.device
            )
            chunks.append(self.model.sample_conditional(batch).cpu())
            done: int = min(start + self.sample_chunk, cond_norm.shape[0])
            self.log.info(
                f"Diffusion sampling: {done}/{cond_norm.shape[0]} arrangements "
                f"(seed={seed})."
            )
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def render_overlay_chunk(
        self,
        floor_ids: torch.Tensor,
        arrangement_channels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend one chunk of arrangements over their floorplan base.

        Same code path as the conditional-diffusion grids: the floorplan base
        and the furniture render both come from the datasets' authoritative
        color tables, and only furniture-foreground pixels are blended with
        ``REDRAW_OVERLAY_ALPHA``. ``arrangement_channels`` may be raw decoder
        logits (live redraw) or the one-vs-rest GT masks (deterministic
        val/test): the argmax over the arrangement channels is the class-id
        map in both cases, so one function serves both paths. Inputs may live
        on CPU or already on the stack device (``.to`` is a no-op then).

        Args:
            floor_ids (torch.Tensor): int64 floorplan class ids,
                shape ``(B, S, S)``.
            arrangement_channels (torch.Tensor): float arrangement logits or
                one-hot masks, shape ``(B, C, S, S)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(overlay, furn_ids)`` where
                overlay is uint8 ``(B, 3, S, S)`` on CPU in 0..255 and
                furn_ids is int64 ``(B, S, S)`` on CPU (0 = background).

        Raises:
            ValueError: If the batch sizes disagree.
        """
        if floor_ids.shape[0] != arrangement_channels.shape[0]:
            raise ValueError(
                f"Floorplan batch ({floor_ids.shape[0]}) and arrangement batch "
                f"({arrangement_channels.shape[0]}) must match."
            )
        floor_onehot: torch.Tensor = encode_masks_to_channels(
            [floor_ids], self.cond_layout
        ).to(self.device)
        channels: torch.Tensor = arrangement_channels.to(self.device)
        floor_rgb: torch.Tensor = render_group_rgb(
            floor_onehot, self.cond_group, self.device
        )
        furn_rgb: torch.Tensor = render_group_rgb(
            channels, self.target_group, self.device
        )
        # Same argmax the renderer used, so boxes describe the pixels that are
        # actually written rather than a re-derived variant.
        furn_ids: torch.Tensor = channels.index_select(1, self.target_index).argmax(
            dim=1
        )
        overlay: torch.Tensor = overlay_furniture_on_floorplan(
            floor_rgb, furn_rgb, furn_ids, self.overlay_alpha
        )
        overlay_u8: torch.Tensor = (
            (overlay.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu()
        )
        return overlay_u8, furn_ids.cpu()


# ---------------------------------------------------------------------------
# Row writing shared by both renderers
# ---------------------------------------------------------------------------


def _write_rendered_row(
    overlay_u8: torch.Tensor,
    furn_ids: np.ndarray,
    source: SampleFolder,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    mask_to_yolo_id: dict[int, int],
    image_size: int,
    min_instance_area: int,
    compress_level: int,
) -> list[str]:
    """Write one rendered overlay PNG plus its derived label file.

    Runs in a save-worker thread: only file encoding + I/O happens here, so
    concurrent execution is safe (each call owns distinct tensors/paths).

    Args:
        overlay_u8 (torch.Tensor): uint8 ``(3, S, S)`` overlay image.
        furn_ids (np.ndarray): int arrangement class ids ``(S, S)``.
        source (SampleFolder): Source whose stem names the output files.
        dst_img_dir (Path): Destination images directory.
        dst_lbl_dir (Path): Destination labels directory.
        mask_to_yolo_id (dict[int, int]): Furniture mask class id -> YOLO id.
        image_size (int): Rendered square side length (normalization base).
        min_instance_area (int): Minimum connected-region area for a box.
        compress_level (int): PNG zlib level.

    Returns:
        list[str]: The label lines written for this image (empty for a layout
            with no furniture), so the trainer can sync its in-memory dataset.
    """
    boxes: list[tuple[int, int, int, int, int]] = connected_component_boxes(
        furn_ids, min_instance_area
    )
    lines: list[str] = boxes_to_yolo_lines(
        boxes, mask_to_yolo_id, image_size, image_size
    )
    image_hwc: np.ndarray = overlay_u8.permute(1, 2, 0).numpy()
    Image.fromarray(image_hwc, mode="RGB").save(
        str(dst_img_dir / f"{source.stem}.png"), compress_level=compress_level
    )
    write_label_file(dst_lbl_dir / f"{source.stem}.txt", lines)
    return lines


def _render_and_write_chunk(
    pool: ThreadPoolExecutor,
    stack: FrozenRenderStack,
    floor_ids: torch.Tensor,
    arrangement_channels: torch.Tensor,
    sources: list[SampleFolder],
    dst_img_dir: Path,
    dst_lbl_dir: Path,
) -> tuple[int, dict[str, list[str]]]:
    """Render one chunk and wait for its PNG/label writes to finish.

    The chunk is the memory bound: only ``render_chunk`` overlays are alive at
    once (each ~0.5 MB), so a full-pool redraw never spikes. Worker failures
    are re-raised on the main thread, never swallowed.

    Args:
        pool (ThreadPoolExecutor): Worker pool doing box derivation + writes.
        stack (FrozenRenderStack): Frozen VAEs + overlay renderer.
        floor_ids (torch.Tensor): int64 floorplan class ids ``(B, S, S)``.
        arrangement_channels (torch.Tensor): float arrangement logits or
            one-hot masks ``(B, C, S, S)``.
        sources (list[SampleFolder]): Sources of this chunk (order preserved).
        dst_img_dir (Path): Destination images directory.
        dst_lbl_dir (Path): Destination labels directory.

    Returns:
        tuple[int, dict[str, list[str]]]: Number of boxes written across the
            chunk, and the label lines written per stem (for the trainer's
            in-memory dataset sync; see :func:`apply_label_lines_to_dataset`).
    """
    overlay_u8, furn_ids = stack.render_overlay_chunk(floor_ids, arrangement_channels)
    futures = [
        pool.submit(
            _write_rendered_row,
            overlay_u8[index],
            furn_ids[index].numpy(),
            source,
            dst_img_dir,
            dst_lbl_dir,
            stack.mask_to_yolo_id,
            stack.image_size,
            stack.min_instance_area,
            stack.png_compress_level,
        )
        for index, source in enumerate(sources)
    ]
    wait(futures)
    # result() re-raises any worker exception: a failed write must abort the
    # run rather than silently drop an image from the dataset.
    lines_by_stem: dict[str, list[str]] = {
        source.stem: future.result() for source, future in zip(sources, futures, strict=True)
    }
    return sum(len(lines) for lines in lines_by_stem.values()), lines_by_stem


def _chunked(sources: list[SampleFolder], size: int) -> list[list[SampleFolder]]:
    """Split sources into consecutive chunks of at most ``size`` items.

    Args:
        sources (list[SampleFolder]): Sources to split.
        size (int): Maximum chunk size (>= 1).

    Returns:
        list[list[SampleFolder]]: Consecutive chunks, all but the last full.
    """
    if size < 1:
        raise ValueError(f"Chunk size must be >= 1, got {size}.")
    return [sources[start : start + size] for start in range(0, len(sources), size)]


# ---------------------------------------------------------------------------
# Deterministic val/test renderer
# ---------------------------------------------------------------------------


class DeterministicSplitRenderer:
    """Renders deterministic GT overlays (GT floorplan + GT arrangement) once.

    No diffusion is involved, so a completed render is skipped - but only when
    it also matches this run's signature (class vocabulary, aliases, overlay
    alpha, image size, min instance area). An ontology change keeps the stems
    identical while invalidating every label id, so a stem-only check would
    silently keep validating against the previous run's labels.
    """

    def __init__(
        self,
        stack: FrozenRenderStack,
        hyperparams: SimpleNamespace,
        log: SimpleNamespace,
    ) -> None:
        """Bind the frozen stack and worker count.

        Args:
            stack (FrozenRenderStack): Frozen VAEs + overlay renderer.
            hyperparams (SimpleNamespace): Render knobs (chunk size, workers).
            log (SimpleNamespace): Configured project logger.
        """
        self.stack: FrozenRenderStack = stack
        self.workers: int = max(1, int(hyperparams.REDRAW_IO_WORKERS))
        self.log: SimpleNamespace = log

    def render_split(
        self,
        sources: list[SampleFolder],
        dst_img_dir: Path,
        dst_lbl_dir: Path,
    ) -> int:
        """Render one deterministic split, skipping a current render.

        Args:
            sources (list[SampleFolder]): Split sources to render.
            dst_img_dir (Path): Destination images directory.
            dst_lbl_dir (Path): Destination labels directory.

        Returns:
            int: Number of images rendered (0 when the render was current).
        """
        signature: str = render_signature(
            sources,
            self.stack.class_names,
            self.stack.class_aliases,
            self.stack.overlay_alpha,
            self.stack.image_size,
            self.stack.min_instance_area,
        )
        signature_path: Path = dst_img_dir / RENDER_SIGNATURE_FILE
        if split_is_current(dst_img_dir, sources, signature_path, signature):
            self.log.info(
                f"Deterministic render already current in {dst_img_dir} "
                f"({len(sources)} images); skipping."
            )
            return 0
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        clear_generated_split(dst_img_dir, dst_lbl_dir)
        rendered: int = 0
        boxes: int = 0
        started: float = time.time()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for chunk_sources in _chunked(sources, self.stack.render_chunk):
                floor_ids: torch.Tensor = self.stack.load_floorplan_ids(
                    chunk_sources, pool
                )
                arrangement_ids: torch.Tensor = self.stack.load_arrangement_ids(
                    chunk_sources, pool
                )
                arrangement_onehot: torch.Tensor = encode_masks_to_channels(
                    [arrangement_ids], self.stack.target_layout
                )
                chunk_boxes, _ = _render_and_write_chunk(
                    pool,
                    self.stack,
                    floor_ids,
                    arrangement_onehot,
                    chunk_sources,
                    dst_img_dir,
                    dst_lbl_dir,
                )
                boxes += chunk_boxes
                rendered += len(chunk_sources)
            self.log.info(
                f"Deterministic render: {rendered}/{len(sources)} images "
                f"({boxes} boxes) in {dst_img_dir}."
            )
        write_render_signature(signature_path, signature)
        self.log.info(
            f"Rendered {rendered} deterministic GT overlays ({boxes} boxes) in "
            f"{time.time() - started:.1f}s."
        )
        return rendered


# ---------------------------------------------------------------------------
# Live per-epoch train redrawer
# ---------------------------------------------------------------------------


class LiveEpochRedrawer:
    """Redraws every train stem from a fresh diffusion draw each epoch.

    Only the diffusion noise is fresh: the condition (GT floorplan) is cached
    once per run and reused, and PNG stems never change, so Ultralytics'
    file index survives the rewrite. Boxes are derived from the arrangement
    that was actually drawn, so image and target always agree.
    """

    def __init__(
        self,
        stack: FrozenRenderStack,
        hyperparams: SimpleNamespace,
        log: SimpleNamespace,
    ) -> None:
        """Bind the frozen stack and worker count.

        Args:
            stack (FrozenRenderStack): Frozen VAEs + diffusion denoiser.
            hyperparams (SimpleNamespace): Render knobs (chunk size, workers).
            log (SimpleNamespace): Configured project logger.
        """
        self.stack: FrozenRenderStack = stack
        self.workers: int = max(1, int(hyperparams.REDRAW_IO_WORKERS))
        self.base_seed: int = int(hyperparams.BASE_SEED)
        self.log: SimpleNamespace = log

    def redraw_epoch(
        self,
        sources: list[SampleFolder],
        cond_norm: torch.Tensor,
        epoch: int,
        dst_img_dir: Path,
        dst_lbl_dir: Path,
        manifest_path: Path,
    ) -> RedrawStats:
        """Redraw every train stem from a diffusion draw seeded for ``epoch``.

        Args:
            sources (list[SampleFolder]): The run's train sources (fixed stems).
            cond_norm (torch.Tensor): Cached normalized conditions on CPU,
                shape ``(N, cond_dim)``, same order as ``sources``.
            epoch (int): 0-based draw index; ``seed = BASE_SEED + epoch``.
                Epoch 0 is the pre-training draw; at the end of epoch E the
                trainer calls this with ``E + 1`` so epoch E+1 trains on
                pixels drawn after epoch E's last weight update.
            dst_img_dir (Path): Live train images dir (created if needed).
            dst_lbl_dir (Path): Live train labels dir (created if needed).
            manifest_path (Path): Provenance CSV appended with one row per
                redraw (header written on first call).

        Returns:
            RedrawStats: Provenance of the redraw (images, boxes, seconds).

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
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        seed: int = self.base_seed + epoch
        started: float = time.time()
        sampled: torch.Tensor = self.stack.sample_arrangements(cond_norm, seed)
        rewritten: int = 0
        boxes: int = 0
        label_lines_by_stem: dict[str, list[str]] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for chunk_sources in _chunked(sources, self.stack.render_chunk):
                floor_ids: torch.Tensor = self.stack.load_floorplan_ids(
                    chunk_sources, pool
                )
                latents_raw: torch.Tensor = self.stack.denormalize_latents(
                    sampled[rewritten : rewritten + len(chunk_sources)]
                ).to(self.stack.device)
                # Decode straight on the device: render_overlay_chunk consumes
                # the logits immediately, so no CPU round-trip is needed.
                logits: torch.Tensor = self.stack.target_vae.decode(  # type: ignore[attr-defined]
                    latents_raw
                )
                chunk_boxes, chunk_lines = _render_and_write_chunk(
                    pool,
                    self.stack,
                    floor_ids,
                    logits,
                    chunk_sources,
                    dst_img_dir,
                    dst_lbl_dir,
                )
                boxes += chunk_boxes
                label_lines_by_stem.update(chunk_lines)
                rewritten += len(chunk_sources)
                self.log.info(
                    f"Redraw epoch {epoch}: {rewritten}/{len(sources)} images "
                    f"({boxes} boxes so far)."
                )
        seconds: float = time.time() - started
        _append_manifest_row(
            manifest_path,
            {
                "epoch": epoch,
                "seed": seed,
                "images": rewritten,
                "boxes": boxes,
                "seconds": f"{seconds:.1f}",
            },
        )
        stats: RedrawStats = RedrawStats(
            epoch=epoch,
            seed=seed,
            images=rewritten,
            boxes=boxes,
            seconds=seconds,
            label_lines_by_stem=label_lines_by_stem,
        )
        self.log.info(
            f"Redraw epoch {epoch} done: {rewritten} images, {boxes} boxes in "
            f"{seconds:.1f}s (seed={seed})."
        )
        return stats


def _append_manifest_row(manifest_path: Path, row: dict[str, object]) -> None:
    """Append one provenance row to the redraw manifest CSV.

    Args:
        manifest_path (Path): Manifest file (header written on first call).
        row (dict[str, object]): Values matching :data:`_MANIFEST_FIELDS`.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header: bool = not manifest_path.is_file()
    with open(manifest_path, "a", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=list(_MANIFEST_FIELDS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
