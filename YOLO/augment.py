"""Frozen-VAE latent augmentation pipeline producing a 2x YOLO dataset.

Source layout is folder-per-sample (one folder holds every render of a layout):

.. code-block:: text

    <split_dir>/<hash>_original/            # or <hash>_augmented_<NNN>
        full.png                            # the only render YOLO reads
        floorplan_only.png                  # ignored
        no_tv.png                           # ignored: labels.txt still lists the
                                           # tv box, so boxes would mismatch pixels
        ground_truth.svg                    # ignored
        labels.txt                          # "<Class_Name> cx cy w h", normalized

``labels.txt`` names its class, so every line is resolved to a numeric id
through ``CLASS_NAMES`` before being written out in Ultralytics' id format.

Pipeline per source image (repeated AUG_FACTOR times with distinct seeds):
  1. Load the RGBA ``full.png``, flatten alpha onto white, resize to
     VAE_IMAGE_SIZE square.
  2. Pick a frozen VAE checkpoint uniformly from CHECKPOINT_EPOCHS, encode to
     the latent mean ``mu`` (deterministic; no reparameterization sampling).
  3. Add Gaussian noise ``N(0, s^2)`` to ``mu`` and decode, where the scale ``s``
     is drawn uniformly per variant from ``LATENT_NOISE_SCALES`` (0.0 = no noise).
  4. Apply Gaussian blur to random patches (photometric only; boxes untouched).

No geometric augmentation is applied (no rotation, no flip). Labels are copied
from the source: the VAE resize is a full-image warp so normalized YOLO
``cx cy w h`` coordinates are invariant to it.

Reproducibility: every output index derives ``seed = BASE_SEED + global_index``
so the full 2x run is exactly reproducible from the TOML files plus a
per-row manifest (source sample, checkpoint epoch, blur, seed).
"""

import csv
import os
import random
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageFilter
from torchvision.transforms.functional import to_pil_image, to_tensor
from tqdm import tqdm

# Allow ``from YOLO.augment import ...`` no matter the launch directory.
SCRIPT_DIR: Path = Path(__file__).resolve().parent
ROOT_DIR: Path = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from VAE.model import VAEModel
from VAE.runner import resolve_image_file_name

# Name of the config.toml key holding the render variant this pipeline reads from
# each source sample folder (YOLO/config.toml). Distinct from the trainers'
# TRAIN_IMAGE_VARIANT only because the augment stage reads source samples, not
# the render it is about to write.
SOURCE_IMAGE_VARIANT_KEY: str = "SOURCE_IMAGE_VARIANT"


def resolve_class_name_to_id(
    class_names: list[str], aliases: dict[str, str] | None = None
) -> dict[str, int]:
    """Build the label class-name -> numeric-id map from the TOML vocabulary.

    Ultralytics' label format stores a numeric class id, but the source
    ``labels.txt`` stores the furniture display name. The id is the index of
    the name in ``CLASS_NAMES``, so the generated ``data.yaml`` and the written
    label ids always agree.

    Args:
        class_names (list[str]): Ordered class vocabulary; index == class id.
        aliases (dict[str, str] | None): Optional source-name -> CLASS_NAMES
            entry overrides, used where the dataset's names cannot be told
            apart by the rendered pixels (e.g. the three Dining_table sizes all
            become one ``Dining_table``). Alias targets must exist in
            ``class_names``.

    Returns:
        dict[str, int]: Mapping of source class name to class id.

    Raises:
        ValueError: If the vocabulary is empty, contains a duplicate name (a
            duplicate would silently drop one of the two classes), or an alias
            points at a name outside the vocabulary.
    """
    if not class_names:
        raise ValueError("CLASS_NAMES must list at least one class.")
    name_to_id: dict[str, int] = {}
    for class_id, name in enumerate(class_names):
        if name in name_to_id:
            raise ValueError(
                f"CLASS_NAMES contains duplicate name {name!r} "
                f"(ids {name_to_id[name]} and {class_id}).")
        name_to_id[name] = class_id
    for source_name, target_name in (aliases or {}).items():
        if target_name not in name_to_id:
            raise ValueError(
                f"CLASS_ALIASES maps {source_name!r} -> {target_name!r}, which is "
                f"not in CLASS_NAMES {class_names}.")
        name_to_id[source_name] = name_to_id[target_name]
    return name_to_id


def parse_box_line(
    line: str,
    class_name_to_id: dict[str, int],
) -> tuple[int, float, float, float, float]:
    """Parse one source label line into typed components.

    Source format is ``"<Class_Name> <cx> <cy> <w> <h>"`` with normalized
    coordinates in [0, 1]; the class id is looked up in ``class_name_to_id``.

    Args:
        line (str): Raw line of form ``"<Class_Name> <cx> <cy> <w> <h>"`` with
            normalized coordinates in [0, 1].
        class_name_to_id (dict[str, int]): Class name to class id map from
            :func:`resolve_class_name_to_id`.

    Returns:
        tuple[int, float, float, float, float]: (cls_id, cx, cy, w, h).

    Raises:
        ValueError: If the line does not have exactly 5 fields, the class name
            is not in ``class_name_to_id``, or the coordinates fall outside
            [0, 1] / have a non-positive size.
    """
    parts: list[str] = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 fields per label line, got {len(parts)}: {line!r}")
    class_name: str = parts[0]
    if class_name not in class_name_to_id:
        raise ValueError(
            f"Unknown class name {class_name!r} in label line: {line!r}. "
            f"Known classes: {sorted(class_name_to_id.keys())}. "
            "Add it to CLASS_NAMES or regenerate the dataset.")
    cls_id: int = class_name_to_id[class_name]
    cx: float
    cy: float
    w: float
    h: float
    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    # Fail fast: normalized boxes must live in the unit square.
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
        raise ValueError(f"Box coordinates out of [0, 1] range: {line!r}")
    if w <= 0.0 or h <= 0.0:
        raise ValueError(f"Box width/height must be positive: {line!r}")
    return cls_id, cx, cy, w, h


def format_box_line(cls_id: int, cx: float, cy: float, w: float, h: float) -> str:
    """Format typed box components back into a YOLO label line.

    Args:
        cls_id (int): Class id, the index of the class in ``CLASS_NAMES``.
        cx (float): Normalized center x.
        cy (float): Normalized center y.
        w (float): Normalized width.
        h (float): Normalized height.

    Returns:
        str: YOLO label line with 6 decimal places.
    """
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def clean_label_lines(
    lines: list[str],
    class_name_to_id: dict[str, int],
) -> list[str]:
    """Rewrite name-based source labels into Ultralytics id-based lines.

    No rotation or flip is applied: normalized boxes are invariant to the
    full-image VAE resize, so source boxes stay valid. Lines are still parsed
    and reformatted to fail fast on unknown class names or invalid coordinates
    and to drop blank lines / degenerate zero-area boxes.

    Args:
        lines (list[str]): Raw source label lines (may include blanks).
        class_name_to_id (dict[str, int]): Class name to class id map from
            :func:`resolve_class_name_to_id`.

    Returns:
        list[str]: Validated label lines with numeric class ids (blank lines
            dropped; degenerate zero-area boxes dropped).

    Raises:
        ValueError: Propagated from :func:`parse_box_line` for malformed lines,
            unknown class names or out-of-range coordinates.
    """
    out_lines: list[str] = []
    for raw in lines:
        if not raw.strip():
            continue
        cls_id, cx, cy, w, h = parse_box_line(raw, class_name_to_id)
        # No geometric transform: copy boxes as-is.
        if w <= 0.0 or h <= 0.0:
            continue
        out_lines.append(format_box_line(cls_id, cx, cy, w, h))
    return out_lines


def apply_patch_blur(
    img: Image.Image,
    rng: random.Random,
    min_patches: int,
    max_patches: int,
    min_size_frac: float,
    max_size_frac: float,
    sigma_min: float,
    sigma_max: float,
) -> tuple[Image.Image, int]:
    """Blur a few random rectangular patches of the image in place.

    This is photometric-only (no geometry change), so bounding boxes are
    unaffected. Each patch is cropped, Gaussian-blurred, and pasted back.

    Args:
        img (Image.Image): Square RGB image to augment (copied before edit).
        rng (random.Random): Seeded RNG for reproducibility.
        min_patches (int): Minimum number of blurred patches.
        max_patches (int): Maximum number of blurred patches.
        min_size_frac (float): Patch side as fraction of image side (min).
        max_size_frac (float): Patch side as fraction of image side (max).
        sigma_min (float): Gaussian blur radius lower bound.
        sigma_max (float): Gaussian blur radius upper bound.

    Returns:
        tuple[Image.Image, int]: (augmented image, number of patches applied).
    """
    img = img.copy()
    side: int = img.size[0]
    num_patches: int = rng.randint(min_patches, max_patches)
    for _ in range(num_patches):
        # Random square patch; side is a fraction of the image side.
        frac: float = rng.uniform(min_size_frac, max_size_frac)
        patch_side: int = max(1, int(side * frac))
        x0: int = rng.randint(0, max(0, side - patch_side))
        y0: int = rng.randint(0, max(0, side - patch_side))
        box: tuple[int, int, int, int] = (x0, y0, x0 + patch_side, y0 + patch_side)
        sigma: float = rng.uniform(sigma_min, sigma_max)
        patch: Image.Image = img.crop(box).filter(ImageFilter.GaussianBlur(radius=sigma))
        img.paste(patch, box)
    return img, num_patches


def flatten_alpha(img: Image.Image) -> Image.Image:
    """Composite an RGBA image onto a white backdrop, returning RGB.

    Args:
        img (Image.Image): Input image (any mode).

    Returns:
        Image.Image: RGB image with transparency baked onto white.
    """
    if img.mode != "RGBA":
        return img.convert("RGB")
    backdrop: Image.Image = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.alpha_composite(backdrop, img).convert("RGB")


class FrozenVAEPool:
    """Cache of frozen VAE decoders across the checkpoint epochs.

    All ``next-collie`` checkpoints share one architecture, so a single
    ``VAEModel`` instance is built from the first checkpoint's stored
    hyperparameters and per-sample weights are swapped in via
    ``load_state_dict``. Models are kept in eval mode under ``no_grad``.
    """

    def __init__(
        self,
        vae_models_dir: str,
        run_id: str,
        checkpoint_epochs: list[int],
        device: torch.device,
    ) -> None:
        """Initialize the pool, verifying every checkpoint file exists.

        Args:
            vae_models_dir (str): Directory holding ``vae_<run>_epoch_<N>.pt``.
            run_id (str): Run id prefix (e.g. "next-collie").
            checkpoint_epochs (list[int]): Epochs to sample from.
            device (torch.device): Device to run the VAE on.

        Raises:
            FileNotFoundError: If any requested checkpoint file is missing.
        """
        self.device: torch.device = device
        self.checkpoint_epochs: list[int] = list(checkpoint_epochs)
        self.state_dicts: dict[int, dict] = {}
        arch_params: SimpleNamespace | None = None
        for epoch in self.checkpoint_epochs:
            path: str = os.path.join(vae_models_dir, f"vae_{run_id}_epoch_{epoch}.pt")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"VAE checkpoint not found: {path}")
            blob = torch.load(path, map_location="cpu")
            self.state_dicts[epoch] = blob["model_state_dict"]
            if arch_params is None:
                arch_params = SimpleNamespace(**blob["hyperparameters"])
        assert arch_params is not None, "Checkpoint list must not be empty."
        # Remember the architecture's native image size so callers can assert
        # their resize target matches the checkpoint (guards TOML drift).
        self.arch_image_size: int = int(arch_params.IMAGE_SIZE)
        # Single shared model; weights are swapped per sample below.
        self.model: VAEModel = VAEModel(arch_params).to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def decode_noisy_mu(
        self,
        image_tensor: torch.Tensor,
        epoch: int,
        noise_std: float,
        seed: int,
    ) -> torch.Tensor:
        """Encode one image with the chosen checkpoint, noise the mean, decode.

        The latent noise comes from a per-variant seeded generator (not the
        global torch RNG), so re-running with the same TOMLs and BASE_SEED
        reproduces identical pixels.

        Args:
            image_tensor (torch.Tensor): Preprocessed image batch of shape
                (1, 3, S, S) in [0, 1] on CPU.
            epoch (int): Which cached checkpoint's weights to use.
            noise_std (float): Std of Gaussian noise added to latent mu.
            seed (int): Per-variant seed; ``seed = BASE_SEED + global_index``.

        Returns:
            torch.Tensor: Decoded image tensor of shape (1, 3, S, S) in [0, 1].
        """
        self.model.load_state_dict(self.state_dicts[epoch])
        with torch.no_grad():
            batch: torch.Tensor = image_tensor.to(self.device)
            mu, _ = self.model.encoder(batch)
            # Seeded generator: randn(shape, generator=...) keeps the global
            # torch RNG untouched and makes each variant reproducible.
            # NOTE (MPS portability): the noise is drawn on CPU with a CPU
            # generator and then moved to the model device, because an MPS
            # tensor cannot consume a CPU generator ("Expected a 'mps'
            # device type for generator but found 'cpu'").
            gen: torch.Generator = torch.Generator().manual_seed(seed)
            noise: torch.Tensor = torch.randn(
                mu.shape, generator=gen, dtype=mu.dtype).to(mu.device) * noise_std
            decoded: torch.Tensor = self.model.decoder(mu + noise)
            return decoded.detach().cpu()

    def decode_batch_noisy_mu(
        self,
        image_tensors: torch.Tensor,
        epoch: int,
        noise_stds: list[float],
        seeds: list[int],
    ) -> torch.Tensor:
        """Encode a batch with one checkpoint, noise each latent, decode all.

        Batched twin of ``decode_noisy_mu``: weights load once per call and a
        single encode/decode forward covers the whole batch, which is what
        keeps the GPU saturated. Per-row noise uses that row's own seeded CPU
        generator, so row ``i`` is bit-identical to ``decode_noisy_mu`` called
        with ``(image_tensors[i:i+1], epoch, noise_stds[i], seeds[i])``.

        Args:
            image_tensors (torch.Tensor): Preprocessed images, shape
                (B, 3, S, S) in [0, 1] on CPU.
            epoch (int): Which cached checkpoint's weights to use for the batch.
            noise_stds (list[float]): Per-row latent noise stds.
            seeds (list[int]): Per-row variant seeds.

        Returns:
            torch.Tensor: Decoded images, shape (B, 3, S, S) in [0, 1] on CPU.

        Raises:
            ValueError: If the batch is empty or the per-row lists mismatch.
        """
        batch_size: int = image_tensors.shape[0]
        if batch_size == 0:
            raise ValueError("decode_batch_noisy_mu requires a non-empty batch.")
        if len(noise_stds) != batch_size or len(seeds) != batch_size:
            raise ValueError(
                f"Per-row lists ({len(noise_stds)}, {len(seeds)}) must match "
                f"batch size {batch_size}.")
        self.model.load_state_dict(self.state_dicts[epoch])
        with torch.no_grad():
            batch: torch.Tensor = image_tensors.to(self.device)
            mu: torch.Tensor = self.model.encoder(batch)[0]
            # One seeded CPU draw per row (same MPS-portability rule as the
            # single-image path), stacked once and moved to the device.
            noise_rows: list[torch.Tensor] = []
            for row_std, row_seed in zip(noise_stds, seeds):
                row_gen: torch.Generator = torch.Generator().manual_seed(row_seed)
                noise_rows.append(torch.randn(
                    mu.shape[1], generator=row_gen, dtype=mu.dtype) * row_std)
            noise: torch.Tensor = torch.stack(noise_rows).to(mu.device)
            decoded: torch.Tensor = self.model.decoder(mu + noise)
            return decoded.detach().cpu()


def list_sample_dirs(
    split_dir: str,
    image_file_name: str,
    label_file_name: str,
) -> list[Path]:
    """Enumerate the per-sample folders of a split that hold an image and a label.

    The source dataset is folder-per-sample: each folder carries every render of
    one layout plus its ``labels.txt``. Only folders providing *both* the
    selected render and the label file are returned, so a partially written
    sample can never reach the pipeline. Enumeration is a single non-recursive
    readdir, sorted by folder name for reproducible seeding.

    Args:
        split_dir (str): Split folder holding one sub-folder per sample.
        image_file_name (str): Render required inside every sample folder
            (e.g. ``"full.png"``).
        label_file_name (str): Label file required inside every sample folder
            (e.g. ``"labels.txt"``).

    Returns:
        list[Path]: Sorted sample folders, each holding the image and the label.

    Raises:
        FileNotFoundError: If ``split_dir`` does not exist.
    """
    split_path: Path = Path(split_dir)
    if not split_path.is_dir():
        raise FileNotFoundError(f"Source split directory missing: {split_dir}")
    return sorted(
        sample_dir
        for sample_dir in split_path.iterdir()
        if sample_dir.is_dir()
        and (sample_dir / image_file_name).is_file()
        and (sample_dir / label_file_name).is_file()
    )


def read_sample_labels(
    sample_dir: Path,
    label_file_name: str,
    class_name_to_id: dict[str, int],
) -> list[str]:
    """Read one sample's label file and convert it to Ultralytics id format.

    The source file names its classes; the returned lines carry the numeric ids
    that match ``CLASS_NAMES``, so the same file can be written straight into
    the generated ``labels/`` tree.

    Args:
        sample_dir (Path): Sample folder holding the label file.
        label_file_name (str): Label filename inside the sample folder
            (e.g. ``"labels.txt"``).
        class_name_to_id (dict[str, int]): Class name to class id map from
            :func:`resolve_class_name_to_id`.

    Returns:
        list[str]: YOLO label lines with numeric class ids. May be empty for a
            layout with no furniture.

    Raises:
        FileNotFoundError: If the sample folder has no label file.
        ValueError: Propagated from :func:`clean_label_lines` for malformed
            lines, unknown class names or out-of-range coordinates.
    """
    label_path: Path = sample_dir / label_file_name
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file missing: {label_path}")
    with open(label_path, encoding="utf-8") as f:
        raw_lines: list[str] = f.read().splitlines()
    return clean_label_lines(raw_lines, class_name_to_id)


def preprocess_for_vae(image_path: str, image_size: int) -> torch.Tensor:
    """Load an RGBA floor-plan image into a VAE-ready tensor batch.

    Args:
        image_path (str): Path to the source PNG.
        image_size (int): Square side length the VAE expects (256).

    Returns:
        torch.Tensor: Float tensor of shape (1, 3, image_size, image_size)
            in [0, 1].
    """
    img: Image.Image = Image.open(image_path)
    img = flatten_alpha(img).resize((image_size, image_size), Image.BILINEAR)
    # C-speed conversion (RGBA-flattened RGB bytes -> float32 tensor in one
    # call); replaces the previous per-byte Python loop (~196k iterations).
    return to_tensor(img).unsqueeze(0)


def decoded_tensor_to_pil(decoded: torch.Tensor) -> Image.Image:
    """Convert a decoded (1, 3, S, S) float tensor in [0, 1] to RGB PIL.

    Args:
        decoded (torch.Tensor): Decoder output batch of size 1.

    Returns:
        Image.Image: RGB PIL image of side S.
    """
    clamped: torch.Tensor = decoded.squeeze(0).clamp(0.0, 1.0)
    # C-speed conversion (float tensor -> RGB bytes in one call); replaces the
    # previous per-pixel Python repacking loop (~196k iterations).
    return to_pil_image(clamped)


def save_variant_files(
    img: Image.Image,
    label_lines: list[str],
    img_path: str,
    lbl_path: str,
    compress_level: int = 6,
) -> None:
    """Write one variant's PNG image and TXT label file.

    Runs in a save-worker thread: only file encoding + I/O happens here, so
    concurrent execution is safe (each call owns distinct objects/paths) and
    output bytes are identical to a serial save.

    Args:
        img (Image.Image): Final augmented RGB image (not mutated after submit).
        label_lines (list[str]): Transformed YOLO label lines for this variant.
        img_path (str): Destination PNG path.
        lbl_path (str): Destination TXT label path.
        compress_level (int): PNG zlib level 0-9; 1 encodes ~3x faster than
            the default 6 with slightly larger files (pixels identical).
    """
    img.save(img_path, compress_level=compress_level)
    with open(lbl_path, "w", encoding="utf-8") as lf:
        lf.write("\n".join(label_lines) + ("\n" if label_lines else ""))


def raise_if_save_failed(futures: list[Future[None]]) -> list[Future[None]]:
    """Re-raise any completed save-worker exception; return still-pending futures.

    Non-blocking: only already-finished futures are checked, so the main VAE
    loop never stalls waiting on disk I/O here (the final drain blocks).

    Args:
        futures (list[Future[None]]): Outstanding save futures.

    Returns:
        list[Future[None]]: The subset still pending.

    Raises:
        Exception: Whatever the first failed worker raised, if any completed
            future failed.
    """
    pending: list[Future[None]] = []
    for fut in futures:
        if fut.done():
            fut.result()  # re-raises the worker exception on the main thread
        else:
            pending.append(fut)
    return pending


def clear_generated_split(img_dir: Path, lbl_dir: Path) -> None:
    """Delete previously generated PNG/TXT files so shrunk configs leave no orphans.

    Only ``*.png`` and ``*.txt`` files are removed; the directories themselves
    (and unrelated files such as ``data.yaml``/manifests elsewhere) are kept.

    Args:
        img_dir (Path): Generated images directory to empty.
        lbl_dir (Path): Generated labels directory to empty.
    """
    for d, pattern in ((img_dir, "*.png"), (lbl_dir, "*.txt")):
        if d.is_dir():
            for p in d.glob(pattern):
                p.unlink()


def mirror_split(
    src_split_dir: str,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    vae_size: int,
    image_file_name: str,
    label_file_name: str,
    class_name_to_id: dict[str, int],
) -> int:
    """Mirror a source split resized to the VAE square domain, labels converted.

    Normalized YOLO boxes are invariant to full-image resize, so only the class
    field changes (name -> id); images are flattened from RGBA and resized.

    Args:
        src_split_dir (str): Source split folder holding one sub-folder per
            sample.
        dst_img_dir (Path): Destination images directory (created if needed).
        dst_lbl_dir (Path): Destination labels directory (created if needed).
        vae_size (int): Square side length both domains share.
        image_file_name (str): Render to mirror out of each sample folder.
        label_file_name (str): Label file inside each sample folder.
        class_name_to_id (dict[str, int]): Class name to class id map used to
            convert the labels.

    Returns:
        int: Number of images mirrored.
    """
    sample_dirs: list[Path] = list_sample_dirs(
        src_split_dir, image_file_name, label_file_name)
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    clear_generated_split(dst_img_dir, dst_lbl_dir)
    count: int = 0
    for sample_dir in sample_dirs:
        stem: str = sample_dir.name
        img_v: Image.Image = flatten_alpha(
            Image.open(sample_dir / image_file_name)).resize(
                (vae_size, vae_size), Image.BILINEAR)
        img_v.save(str(dst_img_dir / f"{stem}.png"))
        out_lines: list[str] = read_sample_labels(
            sample_dir, label_file_name, class_name_to_id)
        with open(dst_lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as out:
            out.write("\n".join(out_lines) + ("\n" if out_lines else ""))
        count += 1
    return count


def split_complete(
    src_split_dir: str,
    image_file_name: str,
    label_file_name: str,
    dst_img_dir: Path,
) -> bool:
    """Check whether every source sample already has a mirrored PNG.

    Compares sample-folder name sets (cheap readdir, no image reads) so a
    completed val/test mirror is skipped instead of re-encoded on every regen.

    Args:
        src_split_dir (str): Source split folder holding one sub-folder per
            sample.
        image_file_name (str): Render required inside every sample folder.
        label_file_name (str): Label file required inside every sample folder.
        dst_img_dir (Path): Mirrored destination images directory.

    Returns:
        bool: True when the destination holds exactly the source stem set.
    """
    if not dst_img_dir.is_dir():
        return False
    src_stems: set[str] = {
        p.name for p in list_sample_dirs(
            src_split_dir, image_file_name, label_file_name)
    }
    if not src_stems:
        return False
    dst_stems: set[str] = {p.stem for p in dst_img_dir.glob("*.png")}
    return src_stems == dst_stems


@dataclass
class _ChunkVariant:
    """Plan + result slot for one augmented variant inside a chunk.

    The checkpoint epoch and noise scale are drawn up front (phase 1) so
    variants can be grouped by epoch for batched GPU forwards (phase 2);
    the remaining RNG draws are replayed from ``seed`` in phase 3, keeping
    outputs identical to the serial order.
    """
    src_idx: int
    rep: int
    seed: int
    epoch_choice: int
    noise_std: float
    input_index: int
    src_path: str
    stem: str
    label_lines: list[str]
    decoded: torch.Tensor | None = None


def run_augmentation(
    config: SimpleNamespace,
    hyperparams: SimpleNamespace,
    log: SimpleNamespace,
) -> dict[str, int]:
    """Generate the full AUG_FACTOR x augmented YOLO dataset from frozen VAE latents.

    For each source train sample, AUG_FACTOR variants are produced. Each
    variant draws an independent seed, a uniform checkpoint epoch, seeded VAE
    latent noise and blur patches. No rotation or flip is applied. Images land
    in ``<root>/images/train`` and labels in ``<root>/labels/train``; the val
    and test splits are mirrored from the same folder-per-sample source layout,
    resized to the VAE square domain with boxes copied verbatim (normalized
    boxes are resize-invariant) and class names resolved to ids. A manifest CSV
    records every train variant's provenance. Train/val/test output dirs are
    cleared first so re-runs with smaller configs leave no orphaned files.

    Args:
        config (SimpleNamespace): Runtime config with source/dest directories,
            VAE location, run id, device string, and the source render variant
            (``SOURCE_IMAGE_VARIANT`` + ``IMAGE_VARIANTS``) and label filename.
        hyperparams (SimpleNamespace): All augmentation knobs (factor, seeds,
            noise std, blur ranges, class names).
        log (SimpleNamespace): Configured project logger.

    Returns:
        dict[str, int]: Counts {"train_images", "val_images", "test_images"}.

    Raises:
        FileNotFoundError: If a required source split directory is missing.
        ValueError: If VAE_IMAGE_SIZE disagrees with the checkpoint architecture.
        AttributeError: If config.toml lacks the source render/label keys.
    """
    for key in ("SOURCE_TRAIN_SPLIT_DIR", "SOURCE_VAL_SPLIT_DIR"):
        if not os.path.isdir(getattr(config, key)):
            raise FileNotFoundError(f"Source directory missing: {getattr(config, key)}")

    # Resolve device with graceful fallback (MPS may be unavailable off-Mac).
    requested: str = str(config.DEVICE)
    if requested == "mps" and torch.backends.mps.is_available():
        device: torch.device = torch.device("mps")
    elif requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if requested not in ("cpu",):
            log.warning(f"Device '{requested}' unavailable; falling back to CPU.")
        device = torch.device("cpu")
    log.info(f"VAE augmentation running on device: {device}")

    vae_pool: FrozenVAEPool = FrozenVAEPool(
        vae_models_dir=config.VAE_MODELS_DIR,
        run_id=config.VAE_RUN_ID,
        checkpoint_epochs=list(hyperparams.CHECKPOINT_EPOCHS),
        device=device,
    )
    # Fail fast on TOML/architecture drift: the resize target must equal the
    # image size the checkpoint was trained with.
    if int(hyperparams.VAE_IMAGE_SIZE) != vae_pool.arch_image_size:
        raise ValueError(
            f"VAE_IMAGE_SIZE={hyperparams.VAE_IMAGE_SIZE} disagrees with "
            f"checkpoint architecture IMAGE_SIZE={vae_pool.arch_image_size}.")

    root: Path = Path(config.AUGMENTED_DATASET_ROOT)
    train_img_dir: Path = root / "images" / "train"
    train_lbl_dir: Path = root / "labels" / "train"
    val_img_dir: Path = root / "images" / "val"
    val_lbl_dir: Path = root / "labels" / "val"
    test_img_dir: Path = root / "images" / "test"
    test_lbl_dir: Path = root / "labels" / "test"
    for d in (train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir,
              test_img_dir, test_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Clear previously generated train files so shrinking AUG_FACTOR or the
    # source set cannot leave orphans that Ultralytics would glob into training.
    clear_generated_split(train_img_dir, train_lbl_dir)

    # Which render of each source folder is read comes from config.toml, so no
    # source filename is hardcoded here. Resolve it once and reuse it below.
    image_file_name: str = resolve_image_file_name(config, SOURCE_IMAGE_VARIANT_KEY)
    label_file_name: str = str(config.SOURCE_LABEL_FILE_NAME)
    source_samples: list[Path] = list_sample_dirs(
        config.SOURCE_TRAIN_SPLIT_DIR,
        image_file_name,
        label_file_name,
    )
    max_samples: int = int(hyperparams.MAX_SAMPLES)
    if max_samples >= 0:
        source_samples = source_samples[:max_samples]
    aug_factor: int = int(hyperparams.AUG_FACTOR)
    class_name_to_id: dict[str, int] = resolve_class_name_to_id(
        list(hyperparams.CLASS_NAMES),
        dict(getattr(hyperparams, "CLASS_ALIASES", {}) or {}),
    )
    log.info(
        f"Found {len(source_samples)} labeled train samples using "
        f"'{image_file_name}' (variant '{config.SOURCE_IMAGE_VARIANT}') + "
        f"'{label_file_name}'; augmenting x{aug_factor} "
        f"= {len(source_samples) * aug_factor} outputs over "
        f"{len(class_name_to_id)} classes.")

    manifest_path: Path = Path(config.AUGMENT_MANIFEST_PATH)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_file = open(manifest_path, "w", newline="", encoding="utf-8")
    manifest_writer = csv.DictWriter(
        manifest_file,
        fieldnames=["output", "source", "checkpoint_epoch", "noise_std",
                    "blur_patches", "seed"],
    )
    manifest_writer.writeheader()

    train_count: int = 0
    pending_saves: list[Future[None]] = []
    save_workers: int = max(1, int(getattr(hyperparams, "SAVE_WORKERS", 1)))
    compress_level: int = int(getattr(hyperparams, "PNG_COMPRESS_LEVEL", 6))
    progress_every: int = max(1, int(getattr(hyperparams, "PROGRESS_LOG_SOURCES", 200)))
    vae_batch: int = max(1, int(getattr(hyperparams, "VAE_BATCH_SIZE", 32)))
    chunk_sources: int = max(1, int(getattr(hyperparams, "AUG_CHUNK_SOURCES", 32)))
    total_outputs: int = len(source_samples) * aug_factor
    loop_start: float = time.time()
    # VAE forwards stay on the main thread (MPS is not thread-safe for
    # concurrent forwards) but run in full batches grouped by checkpoint epoch
    # (one weight load per group); only PNG/label file writes are offloaded to
    # the pool. Manifest rows are written on the main thread in global order,
    # so output bytes are identical to a serial run.
    checkpoint_list: list[int] = list(hyperparams.CHECKPOINT_EPOCHS)
    noise_list: list[float] = [float(s) for s in hyperparams.LATENT_NOISE_SCALES]
    executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=save_workers)
    try:
        with tqdm(total=len(source_samples), desc="VAE augment", unit="sample") as bar:
            sources_done: int = 0
            next_log_at: int = progress_every
            for chunk_start in range(0, len(source_samples), chunk_sources):
                chunk_samples: list[Path] = source_samples[chunk_start:chunk_start + chunk_sources]
                # Phase 1: preprocess each labeled source once; draw each
                # variant's checkpoint + noise scale from its seeded RNG.
                # list_sample_dirs already guaranteed full.png + labels.txt
                # exist, so a missing label here is a genuine read failure and
                # propagates instead of being silently skipped.
                chunk_inputs: list[torch.Tensor] = []
                chunk_specs: list[_ChunkVariant] = []
                for src_idx, sample_dir in enumerate(chunk_samples, start=chunk_start):
                    stem: str = sample_dir.name
                    src_label_lines: list[str] = read_sample_labels(
                        sample_dir, label_file_name, class_name_to_id)
                    vae_input: torch.Tensor = preprocess_for_vae(
                        str(sample_dir / image_file_name), int(hyperparams.VAE_IMAGE_SIZE))
                    input_index: int = len(chunk_inputs)
                    chunk_inputs.append(vae_input)
                    for rep in range(aug_factor):
                        global_index: int = src_idx * aug_factor + rep
                        seed: int = int(hyperparams.BASE_SEED) + global_index
                        rng: random.Random = random.Random(seed)
                        epoch_choice: int = rng.choice(checkpoint_list)
                        noise_std: float = float(rng.choice(noise_list))
                        chunk_specs.append(_ChunkVariant(
                            src_idx=src_idx, rep=rep, seed=seed,
                            epoch_choice=epoch_choice, noise_std=noise_std,
                            input_index=input_index, src_path=str(sample_dir),
                            stem=stem, label_lines=src_label_lines,
                        ))
                    bar.update(1)
                    sources_done += 1
                # Phase 2: group chunk variants by checkpoint epoch — one
                # weight load plus full-batch forwards per group saturates GPU.
                by_epoch: dict[int, list[int]] = {}
                for pos, spec in enumerate(chunk_specs):
                    by_epoch.setdefault(spec.epoch_choice, []).append(pos)
                for epoch, positions in by_epoch.items():
                    for b in range(0, len(positions), vae_batch):
                        group: list[int] = positions[b:b + vae_batch]
                        batch_inputs: torch.Tensor = torch.cat(
                            [chunk_inputs[chunk_specs[p].input_index] for p in group],
                            dim=0,
                        )
                        decoded_batch: torch.Tensor = vae_pool.decode_batch_noisy_mu(
                            batch_inputs,
                            epoch,
                            [chunk_specs[p].noise_std for p in group],
                            [chunk_specs[p].seed for p in group],
                        )
                        for p, row in zip(group, range(decoded_batch.shape[0])):
                            chunk_specs[p].decoded = decoded_batch[row:row + 1]
                # Phase 3: global order — replay the two phase-1 RNG draws so
                # the blur draws match the serial order exactly, then
                # the unchanged CPU post-processing, threaded saves, manifest.
                for spec in chunk_specs:
                    rng = random.Random(spec.seed)
                    _ = rng.choice(checkpoint_list)  # replay phase-1 draws
                    _ = rng.choice(noise_list)
                    if spec.decoded is None:
                        raise RuntimeError(f"Phase 2 missed variant {spec.stem} rep {spec.rep}.")
                    img: Image.Image = decoded_tensor_to_pil(spec.decoded)
                    # 4: photometric patch blur (boxes unaffected, no geometry).
                    img, num_blur = apply_patch_blur(
                        img, rng,
                        int(hyperparams.BLUR_MIN_PATCHES), int(hyperparams.BLUR_MAX_PATCHES),
                        float(hyperparams.BLUR_MIN_SIZE_FRAC), float(hyperparams.BLUR_MAX_SIZE_FRAC),
                        float(hyperparams.BLUR_SIGMA_MIN), float(hyperparams.BLUR_SIGMA_MAX),
                    )
                    # No geometric augmentation: boxes are copied from the
                    # source. read_sample_labels already resolved the class
                    # names to ids in phase 1, so no second pass is needed.
                    out_lines: list[str] = list(spec.label_lines)
                    out_stem: str = f"{spec.stem}_vae{spec.rep:02d}"
                    pending_saves.append(executor.submit(
                        save_variant_files, img, out_lines,
                        str(train_img_dir / f"{out_stem}.png"),
                        str(train_lbl_dir / f"{out_stem}.txt"),
                        compress_level,
                    ))
                    manifest_writer.writerow({
                        "output": f"{out_stem}.png", "source": spec.stem,
                        "checkpoint_epoch": spec.epoch_choice, "noise_std": spec.noise_std,
                        "blur_patches": num_blur, "seed": spec.seed,
                    })
                    train_count += 1
                # Surface worker failures early without stalling on I/O.
                pending_saves = raise_if_save_failed(pending_saves)
                if sources_done >= next_log_at:
                    next_log_at += progress_every
                    elapsed: float = time.time() - loop_start
                    rate: float = train_count / elapsed if elapsed > 0 else 0.0
                    remaining: float = (total_outputs - train_count) / rate if rate > 0 else 0.0
                    log.info(
                        f"Augmented {sources_done}/{len(source_samples)} sources "
                        f"({train_count} outputs, {rate:.1f} img/s, ETA {remaining / 60.0:.1f} min).")
        # Final blocking drain: every save must land before we report counts.
        for fut in pending_saves:
            fut.result()
        pending_saves = []
    finally:
        executor.shutdown(wait=True)
        manifest_file.close()
    total_time: float = time.time() - loop_start
    log.info(f"Wrote {train_count} augmented train images + manifest {manifest_path} "
             f"in {total_time / 60.0:.1f} min ({train_count / total_time:.1f} img/s).")

    # Mirror val (required) and test (optional) resized to the shared square
    # domain. Normalized boxes are resize-invariant, so only class names are
    # resolved to ids. The untouched test split backs the final unbiased
    # evaluation.
    vae_size: int = int(hyperparams.VAE_IMAGE_SIZE)
    if split_complete(
        config.SOURCE_VAL_SPLIT_DIR, image_file_name, label_file_name, val_img_dir
    ):
        val_count: int = len(list(val_img_dir.glob("*.png")))
        log.info(f"Val mirror complete ({val_count} images); skipping re-encode.")
    else:
        val_count = mirror_split(
            config.SOURCE_VAL_SPLIT_DIR, val_img_dir, val_lbl_dir, vae_size,
            image_file_name, label_file_name, class_name_to_id,
        )
        log.info(f"Mirrored {val_count} val images.")
    test_count: int = 0
    test_split_src: str = str(getattr(config, "SOURCE_TEST_SPLIT_DIR", ""))
    if test_split_src and os.path.isdir(test_split_src):
        if split_complete(
            test_split_src, image_file_name, label_file_name, test_img_dir
        ):
            test_count = len(list(test_img_dir.glob("*.png")))
            log.info(f"Test mirror complete ({test_count} images); skipping re-encode.")
        else:
            test_count = mirror_split(
                test_split_src, test_img_dir, test_lbl_dir, vae_size,
                image_file_name, label_file_name, class_name_to_id)
            log.info(f"Mirrored {test_count} test images.")
    else:
        log.warning("Test source dir missing; skipping test mirroring.")
    return {"train_images": train_count, "val_images": val_count, "test_images": test_count}


def write_data_yaml(
    config: SimpleNamespace,
    hyperparams: SimpleNamespace,
) -> str:
    """Write the Ultralytics ``data.yaml`` for the augmented dataset.

    Args:
        config (SimpleNamespace): Runtime config with dataset root + yaml path.
        hyperparams (SimpleNamespace): Holds CLASS_NAMES (the furniture
            vocabulary; the list index is the numeric class id written to
            ``data.yaml`` and to every generated label file).

    Returns:
        str: Absolute path of the written yaml file.
    """
    root_abs: str = str(Path(config.AUGMENTED_DATASET_ROOT).resolve())
    yaml_path: Path = Path(config.DATA_YAML_PATH)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = list(hyperparams.CLASS_NAMES)
    # Quote the path: dataset roots may contain spaces.
    lines: list[str] = [
        f'path: "{root_abs}"',
        "train: images/train",
        "val: images/val",
        # Held-out split for the final unbiased report (never early-stopped on).
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(yaml_path)
