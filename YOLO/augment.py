"""Frozen-VAE latent augmentation pipeline producing a 15x YOLO dataset.

Pipeline per source image (repeated AUG_FACTOR times with distinct seeds):
  1. Load RGBA image, flatten alpha onto white, resize to VAE_IMAGE_SIZE square.
  2. Pick a frozen VAE checkpoint uniformly from CHECKPOINT_EPOCHS, encode to
     the latent mean ``mu`` (deterministic; no reparameterization sampling).
  3. Add Gaussian noise ``N(0, s^2)`` to ``mu`` and decode, where the scale ``s``
     is drawn uniformly per variant from ``LATENT_NOISE_SCALES`` (0.0 = no noise).
  4. Apply Gaussian blur to 1-3 random patches (photometric only; boxes untouched).
  5. Rotate by k*90 degrees CW (exact box transform, no approximation).
  6. Randomly flip horizontally / vertically (exact box transform).

All geometry stays in normalized YOLO ``cx cy w h`` coordinates. The VAE
resize is a full-image warp so normalized boxes are invariant to it; only the
explicit rot90 + flip steps below change the boxes, and each has a closed form.

Reproducibility: every output index derives ``seed = BASE_SEED + global_index``
so the full 226k-image run is exactly reproducible from the TOML files plus a
per-row manifest (source file, checkpoint epoch, k, flips, blur, seed).
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


def parse_box_line(line: str) -> tuple[int, float, float, float, float]:
    """Parse one YOLO label line into typed components.

    Args:
        line (str): Raw line of form ``"<cls> <cx> <cy> <w> <h>"`` with
            normalized coordinates in [0, 1].

    Returns:
        tuple[int, float, float, float, float]: (cls_id, cx, cy, w, h).

    Raises:
        ValueError: If the line does not have exactly 5 fields or the
            coordinates fall outside [0, 1].
    """
    parts: list[str] = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 fields per label line, got {len(parts)}: {line!r}")
    cls_id: int = int(parts[0])
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
        cls_id (int): Class id (0-9).
        cx (float): Normalized center x.
        cy (float): Normalized center y.
        w (float): Normalized width.
        h (float): Normalized height.

    Returns:
        str: YOLO label line with 6 decimal places.
    """
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def transform_box_rot90_flip(
    cx: float,
    cy: float,
    w: float,
    h: float,
    rot_k: int,
    flip_h: bool,
    flip_v: bool,
) -> tuple[float, float, float, float]:
    """Apply the exact rot90-then-flip transform to one normalized box.

    Rotation is k * 90 degrees clockwise about the image center, applied
    FIRST; flips are applied AFTER rotation. All operations are exact on
    axis-aligned boxes because angles are multiples of 90 degrees.

    Args:
        cx (float): Normalized center x in [0, 1].
        cy (float): Normalized center y in [0, 1].
        w (float): Normalized width.
        h (float): Normalized height.
        rot_k (int): Clockwise quarter-turns, one of {0, 1, 2, 3}.
        flip_h (bool): Whether a horizontal flip follows the rotation.
        flip_v (bool): Whether a vertical flip follows the rotation.

    Returns:
        tuple[float, float, float, float]: Transformed (cx, cy, w, h),
            clipped to [0, 1].

    Raises:
        ValueError: If rot_k is not in {0, 1, 2, 3}.
    """
    if rot_k not in (0, 1, 2, 3):
        raise ValueError(f"rot_k must be 0-3 (quarter turns CW), got {rot_k}")
    # Closed-form rotation about the unit-square center (x right, y DOWN,
    # matching PIL image coordinates). Verified against PIL transposes:
    # k=1 (90 CW, PIL ROTATE_270) sends TL(0,0) -> TR(1,0): (x,y) -> (1-y,x).
    # k=2 (180, PIL ROTATE_180) sends TL -> BR: (x,y) -> (1-x,1-y).
    # k=3 (270 CW = 90 CCW, PIL ROTATE_90) sends TL -> BL: (x,y) -> (y,1-x).
    # Quarter turns swap w/h; half turns leave sizes unchanged.
    if rot_k == 1:
        cx, cy, w, h = 1.0 - cy, cx, h, w
    elif rot_k == 2:
        cx, cy = 1.0 - cx, 1.0 - cy
    elif rot_k == 3:
        cx, cy, w, h = cy, 1.0 - cx, h, w
    # Flips mirror the center coordinate; sizes are unaffected.
    if flip_h:
        cx = 1.0 - cx
    if flip_v:
        cy = 1.0 - cy
    # Clip defensively against float drift at the borders.
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    return cx, cy, w, h


def transform_label_lines(
    lines: list[str],
    rot_k: int,
    flip_h: bool,
    flip_v: bool,
) -> list[str]:
    """Transform a full label file's lines through rot90 + flip.

    Args:
        lines (list[str]): Raw YOLO label lines (may include blanks).
        rot_k (int): Clockwise quarter-turns {0, 1, 2, 3}.
        flip_h (bool): Horizontal flip after rotation.
        flip_v (bool): Vertical flip after rotation.

    Returns:
        list[str]: Transformed label lines (blank lines dropped;
            degenerate zero-area boxes dropped).
    """
    out_lines: list[str] = []
    for raw in lines:
        if not raw.strip():
            continue
        cls_id, cx, cy, w, h = parse_box_line(raw)
        cx, cy, w, h = transform_box_rot90_flip(cx, cy, w, h, rot_k, flip_h, flip_v)
        # Drop boxes that collapsed to zero area (can happen at exact borders).
        if w <= 0.0 or h <= 0.0:
            continue
        out_lines.append(format_box_line(cls_id, cx, cy, w, h))
    return out_lines


def rotate_pil_image(img: Image.Image, rot_k: int) -> Image.Image:
    """Rotate a square PIL image by k * 90 degrees clockwise.

    Args:
        img (Image.Image): Square input image.
        rot_k (int): Clockwise quarter-turns {0, 1, 2, 3}.

    Returns:
        Image.Image: Rotated image (same size; square so no canvas change).
    """
    if rot_k == 0:
        return img
    # PIL's ROTATE_90/180/270 are counter-clockwise; 90 CW == 270 CCW.
    method = {1: Image.ROTATE_270, 2: Image.ROTATE_180, 3: Image.ROTATE_90}[rot_k]
    return img.transpose(method)


def flip_pil_image(img: Image.Image, flip_h: bool, flip_v: bool) -> Image.Image:
    """Apply horizontal / vertical flips to a PIL image.

    Args:
        img (Image.Image): Input image.
        flip_h (bool): Mirror left-right.
        flip_v (bool): Mirror top-bottom.

    Returns:
        Image.Image: Flipped image (may be the original object if no flips).
    """
    if flip_h:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


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
    src_img_dir: str,
    src_lbl_dir: str,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    vae_size: int,
) -> int:
    """Mirror a source split resized to the VAE square domain, labels verbatim.

    Normalized YOLO boxes are invariant to full-image resize, so label files
    are copied unchanged; images are flattened from RGBA and resized.

    Args:
        src_img_dir (str): Source images directory.
        src_lbl_dir (str): Source labels directory.
        dst_img_dir (Path): Destination images directory (created if needed).
        dst_lbl_dir (Path): Destination labels directory (created if needed).
        vae_size (int): Square side length both domains share.

    Returns:
        int: Number of images mirrored.
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    clear_generated_split(dst_img_dir, dst_lbl_dir)
    count: int = 0
    for src in sorted(
        p for p in Path(src_img_dir).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ):
        img_v: Image.Image = flatten_alpha(Image.open(src)).resize(
            (vae_size, vae_size), Image.BILINEAR)
        img_v.save(str(dst_img_dir / f"{src.stem}.png"))
        src_lbl: Path = Path(src_lbl_dir) / f"{src.stem}.txt"
        if src_lbl.is_file():
            with open(src_lbl, encoding="utf-8") as f:
                content: str = f.read()
            with open(dst_lbl_dir / f"{src.stem}.txt", "w", encoding="utf-8") as out:
                out.write(content)
        count += 1
    return count


def split_complete(src_img_dir: str, dst_img_dir: Path) -> bool:
    """Check whether every source image already has a mirrored PNG.

    Compares filename stem sets (cheap readdir, no image reads) so a completed
    val/test mirror is skipped instead of re-encoded on every regen.

    Args:
        src_img_dir (str): Source images directory.
        dst_img_dir (Path): Mirrored destination images directory.

    Returns:
        bool: True when the destination holds exactly the source stem set.
    """
    if not dst_img_dir.is_dir():
        return False
    valid_suffixes: set[str] = {".png", ".jpg", ".jpeg"}
    src_stems: set[str] = {
        p.stem for p in Path(src_img_dir).iterdir()
        if p.suffix.lower() in valid_suffixes
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

    For each source train image, AUG_FACTOR variants are produced. Each
    variant draws an independent seed, a uniform checkpoint epoch, seeded VAE
    latent noise, blur patches, a rot90 quarter-turn and flips. Images land in
    ``<root>/images/train`` and labels in ``<root>/labels/train``; the val and
    test splits are mirrored resized to the VAE square domain with labels
    copied verbatim (normalized boxes are resize-invariant). A manifest CSV
    records every train variant's provenance. Train/val/test output dirs are
    cleared first so re-runs with smaller configs leave no orphaned files.

    Args:
        config (SimpleNamespace): Runtime config with source/dest directories,
            VAE location, run id and device string.
        hyperparams (SimpleNamespace): All augmentation knobs (factor, seeds,
            noise std, blur ranges, flip probabilities, class names).
        log (SimpleNamespace): Configured project logger.

    Returns:
        dict[str, int]: Counts {"train_images", "val_images", "test_images"}.

    Raises:
        FileNotFoundError: If a required source image/label directory is missing.
        ValueError: If VAE_IMAGE_SIZE disagrees with the checkpoint architecture.
    """
    for key in (
        "SOURCE_TRAIN_IMAGES_DIR",
        "SOURCE_TRAIN_LABELS_DIR",
        "SOURCE_VAL_IMAGES_DIR",
        "SOURCE_VAL_LABELS_DIR",
    ):
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

    source_images: list[str] = sorted(
        str(p) for p in Path(config.SOURCE_TRAIN_IMAGES_DIR).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    max_samples: int = int(hyperparams.MAX_SAMPLES)
    if max_samples >= 0:
        source_images = source_images[:max_samples]
    aug_factor: int = int(hyperparams.AUG_FACTOR)
    log.info(f"Augmenting {len(source_images)} source images x{aug_factor} "
             f"= {len(source_images) * aug_factor} outputs.")

    manifest_path: Path = Path(config.AUGMENT_MANIFEST_PATH)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_file = open(manifest_path, "w", newline="", encoding="utf-8")
    manifest_writer = csv.DictWriter(
        manifest_file,
        fieldnames=["output", "source", "checkpoint_epoch", "noise_std", "rot_k",
                    "flip_h", "flip_v", "blur_patches", "seed"],
    )
    manifest_writer.writeheader()

    train_count: int = 0
    pending_saves: list[Future[None]] = []
    save_workers: int = max(1, int(getattr(hyperparams, "SAVE_WORKERS", 1)))
    compress_level: int = int(getattr(hyperparams, "PNG_COMPRESS_LEVEL", 6))
    progress_every: int = max(1, int(getattr(hyperparams, "PROGRESS_LOG_SOURCES", 200)))
    vae_batch: int = max(1, int(getattr(hyperparams, "VAE_BATCH_SIZE", 32)))
    chunk_sources: int = max(1, int(getattr(hyperparams, "AUG_CHUNK_SOURCES", 32)))
    total_outputs: int = len(source_images) * aug_factor
    loop_start: float = time.time()
    # VAE forwards stay on the main thread (MPS is not thread-safe for
    # concurrent forwards) but run in full batches grouped by checkpoint epoch
    # (one weight load per group); only PNG/label file writes are offloaded to
    # the pool. Manifest rows are written on the main thread in global order,
    # so output bytes are identical to a serial run.
    checkpoint_list: list[int] = list(hyperparams.CHECKPOINT_EPOCHS)
    noise_list: list[float] = [float(s) for s in hyperparams.LATENT_NOISE_SCALES]
    rotation_list: list[int] = list(hyperparams.ROTATION_CHOICES)
    executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=save_workers)
    try:
        with tqdm(total=len(source_images), desc="VAE augment", unit="img") as bar:
            sources_done: int = 0
            next_log_at: int = progress_every
            for chunk_start in range(0, len(source_images), chunk_sources):
                chunk_paths: list[str] = source_images[chunk_start:chunk_start + chunk_sources]
                # Phase 1: preprocess each labeled source once; draw each
                # variant's checkpoint + noise scale from its seeded RNG.
                chunk_inputs: list[torch.Tensor] = []
                chunk_specs: list[_ChunkVariant] = []
                for src_idx, src_path in enumerate(chunk_paths, start=chunk_start):
                    stem: str = Path(src_path).stem
                    label_path: Path = Path(config.SOURCE_TRAIN_LABELS_DIR) / f"{stem}.txt"
                    if not label_path.is_file():
                        log.warning(f"No label file for {stem}; skipping.")
                        bar.update(1)
                        sources_done += 1
                        continue
                    with open(label_path, encoding="utf-8") as f:
                        src_label_lines: list[str] = f.read().splitlines()
                    vae_input: torch.Tensor = preprocess_for_vae(
                        src_path, int(hyperparams.VAE_IMAGE_SIZE))
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
                            input_index=input_index, src_path=src_path,
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
                # the blur/rot/flip draws match the serial order exactly, then
                # the unchanged CPU post-processing, threaded saves, manifest.
                for spec in chunk_specs:
                    rng = random.Random(spec.seed)
                    _ = rng.choice(checkpoint_list)  # replay phase-1 draws
                    _ = rng.choice(noise_list)
                    if spec.decoded is None:
                        raise RuntimeError(f"Phase 2 missed variant {spec.stem} rep {spec.rep}.")
                    img: Image.Image = decoded_tensor_to_pil(spec.decoded)
                    # 4: photometric patch blur (boxes unaffected).
                    img, num_blur = apply_patch_blur(
                        img, rng,
                        int(hyperparams.BLUR_MIN_PATCHES), int(hyperparams.BLUR_MAX_PATCHES),
                        float(hyperparams.BLUR_MIN_SIZE_FRAC), float(hyperparams.BLUR_MAX_SIZE_FRAC),
                        float(hyperparams.BLUR_SIGMA_MIN), float(hyperparams.BLUR_SIGMA_MAX),
                    )
                    # 5-6: geometric steps with exact box counterparts.
                    rot_k: int = rng.choice(rotation_list)
                    flip_h: bool = rng.random() < float(hyperparams.FLIP_H_PROB)
                    flip_v: bool = rng.random() < float(hyperparams.FLIP_V_PROB)
                    img = flip_pil_image(rotate_pil_image(img, rot_k), flip_h, flip_v)
                    out_lines: list[str] = transform_label_lines(
                        spec.label_lines, rot_k, flip_h, flip_v)
                    out_stem: str = f"{spec.stem}_vae{spec.rep:02d}"
                    pending_saves.append(executor.submit(
                        save_variant_files, img, out_lines,
                        str(train_img_dir / f"{out_stem}.png"),
                        str(train_lbl_dir / f"{out_stem}.txt"),
                        compress_level,
                    ))
                    manifest_writer.writerow({
                        "output": f"{out_stem}.png", "source": Path(spec.src_path).name,
                        "checkpoint_epoch": spec.epoch_choice, "noise_std": spec.noise_std,
                        "rot_k": rot_k, "flip_h": flip_h, "flip_v": flip_v,
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
                        f"Augmented {sources_done}/{len(source_images)} sources "
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
    # domain. Normalized boxes are resize-invariant, hence copied verbatim.
    # The untouched test split backs the final unbiased evaluation.
    vae_size: int = int(hyperparams.VAE_IMAGE_SIZE)
    if split_complete(config.SOURCE_VAL_IMAGES_DIR, val_img_dir):
        val_count: int = len(list(val_img_dir.glob("*.png")))
        log.info(f"Val mirror complete ({val_count} images); skipping re-encode.")
    else:
        val_count = mirror_split(
            config.SOURCE_VAL_IMAGES_DIR, config.SOURCE_VAL_LABELS_DIR,
            val_img_dir, val_lbl_dir, vae_size,
        )
        log.info(f"Mirrored {val_count} val images.")
    test_count: int = 0
    test_img_src: str = str(getattr(config, "SOURCE_TEST_IMAGES_DIR", ""))
    test_lbl_src: str = str(getattr(config, "SOURCE_TEST_LABELS_DIR", ""))
    if test_img_src and test_lbl_src and os.path.isdir(test_img_src) and os.path.isdir(test_lbl_src):
        if split_complete(test_img_src, test_img_dir):
            test_count = len(list(test_img_dir.glob("*.png")))
            log.info(f"Test mirror complete ({test_count} images); skipping re-encode.")
        else:
            test_count = mirror_split(
                test_img_src, test_lbl_src, test_img_dir, test_lbl_dir, vae_size)
            log.info(f"Mirrored {test_count} test images.")
    else:
        log.warning("Test source dirs missing; skipping test mirroring.")
    return {"train_images": train_count, "val_images": val_count, "test_images": test_count}


def write_data_yaml(
    config: SimpleNamespace,
    hyperparams: SimpleNamespace,
) -> str:
    """Write the Ultralytics ``data.yaml`` for the augmented dataset.

    Args:
        config (SimpleNamespace): Runtime config with dataset root + yaml path.
        hyperparams (SimpleNamespace): Holds CLASS_NAMES (10 furniture names).

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
