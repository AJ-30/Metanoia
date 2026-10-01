"""
Dataset for mask_VAE: reads the integer semantic segmentation masks of a split.

The dataset is folder-per-sample: a split folder holds one sub-folder per
layout, and each sub-folder carries every render of that layout. This module
reads **only** the segmentation masks selected by ``MASK_GROUPS`` in
``mask_VAE/config.toml`` — the rendered floor-plan images are never opened,
because this model is unconditional and its target is the mask itself.

Two details of the source data drive the implementation:

* The masks are single-channel 8-bit ``L`` PNGs holding plain integer class
  ids, not colors. Reading a colored image here would produce a tensor of
  arbitrary numbers that silently means nothing, so the image mode is checked
  on every load.
* Resizing must use nearest-neighbour interpolation. Any interpolating filter
  would blend neighbouring class ids into values that belong to no class at all,
  so the one-vs-rest target would stop being a valid one-vs-rest target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mask_VAE.masking import MaskLayout, encode_masks_to_channels

# Image mode the source masks are stored in, and the mode they are read back as.
MASK_IMAGE_MODE: str = "L"

# Image modes accepted when opening a source mask. Palette images are accepted
# because Pillow reports them as "P", but they are converted to single channel
# before use: the payload is class ids either way, not colors.
ACCEPTED_MASK_IMAGE_MODES: frozenset[str] = frozenset({"L", "P"})

# Resampling filter used to resize the class-id masks. Never change this to an
# interpolating filter, for the reason given in the module docstring.
MASK_RESAMPLE: Image.Resampling = Image.Resampling.NEAREST


class MaskDataset(Dataset):
    """
    Torch dataset yielding the one-vs-rest binary mask tensor of each sample.

    One item is the encoded mask of a single layout, shaped
    ``(num_channels, image_size, image_size)`` and valued ``0.0`` or ``1.0``.
    The same encoding is used for the loss target and for the encoder input, so
    an item doubles as both.

    Args:
        split_dir (str): Split folder holding one sub-folder per sample.
        layout (MaskLayout): Channel layout to encode into. Also fixes the mask
            filenames, via the groups it carries.
        image_size (int): Square side length the masks are resized to.
        max_samples (int): Cap on the number of samples. A negative value keeps
            every sample found.
        preload (bool): Decode and encode every sample once at construction and
            keep it in RAM as ``uint8``. Costs
            ``num_channels * image_size**2`` bytes per sample.

    Raises:
        ValueError: If ``image_size`` is not positive.
        FileNotFoundError: If the split folder is missing, or if no sample
            folder provides all the required masks.
    """

    def __init__(
        self,
        split_dir: str,
        layout: MaskLayout,
        image_size: int,
        max_samples: int,
        preload: bool = False,
    ) -> None:
        if image_size < 1:
            raise ValueError(f"image_size must be positive, got {image_size}.")

        self.split_dir: str = split_dir
        self.layout: MaskLayout = layout
        self.image_size: int = image_size

        self.sample_dirs: list[Path] = self._list_complete_sample_dirs(split_dir, layout)
        if max_samples >= 0:
            self.sample_dirs = self.sample_dirs[:max_samples]

        # Forward reference so the cache can be typed as a list of tensors.
        self.cache: list[torch.Tensor] | None = None
        if preload:
            self.cache = [self._load_uint8(sample_dir) for sample_dir in self.sample_dirs]

    def __len__(self) -> int:
        """
        Number of samples in the split.

        Returns:
            int: Length of the resolved sample list.
        """
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Return the encoded mask of one sample.

        Args:
            index (int): Position in the sorted sample list.

        Returns:
            torch.Tensor: Float32 ``(num_channels, image_size, image_size)``
                binary tensor, ``0.0`` and ``1.0`` only.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if self.cache is not None:
            return self.cache[index].float() / 255.0
        return self._load_float(self.sample_dirs[index])

    def _list_complete_sample_dirs(
        self,
        split_dir: str,
        layout: MaskLayout,
    ) -> list[Path]:
        """
        Find every sample folder of the split that provides all selected masks.

        Iterates non-recursively so a stray nested folder is never mistaken for
        a sample, and sorts the result so the dataset order is stable across
        machines and across runs.

        Args:
            split_dir (str): Split folder to scan.
            layout (MaskLayout): Layout whose groups name the required masks.

        Returns:
            list[Path]: Sorted sample folders, one entry per complete sample.

        Raises:
            FileNotFoundError: If the split folder is missing, or if no sample
                folder provides every required mask.
        """
        split_path: Path = Path(split_dir)
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Dataset split directory missing: {split_dir}. The masks live "
                "next to the rendered images, so the split must be the folder "
                "holding the per-sample sub-folders."
            )

        required_files: list[str] = [group.mask_file_name for group in layout.groups]
        sample_dirs: list[Path] = sorted(
            candidate
            for candidate in split_path.iterdir()
            if candidate.is_dir()
            and all((candidate / name).is_file() for name in required_files)
        )
        if not sample_dirs:
            raise FileNotFoundError(
                f"No sample folder under {split_dir} provides all of "
                f"{required_files}. Check the split path and MASK_GROUPS in "
                "mask_VAE/config.toml."
            )
        return sample_dirs

    def _load_mask(self, mask_path: Path) -> torch.Tensor:
        """
        Read one mask PNG as a class-id tensor.

        Args:
            mask_path (Path): Path of the ``*_segmentation.png`` file.

        Returns:
            torch.Tensor: int64 ``(image_size, image_size)`` tensor of class
                ids, resized with nearest-neighbour interpolation.

        Raises:
            FileNotFoundError: If the mask file is missing.
            ValueError: If the image is not a single-channel 8-bit mask, which
                means the wrong render was configured.
        """
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask file disappeared after indexing: {mask_path}.")

        with Image.open(mask_path) as image:
            # A colored render here means the configured filename points at an
            # image instead of at the integer class-id mask. Fail loudly rather
            # than feeding meaningless values to the loss.
            if image.mode not in ACCEPTED_MASK_IMAGE_MODES:
                raise ValueError(
                    f"{mask_path} has image mode {image.mode!r}, but a "
                    f"{MASK_IMAGE_MODE!r} integer class-id mask is required. "
                    "Check MASK_FILES in mask_VAE/config.toml."
                )
            # convert(MASK_IMAGE_MODE) collapses a palette image to its class
            # ids, since a "P" image stores the palette out of band.
            resized: Image.Image = image.convert(MASK_IMAGE_MODE).resize(
                (self.image_size, self.image_size),
                resample=MASK_RESAMPLE,
            )
            array: np.ndarray = np.array(resized, dtype=np.uint8)
        if array.ndim != 2:
            raise ValueError(
                f"{mask_path} decoded to shape {array.shape}; a mask must be a "
                "single 2-D class-id map."
            )
        return torch.from_numpy(array.astype(np.int64))

    def _encode(self, sample_dir: Path) -> torch.Tensor:
        """
        Read and encode every selected mask of one sample.

        Args:
            sample_dir (Path): Sample folder holding the mask files.

        Returns:
            torch.Tensor: Float32 ``(num_channels, image_size, image_size)``
                binary tensor. The batch dimension the encoder introduces for a
                single sample is squeezed off here, so a dataset item is exactly
                the tensor shape the model takes before batching.
        """
        mask_arrays: list[torch.Tensor] = [
            self._load_mask(sample_dir / group.mask_file_name)
            for group in self.layout.groups
        ]
        return encode_masks_to_channels(mask_arrays, self.layout).squeeze(0)

    def _load_uint8(self, sample_dir: Path) -> torch.Tensor:
        """
        Read one sample's masks as ``uint8`` for the RAM cache.

        The encoded tensor is strictly ``0.0`` or ``1.0``, so it round-trips
        through ``uint8`` exactly. That makes the cache four times smaller than
        keeping floats, and ``__getitem__`` scales it back on the way out.

        Args:
            sample_dir (Path): Sample folder holding the mask files.

        Returns:
            torch.Tensor: ``uint8`` tensor of the same shape as ``_encode``.
        """
        return (self._encode(sample_dir) * 255.0).round().to(torch.uint8)

    def _load_float(self, sample_dir: Path) -> torch.Tensor:
        """
        Read one sample's masks on demand, without caching.

        Args:
            sample_dir (Path): Sample folder holding the mask files.

        Returns:
            torch.Tensor: Float32 binary channel tensor, straight from disk.
        """
        return self._encode(sample_dir)
