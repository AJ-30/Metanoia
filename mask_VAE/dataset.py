"""
Dataset for mask_VAE: reads the integer semantic segmentation masks of a split.

The dataset is folder-per-sample: a split folder holds one sub-folder per
layout, and each sub-folder carries every render of that layout. This module
reads **only** the segmentation masks selected by ``MASK_MODE`` in
``mask_VAE/config.toml`` — the rendered floor-plan images are never opened,
because this model is unconditional and its target is the mask itself.

Each item serves two tensors of the same masks: the single-channel float
tensor the custom encoder reads (one ``0..1`` class-id channel per group)
and the one-vs-rest binary tensor the BCE loss is computed against.

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

from mask_VAE.masking import (
    MaskLayout,
    encode_masks_to_channels,
    encode_masks_to_single_channel,
)

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
    Torch dataset yielding the mask tensors of each sample.

    One item is a ``(single, onehot)`` pair for a single layout: ``single`` is
    the Float32 ``(num_groups, image_size, image_size)`` class-id tensor in
    ``0..1`` the custom encoder reads, and ``onehot`` is the Float32
    ``(num_channels, image_size, image_size)`` binary tensor the BCE loss is
    computed against.

    Args:
        split_dir (str): Split folder holding one sub-folder per sample.
        layout (MaskLayout): Channel layout to encode into. Also fixes the mask
            filenames, via the groups it carries.
        image_size (int): Square side length the masks are resized to.
        max_samples (int): Cap on the number of samples. A negative value keeps
            every sample found.
        preload (bool): Decode and encode every sample once at construction and
            keep it in RAM as ``uint8``. Costs
            ``(num_groups + num_channels) * image_size**2`` bytes per sample.

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

        # Forward reference so the cache can be typed as a list of pairs.
        self.cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if preload:
            self.cache = [self._load_uint8(sample_dir) for sample_dir in self.sample_dirs]

    def __len__(self) -> int:
        """
        Number of samples in the split.

        Returns:
            int: Length of the resolved sample list.
        """
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the encoded masks of one sample.

        Args:
            index (int): Position in the sorted sample list.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(single, onehot)`` Float32
                tensors — ``single`` of shape
                ``(num_groups, image_size, image_size)`` in ``0..1`` for the
                encoder, ``onehot`` of shape
                ``(num_channels, image_size, image_size)`` of ``0.0``/``1.0``
                for the loss.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if self.cache is not None:
            cached_single, cached_onehot = self.cache[index]
            return self._decode_single(cached_single), cached_onehot.float() / 255.0
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

    def _load_ids(self, sample_dir: Path) -> list[torch.Tensor]:
        """
        Read every selected mask of one sample as class-id tensors.

        Classes the layout excludes (e.g. ``tv`` in ``arrangement_filtered``)
        are remapped to background id ``0`` here, so the RAM cache, the
        encoder input and the one-vs-rest target all agree that an excluded
        pixel is background.

        Args:
            sample_dir (Path): Sample folder holding the mask files.

        Returns:
            list[torch.Tensor]: One int64 ``(image_size, image_size)`` tensor
                per group, ordered like ``layout.groups``.
        """
        return [
            group.remap_ids_to_background(
                self._load_mask(sample_dir / group.mask_file_name)
            )
            for group in self.layout.groups
        ]

    def _encode(
        self, sample_dir: Path
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read and encode every selected mask of one sample.

        Args:
            sample_dir (Path): Sample folder holding the mask files.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(single, onehot)`` Float32
                tensors. The batch dimension the encoders introduce for a
                single sample is squeezed off here, so a dataset item is
                exactly the tensor shapes the model takes before batching.
        """
        mask_arrays: list[torch.Tensor] = self._load_ids(sample_dir)
        single: torch.Tensor = encode_masks_to_single_channel(
            mask_arrays, self.layout
        ).squeeze(0)
        onehot: torch.Tensor = encode_masks_to_channels(
            mask_arrays, self.layout
        ).squeeze(0)
        return single, onehot

    def _decode_single(self, cached_ids: torch.Tensor) -> torch.Tensor:
        """
        Restore the encoder input from cached class ids.

        The cached ids are exact, so dividing by each group's id range
        reproduces the non-cached encoding bit-identically.

        Args:
            cached_ids (torch.Tensor): ``uint8`` ``(num_groups, H, W)`` ids.

        Returns:
            torch.Tensor: Float32 ``(num_groups, H, W)`` tensor in ``0..1``.
        """
        channels: list[torch.Tensor] = []
        for index, group in enumerate(self.layout.groups):
            scale: float = float(max(group.num_classes - 1, 1))
            channels.append(cached_ids[index].float().unsqueeze(0) / scale)
        return torch.cat(channels, dim=0)

    def _load_uint8(
        self, sample_dir: Path
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read one sample's masks as ``uint8`` for the RAM cache.

        The class ids fit in ``uint8`` natively and the one-vs-rest tensor is
        strictly ``0.0`` or ``1.0``, so both round-trip through ``uint8``
        exactly. That makes the cache four times smaller than keeping floats,
        and ``__getitem__`` scales both back on the way out.

        Args:
            sample_dir (Path): Sample folder holding the mask files.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``uint8`` ``(single_ids,
                onehot)`` pair matching ``_encode`` shapes.
        """
        mask_arrays: list[torch.Tensor] = self._load_ids(sample_dir)
        single_ids: torch.Tensor = torch.stack(
            [array.to(torch.uint8) for array in mask_arrays], dim=0
        )
        onehot: torch.Tensor = (
            encode_masks_to_channels(mask_arrays, self.layout).squeeze(0) * 255.0
        ).round().to(torch.uint8)
        return single_ids, onehot

    def _load_float(
        self, sample_dir: Path
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read one sample's masks on demand, without caching.

        Args:
            sample_dir (Path): Sample folder holding the mask files.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(single, onehot)`` Float32
                tensors, straight from disk.
        """
        return self._encode(sample_dir)
