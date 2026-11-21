"""PyTorch dataset of missing-component cases (thin wrapper over ``CaseSource``).

Each item is a dict:

* ``incomplete`` float32 (C, H, W) in {0, 1}: model input I.
* ``target``     float32 (C, H, W): missing part R = F AND NOT I.
* ``full``       float32 (C, H, W): complete plan F.
* ``mask``       float32 (1, H, W): erased pixels M (oracle / metrics only).
* ``case_id``    str, ``info`` str (JSON metadata; strings collate cleanly).

Training items can be randomly rotated by multiples of 90 degrees and
mirrored (``missing.train.DIHEDRAL_AUGMENT``); call ``set_epoch`` every epoch
so a new case is drawn per plan.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from Reconstruction.data_prep.case_source import TRAIN_SPLIT, CaseSource
from Reconstruction.data_prep.missing import Case


def _dihedral(arrays: list[np.ndarray], rotations: int, flip: bool) -> list[np.ndarray]:
    """Apply the same rotation/flip to arrays whose last two axes are (H, W).

    Args:
        arrays (list[np.ndarray]): Arrays to transform.
        rotations (int): Number of 90-degree rotations (0-3).
        flip (bool): Mirror left-right after rotating.

    Returns:
        list[np.ndarray]: Transformed contiguous copies.
    """
    out: list[np.ndarray] = []
    for array in arrays:
        moved: np.ndarray = np.rot90(array, k=rotations, axes=(-2, -1))
        out.append(np.ascontiguousarray(moved[..., ::-1] if flip else moved))
    return out


class MissingComponentDataset(Dataset):
    """Cases of one split as tensors."""

    def __init__(self, processed_dir: Path, split: str, seed: int, augment: bool | None = None) -> None:
        """Open a split.

        Args:
            processed_dir (Path): ``PROCESSED_ROOT/<dataset>`` written by build_dataset.py.
            split (str): "train", "val" or "test".
            seed (int): Seed for on-the-fly training cases and augmentation.
            augment (bool | None): Dihedral augmentation; None uses
                ``missing.train.DIHEDRAL_AUGMENT`` for train and False otherwise.
        """
        self.source = CaseSource(processed_dir, split, seed)
        self.seed: int = seed
        self.epoch: int = 0
        default_augment: bool = split == TRAIN_SPLIT and self.source.params.missing.train.DIHEDRAL_AUGMENT
        self.augment: bool = default_augment if augment is None else augment

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch, which changes the on-the-fly training cases.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Number of plans (train) or frozen cases (val/test)."""
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Tensors and metadata of one case (see module docstring).

        Args:
            index (int): Dataset index.

        Returns:
            dict[str, Any]: Item dict.
        """
        case: Case = self.source.case(index, self.epoch)
        arrays: list[np.ndarray] = [case.incomplete, case.target, case.full, case.mask[None]]
        if self.augment:
            # Separate random stream from case sampling, so toggling
            # augmentation does not change which cases are drawn.
            aug_rng: np.random.Generator = np.random.default_rng([self.seed, self.epoch, index, 1])
            arrays = _dihedral(arrays, int(aug_rng.integers(4)), bool(aug_rng.random() < 0.5))
        incomplete, target, full, mask = (torch.from_numpy(a.astype(np.float32)) for a in arrays)
        return {"incomplete": incomplete, "target": target, "full": full, "mask": mask,
                "case_id": case.info["case_id"], "info": json.dumps(case.info)}
