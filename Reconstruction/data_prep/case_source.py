"""Framework-free access to the missing-component cases of one split.

* train: a new random case per plan per epoch. Randomness is keyed on
  (seed, epoch, index), so results do not depend on worker count or order.
* val / test: the frozen case list written by ``build_dataset.py``
  (``cases_<split>.jsonl``), regenerated exactly from the stored specs.

Used by the PyTorch dataset (``dataset.py``) and by ``visualize.py``.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from Reconstruction.common.config_loader import load_params
from Reconstruction.data_prep.missing import Case, CaseSpec, MissingComponentGenerator
from Reconstruction.data_prep.plan_store import PlanStore
from Reconstruction.data_prep.schema import Schema

TRAIN_SPLIT: str = "train"
# Files written next to the split stores by build_dataset.py.
PARAMS_COPY_FILE: str = "params.toml"
SIZE_BINS_FILE: str = "size_bins.json"
CASES_FILE_TEMPLATE: str = "cases_{split}.jsonl"


def read_case_specs(path: Path) -> list[CaseSpec]:
    """Read a frozen case list.

    Args:
        path (Path): ``cases_<split>.jsonl`` file.

    Returns:
        list[CaseSpec]: Specs in file order.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return [CaseSpec.from_dict(json.loads(line)) for line in handle]


def load_generator(processed_dir: Path) -> tuple[SimpleNamespace, MissingComponentGenerator]:
    """Recreate the case generator exactly as configured at build time.

    Args:
        processed_dir (Path): ``PROCESSED_ROOT/<dataset>`` written by build_dataset.py.

    Returns:
        tuple[SimpleNamespace, MissingComponentGenerator]: Build-time params
        and the generator (with train-split size bins).

    Raises:
        FileNotFoundError: If the params copy or size bins are missing.
    """
    # The params copy saved at build time guarantees the same taxonomy and
    # generation rules that produced the frozen cases.
    params: SimpleNamespace = load_params(processed_dir / PARAMS_COPY_FILE)
    with open(processed_dir / SIZE_BINS_FILE, "r", encoding="utf-8") as handle:
        size_bins: dict[str, list[float]] = json.load(handle)
    return params, MissingComponentGenerator(Schema.from_params(params), params, size_bins)


class CaseSource:
    """Builds the cases of one split from the processed plan store.

    Attributes:
        split (str): Split name.
        schema (Schema): Taxonomy used at build time.
        params (SimpleNamespace): Parameters used at build time.
    """

    def __init__(self, processed_dir: Path, split: str, seed: int) -> None:
        """Open a split.

        Args:
            processed_dir (Path): ``PROCESSED_ROOT/<dataset>`` written by build_dataset.py.
            split (str): "train", "val" or "test".
            seed (int): Seed for on-the-fly training cases.

        Raises:
            FileNotFoundError: If the processed data is incomplete.
        """
        self.params, self._generator = load_generator(processed_dir)
        self.schema: Schema = self._generator.schema
        self.split: str = split
        self._seed: int = seed
        self._store = PlanStore(processed_dir / split)
        self._specs: list[CaseSpec] | None = None
        if split != TRAIN_SPLIT:
            self._specs = read_case_specs(processed_dir / CASES_FILE_TEMPLATE.format(split=split))

    def __len__(self) -> int:
        """Number of plans (train) or frozen cases (val/test)."""
        return len(self._specs) if self._specs is not None else len(self._store)

    def spec(self, index: int, epoch: int) -> CaseSpec:
        """Spec of a case: frozen for val/test, sampled for train.

        Args:
            index (int): Case index (plan index for train).
            epoch (int): Epoch; only affects train sampling.

        Returns:
            CaseSpec: The spec.
        """
        if self._specs is not None:
            return self._specs[index]
        rng: np.random.Generator = np.random.default_rng([self._seed, epoch, index])
        return self._generator.sample_train_spec(self.split, index, self._store.record(index), rng)

    def build(self, spec: CaseSpec) -> Case:
        """Generate the case for a spec of this split.

        Args:
            spec (CaseSpec): Spec whose ``split`` matches this source.

        Returns:
            Case: Generated case.
        """
        assert spec.split == self.split, f"spec of split {spec.split} given to {self.split} source"
        return self._generator.build_case(self._store.layers(spec.plan_index), self._store.record(spec.plan_index), spec)

    def case(self, index: int, epoch: int = 0) -> Case:
        """Generate the case at an index.

        Args:
            index (int): Case index (plan index for train).
            epoch (int): Epoch; only affects train sampling.

        Returns:
            Case: Generated case.
        """
        return self.build(self.spec(index, epoch))
