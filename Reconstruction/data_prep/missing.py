"""Missing-component case generation.

A *case* is built from one rasterized plan and a ``CaseSpec`` naming the
component to remove, the removal mode and the erased margin:

* ``full``        uint8 (C, H, W) semantic map of the complete plan (F).
* ``incomplete``  uint8 (C, H, W) semantic map after removal (I) - the only
                  input the model gets at test time.
* ``target``      uint8 (C, H, W) what is missing: F AND NOT I (R).
* ``mask``        uint8 (H, W) pixels erased (M). Known only to oracles/metrics.
* ``primary``     uint8 (H, W) pixels of the chosen (primary) instance, which
                  metrics use to find the main ground-truth component.
* ``info``        metadata for stratified evaluation (type, size bin,
                  location, neighbours, ...).

Removal modes (see README for the rationale):

* easy - erase only the chosen instance. A removed door/window leaves a gap
  in the wall; a removed room leaves an unlabelled area enclosed by walls.
* hard - door/window: the gap is filled with wall, so the plan looks
  complete. wall: also erase the doors/windows in that wall. room: also erase
  walls and openings that touch no other room.

A margin > 0 additionally erases everything within ``margin`` px of the
removed pixels (missing-region-size sweep).

Specs are small and deterministic, so frozen evaluation sets are stored as
spec lists and regenerated exactly on demand.
"""

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import ndimage
from skimage.morphology import disk

from Reconstruction.data_prep.schema import LAYERS, ROOM_LAYER, STRUCTURE_LAYER, Schema

EASY: str = "easy"
HARD: str = "hard"
# Component type of a case where nothing is removed (empty target).
NONE_TYPE: str = "none"
# Types whose hard mode fills the removed pixels with wall.
_OPENING_TYPES: tuple[str, str] = ("door", "window")


@dataclass(frozen=True)
class CaseSpec:
    """Deterministic description of one missing-component case.

    Attributes:
        split (str): Split name.
        plan_index (int): Index of the plan inside the split store.
        plan_id (int): ResPlan plan id (for traceability).
        comp_type (str): Component type, or ``NONE_TYPE``.
        layer (str): Layer of the primary instance ("" for NONE_TYPE).
        instance_id (int): Layer-local id of the primary instance (0 for NONE_TYPE).
        mode (str): ``EASY`` or ``HARD``.
        margin_px (int): Extra erased margin in pixels.
    """

    split: str
    plan_index: int
    plan_id: int
    comp_type: str
    layer: str
    instance_id: int
    mode: str
    margin_px: int

    @property
    def case_id(self) -> str:
        """Unique, readable case identifier."""
        return f"{self.split}-{self.plan_id}-{self.comp_type}-{self.layer[:1]}{self.instance_id}-{self.mode}-m{self.margin_px}"

    def to_dict(self) -> dict[str, Any]:
        """Serializable form (includes ``case_id``)."""
        return {"case_id": self.case_id, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseSpec":
        """Rebuild from ``to_dict`` output (extra keys are ignored).

        Args:
            data (dict[str, Any]): Serialized spec.

        Returns:
            CaseSpec: The spec.
        """
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})


@dataclass
class Case:
    """One generated case; see module docstring for array meanings."""

    full: np.ndarray
    incomplete: np.ndarray
    target: np.ndarray
    mask: np.ndarray
    primary: np.ndarray
    info: dict[str, Any]


def compute_size_bins(instance_areas: list[tuple[str, int]], schema: Schema, quantiles: list[float]) -> dict[str, list[float]]:
    """Per-component-type area thresholds for small/medium/large bins.

    Must be computed on the TRAIN split only, then reused for val/test.

    Args:
        instance_areas (list[tuple[str, int]]): (class, pixel area) pairs.
        schema (Schema): Taxonomy (class -> component type).
        quantiles (list[float]): Quantiles separating the bins, ascending.

    Returns:
        dict[str, list[float]]: Component type -> area thresholds.
    """
    by_type: dict[str, list[int]] = {}
    for cls, area in instance_areas:
        by_type.setdefault(schema.component_type[cls], []).append(area)
    return {ctype: np.quantile(areas, quantiles).tolist() for ctype, areas in sorted(by_type.items())}


class MissingComponentGenerator:
    """Samples case specs and builds cases from rasterized plans."""

    def __init__(self, schema: Schema, params: SimpleNamespace, size_bins: dict[str, list[float]] | None) -> None:
        """Store taxonomy and generation parameters.

        Args:
            schema (Schema): Class taxonomy.
            params (SimpleNamespace): Loaded ``params.toml``.
            size_bins (dict[str, list[float]] | None): Output of
                ``compute_size_bins`` (None: size bins reported as null).

        Raises:
            ValueError: If margin probabilities do not match margins.
        """
        self.schema: Schema = schema
        self.cfg: SimpleNamespace = params.missing
        self.size_bins: dict[str, list[float]] | None = size_bins
        self.canvas_size: int = params.canvas.SIZE
        if len(self.cfg.train.MARGIN_PROBS) != len(self.cfg.MARGINS_PX):
            raise ValueError("missing.train.MARGIN_PROBS must have one entry per missing.MARGINS_PX")

    # ------------------------------------------------------------------
    # Spec sampling
    # ------------------------------------------------------------------

    def _eligible(self, record: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """Removable instances of a plan grouped by component type.

        Args:
            record (dict[str, Any]): Plan record from the store.

        Returns:
            dict[str, list[dict[str, Any]]]: Type -> instances (types with none omitted).
        """
        grouped: dict[str, list[dict[str, Any]]] = {}
        for inst in record["instances"]:
            ctype: str = self.schema.component_type[inst["cls"]]
            if ctype in self.cfg.COMPONENT_TYPES:
                grouped.setdefault(ctype, []).append(inst)
        return grouped

    def _spec(self, split: str, plan_index: int, record: dict[str, Any], ctype: str,
              inst: dict[str, Any] | None, mode: str, margin: int) -> CaseSpec:
        """Assemble a ``CaseSpec`` (``inst`` None means an empty case).

        Args:
            split (str): Split name.
            plan_index (int): Plan index in the store.
            record (dict[str, Any]): Plan record.
            ctype (str): Component type or NONE_TYPE.
            inst (dict[str, Any] | None): Primary instance.
            mode (str): Removal mode.
            margin (int): Margin in px.

        Returns:
            CaseSpec: The spec.
        """
        layer: str = inst["layer"] if inst else ""
        instance_id: int = int(inst["id"]) if inst else 0
        return CaseSpec(split, plan_index, int(record["plan_id"]), ctype, layer, instance_id, mode, int(margin))

    def sample_train_spec(self, split: str, plan_index: int, record: dict[str, Any], rng: np.random.Generator) -> CaseSpec:
        """Draw a random training case for one plan.

        The component type is drawn uniformly among the types present in the
        plan (balancing rare types), then an instance uniformly within it.

        Args:
            split (str): Split name.
            plan_index (int): Plan index in the store.
            record (dict[str, Any]): Plan record.
            rng (np.random.Generator): Random generator.

        Returns:
            CaseSpec: Sampled spec.
        """
        eligible: dict[str, list[dict[str, Any]]] = self._eligible(record)
        if not eligible or rng.random() < self.cfg.train.EMPTY_CASE_PROB:
            return self._spec(split, plan_index, record, NONE_TYPE, None, EASY, 0)
        ctype: str = sorted(eligible)[int(rng.integers(len(eligible)))]
        inst: dict[str, Any] = eligible[ctype][int(rng.integers(len(eligible[ctype])))]
        hard: bool = ctype in self.cfg.HARD_MODE_TYPES and rng.random() < self.cfg.train.HARD_MODE_PROB
        margin: int = int(rng.choice(self.cfg.MARGINS_PX, p=self.cfg.train.MARGIN_PROBS))
        return self._spec(split, plan_index, record, ctype, inst, HARD if hard else EASY, margin)

    def enumerate_eval_specs(self, split: str, plan_index: int, record: dict[str, Any], seed: int) -> list[CaseSpec]:
        """Fixed evaluation cases for one plan.

        For every component type present: one instance, one easy case and (if
        supported) one hard case at margin 0. A seeded fraction of plans also
        gets easy cases at every positive margin. The RNG is keyed on
        (seed, plan_id), so results do not depend on processing order.

        Args:
            split (str): Split name.
            plan_index (int): Plan index in the store.
            record (dict[str, Any]): Plan record.
            seed (int): Master seed.

        Returns:
            list[CaseSpec]: Specs for this plan.
        """
        rng: np.random.Generator = np.random.default_rng([seed, int(record["plan_id"])])
        sweep: bool = bool(rng.random() < self.cfg.eval.MARGIN_SWEEP_FRACTION)
        eligible: dict[str, list[dict[str, Any]]] = self._eligible(record)
        specs: list[CaseSpec] = []
        for ctype in self.cfg.COMPONENT_TYPES:
            if ctype not in eligible:
                continue
            inst: dict[str, Any] = eligible[ctype][int(rng.integers(len(eligible[ctype])))]
            specs.append(self._spec(split, plan_index, record, ctype, inst, EASY, 0))
            if ctype in self.cfg.HARD_MODE_TYPES:
                specs.append(self._spec(split, plan_index, record, ctype, inst, HARD, 0))
            if sweep:
                specs.extend(self._spec(split, plan_index, record, ctype, inst, EASY, m) for m in self.cfg.MARGINS_PX if m > 0)
        return specs

    # ------------------------------------------------------------------
    # Case construction
    # ------------------------------------------------------------------

    def _semantic(self, layers: dict[str, np.ndarray], record: dict[str, Any]) -> np.ndarray:
        """Multi-hot semantic map from the two instance layers.

        Args:
            layers (dict[str, np.ndarray]): uint8 instance layers.
            record (dict[str, Any]): Plan record (instance classes).

        Returns:
            np.ndarray: uint8 (C, H, W); a pixel can be set in one structure
            and one room channel at the same time.
        """
        luts: dict[str, np.ndarray] = {layer: np.full(256, -1, np.int16) for layer in LAYERS}
        for inst in record["instances"]:
            luts[inst["layer"]][inst["id"]] = self.schema.channel_index(inst["cls"])
        semantic: np.ndarray = np.zeros((self.schema.num_channels,) + layers[ROOM_LAYER].shape, np.uint8)
        for layer in LAYERS:
            channel: np.ndarray = luts[layer][layers[layer]]
            rows, cols = np.nonzero(channel >= 0)
            semantic[channel[rows, cols], rows, cols] = 1
        return semantic

    def _neighbours(self, layers: dict[str, np.ndarray], layer: str, instance_id: int, radius: int) -> set[tuple[str, int]]:
        """Instances within ``radius`` px of an instance (both layers, excluding itself).

        Args:
            layers (dict[str, np.ndarray]): Instance layers.
            layer (str): Layer of the query instance.
            instance_id (int): Id of the query instance.
            radius (int): Adjacency radius in px.

        Returns:
            set[tuple[str, int]]: (layer, id) of neighbouring instances.
        """
        rows, cols = np.nonzero(layers[layer] == instance_id)
        if rows.size == 0:
            return set()
        # Work on the bounding box grown by the radius, for speed.
        size: int = layers[layer].shape[0]
        crop = (slice(max(rows.min() - radius, 0), min(rows.max() + radius + 1, size)),
                slice(max(cols.min() - radius, 0), min(cols.max() + radius + 1, size)))
        grown: np.ndarray = ndimage.binary_dilation(layers[layer][crop] == instance_id, structure=disk(radius))
        found: set[tuple[str, int]] = set()
        for other in LAYERS:
            found.update((other, int(i)) for i in np.unique(layers[other][crop][grown]) if i > 0)
        found.discard((layer, instance_id))
        return found

    def _removal_set(self, spec: CaseSpec, layers: dict[str, np.ndarray], record: dict[str, Any]) -> list[tuple[str, int]]:
        """Instances erased for a spec, primary first (see module docstring).

        Args:
            spec (CaseSpec): Case spec.
            layers (dict[str, np.ndarray]): Instance layers.
            record (dict[str, Any]): Plan record.

        Returns:
            list[tuple[str, int]]: (layer, id) pairs.
        """
        if spec.comp_type == NONE_TYPE:
            return []
        primary: tuple[str, int] = (spec.layer, spec.instance_id)
        if spec.mode == EASY or spec.comp_type not in ("wall", "room"):
            return [primary]
        type_of: dict[tuple[str, int], str] = {(i["layer"], i["id"]): self.schema.component_type[i["cls"]] for i in record["instances"]}
        radius: int = self.cfg.NEIGHBOR_RADIUS_PX
        touching: set[tuple[str, int]] = self._neighbours(layers, spec.layer, spec.instance_id, radius)
        if spec.comp_type == "wall":
            extra = [n for n in touching if type_of.get(n) in _OPENING_TYPES]
        else:
            # Structure elements of this room that border no other room.
            extra = [n for n in touching if n[0] == STRUCTURE_LAYER
                     and {m for m in self._neighbours(layers, *n, radius) if m[0] == ROOM_LAYER} <= {primary}]
        return [primary] + sorted(extra)

    def build_case(self, layers: dict[str, np.ndarray], record: dict[str, Any], spec: CaseSpec) -> Case:
        """Generate the arrays and metadata of one case.

        Args:
            layers (dict[str, np.ndarray]): Instance layers of the plan (not modified).
            record (dict[str, Any]): Plan record.
            spec (CaseSpec): What to remove.

        Returns:
            Case: The generated case.
        """
        full: np.ndarray = self._semantic(layers, record)
        removed: list[tuple[str, int]] = self._removal_set(spec, layers, record)
        cut: dict[str, np.ndarray] = {layer: layers[layer].copy() for layer in LAYERS}
        erased: np.ndarray = np.zeros(full.shape[1:], bool)
        for layer, instance_id in removed:
            erased |= layers[layer] == instance_id
        if spec.margin_px > 0:
            erased = ndimage.binary_dilation(erased, structure=disk(spec.margin_px))
        for layer in LAYERS:
            # Without a margin only the removed ids are cleared: a wall pixel
            # that also carries a room label keeps the room label.
            removed_ids: list[int] = [instance_id for removed_layer, instance_id in removed if removed_layer == layer]
            cut[layer][erased & np.isin(cut[layer], removed_ids)] = 0
            if spec.margin_px > 0:
                cut[layer][erased] = 0
        incomplete: np.ndarray = self._semantic(cut, record)
        if spec.mode == HARD and spec.comp_type in _OPENING_TYPES:
            # Close the opening with wall so the plan looks complete.
            incomplete[self.schema.channel_index("wall")][layers[spec.layer] == spec.instance_id] = 1
        target: np.ndarray = full & (1 - incomplete)
        info: dict[str, Any] = self._describe(spec, layers, record, full, removed, target)
        primary: np.ndarray = np.zeros(erased.shape, np.uint8)
        if spec.comp_type != NONE_TYPE:
            primary[layers[spec.layer] == spec.instance_id] = 1
        return Case(full, incomplete, target, erased.astype(np.uint8), primary, info)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _size_bin(self, comp_type: str, area: int) -> str | None:
        """Size bin name of a component, from train-split thresholds.

        Args:
            comp_type (str): Component type.
            area (int): Pixel area.

        Returns:
            str | None: Bin name, or None if bins are unavailable.
        """
        if self.size_bins is None or comp_type not in self.size_bins:
            return None
        return self.cfg.SIZE_BIN_NAMES[int(np.searchsorted(self.size_bins[comp_type], area, side="right"))]

    def _exterior(self, full: np.ndarray) -> np.ndarray:
        """Empty pixels connected to the canvas border (outside the building).

        Args:
            full (np.ndarray): Semantic map of the complete plan.

        Returns:
            np.ndarray: bool (H, W).
        """
        labels, _ = ndimage.label(~full.any(axis=0))
        border: np.ndarray = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        return np.isin(labels, border[border > 0])

    def _describe(self, spec: CaseSpec, layers: dict[str, np.ndarray], record: dict[str, Any],
                  full: np.ndarray, removed: list[tuple[str, int]], target: np.ndarray) -> dict[str, Any]:
        """Metadata for stratified evaluation of one case.

        Args:
            spec (CaseSpec): Case spec.
            layers (dict[str, np.ndarray]): Instance layers.
            record (dict[str, Any]): Plan record.
            full (np.ndarray): Semantic map of the complete plan.
            removed (list[tuple[str, int]]): Erased instances.
            target (np.ndarray): Target map.

        Returns:
            dict[str, Any]: JSON-serializable metadata.
        """
        classes: dict[tuple[str, int], str] = {(i["layer"], i["id"]): i["cls"] for i in record["instances"]}
        info: dict[str, Any] = {**spec.to_dict(), "metres_per_px": record["metres_per_px"],
                                "removed": [{"layer": layer, "id": iid, "cls": classes[(layer, iid)]} for layer, iid in removed],
                                "target_area_px": int(target.any(axis=0).sum())}
        if spec.comp_type == NONE_TYPE:
            return {**info, "primary_cls": None, "primary_area_px": 0, "size_bin": None,
                    "location": None, "num_neighbours": 0, "centre_offset": None}
        primary: np.ndarray = layers[spec.layer] == spec.instance_id
        near_exterior: np.ndarray = ndimage.binary_dilation(primary, structure=disk(self.cfg.EXTERIOR_RADIUS_PX))
        occupied_rows, occupied_cols = np.nonzero(full.any(axis=0))
        rows, cols = np.nonzero(primary)
        # Offset of the component from the plan centre, as a fraction of the canvas.
        offset: float = float(np.hypot(rows.mean() - occupied_rows.mean(), cols.mean() - occupied_cols.mean()) / self.canvas_size)
        area: int = int(primary.sum())
        return {**info, "primary_cls": classes[(spec.layer, spec.instance_id)], "primary_area_px": area,
                "size_bin": self._size_bin(spec.comp_type, area),
                "location": "exterior" if (near_exterior & self._exterior(full)).any() else "interior",
                "num_neighbours": len(self._neighbours(layers, spec.layer, spec.instance_id, self.cfg.NEIGHBOR_RADIUS_PX)),
                "centre_offset": round(offset, 4)}
