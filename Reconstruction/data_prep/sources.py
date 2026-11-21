"""Dataset adapters for ``build_dataset.py``.

Each adapter tells the build script, for one dataset:

* which input files it reads (hashed for provenance),
* the item keys of each official split,
* how to load raw payloads for a list of keys (main process, read-only),
* how a worker turns one payload into ``PlanGeometry`` objects (one per
  floor/plan) plus notes about skipped or suspicious plans.

Adding a dataset = writing one ``make_<name>_source`` function and one
module-level geometry function, and registering it in ``SOURCES``.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from Reconstruction.common.config_loader import resolve_path
from Reconstruction.data_prep.cubicasa_source import (load_cubicasa_splits, parse_house, read_svg_bytes,
                                                      validate_class_maps)
from Reconstruction.data_prep.geometry import PlanGeometry
from Reconstruction.data_prep.resplan_source import extract_plan_geometry, load_resplan_plans, load_split_ids
from Reconstruction.data_prep.schema import Schema

# Worker-side conversion: payload + params -> (plans, notes).
GeometryFn = Callable[[Any, SimpleNamespace], tuple[list[PlanGeometry], list[dict[str, Any]]]]


@dataclass
class DatasetSource:
    """Everything the build script needs to know about one dataset.

    Attributes:
        name (str): Dataset name (also the output sub-directory).
        input_files (list[Path]): Original files read (for provenance hashes).
        split_keys (dict[str, list[Any]]): Split -> item keys.
        load_payloads (Callable[[list[Any]], list[Any]]): Keys -> raw payloads.
        to_geometries (GeometryFn): Picklable module-level conversion function.
    """

    name: str
    input_files: list[Path]
    split_keys: dict[str, list[Any]]
    load_payloads: Callable[[list[Any]], list[Any]]
    to_geometries: GeometryFn


def resplan_geometries(plan: dict[str, Any], params: SimpleNamespace) -> tuple[list[PlanGeometry], list[dict[str, Any]]]:
    """Worker conversion for ResPlan (one plan per payload, no notes).

    Args:
        plan (dict[str, Any]): Raw ResPlan plan.
        params (SimpleNamespace): Loaded ``params.toml``.

    Returns:
        tuple[list[PlanGeometry], list[dict[str, Any]]]: One plan, no notes.
    """
    return [extract_plan_geometry(plan, dict(vars(params.taxonomy.resplan_class_map)))], []


def cubicasa_geometries(payload: tuple[int, bytes], params: SimpleNamespace) -> tuple[list[PlanGeometry], list[dict[str, Any]]]:
    """Worker conversion for CubiCasa5K (one plan per usable floor).

    Args:
        payload (tuple[int, bytes]): (house id, model.svg content).
        params (SimpleNamespace): Loaded ``params.toml``.

    Returns:
        tuple[list[PlanGeometry], list[dict[str, Any]]]: Floors and notes.
    """
    house_id, svg = payload
    return parse_house(svg, house_id, params)


def make_resplan_source(config: SimpleNamespace, params: SimpleNamespace, schema: Schema) -> DatasetSource:
    """ResPlan adapter (loads the pickle once, read-only).

    Args:
        config (SimpleNamespace): Run config.
        params (SimpleNamespace): Parameters (unused; uniform signature).
        schema (Schema): Taxonomy (unused; uniform signature).

    Returns:
        DatasetSource: Adapter; keys are plan ids.
    """
    pickle_path: Path = resolve_path(config.RESPLAN_PICKLE_PATH)
    split_path: Path = resolve_path(config.RESPLAN_SPLIT_PATH)
    by_id: dict[int, dict[str, Any]] = {int(p["id"]): p for p in load_resplan_plans(pickle_path)}
    return DatasetSource("resplan", [pickle_path, split_path],
                         load_split_ids(split_path, config.INCLUDE_AUGMENTED_IN_TRAIN),
                         lambda keys: [by_id[key] for key in keys], resplan_geometries)


def make_cubicasa_source(config: SimpleNamespace, params: SimpleNamespace, schema: Schema) -> DatasetSource:
    """CubiCasa5K adapter (reads SVGs straight from the zip, read-only).

    Args:
        config (SimpleNamespace): Run config.
        params (SimpleNamespace): Parameters (class maps validated here).
        schema (Schema): Taxonomy.

    Returns:
        DatasetSource: Adapter; keys are (house id, svg member path).
    """
    validate_class_maps(params, schema.channels)
    zip_path: Path = resolve_path(config.CUBICASA_ZIP_PATH)

    def load(keys: list[tuple[int, str]]) -> list[tuple[int, bytes]]:
        return list(zip([hid for hid, _ in keys], read_svg_bytes(zip_path, [member for _, member in keys])))

    return DatasetSource("cubicasa", [zip_path], load_cubicasa_splits(zip_path), load, cubicasa_geometries)


SOURCES: dict[str, Callable[[SimpleNamespace, SimpleNamespace, Schema], DatasetSource]] = {
    "resplan": make_resplan_source,
    "cubicasa": make_cubicasa_source,
}
