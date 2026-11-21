"""Build a processed dataset (ResPlan or CubiCasa5K) for missing-component reconstruction.

Steps:
    1. Read the original release and its official splits (read-only).
    2. Rasterize every plan / floor of train/val/test into instance layers, in parallel.
    3. Compute component size bins on TRAIN only.
    4. Flag val/test plans that are near-duplicates of train plans.
    5. Freeze the val/test missing-component cases (specs + metadata).
    6. Write a summary of everything (also logged).

Usage (from the repository root):
    python -m Reconstruction.data_prep.build_dataset --dataset resplan --mode debug
    python -m Reconstruction.data_prep.build_dataset --dataset cubicasa --mode prod [--overwrite]

Nothing is written next to the original data; outputs go to PROCESSED_ROOT/<dataset>.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from shapely.errors import GEOSException

from Reconstruction.common.config_loader import load_params, load_run_config, resolve_path
from Reconstruction.common.logging_setup import make_run_id, setup_logger
from Reconstruction.data_prep.case_source import (CASES_FILE_TEMPLATE, PARAMS_COPY_FILE, SIZE_BINS_FILE,
                                                  TRAIN_SPLIT, load_generator)
from Reconstruction.data_prep.dedup import find_near_duplicates
from Reconstruction.data_prep.geometry import SPLIT_NAMES
from Reconstruction.data_prep.missing import CaseSpec, MissingComponentGenerator, compute_size_bins
from Reconstruction.data_prep.plan_store import PlanStore, write_split
from Reconstruction.data_prep.rasterize import RasterPlan, rasterize_plan
from Reconstruction.data_prep.schema import Schema
from Reconstruction.data_prep.sources import SOURCES, DatasetSource, GeometryFn

SCRIPT_DIR: Path = Path(__file__).resolve().parent
CONFIG_FILE: Path = SCRIPT_DIR / "config.toml"
PARAMS_FILE: Path = SCRIPT_DIR / "params.toml"
# Marker proving a directory was created by this script (safe to overwrite).
RUN_INFO_FILE: str = "run_info.json"
DUPLICATES_FILE: str = "near_duplicates.json"
PROBLEMS_FILE: str = "problems.json"
SUMMARY_FILE: str = "summary.json"
EVAL_SPLITS: tuple[str, str] = ("val", "test")
# Items per task handed to a worker process.
POOL_CHUNK_SIZE: int = 16


def parse_args() -> argparse.Namespace:
    """Command-line arguments.

    Returns:
        argparse.Namespace: ``dataset``, ``mode`` and ``overwrite``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(SOURCES), required=True, help="Dataset to build")
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug", help="Config table to use")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output made by this script")
    return parser.parse_args()


def prepare_output_dir(out_dir: Path, overwrite: bool) -> None:
    """Create the output directory, refusing to clobber foreign data.

    Args:
        out_dir (Path): Output directory.
        overwrite (bool): Replace a previous output of this script.

    Raises:
        FileExistsError: If the directory exists and overwriting is not
            allowed, or it was not created by this script.
    """
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass --overwrite to rebuild it")
        if not (out_dir / RUN_INFO_FILE).is_file():
            raise FileExistsError(f"{out_dir} has no {RUN_INFO_FILE}; refusing to delete a directory this script did not create")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)


def _file_md5(path: Path) -> str:
    """MD5 of a file, recorded so a build can be traced to its exact input.

    Args:
        path (Path): File to hash.

    Returns:
        str: Hex digest.
    """
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run_info(out_dir: Path, source: DatasetSource, config: SimpleNamespace, run_id: str) -> None:
    """Save provenance: run id, mode, git commit, input hashes, params copy.

    Args:
        out_dir (Path): Output directory.
        source (DatasetSource): Dataset adapter.
        config (SimpleNamespace): Run config.
        run_id (str): Run identifier.
    """
    commit: subprocess.CompletedProcess = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR,
                                                         capture_output=True, text=True, check=False)
    info: dict[str, Any] = {
        "run_id": run_id, "dataset": source.name, "mode": config.MODE, "git_commit": commit.stdout.strip() or None,
        "inputs": {str(path): _file_md5(path) for path in source.input_files},
        "config": dict(vars(config)),
    }
    (out_dir / RUN_INFO_FILE).write_text(json.dumps(info, indent=2))
    shutil.copyfile(PARAMS_FILE, out_dir / PARAMS_COPY_FILE)


def select_keys(split_keys: dict[str, list[Any]], max_per_split: int, seed: int) -> dict[str, list[Any]]:
    """Optionally subsample each split (debug runs), keeping key order.

    Args:
        split_keys (dict[str, list[Any]]): Split -> item keys.
        max_per_split (int): Maximum items per split (-1 = all).
        seed (int): Sampling seed.

    Returns:
        dict[str, list[Any]]: Split -> selected keys.
    """
    if max_per_split < 0:
        return split_keys
    rng: np.random.Generator = np.random.default_rng(seed)
    selected: dict[str, list[Any]] = {}
    for name, keys in split_keys.items():
        indices = sorted(rng.choice(len(keys), size=min(max_per_split, len(keys)), replace=False).tolist())
        selected[name] = [keys[i] for i in indices]
    return selected


def _rasterize_worker(payload: Any, to_geometries: GeometryFn, schema: Schema,
                      params: SimpleNamespace) -> tuple[list[RasterPlan], list[dict[str, Any]]]:
    """Convert and rasterize one payload; problems are returned, not raised.

    Args:
        payload (Any): Raw item (ResPlan dict or CubiCasa (house id, svg)).
        to_geometries (GeometryFn): Dataset conversion function.
        schema (Schema): Taxonomy.
        params (SimpleNamespace): Parameters.

    Returns:
        tuple[list[RasterPlan], list[dict[str, Any]]]: Rasters and problem
        records (``plan_id`` plus ``error`` or ``note``).
    """
    try:
        geometries, problems = to_geometries(payload, params)
    except (ValueError, GEOSException, ET.ParseError) as error:
        return [], [{"plan_id": None, "error": f"{type(error).__name__}: {error}"}]
    rasters: list[RasterPlan] = []
    for geometry in geometries:
        try:
            rasters.append(rasterize_plan(geometry, schema, params))
        except (ValueError, GEOSException) as error:
            problems.append({"plan_id": geometry.plan_id, "error": f"{type(error).__name__}: {error}"})
    return rasters, problems


def rasterize_splits(source: DatasetSource, split_keys: dict[str, list[Any]], schema: Schema, params: SimpleNamespace,
                     out_dir: Path, workers: int, log: logging.Logger) -> dict[str, list[dict[str, Any]]]:
    """Rasterize and store every split.

    Args:
        source (DatasetSource): Dataset adapter.
        split_keys (dict[str, list[Any]]): Split -> item keys.
        schema (Schema): Taxonomy.
        params (SimpleNamespace): Parameters.
        out_dir (Path): Output directory.
        workers (int): Worker processes.
        log (logging.Logger): Logger.

    Returns:
        dict[str, list[dict[str, Any]]]: Split -> problem records.
    """
    worker = partial(_rasterize_worker, to_geometries=source.to_geometries, schema=schema, params=params)
    problems: dict[str, list[dict[str, Any]]] = {}
    with Pool(workers) as pool:
        for split in SPLIT_NAMES:
            results = pool.map(worker, source.load_payloads(split_keys[split]), chunksize=POOL_CHUNK_SIZE)
            rasters: list[RasterPlan] = [raster for plans, _ in results for raster in plans]
            problems[split] = [problem for _, found in results for problem in found]
            write_split(out_dir / split, rasters)
            errors: int = sum("error" in p for p in problems[split])
            log.info(f"{split}: {len(split_keys[split])} items -> {len(rasters)} plans; "
                     f"{errors} errors, {len(problems[split]) - errors} notes")
            for problem in problems[split]:
                log.debug(f"{split}: plan {problem['plan_id']}: {problem.get('error') or problem.get('note')}")
    (out_dir / PROBLEMS_FILE).write_text(json.dumps(problems, indent=2))
    return problems


def flag_duplicates(out_dir: Path, schema: Schema, params: SimpleNamespace, log: logging.Logger) -> dict[str, list[dict[str, Any]]]:
    """Find val/test plans that nearly duplicate a train plan.

    Args:
        out_dir (Path): Output directory with split stores.
        schema (Schema): Taxonomy.
        params (SimpleNamespace): Parameters.
        log (logging.Logger): Logger.

    Returns:
        dict[str, list[dict[str, Any]]]: Eval split -> duplicates (with plan ids).
    """
    if not params.dedup.ENABLED:
        log.info("Near-duplicate filtering disabled")
        return {split: [] for split in EVAL_SPLITS}
    class_index: dict[str, int] = {name: schema.channel_index(name) for name in schema.channels}
    train = PlanStore(out_dir / TRAIN_SPLIT)
    train_thumbs: np.ndarray = train.label_thumbnails(params.dedup.THUMB_SIZE, class_index)
    result: dict[str, list[dict[str, Any]]] = {}
    for split in EVAL_SPLITS:
        store = PlanStore(out_dir / split)
        found = find_near_duplicates(train_thumbs, store.label_thumbnails(params.dedup.THUMB_SIZE, class_index), params.dedup.THRESHOLD)
        result[split] = [{**d, "eval_plan_id": store.record(d["eval_index"])["plan_id"],
                          "train_plan_id": train.record(d["train_index"])["plan_id"]} for d in found]
        log.info(f"{split}: {len(found)} of {len(store)} plans are near-duplicates of train plans (excluded from cases)")
    (out_dir / DUPLICATES_FILE).write_text(json.dumps(result, indent=2))
    return result


def _describe_worker(spec_dicts: list[dict[str, Any]], out_dir: Path, split: str) -> list[dict[str, Any]]:
    """Build cases for a batch of specs and return their metadata.

    Args:
        spec_dicts (list[dict[str, Any]]): Serialized specs of one split.
        out_dir (Path): Output directory.
        split (str): Split name.

    Returns:
        list[dict[str, Any]]: Case metadata (spec fields included).
    """
    _, generator = load_generator(out_dir)
    store = PlanStore(out_dir / split)
    infos: list[dict[str, Any]] = []
    for data in spec_dicts:
        spec = CaseSpec.from_dict(data)
        infos.append(generator.build_case(store.layers(spec.plan_index), store.record(spec.plan_index), spec).info)
    return infos


def freeze_eval_cases(out_dir: Path, generator: MissingComponentGenerator, seed: int, duplicates: dict[str, list[dict[str, Any]]],
                      workers: int, log: logging.Logger) -> dict[str, list[dict[str, Any]]]:
    """Enumerate and store the fixed val/test cases with their metadata.

    Args:
        out_dir (Path): Output directory.
        generator (MissingComponentGenerator): Case generator.
        seed (int): Master seed.
        duplicates (dict[str, list[dict[str, Any]]]): Near-duplicates to exclude.
        workers (int): Worker processes.
        log (logging.Logger): Logger.

    Returns:
        dict[str, list[dict[str, Any]]]: Split -> case metadata, in file order.
    """
    infos_by_split: dict[str, list[dict[str, Any]]] = {}
    with Pool(workers) as pool:
        for split in EVAL_SPLITS:
            store = PlanStore(out_dir / split)
            excluded: set[int] = {d["eval_index"] for d in duplicates[split]}
            specs: list[dict[str, Any]] = [spec.to_dict() for index in range(len(store)) if index not in excluded
                                           for spec in generator.enumerate_eval_specs(split, index, store.record(index), seed)]
            batches = [specs[i:i + POOL_CHUNK_SIZE] for i in range(0, len(specs), POOL_CHUNK_SIZE)]
            infos: list[dict[str, Any]] = [info for batch in pool.map(partial(_describe_worker, out_dir=out_dir, split=split), batches)
                                           for info in batch]
            with open(out_dir / CASES_FILE_TEMPLATE.format(split=split), "w", encoding="utf-8") as handle:
                handle.writelines(json.dumps(info) + "\n" for info in infos)
            infos_by_split[split] = infos
            log.info(f"{split}: froze {len(infos)} cases from {len(store) - len(excluded)} plans")
    return infos_by_split


def summarize(out_dir: Path, split_keys: dict[str, list[Any]], problems: dict[str, list[dict[str, Any]]],
              duplicates: dict[str, list[dict[str, Any]]], eval_infos: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Collect dataset statistics for the report and the log.

    Args:
        out_dir (Path): Output directory.
        split_keys (dict[str, list[Any]]): Selected item keys per split.
        problems (dict[str, list[dict[str, Any]]]): Problem records per split.
        duplicates (dict[str, list[dict[str, Any]]]): Near-duplicates per eval split.
        eval_infos (dict[str, list[dict[str, Any]]]): Frozen case metadata.

    Returns:
        dict[str, Any]: JSON-serializable summary.
    """
    summary: dict[str, Any] = {"splits": {}, "eval_cases": {}}
    for split in SPLIT_NAMES:
        store = PlanStore(out_dir / split)
        scales: list[float] = [store.record(i)["metres_per_px"] for i in range(len(store))]
        known: list[float] = [s for s in scales if s is not None]
        summary["splits"][split] = {
            "source_items": len(split_keys[split]), "plans": len(store),
            "errors": sum("error" in p for p in problems[split]),
            "notes": dict(Counter(p["note"].split(":")[0] for p in problems[split] if "note" in p)),
            "near_duplicates_excluded": len(duplicates.get(split, [])),
            "instances_per_class": dict(sorted(Counter(cls for cls, _ in store.instance_areas()).items())),
            "metres_per_px_median": float(np.median(known)) if known else None,
            "metres_per_px_unknown": len(scales) - len(known),
        }
    for split, infos in eval_infos.items():
        summary["eval_cases"][split] = {
            "total": len(infos),
            "type_mode": dict(sorted(Counter(f"{i['comp_type']}/{i['mode']}" for i in infos).items())),
            "margin_px": dict(sorted(Counter(str(i["margin_px"]) for i in infos).items())),
            "size_bin": dict(sorted(Counter(str(i["size_bin"]) for i in infos).items())),
            "location": dict(sorted(Counter(str(i["location"]) for i in infos).items())),
        }
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    """Run the full build (see module docstring)."""
    args: argparse.Namespace = parse_args()
    config: SimpleNamespace = load_run_config(CONFIG_FILE, args.mode)
    params: SimpleNamespace = load_params(PARAMS_FILE)
    run_id: str = make_run_id(f"dataprep-{args.dataset}")
    log: logging.Logger = setup_logger("build_dataset", resolve_path(config.LOG_FILE_PATH), run_id, config.IS_DEBUG)
    schema: Schema = Schema.from_params(params)
    workers: int = config.NUM_WORKERS if config.NUM_WORKERS > 0 else (os.cpu_count() or 1)
    out_dir: Path = resolve_path(config.PROCESSED_ROOT) / args.dataset
    log.info(f"Run {run_id} ({args.dataset}, mode={args.mode}) -> {out_dir} with {workers} workers")

    prepare_output_dir(out_dir, args.overwrite)
    source: DatasetSource = SOURCES[args.dataset](config, params, schema)
    write_run_info(out_dir, source, config, run_id)
    split_keys = select_keys(source.split_keys, config.MAX_ITEMS_PER_SPLIT, config.SEED)
    log.info("Source items per split: " + ", ".join(f"{k}={len(v)}" for k, v in split_keys.items()))

    problems = rasterize_splits(source, split_keys, schema, params, out_dir, workers, log)
    size_bins = compute_size_bins(PlanStore(out_dir / TRAIN_SPLIT).instance_areas(), schema, params.missing.SIZE_BIN_QUANTILES)
    (out_dir / SIZE_BINS_FILE).write_text(json.dumps(size_bins, indent=2))
    log.info(f"Size-bin area thresholds (px, train only): {size_bins}")

    duplicates = flag_duplicates(out_dir, schema, params, log)
    _, generator = load_generator(out_dir)
    eval_infos = freeze_eval_cases(out_dir, generator, config.SEED, duplicates, workers, log)
    summary = summarize(out_dir, split_keys, problems, duplicates, eval_infos)
    log.info("Summary:\n" + json.dumps(summary, indent=2))
    log.info(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
