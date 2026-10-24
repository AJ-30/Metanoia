"""Evaluate a predictor on the frozen val/test cases of a processed dataset.

Predictors are the built-in baselines (``empty``, ``oracle``, ``rules``) or
a directory of saved model predictions (``--predictions DIR`` with one
``<case_id>.npz`` holding ``samples`` of shape (K, C, H, W)).

Outputs in OUTPUT_DIR/<dataset>/<split>/<name>/:
    * ``cases.csv``     one row per case: case metadata + all metrics
    * ``summary.json``  every report table with every metric
    * ``summary.md``    headline columns of every table
    * ``examples_<mode>.png``  qualitative grid per removal mode (easy, hard):
                               input / prediction / composed / truth

Usage (from the repository root):
    python -m Reconstruction.evaluation.evaluate --dataset resplan --predictor rules --mode prod
    python -m Reconstruction.evaluation.evaluate --dataset cubicasa --predictions path/to/npz --name diffusion_v1
"""

import argparse
import json
import logging
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from Reconstruction.common.config_loader import load_params, load_run_config, resolve_path
from Reconstruction.common.logging_setup import make_run_id, setup_logger
from Reconstruction.data_prep.case_source import CaseSource
from Reconstruction.data_prep.missing import EASY, HARD
from Reconstruction.data_prep.schema import Schema
from Reconstruction.evaluation.baselines import BUILTIN_PREDICTORS, FilePredictor, PredictorFn
from Reconstruction.evaluation.case_metrics import RECORD_FIELDS, prepare_case, score_samples
from Reconstruction.evaluation.examples import save_examples
from Reconstruction.evaluation.report import build_tables, tables_to_json, to_markdown

SCRIPT_DIR: Path = Path(__file__).resolve().parent
CONFIG_FILE: Path = SCRIPT_DIR / "config.toml"
PARAMS_FILE: Path = SCRIPT_DIR / "params.toml"


def parse_args() -> argparse.Namespace:
    """Command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Processed dataset name (resplan, cubicasa)")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictor", choices=sorted(BUILTIN_PREDICTORS), help="Built-in baseline")
    source.add_argument("--predictions", type=Path, help="Directory of <case_id>.npz model predictions")
    parser.add_argument("--name", default=None, help="Run name for the output folder (default: predictor name)")
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    return parser.parse_args()


def make_predictor(name: str | None, predictions: Path | None) -> PredictorFn:
    """Build the predictor selected on the command line.

    Args:
        name (str | None): Built-in predictor name.
        predictions (Path | None): Directory of saved predictions.

    Returns:
        PredictorFn: Predictor.
    """
    return BUILTIN_PREDICTORS[name] if name else FilePredictor(predictions)


def validate_params(params: SimpleNamespace, schema: Schema) -> None:
    """Fail fast if evaluation parameters name classes the dataset lacks.

    Args:
        params (SimpleNamespace): Evaluation parameters.
        schema (Schema): Dataset taxonomy.

    Raises:
        ValueError: On unknown class names.
    """
    named: set[str] = set(params.structure.SHELL_CLASSES) | set(params.structure.OPEN_ROOM_CLASSES) | {params.rules.FILL_ROOM_CLASS}
    unknown: set[str] = named - set(schema.channels)
    if unknown:
        raise ValueError(f"Evaluation params use classes missing from the taxonomy: {sorted(unknown)}")


def select_cases(source: CaseSource, limit: int, predictions: Path | None, log: logging.Logger) -> list[int]:
    """Case indices to evaluate: the first ``limit`` cases, restricted to those
    with a prediction file when evaluating saved predictions.

    Args:
        source (CaseSource): Frozen cases of the split.
        limit (int): Maximum number of cases.
        predictions (Path | None): Saved-predictions directory, if any.
        log (logging.Logger): Logger (warns about missing predictions).

    Returns:
        list[int]: Case indices.

    Raises:
        FileNotFoundError: If no case has a prediction file.
    """
    indices: list[int] = list(range(limit))
    if predictions is None:
        return indices
    present: list[int] = [i for i in indices if (predictions / f"{source.spec(i, 0).case_id}.npz").is_file()]
    if not present:
        raise FileNotFoundError(f"No prediction files for the selected cases in {predictions}")
    if len(present) < len(indices):
        log.warning(f"Only {len(present)} of {len(indices)} cases have predictions; evaluating those only "
                    f"(results are NOT comparable to a full run)")
    return present


def evaluate_batch(indices: list[int], processed_dir: Path, split: str, predictor_name: str | None,
                   predictions: Path | None) -> list[dict[str, Any]]:
    """Evaluate a batch of cases (runs in a worker process).

    Args:
        indices (list[int]): Case indices.
        processed_dir (Path): Processed dataset directory.
        split (str): Split name.
        predictor_name (str | None): Built-in predictor name.
        predictions (Path | None): Saved predictions directory.

    Returns:
        list[dict[str, Any]]: One record per case.
    """
    source = CaseSource(processed_dir, split, seed=0)
    params: SimpleNamespace = load_params(PARAMS_FILE)
    predictor: PredictorFn = make_predictor(predictor_name, predictions)
    records: list[dict[str, Any]] = []
    for index in indices:
        case = source.case(index)
        samples: np.ndarray = predictor(case, source.schema, params)
        metrics: dict[str, Any] = score_samples(prepare_case(case, source.schema, params), samples, source.schema, params)
        records.append({**{f: case.info.get(f) for f in RECORD_FIELDS}, **metrics})
    return records


def write_examples(source: CaseSource, indices: list[int], predictor: PredictorFn, params: SimpleNamespace, count: int,
                   seed: int, title: str, path: Path) -> None:
    """Save a qualitative grid for ``count`` seeded random cases among the evaluated ones.

    Args:
        source (CaseSource): Case source.
        indices (list[int]): Evaluated case indices to choose from.
        predictor (PredictorFn): Predictor.
        params (SimpleNamespace): Evaluation parameters.
        count (int): Number of rows.
        seed (int): Selection seed.
        title (str): Figure title.
        path (Path): Output PNG.
    """
    picks: np.ndarray = np.random.default_rng(seed).choice(indices, size=min(count, len(indices)), replace=False)
    rows = []
    for index in sorted(picks.tolist()):
        case = source.case(index)
        info: dict[str, Any] = case.info
        rows.append((case, predictor(case, source.schema, params)[0],
                     f"{info['case_id']}\n{info['primary_cls']} | {info['mode']} | m={info['margin_px']}"))
    save_examples(rows, source.schema, params.decode.THRESHOLD, title, path)


def main() -> None:
    """Run the evaluation (see module docstring)."""
    args: argparse.Namespace = parse_args()
    config: SimpleNamespace = load_run_config(CONFIG_FILE, args.mode)
    params: SimpleNamespace = load_params(PARAMS_FILE)
    name: str = args.name or args.predictor or args.predictions.name
    run_id: str = make_run_id(f"eval-{name}")
    log: logging.Logger = setup_logger("evaluate", resolve_path(config.LOG_FILE_PATH), run_id, config.IS_DEBUG)
    processed_dir: Path = resolve_path(config.PROCESSED_ROOT) / args.dataset
    source = CaseSource(processed_dir, args.split, seed=0)
    validate_params(params, source.schema)
    out_dir: Path = resolve_path(config.OUTPUT_DIR) / args.dataset / args.split / name
    out_dir.mkdir(parents=True, exist_ok=True)

    limit: int = len(source) if config.MAX_CASES < 0 else min(config.MAX_CASES, len(source))
    indices: list[int] = select_cases(source, limit, args.predictions, log)
    total: int = len(indices)
    batches: list[list[int]] = [indices[s:s + config.BATCH_SIZE] for s in range(0, total, config.BATCH_SIZE)]
    workers: int = config.NUM_WORKERS if config.NUM_WORKERS > 0 else (os.cpu_count() or 1)
    log.info(f"Run {run_id}: {name} on {args.dataset}/{args.split}, {total} cases, {workers} workers -> {out_dir}")
    worker = partial(evaluate_batch, processed_dir=processed_dir, split=args.split,
                     predictor_name=args.predictor, predictions=args.predictions)
    with Pool(workers) as pool:
        records = pd.DataFrame([r for batch in pool.map(worker, batches) for r in batch])
    records.to_csv(out_dir / "cases.csv", index=False)

    tables = build_tables(records, params.components.IOU_THRESHOLDS)
    (out_dir / "summary.json").write_text(json.dumps({"run_id": run_id, "cases": total, "tables": tables_to_json(tables)}, indent=2))
    markdown: str = to_markdown(tables, params.report.HEADLINE_COLUMNS)
    (out_dir / "summary.md").write_text(f"# {name} on {args.dataset}/{args.split} ({total} cases)\n\n{markdown}")
    log.info("Results:\n" + to_markdown({k: tables[k] for k in ("overall", "by_type_mode")}, params.report.HEADLINE_COLUMNS))
    predictor: PredictorFn = make_predictor(args.predictor, args.predictions)
    for mode in (EASY, HARD):
        mode_indices: list[int] = [i for i in indices if source.spec(i, 0).mode == mode]
        if mode_indices:
            write_examples(source, mode_indices, predictor, params, config.NUM_EXAMPLES, config.SEED,
                           f"{name} on {args.dataset}/{args.split} ({mode} cases)", out_dir / f"examples_{mode}.png")
    log.info(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
