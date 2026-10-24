"""Post-process saved model predictions (CPU only, no model needed).

Per case: fuse the K saved samples by voting (``[aggregate]``), then apply the
steps of the chosen pipeline (``params.toml`` ``[pipelines]``) to the added
labels. Writes ``<case_id>.npz`` with ``samples`` of shape (1, C, H, W), uint8
{0, 255}, in the format ``evaluation.evaluate --predictions`` reads, so the
processed predictions are scored exactly like raw ones.

Usage (from the repository root):
    python -m Reconstruction.postprocess.run --dataset resplan --split val \
        --predictions Reconstruction/runs/<run>/predictions/resplan_val/step_0060000_k1 --pipeline full --mode prod
    python -m Reconstruction.evaluation.evaluate --dataset resplan --split val \
        --predictions <printed output directory> --name <run>_full --mode prod

The output directory defaults to ``<predictions>+<pipeline>`` next to the input.
"""

import argparse
import logging
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from Reconstruction.common.config_loader import load_params, load_run_config, resolve_path
from Reconstruction.common.logging_setup import make_run_id, setup_logger
from Reconstruction.data_prep.case_source import CaseSource
from Reconstruction.data_prep.schema import Schema
from Reconstruction.evaluation.labels import LabelMaps, added, compose, labels_to_semantic, semantic_to_labels
from Reconstruction.postprocess.steps import STEPS

SCRIPT_DIR: Path = Path(__file__).resolve().parent
CONFIG_FILE: Path = SCRIPT_DIR / "config.toml"
PARAMS_FILE: Path = SCRIPT_DIR / "params.toml"
EVAL_PARAMS_FILE: Path = SCRIPT_DIR.parent / "evaluation" / "params.toml"
SCORE_SCALE: int = 255


def parse_args() -> argparse.Namespace:
    """Command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Processed dataset name (resplan, cubicasa)")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--predictions", type=Path, required=True, help="Directory of <case_id>.npz model predictions")
    parser.add_argument("--pipeline", required=True, help="Pipeline name from params.toml [pipelines]")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: <predictions>+<pipeline>)")
    parser.add_argument("--mode", choices=["debug", "prod"], default="debug")
    return parser.parse_args()


def aggregate_samples(samples: np.ndarray, vote_threshold: float) -> np.ndarray:
    """Fuse K binary samples by per-pixel, per-channel voting.

    Args:
        samples (np.ndarray): uint8 (K, C, H, W) saved samples ({0, 255}).
        vote_threshold (float): Fraction of samples that must agree.

    Returns:
        np.ndarray: float32 (C, H, W) in {0, 1}.
    """
    votes: np.ndarray = (samples > SCORE_SCALE // 2).mean(axis=0)
    return (votes >= vote_threshold).astype(np.float32)


def postprocess_case(samples: np.ndarray, incomplete: np.ndarray, schema: Schema, params: SimpleNamespace,
                     steps: list[str], decode_threshold: float) -> np.ndarray:
    """Aggregate and post-process the prediction of one case.

    Args:
        samples (np.ndarray): uint8 (K, C, H, W) saved samples.
        incomplete (np.ndarray): (C, H, W) incomplete input plan.
        schema (Schema): Taxonomy.
        params (SimpleNamespace): Post-processing parameters.
        steps (list[str]): Ordered step names.
        decode_threshold (float): Evaluation decode threshold.

    Returns:
        np.ndarray: uint8 (1, C, H, W) {0, 255} processed prediction.
    """
    base: LabelMaps = semantic_to_labels(incomplete, schema, decode_threshold)
    fused: np.ndarray = aggregate_samples(samples, params.aggregate.VOTE_THRESHOLD)
    # Work on what the prediction actually adds (the evaluation composes it the same way).
    current: LabelMaps = added(base, compose(base, semantic_to_labels(fused, schema, decode_threshold)))
    for name in steps:
        current = STEPS[name](current, base, schema, getattr(params.steps, name))
    return (labels_to_semantic(current, schema) * SCORE_SCALE)[None]


def process_batch(indices: list[int], processed_dir: Path, split: str, predictions: Path, output: Path,
                  steps: list[str]) -> tuple[int, int]:
    """Post-process a batch of cases (runs in a worker process).

    Args:
        indices (list[int]): Case indices.
        processed_dir (Path): Processed dataset directory.
        split (str): Split name.
        predictions (Path): Directory of raw predictions.
        output (Path): Output directory.
        steps (list[str]): Ordered step names.

    Returns:
        tuple[int, int]: (cases written, cases whose prediction the steps changed).
    """
    source = CaseSource(processed_dir, split, seed=0)
    params: SimpleNamespace = load_params(PARAMS_FILE)
    threshold: float = load_params(EVAL_PARAMS_FILE).decode.THRESHOLD
    changed: int = 0
    for index in indices:
        case = source.case(index)
        with np.load(predictions / f"{case.info['case_id']}.npz") as data:
            samples: np.ndarray = data["samples"]
        result: np.ndarray = postprocess_case(samples, case.incomplete, source.schema, params, steps, threshold)
        reference: np.ndarray = postprocess_case(samples, case.incomplete, source.schema, params, [], threshold)
        changed += int(not np.array_equal(result, reference))
        np.savez_compressed(output / f"{case.info['case_id']}.npz", samples=result)
    return len(indices), changed


def main() -> None:
    """Entry point (see module docstring)."""
    args: argparse.Namespace = parse_args()
    config: SimpleNamespace = load_run_config(CONFIG_FILE, args.mode)
    params: SimpleNamespace = load_params(PARAMS_FILE)
    pipelines: dict[str, list[str]] = vars(params.pipelines)
    if args.pipeline not in pipelines:
        raise KeyError(f"Unknown pipeline '{args.pipeline}'; available: {sorted(pipelines)}")
    steps: list[str] = pipelines[args.pipeline]
    unknown: list[str] = [s for s in steps if s not in STEPS]
    if unknown:
        raise KeyError(f"Pipeline '{args.pipeline}' uses unknown steps {unknown}; available: {sorted(STEPS)}")
    log: logging.Logger = setup_logger("postprocess", resolve_path(config.LOG_FILE_PATH),
                                       make_run_id(f"postprocess-{args.pipeline}"), config.IS_DEBUG)
    processed_dir: Path = resolve_path(config.PROCESSED_ROOT) / args.dataset
    source = CaseSource(processed_dir, args.split, seed=0)
    output: Path = args.output or args.predictions.parent / f"{args.predictions.name}+{args.pipeline}"
    output.mkdir(parents=True, exist_ok=True)

    limit: int = len(source) if config.MAX_CASES < 0 else min(config.MAX_CASES, len(source))
    # Same case selection rule as evaluate.py: cases that have a prediction file.
    indices: list[int] = [i for i in range(limit) if (args.predictions / f"{source.spec(i, 0).case_id}.npz").is_file()]
    batches: list[list[int]] = [indices[s:s + config.BATCH_SIZE] for s in range(0, len(indices), config.BATCH_SIZE)]
    workers: int = config.NUM_WORKERS if config.NUM_WORKERS > 0 else (os.cpu_count() or 1)
    log.info(f"Pipeline '{args.pipeline}' {steps} on {args.dataset}/{args.split}: {len(indices)} cases, {workers} workers -> {output}")
    worker = partial(process_batch, processed_dir=processed_dir, split=args.split, predictions=args.predictions,
                     output=output, steps=steps)
    with Pool(workers) as pool:
        results: list[tuple[int, int]] = pool.map(worker, batches)
    written: int = sum(r[0] for r in results)
    changed: int = sum(r[1] for r in results)
    log.info(f"Wrote {written} files; the steps changed {changed} predictions ({changed / max(written, 1):.1%}). Output: {output}")


if __name__ == "__main__":
    main()
