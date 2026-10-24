"""Qualitative grids of the best and worst predictions of an evaluated run.

Reads the ``cases.csv`` written by ``evaluate.py`` for a run, ranks the cases
of every (component type, removal mode) group by BEST_WORST_RANK_COLUMNS, and
saves one grid per group: the BEST_WORST_COUNT best cases followed by the
BEST_WORST_COUNT worst ones (Input | Prediction | Composed plan | Ground truth).

Outputs: OUTPUT_DIR/<dataset>/<split>/<name>/BEST_WORST_DIR/<type>_<mode>.png

Usage (from the repository root, after evaluate.py has run):
    python -m Reconstruction.evaluation.best_worst --dataset resplan --split test \
        --predictions Reconstruction/runs/run2-a40/predictions/resplan_test/step_0150000_k1 \
        --name run2-a40_step150000_k1 --mode prod
"""

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from Reconstruction.common.config_loader import load_params, load_run_config, resolve_path
from Reconstruction.common.logging_setup import make_run_id, setup_logger
from Reconstruction.data_prep.case_source import CaseSource
from Reconstruction.evaluation.baselines import FilePredictor
from Reconstruction.evaluation.examples import save_examples

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
    parser.add_argument("--predictions", type=Path, required=True, help="Directory of <case_id>.npz model predictions")
    parser.add_argument("--name", required=True, help="Evaluated run name (folder holding cases.csv)")
    parser.add_argument("--mode", choices=["debug", "prod"], default="prod")
    return parser.parse_args()


def rank_group(group: pd.DataFrame, rank_columns: list[str], count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best and worst cases of one group.

    Missing metric values (e.g. no prediction matched) count as 0. Ties are
    broken by case_id so the selection is reproducible.

    Args:
        group (pd.DataFrame): cases.csv rows of one (type, mode) group.
        rank_columns (list[str]): Ranking columns, higher = better.
        count (int): Rows per list.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Best (best first) and worst (worst first) rows.
    """
    scored: pd.DataFrame = group.assign(**{c: group[c].fillna(0.0) for c in rank_columns})
    best: pd.DataFrame = scored.sort_values(rank_columns + ["case_id"], ascending=[False] * len(rank_columns) + [True])
    worst: pd.DataFrame = scored.sort_values(rank_columns + ["case_id"], ascending=[True] * len(rank_columns) + [True])
    return best.head(count), worst.head(count)


def row_label(tag: str, row: pd.Series, rank_columns: list[str]) -> str:
    """Y-axis label of one grid row: tag, case id and its scores.

    Args:
        tag (str): e.g. "BEST 1".
        row (pd.Series): cases.csv row.
        rank_columns (list[str]): Scores to show.

    Returns:
        str: Multi-line label.
    """
    scores: str = " ".join(f"{c}={row[c]:.2f}" for c in rank_columns)
    return f"{tag}\n{row['case_id']}\n{row['primary_cls']} -> {row['predicted_type']}\n{scores}"


def main() -> None:
    """Build all best/worst grids of one evaluated run (see module docstring)."""
    args: argparse.Namespace = parse_args()
    config: SimpleNamespace = load_run_config(CONFIG_FILE, args.mode)
    params: SimpleNamespace = load_params(PARAMS_FILE)
    log: logging.Logger = setup_logger("best_worst", resolve_path(config.LOG_FILE_PATH),
                                       make_run_id(f"bestworst-{args.name}"), config.IS_DEBUG)
    run_dir: Path = resolve_path(config.OUTPUT_DIR) / args.dataset / args.split / args.name
    records: pd.DataFrame = pd.read_csv(run_dir / "cases.csv")
    source = CaseSource(resolve_path(config.PROCESSED_ROOT) / args.dataset, args.split, seed=0)
    # cases.csv holds case ids; map them back to frozen-case indices.
    index_of: dict[str, int] = {source.spec(i, 0).case_id: i for i in range(len(source))}
    predictor = FilePredictor(args.predictions)
    rank_columns: list[str] = list(config.BEST_WORST_RANK_COLUMNS)
    out_dir: Path = run_dir / config.BEST_WORST_DIR

    for (ctype, mode), group in records.groupby(["comp_type", "mode"]):
        best, worst = rank_group(group, rank_columns, config.BEST_WORST_COUNT)
        rows = []
        for tag, chosen in (("BEST", best), ("WORST", worst)):
            for rank, (_, row) in enumerate(chosen.iterrows(), start=1):
                case = source.case(index_of[row["case_id"]])
                rows.append((case, predictor(case, source.schema, params)[0], row_label(f"{tag} {rank}", row, rank_columns)))
        title: str = (f"{args.name} on {args.dataset}/{args.split}: {ctype} / {mode} "
                      f"({len(group)} cases; top {len(best)} best, then {len(worst)} worst by {', '.join(rank_columns)})")
        path: Path = out_dir / f"{ctype}_{mode}.png"
        save_examples(rows, source.schema, params.decode.THRESHOLD, title, path)
        log.info(f"Saved {path}")


if __name__ == "__main__":
    main()
