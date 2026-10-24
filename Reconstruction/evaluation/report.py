"""Aggregation of per-case metrics into report tables.

Detection precision / recall / F1 are *micro* averages (summed true
positives over summed predicted / true components), so large and small cases
weigh by their component counts. Every other metric is a per-case mean
(medians for localization error, which has a long tail). NaN / missing values
(e.g. connection accuracy for non-openings) are ignored.

Tables: overall; by component type; type x mode; margin; size bin;
location; and, for hard door/window cases, split by whether the removal
left a visible clue.
"""

import math
from typing import Any

import pandas as pd

# (table name, grouping columns); no columns = one overall row.
GROUPINGS: tuple[tuple[str, list[str]], ...] = (
    ("overall", []),
    ("by_type", ["comp_type"]),
    ("by_type_mode", ["comp_type", "mode"]),
    ("by_margin", ["margin_px"]),
    ("by_size", ["comp_type", "size_bin"]),
    ("by_location", ["comp_type", "location"]),
)
HARD_CLUE_TABLE: str = "hard_openings_by_clue"


def _as_float(series: pd.Series) -> pd.Series:
    """Numeric view of a column mixing bools, numbers and missing values.

    Args:
        series (pd.Series): Column.

    Returns:
        pd.Series: float column (NaN for missing).
    """
    return series.apply(lambda v: math.nan if v is None or (isinstance(v, float) and math.isnan(v)) else float(v))


def _aggregate(group: pd.DataFrame, thresholds: list[float]) -> dict[str, float]:
    """Summary metrics of a group of case records.

    Args:
        group (pd.DataFrame): Case records.
        thresholds (list[float]): IoU thresholds (first = headline).

    Returns:
        dict[str, float]: Metric name -> value.
    """
    out: dict[str, float] = {"cases": float(len(group))}
    n_pred: float = float(group["n_pred"].sum())
    n_truth: float = float(group["n_truth"].sum())
    for t in thresholds:
        tp: float = float(group[f"tp@{t}"].sum())
        precision: float = tp / n_pred if n_pred else math.nan
        recall: float = tp / n_truth if n_truth else math.nan
        out[f"precision@{t}"] = precision
        out[f"recall@{t}"] = recall
        out[f"f1@{t}"] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    means: dict[str, str] = {
        "primary_detection": "primary_detected", "primary_iou": "primary_iou",
        "bestk_primary_detection": "bestk_primary_detected", "type_accuracy": "type_correct",
        "connection_accuracy": "connection_correct", "bestk_connection_accuracy": "bestk_connection_correct",
        "pixel_iou": "pixel_iou", "openings_on_wall": "openings_on_wall",
        "rooms_enclosed": "rooms_enclosed", "rooms_reachable": "rooms_reachable",
    }
    for name, column in means.items():
        if column in group:
            values: pd.Series = _as_float(group[column])
            out[name] = float(values.mean()) if values.notna().any() else math.nan  # Series.mean skips NaN
    if "loc_error_m" in group:
        errors: pd.Series = _as_float(group["loc_error_m"]).dropna()
        out["loc_error_m_median"] = float(errors.median()) if len(errors) else math.nan
    shell: pd.Series = _as_float(group["shell_pieces_diff"])
    out["shell_exact"] = float((shell == 0).mean())
    return out


def build_tables(records: pd.DataFrame, thresholds: list[float]) -> dict[str, pd.DataFrame]:
    """All report tables from per-case records.

    Args:
        records (pd.DataFrame): One row per case (see ``case_metrics``).
        thresholds (list[float]): IoU thresholds.

    Returns:
        dict[str, pd.DataFrame]: Table name -> table (one row per group).
    """
    tables: dict[str, pd.DataFrame] = {}
    for name, keys in GROUPINGS:
        if not keys:
            tables[name] = pd.DataFrame([_aggregate(records, thresholds)], index=["all"])
            continue
        rows = {group_key: _aggregate(group, thresholds) for group_key, group in records.groupby(keys, dropna=False)}
        tables[name] = pd.DataFrame.from_dict(rows, orient="index")
    if "clue" in records:
        hard: pd.DataFrame = records[(records["mode"] == "hard") & records["clue"].notna()]
        if len(hard):
            rows = {key: _aggregate(group, thresholds) for key, group in hard.groupby(["comp_type", "clue"])}
            tables[HARD_CLUE_TABLE] = pd.DataFrame.from_dict(rows, orient="index")
    return tables


def to_markdown(tables: dict[str, pd.DataFrame], columns: list[str]) -> str:
    """Render selected columns of every table as Markdown (no extra dependency).

    Args:
        tables (dict[str, pd.DataFrame]): Report tables.
        columns (list[str]): Columns to show (missing ones are skipped).

    Returns:
        str: Markdown text.
    """
    lines: list[str] = []
    for name, table in tables.items():
        shown: list[str] = [c for c in columns if c in table.columns]
        lines += [f"### {name}", "", "| group | " + " | ".join(shown) + " |", "|---" * (len(shown) + 1) + "|"]
        for index, row in table.iterrows():
            label: str = " / ".join(map(str, index)) if isinstance(index, tuple) else str(index)
            cells: list[str] = [f"{int(row[c])}" if c == "cases" else ("-" if pd.isna(row[c]) else f"{row[c]:.3f}") for c in shown]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def tables_to_json(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """JSON-serializable form of the tables (NaN -> None).

    Args:
        tables (dict[str, pd.DataFrame]): Report tables.

    Returns:
        dict[str, Any]: Table name -> {group label -> metrics}.
    """
    out: dict[str, Any] = {}
    for name, table in tables.items():
        out[name] = {(" / ".join(map(str, k)) if isinstance(k, tuple) else str(k)):
                     {c: (None if pd.isna(v) else float(v)) for c, v in row.items()} for k, row in table.iterrows()}
    return out
